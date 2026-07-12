import os
import json
import time
import torch
import sys
from pathlib import Path
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Auto-Detect Correct Paths ─────────────────────────────────────────────
if Path("research-rag-assistant/storage/text_db.faiss").exists():
    BASE_PATH = Path("research-rag-assistant")
else:
    BASE_PATH = Path(".")

CHUNKS_PATH = BASE_PATH / "storage" / "chunks.json"
TEXT_INDEX_PATH = BASE_PATH / "storage" / "text_db.faiss"
GOLD_DATASET = Path("gold_qa_dataset.json") if Path("gold_qa_dataset.json").exists() else BASE_PATH / "gold_qa_dataset.json"

# Save to separate file
OUT_FILE = BASE_PATH / "storage" / "rag_hybrid_answers.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── 1. Load Models ───────────────────────────────────────────────────────────
print("Loading BGE Embedder...")
text_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)

print("Loading Cross-Encoder Reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)

print("Loading local LLM Qwen 2.5 3B Generator on GPU...")
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id
generator.tokenizer.padding_side = "left"

# ── 2. Load Text Database & Build Local BM25 Index ──────────────────────────
print("Loading text database chunks...")
with open(CHUNKS_PATH) as f:
    text_chunks = json.load(f)

print("Loading FAISS text index...")
text_index = faiss.read_index(str(TEXT_INDEX_PATH))

print("Tokenizing corpus for BM25...")
tokenized_corpus = [chunk["text"].lower().split() for chunk in text_chunks]
bm25 = BM25Okapi(tokenized_corpus)
print("✅ Local BM25 Keyword Index built successfully!")

# ── 3. Load Gold Dataset and Get First 500 Questions ─────────────────────────
with open(GOLD_DATASET) as f:
    dataset = json.load(f)

all_pairs = []
for paper_id, paper_data in dataset.items():
    for qa in paper_data.get("qa_pairs", []):
        all_pairs.append({
            "question_id": qa["question_id"],
            "paper_title": paper_data.get("paper_title", ""),
            "question": qa["question"],
            "expected_answer": qa["expected_answer"],
            "difficulty": qa.get("difficulty", ""),
            "question_type": qa.get("question_type", "")
        })

# CAPPED TO 500 QUESTIONS
experimental_pairs = all_pairs[:500]
print(f"Loaded {len(experimental_pairs)} questions for Hybrid RAG.")

# Load previous results if resuming
hybrid_results = {}
if OUT_FILE.exists():
    try:
        with open(OUT_FILE) as f:
            hybrid_results = json.load(f)
    except: pass

pending_pairs = [qa for qa in experimental_pairs if qa["question_id"] not in hybrid_results]
print(f"Already generated: {len(hybrid_results)} | Pending: {len(pending_pairs)}")

# ── 4. Hybrid Search + Generation Loop ───────────────────────────────────────
BATCH_SIZE = 8
t_start = time.time()

for i in range(0, len(pending_pairs), BATCH_SIZE):
    batch_qas = pending_pairs[i:i + BATCH_SIZE]
    prompts = []
    batch_contexts = []
    
    for qa in batch_qas:
        query = qa["question"]
        query_words = query.lower().split()
        
        # A. Dense (FAISS) Retrieval
        query_emb = text_model.encode(f"Represent this sentence for searching relevant passages: {query}", convert_to_numpy=True)
        dense_scores, dense_indices = text_index.search(np.array([query_emb], dtype="float32"), 30)
        dense_ranks = {idx: rank for rank, idx in enumerate(dense_indices[0])}
        
        # B. Sparse (BM25) Retrieval
        sparse_scores = bm25.get_scores(query_words)
        sparse_indices = np.argsort(sparse_scores)[::-1][:30]
        sparse_ranks = {idx: rank for rank, idx in enumerate(sparse_indices)}
        
        # C. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for idx in set(dense_ranks.keys()).union(sparse_ranks.keys()):
            r_dense = dense_ranks.get(idx, 999)
            r_sparse = sparse_ranks.get(idx, 999)
            rrf_scores[idx] = 1 / (60 + r_dense) + 1 / (60 + r_sparse)
            
        top_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:15]
        candidate_chunks = [text_chunks[idx] for idx in top_indices if idx < len(text_chunks)]
        
        # D. Reranking
        retrieved_text_chunks = []
        context_blocks = []
        if candidate_chunks:
            pairs = [[query, c["text"]] for c in candidate_chunks]
            rerank_scores = reranker.predict(pairs)
            ranked_indices = np.argsort(rerank_scores)[::-1]
            
            for rank_pos, r_idx in enumerate(ranked_indices[:3], start=1):
                chunk = candidate_chunks[r_idx]
                retrieved_text_chunks.append({
                    "score": float(rerank_scores[r_idx]),
                    "source_file": chunk.get("source_file", ""),
                    "page": chunk.get("page_number", 1),
                    "text": chunk.get("text", "")[:800]
                })
                context_blocks.append(f"[{rank_pos}] Source: {chunk.get('source_file','')}, Page: {chunk.get('page_number','')}\nText: {chunk.get('text','')}")
        
        batch_contexts.append(retrieved_text_chunks)
        context_str = "\n\n".join(context_blocks)
        
        prompt_text = (
            "You are a helpful research assistant. Answer the following question based ONLY on the provided context.\n"
            "CRITICAL RULES:\n"
            "1. Use only the provided context. Do NOT use your own knowledge.\n"
            "2. Output ONLY the direct answer. Do NOT include any introduction or prefix.\n"
            "3. If the context does not contain enough information, state that clearly.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        prompts.append(prompt_text)
        
    t_batch = time.time()
    try:
        outputs = generator(
            prompts,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            batch_size=len(prompts)
        )
    except Exception as e:
        print(f"Batch generation error: {e}")
        outputs = [{"generated_text": p + f" [ERROR: Generation failed: {e}]"} for p in prompts]
        
    for qa, context, prompt_text, out in zip(batch_qas, batch_contexts, prompts, outputs):
        full_text = out["generated_text"] if isinstance(out, dict) else out[0]["generated_text"]
        ans = full_text[len(prompt_text):].strip()
        
        qid = qa["question_id"]
        hybrid_results[qid] = {
            "question_id": qid,
            "paper_title": qa["paper_title"],
            "question_type": qa["question_type"],
            "difficulty": qa["difficulty"],
            "question": qa["question"],
            "expected_answer": qa["expected_answer"],
            "rag_answer": ans,
            "retrieved_chunks": context,
            "latency_sec": time.time() - t_batch
        }
        
    with open(OUT_FILE, "w") as f:
        json.dump(hybrid_results, f, indent=2, ensure_ascii=False)
        
    finished = len(hybrid_results)
    elapsed = time.time() - t_start
    est_remaining = (elapsed / (i + BATCH_SIZE)) * (len(pending_pairs) - (i + BATCH_SIZE)) if (i + BATCH_SIZE) < len(pending_pairs) else 0
    print(f"Generated [{finished}/500] answers using Hybrid Search | Est. remaining: {est_remaining/60:.1f} mins", flush=True)

print(f"\n🎉 Hybrid generation finished! Answers saved to: {OUT_FILE}")
