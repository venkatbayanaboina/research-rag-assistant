import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf      import partition_pdf
from unstructured.chunking.title     import chunk_by_title

# Resolve paths relative to this script's directory for robust execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../docs/attention-is-all-you-need-Paper.pdf"))
CHUNKS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../storage/chunks.json"))
INDEX_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../storage/embedding_db.faiss"))

embedding_model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5"
)

print(f"Starting partition for: {PDF_PATH}")
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

# Use "fast" strategy on local macOS / Colab to avoid ONNX/Multiprocessing segmentation faults
elements = partition_pdf(
    filename = PDF_PATH,
    strategy="fast"
)
print(f"Partition finished. Extracted {len(elements)} elements.")

# Chunk with overlap and limits
chunks = chunk_by_title(
    elements,
    max_characters=2000,
    new_after_n_chars=1500,
    combine_text_under_n_chars=100,
    overlap=200
) 

def chunk_to_dict(chunk, chunk_id):
    result = {
        "chunk_id": chunk_id,
        "text": chunk.text,
        "page": chunk.metadata.page_number,
        "images": [],
        "tables": []
    }

    # Extract elements if they exist (fast strategy parses primarily text)
    if hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            if element.category == "Image":
                result["images"].append({
                    "page": element.metadata.page_number,
                    "ocr_text": element.text,
                    "image_base64": getattr(element.metadata, "image_base64", "")
                })
            elif element.category == "Table":
                result["tables"].append({
                    "page": element.metadata.page_number,
                    "text": element.text,
                    "html": getattr(element.metadata, "text_as_html", "")
                })

    return result

all_chunks = []
for i, chunk in enumerate(chunks):
    all_chunks.append(
        chunk_to_dict(chunk, i)
    )

print(f"Ingestion generated {len(all_chunks)} chunks.")
if len(all_chunks) > 0:
    print("First chunk text preview:")
    print(all_chunks[0]["text"][:300])
    print("="*80)
else:
    print("Warning: No chunks generated. Skipping file save.")
    exit(1)

# Save the structured chunks directly to storage directory
with open(CHUNKS_PATH, "w") as f:
    json.dump(
        all_chunks,
        f,
        indent=2
    )
print(f"Saved chunks to {CHUNKS_PATH}")

def build_embedding_text(chunk):
    return chunk["text"]

index = faiss.IndexFlatIP(1024)

print("Generating embeddings and building FAISS index...")
for chunk in all_chunks:
    text = build_embedding_text(chunk)
    embedding = embedding_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embedding = embedding.reshape(1, -1)
    index.add(embedding)

print(f"Total vectors in FAISS index: {index.ntotal}")

# Save the index directly to storage directory
faiss.write_index(
    index,
    INDEX_PATH
)
print(f"Saved FAISS index to {INDEX_PATH}")
print("Ingestion completed successfully!")
