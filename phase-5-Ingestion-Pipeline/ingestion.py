import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf      import partition_pdf
from unstructured.chunking.title     import chunk_by_title

embedding_model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5"
)

print("Starting partition")

# Use "fast" strategy on local macOS to avoid ONNX/Multiprocessing segmentation faults
elements = partition_pdf(
    filename = "../docs/attention-is-all-you-need-Paper.pdf",
    strategy="fast"
)
print("Partition finished")

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
print("First chunk text preview:")
print(all_chunks[0]["text"][:300])
print("="*80)

# Save the structured chunks directly to storage directory
with open("../storage/chunks.json", "w") as f:
    json.dump(
        all_chunks,
        f,
        indent=2
    )
print("Saved chunks to ../storage/chunks.json")

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
    "../storage/embedding_db.faiss"
)
print("Saved FAISS index to ../storage/embedding_db.faiss")
print("Ingestion completed successfully!")
