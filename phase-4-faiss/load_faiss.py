import faiss

index= faiss.read_index(
    "research_index.faiss"
)

print("index loaded")
print("vectors:",index.ntotal)
print("dimension :", index.d )
