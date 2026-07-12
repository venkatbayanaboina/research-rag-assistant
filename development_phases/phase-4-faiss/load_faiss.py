import faiss

db= faiss.read_index(
    "research_index.faiss"
)

print("index loaded")
print("vectors:",db.ntotal)
print("dimension :", db.d )
