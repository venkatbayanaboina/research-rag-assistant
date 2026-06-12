import faiss
from sentence_transformers import SentenceTransformer

import json



embedding_model= SentenceTransformer(
    "BAAI/bge-large-en-v1.5"
)


index= faiss.read_index(
    "../storage/embedding_db.faiss"
)

#print("vectors in faiss:",index.ntotal)


with open("../storage/chunks.json","r") as f:
    all_chunks=json.load(f)

#print("chunks loaded :" ,(len(all_chunks)))




query = "What is  the transformer architecture?"

query_embedding=embedding_model.encode(
    query,
    convert_to_numpy=True,
    normalize_embeddings=True   
                              )

query_embedding=query_embedding.reshape(1,-1)

print(query_embedding.shape)

scores,indices=index.search(query_embedding,k=5)

print(indices)

print()

print(scores)
# print(documents[indices[0][0]])
# print()
# print(documents[indices[0][1]])

# print("Phase completed successfully")

for idx in indices[0]:
    print("chunk Id :",idx)
    print("page: ",all_chunks[idx]["page"])
    print()
    print(all_chunks[idx]["text"][:500])
