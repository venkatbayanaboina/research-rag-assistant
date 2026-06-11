import faiss
from sentence_transformers import SentenceTransformer

import numpy as np

model= SentenceTransformer(
    "BAAI/bge-large-en-v1.5"
)

documents =["Artificial Intelligence will be the next booming domain in the world", "Neural networks made the AI possible " , "India won the last T20 worldcup"]

embeddings= model.encode(documents)

embeddings = np.array(
    embeddings,
    dtype="float32"
)            

dimension =embeddings.shape[1]

embedding_db = faiss.IndexFlatL2(dimension)

embedding_db.add(embeddings)
print(embeddings.shape)
faiss.write_index(
    embedding_db,
     "embedding_db.faiss"             
                  )
print("embedding db created and stored successfully.")
print()

query = "Whats the Ai neural net?"

query_embedding= model.encode(query)
query_embedding=query_embedding.reshape(1,1024)
print(query_embedding.shape)

distances,indices=embedding_db.search(query_embedding,2)


print(documents[indices[0][0]])
print()
print(documents[indices[0][1]])

print("Phase completed successfully")
