import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

texts = [

    "Artificial Intelligence",

    "Neural Network",

    "Cricket World Cup"

]

model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5"
)

embeddings = model.encode(texts)

embeddings = np.array(

    embeddings,

    dtype="float32"

)

dimension= embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index,"research_index.faiss")
print("index saved successfully")

print ("dimension :" ,dimension)

print("total vectors stored :", index.ntotal)
