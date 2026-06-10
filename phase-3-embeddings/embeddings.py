
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


ai= "Artificial Inteligence"
nn= "neural Network "        
cw="cricket worldcup"


model=SentenceTransformer(
    "BAAI/bge-large-en-v1.5"
)

aien=model.encode(ai)
nnen=model.encode(nn)
cwen=model.encode(cw)

sim_ai_nn= cosine_similarity([aien],[nnen])[0][0]
sim_ai_cw=cosine_similarity([aien],[cwen])[0][0]


print("AI ↔ Neural Network:", sim_ai_nn)

print("AI ↔ Cricket:", sim_ai_cw)