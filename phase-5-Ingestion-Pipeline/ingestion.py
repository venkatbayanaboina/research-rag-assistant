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

# elements = partition_pdf(
#                        filename = "../docs/attention-is-all-you-need-Paper.pdf",
#                         strategy="hi_res",
#                         infer_table_structure=True,
#                         extract_image_block_types=["Image"],
#                         extract_image_block_to_payload=True
#                     )
# print("Partition finished")
# chunks = chunk_by_title(elements) 

# def chunk_to_dict(chunk, chunk_id):

#     result = {
#         "chunk_id": chunk_id,
#         "text": chunk.text,
#         "page": chunk.metadata.page_number,
#         "images": [],
#         "tables": []
#     }

#     for element in chunk.metadata.orig_elements:

#         if element.category == "Image":

#             result["images"].append({
#                 "page": element.metadata.page_number,
#                 "ocr_text": element.text,
                
#                 "image_base64": element.metadata.image_base64
#             })

#         elif element.category == "Table":

#             result["tables"].append({
#                 "page": element.metadata.page_number,
#                 "text": element.text,
#                 "html": element.metadata.text_as_html
#             })

#     return result




# all_chunks = []

# for i, chunk in enumerate(chunks):
#     all_chunks.append(
#         chunk_to_dict(chunk,i)
#     )
     


# with open("chunks_json_form.json","w") as f:
#     json.dump(
#         all_chunks,
#         f,
#         indent=2
#     )
with open ("../storage/chunks.json","r") as f:
  all_chunks=json.load(f)

def build_embedding_text(chunk):
  return chunk["text"]

index = faiss.IndexFlatIP(1024)


      
for chunk in all_chunks:

    text= build_embedding_text(chunk)

    embedding= embedding_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True    ) 
    
    print(embedding.shape)

    embedding = embedding.reshape(1, -1)   

    index.add(embedding) 

    #print(embedding.shape)

print(index.ntotal)


# faiss.write_index(

#    index,
#     "embedding_db.faiss"             
#                  )

