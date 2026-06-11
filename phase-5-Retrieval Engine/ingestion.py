import os
from unstructured.partition.pdf      import partition_pdf
from unstructured.chunking.title     import chunk_by_title

elements = partition_pdf(
                       filename = "../docs/attention-is-all-you-need-Paper.pdf",
                        strategy="hi_res",
                        infer_table_structure=True,
                        extract_image_block_types=["Image"],
                        extract_image_block_to_payload=True
                    )

chunks = chunk_by_title(elements) #

print(len(chunks))

print(chunks[17].text)
print((chunks[17].metadata.orig_elements))

images = [e for e in elements if e.category == "Image"]
tables = [e for e in elements if e.category == "Table"]

print("Images:", len(images))
print("Tables:", len(tables))


print(tables[0].to_dict())



uniquetypes=set()

for doc in elements:
    uniquetypes.add(type(doc))

print(len(elements))
print ()

for ut in uniquetypes:
    print(ut)

