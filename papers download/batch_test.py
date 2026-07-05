import requests
import xml.etree.ElementTree as ET

ids = [
    "0903.0340",
    "1103.0398",
    "1104.5557"
]

url = (
    "https://export.arxiv.org/api/query?id_list="
    + ",".join(ids)
)

response = requests.get(
    url,
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=30
)

print("Status:", response.status_code)

root = ET.fromstring(response.text)

ns = {"atom": "http://www.w3.org/2005/Atom"}

for entry in root.findall("atom:entry", ns):
    paper_id = entry.find("atom:id", ns).text.split("/")[-1]
    title = entry.find("atom:title", ns).text.strip()

    print(paper_id)
    print(title)
    print("-" * 50)