import requests
from pathlib import Path
from paper_sources import get_pdf_url

with open("metadata/paper_links.txt", "r", encoding="utf-8") as f:
    url = f.readline().strip()

print("Original URL:")
print(url)

pdf_url = get_pdf_url(url)

if pdf_url is None:
    print(f"Skipping: {url}")
    exit()

print("\nDownloading from:")
print(pdf_url)

Path("pdfs").mkdir(exist_ok=True)

response = requests.get(pdf_url, timeout=30)

if response.status_code == 200:
    with open("pdfs/test.pdf", "wb") as f:
        f.write(response.content)

    print("\n✅ Download successful!")
else:
    print(f"\n❌ Failed ({response.status_code})")