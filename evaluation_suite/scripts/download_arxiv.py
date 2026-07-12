import requests
from pathlib import Path
import time

# Create pdf directory
Path("pdfs").mkdir(exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0"
}

downloaded = 0
failed = 0

with open("metadata/paper_links.txt", "r", encoding="utf-8") as f:
    for url in f:

        url = url.strip()

        # Skip non-arXiv links
        if "arxiv.org/abs/" not in url:
            continue

        # Convert abstract URL to PDF URL
        pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"

        paper_id = pdf_url.split("/")[-1]

        filename = Path("pdfs") / paper_id

        # Skip if already downloaded
        if filename.exists():
            print(f"Already exists: {paper_id}")
            continue

        try:
            r = requests.get(pdf_url, headers=headers, timeout=30)

            if r.status_code == 200:
                with open(filename, "wb") as pdf:
                    pdf.write(r.content)

                downloaded += 1
                print(f"Downloaded: {paper_id}")

            else:
                failed += 1
                print(f"Failed ({r.status_code}): {paper_id}")

        except Exception as e:
            failed += 1
            print(e)

        # Be polite to arXiv
        time.sleep(1)

print("\nFinished")
print("Downloaded:", downloaded)
print("Failed:", failed)    