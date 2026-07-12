import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import csv
import time

PDF_DIR = Path("pdfs")
CSV_FILE = "metadata.csv"

ns = {"atom": "http://www.w3.org/2005/Atom"}

metadata = []

headers = {
    "User-Agent": "Mozilla/5.0"
}

for pdf in PDF_DIR.glob("*.pdf"):

    paper_id = pdf.stem

    # Skip already renamed files
    if "_" in paper_id:
        continue

    print(f"\nProcessing {paper_id}...")

    url = f"https://export.arxiv.org/api/query?id_list={paper_id}"

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )

        if response.status_code != 200:
            print(f"HTTP {response.status_code}")
            continue

        if not response.text.strip():
            print("Empty response")
            continue

        root = ET.fromstring(response.text)

        entry = root.find("atom:entry", ns)

        if entry is None:
            print("No metadata found.")
            continue

        title = entry.find("atom:title", ns)
        published = entry.find("atom:published", ns)

        if title is None or published is None:
            print("Incomplete metadata.")
            continue

        title = title.text.strip()

        authors = []

        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns)
            if name is not None:
                authors.append(name.text.strip())

        year = published.text[:4]

        # Clean filename
        safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
        safe_title = re.sub(r"\s+", "_", safe_title)
        safe_title = safe_title[:150]

        new_name = f"{paper_id}_{safe_title}.pdf"

        pdf.rename(PDF_DIR / new_name)

        metadata.append([
            paper_id,
            title,
            ", ".join(authors),
            year,
            new_name
        ])

        print(f"✓ Renamed -> {new_name}")

    except Exception as e:
        print(f"Error: {e}")

    # Prevent API rate limiting
    time.sleep(3)

# Save metadata
with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Paper ID",
        "Title",
        "Authors",
        "Year",
        "Filename"
    ])

    writer.writerows(metadata)

print("\n===================================")
print(f"Finished! Renamed {len(metadata)} papers.")
print(f"Metadata saved to {CSV_FILE}")
print("===================================")