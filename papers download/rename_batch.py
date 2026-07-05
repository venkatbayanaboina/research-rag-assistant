import csv
import re
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
PDF_DIR = Path("pdfs")
OUTPUT_CSV = "metadata.csv"
LINKS_TXT = Path("metadata/paper_links.txt")

BATCH_SIZE = 50
REQUEST_DELAY = 3          # seconds between successful batches
RATE_LIMIT_DELAY = 60      # seconds after HTTP 429

HEADERS = {
    "User-Agent": "Research-RAG/1.0 (research project)"
}

ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom"
}

# -----------------------------
# Build Link-to-Query ID Map
# -----------------------------
print("Building arXiv ID maps from links metadata...")
url_map = {}
if LINKS_TXT.exists():
    with open(LINKS_TXT, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if "arxiv.org/abs/" not in url:
                continue
            
            # Extract full ID (e.g., 'cs/0309048' or '1511.05644v1')
            full_id = url.split("arxiv.org/abs/")[-1]
            base_id = full_id.split("/")[-1]
            base_id_clean = re.sub(r"v\d+$", "", base_id)
            
            url_map[base_id_clean] = full_id
    print(f"Mapped {len(url_map)} unique IDs from paper_links.txt.")
else:
    print(f"⚠️ Warning: {LINKS_TXT} not found. Fallback mode active.")

# -----------------------------
# Read Remaining PDFs
# -----------------------------
paper_map = {}
for pdf in PDF_DIR.glob("*.pdf"):
    stem = pdf.stem
    
    # Skip already renamed files
    if "_" in stem:
        continue
        
    normalized = re.sub(r"v\d+$", "", stem)
    paper_map[normalized] = pdf

# Sort paper IDs so execution order is predictable
paper_stems = sorted(paper_map.keys())
print(f"Need to process {len(paper_stems)} unrenamed papers.")

if len(paper_stems) == 0:
    print("Everything already renamed.")
    exit()

# -----------------------------
# Resolve Full Query IDs
# -----------------------------
# Query IDs list contains the full category-prefixed IDs (e.g. 'cs/0309048') to prevent HTTP 400
query_ids = []
query_to_stem = {}
for stem in paper_stems:
    full_query_id = url_map.get(stem, stem)
    query_ids.append(full_query_id)
    query_to_stem[full_query_id] = stem

# -----------------------------
# Split into batches
# -----------------------------
batches = [
    query_ids[i:i+BATCH_SIZE]
    for i in range(0, len(query_ids), BATCH_SIZE)
]

print(f"Created {len(batches)} batches of size {BATCH_SIZE}.\n")

# -----------------------------
# Metadata CSV
# -----------------------------
csv_exists = Path(OUTPUT_CSV).exists()
csv_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
writer = csv.writer(csv_file)

if not csv_exists:
    writer.writerow(["Paper ID", "Title", "Authors", "Year", "Filename"])

# -----------------------------
# Helper: Query API with urllib
# -----------------------------
def query_arxiv_api(ids):
    url = f"https://export.arxiv.org/api/query?id_list={','.join(ids)}&max_results=100"
    while True:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as response:
                if response.status == 200:
                    return 200, response.read().decode("utf-8")
                elif response.status == 429:
                    print(f"\nHTTP 429 - Rate limited. Sleeping {RATE_LIMIT_DELAY} seconds...")
                    time.sleep(RATE_LIMIT_DELAY)
                    continue
                else:
                    return response.status, ""
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"\nHTTP 429 - Rate limited. Sleeping {RATE_LIMIT_DELAY} seconds...")
                time.sleep(RATE_LIMIT_DELAY)
                continue
            else:
                return e.code, ""
        except Exception as e:
            return 500, str(e)

# -----------------------------
# Process Batches
# -----------------------------
for batch_number, batch in enumerate(batches, start=1):
    print("=" * 70)
    print(f"Batch {batch_number}/{len(batches)}")
    print("=" * 70)
    
    status, xml_data = query_arxiv_api(batch)
    if status != 200:
        print(f"❌ Batch {batch_number} failed with HTTP {status}. Skipping batch.")
        continue
        
    try:
        root = ET.fromstring(xml_data)
    except Exception as e:
        print("XML Parse Error:", e)
        continue
        
    entries = root.findall("atom:entry", ARXIV_NS)
    if not entries:
        print("No entries returned.")
        continue
        
    metadata = {}
    for entry in entries:
        atom_id_elem = entry.find("atom:id", ARXIV_NS)
        if atom_id_elem is None or not atom_id_elem.text:
            continue
            
        api_id = atom_id_elem.text.split("/")[-1]
        paper_id_clean = re.sub(r"v\d+$", "", api_id)
        
        title_elem = entry.find("atom:title", ARXIV_NS)
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Unknown Title"
        # Normalize title: remove inner newlines/excess spaces
        title = re.sub(r"\s+", " ", title)
        
        published_elem = entry.find("atom:published", ARXIV_NS)
        year = published_elem.text[:4] if published_elem is not None and published_elem.text else "Unknown"
        
        authors = []
        for author in entry.findall("atom:author", ARXIV_NS):
            name_elem = author.find("atom:name", ARXIV_NS)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
                
        metadata[paper_id_clean] = {
            "title": title,
            "authors": authors,
            "year": year
        }
        
    # Rename PDFs and write to CSV
    for query_id in batch:
        stem = query_to_stem[query_id]
        pdf = paper_map.get(stem)
        if not pdf or not pdf.exists():
            continue
            
        # Match using base ID clean
        base_id_clean = re.sub(r"v\d+$", "", query_id.split("/")[-1])
        
        if base_id_clean not in metadata:
            print(f"⚠️ Metadata missing for ID: {query_id} (stem: {stem})")
            continue
            
        info = metadata[base_id_clean]
        safe_title = re.sub(r'[\\/*?:"<>|]', "", info["title"])
        safe_title = re.sub(r"\s+", "_", safe_title)
        
        # Limit title length to prevent filesystem limits
        if len(safe_title) > 120:
            safe_title = safe_title[:120]
            
        new_name = f"{stem}_{safe_title}.pdf"
        new_path = PDF_DIR / new_name
        
        try:
            pdf.rename(new_path)
            writer.writerow([
                stem,
                info["title"],
                ", ".join(info["authors"]),
                info["year"],
                new_name
            ])
            print(f"✓ Renamed: {stem} -> {new_name}")
        except Exception as e:
            print(f"❌ Failed to rename {stem}: {e}")
            
    csv_file.flush()
    print(f"\nBatch {batch_number} complete.")
    
    if batch_number != len(batches):
        print(f"Sleeping {REQUEST_DELAY} seconds...\n")
        time.sleep(REQUEST_DELAY)

csv_file.close()
print("\n======================================")
print("Finished!")
print("metadata.csv updated successfully.")
print("======================================")