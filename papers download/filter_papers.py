from pathlib import Path

PAPER_DOMAINS = [
    "arxiv.org",
    "doi.org",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "springer.com",
    "link.springer.com",
    "usenix.org",
    "openreview.net",
    "papers.nips.cc",
    "proceedings.mlr.press",
    "jmlr.org",
    "aclweb.org",
    "aclanthology.org",
    "cv-foundation.org",
    "openaccess.thecvf.com",
    "nature.com",
    "science.org",
]

paper_links = []

with open("metadata/all_links.txt", "r", encoding="utf-8") as f:
    for line in f:
        url = line.strip()

        if any(domain in url for domain in PAPER_DOMAINS):
            paper_links.append(url)

paper_links = sorted(set(paper_links))

with open("metadata/paper_links.txt", "w", encoding="utf-8") as f:
    for url in paper_links:
        f.write(url + "\n")

print(f"Paper links found: {len(paper_links)}")
print("Saved to metadata/paper_links.txt")