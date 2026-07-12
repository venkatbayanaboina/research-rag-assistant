import re
from pathlib import Path

# Read README.md
readme_path = Path("/Users/nanibayanaboina2750/Desktop/research-rag-assistant/evaluation_suite/research papers.md")

with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

# Find all URLs
urls = re.findall(r'https?://[^\s)"]+', content)

# Remove duplicates
urls = sorted(set(urls))

print(f"Total URLs found: {len(urls)}")

# Save all URLs
Path("metadata").mkdir(exist_ok=True)

with open("metadata/all_links.txt", "w", encoding="utf-8") as f:
    for url in urls:
        f.write(url + "\n")

print("Saved to metadata/all_links.txt")