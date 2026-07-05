import re
import fitz  # PyMuPDF
from pathlib import Path

PDF_DIR = Path("pdfs")
CHECKLIST_PATH = Path("papers_list.md")

def get_uncompleted_papers():
    uncompleted = []
    if CHECKLIST_PATH.exists():
        with open(CHECKLIST_PATH, "r", encoding="utf-8") as f:
            for line in f:
                match = re.match(r'^\s*-\s*\[\s*\]\s*([a-zA-Z0-9\./\-_]+?)\s*-\s*(.*)$', line)
                if match:
                    uncompleted.append(match.group(1).strip())
    return uncompleted

def find_pdf_file(paper_id: str) -> Path:
    for p in PDF_DIR.glob("*.pdf"):
        if p.stem.split("_")[0] == paper_id:
            return p
    return None

def main():
    papers = get_uncompleted_papers()
    if not papers:
        print("No remaining papers found.")
        return

    word_counts = []
    print(f"Scanning {len(papers)} remaining papers to calculate averages...")

    for i, paper_id in enumerate(papers, 1):
        pdf_path = find_pdf_file(paper_id)
        if not pdf_path or not pdf_path.exists():
            continue

        try:
            doc = fitz.open(pdf_path)
            words = 0
            for page in doc:
                text = page.get_text()
                words += len(text.split())
            word_counts.append(words)
        except Exception:
            pass

    if not word_counts:
        print("No PDFs could be read.")
        return

    avg_words = sum(word_counts) / len(word_counts)
    # 1 word is typically ~1.33 tokens
    avg_tokens = avg_words * 1.33

    print("\n--- RESULTS ---")
    print(f"Total PDFs successfully scanned: {len(word_counts)}")
    print(f"Average Words per Paper: {avg_words:.1f}")
    print(f"Average Tokens per Paper (estimated): {avg_tokens:.1f}")
    
    # Suggesting optimal documents per chat
    # We target around 40k-50k tokens max for clean context
    suggested_docs = int(45000 / avg_tokens)
    print(f"Recommended number of papers per ChatGPT session: {suggested_docs}")

if __name__ == "__main__":
    main()
