import json
import os
from pathlib import Path

# Paths to try
paths = [
    Path("storage/rag_hybrid_answers.json"),
    Path("papers download/rag_hybrid_answers.json"),
    Path("research-rag-assistant/storage/rag_hybrid_answers.json")
]

input_file = None
for p in paths:
    if p.exists():
        input_file = p
        break

if not input_file:
    # If the file hasn't been placed in the folder yet, try to look for the baseline file
    paths_baseline = [
        Path("storage/rag_retrieved_answers.json"),
        Path("papers download/rag_retrieved_answers.json"),
        Path("research-rag-assistant/storage/rag_retrieved_answers.json")
    ]
    for p in paths_baseline:
        if p.exists():
            input_file = p
            break

output_file = Path("papers download/rag_sample_qa_report.md")

if not input_file:
    print("⚠️ Could not find any RAG answers JSON file in the project. Please place 'rag_hybrid_answers.json' in your project's storage/ folder first.")
else:
    print(f"📖 Reading answers from: {input_file}")
    with open(input_file) as f:
        data = json.load(f)

    # Filter to valid answers
    items = []
    for qid, item in data.items():
        ans = item.get("rag_answer", "").strip()
        if ans and not ans.startswith("[ERROR"):
            items.append(item)

    print(f"💡 Found {len(items)} valid answers. Formatting the top 50...")

    md_lines = [
        "# 📝 RAG Sample Question-Answering Report",
        "",
        "This report shows a side-by-side comparison of the **Gold Standard (Expected) Answers** and the **RAG Generated Answers**.",
        "",
        "| No. | Question | Gold (Expected) Answer | RAG Generated Answer |",
        "| :--- | :--- | :--- | :--- |"
    ]

    for idx, item in enumerate(items[:50], start=1):
        q = item["question"].replace("\n", " ").strip()
        gold = item["expected_answer"].replace("\n", " ").strip()
        rag = item["rag_answer"].replace("\n", " ").strip()
        md_lines.append(f"| {idx} | {q} | {gold} | {rag} |")

    with open(output_file, "w") as f:
        f.write("\n".join(md_lines))
        
    print(f"🎉 Created beautiful Markdown report at: {output_file}")
