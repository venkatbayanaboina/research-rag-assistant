import os
import re
import sys
import json
import time
from pathlib import Path
import fitz  # PyMuPDF
from playwright.sync_api import sync_playwright

# -----------------------------
# Configuration & Setup
# -----------------------------
PDF_DIR = Path("pdfs")
CHECKLIST_PATH = Path("papers_list.md")
OUTPUT_JSON = Path("evaluation_dataset.json")

def find_pdf_file(paper_id: str) -> Path:
    """Finds the local PDF matching the given arXiv paper_id."""
    for p in PDF_DIR.glob("*.pdf"):
        stem = p.stem
        parts = stem.split("_")
        if parts[0] == paper_id:
            return p
    return None

def determine_allocations(has_figures: bool, has_tables: bool, has_equations: bool) -> dict:
    """Calculates the exact QA type count distribution to total exactly 15 questions."""
    if has_figures and has_tables and has_equations:
        return {"text": 4, "figure": 4, "table": 3, "equation": 4}
    elif has_figures and has_equations:
        return {"text": 5, "figure": 5, "table": 0, "equation": 5}
    elif has_tables and has_equations:
        return {"text": 5, "figure": 0, "table": 5, "equation": 5}
    elif has_figures and has_tables:
        return {"text": 5, "figure": 5, "table": 5, "equation": 0}
    elif has_figures:
        return {"text": 8, "figure": 7, "table": 0, "equation": 0}
    elif has_tables:
        return {"text": 8, "figure": 0, "table": 7, "equation": 0}
    elif has_equations:
        return {"text": 8, "figure": 0, "table": 0, "equation": 7}
    else:
        return {"text": 15, "figure": 0, "table": 0, "equation": 0}

def table_to_markdown(table_data) -> str:
    """Converts raw table row list data to a clean Markdown table format."""
    if not table_data:
        return ""
    # Filter out empty rows
    table_data = [row for row in table_data if any(x is not None and str(x).strip() for x in row)]
    if not table_data:
        return ""
    
    # Identify number of columns from headers
    headers = [str(x).strip() if x is not None else "" for x in table_data[0]]
    num_cols = len(headers)
    
    markdown = "\n"
    markdown += "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * num_cols) + " |\n"
    
    for row in table_data[1:]:
        row_strs = [str(x).strip() if x is not None else "" for x in row]
        # Pad row if it has fewer columns than headers
        if len(row_strs) < num_cols:
            row_strs += [""] * (num_cols - len(row_strs))
        # Truncate row if it has more columns
        row_strs = row_strs[:num_cols]
        markdown += "| " + " | ".join(row_strs) + " |\n"
    markdown += "\n"
    return markdown

def extract_page_content(page_obj, page_num: int) -> str:
    """Extracts text, formats any tables as markdown, and detects image placements."""
    text = page_obj.get_text()
    
    # Prefix text with page marker to help ChatGPT's evidence mapping
    content = f"--- PAGE {page_num} START ---\n{text}\n"
    
    # Extract tables
    try:
        tables = page_obj.find_tables()
        if tables:
            content += f"\n\n### [Page {page_num} Extracted Tables]\n"
            for t_idx, table in enumerate(tables, 1):
                data = table.extract()
                md_table = table_to_markdown(data)
                if md_table.strip():
                    content += f"#### Table {t_idx} (Markdown Format):\n{md_table}"
    except Exception:
        pass
        
    # Detect figures/images on this page
    try:
        images = page_obj.get_images()
        if images:
            content += f"\n\n[Figure/Image present on Page {page_num}: {len(images)} raw images detected]\n"
    except Exception:
        pass
        
    content += f"--- PAGE {page_num} END ---\n\n"
    return content

def dismiss_chatgpt_modals(page):
    """Automatically dismisses common ChatGPT overlay popups/modals."""
    modal_selectors = [
        'button:has-text("Got it")',
        'button:has-text("Dismiss")',
        'button:has-text("Close")',
        'button:has-text("Okay")',
        'button:has-text("Stay on Free")',
        'button:has-text("Keep using")',
        'div[role="dialog"] button:has-text("Got it")',
        'div[role="dialog"] button:has-text("Dismiss")',
        'div[role="dialog"] button:has-text("Close")'
    ]
    for sel in modal_selectors:
        try:
            # We use a short timeout of 500ms since we check proactively
            btn = page.locator(sel).first
            if btn.is_visible(timeout=500):
                print(f"Dismissing ChatGPT modal popup using selector: {sel}")
                btn.click()
                time.sleep(1)
        except Exception:
            pass

def open_new_chat(page) -> bool:
    """Robustly opens a new/fresh ChatGPT chat session.
    Tries multiple methods (clicking sidebar New Chat buttons, navigating with cache-busting params).
    """
    print("🔄 Navigating/opening a fresh ChatGPT session...")
    
    # Try Method 1: Click the sidebar/header 'New chat' button if visible
    new_chat_selectors = [
        '[data-testid="sidebar-new-chat-button"]',
        'a[href="/"]',
        'button[aria-label="New chat"]',
        'a[data-testid="new-chat-button"]'
    ]
    for sel in new_chat_selectors:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=1000):
                print(f"Clicking new chat button using selector: {sel}")
                btn.click()
                time.sleep(2)
                # Verify prompt textarea is present
                page.wait_for_selector("#prompt-textarea", timeout=5000)
                dismiss_chatgpt_modals(page)
                print("✓ Successfully started fresh chat via UI click.")
                return True
        except Exception:
            pass

    # Try Method 2: Navigate with unique timestamp query parameter to force route/caching reset
    try:
        t_val = int(time.time())
        page.goto(f"https://chatgpt.com/?t={t_val}", wait_until="domcontentloaded")
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_selector("#prompt-textarea", timeout=15000)
        dismiss_chatgpt_modals(page)
        print("✓ Successfully loaded fresh chat via cache-busting navigation.")
        return True
    except Exception as e:
        print(f"⚠️ Query-parameter navigation failed: {e}. Trying standard fallback navigation...")
        
    # Try Method 3: Standard fallback goto
    try:
        page.goto("https://chatgpt.com", wait_until="domcontentloaded")
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_selector("#prompt-textarea", timeout=15000)
        dismiss_chatgpt_modals(page)
        print("✓ Successfully loaded fresh chat via fallback navigation.")
        return True
    except Exception as fallback_err:
        print(f"❌ Failed to load fresh chat: {fallback_err}")
        return False

def mark_completed(paper_id: str):
    """Updates papers_list.md checklist to mark the given paper completed."""
    if CHECKLIST_PATH.exists():
        with open(CHECKLIST_PATH, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            
        for i, line in enumerate(lines):
            match = re.match(r'^\s*-\s*\[\s*\]\s*([a-zA-Z0-9\./\-_]+?)\s*-\s*(.*)$', line)
            if match and match.group(1).strip() == paper_id:
                title = match.group(2).strip()
                lines[i] = f"- [x] {paper_id} - {title}"
                break
                
        with open(CHECKLIST_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

def run_automation():
    # 1. Read uncompleted papers from papers_list.md
    if not CHECKLIST_PATH.exists():
        print(f"❌ Error: checklist file not found at {CHECKLIST_PATH}.")
        sys.exit(1)

    unprocessed = []
    with open(CHECKLIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r'^\s*-\s*\[\s*\]\s*([a-zA-Z0-9\./\-_]+?)\s*-\s*(.*)$', line)
            if match:
                unprocessed.append((match.group(1).strip(), match.group(2).strip()))

    print(f"Found {len(unprocessed)} papers remaining to process.")
    if not unprocessed:
        print("All papers have been processed and marked.")
        return

    # Load existing dataset results
    all_dataset_records = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                all_dataset_records = json.load(f)
        except Exception:
            pass

    print("Connecting to your running Chrome instance via CDP on port 9222...")
    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
        except Exception as e:
            print("❌ Failed to connect to Chrome. Make sure Chrome is running with remote debugging enabled:")
            print("Run this command in terminal first:")
            print("/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222 --user-data-dir=\"/Users/nanibayanaboina2750/ChromeProfile\"")
            print(f"Error details: {e}")
            sys.exit(1)

        # Get default context or create page
        context = browser.contexts[0]
        page = context.new_page()
        
        print("Navigating to ChatGPT...")
        page.goto("https://chatgpt.com", wait_until="domcontentloaded")
        page.wait_for_load_state("domcontentloaded")
        
        # Verify if we are logged in by checking for the input box
        try:
            page.wait_for_selector("#prompt-textarea", timeout=10000)
            print("✓ Successfully connected to ChatGPT and ready for automation.")
        except Exception:
            print("⚠️ Prompt textarea not found. Please log in manually to ChatGPT in the Chrome window first, then run this script again.")
            browser.close()
            sys.exit(1)

        processed_in_chat = 0
        for idx, (paper_id, title) in enumerate(unprocessed, start=1):
            # Refresh chat session every 1 paper to prevent context bloat and speed up responses
            if processed_in_chat >= 1:
                print("\n🔄 Starting a new ChatGPT session to clear context memory...")
                if open_new_chat(page):
                    processed_in_chat = 0
                else:
                    print("⚠️ Warning: Failed to refresh chat context. Trying to continue anyway...")
                    
            print("\n" + "="*80)
            print(f"[{idx}/{len(unprocessed)}] Processing: {paper_id} - {title}")
            print("="*80)

            pdf_path = find_pdf_file(paper_id)
            if not pdf_path or not pdf_path.exists():
                print(f"⚠️ Warning: PDF for paper {paper_id} not found. Skipping.")
                continue

            processed_in_chat += 1

            # Extract layout details via PyMuPDF to calculate exact allocations
            try:
                doc = fitz.open(pdf_path)
                num_pages = len(doc)
                full_text = ""
                for page_num in range(1, num_pages + 1):
                    page_obj = doc[page_num - 1]
                    full_text += extract_page_content(page_obj, page_num)
            except Exception as e:
                print(f"❌ Failed to parse PDF text: {e}")
                continue

            figures = set(re.findall(r'\b(?:Figure|Fig\.)\s+(\d+)\b', full_text, re.IGNORECASE))
            tables = set(re.findall(r'\bTable\s+(\d+)\b', full_text, re.IGNORECASE))
            equations = set(re.findall(r'\b(?:Equation|Eq\.)\s+\(?(\d+)\)?\b', full_text, re.IGNORECASE))
            eq_ends = re.findall(r'\(\s*(\d+)\s*\)\s*$', full_text, re.MULTILINE)
            equations.update(eq_ends)

            has_figs = len(figures) > 0
            has_tbls = len(tables) > 0
            has_eqs = len(equations) > 0

            allocations = determine_allocations(has_figs, has_tbls, has_eqs)
            print(f"Modalities: Pages={num_pages}, Figures={len(figures)}, Tables={len(tables)}, Equations={len(equations)}")
            print(f"Allocating QA counts: {allocations}")

            prompt_text = f"""Analyze the uploaded academic paper and generate exactly 15 high-quality, research-grade evaluation QA pairs.

Paper ID: {paper_id}
Paper Title: {title}
Total Pages: {num_pages}

You MUST generate exactly the following counts for each question type:
- 'text': {allocations['text']} questions
- 'figure': {allocations['figure']} questions (Must reference actual figure numbers found in the paper, e.g. Figure 1, Figure 2)
- 'table': {allocations['table']} questions (Must reference actual table numbers found in the paper, e.g. Table 1, Table 2)
- 'equation': {allocations['equation']} questions (Must reference actual equation numbers found in the paper, e.g. Equation (1), Equation (5))

Total: 15 questions.

Difficulty Ratings:
- 'easy': Direct lookup of facts, hyper-parameters, datasets, or simple metrics.
- 'medium': Connecting information across paragraphs, sections, or comparing results.
- 'hard': Reasoning over math/equations, visual figure workflows, or complex multi-row tables.

Each question MUST include precise evidence mapping:
- page: 1-indexed page number containing the evidence (must be between 1 and {num_pages}).
- section: Section name/header (e.g. 'Methodology'), if type is 'text'.
- paragraph: 1-indexed paragraph number within the section, if type is 'text'.
- figure: e.g. 'Figure 3' if type is 'figure'.
- table: e.g. 'Table 2' if type is 'table'.
- equation: e.g. 'Equation (5)' if type is 'equation'.

Only cite figures, tables, and equations that actually appear in the text. Do not invent figure, table, or equation numbers.
Note: The references or bibliography section at the end of the text may be truncated due to length constraints. This is expected and normal, and you must generate the required QA pairs using the main body text of the paper.

You MUST output a JSON object conforming exactly to this JSON Schema (enclose it in standard ```json ... ``` code blocks):
{{
  "paper_id": "{paper_id}",
  "paper_title": "{title}",
  "qa_pairs": [
    {{
      "question_id": "string (format: {paper_id}_Q01 to {paper_id}_Q15)",
      "question_type": "text" | "figure" | "table" | "equation",
      "question": "string",
      "expected_answer": "string",
      "evidence": {{
        "page": integer,
        "section": "string" (optional),
        "paragraph": integer (optional),
        "figure": "string" (optional),
        "table": "string" (optional),
        "equation": "string" (optional)
      }},
      "difficulty": "easy" | "medium" | "hard"
    }}
  ]
}}
"""

            # Skip paper if character count exceeds 120,000
            if len(full_text) > 120000:
                print(f"⏭️ Skipping paper: text size ({len(full_text):,} characters) exceeds 120,000 limit.")
                continue

            print(f"Pasting full paper text ({len(full_text):,} chars)...")
            prompt_text += f"\n\nHere is the full text of the academic paper for your analysis:\n\n[PAPER TEXT START]\n{full_text}\n[PAPER TEXT END]"
            
            # Dismiss any popup modal that might have appeared and is blocking the textarea
            dismiss_chatgpt_modals(page)
            
            def reset_chat_on_timeout(reason):
                """Opens a fresh chat and returns True so the caller can continue to next paper."""
                print(f"⏰ {reason}. Starting fresh chat and skipping paper...")
                if open_new_chat(page):
                    print("✓ Fresh chat opened.")
                else:
                    print("⚠️ Failed to open fresh chat.")

            # Step 2: Input prompt (60-second timeout)
            print("Entering prompt...")
            prompt_entered = False
            prompt_start = time.time()
            textarea = page.locator("#prompt-textarea")
            try:
                textarea.evaluate(f"""(el, text) => {{
                    el.focus();
                    const selection = window.getSelection();
                    const range = document.createRange();
                    range.selectNodeContents(el);
                    selection.removeAllRanges();
                    selection.addRange(range);
                    document.execCommand('insertText', false, text);
                }}""", prompt_text)
                time.sleep(2)
                prompt_entered = True
            except Exception as e:
                print(f"⚠️ JS insertion failed: {e}. Trying Playwright fill...")
                try:
                    textarea.fill(prompt_text, timeout=60000)
                    prompt_entered = True
                except Exception as fill_err:
                    print(f"❌ Fill also failed after 60s: {fill_err}")

            if not prompt_entered or (time.time() - prompt_start) >= 60:
                reset_chat_on_timeout("Prompt entry timed out after 60 seconds")
                processed_in_chat = 0
                continue

            # Wait for send button to become enabled (60-second timeout)
            time.sleep(2)
            send_btn = page.locator('button[data-testid="send-button"], button[data-testid="fruitjuice-send-button"], button[aria-label="Send prompt"]')
            send_wait_start = time.time()
            upload_finished = False
            while time.time() - send_wait_start < 60:
                try:
                    if send_btn.first.is_visible() and send_btn.first.is_enabled():
                        upload_finished = True
                        break
                except Exception:
                    pass
                time.sleep(3)

            if not upload_finished:
                reset_chat_on_timeout("Send button did not activate within 60 seconds")
                processed_in_chat = 0
                continue
                
            # Step 3: Wait for response completion — 60-second hard timeout
            print("Sending prompt and waiting for generation to start...")
            send_btn.first.click()
            time.sleep(10) # wait for submission to go through and generation to start
            print("Monitoring response completion (60s timeout if stuck)...")
            
            start_wait = time.time()
            last_text = ""
            stable_ticks = 0
            heartbeat_interval = 30  # print heartbeat every 30 seconds
            last_heartbeat = time.time()
            RESPONSE_TIMEOUT = 60  # 60 seconds hard timeout
            timed_out = False
            
            while True:
                # A. Check active streaming indicator
                is_streaming = False
                try:
                    if page.locator(".result-streaming").count() > 0:
                        is_streaming = True
                except Exception:
                    pass
                
                # B. Check text content of last assistant turn
                current_text = ""
                try:
                    elements = page.locator('div.markdown, div.prose').all()
                    if elements:
                        current_text = elements[-1].inner_text()
                except Exception:
                    pass
                
                # Text is stable only when content matches last check AND not actively streaming
                if current_text and current_text == last_text and not is_streaming:
                    stable_ticks += 1
                else:
                    stable_ticks = 0
                    if current_text:
                        last_text = current_text
                    
                # Exit when response has been completely stable for 6 seconds
                if stable_ticks >= 2 and len(current_text.strip()) > 100:
                    break
                
                now = time.time()
                elapsed = now - start_wait
                
                # Hard 60-second timeout — start fresh chat and skip paper
                if elapsed >= RESPONSE_TIMEOUT:
                    print(f"⏰ No response after {int(elapsed)}s. Starting fresh chat and skipping paper...")
                    timed_out = True
                    if open_new_chat(page):
                        processed_in_chat = 0
                        print("✓ Fresh chat opened after timeout.")
                    else:
                        print("⚠️ Failed to open fresh chat.")
                    break
                
                # Print heartbeat every 30 seconds
                if now - last_heartbeat >= heartbeat_interval:
                    elapsed_min = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
                    words_so_far = len(current_text.split()) if current_text else 0
                    remaining = int(RESPONSE_TIMEOUT - elapsed)
                    print(f"⏳ Still generating... ({elapsed_min} elapsed, {words_so_far} words, timeout in {remaining}s)")
                    last_heartbeat = now
                    dismiss_chatgpt_modals(page)
                
                time.sleep(3)
            
            if timed_out:
                continue

            print("Extracting response...")
            # Fetch the last assistant message (the latest turn)
            last_text = ""
            for selector in ['div.markdown, div.prose']:
                try:
                    last_el = page.locator(selector).last
                    if last_el.is_visible():
                        last_text = last_el.inner_text()
                        if last_text.strip():
                            break
                except Exception:
                    pass
            
            # Extract JSON block using greedy patterns (safe because last_text is only the last bubble)
            json_blocks = re.findall(r'```json\s*(.*?)\s*```', last_text, re.DOTALL)
            if not json_blocks and last_text:
                json_blocks = re.findall(r'({.*})', last_text, re.DOTALL)
                
            target_block = json_blocks[-1] if json_blocks else None
            
            # Fallback to scraping the body, but filtering for the current paper_id
            if not target_block:
                print("⚠️ Retrying extraction from the entire page body...")
                try:
                    page_text = page.locator("body").inner_text()
                    body_json_blocks = re.findall(r'```json\s*(.*?)\s*```', page_text, re.DOTALL)
                    if not body_json_blocks:
                        body_json_blocks = re.findall(r'({.*})', page_text, re.DOTALL)
                    
                    valid_blocks = []
                    for block in body_json_blocks:
                        if paper_id in block and "qa_pairs" in block:
                            valid_blocks.append(block)
                    if valid_blocks:
                        target_block = valid_blocks[-1]
                except Exception:
                    pass
            
            if not target_block:
                print("❌ Failed to find a JSON block in the ChatGPT response. Skipping.")
                continue
                
            try:
                # Parse JSON
                result_data = json.loads(target_block.strip())
                
                # Validation & structural fixes
                validated_pairs = []
                for qa_idx, pair in enumerate(result_data.get("qa_pairs", []), start=1):
                    pair["question_id"] = f"{paper_id}_Q{qa_idx:02d}"
                    evidence = pair.get("evidence", {})
                    page_val = evidence.get("page", 1)
                    if page_val < 1 or page_val > num_pages:
                        evidence["page"] = max(1, min(page_val, num_pages))
                    validated_pairs.append(pair)
                    
                result_data["qa_pairs"] = validated_pairs
                result_data["paper_id"] = paper_id
                result_data["paper_title"] = title
                
                # Save to evaluation_dataset.json
                all_dataset_records[paper_id] = result_data
                with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(all_dataset_records, f, indent=2)
                    
                print(f"✓ Successfully generated 15 QA pairs for {paper_id}.")
                
                # Mark as completed in papers_list.md
                mark_completed(paper_id)
                print(f"✓ Marked {paper_id} completed in {CHECKLIST_PATH}.")
                
            except Exception as e:
                print(f"❌ Failed to parse/save response: {e}")
                if target_block:
                    print("--- Raw Extracted JSON Block Preview ---")
                    print(target_block[:1000])
                    print("----------------------------------------")
                
            # Sleep 20 seconds between papers to behave nicely and avoid rate limits
            time.sleep(20)

if __name__ == "__main__":
    run_automation()
