import os
import re
import sys
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

# -----------------------------
# Configuration
# -----------------------------
INPUT_JSON  = Path("evaluation_dataset.json")
OUTPUT_JSON = Path("evaluation_results.json")
GEMINI_URL  = "https://gemini.google.com/app"

# How many QA pairs to judge per Gemini chat before starting a fresh one
# (avoids context overflow and rate limits)
PAIRS_PER_CHAT = 10

# Timeouts (seconds)
PROMPT_TIMEOUT   = 60   # max time to enter the prompt
SEND_TIMEOUT     = 60   # max time for send button to become active
RESPONSE_TIMEOUT = 100  # max time to wait for Gemini to finish responding

# -----------------------------
# Gemini selectors
# -----------------------------
GEMINI_INPUT_SELECTORS = [
    'rich-textarea div[contenteditable="true"]',
    'rich-textarea p',
    'div[contenteditable="true"]',
    '.ql-editor',
]
GEMINI_SEND_SELECTORS = [
    'button[aria-label="Send message"]',
    'button[data-mat-icon-name="send"]',
    'button.send-button',
    'button[jsname="Qj7are"]',
]
GEMINI_RESPONSE_SELECTORS = [
    'model-response .markdown',
    'model-response',
    '.response-content-chunk',
    '.model-response-text',
    'message-content',
]

# -----------------------------
# LLM-as-Judge prompt template
# -----------------------------
JUDGE_PROMPT_TEMPLATE = """You are an expert LLM-as-Judge evaluating the quality of QA pairs generated from academic research papers for a RAG (Retrieval-Augmented Generation) evaluation dataset.

For each QA pair below, rate it on these 4 dimensions using a score from 1 to 5:

1. **Question Quality** (1-5): Is the question clear, specific, unambiguous, and non-trivial?
2. **Answer Accuracy** (1-5): Is the expected answer factually correct and complete?
3. **Evidence Relevance** (1-5): Does the cited evidence (page/section/paragraph) logically support the answer?
4. **RAG Suitability** (1-5): Is this QA pair suitable for evaluating a RAG system (requires retrieval, not just common sense)?

Scoring guide:
- 5 = Excellent
- 4 = Good, minor issues
- 3 = Acceptable, some concerns
- 2 = Poor, significant issues
- 1 = Very poor / unusable

Return your evaluation ONLY as a valid JSON array. No markdown, no explanation outside the JSON.
Format:
[
  {{
    "question_id": "...",
    "question_quality": <1-5>,
    "answer_accuracy": <1-5>,
    "evidence_relevance": <1-5>,
    "rag_suitability": <1-5>,
    "overall_score": <average of above, 1 decimal>,
    "comment": "<one sentence critique or praise>"
  }}
]

Here are the QA pairs to evaluate:

{qa_pairs_text}
"""

# -----------------------------
# Helpers
# -----------------------------

def load_results() -> dict:
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON) as f:
            return json.load(f)
    return {}

def save_results(results: dict):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def format_qa_for_judge(qa_pairs: list) -> str:
    lines = []
    for i, qa in enumerate(qa_pairs, 1):
        evidence = qa.get("evidence", {})
        ev_str = ", ".join(f"{k}={v}" for k, v in evidence.items())
        lines.append(
            f"[{i}] question_id: {qa['question_id']}\n"
            f"    type: {qa.get('question_type','')}\n"
            f"    question: {qa['question']}\n"
            f"    expected_answer: {qa['expected_answer']}\n"
            f"    evidence: {ev_str}\n"
            f"    difficulty: {qa.get('difficulty','')}\n"
        )
    return "\n".join(lines)

def find_gemini_input(page):
    for sel in GEMINI_INPUT_SELECTORS:
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=1000):
                return el
        except Exception:
            pass
    return None

def find_send_button(page):
    for sel in GEMINI_SEND_SELECTORS:
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=1000):
                return el
        except Exception:
            pass
    return None

def get_last_response_text(page) -> str:
    for sel in GEMINI_RESPONSE_SELECTORS:
        try:
            els = page.locator(sel).all()
            if els:
                return els[-1].inner_text()
        except Exception:
            pass
    return ""

def dismiss_modals(page):
    dismiss_selectors = [
        'button[aria-label="Close"]',
        'button:has-text("Got it")',
        'button:has-text("Dismiss")',
        'button:has-text("OK")',
        'button:has-text("Accept")',
    ]
    for sel in dismiss_selectors:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=500):
                btn.click()
                time.sleep(0.5)
        except Exception:
            pass

def navigate_to_new_chat(page) -> bool:
    print("🔄 Opening fresh Gemini chat...")
    try:
        page.goto(GEMINI_URL, wait_until="domcontentloaded")
        page.wait_for_load_state("domcontentloaded")
        time.sleep(3)
        dismiss_modals(page)
        input_el = find_gemini_input(page)
        if input_el:
            print("✓ Fresh Gemini chat ready.")
            return True
        # Try waiting a bit more
        time.sleep(3)
        input_el = find_gemini_input(page)
        return input_el is not None
    except Exception as e:
        print(f"⚠️ Failed to navigate: {e}")
        return False

def send_to_gemini(page, prompt_text: str) -> str | None:
    """
    Sends prompt_text to Gemini and returns the response text.
    Returns None if any step times out.
    """

    # --- Step 1: Enter prompt (1 min timeout) ---
    print("  Entering prompt into Gemini...")
    prompt_start = time.time()
    input_el = find_gemini_input(page)
    if not input_el:
        print("  ❌ Could not find Gemini input box.")
        return None

    entered = False
    try:
        # Click to focus, select all existing, then insert text
        input_el.click()
        time.sleep(0.5)
        page.keyboard.press("Control+a")
        time.sleep(0.2)

        # Use JS insertText for speed and correctness with contenteditable
        input_el.evaluate("""(el, text) => {
            el.focus();
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(el);
            selection.removeAllRanges();
            selection.addRange(range);
            document.execCommand('insertText', false, text);
        }""", prompt_text)
        time.sleep(1)
        entered = True
    except Exception as e:
        print(f"  ⚠️ JS insert failed ({e}), trying keyboard type...")
        try:
            # Fallback: use clipboard paste via pyperclip alternative
            input_el.click()
            page.keyboard.press("Control+a")
            page.keyboard.press("Delete")
            # Type only first 2000 chars as fallback
            input_el.type(prompt_text[:2000], delay=0)
            entered = True
        except Exception as e2:
            print(f"  ❌ Keyboard type also failed: {e2}")

    if not entered or (time.time() - prompt_start) >= PROMPT_TIMEOUT:
        print("  ⏰ Prompt entry timed out.")
        navigate_to_new_chat(page)
        return None

    # --- Step 2: Wait for send button and click (1 min timeout) ---
    time.sleep(1)
    send_start = time.time()
    send_btn = None
    while time.time() - send_start < SEND_TIMEOUT:
        send_btn = find_send_button(page)
        if send_btn:
            try:
                if send_btn.is_enabled():
                    break
            except Exception:
                pass
        time.sleep(2)

    if not send_btn:
        print("  ⏰ Send button never became active.")
        navigate_to_new_chat(page)
        return None

    try:
        send_btn.click()
    except Exception as e:
        print(f"  ⚠️ Send click failed: {e}")
        # Try pressing Enter as fallback
        try:
            input_el.press("Enter")
        except Exception:
            navigate_to_new_chat(page)
            return None

    time.sleep(5)  # let generation start

    # --- Step 3: Wait for response (100 sec timeout) ---
    print(f"  Waiting for Gemini response (timeout={RESPONSE_TIMEOUT}s)...")
    resp_start = time.time()
    last_text = ""
    stable_ticks = 0
    last_heartbeat = time.time()

    while True:
        current_text = get_last_response_text(page)

        if current_text and current_text == last_text:
            stable_ticks += 1
        else:
            stable_ticks = 0
            if current_text:
                last_text = current_text

        # Done when text has been stable for 6 seconds and is substantial
        if stable_ticks >= 2 and len(current_text.strip()) > 50:
            break

        elapsed = time.time() - resp_start
        if elapsed >= RESPONSE_TIMEOUT:
            print(f"  ⏰ No response after {int(elapsed)}s.")
            navigate_to_new_chat(page)
            return None

        # Heartbeat every 20 seconds
        if time.time() - last_heartbeat >= 20:
            words = len(current_text.split()) if current_text else 0
            print(f"  ⏳ Still generating... ({int(elapsed)}s, {words} words)")
            last_heartbeat = time.time()
            dismiss_modals(page)

        time.sleep(3)

    return current_text.strip()

def parse_judge_response(response_text: str) -> list:
    """Extract JSON array from Gemini's response."""
    # Try direct parse
    try:
        return json.loads(response_text)
    except Exception:
        pass
    # Try extracting from code block
    match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # Try greedy extract of array
    match = re.search(r'(\[.*\])', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return []

# -----------------------------
# Main
# -----------------------------

def run_evaluation():
    # Load dataset
    if not INPUT_JSON.exists():
        print(f"❌ {INPUT_JSON} not found.")
        sys.exit(1)

    with open(INPUT_JSON) as f:
        dataset = json.load(f)

    # Load existing results to resume
    results = load_results()

    # Collect all QA pairs that haven't been judged yet
    all_qa_pairs = []
    for paper_id, paper_data in dataset.items():
        for qa in paper_data.get("qa_pairs", []):
            qid = qa["question_id"]
            if qid not in results:
                all_qa_pairs.append(qa)

    total = len(all_qa_pairs)
    if total == 0:
        print("✅ All QA pairs already evaluated!")
        print_summary(results)
        return

    print(f"Found {total} QA pairs to evaluate.")
    print(f"Connecting to Chrome on port 9222...")

    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
        except Exception as e:
            print(f"❌ Could not connect to Chrome: {e}")
            print("Make sure Chrome is running with: --remote-debugging-port=9222")
            sys.exit(1)

        context = browser.contexts[0]

        # Find or open a Gemini tab
        page = None
        for pg in context.pages:
            if "gemini.google.com" in pg.url:
                page = pg
                break
        if not page:
            page = context.new_page()

        if not navigate_to_new_chat(page):
            print("❌ Could not open Gemini. Make sure you are logged in to gemini.google.com")
            sys.exit(1)

        processed_in_chat = 0
        evaluated = 0

        # Process in batches of PAIRS_PER_CHAT
        for batch_start in range(0, total, PAIRS_PER_CHAT):
            batch = all_qa_pairs[batch_start: batch_start + PAIRS_PER_CHAT]
            batch_ids = [qa["question_id"] for qa in batch]

            print(f"\n{'='*70}")
            print(f"Batch {batch_start // PAIRS_PER_CHAT + 1}: evaluating {len(batch)} QA pairs")
            print(f"  IDs: {batch_ids[0]} ... {batch_ids[-1]}")
            print(f"{'='*70}")

            # Start fresh chat every PAIRS_PER_CHAT to avoid context overflow
            if processed_in_chat >= PAIRS_PER_CHAT:
                navigate_to_new_chat(page)
                processed_in_chat = 0

            qa_text = format_qa_for_judge(batch)
            prompt = JUDGE_PROMPT_TEMPLATE.format(qa_pairs_text=qa_text)

            response = send_to_gemini(page, prompt)

            if response is None:
                print(f"  ⚠️ Skipping batch (timeout/error).")
                continue

            judgments = parse_judge_response(response)

            if not judgments:
                print(f"  ⚠️ Could not parse Gemini response as JSON. Raw snippet:")
                print(f"  {response[:300]}")
                continue

            # Store results
            for judgment in judgments:
                qid = judgment.get("question_id")
                if qid:
                    results[qid] = judgment
                    evaluated += 1

            save_results(results)
            processed_in_chat += len(batch)

            print(f"  ✓ Judged {len(judgments)} pairs. Total evaluated: {evaluated}/{total}")
            time.sleep(2)

        browser.close()

    print(f"\n✅ Evaluation complete! {evaluated}/{total} QA pairs judged.")
    print(f"Results saved to: {OUTPUT_JSON}")
    print_summary(results)


def print_summary(results: dict):
    if not results:
        return
    scores = {
        "question_quality": [],
        "answer_accuracy": [],
        "evidence_relevance": [],
        "rag_suitability": [],
        "overall_score": [],
    }
    for qid, r in results.items():
        for key in scores:
            val = r.get(key)
            if val is not None:
                try:
                    scores[key].append(float(val))
                except Exception:
                    pass

    print(f"\n{'='*50}")
    print(f"  PIPELINE ACCURACY SUMMARY ({len(results)} QA pairs judged)")
    print(f"{'='*50}")
    for metric, vals in scores.items():
        if vals:
            avg = sum(vals) / len(vals)
            label = metric.replace("_", " ").title()
            bar = "█" * int(avg) + "░" * (5 - int(avg))
            print(f"  {label:<22} {bar}  {avg:.2f}/5.0")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_evaluation()
