import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAG_OUTPUT = ROOT / "evaluation_suite" / "rag_retrieved_answers.json"

# ── Selectors ────────────────────────────────────────────────────────────────
GEMINI_CONFIG = {
    "url": "https://gemini.google.com/app",
    "inputs": [
        'rich-textarea div[contenteditable="true"]',
        'rich-textarea p',
        'div[contenteditable="true"]',
    ],
    "sends": [
        'button[aria-label="Send message"]',
        'button[data-mat-icon-name="send"]',
        'button[jsname="Qj7are"]',
    ],
    "responses": [
        'model-response .markdown',
        'model-response',
        'message-content',
        'div.markdown',
    ],
    "streaming": "div[aria-label='Gemini is responding'], .loading, .thinking"
}

CHATGPT_CONFIG = {
    "url": "https://chatgpt.com",
    "inputs": [
        '#prompt-textarea',
        'div[contenteditable="true"]',
    ],
    "sends": [
        'button[data-testid="send-button"]',
        'button[aria-label="Send prompt"]',
    ],
    "responses": [
        'div.markdown',
        'div.prose',
        '.markdown',
    ],
    "streaming": "button[aria-label='Stop generating'], .streaming"
}

# ── Helper Functions ──────────────────────────────────────────────────────────

def load_data() -> dict:
    if not RAG_OUTPUT.exists():
        print(f"❌ {RAG_OUTPUT} not found. Run run_rag_evaluation.py first.")
        sys.exit(1)
    with open(RAG_OUTPUT) as f:
        return json.load(f)

def save_data(data: dict, out_path: Path):
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def find_element(page, selectors: list):
    for sel in selectors:
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=1000):
                return el
        except Exception:
            pass
    return None

def get_last_response_text(page, selectors: list) -> str:
    for sel in selectors:
        try:
            els = page.locator(sel).all()
            if els:
                return els[-1].inner_text()
        except Exception:
            pass
    return ""

def dismiss_modals(page, browser_type: str):
    """Automatically dismisses overlay popups/modals in Gemini or ChatGPT."""
    dismiss_selectors = [
        'button:has-text("Got it")',
        'button:has-text("Dismiss")',
        'button:has-text("Close")',
        'button:has-text("Okay")',
        'button:has-text("OK")',
        'button:has-text("Stay on Free")',
        'button:has-text("Keep using")',
        'button[aria-label="Close"]',
        'div[role="dialog"] button:has-text("Got it")',
        'div[role="dialog"] button:has-text("Dismiss")',
        'div[role="dialog"] button:has-text("Close")'
    ]
    for sel in dismiss_selectors:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=300):
                print(f"Dismissing {browser_type.upper()} popup modal using selector: {sel}")
                btn.click()
                time.sleep(0.5)
        except Exception:
            pass

def navigate_with_recovery(page, context, url: str) -> tuple:
    """Robustly navigates page to a URL, recreating the tab if it crashes or closes."""
    for attempt in range(3):
        try:
            # Check if page is closed or crashed
            is_closed = True
            try:
                is_closed = page.is_closed()
            except Exception:
                pass
                
            if is_closed:
                print("⚠️ Page was closed or invalid. Recreating tab...")
                page = context.new_page()
            
            print(f"🧭 Navigating to {url} (attempt {attempt+1}/3)...")
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_load_state("domcontentloaded")
            time.sleep(3)
            return page, True
        except Exception as e:
            print(f"⚠️ Navigation failed or page crashed: {e}")
            try:
                page.close()
            except Exception:
                pass
            try:
                page = context.new_page()
            except Exception as creation_err:
                print(f"❌ Failed to create fresh page tab: {creation_err}")
            time.sleep(3)
    return page, False

def is_model_generating(page, browser_type: str) -> bool:
    """Returns True if the model is currently generating a response (Stop button is active)."""
    config = GEMINI_CONFIG if browser_type == "gemini" else CHATGPT_CONFIG
    send_btn = find_element(page, config["sends"])
    if send_btn:
        try:
            aria_label = send_btn.get_attribute("aria-label") or ""
            if any(x in aria_label.lower() for x in ["stop", "cancel", "responding", "answering"]):
                return True
        except Exception:
            pass

    # Secondary check: search for standard stop/cancel buttons on page
    stop_selectors = [
        'button[aria-label*="Stop"]',
        'button[aria-label*="Cancel"]',
        'button[data-testid="stop-button"]',
        'button[aria-label="Stop responding"]',
        'button[aria-label="Stop generating"]',
        'button[aria-label="Stop answering"]',
    ]
    for sel in stop_selectors:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=100):
                return True
        except Exception:
            pass
    return False

# ── Main Automation Loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Automate LLM Generation via Web App")
    parser.add_argument("--browser", type=str, required=True, choices=["gemini", "chatgpt"], help="Which web app to automate")
    parser.add_argument("--port", type=int, required=True, help="Chrome remote debugging port (e.g. 9222)")
    parser.add_argument("--shard", type=int, default=1, help="Shard index (1-indexed)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards")
    parser.add_argument("--refresh-interval", type=int, default=15, help="Start a new chat after this many queries to prevent context drift and browser lag (default: 15)")
    args = parser.parse_args()

    # Load data
    data = load_data()
    
    # Identify pending generation entries
    pending_qids = [
        qid for qid, item in data.items()
        if not item.get("rag_answer", "").strip() or item.get("rag_answer", "").startswith("[ERROR")
    ]

    # Partition by shard
    if args.num_shards > 1:
        sharded_qids = []
        for i, qid in enumerate(pending_qids):
            if (i % args.num_shards) == (args.shard - 1):
                sharded_qids.append(qid)
        pending_qids = sharded_qids

    todo = len(pending_qids)
    shard_out_path = RAG_OUTPUT.parent / f"rag_retrieved_answers_shard_{args.shard}.json"
    
    # Filter data to only contain this shard's entries to prevent concurrent write race conditions
    shard_data = {qid: data[qid] for qid in pending_qids}
    save_data(shard_data, shard_out_path)

    print(f"🚀 Shard {args.shard}/{args.num_shards} | Browser: {args.browser.upper()} | Port: {args.port}")
    print(f"⏳ Pending questions in this shard: {todo}")

    if todo == 0:
        print("✅ No pending questions in this shard!")
        return

    # Select configuration
    config = GEMINI_CONFIG if args.browser == "gemini" else CHATGPT_CONFIG

    with sync_playwright() as p:
        print(f"🔗 Connecting to Chrome over CDP on port {args.port}...")
        try:
            browser = p.chromium.connect_over_cdp(f"http://localhost:{args.port}")
            context = browser.contexts[0]
            page = context.pages[0]
        except Exception as e:
            print(f"❌ Failed to connect to Chrome on port {args.port}: {e}")
            print("   Make sure Chrome is running with remote debugging enabled.")
            sys.exit(1)

        print(f"🌐 Verifying page is on {args.browser}...")
        should_navigate = False
        try:
            if args.browser not in page.url:
                should_navigate = True
        except Exception:
            should_navigate = True

        if should_navigate:
            page, success = navigate_with_recovery(page, context, config["url"])

        for i, qid in enumerate(pending_qids, 1):
            # Verify connection to Chrome is alive, auto-reconnect if lost
            try:
                _ = page.url
            except Exception:
                print(f"⚠️ Shard {args.shard}: Connection to Chrome on port {args.port} was lost! Reconnecting...")
                try:
                    time.sleep(2)
                    browser = p.chromium.connect_over_cdp(f"http://localhost:{args.port}")
                    context = browser.contexts[0]
                    page = context.pages[0]
                    print("✓ Reconnected successfully!")
                except Exception as reconnect_err:
                    print(f"❌ Reconnection failed: {reconnect_err}. Retrying in 10 seconds...")
                    time.sleep(10)
                    continue

            # Periodic refresh to start a fresh chat thread
            if i > 1 and (i - 1) % args.refresh_interval == 0:
                print(f"🔄 Starting a fresh chat session to prevent context drift and browser lag...")
                page, success = navigate_with_recovery(page, context, config["url"])

            item = shard_data[qid]
            question = item["question"]
            
            # Format RAG prompt
            context_blocks = []
            for j, chunk in enumerate(item.get("retrieved_chunks", []), 1):
                context_blocks.append(f"[{j}] Source: {chunk.get('source_file','')}, Page: {chunk.get('page','')}\nText: {chunk.get('text','')}")
            context_str = "\n\n".join(context_blocks)
            
            prompt_text = (
                "You are a helpful research assistant. Answer the following question based ONLY on the provided context.\n"
                "CRITICAL RULES:\n"
                "1. Do NOT search the web. Use only the provided context.\n"
                "2. Output ONLY the direct answer. Do NOT include any conversational introduction, summary, or prefix "
                "(e.g., do NOT say 'Based on the context' or 'Here is your answer'). Start directly with the factual response.\n"
                "3. If the context does not contain enough information to answer, state that clearly.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

            print(f"\n[{i}/{todo}] Generating answer for: {qid}")
            print(f"  Q: {question[:100]}...")

            # Ensure model is not busy generating previous answer before starting
            busy_wait_start = time.time()
            while is_model_generating(page, args.browser):
                if time.time() - busy_wait_start > 90:
                    print("⚠️ Model has been busy for over 90 seconds. Forcing refresh...")
                    page, success = navigate_with_recovery(page, context, config["url"])
                    break
                print("⏳ Model is busy generating previous response. Waiting 3 seconds...")
                time.sleep(3)

            # ── Step 1: Input prompt ──────────────────────────────────────────
            dismiss_modals(page, args.browser)
            input_el = find_element(page, config["inputs"])
            if not input_el:
                print("⚠️ Input element not found. Attempting to recover and reload page...")
                page, success = navigate_with_recovery(page, context, config["url"])
                input_el = find_element(page, config["inputs"])
                if not input_el:
                    print("❌ Skip: Could not find input text area even after page recovery.")
                    item["rag_answer"] = f"[ERROR: Input box not found]"
                    save_data(shard_data, shard_out_path)
                    continue

            # Step 1: Robust prompt input
            prompt_entered = False
            try:
                # Set cursor and insert text instantly
                input_el.click()
                time.sleep(0.3)
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
                prompt_entered = True
            except Exception as e:
                print(f"⚠️ JS prompt insertion failed: {e}. Trying fallback fill...")
                try:
                    input_el.click()
                    input_el.fill(prompt_text)
                    time.sleep(2)
                    prompt_entered = True
                except Exception as fill_err:
                    print(f"❌ Fallback fill failed: {fill_err}")

            try:
                # Press Escape to dismiss any slash-command popups or auto-complete menus
                page.keyboard.press("Escape")
                time.sleep(0.5)
            except Exception:
                pass

            # ── Step 2: Click Send with safety checks ──────────────────────────
            send_btn = find_element(page, config["sends"])
            if not send_btn:
                print("❌ Send button not found. Skipping...")
                item["rag_answer"] = f"[ERROR: Send button not found]"
                save_data(shard_data, shard_out_path)
                continue

            # Verify the button is NOT the microphone dictation button
            aria_label = send_btn.get_attribute("aria-label") or ""
            if any(x in aria_label.lower() for x in ["microphone", "use mic", "dictate", "voice"]):
                print("⚠️ Button is Microphone icon (typing failed to register in UI). Retrying with force fill...")
                try:
                    input_el.click()
                    input_el.fill(prompt_text)
                    time.sleep(2)
                    send_btn = find_element(page, config["sends"])
                    aria_label = send_btn.get_attribute("aria-label") or ""
                except Exception as force_err:
                    print(f"❌ Force fill failed: {force_err}")

            # Verify the button is NOT the Stop responding button (previous question is still running)
            for wait_attempt in range(10):
                if "stop" in aria_label.lower() or "responding" in aria_label.lower():
                    print("⏳ Model is still generating previous answer. Waiting 3 seconds...")
                    time.sleep(3)
                    send_btn = find_element(page, config["sends"])
                    aria_label = send_btn.get_attribute("aria-label") or ""
                else:
                    break

            # Click send button
            print("Clicking Send button...")
            send_btn.click()
            time.sleep(5) # wait for generation to start

            # ── Step 3: Wait for Response Completion ──────────────────────────
            print("⏳ Monitoring response stream...")
            start_wait = time.time()
            last_text = ""
            stable_ticks = 0
            RESPONSE_TIMEOUT = 120 # 2 minutes timeout for long responses
            timed_out = False

            while True:
                # Check if actively streaming
                is_streaming = False
                try:
                    if page.locator(config["streaming"]).count() > 0:
                        is_streaming = True
                except Exception:
                    pass

                # Get response text
                current_text = get_last_response_text(page, config["responses"])
                
                # Check for stability
                if current_text and current_text == last_text and not is_streaming:
                    stable_ticks += 1
                else:
                    stable_ticks = 0
                    if current_text:
                        last_text = current_text

                # Stable for 6 seconds (2 ticks of 3s)
                if stable_ticks >= 2 and len(current_text.strip()) > 5:
                    break

                if time.time() - start_wait >= RESPONSE_TIMEOUT:
                    print("⏰ Response timed out. Saving current state...")
                    timed_out = True
                    break

                dismiss_modals(page, args.browser)
                time.sleep(3)

            # ── Step 4: Extract Response ──
            response_text = get_last_response_text(page, config["responses"])
            if not response_text.strip():
                response_text = "[ERROR: Empty response scraped from web UI]"

            print(f"  ✓ Answer generated successfully (Length: {len(response_text)})")
            item["rag_answer"] = response_text
            save_data(shard_data, shard_out_path)
            
            # Brief cooldown between queries
            time.sleep(2)

    print(f"\n🎉 Shard {args.shard} complete! Results saved to shard file.")

if __name__ == "__main__":
    main()
