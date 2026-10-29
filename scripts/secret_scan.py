"""Scans the full git history for credential patterns.

Patterns: OpenRouter keys (sk-or-...), Google API keys (AIza...), generic
bearer/assignment leaks for the env vars this project uses, and private-key
blocks. Placeholders in .env.example are whitelisted. Exit code 1 on finding.
"""

import re
import subprocess
import sys

PATTERNS = {
    "openrouter_key": re.compile(r"sk-or-v1-[0-9a-zA-Z]{16,}"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    "assigned_secret": re.compile(r"(?:GEMINI_API_KEY|OPENROUTER_API_KEY)\s*[=:]\s*[\"']?(?!your_|\"\"|''|\$\{)[A-Za-z0-9_\-]{20,}"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

WHITELIST = ("your_gemini_api_key_here", "your_openrouter_api_key_here")


def main():
    log = subprocess.run(["git", "log", "-p", "--all", "--", ".", ":(exclude)requirements.lock.txt"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = log.stdout
    findings = []
    for name, pat in PATTERNS.items():
        for m in pat.finditer(text):
            frag = text[max(0, m.start() - 40): m.end() + 20]
            if any(w in frag for w in WHITELIST):
                continue
            # Find the commit line above the match for context
            commit = "unknown"
            c = text.rfind("\ncommit ", 0, m.start())
            if c != -1:
                commit = text[c + 8: c + 48]
            findings.append((name, commit, m.group(0)[:12] + "..."))

    print(f"[secret-scan] scanned {len(text):,} chars of git history (-p --all)")
    if findings:
        for name, commit, frag in findings:
            print(f"  FINDING {name} in {commit}: {frag}")
        print(f"[secret-scan] {len(findings)} FINDING(S)")
        sys.exit(1)
    print("[secret-scan] clean — no credential patterns in git history")


if __name__ == "__main__":
    main()
