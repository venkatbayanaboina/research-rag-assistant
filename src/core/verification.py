"""Claim-level citation verification for generated research answers.

Layer 1 (deterministic, always on): resolves every ``[Source N]`` marker in
the answer against the evidence list actually returned by retrieval, checks
that cited evidence exists, and measures what fraction of claim-like
sentences carry a citation.

Layer 2 (opt-in via config.VERIFY_CLAIMS_WITH_LLM): one batched LLM call that
judges whether each cited chunk text actually supports the claim sentence it
is attached to. LLM verdicts are advisory inputs to the qualification
decision, never ground truth.

Status vocabulary shared with the LangGraph final response:
  verified  — all markers resolve, coverage is acceptable
  qualified — answer is usable but carries listed integrity issues
  failed    — evidence is absent or a majority of citations are dangling
  skipped   — no text evidence to check (special intents, greetings)
"""

import re

import config

_MARKER_RE = re.compile(r"\[Source\s*(\d+)(?:[^\]]*)\]", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Sentence shapes that do not represent factual claims.
_NON_CLAIM_PREFIXES = ("#", "-", "*", "|", ">", "1.", "2.", "3.", "4.", "5.")
_GREETING_TOKENS = {"hi", "hello", "hey", "thanks", "thank", "bye", "welcome"}


def _split_sentences(answer):
    sentences = []
    for raw in _SENTENCE_SPLIT_RE.split(answer):
        s = raw.strip()
        if not s:
            continue
        lowered = s.lower()
        if lowered.startswith(_NON_CLAIM_PREFIXES):
            continue
        if any(lowered.startswith(g) for g in _GREETING_TOKENS):
            continue
        sentences.append(s)
    return sentences


def _is_claim_like(sentence):
    words = sentence.split()
    if len(words) < 8:
        return False
    if sentence.startswith(("Based on", "According to")):
        return True
    # A sentence with a digit or a bracketed citation is carrying factual
    # content; long sentences without either are still claims worth citing.
    return True


def _layer1_checks(answer, evidence):
    issues = []
    markers = _MARKER_RE.findall(answer or "")
    marker_indices = [int(m) for m in markers]

    n_evidence = len(evidence)
    dangling = sorted({m for m in marker_indices if m < 1 or m > n_evidence})
    if dangling:
        issues.append(
            f"Citation markers reference evidence that was not retrieved: "
            f"{', '.join(f'[Source {m}]' for m in dangling)}"
        )

    sentences = [s for s in _split_sentences(answer) if _is_claim_like(s)]
    cited_sentences = [s for s in sentences if _MARKER_RE.search(s)]
    coverage = (len(cited_sentences) / len(sentences)) if sentences else 1.0

    if sentences and not marker_indices:
        issues.append("Answer contains no inline [Source N] citations.")
    elif sentences and coverage < 0.5:
        issues.append(
            f"Only {len(cited_sentences)}/{len(sentences)} claim-like sentences carry citations."
        )

    report = {
        "markers_found": len(marker_indices),
        "dangling_markers": dangling,
        "claim_sentences": len(sentences),
        "cited_sentences": len(cited_sentences),
        "citation_coverage": round(coverage, 4),
        "issues": issues,
    }
    return report


def _layer2_llm_support(answer, evidence):
    """Batched LLM check: does the cited evidence support each claim?"""
    cited_claims = []
    for sentence in _split_sentences(answer):
        for match in _MARKER_RE.finditer(sentence):
            idx = int(match.group(1))
            if 1 <= idx <= len(evidence):
                cited_claims.append({
                    "claim": _MARKER_RE.sub("", sentence).strip(),
                    "source_index": idx,
                    "source_text": evidence[idx - 1]["chunk"].get("text", "")[:1500],
                })
    if not cited_claims:
        return []

    claims_block = "\n".join(
        f"{i+1}. CLAIM: {c['claim']}\n   EVIDENCE [Source {c['source_index']}]: {c['source_text']}"
        for i, c in enumerate(cited_claims[:12])  # bound the prompt size
    )
    prompt = f"""You are verifying whether research claims are supported by their cited evidence.
For each numbered claim below, decide if the evidence text supports the claim (allow reasonable paraphrase).

{claims_block}

Return ONLY valid JSON in this exact schema:
{{"claims": [{{"index": 1, "supported": true, "reason": "short reason"}}, ...]}}
"""
    try:
        import json
        from src.core.generator.client import generate_content_with_retry
        response = generate_content_with_retry(
            prompt, temperature=0.0, response_mime_type="application/json"
        )
        clean = response.strip()
        if "```" in clean:
            clean = clean.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        unsupported = []
        for item in data.get("claims", []):
            if item.get("supported") is False:
                idx = int(item.get("index", 0)) - 1
                if 0 <= idx < len(cited_claims):
                    unsupported.append({
                        "claim": cited_claims[idx]["claim"][:200],
                        "source_index": cited_claims[idx]["source_index"],
                        "reason": str(item.get("reason", ""))[:300],
                    })
        return unsupported
    except Exception as e:
        # LLM verification is advisory; a failure degrades to Layer 1 only.
        print(f"[VERIFICATION] Layer 2 LLM support check failed (advisory only): {e}")
        return []


def verify_citations(answer, evidence, use_llm=None):
    """Verifies an answer's citations against the retrieved evidence.

    Args:
        answer: generated answer text.
        evidence: retrieval results, i.e. [{"score": float, "chunk": dict}, ...];
            [Source N] maps to evidence[N-1].
        use_llm: override for config.VERIFY_CLAIMS_WITH_LLM.

    Returns:
        dict: {status, citation_coverage, markers_found, dangling_markers,
               claim_sentences, cited_sentences, issues, unsupported_claims}
    """
    evidence = evidence or []
    report = _layer1_checks(answer, evidence)

    if use_llm is None:
        use_llm = config.VERIFY_CLAIMS_WITH_LLM
    unsupported = []
    if use_llm and answer and evidence:
        unsupported = _layer2_llm_support(answer, evidence)
    report["unsupported_claims"] = unsupported
    if unsupported:
        report["issues"].append(
            f"{len(unsupported)} cited claim(s) were not supported by their evidence "
            "(LLM support check)."
        )

    n_dangling = len(report["dangling_markers"])
    if not evidence:
        report["status"] = "failed" if report["markers_found"] else "skipped"
    elif n_dangling > max(1, report["markers_found"] // 2):
        report["status"] = "failed"
    elif report["issues"]:
        report["status"] = "qualified"
    else:
        report["status"] = "verified"
    return report


def format_verification_note(report):
    """One-line user-facing note appended to qualified/failed answers."""
    status = report.get("status")
    if status == "verified":
        coverage = report.get("citation_coverage", 0.0)
        return f"\n\n> ✅ *Citation check passed — {coverage*100:.0f}% of claim sentences cite retrieved sources.*"
    if status == "qualified":
        issues = "; ".join(report.get("issues", [])) or "minor integrity issues"
        return f"\n\n> ⚠️ *Citation check flagged this answer: {issues} Treat details without citations with caution.*"
    if status == "failed":
        return ("\n\n> ❌ *Citation check failed: this answer references sources that were not "
                "retrieved. It should not be relied upon.*")
    return ""
