"""Agentic research actions — Phase 5 Features B, C, D (ADR-008).

Each feature is a bounded tool pipeline: deterministic retrieval/provenance
steps interleaved with structured-output LLM calls. Everything the LLM asserts
about a paper is validated against the registry (fabricated page numbers are
dropped, unknown fields become "Not stated in retrieved evidence" — never
guessed), and every action taken is logged into state so the agent's behaviour
is observable in the UI.

Features:
  EVIDENCE_TABLE  — per-paper structured extraction (method, dataset, metric,
                    result, limitations) with evidence page references.
  CONTRADICTION   — per-paper claims on a focus topic, then classification of
                    conflicts (genuine disagreement vs. different datasets /
                    settings / definitions / not a contradiction).
  GAP_EXPLORATION — explicitly-stated limitations (evidenced), corpus-wide
                    patterns, and clearly-labelled inferred hypotheses. Never
                    claims novelty beyond the indexed corpus.
"""

import json
import time

import config

NOT_STATED = "Not stated in retrieved evidence"
MAX_CHARS_PER_DOC = 24000


# ---------------------------------------------------------------- LLM helper

def _llm_json(prompt, system_instruction=None):
    """One structured-output LLM call, parsed and fence-stripped."""
    from src.core.generator.client import generate_content_with_retry
    response = generate_content_with_retry(
        prompt, temperature=0.0,
        system_instruction=system_instruction,
        response_mime_type="application/json",
    )
    clean = response.strip()
    if "```" in clean:
        clean = clean.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


# ------------------------------------------------------------- provenance

def _doc_page_set(registry, doc):
    return {c.get("page") for c in registry
            if c.get("source_file") == doc and isinstance(c.get("page"), int)}


def _valid_pages(pages, allowed):
    """Drops any page the registry cannot confirm (fabrication guard)."""
    out = []
    for p in pages or []:
        try:
            p = int(p)
        except (TypeError, ValueError):
            continue
        if p in allowed and p not in out:
            out.append(p)
    return out


def _field(value):
    return value if value not in (None, "", [], {}) else NOT_STATED


def _doc_text(registry, doc):
    chunks = [c for c in registry if c.get("source_file") == doc]
    text, used = [], 0
    for c in chunks:
        t = c.get("text", "")
        if used + len(t) > MAX_CHARS_PER_DOC:
            break
        text.append(f"[Page {c.get('page')}] {t}")
        used += len(t)
    return "\n\n".join(text), _doc_page_set(registry, doc)


# ------------------------------------------------------------ the actions

def _action_evidence_table(focus, targets, registry, actions, llm_budget):
    """Feature B: structured per-paper evidence table with provenance."""
    rows, data = [], []
    for doc in targets:
        if llm_budget["left"] <= 0:
            actions.append({"tool": "evidence_table.extract", "args": {"doc": doc},
                            "skipped": "model-call budget exhausted"})
            continue
        text, pages = _doc_text(registry, doc)
        t0 = time.time()
        try:
            extracted = _llm_json(f"""Analyse the following text extracted from the paper '{doc}'.
Extract ONLY facts explicitly present in the text. If a field is not stated, use null — never guess.

RESEARCH FOCUS: {focus or 'the paper as a whole'}

PAPER TEXT:
{text}

Return JSON exactly like:
{{"research_question": str|null, "method": str|null, "dataset": str|null,
  "metric": str|null, "result": str|null, "limitations": str|null,
  "evidence_pages": [integer page numbers from the [Page N] markers above that support these fields]}}""")
            llm_budget["left"] -= 1
            actions.append({"tool": "evidence_table.extract", "args": {"doc": doc},
                            "seconds": round(time.time() - t0, 2),
                            "summary": "per-paper field extraction"})
        except Exception as e:
            llm_budget["left"] -= 1
            actions.append({"tool": "evidence_table.extract", "args": {"doc": doc},
                            "seconds": round(time.time() - t0, 2), "error": str(e)[:140]})
            continue

        row = {k: _field(extracted.get(k)) for k in
               ("research_question", "method", "dataset", "metric", "result", "limitations")}
        row["paper"] = doc
        row["evidence_pages"] = _valid_pages(extracted.get("evidence_pages"), pages)
        data.append(row)

    if data:
        rows.append("| Paper | Research question | Method | Dataset | Metric | Reported result | Limitations | Evidence |")
        rows.append("|---|---|---|---|---|---|---|---|")
        for r in data:
            rows.append(f"| {r['paper']} | {r['research_question']} | {r['method']} | {r['dataset']} "
                        f"| {r['metric']} | {r['result']} | {r['limitations']} "
                        f"| pp. {r['evidence_pages'] if r['evidence_pages'] else '—'} |")
        rows.append("\n*Empty cells mean the field was not stated in the retrieved text — values are never guessed.*")
    else:
        rows.append("No evidence could be extracted for the selected papers within the action budget.")
    return rows, data


def _action_contradiction(focus, targets, registry, actions, llm_budget):
    """Feature C: conflict analysis with disagreement-type classification."""
    claims = []
    for doc in targets:
        if llm_budget["left"] <= 0:
            actions.append({"tool": "contradiction.claim", "args": {"doc": doc},
                            "skipped": "model-call budget exhausted"})
            continue
        text, pages = _doc_text(registry, doc)
        t0 = time.time()
        try:
            out = _llm_json(f"""From this paper's text, state what it reports about the focus topic.
Use only the text; if the topic is not addressed, say so.

FOCUS TOPIC: {focus or "the paper's central findings"}

PAPER TEXT ({doc}):
{text}

Return JSON: {{"claim": str|null, "page": int|null, "conditions": str|null}}
"conditions" = the dataset, setup, or experimental setting the claim was made under.""")
            llm_budget["left"] -= 1
            if out.get("claim"):
                claims.append({"paper": doc, "claim": out["claim"],
                               "page": _valid_pages([out.get("page")], pages),
                               "conditions": _field(out.get("conditions"))})
            actions.append({"tool": "contradiction.claim", "args": {"doc": doc},
                            "seconds": round(time.time() - t0, 2),
                            "summary": "claim extracted" if out.get("claim") else "topic not addressed"})
        except Exception as e:
            llm_budget["left"] -= 1
            actions.append({"tool": "contradiction.claim", "args": {"doc": doc},
                            "seconds": round(time.time() - t0, 2), "error": str(e)[:140]})

    if len(claims) < 2:
        actions.append({"tool": "contradiction.compare",
                        "skipped": "fewer than two papers address the focus topic"})
        return ([f"Only {len(claims)} of the selected papers address '{focus or 'this topic'}' — "
                 "a contradiction analysis needs at least two claims to compare."], claims)

    t0 = time.time()
    claims_block = "\n".join(
        f"- {c['paper']} (p. {c['page'] or '?'}): {c['claim']} [under: {c['conditions']}]"
        for c in claims)
    try:
        verdict = _llm_json(f"""Compare these per-paper claims about "{focus or 'the topic'}" and identify conflicts.

{claims_block}

Classify each conflict as exactly one of:
"genuine_disagreement" | "different_datasets" | "different_evaluation_settings" | "different_definitions" | "not_a_contradiction"

Rules: different numbers under different conditions are usually NOT genuine contradictions. Name the papers involved.

Return JSON: {{"contradictions": [{{"papers": [str], "classification": str, "explanation": str}}]}}""")
        llm_budget["left"] -= 1
        actions.append({"tool": "contradiction.compare",
                        "args": {"claims": len(claims)},
                        "seconds": round(time.time() - t0, 2),
                        "summary": f"{len(verdict.get('contradictions', []))} conflict(s) classified"})
    except Exception as e:
        llm_budget["left"] -= 1
        actions.append({"tool": "contradiction.compare",
                        "seconds": round(time.time() - t0, 2), "error": str(e)[:140]})
        return ("The conflict comparison call failed; per-paper claims were still extracted "
                "(see actions).", claims)

    valid_classes = {"genuine_disagreement", "different_datasets", "different_evaluation_settings",
                     "different_definitions", "not_a_contradiction"}
    rows = ["| Papers | Classification | Explanation |", "|---|---|---|"]
    for c in verdict.get("contradictions", []):
        cls = c.get("classification") if c.get("classification") in valid_classes else "unclassified"
        rows.append(f"| {', '.join(c.get('papers', []))} | `{cls}` | {_field(c.get('explanation'))} |")
    rows.append("\n*Different numbers under different datasets/settings are reported as such — not as genuine contradictions.*")
    return rows, {"claims": claims, "verdicts": verdict.get("contradictions", [])}


def _action_gap_exploration(focus, targets, registry, actions, llm_budget):
    """Feature D: evidenced limitations, corpus patterns, labelled hypotheses."""
    limitation_lists = []
    for doc in targets:
        if llm_budget["left"] <= 0:
            actions.append({"tool": "gap.limitations", "args": {"doc": doc},
                            "skipped": "model-call budget exhausted"})
            continue
        text, pages = _doc_text(registry, doc)
        t0 = time.time()
        try:
            out = _llm_json(f"""List ONLY the limitations and future-work items that this paper explicitly states about itself.
Do not invent or infer limitations. Cite the page each item came from.

PAPER TEXT ({doc}):
{text}

Return JSON: {{"limitations": [{{"text": str, "page": int}}]}}""")
            llm_budget["left"] -= 1
            items = [{"text": li.get("text"), "page": p}
                     for li in out.get("limitations", [])
                     if (p := _valid_pages([li.get("page")], pages))]
            limitation_lists.append({"paper": doc, "limitations": items})
            actions.append({"tool": "gap.limitations", "args": {"doc": doc},
                            "seconds": round(time.time() - t0, 2),
                            "summary": f"{len(items)} explicit limitation(s)"})
        except Exception as e:
            llm_budget["left"] -= 1
            actions.append({"tool": "gap.limitations", "args": {"doc": doc},
                            "seconds": round(time.time() - t0, 2), "error": str(e)[:140]})

    all_items = [(d["paper"], li) for d in limitation_lists for li in d["limitations"]]
    rows = [f"### 1. Explicitly stated in the papers (evidenced)\n"]
    if all_items:
        for paper, li in all_items:
            rows.append(f"- **{paper}** (p. {li['page'][0] if li['page'] else '?'}): {li['text']}")
    else:
        rows.append("- None of the selected papers explicitly state limitations in the retrieved text.")

    patterns = hypotheses = []
    if llm_budget["left"] > 0 and all_items:
        t0 = time.time()
        lit_block = "\n".join(f"- {p}: {li['text']}" for p, li in all_items)
        try:
            out = _llm_json(f"""These are self-stated limitations from {len(limitation_lists)} indexed papers:
{lit_block}

Identify: (a) recurring patterns across papers, (b) plausible open research directions.
Research directions must be framed as hypotheses, not established facts.

Return JSON: {{"corpus_patterns": [str], "hypotheses": [str]}}""")
            llm_budget["left"] -= 1
            patterns = out.get("corpus_patterns", []) or []
            hypotheses = out.get("hypotheses", []) or []
            actions.append({"tool": "gap.synthesize",
                            "seconds": round(time.time() - t0, 2),
                            "summary": f"{len(patterns)} pattern(s), {len(hypotheses)} hypothesis(ies)"})
        except Exception as e:
            llm_budget["left"] -= 1
            actions.append({"tool": "gap.synthesize",
                            "seconds": round(time.time() - t0, 2), "error": str(e)[:140]})

    rows.append("\n### 2. Patterns observed across the indexed corpus\n")
    rows += [f"- {p}" for p in patterns] or ["- Not enough stated limitations to identify patterns."]
    rows.append("\n### 3. Hypotheses (inferred by the system — not established facts)\n")
    rows += [f"- {h}" for h in hypotheses] or ["- None."]
    rows.append(f"\n*Scope: derived from {len(targets)} indexed paper(s) only. "
                "This is not a claim about all published research.*")
    return rows, {"limitation_lists": limitation_lists,
                  "patterns": patterns, "hypotheses": hypotheses}


ACTION_RUNNERS = {
    "EVIDENCE_TABLE": _action_evidence_table,
    "CONTRADICTION": _action_contradiction,
    "GAP_EXPLORATION": _action_gap_exploration,
}


def run_research_actions(intent, focus, targets, registry):
    """Executes the planned research actions for a research intent.

    Returns {"actions": [...], "markdown": str, "data": ...}. The LLM-call
    budget left for actions is the agent budget minus what routing already
    spent, so research runs can never loop unbounded.
    """
    runner = ACTION_RUNNERS[intent]
    llm_budget = {"left": max(0, config.AGENT_MODEL_CALL_BUDGET - 1)}  # routing spent 1
    ordered_targets = targets[:5]  # bound per-run scope
    actions = []
    rows, data = runner(focus, ordered_targets, registry, actions, llm_budget)
    title = {
        "EVIDENCE_TABLE": "Research Evidence Table",
        "CONTRADICTION": "Contradiction Analysis",
        "GAP_EXPLORATION": "Research Gap Exploration",
    }[intent]
    header = f"## {title}"
    if focus:
        header += f" — focus: “{focus}”"
    header += f"\n\n*Papers analysed: {', '.join(ordered_targets)}*"
    return {"actions": actions, "markdown": header + "\n\n" + "\n".join(rows), "data": data}
