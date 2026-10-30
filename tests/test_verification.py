import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.core.verification as verification
from src.core.verification import verify_citations, format_verification_note


def _evidence(n=3):
    return [{"score": 1.0, "chunk": {"source_file": f"p{i}.pdf", "page": i,
                                     "text": f"evidence text {i}"}} for i in range(1, n + 1)]


def test_verified_answer_with_resolved_citations():
    answer = ("Self-attention computes pairwise similarities [Source 1]. "
              "The model uses 6 layers [Source 2]. BLEU reached 28.4 on WMT14 [Source 1].")
    report = verify_citations(answer, _evidence())
    assert report["status"] == "verified"
    assert report["dangling_markers"] == []
    assert report["citation_coverage"] == 1.0
    assert report["issues"] == []


def test_dangling_marker_fails_verification():
    answer = ("Grounded claim with a citation [Source 1]. "
              "This one points to evidence that does not exist [Source 9]. "
              "Another grounded sentence citing real evidence [Source 2].")
    report = verify_citations(answer, _evidence())
    assert report["status"] in ("qualified", "failed")
    assert 9 in report["dangling_markers"]
    assert any("not retrieved" in i for i in report["issues"])


def test_majority_dangling_markers_fail():
    answer = "Bad [Source 8]. Worse [Source 9]. Ok [Source 1]."
    report = verify_citations(answer, _evidence())
    assert report["status"] == "failed"


def test_zero_citations_is_qualified_with_issue():
    answer = ("A long enough factual sentence without any citation markers at all. "
              "Another long sentence that also carries no citation marker whatsoever.")
    report = verify_citations(answer, _evidence())
    assert report["status"] == "qualified"
    assert any("no inline" in i.lower() for i in report["issues"])


def test_partial_coverage_is_qualified():
    cited = "Grounded factual sentence carrying a proper citation [Source 1]."
    uncited = ("Long factual sentence with numbers 42 and no citation marker whatsoever. "
               "Another long factual sentence that also carries no citation marker at all.")
    report = verify_citations(f"{cited} {uncited}", _evidence())
    assert report["status"] == "qualified"
    assert 0 < report["citation_coverage"] < 0.5


def test_empty_evidence_skips_or_fails():
    assert verify_citations("hello there", [])["status"] == "skipped"
    assert verify_citations("Claim [Source 1].", [])["status"] == "failed"


def test_greetings_are_not_claims():
    report = verify_citations("Hi there! Thanks for the question.", _evidence())
    assert report["status"] == "verified"  # no claims -> coverage defaults to 1.0


def test_layer2_llm_verification_flags_unsupported(monkeypatch):
    def fake_llm(prompt, **kwargs):
        return '{"claims": [{"index": 1, "supported": false, "reason": "evidence silent on number"}]}'
    monkeypatch.setattr("src.core.generator.client.generate_content_with_retry", fake_llm)
    answer = "The accuracy was 99.9 percent on every dataset [Source 1]."
    report = verify_citations(answer, _evidence(), use_llm=True)
    assert report["status"] == "qualified"
    assert len(report["unsupported_claims"]) == 1
    assert "not supported" in report["issues"][-1]


def test_layer2_failure_degrades_to_layer1(monkeypatch):
    def boom(prompt, **kwargs):
        raise RuntimeError("LLM down")
    monkeypatch.setattr("src.core.generator.client.generate_content_with_retry", boom)
    answer = "Grounded factual sentence with citation [Source 1]."
    report = verify_citations(answer, _evidence(), use_llm=True)
    assert report["status"] == "verified"  # advisory layer failed silently
    assert report["unsupported_claims"] == []


def test_format_verification_note_shapes():
    assert "passed" in format_verification_note({"status": "verified", "citation_coverage": 0.9})
    assert "flagged" in format_verification_note({"status": "qualified", "issues": ["x"]})
    assert "failed" in format_verification_note({"status": "failed"})
    assert format_verification_note({"status": "skipped"}) == ""


def test_real_module_smoke():
    # Guards against accidental regex regressions on the marker format.
    m = verification._MARKER_RE.findall("see [Source 2 - p1.pdf (Page 3)] and [source 3]")
    assert m == ["2", "3"]
