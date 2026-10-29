"""Pure ranking & qualification logic for visual (figure/table) retrieval.

Design rules (fixes for the "irrelevant images always show up" defect):

1. Never compare raw CLIP similarity against keyword scores — they are
   incompatible spaces. Each ranker is filtered on its own calibrated scale
   and the survivors are merged by Reciprocal Rank Fusion.
2. Dense (CLIP) filtering is *relative*: keep hits within a small margin of
   the best hit for this query, subject to an absolute noise floor. A fixed
   absolute cutoff fails because CLIP's similarity scale shifts per query.
3. Sparse (caption) matching is IDF-weighted: tokens that appear in many
   captions ("model", "results", bare digits inside table text) carry almost
   no evidence; rare tokens ("cmrr", "heap") carry a lot. A single ubiquitous
   token can no longer qualify an image.
4. Images attach to an answer only on evidence:
     - explicit reference ("figure 3", "table 4") matched in a caption, OR
     - agreement: found independently by BOTH dense and sparse, OR
     - strong caption evidence (IDF sum >= IMAGE_SPARSE_STRONG_IDF).
   Otherwise the result is empty — the image count is a cap, never a quota.
"""

import math
import re

import config
from src.core.sparse_retrieval import STOP_WORDS, tokenize  # lightweight: no faiss/torch

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_ROMAN = {1: "i", 2: "ii", 3: "iii", 4: "iv", 5: "v",
          6: "vi", 7: "vii", 8: "viii", 9: "ix", 10: "x"}

# Explicit "figure 3" / "fig. 4b" / "table ii" references in the query.
_REF_PATTERN_RE = re.compile(
    r"(fig(?:ure)?|table)\s*(\d+|[ivx]+|[a-z])(?:\s*\(?([a-z])\)?\b)?"
)


def chunk_key(chunk):
    return (chunk.get("source_file", ""), chunk.get("page"), chunk.get("image_path", ""))


def build_caption_stats(registry):
    """Document frequency of each token across all image captions."""
    df = {}
    for item in registry:
        for token in set(tokenize(item.get("caption", ""))):
            df[token] = df.get(token, 0) + 1
    return df


def _idf(df, token, n_docs):
    return math.log(n_docs / (1 + df.get(token, 0)))


def _roman(num_str):
    try:
        return _ROMAN.get(int(num_str), "")
    except ValueError:
        return ""


# Structural/visual nouns are noise in *caption* matching (every caption is a
# figure/table); bare digits match table OCR everywhere. Both only matter via
# the explicit-reference patterns above, never as topical evidence.
_VISUAL_STOPWORDS = frozenset({
    "figure", "figures", "fig", "figs", "table", "tables", "diagram", "diagrams",
    "chart", "charts", "graph", "graphs", "plot", "plots", "image", "images",
    "picture", "pictures", "photo", "photos", "sketch", "sketches", "equation",
    "equations", "appendix", "section",
})


def _query_evidence_tokens(query_lower):
    return [
        t for t in _TOKEN_RE.findall(query_lower)
        if t not in STOP_WORDS and t not in _VISUAL_STOPWORDS and not t.isdigit()
    ]


def sparse_visual_matches(query, registry, caption_df):
    """Ranks registry items whose captions reference the query.

    Token matching uses each caption's token SET (not substring containment) so
    that query tokens which never occur as real caption tokens — e.g. a bare
    "1" substring-matching inside "P1" or "12" — can never score.

    Returns a ranked list of {"score", "chunk", "explicit", "idf_sum"}.
    """
    n_docs = max(1, len(registry))
    query_lower = query.lower()
    ref_patterns = _REF_PATTERN_RE.findall(query_lower)
    q_tokens = _query_evidence_tokens(query_lower)

    results = []
    for item in registry:
        caption = item.get("caption", "").lower()
        caption_tokens = set(tokenize(item.get("caption", "")))

        # 1. Explicit figure/table references — the highest-confidence signal.
        explicit = False
        for ref_type, ref_num, ref_letter in ref_patterns:
            if ref_num in STOP_WORDS and len(ref_num) > 1:
                continue
            roman = _roman(ref_num)
            num_part = r"(?:" + re.escape(ref_num) + r"\b"
            if roman:
                num_part += r"|" + re.escape(roman) + r"\b"
            num_part += r")"
            prefix = r"fig(?:ure)?[\s\.]*" if ref_type.startswith("fig") else r"table[\s\.]*"
            if ref_letter:
                pattern = prefix + num_part + r"(?:[\s\S]{0,100}?[\s\.\(\[-]+" + re.escape(ref_letter) + r"\b)?"
            else:
                pattern = prefix + num_part
            if re.search(pattern, caption):
                explicit = True
                break
        if explicit:
            results.append({"score": 0.45, "chunk": item, "explicit": True, "idf_sum": None})
            continue

        # 2. IDF-weighted caption match — ubiquitous tokens carry ~no evidence.
        matched = [(_idf(caption_df, t, n_docs), t) for t in q_tokens if t in caption_tokens]
        if not matched:
            continue
        matched.sort(reverse=True)
        idf_sum = sum(w for w, _ in matched[:3])
        if idf_sum < config.IMAGE_SPARSE_MIN_IDF:
            continue
        score = 0.28 + 0.12 * min(idf_sum / 4.0, 1.0)
        results.append({"score": score, "chunk": item, "explicit": False, "idf_sum": idf_sum})

    results.sort(key=lambda r: -r["score"])
    return results


def qualify_visual(dense_ranked, sparse_ranked, k):
    """Applies the evidence rules and returns at most k images (possibly 0)."""
    dense_keys = {chunk_key(r["chunk"]) for r in dense_ranked}
    sparse_meta = {chunk_key(r["chunk"]): r for r in sparse_ranked}

    from src.core.fusion import rrf_fuse
    fused = rrf_fuse([dense_ranked, sparse_ranked])

    qualified = []
    for res in fused:
        key = chunk_key(res["chunk"])
        sparse_hit = sparse_meta.get(key)
        if sparse_hit is None:
            continue  # dense-only: no caption evidence, do not attach
        if sparse_hit.get("explicit"):
            qualified.append({"score": res["score"], "chunk": res["chunk"],
                              "explicit": True, "idf_sum": None})
        elif (key in dense_keys) or ((sparse_hit.get("idf_sum") or 0.0)
                                     >= config.IMAGE_SPARSE_STRONG_IDF):
            qualified.append({"score": res["score"], "chunk": res["chunk"],
                              "explicit": False, "idf_sum": sparse_hit.get("idf_sum")})
        if len(qualified) >= k:
            break
    return qualified
