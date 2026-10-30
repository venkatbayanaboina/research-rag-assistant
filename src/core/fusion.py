"""Pure ranking-fusion utilities shared by text and visual retrieval.

Reciprocal Rank Fusion (Cormack et al., 2009): for each result ranked at
position r (1-based) in a ranking, contribute 1 / (RRF_K + r). Contributions
are summed across rankings; ties are broken deterministically by first-seen
order so fusion output is stable for tests.
"""

import config


def rrf_fuse(rankings, k_constant=None, top_k=None):
    """Fuses multiple ranked lists of result dicts into one RRF-scored list.

    Args:
        rankings: iterable of ranked lists; each item must be a dict with a
            "chunk" key. Chunk identity is ("source_file", "page", text-or-path)
            so identical chunks from different rankings merge.
        k_constant: RRF dampening constant (defaults to config.RRF_K_CONSTANT).
        top_k: optional cap on the size of the fused output.

    Returns:
        List of {"score": rrf_score, "chunk": chunk} sorted by descending
        score. The original "chunk" dicts are returned unmodified.
    """
    if k_constant is None:
        k_constant = config.RRF_K_CONSTANT

    scores = {}
    chunks = {}
    order = []
    for ranking in rankings:
        for rank_0, result in enumerate(ranking):
            chunk = result["chunk"]
            key = (chunk.get("source_file", ""),
                   chunk.get("page"),
                   chunk.get("text", chunk.get("image_path", "")))
            if key not in scores:
                scores[key] = 0.0
                chunks[key] = chunk
                order.append(key)
            scores[key] += 1.0 / (k_constant + rank_0 + 1)

    fused = [{"score": scores[key], "chunk": chunks[key]} for key in order]
    fused.sort(key=lambda item: item["score"], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]
    return fused
