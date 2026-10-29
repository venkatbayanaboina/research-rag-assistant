"""Lazy re-exports: importing src.core.generator no longer pulls the entire
generator chain (orchestrator -> vector_store -> faiss/torch) at import time.
Smoke checks and holdout tooling import lightweight submodules (router,
research_actions) without the heavy transitive dependencies. Callers of
`from src.core.generator import X` still work via module __getattr__ (PEP 562).
"""

__all__ = [
    "get_gemini_client",
    "generate_answer",
    "generate_summary",
    "generate_section_summaries",
    "generate_comparison",
    "execute_rag_pipeline",
    "route_user_intent",
    "get_gemini_client",
]


def __getattr__(name):
    if name in __all__:
        from src.core.generator import orchestrator
        return getattr(orchestrator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
