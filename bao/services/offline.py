"""Bao AI - Offline Mode Policy.

Previously every call site independently re-derived "should this go to
Gemini or fall back?" — `if generator.is_available()`, `if
self.llm_service.is_online`, `if edge_mode: ... else: ...` — each with
slightly different logic and none of them accounting for an explicit
user-facing "force offline" toggle at the same time as a missing API key.
One function, one truth.
"""

from __future__ import annotations

from bao.ai.client import GeminiClient


def should_use_offline(client: GeminiClient, force_offline: bool = False) -> bool:
    """True if the pipeline should skip Gemini entirely and rely on the
    offline knowledge base only — either because the user explicitly asked
    for offline mode, or because Gemini genuinely isn't reachable.
    """
    if force_offline:
        return True
    return not client.is_available()
