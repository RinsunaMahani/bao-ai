"""
Bao AI - Security Guardrails.

Screens user input before it reaches language detection, retrieval, or the
Gemini API: rejects empty and oversized input, and blocks a small set of
known prompt-injection / jailbreak phrasings.

Scope, stated honestly: this is regex screening, not "AI security." It
catches the obvious, well-known phrasings and nothing subtler. Its real
job is keeping malformed or hostile input out of the pipeline and out of
an outbound API call, not defeating a determined adversary.

Two entry points on purpose:
  - `validate_input()` returns a (bool, reason) tuple, for callers that
    want to branch on the result (the UI, tests).
  - `validate_or_raise()` raises `SecurityViolationError`, for callers
    inside the pipeline that should abort rather than continue — this is
    what `ai/orchestrator.py` uses, so a blocked turn short-circuits
    before detection, retrieval, or generation ever run.
"""
from __future__ import annotations

import re

from bao.core.exceptions import SecurityViolationError

_BLOCK_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"system\s*prompt\s*(override|leak)",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"bypass\s+(all\s+)?(rules|restrictions|guardrails)",
)


class SecurityGuardrails:
    """Stateless input screening. Constructed once per session with the
    configured `max_query_length` from config.toml.
    """

    def __init__(self, max_length: int = 500):
        self.max_length = max_length
        self._block_patterns = [re.compile(p, re.IGNORECASE) for p in _BLOCK_PATTERNS]

    def validate_input(self, text: str) -> tuple[bool, str | None]:
        """Returns (True, None) if the input is safe to process, or
        (False, reason) if it should be rejected. The reason is shown to
        the user, so it explains what to do differently.
        """
        if not text or not text.strip():
            return False, "Input cannot be empty."

        if len(text) > self.max_length:
            return False, (
                f"Input exceeds the maximum length of {self.max_length} characters."
            )

        for pattern in self._block_patterns:
            if pattern.search(text):
                return False, (
                    "Input was flagged by the security policy "
                    "(prompt-injection pattern detected)."
                )

        return True, None

    def validate_or_raise(self, text: str) -> None:
        """Raises SecurityViolationError instead of returning a tuple, so
        pipeline code can abort with `try/except` rather than threading a
        boolean through every stage.
        """
        is_safe, reason = self.validate_input(text)
        if not is_safe:
            raise SecurityViolationError(reason)
