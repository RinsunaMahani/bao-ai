"""Bao AI - Conversation Memory.

This didn't exist before. Both Streamlit entrypoints kept `st.session_state
["messages"]` purely to *render* chat bubbles — every call to Gemini was
still constructed from a single, memory-less prompt, so the assistant had
no way to answer "what did I just ask you?" or resolve a follow-up like
"and in Afrikaans?". This module makes conversation history part of the
pipeline (see ai/orchestrator.py), not just part of the UI.

Deliberately a plain rolling window, not a summarizing/embedding memory —
that's a reasonable v2 (see docs/ARCHITECTURE.md), but a fixed window is
the right amount of complexity for a single-session assistant today.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    language: str | None = None


@dataclass
class ConversationMemory:
    max_turns: int = 6
    # How much of one turn to keep. The window bounded how MANY turns were
    # carried and nothing bounded how LARGE each was, which is the half
    # that grows: the security guardrail caps user input at 500 characters,
    # but a generated reply is uncapped, and a measured one ran to 2,059.
    # Six such exchanges prepended 12,656 characters — roughly 3,200 tokens
    # — to every subsequent request. On a free tier with a small daily
    # allowance, that is paid again on each turn, and a longer prompt also
    # delays the first word, which is the part an audience watches.
    #
    # 400 characters because of what this memory is FOR: resolving a
    # follow-up like "and in Afrikaans?" or "what did I just ask you"
    # needs the gist of the exchange, not the full text of an essay. The
    # cut is marked so the model can see it is reading an excerpt rather
    # than a complete previous answer.
    max_chars_per_turn: int = 400
    turns: list[Turn] = field(default_factory=list)

    def add(self, role: str, content: str, language: str | None = None) -> None:
        if self.max_chars_per_turn and len(content) > self.max_chars_per_turn:
            content = content[: self.max_chars_per_turn].rstrip() + " […]"
        self.turns.append(Turn(role=role, content=content, language=language))
        # Keep the last max_turns*2 entries (user+assistant pairs).
        overflow = len(self.turns) - (self.max_turns * 2)
        if overflow > 0:
            self.turns = self.turns[overflow:]

    def as_context(self) -> str:
        """Renders recent history as plain text suitable for prepending to
        a Gemini prompt. Returns "" if there's no history yet, so callers
        can unconditionally prepend it without an extra branch.
        """
        if not self.turns:
            return ""
        lines = [f"{t.role.capitalize()}: {t.content}" for t in self.turns]
        return "Recent conversation:\n" + "\n".join(lines)

    def clear(self) -> None:
        self.turns = []

    def __len__(self) -> int:
        return len(self.turns)
