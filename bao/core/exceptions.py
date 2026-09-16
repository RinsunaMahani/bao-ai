"""Bao AI - Custom Exception Hierarchy.

The original codebase caught `Exception` broadly at nearly every call site
and turned failures into log lines or generic apology strings. That's fine
as a last-resort safety net, but it made it impossible for a caller (e.g.
the orchestrator, or a test) to distinguish "the security filter blocked
this" from "the knowledge base file is missing" from "Gemini's network call
failed." These types make that distinction explicit; modules should raise
the most specific one that applies, and only the outermost layer (the UI)
should catch the broad `BaoError` base class as a final fallback.
"""


class BaoError(Exception):
    """Base class for all Bao AI application errors."""


class SecurityViolationError(BaoError):
    """Raised when user input fails the security guardrail checks."""


class ConfigurationError(BaoError):
    """Raised when required configuration is missing or invalid."""


class RetrievalError(BaoError):
    """Raised when the knowledge base fails to load or query."""


class GenerationError(BaoError):
    """Raised when the Gemini generation call fails in a way that should
    not be silently swallowed (as opposed to a normal "offline, no key"
    condition, which is not an error - see ai/client.py `is_available()`).
    """


class SpeechError(BaoError):
    """Raised when text-to-speech or speech-to-text processing fails."""


class UnsupportedLanguageError(BaoError):
    """Raised when a requested language isn't in the supported set."""
