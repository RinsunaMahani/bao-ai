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


class GenerationUnavailableError(GenerationError):
    """Raised when generation failed for a reason that is the provider's
    and is expected to pass: the model is overloaded (503), the request was
    rate-limited (429), or the network dropped — and retrying did not clear
    it within the configured attempts.

    A subclass rather than a flag because callers treat it differently. An
    ordinary GenerationError means something about the request was wrong
    and repeating it will not help; this one means the request was fine and
    the user should simply try again, which is a different sentence to put
    on screen.

    `retry_after` carries the provider's own estimate in seconds when it
    gave one, so the interface can say "try again in 13 seconds" instead of
    "try again in a moment" and be right.
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class NetworkUnavailableError(GenerationError):
    """Raised when Google could not be reached at all: no connection, or
    the name could not be resolved (Wi-Fi off reads as getaddrinfo failing).

    Not "busy", and not worth retrying or handing to a backup model, which
    would need the same connection. The caller answers offline instead.
    """


class ModelRejectedError(GenerationError):
    """Raised when one model refused the request outright: it no longer
    exists (404, a retired model), the key may not use it (403), or it does
    not accept the request's settings (400; gemini-3.7-flash and 3.8-flash
    reject the "minimal" thinking level, for one).

    Retrying the same model cannot help, but another model may well
    accept the identical request, so the client tries its backups.
    """


class SpeechError(BaoError):
    """Raised when text-to-speech or speech-to-text processing fails."""


class UnsupportedLanguageError(BaoError):
    """Raised when a requested language isn't in the supported set."""
