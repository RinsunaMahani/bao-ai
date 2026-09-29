"""Bao AI - Gemini API Client.

The *only* module in the codebase that imports google-genai. Previously
this logic existed twice (generative.py's GeminiGenerator and
services/llm_service.py's LLMService) with different constructor
signatures, different error handling, and both hardcoding their own
"gemini-3.5-flash" default instead of reading core.config. Everything else
in the app — orchestrator, translation, evaluation — talks to Gemini
through this one class.
"""

from __future__ import annotations

import random
import re
import threading
import time

from bao.core.config import Settings
from bao.core.exceptions import GenerationError, GenerationUnavailableError
from bao.core.logging import get_logger

logger = get_logger(__name__)

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False
    logger.warning("google-genai library not found. Online generation disabled.")


# Statuses worth trying again. All of these say "the request was fine, the
# service could not serve it right now", which is exactly the case where a
# retry is the correct response rather than an apology:
#
#   429  rate limited
#   500  internal error
#   502  bad gateway
#   503  UNAVAILABLE — "this model is currently experiencing high demand"
#   504  gateway timeout
#
# 503 is the one observed in practice on gemini-3.6-flash and is what
# motivated this: a single spike turned a perfectly good question into
# "I ran into a problem generating a response", and the turn was lost. The
# same request succeeded on the very next attempt.
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Network-level failures have no status code but are equally worth one more
# try. Matched on type name rather than by importing httpx, which is a
# transitive dependency of the SDK and not one this module should require.
_TRANSIENT_ERROR_NAMES = (
    "timeout", "timedout", "connecterror", "connectionerror", "readerror",
    "remoteprotocolerror", "connectionreset", "incompleteread",
)


# Gemini reports how long to wait in two different shapes depending on
# where in the payload it appears: a RetryInfo field ("'retryDelay': '13s'")
# and prose in the message ("Please retry in 13.902226599s").
_RETRY_DELAY_RE = re.compile(
    r"retry(?:Delay['\"]?\s*:\s*['\"]?|\s+in\s+)(\d+(?:\.\d+)?)s", re.IGNORECASE
)

# Above this, retrying immediately is worse than not retrying.
#
# The free tier allows 5 requests per minute, and a 429 comes back asking
# for ~13 seconds. Three quick attempts do not wait that out — they just
# spend three more of the five requests the quota allows, so the retry
# makes the very condition it is reacting to worse, and then reports
# "busy" anyway. When the provider names a wait longer than this, the
# honest move is to stop and pass the number on.
_MAX_SERVER_RETRY_WAIT_SECONDS = 2.0


def _server_retry_delay(error: Exception) -> float | None:
    """How long the provider asked us to wait, in seconds, if it said."""
    match = _RETRY_DELAY_RE.search(str(error))
    return float(match.group(1)) if match else None


def _is_transient(error: Exception) -> bool:
    """True when retrying the identical request might succeed.

    Reads `code` off google-genai's APIError rather than matching on the
    message text: the message is prose that can be reworded upstream at any
    time, while the status code is the contract.
    """
    code = getattr(error, "code", None)
    if isinstance(code, int):
        return code in _TRANSIENT_STATUS_CODES

    name = type(error).__name__.lower()
    return any(marker in name for marker in _TRANSIENT_ERROR_NAMES)


class GeminiClient:
    """Thin, testable wrapper around the Gemini API.

    `is_available()` is the single check every caller should use to decide
    whether to route to Gemini or fall back to offline behavior — it's
    false whenever the SDK isn't installed, no API key is configured, or
    client construction failed, so callers don't need to duplicate that
    three-way check themselves.
    """

    # Backoff between attempts, in seconds, plus jitter. Short because a
    # person is watching a spinner: three attempts cost under a second of
    # added latency in the worst case, which is a far better trade than
    # losing the turn. This is not a batch job and must not behave like one.
    _RETRY_BACKOFF_SECONDS = (0.25, 0.75)

    def __init__(self, settings: Settings, api_key: str | None = None):
        self.settings = settings
        self.model_name = settings.gemini_model
        self.fallback_models = [
            m for m in settings.gemini_fallback_models if m != self.model_name
        ]
        self.max_attempts = max(1, settings.generation_max_attempts)
        self.thinking_level = settings.thinking_level
        self.max_output_tokens = settings.max_output_tokens
        self.timeout_seconds = settings.generation_timeout_seconds
        self._api_key = api_key
        self.client = None
        # Per thread, because one client serves every browser session and
        # Streamlit runs each session's turn on its own thread. An
        # attribute shared by all of them would report another visitor's
        # fallback on this visitor's badge.
        self._local = threading.local()
        self._setup_client()

    def last_fallback_model(self) -> str | None:
        """The backup model that answered this thread's most recent call,
        or None when the main model answered it.
        """
        return getattr(self._local, "fallback_model", None)

    def _sleep_before_retry(self, attempt: int) -> None:
        """Waits before the next attempt, with jitter so several clients
        recovering from the same outage do not retry in lockstep.
        """
        index = min(attempt, len(self._RETRY_BACKOFF_SECONDS) - 1)
        delay = self._RETRY_BACKOFF_SECONDS[index]
        time.sleep(delay + random.uniform(0, delay / 2))  # noqa: S311 - jitter, not a secret

    def _setup_client(self) -> None:
        if not _HAS_GENAI:
            return

        import os
        api_key = self._api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found. Running in offline-only mode.")
            return

        try:
            self.client = genai.Client(
                api_key=api_key,
                # Milliseconds. Unset, the SDK waits forever.
                http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
            )
            logger.info(f"Gemini client initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    def is_available(self) -> bool:
        return self.client is not None

    def _config(self, system_instruction: str | None, temperature: float):
        config_kwargs = {"temperature": temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if self.max_output_tokens:
            config_kwargs["max_output_tokens"] = self.max_output_tokens
        # No tools are passed, so automatic function calling has nothing it
        # could call. Switched off explicitly rather than left idle: the
        # model is given no agency this app does not use, and the SDK stops
        # logging "AFC is enabled with max remote calls: 10" on every call.
        if hasattr(types, "AutomaticFunctionCallingConfig"):
            config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )

        # Gemini 3.x models reason before emitting anything, so this
        # decides how long someone watches a spinner — streaming cannot
        # shorten it, because there is nothing to stream until thinking
        # ends. See Settings.thinking_level for the measurements.
        #
        # Guarded rather than assumed: older SDKs have no ThinkingConfig,
        # and a model that does not support the field rejects the request
        # outright. Neither should cost the app its generation path.
        level = (self.thinking_level or "").upper()
        if level and level != "DEFAULT" and hasattr(types, "ThinkingConfig"):
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=level
                )
            except Exception as e:  # unknown level, or a stricter SDK
                logger.warning(
                    f"Ignoring thinking_level={level!r} ({e}); using the model default."
                )
        return types.GenerateContentConfig(**config_kwargs)

    def generate_stream(
        self, prompt: str, system_instruction: str | None = None, temperature: float = 0.3
    ):
        """Yields the reply in pieces as the API produces them.

        Same request as `generate()`; the only difference is that the
        caller can show text while it's still being written. That matters
        here because the long answers in this app (a full explanation
        translated into an African language) are exactly the ones where
        waiting for the complete response feels broken, and it's free —
        the total time is unchanged, but time-to-first-word drops from
        the whole generation to a fraction of it.

        When the main model stays busy through its retries, the fallback
        models are tried in turn, but only while nothing has been shown.
        Once a piece has reached the caller, a second model would add a
        second answer after the first, so a failure from then on is final.
        """
        if not self.is_available():
            raise GenerationError("Gemini client is not available (missing SDK or API key).")

        self._local.fallback_model = None
        models = [self.model_name, *self.fallback_models]
        busy: GenerationUnavailableError | None = None
        for index, model in enumerate(models):
            state = {"produced": False}
            try:
                yield from self._stream_from(model, prompt, system_instruction, temperature, state)
                return
            except GenerationUnavailableError as e:
                if state["produced"]:
                    self._local.fallback_model = None
                    raise
                busy = e
                if index + 1 < len(models):
                    logger.warning(f"{model} is unavailable; trying {models[index + 1]}.")
                    self._local.fallback_model = models[index + 1]
            except GenerationError as e:
                if index == 0 or state["produced"]:
                    self._local.fallback_model = None
                    raise
                # A backup failing for its own reasons (a retired model
                # name, say) must not replace the real story, which is that
                # the main model was busy.
                logger.error(f"Fallback model {model} failed: {e}")
                break
        self._local.fallback_model = None
        raise busy

    def _stream_from(self, model: str, prompt: str, system_instruction, temperature, state: dict):
        """One model's streamed attempts, retrying transient failures.

        `state["produced"]` is set once any chunk has been yielded, so the
        caller can tell a failure before the answer began (safe to hand to
        another model) from one after (not safe).
        """
        for attempt in range(self.max_attempts):
            # Tracked per attempt, because it decides whether retrying is
            # even legal: once a chunk has reached the caller it has been
            # rendered on screen, and starting over would append a second
            # copy of the answer rather than replace the first. A stream
            # that has begun can only be failed, never retried.
            produced_any = False
            try:
                stream = self.client.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                    config=self._config(system_instruction, temperature),
                )
                for chunk in stream:
                    if chunk.text:
                        produced_any = True
                        state["produced"] = True
                        yield chunk.text
                if not produced_any:
                    raise GenerationError("Gemini returned an empty response.")
                return
            except GenerationError:
                raise
            except Exception as e:
                wait = _server_retry_delay(e)
                too_long = wait is not None and wait > _MAX_SERVER_RETRY_WAIT_SECONDS
                retriable = (
                    _is_transient(e)
                    and not produced_any
                    and not too_long
                    and attempt < self.max_attempts - 1
                )
                if not retriable:
                    if _is_transient(e):
                        logger.warning(f"Gemini stream unavailable: {e}")
                        raise GenerationUnavailableError(str(e), retry_after=wait) from e
                    logger.error(f"Gemini streaming API error: {e}")
                    raise GenerationError(str(e)) from e
                logger.warning(
                    f"Gemini stream failed with a transient error "
                    f"(attempt {attempt + 1}/{self.max_attempts}); retrying: {e}"
                )
                self._sleep_before_retry(attempt)

    def generate(self, prompt: str, system_instruction: str | None = None, temperature: float = 0.3) -> str:
        """Generates a response from Gemini.

        Raises GenerationError on failure rather than returning an apology
        string baked into the return value — callers (the orchestrator)
        decide what the user-facing fallback text should be, which keeps
        that copy in one place (ai/prompts.py) instead of scattered through
        every service that might call Gemini.

        Falls back to the next model in `fallback_models` when one stays
        busy or out of quota; see generate_stream.
        """
        if not self.is_available():
            raise GenerationError("Gemini client is not available (missing SDK or API key).")

        self._local.fallback_model = None
        models = [self.model_name, *self.fallback_models]
        busy: GenerationUnavailableError | None = None
        for index, model in enumerate(models):
            try:
                text = self._generate_from(model, prompt, system_instruction, temperature)
                self._local.fallback_model = model if index else None
                return text
            except GenerationUnavailableError as e:
                busy = e
                if index + 1 < len(models):
                    logger.warning(f"{model} is unavailable; trying {models[index + 1]}.")
            except GenerationError as e:
                if index == 0:
                    raise
                logger.error(f"Fallback model {model} failed: {e}")
                break
        raise busy

    def _generate_from(self, model: str, prompt: str, system_instruction, temperature) -> str:
        """One model's attempts, retrying transient failures."""
        for attempt in range(self.max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=self._config(system_instruction, temperature),
                )
                if not response.text:
                    raise GenerationError("Gemini returned an empty response.")
                return response.text
            except GenerationError:
                raise
            except Exception as e:
                wait = _server_retry_delay(e)
                too_long = wait is not None and wait > _MAX_SERVER_RETRY_WAIT_SECONDS
                if not _is_transient(e) or too_long or attempt == self.max_attempts - 1:
                    if _is_transient(e):
                        if too_long:
                            logger.warning(
                                f"Gemini asked for a {wait:.0f}s wait; not retrying, "
                                "since quick retries would only spend more quota."
                            )
                        else:
                            logger.error(f"Gemini API error: {e}")
                        raise GenerationUnavailableError(str(e), retry_after=wait) from e
                    logger.error(f"Gemini API error: {e}")
                    raise GenerationError(str(e)) from e
                logger.warning(
                    f"Gemini call failed with a transient error "
                    f"(attempt {attempt + 1}/{self.max_attempts}); retrying: {e}"
                )
                self._sleep_before_retry(attempt)

        # Unreachable: the final attempt either returns or raises above.
        raise GenerationError("Generation failed after exhausting all attempts.")
