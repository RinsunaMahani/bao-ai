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

from bao.core.config import Settings
from bao.core.exceptions import GenerationError
from bao.core.logging import get_logger

logger = get_logger(__name__)

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False
    logger.warning("google-genai library not found. Online generation disabled.")


class GeminiClient:
    """Thin, testable wrapper around the Gemini API.

    `is_available()` is the single check every caller should use to decide
    whether to route to Gemini or fall back to offline behavior — it's
    false whenever the SDK isn't installed, no API key is configured, or
    client construction failed, so callers don't need to duplicate that
    three-way check themselves.
    """

    def __init__(self, settings: Settings, api_key: str | None = None):
        self.settings = settings
        self.model_name = settings.gemini_model
        self._api_key = api_key
        self.client = None
        self._setup_client()

    def _setup_client(self) -> None:
        if not _HAS_GENAI:
            return

        import os
        api_key = self._api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found. Running in offline-only mode.")
            return

        try:
            self.client = genai.Client(api_key=api_key)
            logger.info(f"Gemini client initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    def is_available(self) -> bool:
        return self.client is not None

    def _config(self, system_instruction: str | None, temperature: float):
        config_kwargs = {"temperature": temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
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
        """
        if not self.is_available():
            raise GenerationError("Gemini client is not available (missing SDK or API key).")

        try:
            stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=self._config(system_instruction, temperature),
            )
            produced_any = False
            for chunk in stream:
                if chunk.text:
                    produced_any = True
                    yield chunk.text
            if not produced_any:
                raise GenerationError("Gemini returned an empty response.")
        except GenerationError:
            raise
        except Exception as e:
            logger.error(f"Gemini streaming API error: {e}")
            raise GenerationError(str(e)) from e

    def generate(self, prompt: str, system_instruction: str | None = None, temperature: float = 0.3) -> str:
        """Generates a response from Gemini.

        Raises GenerationError on failure rather than returning an apology
        string baked into the return value — callers (the orchestrator)
        decide what the user-facing fallback text should be, which keeps
        that copy in one place (ai/prompts.py) instead of scattered through
        every service that might call Gemini.
        """
        if not self.is_available():
            raise GenerationError("Gemini client is not available (missing SDK or API key).")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self._config(system_instruction, temperature),
            )
            if not response.text:
                raise GenerationError("Gemini returned an empty response.")
            return response.text
        except GenerationError:
            raise
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise GenerationError(str(e)) from e
