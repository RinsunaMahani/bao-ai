"""Bao AI - Translation Service.

Previously "translate this fact" was one inline f-string built directly
inside app.py's response-generation branch, indistinguishable from the
"answer this open-ended question" branch except by reading the surrounding
if/else. Pulling it out means the pipeline diagram (User -> ... ->
Translation (if needed) -> ...) corresponds to an actual, named function
call in orchestrator.py, not an implicit side effect of prompt phrasing.
"""

from __future__ import annotations

from collections import OrderedDict

from bao.ai.client import GeminiClient
from bao.ai.prompts import translate_fact_prompt
from bao.core.exceptions import GenerationError
from bao.core.logging import get_logger

logger = get_logger(__name__)

# Knowledge-base facts are a small, fixed set (32 rows) and translation of
# a given fact into a given language is deterministic, so the same network
# round trip was being paid every single time. A demo that asks the same
# greeting three times paid for three identical Gemini calls. Process-local
# and unbounded on purpose: the key space is (rows x 11 languages), which
# is a few hundred short strings at absolute worst.
_TRANSLATION_CACHE: OrderedDict[tuple[str, str], str] = OrderedDict()

# Bounded because "the key space is small" holds for knowledge-base facts
# and stops holding the moment anything else is translated. An LRU cap
# costs nothing and removes the assumption entirely; 512 short strings is
# far more than the 32-row KB across 11 languages ever needs.
_TRANSLATION_CACHE_MAX = 512


def clear_translation_cache() -> None:
    _TRANSLATION_CACHE.clear()


def translate_fact(fact: str, target_language: str, client: GeminiClient) -> str:
    """Translates a verified, already-correct fact into the target
    language. Only used for facts pulled from the offline knowledge base —
    open-ended Gemini answers are already generated directly in the target
    language (see ai/prompts.system_instruction), so they never need this
    extra round trip.
    """
    if target_language.lower() == "english":
        return fact

    cache_key = (fact, target_language)
    if cache_key in _TRANSLATION_CACHE:
        _TRANSLATION_CACHE.move_to_end(cache_key)
        logger.info(f"Translation cache hit for {target_language}; skipping API call.")
        return _TRANSLATION_CACHE[cache_key]

    if not client.is_available():
        # No online translation available — better to show the verified
        # English fact than to show nothing.
        logger.info("Translation requested but Gemini unavailable; returning source fact untranslated.")
        return fact

    try:
        translated = client.generate(translate_fact_prompt(fact, target_language))
        _TRANSLATION_CACHE[cache_key] = translated
        while len(_TRANSLATION_CACHE) > _TRANSLATION_CACHE_MAX:
            _TRANSLATION_CACHE.popitem(last=False)
        return translated
    except GenerationError as e:
        # Deliberately NOT cached: a failure is usually transient (network,
        # rate limit), and caching it would make one blip permanent for the
        # rest of the session.
        logger.warning(f"Translation failed, returning source fact untranslated: {e}")
        return fact
