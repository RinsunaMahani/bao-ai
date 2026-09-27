"""Bao AI service layer.

Deliberately left with no re-exports, like every other package in `bao`
(see bao/__init__.py). Import from the module that owns the name:

    from bao.services.language_detector import get_language_detector
    from bao.services.speech import synthesize_speech

This file used to re-export from every submodule, and that had a cost
nobody could see from a call site. speech.py imports torch, transformers
and the Coqui TTS stack when it loads, so ANY import from this package —
even just the language detector, which uses none of them — loaded the
whole speech stack first: measured at 562 MB and 19 seconds. That landed
on evaluate.py, the benchmark, the preflight's detection check and every
test that touches detection. The web app still loads speech, because it
uses it; nothing else pays for it any more.

Its __all__ also still listed SignLanguageRecognizer, from the removed
sign-language prototype, so `from bao.services import *` raised
AttributeError.
"""
