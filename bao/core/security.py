"""
Bao AI - Security Guardrails.

Screens user input before it reaches language detection, retrieval, or the
Gemini API: rejects empty and oversized input, and blocks a small set of
known prompt-injection / jailbreak phrasings. Also holds the output-side
counterpart, `safe_markdown`, for text the page renders.

Scope, stated honestly: this is regex screening, not "AI security." It
catches the well-known English phrasings, including the usual disguises
(invisible characters, full-width letters, look-alike Cyrillic letters,
punctuation between the words), and nothing subtler. It does not read the
other ten official languages: "ziba imiyalelo edlule" passes. Its real job
is keeping malformed or hostile input out of the pipeline and out of an
outbound API call, not defeating a determined adversary. The defences that
do not depend on spotting a phrase are elsewhere: the untrusted-document
fence in ai/prompts.py, and safe_markdown below.

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
import unicodedata

from bao.core.exceptions import SecurityViolationError

# Characters that display as nothing, or reorder what is displayed, but are
# still read by a language model. They are how an instruction is hidden in
# text a person reviews and sees nothing wrong with:
#
#   - Unicode TAG characters (U+E0000-E007F) spell out ASCII invisibly,
#     and models read them ("ASCII smuggling").
#   - Bidirectional overrides and isolates make displayed text differ from
#     stored text ("Trojan Source", CVE-2021-42574).
#   - Zero-width spaces and word joiners split a phrase so a filter no
#     longer sees it, while a model still does.
#
# None has a use in a chat message or an uploaded study document. The
# zero-width JOINER and NON-JOINER are kept: emoji sequences such as the
# family emoji are built from them, and the screening below ignores them
# anyway, so keeping them cannot hide a phrase from it.
_INVISIBLE = re.compile(
    "["
    "­"              # soft hyphen
    "᠎"              # Mongolian vowel separator
    "​"              # zero-width space
    "‎‏"        # left-to-right / right-to-left marks
    "‪-‮"       # bidi embeddings and overrides
    "⁠-⁤"       # word joiner, invisible operators
    "⁦-⁩"       # bidi isolates
    "﻿"              # zero-width no-break space (BOM)
    "\U000e0000-\U000e007f"  # tag characters
    "]"
)
# C0 controls other than tab, newline and carriage return, and DEL.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_invisible(text: str) -> str:
    """Removes invisible, reordering and control characters, and normalises
    to NFC. Applied to what the user types AND to uploaded documents, since
    both end up in a prompt.

    NFC, not NFKC: NFKC would also rewrite visible text ("x²" becomes "x2"),
    which changes what the user actually wrote. NFKC is used only for the
    screening copy below, which is never shown or sent anywhere.
    """
    return unicodedata.normalize("NFC", _CONTROL.sub("", _INVISIBLE.sub("", text)))


# Latin look-alikes in other scripts, for the screening copy only: "ignоre"
# with a Cyrillic "о" is a different string to a regex and the same word to
# a model. Only letters that are near-identical in common fonts; a
# Cyrillic sentence mapped through this becomes Latin gibberish, which
# matches none of the English patterns, so real Cyrillic text is unaffected.
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l", "ɡ": "g",
    "ο": "o", "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ρ": "p",
    "τ": "t", "υ": "u", "χ": "x",
})


def _screening_forms(text: str) -> tuple[str, str]:
    """The copies the block patterns are matched against: full-width and
    other compatibility forms folded (NFKC), look-alikes mapped to Latin,
    case folded, and every run of non-letters collapsed to one space - so
    "IGNORE_all_PREVIOUS...instructions" and "ｉｇｎｏｒｅ all previous
    instructions" both read "ignore all previous instructions".

    Two copies, because an invisible character can be doing either of two
    jobs. Inside a word ("ig​nore") it has to be deleted to rejoin the
    word; between words ("ignore⁠all⁠previous") it has to become a
    space, or deleting it glues the phrase into one long word. A pattern
    matching either copy counts.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_CONFUSABLES).casefold()
    forms = []
    for replacement in ("", " "):
        visible = "".join(
            replacement if unicodedata.category(c) == "Cf" else c for c in folded
        )
        forms.append(re.sub(r"[\W_]+", " ", visible).strip())
    return forms[0], forms[1]


# Matched against _screening_forms, so words are lower case and separated
# by exactly one space. Each is specific enough not to fire on an ordinary
# question - "how do I enable developer mode on my phone", "ignore the
# noise", "you can do anything now that exams are over" all pass - and
# tests/test_security.py holds a list of such sentences to keep it so.
_BLOCK_PATTERNS: tuple[str, ...] = (
    r"\b(ignore|disregard|forget|override)( all| any)?( of)?( the| your| my)? "
    r"(previous|prior|above|earlier|preceding|system) (instructions|prompts?|rules|directions)\b",
    r"\bsystem prompt (override|leak)\b",
    r"\b(reveal|show|print|repeat|output|display|tell me|give me)( me)?( the| your)?"
    r"( full| entire| original| hidden| initial)? "
    r"(system prompt|system instructions|hidden instructions|initial instructions)\b",
    r"\byou are now in (developer|dev|god|dan) mode\b",
    r"\b(enable|enter|activate|switch to) dan mode\b",
    r"\bbypass( all| any| your)? (rules|restrictions|guardrails)\b",
    r"\bpretend (that )?you (are|have) no (rules|restrictions|guidelines|filters)\b",
    r"\bact as (an? )?(unrestricted|unfiltered|uncensored|jailbroken) (ai|assistant|model|chatbot)\b",
)


class SecurityGuardrails:
    """Stateless input screening. Constructed once per session with the
    configured `max_query_length` from config.toml.
    """

    def __init__(self, max_length: int = 500):
        self.max_length = max_length
        self._block_patterns = [re.compile(p) for p in _BLOCK_PATTERNS]

    @staticmethod
    def sanitize(text: str) -> str:
        """The input as the pipeline should see it: invisible characters
        removed and surrounding whitespace trimmed. See strip_invisible.
        """
        return strip_invisible(text or "").strip()

    def validate_input(self, text: str) -> tuple[bool, str | None]:
        """Returns (True, None) if the input is safe to process, or
        (False, reason) if it should be rejected. The reason is shown to
        the user, so it explains what to do differently.
        """
        if not text or not strip_invisible(text).strip():
            return False, "Input cannot be empty."

        if len(text) > self.max_length:
            return False, (
                f"Input exceeds the maximum length of {self.max_length} characters."
            )

        forms = _screening_forms(text)
        for pattern in self._block_patterns:
            if any(pattern.search(form) for form in forms):
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


# --- output ---------------------------------------------------------------

# ![alt](url) and ![alt](url "title"), with an optional <...> around the url.
#
# Every repetition is bounded. Unbounded, `[^\]\n]*` let each "![" scan to
# the end of the text when no "]" followed, so a reply of many "![" took
# time proportional to the square of its length (CodeQL
# py/polynomial-redos) - and the text is the model's output, which a
# document can steer. Bounded, the work is linear. Nothing becomes unsafe
# past the bounds: an image this misses is still escaped by the replace()
# in safe_markdown.
_MARKDOWN_IMAGE = re.compile(
    r"!\[([^\]\n]{0,300})\]\(\s{0,20}<?([^()\s<>]{1,2048})>?"
    r"(?:\s{1,20}(?:\"[^\"\n]{0,300}\"|'[^'\n]{0,300}'|\([^()\n]{0,300}\)))?\s{0,20}\)"
)


def safe_markdown(text: str) -> str:
    """Model output and uploaded-document text, made safe to render with
    st.markdown.

    Markdown images are the problem. Streamlit renders `![](url)` as an
    <img>, and the viewer's browser fetches that URL the moment the reply
    appears, with no click. A document carrying hidden instructions can ask
    the model to "include this image" with the conversation encoded into
    the address, and the browser then delivers the conversation to that
    server. This is the standard way data leaks out of an LLM chat
    interface, and it needs no bug in the model, only obedience.

    Images become ordinary links, marked with an icon, so nothing loads
    until someone chooses to click. Any "![" left over (reference-style
    images) is escaped so it cannot form an image either. Raw HTML needs no
    handling here: st.markdown escapes it unless unsafe_allow_html is set,
    and this app sets that only for its own fixed thinking indicator.
    """
    if not text:
        return text
    linked = _MARKDOWN_IMAGE.sub(
        lambda m: f"[🖼 {m.group(1).strip() or 'image'}]({m.group(2)})", text
    )
    return linked.replace("![", "!\\[")


_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~:$])")


def escape_markdown(text: str) -> str:
    """Shows `text` literally inside Markdown: for strings the user chose,
    such as an uploaded file's name, which would otherwise be interpreted.
    Unescaped, a file named "![x](https://example.org/t.png)" would be
    drawn as a picture in the sidebar rather than shown as a name.
    """
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text or "")
