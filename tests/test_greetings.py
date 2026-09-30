"""Greetings must be written in-language, with no English mixed in.

This is a speech-quality requirement, not a stylistic one. A bilingual row
like "Avuxeni! How can Bao assist you today?" is handed to MMS-TTS with a
single voice code, so the Xitsonga model applies Xitsonga letter-to-sound
rules to the English half. The greeting sounds right and the rest comes
out as noise. There is no voice code that reads both halves correctly, so
the fix has to be in the data.
"""

import re

import pytest

from bao.core.config import LABELS, Settings
from bao.knowledge.loader import load_knowledge_csv
from bao.knowledge.retriever import KnowledgeRetriever

# Function words common enough that finding one means an English clause is
# present. Deliberately not a full dictionary — "nga", "ka", "ni" and other
# short strings are real words in these languages, so a naive wordlist would
# fire constantly.
ENGLISH_MARKERS = {
    "how", "can", "help", "what", "you", "your", "today", "the", "with",
    "assist", "need", "i", "am", "is", "are", "do", "does", "may",
}


@pytest.fixture
def orchestrator_online(monkeypatch):
    """An orchestrator wired to a stub Gemini that flags any translation
    call, so a test can assert translation did NOT happen.
    """
    from bao.ai.client import GeminiClient
    from bao.ai.orchestrator import Orchestrator
    from bao.core.security import SecurityGuardrails
    from bao.services.language_detector import HeuristicLanguageDetector
    from bao.services.translation import clear_translation_cache

    clear_translation_cache()
    settings = Settings()
    client = GeminiClient(settings)
    monkeypatch.setattr(client, "is_available", lambda: True)
    monkeypatch.setattr(client, "generate", lambda prompt, **kw: "TRANSLATED")
    return Orchestrator(
        SecurityGuardrails(),
        HeuristicLanguageDetector(),
        KnowledgeRetriever(
            data_path=settings.knowledge_base_path,
            threshold=settings.similarity_threshold,
            min_coverage=settings.min_query_coverage,
        ),
        client,
    )


@pytest.fixture(scope="module")
def knowledge():
    return load_knowledge_csv(Settings().knowledge_base_path)


@pytest.fixture(scope="module")
def greetings(knowledge):
    rows = knowledge[knowledge["Category"].str.strip().str.lower() == "greeting"]
    assert not rows.empty
    return rows


# Afrikaans is exempt. It is a Germanic language that genuinely shares
# vocabulary with English — "help", "is", "kan", "hand" are all real
# Afrikaans words — so a marker wordlist produces false positives there
# rather than catching anything. ("Hoe kan ek jou help?" tripped exactly
# this check while being correct Afrikaans.) Afrikaans also has its own
# MMS voice, so the mixed-phonetics problem doesn't arise for it.
_MARKER_CHECK_EXEMPT = {"english", "afrikaans"}


def test_non_english_greetings_contain_no_english(greetings):
    for _, row in greetings.iterrows():
        if row["Language"].strip().lower() in _MARKER_CHECK_EXEMPT:
            continue
        words = set(re.findall(r"[a-z']+", row["Answer"].lower()))
        intruders = words & ENGLISH_MARKERS
        assert not intruders, (
            f"{row['Language']} greeting {row['Answer']!r} contains English "
            f"{sorted(intruders)} — TTS will read it with the wrong phonetics"
        )


def test_every_official_language_has_a_greeting(greetings):
    covered = {lang.strip().lower() for lang in greetings["Language"]}
    missing = [label for label in LABELS if label.lower() not in covered]
    assert not missing, f"no greeting for: {missing}"


def test_each_greeting_is_retrievable_by_its_trigger_word(greetings):
    settings = Settings()
    retriever = KnowledgeRetriever(
        data_path=settings.knowledge_base_path,
        threshold=settings.similarity_threshold,
        min_coverage=settings.min_query_coverage,
    )
    for _, row in greetings.iterrows():
        fact = retriever.lookup(row["Question"])
        assert fact is not None, f"{row['Question']!r} no longer retrieves"
        assert fact.language == row["Language"]


def test_sotho_tswana_rows_share_the_dumela_trigger(greetings):
    """Sepedi, Sesotho and Setswana all greet with "Dumela". All three rows
    answer to it, deliberately — retrieval is not supposed to resolve this,
    because the information needed is not in the query.

    An earlier version gave each row a distinct trigger to avoid the
    collision. That hid the ambiguity rather than handling it: typing
    "dumela" reported "Language: Sesotho" while answering with the Setswana
    row, badge and answer disagreeing on screen.
    """
    rows = {
        row["Language"]: row["Question"]
        for _, row in greetings.iterrows()
        if row["Language"] in {"Sepedi", "Sesotho", "Setswana"}
    }
    assert len(rows) == 3
    for language, trigger in rows.items():
        assert "dumela" in trigger, f"{language} must answer to the shared greeting"


def test_the_classifier_resolves_the_shared_greeting(orchestrator_online):
    """The disambiguation the LSTM exists for, end to end: the detected
    language selects among rows that retrieval scores equally.
    """
    for language in ("Sesotho", "Setswana", "Sepedi"):
        fact = orchestrator_online.knowledge_retriever.lookup(
            "dumela", prefer_language=language
        )
        assert fact is not None
        assert fact.language == language, (
            f"asked for {language}, got the {fact.language} row — badge and "
            "answer would disagree on screen"
        )


def test_preference_is_ignored_when_no_row_matches_it(orchestrator_online):
    """A language with no row for that query must still get the best
    available answer, not nothing.
    """
    fact = orchestrator_online.knowledge_retriever.lookup("avuxeni", prefer_language="Afrikaans")
    assert fact is not None
    assert fact.language == "Xitsonga"


def test_language_column_defaults_when_absent(tmp_path):
    """Older knowledge bases and test fixtures have no Language column.
    They must still load rather than raising.
    """
    path = tmp_path / "kb.csv"
    path.write_text("Question,Answer\nhello,Hi there.\n")
    df = load_knowledge_csv(str(path))
    assert df.iloc[0]["Language"] == "English"


def test_curated_non_english_answers_are_never_machine_translated(orchestrator_online):
    """Observed live: "dumela" was detected as Sesotho, matched the
    Setswana row, and was sent to Gemini to be "translated" — which
    returned `"Lumela! Nka o me/thusa jwang?"` with quote marks, a slash
    artifact, and a parenthetical note about Lesotho orthography, all of
    which the voice then read aloud.

    A row written and checked in a South African language is served as
    written. The detector is least reliable exactly here — "dumela" is
    shared by three languages — so translating between two guesses
    destroys human-verified text for nothing.
    """
    result = orchestrator_online.handle("dumela")
    assert result.source == "knowledge_base"
    # Sesotho, because the detector says Sesotho and a Sesotho row exists —
    # served exactly as written in the CSV, with no translator involved.
    assert result.text == "Lumela! Nka o thusa jwang?"
    assert '"' not in result.text and "(" not in result.text


def test_english_facts_are_still_translated(orchestrator_online):
    """The exemption is for in-language rows only. Factual rows are
    written in English and genuinely do need translating for a non-English
    user — the fix must not disable translation across the board.

    language_override is used because the query itself is English; without
    it translate_fact correctly short-circuits on an English target and
    the test would pass for the wrong reason.
    """
    result = orchestrator_online.handle(
        "what is the capital of south africa", language_override="isiZulu"
    )
    assert result.source == "knowledge_base"
    assert result.text == "TRANSLATED"


# --- a language the visitor CHOSE ---------------------------------------
#
# Observed live on 2026-09-28: the "Reply in" picker on Sesotho, "avuxeni"
# typed, and the Xitsonga greeting shown under "Replying in Sesotho", then
# read aloud by the Sesotho voice.


@pytest.fixture
def orchestrator_offline(monkeypatch):
    from bao.ai.client import GeminiClient
    from bao.ai.orchestrator import Orchestrator
    from bao.core.security import SecurityGuardrails
    from bao.services.language_detector import HeuristicLanguageDetector

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings()
    return Orchestrator(
        SecurityGuardrails(),
        HeuristicLanguageDetector(),
        KnowledgeRetriever(
            data_path=settings.knowledge_base_path,
            threshold=settings.similarity_threshold,
            min_coverage=settings.min_query_coverage,
        ),
        GeminiClient(settings),
    )


@pytest.mark.parametrize("mode", ["orchestrator_online", "orchestrator_offline"])
def test_a_chosen_language_is_greeted_in_that_language(mode, request):
    """Every language has its own curated greeting, so a chosen language
    needs no translation, online or off: serve that language's row.
    """
    orchestrator = request.getfixturevalue(mode)
    result = orchestrator.handle("avuxeni", language_override="Sesotho")

    assert result.text == "Lumela! Nka o thusa jwang?"
    assert result.text_language == "Sesotho"


def test_a_chosen_language_with_no_greeting_is_translated_online(orchestrator_online):
    result = orchestrator_online.handle("avuxeni", language_override="Swahili")
    assert result.text == "TRANSLATED"
    assert result.text_language is None, "machine output: intended, not known"


def test_offline_the_greeting_is_served_as_written_and_labelled(orchestrator_offline):
    """Nothing can translate offline, so the curated greeting is served as
    written - and the result says which language it is really in, for the
    badge and the voice.
    """
    result = orchestrator_offline.handle("avuxeni", language_override="Swahili")
    assert result.text == "Avuxeni! Ndzi nga ku pfuna njhani?"
    assert result.text_language == "Xitsonga"


def test_a_detected_language_does_not_swap_the_greeting(orchestrator_online):
    """Only a CHOICE swaps rows. A detector unsure between Xitsonga and
    siSwati must not replace the greeting the user typed with one in a
    language they may not speak.
    """
    result = orchestrator_online.handle("avuxeni")
    assert result.text == "Avuxeni! Ndzi nga ku pfuna njhani?"


def test_an_unreviewed_translation_is_served_as_a_counterpart_only_marked(knowledge):
    """The isiZulu police row shares Canonical_Id en-020 with the English
    one and is still needs-review. Its numbers check out, so counterpart()
    may return it, but always carrying reviewed=False: it must not become
    a side door that serves an unchecked translation as a checked one.
    """
    settings = Settings()
    english = knowledge[
        (knowledge["Canonical_Id"] == "en-020") & (knowledge["Language"] == "English")
    ]
    assert not english.empty, "fixture assumption: en-020 has an English row"
    question = english.iloc[0]["Question"]

    served = KnowledgeRetriever(data_path=settings.knowledge_base_path)
    fact = served.lookup(question)
    assert fact is not None and fact.canonical_id == "en-020" and fact.reviewed
    zulu = served.counterpart(fact, "isiZulu")
    assert zulu is not None and zulu.language == "isiZulu" and zulu.canonical_id == "en-020"
    assert zulu.reviewed is False
