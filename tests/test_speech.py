"""Tests for the speech layer.

Neither TTS backend can run here: MMS needs a Hugging Face download and
edge-tts needs Microsoft's endpoint. So these cover the parts that are
pure logic — which backend gets picked, and how text is chunked before it
reaches one — which is exactly where the bugs were. Audio quality still
has to be judged by ear.
"""

import pytest

import bao.services.speech as speech_module
from bao.services.speech import (
    clean_text_for_speech,
    select_backend,
    split_into_sentences,
)

# --- Backend routing ---------------------------------------------------


def test_english_prefers_edge_when_available(monkeypatch):
    """The whole point of adding edge-tts: English must NOT go to the
    Afrikaans MMS voice when a real en-ZA voice exists.
    """
    monkeypatch.setattr(speech_module, "_HAS_EDGE_BACKEND", True)
    monkeypatch.setattr(speech_module, "_HAS_MMS_BACKEND", True)
    assert select_backend("English", "auto") == "edge"
    assert select_backend("Afrikaans", "auto") == "edge"
    assert select_backend("isiZulu", "auto") == "edge"


def test_languages_without_an_sa_voice_stay_on_mms(monkeypatch):
    """Only three of the eleven have a Microsoft locale. The other eight
    must fall through to MMS rather than silently getting no audio.
    """
    monkeypatch.setattr(speech_module, "_HAS_EDGE_BACKEND", True)
    monkeypatch.setattr(speech_module, "_HAS_MMS_BACKEND", True)
    for language in ("Xitsonga", "Sepedi", "Sesotho", "Setswana",
                     "siSwati", "Tshivenda", "isiXhosa", "isiNdebele"):
        assert select_backend(language, "auto") == "mms", language


def test_falls_back_to_mms_when_edge_not_installed(monkeypatch):
    monkeypatch.setattr(speech_module, "_HAS_EDGE_BACKEND", False)
    monkeypatch.setattr(speech_module, "_HAS_MMS_BACKEND", True)
    assert select_backend("English", "auto") == "mms"


def test_no_backend_returns_none_rather_than_raising(monkeypatch):
    monkeypatch.setattr(speech_module, "_HAS_EDGE_BACKEND", False)
    monkeypatch.setattr(speech_module, "_HAS_MMS_BACKEND", False)
    assert select_backend("English", "auto") is None


def test_explicit_preference_is_not_silently_overridden(monkeypatch):
    """backend = "mms" in config.toml must actually force MMS — a config
    knob that quietly does something else is worse than no knob.
    """
    monkeypatch.setattr(speech_module, "_HAS_EDGE_BACKEND", True)
    monkeypatch.setattr(speech_module, "_HAS_MMS_BACKEND", True)
    assert select_backend("English", "mms") == "mms"
    assert select_backend("Xitsonga", "edge") is None


# --- Text preparation --------------------------------------------------


def test_apostrophes_survive_cleaning():
    """"ematshan'weni" (Xitsonga) and similar forms use the apostrophe as
    part of the orthography. Stripping it changes the word rather than
    tidying the markdown.
    """
    assert "ematshan'weni" in clean_text_for_speech("**ematshan'weni**")


def test_markdown_is_stripped_so_it_is_not_read_aloud():
    cleaned = clean_text_for_speech("**Cloud** computing (storage) uses `servers`")
    for token in ("**", "(", ")", "`"):
        assert token not in cleaned


def test_sentence_terminators_are_preserved():
    """Both backends use punctuation for prosody. The earlier splitter
    discarded it, which flattened long replies into a monotone.
    """
    sentences = split_into_sentences("First sentence. Second one! Third?")
    assert sentences == ["First sentence.", "Second one!", "Third?"]


def test_long_unpunctuated_text_is_still_chunked():
    """A bullet list with no full stops is a normal LLM reply. Without a
    word-boundary backstop the whole thing went to VITS as one sequence.
    """
    long_text = " ".join(["word"] * 300)
    chunks = split_into_sentences(long_text)
    assert len(chunks) > 1
    assert all(len(c) <= speech_module._MAX_CHUNK_CHARS for c in chunks)
    # No word may be cut in half by the split.
    assert "".join(chunks).replace(" ", "") == long_text.replace(" ", "")


def test_empty_and_whitespace_text_produce_no_chunks():
    assert split_into_sentences("") == []
    assert split_into_sentences("   \n  ") == []


def test_rate_percent_conversion():
    """One config value drives both backends, so the multiplier has to map
    onto edge-tts's percentage string correctly.
    """
    assert speech_module._rate_percent(0.85) == "-15%"
    assert speech_module._rate_percent(1.0) == "+0%"
    assert speech_module._rate_percent(None) == "+0%"


def test_refused_request_is_not_reported_as_a_missing_model():
    """The correction that matters most in this file.

    transformers renders an absent repo, a gated repo and a refused request
    with the same sentence: "is not a local folder and is not a valid model
    identifier". An earlier version of _explain_failure read that sentence
    as proof of absence and told users the model "is not a published model,
    this cannot be fixed in code".

    A live probe then showed nine repos returning HTTP 401 while a tenth
    returned 200 and loaded — so the message was confidently wrong, and it
    sent someone off to redesign around a problem that was authentication.
    401 is not 404.
    """
    error = Exception(
        "facebook/mms-tts-sot is not a local folder and is not a valid model "
        "identifier listed on 'https://huggingface.co/models' "
        "(401 Client Error: Unauthorized)"
    )
    message = speech_module._explain_failure("mms", "Sesotho", {"Sesotho": "sot"}, error)
    assert "HF_TOKEN" in message
    assert "not a missing model" in message
    assert "not a published model" not in message


def test_genuine_404_is_still_reported_as_absence():
    """The other half: a real 404 should say so plainly, because that one
    genuinely cannot be fixed in code.
    """
    error = Exception("404 Client Error: RepositoryNotFound for url ...")
    message = speech_module._explain_failure("mms", "Sesotho", {"Sesotho": "sot"}, error)
    assert "404" in message
    assert "HF_TOKEN" not in message


def test_ambiguous_message_admits_it_is_ambiguous():
    """When transformers gives no status code, the honest answer is "this
    could be either" — not a guess dressed as a finding.
    """
    error = Exception("facebook/mms-tts-ven is not a local folder and is not a valid model identifier")
    message = speech_module._explain_failure("mms", "Tshivenda", {"Tshivenda": "ven"}, error)
    assert "may be" in message or "could not be fetched" in message
    assert "HF_TOKEN" in message


def test_network_error_is_distinguished_from_a_missing_model():
    """Different cause, different remedy: one is worth retrying, the other
    never is.
    """
    error = Exception("Connection timeout while resolving huggingface.co")
    message = speech_module._explain_failure("mms", "Xitsonga", {"Xitsonga": "tso"}, error)
    assert "download" in message.lower()
    assert "not a published model" not in message


def test_edge_backend_cleans_markdown_before_speaking(monkeypatch):
    """Reported from a live session: a reply containing "**Differential
    Calculus**" was spoken with the asterisks read aloud.

    The cause was a missing step rather than a broken one. Only the MMS
    branch cleaned its input, because cleaning happened inside
    split_into_sentences() — which the edge branch does not call, since
    edge-tts handles long text itself. So raw Gemini markdown went straight
    to the voice.
    """
    captured = {}

    def fake_edge(text, voice, rate):
        captured["text"] = text
        return speech_module.SpeechAudio(data=b"x", mime="audio/mpeg")

    monkeypatch.setattr(speech_module, "_HAS_EDGE_BACKEND", True)
    monkeypatch.setattr(speech_module, "_synthesize_edge", fake_edge)

    speech_module.synthesize_speech(
        "Yi katsa:\n\n1. **Differential Calculus** yi lavisisa mpimo.",
        "English",
        backend="edge",
    )

    assert "*" not in captured["text"], "markdown reached the voice"
    assert "Differential Calculus" in captured["text"], "content must survive"


@pytest.mark.parametrize("raw,expected_gone", [
    ("**bold**", "*"), ("__under__", "_"), ("# Heading", "#"),
    ("`code`", "`"), ("| a | b |", "|"), ("> quote", ">"), ("~~strike~~", "~"),
])
def test_markdown_tokens_are_stripped(raw, expected_gone):
    assert expected_gone not in speech_module.clean_text_for_speech(raw)


def test_apostrophes_still_survive_the_wider_cleaner():
    """Widening the token list must not catch the apostrophe: it is part of
    the orthography in Xitsonga ("ematshan'weni") and isiZulu, so removing
    it changes the word rather than tidying it.
    """
    assert "ematshan'weni" in speech_module.clean_text_for_speech("**ematshan'weni**")


def test_local_checkpoint_unlocks_sepedi_and_the_sotho_tswana_routes(tmp_path, monkeypatch):
    """One locally trained checkpoint changes the tier of THREE languages.

    Sotho-Tswana substitution was originally dropped outright because
    Sesotho, Setswana and Sepedi all 404 on Hugging Face — there was no
    voice anywhere in the family to substitute from. That reasoning was
    correct for pretrained voices and wrong the moment this project
    trained its own Sepedi model, which the speech layer previously had
    no way to load at all.

    Pins the coupling so it cannot regress: register a Sepedi checkpoint
    and Sepedi becomes native while Sesotho and Setswana become routable;
    remove it and all three fall back. A route that outlives its target
    voice promises audio that never arrives.
    """
    from bao.services import speech

    # A local checkpoint is a VITS model, so it needs the same
    # torch/transformers stack MMS does. Pinned rather than left to this
    # machine's installed packages: CI installs no speech extras, and a
    # test whose result depends on that is testing the runner, not the
    # policy. (Before coverage was derived from the resolver, this test
    # passed in CI precisely BECAUSE coverage ignored the backend — it was
    # asserting a claim the audio path could not honour.)
    monkeypatch.setattr(speech, "_HAS_MMS_BACKEND", True)
    monkeypatch.setattr(speech, "_HAS_EDGE_BACKEND", True)

    original = dict(speech.LOCAL_VOICE_MODELS)
    try:
        speech.load_local_voices({})
        assert "Sesotho" not in speech.related_language_voices()
        assert "Setswana" not in speech.related_language_voices()

        checkpoint = tmp_path / "sepedi_vits"
        checkpoint.mkdir()
        speech.load_local_voices({"Sepedi": str(checkpoint)})

        routes = speech.related_language_voices()
        assert routes["Sesotho"] == "Sepedi"
        assert routes["Setswana"] == "Sepedi"
        coverage = speech.voice_coverage()
        assert coverage["Sepedi"] == "native (locally trained)"
        assert coverage["Sesotho"] == "related (Sepedi)"

        # Nguni routes are independent of any local checkpoint.
        assert routes["isiXhosa"] == "isiZulu"
    finally:
        speech.load_local_voices(original)


def test_a_configured_but_missing_checkpoint_is_ignored(tmp_path):
    """The app must start on a machine that has not synced model files.

    A configured path that does not exist is dropped with a warning
    rather than raising, and must not leave a Tier 2 route pointing at a
    voice that cannot speak.
    """
    from bao.services import speech

    original = dict(speech.LOCAL_VOICE_MODELS)
    try:
        registered = speech.load_local_voices({"Sepedi": str(tmp_path / "absent")})
        assert registered == {}
        assert "Sesotho" not in speech.related_language_voices()
    finally:
        speech.load_local_voices(original)


# --- how much of a reply actually gets spoken ---------------------------


def test_a_long_reply_is_not_read_out_in_full():
    """Nothing capped this, and the cost is invisible from the text.

    MMS runs a VITS forward pass per sentence on CPU, so a detailed answer
    measured here produced 4.7 MB of WAV - two and a half minutes of audio
    - and took 60 seconds to synthesize, while the text had been on screen
    and readable the whole time.
    """
    from bao.services.speech import trim_for_speech

    long_answer = "This is a sentence about calculus. " * 40
    spoken, trimmed = trim_for_speech(long_answer, 400)
    assert trimmed
    assert len(spoken) <= 400


def test_the_cut_lands_on_a_sentence_boundary():
    """A clip that stops mid-word sounds like a fault rather than a
    summary.
    """
    from bao.services.speech import trim_for_speech

    spoken, _ = trim_for_speech("One. Two. Three. " * 40, 400)
    assert spoken.endswith("."), spoken[-30:]


def test_a_reply_shorter_than_the_cap_is_untouched():
    from bao.services.speech import trim_for_speech

    assert trim_for_speech("Avuxeni, hi njhani?", 400) == ("Avuxeni, hi njhani?", False)


def test_text_with_no_sentence_or_word_breaks_still_gets_cut():
    """Degenerate input must not defeat the cap - the point is bounding
    synthesis time, and an unbroken string is the worst case for it.
    """
    from bao.services.speech import trim_for_speech

    spoken, trimmed = trim_for_speech("x" * 900, 400)
    assert trimmed and len(spoken) == 400


def test_the_cap_can_be_switched_off():
    from bao.services.speech import trim_for_speech

    long_answer = "One. Two. " * 100
    assert trim_for_speech(long_answer, 0) == (long_answer, False)


def test_trimming_is_disclosed_to_the_caller(monkeypatch):
    """A shortened reading the listener is told about is a summary; one
    they are not told about is the app appearing to lose the end of its own
    answer.
    """
    monkeypatch.setattr(speech_module, "_HAS_EDGE_BACKEND", False)
    monkeypatch.setattr(speech_module, "_HAS_MMS_BACKEND", False)

    notes = []
    speech_module.synthesize_speech(
        "This is a sentence about calculus. " * 40, "English",
        max_characters=400, on_trim=notes.append,
    )
    assert notes and "full text" in notes[0]
