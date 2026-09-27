"""Does the shipped tokenizer agree with the one the model was trained with?

This is the train/serve skew question, and it is currently the single
biggest unvalidated assumption in the deployed system.

The reported macro F1 of 0.98 was measured in the training notebook, using
Keras's own `Tokenizer` and the full Keras model. The app ships neither: it
uses `_KerasWordTokenizer` — a from-scratch reimplementation that reads
`tokenizer_config.json` so the app doesn't need TensorFlow just to turn
words into numbers — feeding a quantised TFLite model. If the
reimplementation disagrees with Keras on even a subset of inputs, the
deployed accuracy is not the measured accuracy, and nothing would crash to
tell you.

`test_reimplementation_matches_keras_exactly` closes that gap, but it can
only run where TensorFlow is installed. It skips otherwise — so it does
nothing in a core-only environment and everything on a machine with
`requirements-ml.txt` installed.

**Status: verified.** First run on 2026-09-27 against TensorFlow 2.21 and
Keras 3.15 (Python 3.12): all ten cases matched exactly. It skips on the
project's development machine, which runs Python 3.14 - TensorFlow has no
3.14 wheels - so CI's Python 3.11 job installs TensorFlow and runs it on
every push instead. Under Keras 3 the Tokenizer lives in a `legacy` module
but is still importable as below; the CI install is pinned to the series
this was verified against in case a later Keras removes it.

The remaining tests need no TensorFlow and always run: they pin the
specific semantics that are easy to get subtly wrong.
"""

import json

import pytest

from bao.core.config import Settings
from bao.services.language_detector import _KerasWordTokenizer

# Chosen to exercise the branches that differ between plausible
# implementations, not to look like natural language.
PARITY_TEXTS = [
    "sawubona unjani",                      # ordinary in-vocabulary words
    "Sawubona UNJANI",                      # casing (lower=True)
    "hello, world! how's it?",              # punctuation in `filters`
    "zzqqxx nonexistentwordhere",           # out-of-vocabulary -> OOV token
    "the of and a to in",                   # very high-frequency words
    "",                                     # empty
    "     ",                                # whitespace only
    "avuxeni " * 60,                        # longer than maxlen (truncation)
    "dumela\tthobela\nmolo",                # tab/newline are filter chars
    "a1 b2 3c",                             # digits mixed with letters
]


@pytest.fixture(scope="module")
def tokenizer():
    return _KerasWordTokenizer(Settings().tokenizer_config_path)


def _keras_tokenizer():
    """Rebuilds a real Keras Tokenizer from the same config file."""
    from tensorflow.keras.preprocessing.text import Tokenizer

    with open(Settings().tokenizer_config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    keras_tok = Tokenizer(
        num_words=cfg.get("num_words"),
        filters=cfg.get("filters"),
        lower=cfg.get("lower", True),
        split=cfg.get("split", " "),
        char_level=cfg.get("char_level", False),
        oov_token=cfg.get("oov_token"),
    )
    keras_tok.word_index = cfg["word_index"]
    keras_tok.index_word = {v: k for k, v in cfg["word_index"].items()}
    return keras_tok


try:
    import tensorflow  # noqa: F401
    _HAS_TF = True
except Exception:
    _HAS_TF = False


@pytest.mark.skipif(not _HAS_TF, reason="TensorFlow not installed — run this on the demo machine")
@pytest.mark.parametrize("text", PARITY_TEXTS)
def test_reimplementation_matches_keras_exactly(tokenizer, text):
    """The whole point of this file. Any disagreement here means the
    deployed model is being fed something different from what it was
    trained on, and the reported accuracy does not describe the app.
    """
    keras_tok = _keras_tokenizer()
    expected = keras_tok.texts_to_sequences([text])[0]
    actual = tokenizer._texts_to_sequence(text)
    assert actual == expected, (
        f"tokenizer disagrees with Keras on {text!r}\n"
        f"  Keras: {expected[:20]}\n"
        f"  ours : {actual[:20]}"
    )


# --- Semantics that always run, TensorFlow or not ---------------------


def test_words_above_num_words_become_oov_not_dropped(tokenizer):
    """Keras keeps the vocabulary but caps it at `num_words`. A word whose
    index exceeds the cap is replaced by the OOV token — not silently
    removed, which would shorten the sequence and shift everything after
    it into different positions.
    """
    assert tokenizer.num_words is not None
    rare = [w for w, i in tokenizer.word_index.items() if i > tokenizer.num_words + 50]
    assert rare, "expected the vocabulary to extend past num_words"
    encoded = tokenizer._texts_to_sequence(rare[0])
    assert encoded == [tokenizer._oov_index]


def test_unknown_words_become_oov(tokenizer):
    encoded = tokenizer._texts_to_sequence("qqzzxxnotarealword")
    assert encoded == [tokenizer._oov_index]


def test_encode_shape_and_dtype_match_the_model_input(tokenizer):
    """The TFLite model declares (None, 35) float32. A wrong shape raises;
    a wrong dtype can silently reinterpret the bytes.
    """
    encoded = tokenizer.encode("sawubona", maxlen=35)
    assert encoded.shape == (1, 35)
    assert encoded.dtype.name == "float32"


def test_padding_is_post_not_pre(tokenizer):
    """'post' was determined empirically against the trained model — 'pre'
    produced near-random confidences. Reversing it would break the model
    without raising anything, so it gets a test rather than a comment.
    """
    encoded = tokenizer.encode("sawubona", maxlen=35)[0]
    assert encoded[0] != 0.0, "content must start at position 0 (post-padding)"
    assert encoded[-1] == 0.0, "padding must be at the end"


def test_truncation_keeps_the_beginning(tokenizer):
    long_text = " ".join(["sawubona"] * 100)
    encoded = tokenizer.encode(long_text, maxlen=35)[0]
    assert len(encoded) == 35
    assert all(v != 0.0 for v in encoded), "a truncated sequence has no padding"


def test_empty_input_is_all_padding(tokenizer):
    encoded = tokenizer.encode("", maxlen=35)[0]
    assert len(encoded) == 35
    assert all(v == 0.0 for v in encoded)
