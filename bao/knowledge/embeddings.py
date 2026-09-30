"""Bao AI - Text Embedding Strategies.

Defines a small interface so the retrieval layer doesn't care *how* text
gets turned into vectors. Today there's one implementation (TF-IDF, cheap
and fully offline). The interface exists so a future upgrade to real
sentence embeddings (e.g. a multilingual sentence-transformers model) is a
new class here, not a rewrite of retriever.py or vector_store.py.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# Words sklearn classifies as English stop words that are CONTENT words in
# this knowledge base. "fire" and "call" are the safety-critical ones — with
# sklearn's unmodified list, "who do i call for fire" scores 0.000 against
# the emergency-services row (0.322 once they're kept). The numerals appear
# in dialable emergency numbers.
_DOMAIN_CONTENT_WORDS = frozenset({
    "fire", "call", "find", "get", "take", "nine", "three", "first",
    "made", "please",
})

KNOWLEDGE_BASE_STOP_WORDS = (
    sorted(ENGLISH_STOP_WORDS - _DOMAIN_CONTENT_WORDS) if _HAS_SKLEARN else []
)
_STOP_WORD_SET = frozenset(KNOWLEDGE_BASE_STOP_WORDS)
_WORD = re.compile(r"\w+")

# Uploaded documents are matched on the first six letters of each content
# word, so the forms of one word meet: "presentation", "presentations" and
# "present"; "lecturer" and "lecture"; "submitted" and "submit".
#
# Exact words missed the case documents exist for. A short note a user
# typed up - office hours, a presentation date, a deadline - was found by 1
# of 8 plain questions about it: "how long is each presentation" scored
# 0.000 against a note about "presentations". Six letters found the right
# passage for all 8, mixed with a 5,000-word distractor. Character
# n-grams also found all 8, but matched across unrelated words
# ("photosynthesis" against "speech synthesis"); five letters confused
# "office" with "official".
DOCUMENT_STEM_LENGTH = 6


def document_terms(text: str) -> list[str]:
    """The terms an uploaded document is indexed and searched by: its
    content words, each cut to DOCUMENT_STEM_LENGTH letters.
    """
    return [
        word[:DOCUMENT_STEM_LENGTH]
        for word in _WORD.findall(text.lower())
        if word not in _STOP_WORD_SET
    ]


def has_sklearn_support() -> bool:
    return _HAS_SKLEARN


class EmbeddingModel(ABC):
    """Turns text into vectors that support similarity comparison."""

    @abstractmethod
    def fit(self, corpus: Iterable[str]) -> None: ...

    @abstractmethod
    def transform(self, texts: Iterable[str]):
        """Returns a vector/matrix representation. Shape and type are
        implementation-defined; callers should only pass the result into
        the matching `Similarity` function from vector_store.py.
        """


class TfidfEmbeddings(EmbeddingModel):
    """Default, dependency-light embedding strategy — no network, no GPU,
    no model download. Good enough for keyword-overlapping queries; the
    known limitation (documented, not hidden) is that it won't match true
    paraphrases with little lexical overlap. That's exactly the gap a
    future sentence-embedding implementation of this same interface would
    close.

    Stop words are filtered because without them, function words alone
    cleared the similarity threshold on a 32-row corpus of short texts:
    "what is quantom physics" scored 0.352 against the South African
    currency fact on the shared words "what is" and was returned to a user
    as a verified answer. Filtering them takes false positives on the
    70-query labelled set from 37 to 22 at threshold 0.20 with no recall
    loss (F1 0.59 -> 0.70), and drops that query to 0.000 so it correctly
    falls through to generation.

    The remaining 22 false positives are a genuinely harder class that
    stop words cannot touch: queries sharing real content words with a KB
    row. "what is the minimum wage in south africa" still scores 0.666
    against the currency fact on "south africa". No threshold separates
    those from true paraphrases — closing that gap needs semantic
    embeddings (see SentenceEmbeddings below) or broader KB coverage.
    """

    def __init__(self):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for TfidfEmbeddings.")
        self._vectorizer = TfidfVectorizer(
            lowercase=True, stop_words=KNOWLEDGE_BASE_STOP_WORDS
        )
        self._is_fitted = False

    def fit(self, corpus: Iterable[str]) -> None:
        self._matrix = self._vectorizer.fit_transform(corpus)
        self._is_fitted = True

    def transform(self, texts: Iterable[str]):
        if not self._is_fitted:
            raise RuntimeError("TfidfEmbeddings.fit() must be called before transform().")
        return self._vectorizer.transform(texts)

    @property
    def fitted_matrix(self):
        if not self._is_fitted:
            raise RuntimeError("TfidfEmbeddings.fit() must be called before accessing fitted_matrix.")
        return self._matrix


class DocumentTfidfEmbeddings(TfidfEmbeddings):
    """TF-IDF over `document_terms`, for uploaded documents.

    Kept apart from the knowledge base's vectorizer on purpose. The
    knowledge base's threshold and coverage gate were measured on whole
    words (see TfidfEmbeddings), and shortening its words would move every
    one of those numbers. Documents are the opposite case: long prose the
    user wrote in their own words, where a question rarely repeats the
    exact form of a word. Sublinear term frequency stops a word repeated
    through a long passage from outweighing the rest of the question.
    """

    def __init__(self):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for DocumentTfidfEmbeddings.")
        self._vectorizer = TfidfVectorizer(analyzer=document_terms, sublinear_tf=True)
        self._is_fitted = False


def has_semantic_support() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class SentenceEmbeddings(EmbeddingModel):
    """Multilingual sentence-embedding strategy — the intended upgrade
    path from TfidfEmbeddings, implementing the same interface so
    retriever.py and vector_store.py need no changes at all.

    The default model (`paraphrase-multilingual-MiniLM-L12-v2`, ~470 MB)
    is chosen for two specific reasons over a larger or English-only one:
    it covers 50+ languages including the South African ones this project
    targets, and it's trained specifically on paraphrase similarity, which
    is the exact failure mode TF-IDF exhibits here.

    NOT YET BENCHMARKED against TF-IDF in this repository — the comparison
    requires downloading model weights from Hugging Face, which wasn't
    possible in the environment where this class was written. Run
    `python compare_retrievers.py` to produce the comparison on your own
    machine; that script prints both backends' numbers side by side
    against the same 70-query evaluation set. Adopt this only if it
    actually wins on paraphrase recall and negative rejection — if it
    doesn't, keep TF-IDF and document why. That decision should come from
    the measurement, not from "embeddings are more modern."

    Practical warning found the hard way: installing
    `sentence-transformers` pulls in torch, which can collide with an
    existing TensorFlow install at the native-library level (observed as a
    hard segfault when both were present). If the TFLite language detector
    stops working after installing this, that collision is the first thing
    to check — consider a separate virtualenv for the retrieval
    experiment.
    """

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SentenceEmbeddings. "
                "Install with: pip install -r requirements-semantic.txt"
            ) from e
        self._model = SentenceTransformer(model_name)
        self._matrix = None
        self._is_fitted = False

    def fit(self, corpus: Iterable[str]) -> None:
        # Unlike TF-IDF there's nothing to "learn" from the corpus — the
        # model is pretrained. This just encodes it once so queries can be
        # compared against a precomputed matrix, matching the interface's
        # fit/transform contract.
        self._matrix = self._model.encode(list(corpus), normalize_embeddings=True)
        self._is_fitted = True

    def transform(self, texts: Iterable[str]):
        if not self._is_fitted:
            raise RuntimeError("SentenceEmbeddings.fit() must be called before transform().")
        return self._model.encode(list(texts), normalize_embeddings=True)

    @property
    def fitted_matrix(self):
        if not self._is_fitted:
            raise RuntimeError("SentenceEmbeddings.fit() must be called before accessing fitted_matrix.")
        return self._matrix