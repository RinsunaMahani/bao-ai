"""The coverage gate: a high similarity score can still be answering a
different question from the one asked.

A TF-IDF vectorizer silently drops words outside its fitted vocabulary, so
an unanswerable question gets scored on whatever fragment remains. These
pin the three real false positives found on a live knowledge base.
"""

import pytest

from bao.core.config import Settings
from bao.knowledge.retriever import KnowledgeRetriever


@pytest.fixture
def retriever():
    settings = Settings()
    return KnowledgeRetriever(
        data_path=settings.knowledge_base_path,
        threshold=settings.similarity_threshold,
        min_coverage=settings.min_query_coverage,
    )


@pytest.mark.parametrize("query", [
    "what is the minimum wage in south africa",
    "what is the population of south africa",
])
def test_questions_the_kb_cannot_answer_are_not_returned_as_verified(retriever, query):
    """Each of these scored above the similarity threshold against the
    currency fact — on the shared words "south africa" alone — and was
    returned tagged source="knowledge_base", i.e. verified. A confidently
    wrong verified answer is worse than no answer, because a KB match
    bypasses generation entirely.
    """
    match = retriever.best_match_index(query)
    assert match is not None
    _, score = match
    assert score >= retriever.threshold, "precondition: similarity still clears the bar"
    assert retriever.query_fact(query) is None, f"{query!r} must fall through to generation"


@pytest.mark.parametrize("query", [
    "sawubona",
    "what is the capital of south africa",
    "who do i call for fire",
    "how do i apply for nsfas",
])
def test_genuine_questions_still_retrieve(retriever, query):
    """The gate must not be a blunt instrument — these are exactly the
    questions the knowledge base exists to answer.
    """
    assert retriever.query_coverage(query) >= retriever.min_coverage
    assert retriever.query_fact(query) is not None


def test_coverage_is_one_when_every_word_is_known(retriever):
    assert retriever.query_coverage("what is the capital of south africa") == pytest.approx(1.0)


def test_coverage_falls_when_the_informative_word_is_unknown(retriever):
    """"population" is the whole question; "south africa" is the part the
    KB happens to share with an unrelated row.
    """
    known = retriever.query_coverage("what is the capital of south africa")
    unknown = retriever.query_coverage("what is the population of south africa")
    assert unknown < known


def test_empty_and_stopword_only_queries_have_zero_coverage(retriever):
    assert retriever.query_coverage("") == 0.0
    assert retriever.query_coverage("what is the") == 0.0


def test_gate_can_be_disabled_for_comparison(tmp_path):
    """min_coverage=0 restores similarity-only behaviour, which is how the
    before/after numbers in REVIEW.md were produced.
    """
    settings = Settings()
    ungated = KnowledgeRetriever(
        data_path=settings.knowledge_base_path,
        threshold=settings.similarity_threshold,
        min_coverage=0.0,
    )
    assert ungated.query_fact("what is the population of south africa") is not None


@pytest.mark.parametrize("query", [
    "calculus in xitsonga",
    "explain calculus in xitsonga",
    "what is car in isiXhosa",
    "explain maths in isiZulu",
])
def test_queries_naming_a_target_language_go_to_generation(retriever, query):
    """A question asking for content RENDERED IN a language cannot be
    answered from a knowledge base that holds no per-language content.

    Observed live: "calculus in xitsonga" was answered with "Advanced
    multivariable calculus tutorials are usually held in the main sports
    complex hall" — a campus-venue row, returned as a VERIFIED answer.

    The coverage gate cannot catch this. Coverage detects questions the
    corpus has no words for; this is the opposite failure — every word is
    known ("calculus" from the tutorial row, the language name from
    another), but the combination means something the corpus does not
    contain. Similarity was 0.303 at 100% coverage.
    """
    assert retriever.lookup(query) is None


@pytest.mark.parametrize("query", [
    "where do i find my multivariable calculus tutorial",
    "how many official languages does south africa have",
    "what is the emergency number",
    "sawubona",
])
def test_the_language_guard_is_narrow_enough(retriever, query):
    """It must fire only on "in <language>", not on any mention of
    languages — the second query here names languages and is a legitimate
    knowledge-base question.
    """
    assert retriever.lookup(query) is not None
