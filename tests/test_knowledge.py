import pytest

from bao.knowledge.loader import chunk_text
from bao.knowledge.retriever import DocumentRetriever, KnowledgeRetriever

SAMPLE_CSV = """Category,Trigger,Answer
Greeting,hello bao,"Hello! I am Bao, your offline-first assistant."
Fact,capital of south africa,"South Africa has three capital cities: Pretoria, Cape Town, and Bloemfontein."
"""


@pytest.fixture
def knowledge_csv(tmp_path):
    path = tmp_path / "test_data.csv"
    path.write_text(
        "Question,Answer\n"
        "hello bao,Hello! I am Bao.\n"
        "capital of south africa,Pretoria is the administrative capital.\n"
    )
    return str(path)


def test_retriever_finds_exact_match(knowledge_csv):
    retriever = KnowledgeRetriever(data_path=knowledge_csv, threshold=0.15)
    assert retriever.is_initialized
    answer = retriever.query_fact("hello bao")
    assert answer is not None and "Bao" in answer


def test_retriever_returns_none_below_threshold(knowledge_csv):
    # threshold 1.01 is unreachable — cosine similarity maxes at 1.0
    retriever = KnowledgeRetriever(data_path=knowledge_csv, threshold=1.01)
    answer = retriever.query_fact("hello bao")
    assert answer is None


def test_retriever_missing_file_does_not_crash(tmp_path):
    retriever = KnowledgeRetriever(data_path=str(tmp_path / "does_not_exist.csv"))
    assert not retriever.is_initialized
    assert retriever.query_fact("anything") is None


def test_known_limitation_tfidf_false_positive_on_shared_stopwords(knowledge_csv):
    """Documents the limitation described in docs/ARCHITECTURE.md: an
    unrelated query can still clear a low similarity threshold purely on
    shared common words. This test exists so a future embeddings.py swap
    (see knowledge/embeddings.py) has something concrete to improve on —
    if this starts failing, retrieval precision has gotten better, which
    is a reason to update the docs, not the test.
    """
    retriever = KnowledgeRetriever(data_path=knowledge_csv, threshold=0.15)
    result = retriever.best_match_index("what is the capital of a fictional planet")
    assert result is not None
    _, score = result
    assert score >= 0.15  # matches the documented false positive


def test_function_word_only_queries_are_rejected():
    """Regression test for a bug that reached a live user: "what is quantom
    physics" scored 0.352 against the South African currency fact — on the
    shared words "what is" alone — cleared the threshold, and was returned
    verbatim as a *verified* knowledge-base answer, since a KB match
    bypasses generation entirely.

    Filtering stop words drops these to 0.0. This test pins the class of
    failure, not one embarrassing example: every query below shares only
    function words with the knowledge base and must retrieve nothing.
    Uses the real 32-row data/african_data.csv, because the bug depends on
    a small corpus of short texts, not on a synthetic fixture.
    """
    from bao.core.config import Settings

    settings = Settings()
    retriever = KnowledgeRetriever(data_path=settings.knowledge_base_path, threshold=settings.similarity_threshold)

    for query in (
        "what is quantom physics",
        "what is the speed of light",
        "what is the airspeed velocity of an unladen swallow",
    ):
        assert retriever.query_fact(query) is None, f"{query!r} should reach generation, not the KB"
        _, score = retriever.best_match_index(query)
        assert score == 0.0, f"{query!r} scored {score:.3f} on function words alone"


def test_domain_content_words_survive_stop_word_filtering():
    """sklearn's English stop list contains "fire" and "call", which are
    content words here — filtering them blindly takes "who do i call for
    fire" to 0.000 against the emergency-services row. The curated list
    keeps them. This is safety-relevant, so it gets a test.
    """
    from bao.core.config import Settings

    settings = Settings()
    retriever = KnowledgeRetriever(data_path=settings.knowledge_base_path, threshold=settings.similarity_threshold)
    _, score = retriever.best_match_index("who do i call for fire")
    assert score > 0.3


def test_document_retriever_chunks_and_searches():
    dr = DocumentRetriever()
    dr.add_document("doc.txt", "Bao AI supports eleven official South African languages.")
    result = dr.search("which languages are supported")
    assert "languages" in result.lower()


def test_document_retriever_clear():
    dr = DocumentRetriever()
    dr.add_document("doc.txt", "some content here")
    dr.clear()
    assert len(dr) == 0
    assert dr.search("anything") == ""


def test_chunk_text_short_document_single_chunk():
    chunks = chunk_text("just a few words", source_name="short.txt", chunk_size=300)
    assert len(chunks) == 1


def test_chunk_text_long_document_multiple_chunks():
    long_text = " ".join(["word"] * 1000)
    chunks = chunk_text(long_text, source_name="long.txt", chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1