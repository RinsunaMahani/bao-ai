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

# --- re-indexing the same upload ----------------------------------------


def test_reindexing_a_document_replaces_it_rather_than_duplicating():
    """The uploader keeps its files across reruns, so pressing "Index
    Uploaded Documents" again re-indexed everything. The visible cost was
    in the prompt: the same passage was handed to Gemini twice, paying for
    the tokens and inviting the model to read a repetition as emphasis. It
    also distorts TF-IDF, which weights terms by how many chunks hold them.
    """
    from bao.knowledge.retriever import DocumentRetriever

    dr = DocumentRetriever()
    doc = "The submission deadline is 14 November 2026. The supervisor is Dr Mokoena."
    for _ in range(3):
        dr.add_document("handbook.pdf", doc)

    assert len(dr) == 1
    assert dr.search("when is the deadline").count("[Document:") == 1


def test_a_different_document_still_adds():
    """Replacement must key on the document, not clear the store."""
    from bao.knowledge.retriever import DocumentRetriever

    dr = DocumentRetriever()
    dr.add_document("a.txt", "Registration opens in January.")
    dr.add_document("b.txt", "Graduation is held in April.")
    dr.add_document("a.txt", "Registration opens in January.")
    assert dr.sources == ["a.txt", "b.txt"]
    assert len(dr) == 2


def test_a_document_whose_name_prefixes_another_is_not_swallowed():
    """Chunk ids embed the source name, so matching on an id prefix would
    make re-indexing "notes.txt" also delete "notes.txt.backup".
    """
    from bao.knowledge.retriever import DocumentRetriever

    dr = DocumentRetriever()
    dr.add_document("notes.txt", "Registration opens in January.")
    dr.add_document("notes.txt.backup", "Graduation is held in April.")
    dr.add_document("notes.txt", "Registration opens in February.")
    assert set(dr.sources) == {"notes.txt", "notes.txt.backup"}


def test_the_store_has_a_ceiling():
    """Nothing bounded this: every upload appended, so a large PDF
    re-indexed a few times grew both the store and the per-add refit cost
    without limit. Reaching the ceiling is reported, not silent.
    """
    from bao.knowledge.retriever import DocumentRetriever

    dr = DocumentRetriever(chunk_size=10, chunk_overlap=0, max_chunks=5)
    added = dr.add_document("big.txt", " ".join(f"word{i}" for i in range(500)))
    assert added == 5
    assert len(dr) == 5
    assert dr.add_document("another.txt", "more text entirely") == 0


def test_clearing_really_drops_everything():
    """Uploaded documents are the user's own, so `clear` has to mean it."""
    from bao.knowledge.retriever import DocumentRetriever

    dr = DocumentRetriever()
    dr.add_document("private.txt", "Confidential medical results for one person.")
    dr.clear()
    assert len(dr) == 0
    assert dr.sources == []
    assert dr.search("medical results") == ""


# --- uploads that are not what they claim to be --------------------------


@pytest.mark.parametrize("filename,payload", [
    ("corrupt.pdf", b"this is definitely not a pdf"),
    ("empty.pdf", b""),
    ("truncated.pdf", b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog"),
    ("weird.exe", b"MZ\x90\x00binary"),
    ("noext", b"some text with no extension"),
    ("bad_utf8.txt", b"\xff\xfe\x00invalid utf8 \xc3\x28"),
    ("malformed.csv", b'a,b\n"unclosed quote,c\n'),
])
def test_an_unreadable_upload_returns_nothing_rather_than_raising(filename, payload):
    """This is where a file chosen by someone else enters the app, so "not
    readable" has to be an ordinary outcome.

    A corrupt or empty PDF used to raise out of here, and the Streamlit
    upload handler has no try/except - so picking the wrong file replaced
    the page with a traceback during a demo. pypdf raises several distinct
    types (PdfStreamError, EmptyFileError), which is why the guard is not
    written against a list of them.
    """
    from bao.knowledge.loader import extract_text_from_bytes

    assert isinstance(extract_text_from_bytes(payload, filename), str)


def test_a_readable_upload_still_works():
    """The guard must not swallow success."""
    from bao.knowledge.loader import extract_text_from_bytes

    assert "deadline" in extract_text_from_bytes(
        b"The deadline is 14 November.", "notes.txt")
