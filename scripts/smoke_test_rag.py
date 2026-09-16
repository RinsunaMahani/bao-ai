"""
Bao AI - RAG & Edge Ingestion Smoke Test.

Run manually:  python scripts/smoke_test_rag.py

Moved out of the repository root and renamed. It used to be
`test_rag_edge.py`, which was misleading in both directions: pytest never
collected it (pyproject sets `testpaths = ["tests"]`), so it looked like a
test that silently never ran — while a bare `pytest` invocation from a
different working directory WOULD have collected it and executed a script
that prints rather than asserts. It is a smoke script; it is now named and
located like one.
"""

from bao.knowledge.retriever import DocumentRetriever


def run_test():
    retriever = DocumentRetriever()

    sample_doc = """
    Bao AI is a hybrid intelligence architecture designed for multilingual edge-first computing.
    The system features an offline detection engine capable of identifying all 11 official South African languages
    including isiZulu, Sepedi, Xitsonga, and Afrikaans with sub-millisecond latency.
    The local edge layer handles retrieval directly on device.
    When complex generative reasoning or multi-turn translation is required, local RAG context is packaged and sent
    to Gemini via ai/client.py.
    """

    print("--- 1. Indexing Document ---")
    chunk_count = retriever.add_document(doc_id="bao_architecture.txt", content=sample_doc)
    print(f"Indexed document into {chunk_count} chunks.")
    print(f"Total stored chunks in memory: {len(retriever)}\n")

    print("--- 2. Testing Edge Retrieval ---")
    query = "Which South African languages are supported on the edge?"
    print(f"Query: \"{query}\"\n")

    results = retriever.search(query, top_k=2)
    if results:
        print("MATCHED CONTEXT RETRIEVED:")
        print("=" * 60)
        print(results)
        print("=" * 60)
    else:
        print("No matching context found.")


if __name__ == "__main__":
    run_test()
