"""The evaluation harness had no tests, which is the wrong way round.

Every other module is tested so the app behaves. This module produces the
numbers that go into the report and onto the slides — so a defect here
doesn't cause a visible failure, it causes a wrong claim made confidently
in front of judges. That is worse.
"""

import pytest

from bao.core.config import Settings
from bao.evaluation import evaluate_retrieval_self_consistency
from bao.knowledge.retriever import KnowledgeRetriever


@pytest.fixture(scope="module")
def retriever():
    settings = Settings()
    return KnowledgeRetriever(
        data_path=settings.knowledge_base_path,
        threshold=settings.similarity_threshold,
        min_coverage=settings.min_query_coverage,
    )


def test_self_consistency_counts_every_row(retriever):
    report = evaluate_retrieval_self_consistency(retriever)
    assert report.total == len(retriever.dataframe)
    assert report.correct <= report.total


def test_every_counted_row_is_either_correct_or_explained(retriever):
    """A row that fails to retrieve is skipped by the loop but still counted
    in `total`, so it lowers the reported accuracy while appearing in
    neither `correct` nor `misses`. That makes a number impossible to
    reconcile with its own explanation — the worst kind of evaluation bug,
    because it looks fine.
    """
    report = evaluate_retrieval_self_consistency(retriever)
    unexplained = report.total - report.correct - len(report.misses)
    assert unexplained == 0, (
        f"{unexplained} row(s) counted against accuracy but absent from "
        "`misses` — the report cannot be reconciled with itself"
    )


def test_report_carries_its_own_caveat(retriever):
    """~100% here is the expected result, not an achievement: the check
    asks the knowledge base its own questions verbatim. The rendered report
    must say so, or the figure gets quoted as an accuracy claim.
    """
    rendered = evaluate_retrieval_self_consistency(retriever).to_markdown()
    lowered = rendered.lower()
    assert "regression check" in lowered
    assert "rag_eval" in lowered, "must point at the number actually worth reporting"


def test_accuracy_is_not_silently_divided_by_zero(tmp_path):
    """An empty or unreadable knowledge base must not produce a confident
    figure — an uninitialised retriever should raise rather than report.
    """
    broken = KnowledgeRetriever(data_path=str(tmp_path / "missing.csv"))
    assert not broken.is_initialized
    with pytest.raises(Exception):
        evaluate_retrieval_self_consistency(broken)


def test_misses_record_what_was_expected_and_what_came_back(tmp_path):
    """When a row does miss, the report has to say which answer was
    returned instead — otherwise a failing evaluation gives no lead to
    follow.
    """
    path = tmp_path / "kb.csv"
    # Two rows worded so similarly that at least one must mis-retrieve.
    path.write_text(
        "Question,Answer\n"
        "capital city information,Pretoria is the administrative capital.\n"
        "capital city information detail,Cape Town is the legislative capital.\n"
    )
    report = evaluate_retrieval_self_consistency(
        KnowledgeRetriever(data_path=str(path), threshold=0.0, min_coverage=0.0)
    )
    for question, expected, retrieved in report.misses:
        assert question and expected and retrieved
        assert expected != retrieved, "a recorded miss must actually differ"
