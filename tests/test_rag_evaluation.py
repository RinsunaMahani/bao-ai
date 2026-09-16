from pathlib import Path

import pytest

from bao.core.config import Settings
from bao.knowledge.retriever import KnowledgeRetriever
from bao.rag_evaluation import (
    RagEvalCase,
    ThresholdResult,
    evaluate_threshold,
    load_rag_eval_cases,
    per_category_breakdown,
    sweep_thresholds,
)

_settings = Settings()
_eval_csv = Path(__file__).parent.parent / "eval" / "rag_eval.csv"


def test_threshold_result_metrics():
    r = ThresholdResult(threshold=0.25, true_positives=8, false_positives=2, true_negatives=6, false_negatives=4)
    assert r.precision == pytest.approx(0.8)
    assert r.recall == pytest.approx(8 / 12)
    assert r.f1 == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))
    assert r.false_positive_rate == pytest.approx(2 / 8)


def test_threshold_result_handles_empty_division():
    """All-zero counts must not raise ZeroDivisionError — a sweep at an
    extreme threshold can legitimately produce empty buckets.
    """
    r = ThresholdResult(threshold=0.9, true_positives=0, false_positives=0, true_negatives=0, false_negatives=0)
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert r.f1 == 0.0
    assert r.false_positive_rate == 0.0


@pytest.mark.skipif(not _eval_csv.exists(), reason="eval/rag_eval.csv not present")
def test_eval_set_loads_and_has_all_three_categories():
    cases = load_rag_eval_cases(_eval_csv)
    assert len(cases) >= 40
    categories = {c.category for c in cases}
    assert {"exact", "paraphrase", "negative"} <= categories


@pytest.mark.skipif(not _eval_csv.exists(), reason="eval/rag_eval.csv not present")
def test_exact_queries_all_retrieve_correctly_at_default_threshold():
    """Verbatim knowledge-base questions are the easy case — if these ever
    stop working, retrieval is broken outright, not merely imprecise.
    """
    retriever = KnowledgeRetriever(
        data_path=_settings.knowledge_base_path, threshold=_settings.similarity_threshold
    )
    cases = load_rag_eval_cases(_eval_csv)
    breakdown = per_category_breakdown(retriever, cases, _settings.similarity_threshold)
    correct, total = breakdown["exact"]
    assert correct == total, f"exact-match retrieval regressed: {correct}/{total}"


@pytest.mark.skipif(not _eval_csv.exists(), reason="eval/rag_eval.csv not present")
def test_documented_tfidf_precision_recall_tradeoff_still_holds():
    """Pins the central empirical finding from the threshold sweep: with
    TF-IDF there is NO threshold that handles both paraphrases and
    unrelated queries well. A high threshold rejects negatives but kills
    paraphrase recall; a low one keeps paraphrases but accepts almost
    every unrelated query.

    This is the evidence behind not simply "raising the threshold" to fix
    the documented false positive, and the evidence for eventually moving
    to semantic embeddings. If this test starts failing, retrieval quality
    has fundamentally changed and the docs need updating — that's a good
    outcome, not a broken test.
    """
    retriever = KnowledgeRetriever(
        data_path=_settings.knowledge_base_path, threshold=_settings.similarity_threshold
    )
    cases = load_rag_eval_cases(_eval_csv)

    # apply_coverage_gate=False is deliberate and load-bearing: this test
    # pins the behaviour of SIMILARITY ALONE, which is precisely the
    # argument for why the coverage gate had to be added. Scoring it with
    # the gate on would measure a different system and destroy the claim.
    # See test_shipped_pipeline_trades_paraphrase_recall_for_precision for
    # what the deployed pipeline actually does.
    low = per_category_breakdown(retriever, cases, 0.15, apply_coverage_gate=False)
    high = per_category_breakdown(retriever, cases, 0.55, apply_coverage_gate=False)

    # Low threshold: paraphrases mostly work, negatives still mostly aren't
    # rejected. Stop-word filtering moved this last number from 0.10 to
    # 0.50 by eliminating pure function-word matches — a real improvement,
    # and still only a coin flip, which is the point: half the unrelated
    # queries share genuine content words with a KB row.
    assert low["paraphrase"][0] / low["paraphrase"][1] >= 0.6
    assert 0.4 <= low["negative"][0] / low["negative"][1] <= 0.7

    # High threshold: the trade-off inverts — paraphrase recall collapses.
    assert high["paraphrase"][0] / high["paraphrase"][1] <= 0.3
    # Note the high-threshold negative rejection is asserted loosely (>=0.7,
    # not >=0.9): expanding the negative set from 20 to 40 with deliberate
    # lexical traps ("what is the capital of Australia", "who is the
    # president of South Africa") showed TF-IDF still can't reject all of
    # them even at 0.55, because they share so much surface vocabulary with
    # real KB entries. That's the finding, not a slack assertion.
    assert high["negative"][0] / high["negative"][1] >= 0.7


def test_lexical_trap_negatives_are_the_hard_case():
    """Documents the retrieval failure that REMAINS after stop-word
    filtering, and is therefore the real argument for semantic embeddings.

    The function-word version of this trap ("what is the speed of light"
    scoring 0.53 on the shared words "what is the of") is fixed and now
    scores 0.0 — see test_function_word_only_queries_are_rejected. What
    stop words cannot fix is genuine content-word overlap: "what is the
    minimum wage in south africa" scores 0.666 against the currency fact
    on "south africa" alone.

    This is the concrete evidence that the residual problem is TF-IDF's
    lexical nature, not a threshold that needs nudging — no cutoff
    separates 0.666 from the legitimate paraphrases scoring in the same
    band. That gap is what SentenceEmbeddings exists to close, and
    compare_retrievers.py is what should decide whether it actually does.
    """
    retriever = KnowledgeRetriever(
        data_path=_settings.knowledge_base_path, threshold=_settings.similarity_threshold
    )
    result = retriever.best_match_index("what is the minimum wage in south africa")
    assert result is not None
    _, score = result
    assert score > 0.4, "the documented lexical-overlap false positive no longer reproduces — update the docs"


@pytest.mark.skipif(not _eval_csv.exists(), reason="eval/rag_eval.csv not present")
def test_sweep_covers_a_range_and_returns_sorted_thresholds():
    retriever = KnowledgeRetriever(
        data_path=_settings.knowledge_base_path, threshold=_settings.similarity_threshold
    )
    cases = load_rag_eval_cases(_eval_csv)
    results = sweep_thresholds(retriever, cases, thresholds=[0.15, 0.25, 0.35])
    assert [r.threshold for r in results] == [0.15, 0.25, 0.35]


def test_wrong_row_counts_as_false_positive_not_true_positive():
    """A confidently wrong answer must never score as a success — this is
    the distinction that makes the precision number meaningful rather than
    flattering.
    """
    retriever = KnowledgeRetriever(
        data_path=_settings.knowledge_base_path, threshold=_settings.similarity_threshold
    )
    # Deliberately mislabel the expected row so any match is the "wrong" one.
    cases = [RagEvalCase(query="what is the capital of south africa", expected_row=999, category="exact")]
    result = evaluate_threshold(retriever, cases, threshold=0.15)
    assert result.true_positives == 0
    assert result.false_positives == 1

@pytest.mark.skipif(not _eval_csv.exists(), reason="eval/rag_eval.csv not present")
def test_shipped_pipeline_trades_paraphrase_recall_for_precision():
    """Pins what the DEPLOYED pipeline does — similarity plus the coverage
    gate — as opposed to similarity alone.

    This test exists because the two were conflated. The per-category
    table was computed without the gate while the precision/recall/F1
    headline was computed with it, so the documented weakness of the
    retrieval layer was the wrong one: negative rejection looked like a
    coin flip when the shipped system actually rejects 37/40, and
    paraphrase recall looked healthy when the shipped system only manages
    4/15.

    The real finding, pinned here: the coverage gate buys a large
    reduction in false positives and pays for it with paraphrase recall.
    Paraphrase is therefore the retrieval layer's actual weak point, and
    the case for semantic embeddings rests on closing THAT gap without
    giving back the negative rejection the gate won.
    """
    retriever = KnowledgeRetriever(
        data_path=_settings.knowledge_base_path, threshold=_settings.similarity_threshold
    )
    cases = load_rag_eval_cases(_eval_csv)
    gated = per_category_breakdown(retriever, cases, _settings.similarity_threshold)
    ungated = per_category_breakdown(
        retriever, cases, _settings.similarity_threshold, apply_coverage_gate=False
    )

    # Exact matches are unaffected by the gate — they cover their own query.
    assert gated["exact"][0] == gated["exact"][1]

    # The gate's win: negative rejection improves sharply.
    assert gated["negative"][0] > ungated["negative"][0]
    assert gated["negative"][0] / gated["negative"][1] >= 0.85

    # The gate's cost: paraphrase recall degrades, and remains the weak
    # point. The bound was raised from 0.4 to 0.55 when Alt_Questions was
    # added to the knowledge base: alternative phrasings lifted gated
    # paraphrase recall from 4/15 to 7/15 with negatives unchanged at
    # 37/40. That is a real improvement in the system, not a loosened
    # test, and the README numbers were re-measured with it.
    assert gated["paraphrase"][0] < ungated["paraphrase"][0]
    assert gated["paraphrase"][0] / gated["paraphrase"][1] <= 0.55
