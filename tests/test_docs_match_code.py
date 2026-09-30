"""Documentation that states a number must state the right one.

Adding six greeting rows left `32 rows` in the README and in the technical
report while the knowledge base held 38. Nothing failed — the docs simply
became quietly wrong, which in a graded submission is worse than a crash,
because it gets read and believed.

These are cheap guards on the few figures that actually appear in prose.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from bao.core.config import LABELS, Settings

REPO = Path(__file__).resolve().parent.parent
DOCS = [REPO / "README.md", *(REPO / "docs").glob("*.md")]

# Every read below passes encoding="utf-8" explicitly. Path.read_text()
# defaults to the platform encoding, which is cp1252 on Windows, so these
# checks crashed with UnicodeDecodeError the moment a document quoted the
# orthography it is documenting — Tshivenda's ṱ ḓ ṋ ḽ or Sepedi's š. A
# guard on a multilingual project's docs cannot assume Latin-1.


@pytest.fixture(scope="module")
def knowledge_rows():
    """Both counts are legitimate figures for the docs to quote.

    The knowledge base carries draft rows flagged `needs-review`, and a
    draft serves only if its numbers check out (see KnowledgeRetriever).
    So "the size of the knowledge base" is two numbers: what has been
    written (all rows), and what actually serves, counted by the retriever
    itself rather than re-derived here. A document quoting either is
    accurate; a document quoting neither is stale, which is what this
    test is for.
    """
    from bao.knowledge.retriever import KnowledgeRetriever

    path = Settings().knowledge_base_path
    return {len(pd.read_csv(path)), len(KnowledgeRetriever(data_path=path))}


def test_no_document_states_a_stale_row_count(knowledge_rows):
    """Catches "32 rows" / "32 entries" / "32/32" left behind after the
    knowledge base grew.
    """
    # Deliberately narrow. Two earlier versions of this test fired on
    # "2/40" and "20/20" from unrelated sentences — a threshold table and a
    # detector spot-check. A guard that cries wolf gets deleted, so it only
    # looks at the two forms that unambiguously encode the corpus size:
    #   - "38 rows" / "38 entries"
    #   - "38/38", and only on a line that says what it is measuring
    #
    # Widened once, after two stale counts survived the original pattern
    # and shipped in the README:
    #   - "32-entry dataset"        — hyphenated, and singular "entry"
    #   - "rows evaluated: 32"      — the count FOLLOWS the noun
    # Both unambiguously encode the corpus size, so both are now caught.
    # The singular is admitted only when hyphenated ("32-entry"), never
    # bare, so ordinary prose like "each row" cannot trip it.
    explicit = re.compile(
        r"\b(\d+)\s*(?:rows|entries)\b"
        r"|\b(\d+)-(?:row|entry)\b"
        r"|\b(?:rows|entries|records)\s+\w+:?\s*(\d+)\b"
    )
    self_retrieval = re.compile(r"\b(\d+)/(\1)\b")

    def check(doc_name, found, value):
        if 20 <= value <= 200 and value not in knowledge_rows:
            expected = " or ".join(str(v) for v in sorted(knowledge_rows))
            pytest.fail(
                f"{doc_name} says {found!r} but the knowledge base has "
                f"{expected} rows (authored / reviewed-and-served)"
            )

    for doc in DOCS:
        if not doc.exists():
            continue
        for line in doc.read_text(encoding="utf-8").splitlines():
            for match in explicit.finditer(line):
                # One group per alternation branch; exactly one is set.
                digits = next(g for g in match.groups() if g is not None)
                check(doc.name, match.group(0), int(digits))
            if re.search(r"self.retrieval|self.consistency", line, re.I):
                for match in self_retrieval.finditer(line):
                    check(doc.name, match.group(0), int(match.group(1)))


def test_language_count_is_consistent_everywhere():
    assert len(LABELS) == 11
    for doc in DOCS:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        wrong = re.findall(r"\b(\d+)\s+official (?:South African )?languages", text)
        for n in wrong:
            assert int(n) in (11, 12), (
                f"{doc.name} claims {n} official languages; the app covers 11 "
                "spoken languages (12 including SASL)"
            )


def test_every_command_the_readme_gives_actually_exists():
    """A README that tells a marker to run a file that isn't there costs
    documentation marks for no reason.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    scripts = set(re.findall(r"python (\S+\.py)", readme))
    scripts |= set(re.findall(r"streamlit run (\S+\.py)", readme))
    missing = [s for s in scripts if not (REPO / s).exists()]
    assert not missing, f"README references missing files: {missing}"
