"""Audit a training corpus before training anything on it.

    python scripts/audit_corpus.py --corpus path/to/data.csv

Reads a CSV with columns `text,language` and reports whether the data looks
like real language or like generated filler.

Why this exists, concretely. A 16-language classifier was submitted to this
project earlier. Its artifacts showed:

  - 2,400 documents across 16 classes — exactly 150 each;
  - each language's NAME appearing exactly 150 times, i.e. once in every
    one of its own training sentences;
  - a total vocabulary of 592 words for sixteen languages;
  - training examples of the form
    "sample text sentence for language classification in swahili sequence 42".

That model reported near-perfect validation accuracy and recognised ZERO
words of a real Swahili, Yoruba, Hausa or French sentence. It had not
learned to identify languages; it had learned to read the answer out of the
input.

**The dangerous part is what that does to a combined corpus.** Merge real
data with data like this and the fake classes become trivially separable,
so they score near 1.00 and pull macro F1 *up*. The combined model looks
better than the real one while being worse. Nothing warns you.

So: run this on any corpus before it is trained on, and run it on each
language source separately before merging. It cannot prove data is genuine.
It reliably catches the specific ways generated data gives itself away.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Thresholds are deliberately loose. The goal is to flag corpora worth
# looking at by eye, not to automate the judgement.
MIN_VOCAB_PER_CLASS = 500        # real prose in one language exceeds this easily
MAX_LABEL_LEAK_RATIO = 0.20      # class name appearing in >20% of its own rows
MAX_TEMPLATE_RATIO = 0.30        # share of rows sharing a 5-word opening
MIN_TYPE_TOKEN_RATIO = 0.02      # vocabulary size / total tokens


def load(path: Path) -> list[tuple[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise SystemExit("Corpus needs columns: text,language")
        label_field = "language" if "language" in reader.fieldnames else "label"
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get(label_field) or "").strip()
            if text and label:
                rows.append((text, label))
    return rows


def audit(rows: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Returns (severity, check, detail). Severity is OK / WARN / FAIL."""
    findings: list[tuple[str, str, str]] = []

    by_class: dict[str, list[str]] = defaultdict(list)
    for text, label in rows:
        by_class[label].append(text)

    all_tokens: list[str] = []
    class_vocab: dict[str, set[str]] = {}
    for label, texts in by_class.items():
        tokens = [w.lower() for t in texts for w in WORD.findall(t)]
        all_tokens.extend(tokens)
        class_vocab[label] = set(tokens)

    total_vocab = len(set(all_tokens))
    findings.append((
        "OK", "size",
        f"{len(rows)} rows, {len(by_class)} classes, {total_vocab} unique words",
    ))

    # 1. Label leakage — the signature that matters most. A language's own
    #    name appearing in its own training text means the model can read
    #    the answer off the input.
    leaks = []
    for label, texts in by_class.items():
        needle = label.lower()
        hits = sum(1 for t in texts if needle in t.lower())
        ratio = hits / len(texts) if texts else 0.0
        if ratio > MAX_LABEL_LEAK_RATIO:
            leaks.append((label, hits, len(texts), ratio))
    if leaks:
        worst = ", ".join(f"{label} {hits}/{n} ({ratio:.0%})" for label, hits, n, ratio in leaks[:5])
        findings.append((
            "FAIL", "label leakage",
            f"{len(leaks)} class(es) contain their own name in their own rows: {worst}. "
            "The model can classify by reading the label out of the input.",
        ))
    else:
        findings.append(("OK", "label leakage", "no class name appears in its own rows at scale"))

    # 2. Vocabulary per class, judged RELATIVE to how much text there is.
    #    An absolute floor would unfairly fail a small but genuine corpus —
    #    found by testing this script against a deliberately small control.
    #    What matters is a class with plenty of text and almost no variety,
    #    which is what templating produces.
    thin = {}
    for label, texts in by_class.items():
        vocab = len(class_vocab[label])
        if len(texts) >= 500 and vocab < MIN_VOCAB_PER_CLASS:
            thin[label] = (vocab, len(texts))
    if thin:
        listed = ", ".join(
            f"{label} ({v} words in {n} rows)" for label, (v, n) in sorted(thin.items())[:5]
        )
        findings.append((
            "WARN", "vocabulary size",
            f"{len(thin)} class(es) have lots of text but little variety: {listed}",
        ))
    else:
        smallest = min((len(v), k) for k, v in class_vocab.items())
        findings.append((
            "OK", "vocabulary size",
            f"smallest class vocabulary: {smallest[1]} ({smallest[0]} distinct words)",
        ))

    # 3. Cross-class vocabulary sharing. Generated data reuses one English
    #    scaffold across every language, so unrelated languages share far
    #    more vocabulary than they should.
    labels = list(class_vocab)
    if len(labels) >= 2:
        overlaps = []
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                smaller = min(len(class_vocab[a]), len(class_vocab[b])) or 1
                shared = len(class_vocab[a] & class_vocab[b]) / smaller
                overlaps.append((shared, a, b))
        overlaps.sort(reverse=True)
        worst_share, a, b = overlaps[0]
        median_share = sorted(s for s, _, _ in overlaps)[len(overlaps) // 2]
        if median_share > 0.5:
            findings.append((
                "FAIL", "cross-class overlap",
                f"unrelated languages share {median_share:.0%} of vocabulary (median). "
                "Real distinct languages do not — this is one scaffold reused.",
            ))
        else:
            findings.append((
                "OK", "cross-class overlap",
                f"median {median_share:.0%}, highest {worst_share:.0%} ({a}/{b})",
            ))

    # 4. Templating — many rows sharing an identical opening.
    openings = Counter(" ".join(t.lower().split()[:5]) for t, _ in rows)
    if openings:
        common, count = openings.most_common(1)[0]
        ratio = count / len(rows)
        if ratio > MAX_TEMPLATE_RATIO:
            findings.append((
                "FAIL", "templating",
                f"{ratio:.0%} of rows begin with the same five words: {common!r}",
            ))
        else:
            findings.append(("OK", "templating", f"most common opening covers {ratio:.1%} of rows"))

    # 5. Type-token ratio — vocabulary richness overall.
    ttr = total_vocab / len(all_tokens) if all_tokens else 0.0
    if ttr < MIN_TYPE_TOKEN_RATIO:
        findings.append((
            "WARN", "lexical richness",
            f"type-token ratio {ttr:.4f} — very repetitive for natural text",
        ))
    else:
        findings.append(("OK", "lexical richness", f"type-token ratio {ttr:.4f}"))

    # 6. Class balance, reported rather than judged. Exactly equal counts
    #    are a mild tell for generated data; NCHLT is naturally uneven.
    counts = {label: len(texts) for label, texts in by_class.items()}
    if len(set(counts.values())) == 1 and len(counts) > 2:
        findings.append((
            "WARN", "class balance",
            f"every class has exactly {next(iter(counts.values()))} rows — "
            "real corpora are rarely this even",
        ))
    else:
        smallest, largest = min(counts.values()), max(counts.values())
        findings.append((
            "OK", "class balance",
            f"{smallest}-{largest} rows per class (ratio {largest / max(smallest, 1):.1f}x)",
        ))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="CSV with columns text,language")
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"No such file: {args.corpus}", file=sys.stderr)
        return 2

    rows = load(args.corpus)
    if not rows:
        print("Corpus is empty.", file=sys.stderr)
        return 2

    findings = audit(rows)

    print(f"\nCORPUS AUDIT — {args.corpus}")
    print("=" * 74)
    for severity, check, detail in findings:
        marker = {"OK": "  ok  ", "WARN": " warn ", "FAIL": " FAIL "}[severity]
        print(f"[{marker}] {check:<22} {detail}")
    print("=" * 74)

    # FAIL is reserved for signals that generated data essentially cannot
    # avoid producing: the label inside its own rows, one scaffold reused
    # across unrelated languages, and mass-identical openings. Vocabulary
    # richness and class balance are suggestive, not decisive, so they warn.
    failures = [f for f in findings if f[0] == "FAIL"]
    warnings = [f for f in findings if f[0] == "WARN"]

    if failures:
        print(f"\n  {len(failures)} check(s) failed. This corpus shows the signature of")
        print("  generated filler rather than real language data.")
        print("\n  Do NOT merge it with real data. A model trained on the mixture will")
        print("  score HIGHER on macro F1 than one trained on the real data alone,")
        print("  because the fake classes are trivially separable — so the number")
        print("  improves while the system gets worse, and nothing warns you.\n")
        return 1
    if warnings:
        print(f"\n  {len(warnings)} warning(s). Worth a look by eye before training.\n")
        return 0

    print("\n  No signature of generated data. This does not prove the corpus is")
    print("  good — only that it does not fail these specific checks.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
