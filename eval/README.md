# Building a Language Detector Evaluation Set

`evaluate.py --detector-eval eval/language_eval_template.csv` scores the
active `LanguageDetector` (heuristic by default, or the TFLite backend with
`--use-ml-detector`) against a labelled CSV of `text,label` pairs.

## Why this needs real data (unlike the retrieval evaluation)

`evaluate_retrieval_self_consistency()` generates its own test cases from
the knowledge base (asking each row its own question), so it needs no
external labels. Language detection accuracy can't be measured that way —
there's no self-consistent proxy for "is this the correct language,"
so it needs genuinely labelled examples.

## Building a trustworthy set

- **Cover all 11 languages**, not just the ones you speak. A detector that
  looks accurate because 8 of 11 languages went untested isn't accurate.
- **Include short, ambiguous input.** Greetings and single words are the
  hardest case for both the heuristic (few keyword matches to key off) and
  the ML backend (little signal in 35 characters) — and the most common
  real input in a chat UI.
- **Include code-switching**, common in South African speech (a sentence
  mixing English and isiZulu, for example) — decide up front what the
  "correct" label is for a mixed-language sentence and be consistent.
- **Don't reuse text the model may have already seen.** If you're
  evaluating the TFLite classifier, avoid pulling examples from whatever
  corpus trained it — that measures memorization, not generalization.

## Format

```csv
text,label
Sawubona,isiZulu
Dumela,Sesotho
How do I reset my password?,English
```

Labels must exactly match the strings in `bao/core/config.py::LABELS`.

`language_eval_template.csv` is intentionally not checked in with real
data — replace it with your own labelled examples before running
`--detector-eval` against it.
