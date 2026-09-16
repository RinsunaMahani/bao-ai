# Bao AI — Technical Report

*Prepared for presentation, 30 August 2026.*

## 1. Overview

Bao AI ("Bao," short for baobab) is an offline-first multilingual
assistant for South Africa's 11 spoken official languages — of 12
official languages in total, South African Sign Language being the 12th,
and outside Bao's scope. The core design
principle: verified local knowledge and generative AI are treated as two
distinct answer sources, not blended into one system. A question is
answered from a curated, human-written knowledge base when possible; only
questions with no local match are passed to Gemini for a generated
response — and even that path degrades to a clear offline message when no
network connection is available at all.

## 2. Architecture

```
User
  |
  v
Security guardrails        (prompt-injection / input screening)
  |
  v
Language detection         (on-device TFLite LSTM, default;
  |                          heuristic keyword matching as fallback)
  v
Knowledge retrieval    ----+---- verified curated facts (TF-IDF)
  |                        +---- uploaded-document context (TF-IDF, session-scoped)
  v
      match?  -----yes---->  Verified answer (translated if needed)
        |
        no
        v
Gemini generation (if reachable) --------> Generated answer, grounded in
        |                                   any uploaded-document context
        no (offline / unreachable)
        v
Offline fallback: matched document excerpt, or a clear "no answer" message
  |
  v
Conversation memory update
  |
  v
Speech output (optional, if installed)
  |
  v
Response
```

Every stage is an independently testable module (see Section 5). The two
knowledge-retrieval branches are deliberately separate, not merged: a
curated-knowledge-base match is treated as verified and returned exactly
(or translated), never rephrased by an LLM; document-upload matches are
treated as unverified context fed *into* a Gemini prompt. This is the
project's central architectural claim: Bao doesn't treat generated
information and verified local knowledge as the same thing.

## 3. Methodology

### 3.1 Retrieval

TF-IDF vectorization (scikit-learn) over a curated CSV of question/answer
pairs, matched by cosine similarity against a configurable threshold
(default 0.25) plus a query-coverage gate (default 0.70; see 3.1.1).
Chosen over a vector database or embedding model for two
reasons: it requires no network call or GPU, and at the current knowledge
base size (38 rows), the added complexity of an embedding pipeline isn't
justified by the data volume.

### 3.2 Language detection

Two interchangeable implementations behind one interface:

- **On-device ML (default):** an LSTM classifier trained on the NCHLT
  South African text corpus, deployed via TFLite. See Section 4.4 for the
  evaluation and a real deployment-bug story worth reading before citing
  the underlying model's training-time accuracy figure as this
  deployment's own.
- **Heuristic (automatic fallback):** keyword-marker matching across all
  11 languages, zero model-loading cost, used automatically if the ML
  model or its tokenizer files are missing.

### 3.3 Generation

Google's Gemini API, used only when a query has no verified local answer.
The model in use is centrally configured (`config.toml` /
`bao/core/config.py`) rather than hardcoded per call site.

### 3.4 Optional modalities

Text-to-speech output (MMS-TTS, per-language voice models) and a static,
single-frame sign-language hand-shape classifier (MediaPipe hand-landmark
extraction plus a small scikit-learn classifier trained on
self-collected data) are both implemented as genuinely optional
capabilities: absent dependencies or missing trained assets degrade the
specific feature, never the core application.

## 4. Evaluation

### 4.1 Retrieval self-consistency

`python evaluate.py` queries the knowledge base with each entry's own
question and checks whether that entry is the top match. Current result:
100% (38/38). This measures whether the retrieval mechanism works at
all — not whether it generalizes to real user phrasing — and is presented
here with that caveat, not as a headline accuracy claim.

### 4.2 Language detection

`python benchmark_bao.py` runs the active detector against a small,
hand-labelled query set spanning 4 languages. Current result: 80%
(20/25), with the one miss being a Sepedi greeting ("Dumela...")
detected as Sesotho — a specific, known keyword-overlap case (both
languages share the marker word "dumela"), not a random failure.

### 4.2b Disambiguating closely-related languages

Sepedi, Sesotho and Setswana form the most closely related group in the
label set and share the greeting "Dumela", making them the natural stress
test for language detection. Evaluated with three sentences per language,
including all three "Dumela" greetings, the LSTM classifier scored 9/9
with 0.993 confidence on the ambiguous greetings, while the keyword
heuristic collapses them (it returns Sesotho for all three).

This isolates the failure as a limitation of lexical keyword matching
rather than of the trained model, and is the empirical justification for
the ML classifier being the production default.

### 4.3 Retrieval precision and recall: a measured limitation

`python evaluate.py --rag-eval eval/rag_eval.csv` scores retrieval against
a 70-query hand-labelled set (15 verbatim knowledge-base questions, 15
paraphrases of them, 40 topically unrelated queries — including lexical
traps that share vocabulary with knowledge-base entries, and
South-Africa-related questions absent from the knowledge base) and sweeps the
similarity threshold from 0.05 to 0.60. This exists because the
self-consistency result in 4.1 only proves the knowledge base can find its
own rows; it cannot detect the failure mode that matters in use — an
unrelated query clearing the threshold and receiving a confidently wrong
answer.

| Threshold | Exact | Paraphrase | Negative (correctly rejected) |
|---|---|---|---|
| 0.15 | 15/15 | 12/15 | 2/40 |
| **0.25 (current default)** | **15/15** | **11/15** | **8/40** |
| 0.45 | 15/15 | 3/15 | 22/40 |
| 0.55 | 13/15 | 0/15 | 35/40 |

The result is a genuine, measured limitation rather than a tuning
oversight: **no threshold performs acceptably on both paraphrases and
unrelated queries.** Raising it to reject out-of-scope questions destroys
paraphrase recall; lowering it to catch paraphrases means nearly every
out-of-scope question receives an incorrect answer. The default was moved
from 0.15 to 0.25 (the best-F1 point, F1 = 0.765) on this evidence, but
that is a compromise, not a fix.

The underlying cause is that TF-IDF measures lexical overlap, not meaning
— a query sharing common words with a knowledge base entry scores highly
regardless of topic. Concrete instances, all matching against a fact about South Africa's
*currency*: "who is the current president of south africa" (0.765), "what
is the history of the roman empire" (0.543), and — most starkly — "what
is the speed of light" (0.532), which shares nothing with the entry but
the function words "what is the of". This is the empirical case for replacing TF-IDF with a
multilingual sentence-embedding model, and the reason
`knowledge/embeddings.py` was built as a swappable interface from the
start — that upgrade is a new class implementing the same interface, not
a rewrite of the retrieval layer.

### 4.4 The on-device ML language classifier

The model is an LSTM (embedding dim 128, 100 LSTM units, 3,292,711
trainable parameters — confirmed by direct inspection of the deployed
`.keras` file) trained on 636,123 NCHLT-corpus sentences across all 11
languages, reporting a 0.98 Macro F1 on a 63,613-sentence held-out test
set during training (documented in the project's original training
presentation, `docs/training/nchlt_training_presentation.pptx`;
independently corroborated here by an exact match on every checkable
architecture detail, including the parameter count to the digit).

Two deployment bugs were found and fixed before this model could be
trusted in the running application, neither of which crashed anything —
both produced confidently wrong output instead:

1. **Preprocessing mismatch:** the shipped code encoded text as raw
   character codes; the model's own `tokenizer_config.json` proves it was
   trained on a 25,000-word, word-level vocabulary. Fixed by reconstructing
   the real tokenization pipeline from the vocabulary file.
2. **Class-order mismatch:** even after fixing preprocessing, the code
   mapped the model's output index back to a language name using an
   ordering chosen for the UI, not the model's actual (alphabetical)
   training-time class order. Found by feeding one clearly-monolingual
   sentence per language through the model and recording which output
   index fired for each — every language landed on a distinct index at
   97-100% confidence once the correct order was used.

With both fixed, the deployed `.tflite` file was tested directly (not
just the Keras source it was converted from) against three independent
sets: 11/11 on formal single-language sentences, 4/4 on short informal
greetings — a different register from NCHLT's training text and closer to
real chat input — and 5/5 on this project's own pre-existing benchmark
query set (versus the heuristic detector's 4/5 on the same set).

This 20/20 result is real, reproducible evidence, but it is not the same
claim as "0.98 Macro F1 confirmed for this deployment" — the original
63,613-sentence held-out test set isn't available to re-run here. The
model is the default detector in the running application
(`prefer_ml=True`), with the heuristic as an automatic fallback if the
model or tokenizer files are ever missing, on the strength of the 20/20
evidence and the independently-confirmed architecture match — not on the
strength of the unreproduced training-time number alone.

## 5. Testing

37 unit and integration tests (`pytest tests/ -v`), covering security
input validation, both language-detection backends, knowledge retrieval
(including the TF-IDF limitation above, pinned as a regression test),
conversation memory, sign-language feature extraction, and the full
request pipeline end to end — including a mocked test of the
Gemini-generation path, since no live API dependency is required to run
the suite. The full pipeline was also driven through real, non-mocked
Streamlit sessions (`streamlit.testing.v1.AppTest`) for manual-equivalent
test cases: an English knowledge-base question, an isiZulu question, an
out-of-scope question with the system offline, and a document-upload
question — all verified to behave correctly with no exceptions.

## 6. Known limitations

- **Small knowledge base** (38 entries): appropriate for a prototype
  demonstrating an architecture, not a claim of comprehensive coverage.
  Several entries are intentionally campus-specific.
- **TF-IDF false positives** on topically-unrelated queries (Section 4.3)
  — architected around, not yet fixed.
- **No query resolution for follow-ups against the knowledge base**:
  conversation memory feeds Gemini's prompts, but not the retrieval query
  itself, so a follow-up like "what about its opening hours?" only
  resolves correctly when it reaches Gemini, not when answered from the
  local knowledge base.
- **KB-fact translation requires a Gemini API call**: a verified English
  fact translated into another language still costs one API round trip,
  a real reliability dependency for offline-first claims about that
  specific path.
- **Speech coverage is partial**: four of the eleven languages have a
  voice (English, Afrikaans and isiZulu via edge-tts; Xitsonga via Meta
  MMS). The other seven have no open text-to-speech model — verified
  against Hugging Face with an authenticated request on 2026-08-30, where
  `mms-tts-xho`, `-sot`, `-tsn`, `-nso`, `-ven`, `-ssw` and `-nbl` all
  return 404. Replies in those languages are shown as text.
- **The 0.98 macro F1 is the original training-corpus evaluation**, not a
  figure this project independently reproduced. What was independently
  validated is the *deployed inference path*: 20/20 across three small
  probe sets (11 formal sentences, 4 greetings, 5 benchmark queries),
  which confirms the TFLite path and tokenizer work end to end but is not
  equivalent to re-running the 63,613-sentence held-out evaluation.

## 7. Future work

- Real sentence-embedding retrieval to close the TF-IDF false-positive
  gap, behind the existing swappable `knowledge/embeddings.py` interface.
- A properly labelled, multi-language evaluation dataset (currently a
  small template) to validate the on-device ML language classifier before
  considering it for production use.
- Query resolution against conversation memory before knowledge-base
  retrieval, not just before Gemini generation.
- Continuous/dynamic sign-language recognition, requiring sequence
  modeling rather than single-frame classification — a materially larger
  project than the current static prototype.
