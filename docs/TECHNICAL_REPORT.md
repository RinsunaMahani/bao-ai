# Bao AI — Technical Report

*Prepared for presentation, 30 August 2026; figures updated 29 September 2026.*

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
- **Keyword matcher (paired with the classifier):** marker-word matching
  across all 11 languages, with zero model-loading cost. It stands in
  entirely if the model or its tokenizer files are missing, and it also
  answers, at runtime, the input the classifier cannot read. The
  classifier's vocabulary comes from NCHLT news sentences, so a lone
  greeting such as "Sawubona" or "Avuxeni" is wholly out of vocabulary to
  it and it returns the same constant prior for all of them — measured,
  nine greetings across eight languages all came back as siSwati at
  exactly 35%. The keyword list contains precisely those words. It is
  consulted only when the classifier is unsure and only when it actually
  recognised a word, and it can never overrule a confident classification.
  Single-word greetings went from 6/15 to 14/15 answered in the right
  language; the remaining miss, "Dumela", is the greeting in Sepedi,
  Sesotho and Setswana alike and cannot be resolved from the word alone.
- **Pan-African extension (optional):** a character n-gram classifier for
  14 further languages (Amharic, French, Hausa, Igbo, Lingala, Luganda,
  Oromo, Nigerian Pidgin, Kirundi, Shona, Somali, Swahili, Tigrinya,
  Yoruba). The two models are not calibrated against each other, so their
  confidences are never compared directly; the pan-African model's own
  confidence decides, with a threshold taken from measurement — it never
  exceeded 44% on 24 South African inputs, and was correct on all 14
  pan-African sentences tested at 43–89%. Walking one sentence per language
  through the whole pipeline (`scripts/probe_language_routing.py`), 24 of
  25 are answered in the language they were written in. Luganda is the
  miss: identified correctly but at 43%, just under the 44% English
  reaches as Nigerian Pidgin, and admitting it would mean relabelling
  English.

### 3.3 Generation

Google's Gemini API, used only when a query has no verified local answer.
The model in use is centrally configured (`config.toml` /
`bao/core/config.py`) rather than hardcoded per call site.

### 3.4 Optional modalities

Text-to-speech output is implemented as a genuinely optional capability:
absent dependencies or missing trained assets degrade the specific
feature, never the core application. Three backends are selected per
language, each only where it is the best available: Microsoft's neural
voices (online; real en-ZA, af-ZA and zu-ZA locales), Meta's MMS-TTS
(offline; Xitsonga, and twelve of the pan-African languages), and a
multilingual South African VITS model covering all eleven, enabled in this
deployment for the seven that have no other voice. A voice is chosen from
the language the reply is SPOKEN in — its opening — rather than from every
language that appears in it, because replies routinely quote lyrics or
terms in another language. Where no voice exists the interface says so
rather than substituting one silently. Replies are read in full: for a
user who cannot easily read the screen the voice is the answer, not a
flourish on it.

A static, single-frame sign-language hand-shape classifier (MediaPipe
hand landmarks feeding a small scikit-learn classifier) was prototyped
and then **removed**; it was never trained, and static hand poses cannot
represent a language whose vocabulary is movement and whose grammar is
carried partly by non-manual markers.

South African Sign Language is therefore not supported, and the reason
was checked rather than assumed. No public SASL dataset exists: the field's
own dataset catalogue lists none, against roughly 8–10 for ASL. The only
SASL video corpus found — 5,047 sentences, about five hours, from a 2024
University of Cape Town master's thesis — is not publicly released, and
the thesis's own sign-to-text result on it is BLEU-4 1.35, described by
its author as "very far from practical" (the same approach scores 13.23 on
a German benchmark). The leading open-source text↔sign system, sign.mt,
works reasonably well for American, German and Brazilian sign languages
only. The blocker is data and Deaf-community partnership, not tooling:
MediaPipe installs and runs on this project's Python. Sources and the full
reasoning are in "Accessibility and SASL" in `docs/ARCHITECTURE.md`.

## 4. Evaluation

### 4.1 Retrieval self-consistency

`python evaluate.py` queries the knowledge base with each entry's own
question and checks whether that entry is the top match. Current result:
100% (38/38). This measures whether the retrieval mechanism works at
all — not whether it generalizes to real user phrasing — and is presented
here with that caveat, not as a headline accuracy claim.

### 4.2 Language detection

`python benchmark_bao.py` runs both detectors against a small,
hand-labelled set of 5 queries spanning 4 languages (each run 5 times for
latency, so 25 inference runs but 5 independent examples). Current result:
the LSTM classifier 5/5 at 97.8% average confidence; the keyword matcher
4/5, its miss being a Sepedi greeting ("Dumela...") read as Sesotho — a
specific, known keyword-overlap case (the languages share the marker word
"dumela"), not a random failure.

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

| System | Exact | Paraphrase | Negative (correctly rejected) |
|---|---|---|---|
| Similarity alone, threshold 0.25 | 15/15 | 12/15 | 20/40 |
| **Shipped: similarity 0.25 and coverage 0.70** | **15/15** | **7/15** | **37/40** |

Shipped: precision 0.880, recall 0.733, F1 0.800.

TF-IDF measures word overlap, not meaning, and no similarity threshold
alone separates paraphrases from unrelated questions. Two fixes closed
most of the false positives: a curated stop-word list (a question matching
only on "what is the of" now scores 0.000) and the IDF-weighted coverage
gate (a question whose informative words the knowledge base does not know,
such as "what is the minimum wage in south africa", is no longer served as
a verified answer). The cost is paraphrase recall, 7/15: rephrasings that
share few words with the stored question fall through to generation.

Sweeping both gates together (24 combinations) shows the tuning is
exhausted. A semantic embedding model
(`paraphrase-multilingual-MiniLM-L12-v2`) was measured against the same 70
questions: it finds more paraphrases (12/15 at its best threshold) but
answers more unrelated questions as verified (34/40 rejected), and its
training languages include none of South Africa's other ten. It was not
adopted; the next step is an embedding model trained on South African
languages, measured on held-out data. `knowledge/embeddings.py` was built
as a swappable interface for exactly that change.

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
(`prefer_ml=True`), paired with the keyword matcher for input it cannot
read and replaced by it if the model or tokenizer files are ever missing
(Section 3.2), on the strength of the 20/20
evidence and the independently-confirmed architecture match — not on the
strength of the unreproduced training-time number alone.

## 5. Testing

449 automated tests in 20 files (`pytest tests/ -v`), covering security
and prompt injection, all three language detectors, knowledge retrieval
and its documented trade-offs, conversation memory, speech routing, the
Gemini client's retries and limits, per-session isolation, the Docker
build context, and the full request pipeline end to end, offline. CI runs
them on Python 3.11 and 3.14 on every change. The full pipeline was also driven through real, non-mocked
Streamlit sessions (`streamlit.testing.v1.AppTest`) for manual-equivalent
test cases: an English knowledge-base question, an isiZulu question, an
out-of-scope question with the system offline, and a document-upload
question — all verified to behave correctly with no exceptions.

## 6. Known limitations

- **Small knowledge base** (38 entries): appropriate for a prototype
  demonstrating an architecture, not a claim of comprehensive coverage.
  Several entries are intentionally campus-specific.
- **Paraphrase recall** (7/15, Section 4.3) is the retrieval weakness;
  unrelated questions are rejected 37 times out of 40.
- **No query resolution for follow-ups against the knowledge base**:
  conversation memory feeds Gemini's prompts, but not the retrieval query
  itself, so a follow-up like "what about its opening hours?" only
  resolves correctly when it reaches Gemini, not when answered from the
  local knowledge base.
- **KB-fact translation requires a Gemini API call**: a verified English
  fact translated into another language still costs one API round trip,
  a real reliability dependency for offline-first claims about that
  specific path.
- **Speech coverage depends on one configuration switch**: four of the
  eleven languages have a voice with the shipped code defaults (English, Afrikaans and
  isiZulu via edge-tts; Xitsonga via Meta MMS). No pretrained MMS voice
  exists for the other seven — verified against Hugging Face with an
  authenticated request on 2026-08-30, where `mms-tts-xho`, `-sot`,
  `-tsn`, `-nso`, `-ven`, `-ssw` and `-nbl` all return 404. A multilingual
  South African VITS model covers all eleven and is enabled in this
  deployment, taking native coverage to 11 of 11 South African and 23 of
  25 detectable languages. It is off in the code defaults because it is
  licensed cc-by-nc-4.0 while this project is MIT, so enabling it makes a
  deployment non-commercial, and because its model card carries no
  evaluation this project could check — its audio quality is therefore
  reported as unverified rather than measured.
- **Alternatives were ruled out by measurement, not assumption**: Google
  Translate's TTS endpoint returns HTTP 400 for all nine indigenous South
  African languages, so it can translate them but not speak them;
  Microsoft publishes 322 voices of which 11 are relevant locales, which
  is the three already used; and the one apache-2.0 alternative covering
  part of the gap is a 6.63 GB 3B-parameter model, the wrong shape for an
  offline-first system on modest hardware.
- **The 0.98 macro F1 is the original training-corpus evaluation**, not a
  figure this project independently reproduced. What was independently
  validated is the *deployed inference path*: 20/20 across three small
  probe sets (11 formal sentences, 4 greetings, 5 benchmark queries),
  which confirms the TFLite path and tokenizer work end to end but is not
  equivalent to re-running the 63,613-sentence held-out evaluation.

## 7. Future work

- An embedding model trained on South African languages, behind the
  existing swappable `knowledge/embeddings.py` interface, measured on a
  held-out set (a general multilingual model was measured and not adopted).
- A properly labelled, multi-language evaluation dataset (currently a
  small template) to measure the classifier beyond the current probe sets.
- Query resolution against conversation memory before knowledge-base
  retrieval, not just before Gemini generation.
- South African Sign Language. Continuous recognition needs pose, hand
  and face tracking with a sequence model, and a public annotated SASL
  corpus that does not yet exist — so the first step is building one with
  the Deaf community, not writing a model. A narrower, feasible piece is
  SASL fingerspelling (the one-handed manual alphabet), but it needs
  self-recorded data from several signers checked letter by letter by a
  SASL signer, and it covers spelling rather than conversation.
