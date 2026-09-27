<p align="center">
  <img src="docs/assets/logo.jpg" alt="Bao AI logo — a baobab tree, canopy and root system mirrored" width="220">
</p>

<h1 align="center">Bao AI</h1>

<p align="center">
  <strong>An offline-first, multilingual AI assistant for the 11 spoken official languages of South Africa.</strong>
</p>

<p align="center">
  <img alt="Final Year Project" src="https://img.shields.io/badge/Final%20Year%20Project-2026-8B5A2B?style=for-the-badge">
</p>

<p align="center">
  <a href="https://github.com/RinsunaMahani/bao-ai/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/RinsunaMahani/bao-ai/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%20%7C%203.14-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="TensorFlow Lite" src="https://img.shields.io/badge/On--device%20ML-LiteRT%20%2F%20TFLite-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/Retrieval-scikit--learn-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Gemini" src="https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

> **🎓 Final Year Project.** Bao AI was designed, built and evaluated by
> **Rinsuna Blessing Mahani** as a final-year project for the
> **Bachelor of Science (BSc) in Mathematics** at
> **Sefako Makgatho Health Sciences University**, South Africa.
> It is an academic project: it is not a commercial product and it is
> not maintained as a production service.

## At a glance

| | |
|---|---|
| **Problem** | Most AI assistants work poorly, or not at all, in South Africa's indigenous languages, and they assume an internet connection is always available. |
| **Approach** | An offline retrieval "root system" and on-device language identification, with cloud generation used only when a connection is available. |
| **Languages** | English, Afrikaans, isiZulu, isiXhosa, Sepedi, Sesotho, Setswana, siSwati, Tshivenda, Xitsonga, isiNdebele |
| **ML** | An LSTM language classifier (3.29M parameters) trained on the NCHLT corpus and quantised to TFLite for on-device inference |
| **Retrieval** | Offline TF-IDF RAG over a curated South African knowledge base, gated on both similarity and IDF-weighted coverage |
| **Engineering** | A typed request pipeline, prompt-injection guardrails, conversation memory, speech output, Docker, and CI on every push |
| **Evaluation** | Reproducible scripts for retrieval precision and recall, detector benchmarks, and gate sweeps, with limitations documented rather than hidden |

📄 **Read the write-up:** [Technical report](docs/TECHNICAL_REPORT.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Development review log](REVIEW.md)

---

## About the name

"Bao" is short for **baobab** — the tree, not an acronym. The name is the
design brief, not just a label: a baobab survives on deep, wide roots that
work with or without rain, and a canopy that reaches in many directions
from one trunk. That's the actual architecture here — an offline-first
retrieval "root system" that works with no network connection at all, and
a canopy of the 11 spoken official South African languages growing from
it. (South Africa has 12 official languages: South African Sign Language
was recognised as the 12th in 2023. Bao does not support SASL — see
"Accessibility and SASL" below.)

**For a presentation-facing summary — architecture, methodology,
evaluation results, and known limitations, without development
history — see [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).**
This README covers the same ground at more length, plus setup
instructions.

Bao AI is a South African multilingual assistant built for research,
accessibility, and offline-first knowledge delivery — combining offline
TF-IDF retrieval, on-device language identification (an LSTM classifier,
with a keyword matcher for the short greetings it cannot read), and
Gemini-powered generation across all 11 spoken official South African
languages.

## Why this project stands out

- A real, testable **request pipeline** (`ai/orchestrator.py`), not
  branching logic scattered across the UI layer — see
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full diagram and
  the reasoning behind every major design decision.
- Offline RAG retrieval with local TF-IDF search across a curated South
  African knowledge base — **100% top-1 self-retrieval accuracy** over the
  38 reviewed rows it serves (`python evaluate.py`, reproducible). The
  knowledge base holds 52 rows; the other 14 are translations awaiting a
  first-language speaker's review and are deliberately not served until
  then.
- A pluggable language detector: an LSTM classifier (trained on the NCHLT
  South African corpus, 3.29M parameters) as the default, paired with a
  zero-dependency keyword matcher — behind the exact same interface
  (`services/language_detector.py`). The two fail on opposite inputs: the
  classifier's vocabulary is news sentences, so a lone greeting like
  "Sawubona" is entirely out of vocabulary to it, while the keyword list
  knows exactly those words. Pairing them took single-word greetings from
  6/15 to 14/15 answered in the right language. Two real,
  silent deployment bugs (wrong text preprocessing, wrong output-class
  ordering) were found and fixed by testing against the real model, not
  assumed correct because the training results looked good — see
  `docs/ARCHITECTURE.md` and `REVIEW.md` §5 for the full story. Result:
  **5/5 unique benchmark queries correctly classified (100%)** at 97.8%
  average model confidence, versus 4/5 for the heuristic fallback on the
  same queries (`python benchmark_bao.py`, reproducible). Note those 5
  queries are run 5 times each for latency measurement — 25 inference
  runs, but 5 independent language examples, not 25.
- Conversation memory that's actually part of the pipeline, not just chat
  history rendered in the UI — the assistant can resolve follow-ups like
  "and in Afrikaans?" (`ai/memory.py`).
- Multilingual generation via Gemini, with an automatic, tested offline
  fallback when the API is unavailable.
- Security guardrails for prompt sanitization and injection detection —
  raises a typed exception (`SecurityViolationError`) rather than a
  boolean a caller could silently ignore.
- A reproducible evaluation harness (`evaluate.py`) and a real benchmark
  suite (`benchmark_bao.py`) that exercises the actual pipeline, not a
  mocked approximation of it.
- A passing unit and integration test suite (`pytest tests/ -v` — run it
  for the current count; deliberately not hardcoded here, since a number
  written into three documents inevitably drifts out of sync with the
  actual suite), including a
  test that pins a documented, known retrieval limitation (see
  "Evaluation" below) rather than hiding it.
- Containerized deployment and CI that lints and tests every push.

## Architecture

```
User
  │
  ▼
Security guardrails        core/security.py
  ▼
Language detection         services/language_detector.py
  ▼
Knowledge retrieval        knowledge/retriever.py
  ▼
Gemini generation          ai/client.py
  ▼
Translation (if needed)    services/translation.py
  ▼
Conversation memory        ai/memory.py
  ▼
Speech output (optional)   services/speech.py
  ▼
Response
```

Full module responsibilities and the reasoning behind each design decision
live in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Project structure

```
bao/
├── ai/            # client.py (Gemini), orchestrator.py (the pipeline), prompts.py, memory.py
├── knowledge/     # retriever.py, vector_store.py, embeddings.py, loader.py
├── services/      # language_detector.py, translation.py, speech.py, offline.py
├── core/          # config.py, logging.py, security.py, exceptions.py
├── ui/            # streamlit_app.py, console_app.py
└── evaluation.py  # retrieval + language-detector accuracy harness

data/               # african_data.csv — offline knowledge base
models/             # language_classifier.tflite — optional ML language detector
tests/              # pipeline, retrieval, detection, memory, security, RAG evaluation
docs/               # ARCHITECTURE.md, TECHNICAL_REPORT.md
docker/             # Dockerfile, docker-compose.yml
scripts/            # gates.py, scorecard.py, probe_tts.py, compare_detectors.py, audit_corpus.py
evaluate.py         # CLI evaluation runner
benchmark_bao.py    # CLI benchmark runner
bao_console.py      # console entrypoint
```

### Model assets: runtime vs training

| File | Role | Needed to run? |
|---|---|---|
| `models/language_classifier.tflite` | quantised classifier | **yes** |
| `tokenizer_config.json` | vocabulary the model was trained with | **yes** |
| `models/language_detector_pan_african.joblib` | optional 14-language classifier | only if enabled |
| `ultimate_african_ai.keras` | Keras source for `convert_model.py` | no |
| `tokenizer.pickle` | training-time tokenizer | no |

The last two are **training and conversion artifacts**. Nothing in the
running application loads them — `bootstrap.py` passes no
`fallback_keras_path` — and `.dockerignore` excludes them, which is ~69 MB
of the ~96 MB of model weight kept out of any deployment image.

## Getting started

### Prerequisites

- Python 3.11+ (3.10 cannot load the pickled language model faithfully — see `pyproject.toml`)
- Docker Desktop (optional, for containerized deployment)
- A `GEMINI_API_KEY` in a `.env` file for online generation (the app runs
  fully offline without one — see Architecture above)

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Recommended — the app's language detector defaults to this (an LSTM
# classifier, 100% on the benchmark set vs. 80% for the zero-dependency
# fallback that runs automatically without it — see the note above):
python -m pip install -r requirements-ml.txt
# Other optional extras, install what you need:
python -m pip install -r requirements-speech.txt   # TTS/STT
python -m pip install -r requirements-pdf.txt       # PDF document uploads
python -m pip install -r dev-requirements.txt       # pytest, ruff
```

### Check the machine is ready

```bash
python scripts/preflight.py            # no API calls
python scripts/preflight.py --online   # also spends one Gemini request
```

The test suite proves the code is correct on any machine. This proves
*this* machine is ready — that the Git LFS model files actually
downloaded, the voices load, the API key still works. Those are what fail
on the day, and none of them are code bugs, so nothing in the test suite
would catch them. `WARN` means a feature is unavailable and the app will
say so; only `FAIL` blocks a demo.

### Run locally

```bash
streamlit run bao/ui/streamlit_app.py
```

Then open `http://localhost:8501`. For the console/voice demo:

```bash
python bao_console.py
```

### A note on voices

Speech output uses two backends, picked per language in
`services/speech.py::select_backend` and controlled by `[speech] backend`
in `config.toml`:

| Language | Backend | Voice | Offline? |
|---|---|---|---|
| English | edge-tts | `en-ZA-LukeNeural` | no |
| Afrikaans | edge-tts | `af-ZA-WillemNeural` | no |
| isiZulu | edge-tts | `zu-ZA-ThembaNeural` | no |
| The other eight | Meta MMS-TTS | `facebook/mms-tts-*` | yes |

Meta's MMS has no South African English model, so with `backend = "mms"`
(or no network) English is routed through the Afrikaans model. That does
sound local, but it is Afrikaans letter-to-sound rules applied to English
words — an Afrikaans accent, not South African English. `edge-tts` has a
real en-ZA voice, which is why it wins for those three when available. It
is free and needs no API key, but it is an online call, so the
offline-first claim in this README applies to the other eight languages
and to retrieval, not to English/Afrikaans/isiZulu speech.

Both paths fail soft: if the voice can't be produced the turn still
returns its text answer.

### Run with Docker

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

## Testing & linting

```bash
pytest tests/ -v
ruff check .
```

The suite covers the security guardrails, language
detector, knowledge retrieval (including the documented TF-IDF false
positive limitation below), conversation memory, and the full orchestrator
pipeline end to end, offline. CI (`.github/workflows/ci.yml`) runs both
`ruff check .` and the full test suite, plus `evaluate.py`, on Python 3.11
(the supported floor) and 3.14 for every push and pull request.

## Evaluation

```bash
python evaluate.py
python evaluate.py --detector-eval eval/language_eval_template.csv --use-ml-detector --out eval/report.md
python benchmark_bao.py
```

Current, reproducible result on the checked-in `data/african_data.csv`:

```
Knowledge base rows evaluated: 52
Top-1 self-retrieval accuracy: 100.0% (38/38 reviewed rows)
Mean similarity on correct matches: 0.695
```

`benchmark_bao.py` also runs a head-to-head language-detection comparison
between the default LSTM classifier and the heuristic fallback:

```
heuristic : 4/5 (80%)
tflite    : 5/5 (100%)

Unique queries classified correctly : 5/5
  (25 total inference runs = 5 queries x 5 iterations,
   repeated for latency measurement — NOT independent language examples)
Average tflite model score : 97.8%
```

This detector was trained on 636,123 NCHLT-corpus sentences across all 11
languages (reported 0.98 Macro F1 on a 63,613-sentence held-out test set
during training — see `docs/TECHNICAL_REPORT.md`). Two real deployment
bugs — mismatched text preprocessing and a mismatched output-class
ordering, both silent, neither a crash — were found and fixed by testing
against the real model rather than trusting the training numbers applied
automatically to this deployment; see `REVIEW.md` §5 for the full,
evidence-based account of that process.

### Language detection: the closely-related language group

Sepedi, Sesotho and Setswana are the most closely related languages in
the label set, and all three share the greeting "Dumela" — which is
exactly where keyword matching breaks down. Tested with three sentences
per language, including all three "Dumela" greetings:

| Detector | Sotho-Tswana group (9 sentences) |
|---|---|
| Keyword heuristic | fails to separate them — "Dumela" alone is ambiguous |
| **LSTM classifier (default)** | **9/9 correct**, all three "Dumela" greetings correctly separated at 0.993 confidence |

This is the clearest single justification for the ML detector being the
production default rather than an optional extra: it resolves precisely
the case the simpler approach cannot. Pinned by
`tests/test_language_detector.py::test_disambiguates_the_closely_related_sotho_tswana_group`.

### Retrieval precision and recall

`python evaluate.py --rag-eval eval/rag_eval.csv` scores retrieval against
a 70-query labelled set (15 verbatim, 15 paraphrased, 40 topically
unrelated — including deliberate lexical traps and South-Africa-related
questions that aren't in the knowledge base) and sweeps the similarity
threshold from 0.05 to 0.60.

The self-consistency number above only proves the knowledge base can find
its own rows. This measures the thing that actually matters: whether
retrieval can tell a relevant question from an irrelevant one. Current
per-category accuracy:

Retrieval has **two** gates, not one: the similarity threshold, and the
IDF-weighted coverage gate (`min_coverage`, default 0.7) described above.
Numbers below are for both, because they must be read together — an
earlier revision of this table reported similarity alone next to an F1
computed with the gate applied, which described two different systems and
pointed at the wrong weakness.

Similarity alone, i.e. what the coverage gate was introduced to fix:

| Threshold | Exact | Paraphrase | Negative (correctly rejected) |
|---|---|---|---|
| 0.25 | 15/15 | 12/15 | 20/40 |

The shipped pipeline, similarity **and** coverage gate, at the default
threshold 0.25 and coverage 0.7:

| | Exact | Paraphrase | Negative (correctly rejected) |
|---|---|---|---|
| **Shipped** | **15/15** | **7/15** | **37/40** |

Precision 0.880, recall 0.733, **F1 0.800**.

**The honest finding: the coverage gate works, and paraphrase recall is
what it costs.** The gate takes negative rejection from 20/40 to 37/40 —
unrelated queries very rarely receive a `source="knowledge_base"` answer
any more. It pays for that by rejecting genuine rephrasings: 7/15. A user
who asks "what money does south africa use" instead of "what is the
currency of south africa" gets nothing, because the two share almost no
content words and TF-IDF has nothing else to go on.

That trade-off is not a tuning mistake. `python sweep_gates.py` sweeps
both gates together across 24 combinations; F1 varies by only 0.128
across the entire grid and the shipped configuration is already the
global best. The gates trade against each other rather than compounding,
so **no setting of these constants closes the paraphrase gap** — that
requires a different representation. This is the measured,
evidence-based case for semantic embeddings, and it names the specific
number they have to beat: paraphrase 7/15 without giving back negative
37/40.

Note that the coverage gate is a TF-IDF-specific remedy — it exists
because a TF-IDF vectorizer silently drops out-of-vocabulary terms.
Sentence-transformer backends subword-tokenize and do not, so 0.7 should
be re-swept per backend rather than carried over as a constant.
`tests/test_rag_evaluation.py` pins both the similarity-only trade-off
and the shipped behaviour separately, so if either stops holding, the
docs are what need updating.

Concrete examples of the failure mode, all scoring against a fact about
South Africa's *currency*:

| Unrelated query | Similarity |
|---|---|
| "who is the current president of south africa" | 0.765 |
| "what is the population of south africa" | 0.759 |
| "what is the history of the roman empire" | 0.543 |
| "what is the speed of light" | 0.532 |

That last one is the clearest statement of the problem: *"what is the
speed of light"* scores 0.53 against a currency fact purely on the shared
function words "what is the of". No threshold separates that from a
legitimate paraphrase scoring in the same band — which is why this is a
retrieval-method limitation, not a tuning problem.

`python compare_retrievers.py` runs a semantic-embedding backend against
this same evaluation set for a like-for-like comparison (requires
`requirements-semantic.txt`). That experiment has not been run in this
repository yet — the TF-IDF numbers above are measured; the semantic
column is whatever your run produces.

## Developer notes

- Config resolution, model paths, and the Gemini model name all come from
  one place: `core/config.py::Settings`. Nothing else hardcodes them.
- The read-only parts of the pipeline — the language classifier, the
  knowledge base and its TF-IDF index, the Gemini client — are built once
  per server (`st.cache_resource`) and shared. The parts that hold a
  visitor's data — conversation memory and uploaded documents — are built
  per session (`bootstrap.for_session`). `st.cache_resource` is shared
  across *all* visitors, so an earlier version that handed its object
  straight to the page let one visitor's conversation and uploads reach
  the next. See "Sessions and concurrency" in `docs/ARCHITECTURE.md`.
- Speech synthesis and the ML language detector both degrade gracefully —
  missing `torch`/`transformers`/`tensorflow`/`edge-tts` disables one
  feature, never crashes the app at import time. A speech failure (no
  network for the online voices, say) is logged as one warning line and
  the turn still returns its text answer.
- Speech is routed by the language of the **reply**, not the question, so
  a cross-lingual turn ("explain X in Xitsonga", asked in English) is
  spoken with a Xitsonga voice rather than an English one.
- Structured JSON logging (`core/logging.py`) carries live CPU/RAM
  telemetry on every log line.

## Accessibility and SASL

South African Sign Language became the 12th official language on 19 July
2023. **Bao does not support it**, and that was checked rather than
assumed:

- **No public SASL dataset exists.** The field's own dataset catalogue
  lists none, against roughly 8–10 for American Sign Language. A 2014
  University of Cape Town dataset is data-glove sensor readings (it cannot
  train a camera-based system), and a 2024 UCT corpus of 5,047 signed
  sentences was never publicly released.
- **The state of the art is not yet usable.** That 2024 thesis reports
  sign-to-text translation at **BLEU-4 1.35** on its own corpus, "very far
  from practical" in the author's words; the same method scores 13.23 on
  a German benchmark.
- **Open tools do not cover it.** sign.mt, the leading open-source
  text↔sign system, works reasonably well for American, German and
  Brazilian sign languages only.
- **Tooling is not the blocker; data is.** MediaPipe installs and runs on
  this project's Python. What is missing is an annotated corpus, and the
  Deaf-community partnership needed to build one properly.

What Bao does offer Deaf users is that it is **text-first and works fully
with no audio at all** — every answer is on screen before any voice
begins, and removing the whole speech stack changes nothing about the
answers. Full reasoning, the one narrow feasible piece (fingerspelling)
and why it alone should not be called SASL support, plus sources:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), "Accessibility and SASL".

## Known limitations

Stated here so they are read rather than discovered.

- **Speech covers 11 of 11 South African languages in this deployment,
  4 of 11 on a fresh clone.**
  English, Afrikaans and isiZulu have real South African voices (edge-tts,
  online); Xitsonga has an offline one. No *pretrained MMS* voice exists
  for the other seven — verified against Hugging Face with an
  authenticated request on 2026-08-30, where `mms-tts-xho`, `-sot`,
  `-tsn`, `-nso`, `-ven`, `-ssw` and `-nbl` all return 404. Those seven
  are covered by a multilingual South African VITS model, enabled here via
  `[speech].enable_coqui_sa` in `config.toml`. **The code default is off,
  and deliberately so: the model is cc-by-nc-4.0 while this project is
  MIT, so enabling it makes a deployment non-commercial.** That is
  appropriate for an academic project and would not be for a product. The
  model is gated on Hugging Face (automatic approval) and downloaded at
  runtime, never vendored. Its own model card is an unfilled template with
  no published evaluation, so its quality is not independently verified
  here. Its 138-symbol vocabulary also lacks `š`, the Tshivenda dental set
  `ṱ ḓ ṋ ḽ`, and `ō ē`; Coqui discards unknown symbols, so those are
  folded to their nearest equivalents before synthesis rather than being
  dropped mid-word.
- **Pan-African speech covers 12 of 14.** Enabling the pan-African
  detector brings twelve offline MMS voices, four of which also have a
  Microsoft neural locale (Amharic, French, Somali, Swahili) — all probed
  on 2026-09-18 rather than assumed. Igbo and Lingala have no voice:
  `mms-tts-ibo` and `mms-tts-lin` 404 under every code tried.
- **Luganda is detected as Xitsonga.** The pan-African model identifies it
  correctly but at 43% confidence, just under the 44% that English reaches
  as Nigerian Pidgin. Admitting Luganda would mean relabelling English, so
  24 of 25 languages route correctly and this one does not. Reproduce with
  `python scripts/probe_language_routing.py`.
- **Gemini free-tier quotas are small, and they are per model.** Measured
  on `gemini-3.6-flash`: 5 requests a minute and 20 a day, which is about
  one demo. This deployment uses `gemini-3.5-flash`, chosen by measuring
  availability across the flash models (see `config.toml`); switching
  `[model].gemini_model` gives a fresh budget because each model has its
  own. A knowledge-base fact translated into another language costs a
  request of its own. The app reports this as a quota
  message carrying the provider's own wait time rather than as a failure,
  and does not retry a wait it has been told is long — quick retries would
  spend more of the same quota. Transient 503s (the model being busy) *are*
  retried, up to `[model].generation_max_attempts`.
- **The 0.98 macro F1 is the original training-corpus evaluation.** This
  project independently validated the *deployed* inference path (20/20
  across three small probe sets), which is not the same claim.
- **The 70-query retrieval set was used to tune the coverage threshold**,
  so it is a development set, not an independent test set. Precision 0.826
  should be read as "measured on the set it was tuned on".
- **Language detection is out of distribution on short queries.** The
  classifier was trained on NCHLT news sentences; short imperatives score
  35-42% where full sentences score 64-91%. Detections below
  `min_detection_confidence` are reported but not acted on.
- **Prompt-injection handling is mitigation, not protection.** Regex
  screening of user input plus an untrusted-document boundary. The model
  still interprets the content.
- **Offline-first, not offline-only.** Retrieval and detection run
  on-device; open-ended generation and 3 of the 4 voices need a network.

## Next improvements

- **South African Sign Language** — the right long-term direction, and
  blocked on data rather than code: see "Accessibility and SASL" above.
  The first step is an annotated corpus built with the Deaf community.
- Swap `knowledge/embeddings.py`'s TF-IDF implementation for a real
  multilingual sentence-embedding model to close the false-positive gap
  documented above.
- Reproduce the original 63,613-sentence NCHLT held-out test evaluation
  against this exact deployment, if that dataset becomes available — the
  current 20/20 spot-check and 100% benchmark result are strong evidence
  the deployment bugs are fixed, but they're a different, smaller claim
  than "0.98 Macro F1 confirmed for this deployment."
- Populate `eval/language_eval_template.csv` with a larger, independently
  labelled set spanning more registers (formal, informal, code-switched)
  to track detector accuracy over time beyond the current benchmark set.
- An intent-routing pipeline stage (document Q&A vs. translation request
  vs. general chat) — an early version of this existed as
  `router_service.py` before this refactor and was intentionally deferred
  rather than force-fit into the current pipeline; see
  `docs/ARCHITECTURE.md`.

## Author

**Rinsuna Blessing Mahani** · [@RinsunaMahani](https://github.com/RinsunaMahani)

BSc Mathematics · Sefako Makgatho Health Sciences University

Built as a final-year project. Feedback and questions are welcome through
[GitHub Issues](https://github.com/RinsunaMahani/bao-ai/issues).

## License

Released under the [MIT License](LICENSE).
