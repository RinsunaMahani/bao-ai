# Bao AI — Technical Review

**Version 9 · 2026-08-08 · Status: pre-presentation (target: Aug 30)**

This is a development log across nine rounds of review (one internal,
eight external), kept as the honest record of what was checked and how.
For a presentation-facing document without the debugging history, see
`docs/TECHNICAL_REPORT.md`.

Rounds 2, 3, 4, and 6 each found something the previous round's
"verified" state had missed (§9). Round 5 was different in kind: a
reversal of a prior recommendation, backed by new evidence the project
owner provided — not a missed bug, but a case where "don't trust an
unverified claim" turned into "verify it, and it holds up." Round 6
immediately found that Round 5's own "reproducible" claim wasn't quite
reproducible outside the exact environment it was checked in — the
pattern repeats, and that's the point of §9, not a discouraging sign.

---

## 1. Round 2 — verified claim by claim

Every claim below was independently checked against the actual code and,
where possible, reproduced or empirically tested before being accepted —
"an AI (or anyone) told me X" is a lead to verify, not a fact to relay.

| # | Claim | Verified how | Verdict |
|---|---|---|---|
| 1 | Unguarded `speech_recognition` import breaks `pytest`/`evaluate.py` on a core-only install | Uninstalled `SpeechRecognition`, ran both commands, watched them fail with the exact traceback described; fixed; re-ran to confirm | **Confirmed, fixed** |
| 2 | Heuristic detector has real keyword-overlap ambiguity (e.g. "dumela" in 3+ languages) | Read `KEYWORD_MAP` directly — accurate | **Confirmed, documented — later corroborated by a real benchmark miss, see §4** |
| 3 | TFLite classifier's `ord(c) % 256` preprocessing doesn't use the shipped tokenizer | Inspected `tokenizer_config.json` directly: `char_level: false`, 679,385-word vocabulary, `num_words: 25000` — definitively a word-level tokenizer, not character-level | **Confirmed, more serious than stated — see §5** |
| 4 | "100% self-retrieval accuracy" is a weak metric (asking the DB its own question) | Already self-disclosed in the README before this review arrived | **Confirmed, already documented, no action needed** |
| 5 | KB is too small (32 rows) for "comprehensive SA knowledge" claims | Counted rows directly | **Confirmed — reframing applied, see §6** |
| 6 | Document context gets discarded when offline with no KB match | Read `orchestrator.py`'s branching directly — this exact gap was independently found one conversation earlier, from a different angle ("what does document ingestion do") | **Confirmed, fixed** |
| 7 | Translating a KB fact costs a Gemini API call, a demo-day reliability risk | Confirmed via code read — accurate, not fixed (out of scope this close to the deadline; see §7) | **Confirmed, deferred** |
| 8 | Memory isn't used to resolve follow-up references in KB retrieval ("and its hours?") | Confirmed via code read: `knowledge_retriever.query_fact()` gets only the raw current turn, never memory context | **Confirmed, documented as a known limitation, not fixed** |
| 9 | Security guardrails are basic regex screening, shouldn't be oversold as "AI security" | Confirmed — this project never used that phrase, but the caution is fair | **Confirmed, wording reinforced** |
| 10 | Tests exercise implementation more than real-world behavior | Fair — added black-box tests exercising actual behavior | **Confirmed, addressed** |
| 11 | `REVIEW.md` was stale, describing removed architecture | Confirmed by re-reading it | **Confirmed — corrected each round since** |
| 12 | `gemini_model` should be checked before the presentation | Web-searched current Gemini API docs (checked 2026-08-08) | **`gemini-3.5-flash` valid but one generation behind — updated to `gemini-3.6-flash`. Re-verify before the actual presentation; this changes often** |
| 13 | Docker image only has core deps, README could imply otherwise | Confirmed via `Dockerfile` read | **Confirmed, docs clarified** |
| 14 | UI shouldn't present heuristic confidence as if calibrated | Confirmed as reasonable UX/integrity concern | **Confirmed, wording updated** |

**What the verification process itself found, beyond the claims list:**

- Fixing the TFLite preprocessing required installing `tensorflow-cpu` and
  running the actual `.tflite` model to inspect its real input tensor
  shape (`(1, 35)` float32) — confirmed the 35-token length assumption was
  at least consistent between old and new preprocessing, which is exactly
  why the bug never threw a shape error and looked like it was "working."
- A pandas 3.0 / sklearn incompatibility, unrelated to the external
  review, was found while stress-testing the sign-language training
  script: pandas 3.0's new Arrow-backed string dtype breaks
  `sklearn.train_test_split(..., stratify=y)` with a cryptic `TypeError`.
  Fixed in `scripts/train_sign_classifier.py`.

## 2. Round 3 — verified claim by claim

| Claim | Verified how | Verdict |
|---|---|---|
| `sign_language.py` binds `numpy` only inside the `mediapipe` try/except, so `landmarks_to_feature_vector()` — explicitly designed to work without mediapipe — throws `NameError` when mediapipe isn't installed | Uninstalled `mediapipe`, ran `pytest`, reproduced exactly: `2 failed, 31 passed, 1 skipped`. Fixed (moved `numpy` import outside the guard — already a core dependency used everywhere else). Re-ran in the same broken environment: `33 passed, 1 skipped` | **Confirmed, fixed, re-verified in the failing environment, not just the working one** |
| Round 2 of this document claimed "33 passing" while the real count (in a mediapipe-less environment) was 31 passing + 2 failing | Corrected in Round 3 | **Confirmed — corrected** |
| `services/__init__.py` still eagerly imports `speech` (safely guarded internally now, but still eager) | Confirmed via read — accurate, low-priority, not fixed | **Confirmed, deferred (§7)** |
| Sign-language UI status should be unambiguous ("Experimental SASL prototype — not part of the primary demo") when untrained | Updated both the sidebar status line and the input expander's caption | **Confirmed, fixed** |

**What checking this round surfaced, beyond the claim:**

- Running 5 manual Streamlit test cases via `streamlit.testing.v1.AppTest`
  (actually driving them, not assuming they'd probably work) surfaced a
  **more dramatic version of the already-documented TF-IDF false-positive
  limitation**: the nonsense query *"what is the airspeed velocity of an
  unladen swallow"* scores **0.443** against a fact about South Africa's
  currency — nearly 3x the threshold, and stronger evidence than the
  previous example (0.186, a query that at least shared the word
  "capital"). Now the lead example in the README and `ARCHITECTURE.md`;
  both are pinned as separate regression tests.
- Driving those same manual cases exposed a **real, previously unnoticed
  test gap**: no test — unit or otherwise — exercised the "Gemini
  reachable, generates a response" branch of the orchestrator. Every
  other branch had coverage; this one didn't. Added
  `test_gemini_generates_response_when_available_and_no_kb_match` with a
  mocked client, since no live API key is available in this environment.

## 3. Round 4 — verified claim by claim

| Claim | Verified how | Verdict |
|---|---|---|
| `config.toml`'s `mms_codes`/`stt_codes` keys ("IsiZulu", "SiSwati", "IsiXhosa", "IsiNdebele", "TshiVenda") use different capitalization than the canonical `LABELS` ("isiZulu", "siSwati", ...) | Tested the actual `dict.get()` lookup before fixing: `codes.get("isiZulu", "eng")` returned `"eng"` (the fallback), not the real isiZulu code. Confirmed this is a **real functional bug**, not cosmetic: 5 of 11 languages would silently produce English TTS audio regardless of what was actually detected. Fixed; added `tests/test_config.py` pinning that every canonical label resolves | **Confirmed, more serious than stated — a silent runtime bug, not a style nit** |
| Benchmark's "LANGUAGE DETECTION CONFIDENCE" section mislabels an uncalibrated heuristic score as calibrated confidence | Confirmed via code read — same class of issue already fixed in the UI badge, missed in the benchmark script | **Confirmed, fixed — relabeled by detector backend, with an explicit non-calibration note for the heuristic case** |
| Benchmark has `expected_lang` in its test data but never compares against it, despite the accuracy check being one dict lookup away | Confirmed via code read; added the comparison, then ran the real benchmark | **Confirmed, fixed — real output: 80% (20/25) on the 5-query benchmark set, with the one miss being `"Dumela..."` (expected Sepedi) detected as Sesotho, directly corroborating the "dumela" ambiguity documented in §1** |
| "Multilingual South African Voice & Text Assistant" oversells the primary UI, which has no voice-input workflow (`transcribe_audio_bytes` is never called anywhere in `streamlit_app.py`) | Grepped for every call site — confirmed zero — while the console app does have a real microphone path (bypassing that same function, calling `speech_recognition` directly) | **Confirmed, fixed — sidebar caption now says "Text-first, with optional speech output," and a real TTS-availability status line was added (there wasn't one before)** |
| `services/__init__.py`'s eager (but safely guarded) speech import | Same finding as Rounds 2 and 3 — still accurate, still low priority | **Confirmed, still deferred** |

## 4. Concrete evidence trail: the "dumela" ambiguity

Documented as a keyword-overlap risk in Round 2 (§1, claim 2), then
independently reproduced by the benchmark's newly-added accuracy check in
Round 4 without being specifically aimed at it: `"Dumela, nka thusha bjang
ka system e?"` (expected Sepedi) is detected as Sesotho **by the
heuristic detector**. Two different checks, run at different times,
converging on the same specific failure.

**Completed in Round 9 (§9):** the ML detector classifies that same query
as Sepedi at 99.3% confidence, and handles the full Sotho-Tswana group
(Sepedi / Sesotho / Setswana, all sharing "Dumela") at 9/9. So this is a
limitation of keyword matching specifically, not of the trained model —
which makes it the clearest single justification for the ML detector
being the production default. A good concrete example to have ready if
asked why the project didn't just ship the simpler rule-based approach.

## 5. The TFLite language classifier — reversed, with real evidence

Prior versions of this document said "do not enable `prefer_ml=True`."
**That recommendation is reversed as of Round 5, based on real evidence,
not a change of heart.** Here's the full sequence, since how this
conclusion was reached matters as much as the conclusion itself.

**Round 4's status:** the character-code preprocessing bug was fixed
(reconstructed Keras's `texts_to_sequences` from `tokenizer_config.json`
since the original training script wasn't available), but a 4-sentence
manual spot-check showed no clear improvement over the broken version —
1/4 correct, low confidence. Conclusion at the time: fixing preprocessing
isn't the same as validating the model; keep it off.

**What changed:** the project owner provided the actual training
presentation (`BaoBao_updated.pptx`) documenting the original Google Colab
experiment — 636,123 NCHLT sentences across all 11 languages, an
80/10/10 split, and a reported 0.98 Macro F1 on 63,613 held-out test
sentences. Before accepting any of that, the architectural claims in the
deck were checked against the actual `ultimate_african_ai.keras` file
directly: embedding dimension, LSTM units, output classes, and total
trainable parameter count (3,292,711) all matched **exactly**. That's not
a number someone guesses; it's strong evidence the deck describes the
real model that's actually deployed here.

That raised an obvious question the 4-sentence spot-check couldn't
answer: if the trained model is genuinely that good, why did it score
1/4? Investigating found **two compounding bugs, not one**:

1. **Padding direction was wrong.** The Round 4 fix used Keras's default
   (`padding='pre'`), never confirmed against the real training setup.
   Testing both directions against the real model with unambiguous
   sentences showed the actual signature of right vs. wrong preprocessing:
   `pre` produced near-uniform, ~30% confidence (consistent with
   essentially random input); `post` produced 97-100% confidence,
   consistently.
2. **The class-index-to-language mapping was wrong**, independently of
   padding. The deployment used `core.config.LABELS` (an order chosen for
   UI purposes) to map the model's output index back to a language name.
   The model's actual training-time class order — reverse-engineered by
   feeding one clearly-monolingual sentence per language through the real
   model and recording which output index fired for each — turned out to
   be **alphabetical**, not `LABELS`' order. This is now
   `TFLITE_CLASS_ORDER` in `bao/services/language_detector.py`, pinned by
   `tests/test_language_detector.py::TestTFLiteLanguageDetector`.

With both fixed, tested against the **real deployed `.tflite` file** (not
just the Keras source, so this also confirms the TFLite conversion is
faithful — a question Round 4 hadn't settled either):

- **11/11** correct on one clearly-monolingual formal sentence per
  language (0.97-1.00 confidence)
- **4/4** correct on short, informal, chat-style greetings — a different
  register from NCHLT's formal training text, and closer to real user
  input — including the exact Sepedi "dumela" case the heuristic detector
  is documented to get wrong (§4)
- **5/5** on the pre-existing `benchmark_bao.py` query set, versus the
  heuristic's 4/5 on the same set

20/20 across three independently-constructed test sets is real evidence,
checked directly rather than asserted — **with one important caveat added
in Round 6 (§10): it only reproduces in an environment where tensorflow
is genuinely functional**, which is not the same as "the model files
exist" or even "`import tensorflow` succeeds" (see §10 for exactly why
that distinction turned out to matter). It also doesn't reproduce the
full 63,613-sentence NCHLT test set (that data isn't available here), so
it isn't the same claim as "0.98 Macro F1 confirmed on this deployment" —
but it's strong enough that `prefer_ml=True` is now the default in
`bao/ui/streamlit_app.py` and `bao/ui/console_app.py`, with the heuristic
detector as the automatic fallback if the model or tokenizer files are
ever missing.

## 6. Round 6 — verified claim by claim: the "20/20" claim wasn't actually reproducible

The Round 5 claim (§5) was real evidence, checked directly against the
real model — but a further external review reproduced it in a different
environment and got `38 passed, 2 failed, 1 skipped`, not the claimed
`40 passed, 1 skipped`. Investigated and confirmed as three compounding,
real problems, not a flaky test:

| Claim | Verified how | Verdict |
|---|---|---|
| The TFLite tests' skip condition only checked file existence (`os.path.exists()` on the model and tokenizer), never whether tensorflow itself was actually usable | Uninstalled tensorflow, kept the model/tokenizer files in place, ran the tests: they did NOT skip — they ran and failed, with the failures looking exactly like a broken ML model (`expected: Afrikaans, got: English`) when the real cause was a silent fallback to the heuristic detector | **Confirmed, fixed — and worse than the file-existence framing suggested, see below** |
| `evaluate.py --use-ml-detector` never passed `tokenizer_config_path`, so it could silently evaluate the heuristic while claiming to evaluate the ML model | Read the code directly — confirmed missing. Ran `evaluate.py --use-ml-detector` in the broken-tensorflow environment: it silently produced a "detector accuracy" report that was actually the heuristic's numbers | **Confirmed, fixed — now raises an explicit warning to stderr if the ML detector was requested but the result didn't actually come from `backend == "tflite"`** |
| `REVIEW.md`'s "20/20 ... is real, reproducible evidence" phrasing was too strong given the above | Fair — reworded in §5 to be explicit about the environment dependency | **Confirmed, wording corrected** |

**What checking this round found that went beyond the reported claim:**
fixing the skip condition to check `_HAS_TFLITE` (language_detector.py's
own tensorflow-import flag) looked sufficient and *still wasn't* —
uninstalling tensorflow in this sandbox left a broken namespace-package
stub behind (`import tensorflow` succeeds; `tf.lite` doesn't exist), so
even an import-success check reported "available" for a runtime that
couldn't actually construct the model. The check that can't be fooled by
a half-working install is the one now used: **actually attempt to
construct the detector and confirm `result.backend == "tflite"`**, not
any proxy for whether that should be possible. Both TFLite tests now also
assert `result.backend == "tflite"` as their first check, so a future
silent fallback fails with an obvious "wrong backend" message instead of
a confusing "wrong language" one that looks like the model itself is
broken when it never ran at all.

Re-verified after all three fixes, in both states: **with tensorflow
broken/absent, the 3 TFLite tests now correctly SKIP (5 passed, 3 skipped
in that file) instead of failing; with a genuinely working tensorflow
install, all of them PASS (40 passed, 1 skipped overall).** That's the
actual, now-precise meaning of "reproducible" for this claim.

## 7. Round 7 — the Keras-fallback backend bug, and RAG finally measured

| Claim | Verified how | Verdict |
|---|---|---|
| `TFLiteLanguageDetector.detect()` hardcoded `backend="tflite"` even when the Keras fallback produced the prediction | Read the code: line 285 runs `self.keras_model.predict(...)`, line 291 unconditionally reported `"tflite"`. **More serious than cosmetic**: Round 6 added `assert result.backend == "tflite"` specifically to prove the ML path ran — this bug would have let a Keras fallback satisfy that assertion falsely, quietly defeating the check | **Confirmed, fixed, pinned by two new tests** |
| The UI badge only handled `"heuristic"` and `"tflite"` | Confirmed — a `"keras"` result would fall through to a bare `else` and silently lose its confidence display | **Confirmed, fixed — the Keras fallback is now visible to the user (same weights, but slower and heavier, worth surfacing)** |
| The benchmark's "25/25" is 5 unique queries repeated 5 times, not 25 independent examples | Confirmed by reading the loop | **Confirmed, fixed at the source — the benchmark's own output now says "5/5 unique queries ... 25 total inference runs ... NOT independent language examples", so the misleading framing can't be reintroduced by someone reading raw output** |
| Hardcoded test counts across README/CONTRIBUTING/REVIEW will drift out of sync | Confirmed — README said 37, REVIEW said 40, actual was 42 at the time of checking | **Confirmed, fixed — counts replaced with "run `pytest` for the current count" rather than manually maintained in three places** |

**The substantive new work this round: RAG is now actually measured.**
`eval/rag_eval.csv` (50 hand-labelled queries against the real knowledge
base — 15 verbatim, 15 paraphrase, 20 topically unrelated) plus
`bao/rag_evaluation.py` (precision/recall/F1 and a 0.05-0.60 threshold
sweep), wired into `evaluate.py --rag-eval`.

The finding is worth stating plainly, because it's more useful than a
passing grade would have been:

| Threshold | Exact | Paraphrase | Negative rejected |
|---|---|---|---|
| 0.15 (old default) | 15/15 | 12/15 | **2/20** |
| 0.25 (new default) | 15/15 | 11/15 | 8/20 |
| 0.45 | 15/15 | 3/15 | 13/20 |
| 0.55 | 13/15 | 0/15 | 20/20 |

(These are the original 20-negative numbers. Round 8 doubled the negative
set with harder cases and the picture got materially worse — see §8. The
20-negative figures are kept here as the historical record of what this
round actually measured.)

At the old 0.15 default, **18 of 20 completely unrelated questions
received a confidently wrong knowledge-base answer.** But raising the
threshold to fix that destroys paraphrase handling — at 0.55, negatives
are perfect and paraphrase recall is zero. **No threshold is actually
good.** The default moved to 0.25 (best F1, 0.765) on this evidence, and
that's a compromise, not a solution — which is exactly the measured case
for semantic embeddings that previous rounds had only argued for
qualitatively. `tests/test_rag_evaluation.py` pins the trade-off so a
future embedding upgrade has something concrete to beat.

## 8. Round 8 — harder negatives, and the semantic experiment set up but not run

Acting on the previous round's Priority 1 (expand negatives, then compare
against a semantic backend). Both parts, honestly separated:

**Part 1, done and measured: the negative set doubled, 20 -> 40**, adding
the two harder categories specifically called for — lexical traps
(questions sharing vocabulary with KB entries but different meaning:
"what is the capital of Australia", "where is the national library of
France") and South-Africa-related questions that simply aren't in the
knowledge base ("who is the current president of South Africa", "what is
the minimum wage in South Africa").

The result is materially worse than the easier 20-negative set, which is
exactly why it was worth doing:

| Metric | 20 negatives | 40 negatives (incl. hard cases) |
|---|---|---|
| Negative rejection @ 0.25 | 8/20 (40%) | **8/40 (20%)** |
| Precision @ 0.25 | 65.0% | **43.3%** |
| F1 @ 0.25 | 0.765 | **0.591** |

**Every one of the 20 new hard negatives was falsely matched.** The worst
cases are the most instructive: "who is the current president of south
africa" scores 0.765, and "what is the speed of light" scores 0.532 —
both against a fact about South Africa's *currency*, the latter purely on
the shared function words "what is the of". That single example makes the
case better than any aggregate metric: no threshold can separate 0.53
(complete nonsense) from legitimate paraphrases scoring in the same band.
Pinned in `tests/test_rag_evaluation.py`.

**Part 2, built but NOT run — and this matters:**
`bao/knowledge/embeddings.py::SentenceEmbeddings` implements the same
`EmbeddingModel` interface as TF-IDF, `KnowledgeRetriever` now takes an
injectable `embedding_model` (the swap the interface was designed for,
finally exercised), and `compare_retrievers.py` runs both backends against
the same 70-query set side by side.

**The comparison itself has not been run.** The semantic backend needs
model weights from Hugging Face, which was not reachable from the
environment this was written in — confirmed by trying, not assumed. So
this round delivers a ready-to-run experiment and an honest TF-IDF
baseline to beat, not a result. `compare_retrievers.py` prints the TF-IDF
column and says plainly that the semantic column is unpopulated. Running
it is a one-command task on a machine with normal network access.

One practical finding worth recording, discovered while setting this up:
installing `sentence-transformers` pulls in torch, which **segfaulted the
benchmark** when installed alongside TensorFlow in this environment — a
native-library collision, not a code bug (confirmed by uninstalling torch
and watching the benchmark work again). Documented in
`requirements-semantic.txt` and the class docstring, because it would
otherwise look like the TFLite detector had spontaneously broken.

## 9. Round 9 — the "Sepedi" discrepancy resolved, and it was a masked fallback again

An external review reported the benchmark showing TFLite failing the
Sepedi "Dumela" case (`tflite = Sesotho`), directly contradicting a
passing test asserting the opposite. Both could not be true, so this was
investigated before anything else.

**Resolution: the model is right, the reviewer's environment was missing
tensorflow.** In an environment with a working tensorflow install the
TFLite detector classifies that query as **Sepedi at 99.3% confidence**;
only the heuristic returns Sesotho. The reviewer's "tflite" column was
the heuristic silently falling back — the exact masking failure already
fixed for the *tests* in Round 6, which the *benchmark* had no equivalent
guard against.

**Fixed:** `benchmark_bao.py` now labels its comparison column by the
backend that actually ran, and prints an explicit warning when the ML
detector falls back. Verified by reproducing the reviewer's environment
(uninstalling tensorflow): the benchmark now prints
`WARNING: the ML detector fell back to 'heuristic'` and labels both
columns `heuristic`, so the result can no longer be misattributed. Third
instance of this same class of bug in three different places (tests,
`evaluate.py`, benchmark) — the general lesson is in §11.

**Also answered, a genuinely open question from the review:** is the
Sepedi/Sesotho confusion a systematic model weakness that would justify
retraining? Tested the full Sotho-Tswana group — the three most closely
related languages in the label set, all of which share the greeting
"Dumela" — with three sentences each including all three "Dumela"
greetings:

**9/9 correct**, with all three "Dumela" greetings correctly separated
into Sepedi, Sesotho and Setswana (0.993, 0.993, 0.993 confidence).

So the confusion is a **keyword-matching limitation, not a model
limitation** — precisely the case the ML detector exists to handle. No
retraining is warranted, and this is now pinned by
`test_disambiguates_the_closely_related_sotho_tswana_group`. It's also
the single clearest concrete justification for the ML detector being the
default rather than optional.

## 10. Positioning: what this project should claim, and shouldn't

> **Say:** "An offline-first multilingual assistant designed around South
> African languages and locally relevant knowledge, that only reaches for
> generative AI when its verified local knowledge doesn't have an answer."
>
> **Don't say:** "An AI containing comprehensive South African knowledge."
> **Don't say:** "A voice assistant" — the primary UI is text-first with
> optional speech output; there's no voice-input path in `streamlit_app.py`.

The knowledge base has 32 curated entries, several campus-specific (a
university library, a student card, a calculus course) — a legitimate
prototype scope, not a claim of broad coverage. The verified-KB-vs-
generative split (§1 claim 6's fix, and the orchestrator's own docstring)
is the actual interesting architectural idea here — lead with that in a
presentation, not with knowledge-base size.

## 11. What's fixed vs. deliberately deferred

**Fixed across all seven rounds** (re-verified with `pytest tests/ -v`
after each change — run it for the current count rather than trusting a
number written here; hardcoded counts across three documents drifted out
of sync twice before this was fixed in Round 7 — confirmed in both a
full and a genuinely minimal environment — mediapipe and SpeechRecognition
both actually uninstalled, not just assumed absent):
- Unguarded `speech_recognition` import (critical — broke
  `pytest`/`evaluate.py` on any fresh core install)
- Unguarded `numpy` import in `sign_language.py` (same class of bug, same
  fix pattern, found a round later)
- TFLite preprocessing rebuilt from the real tokenizer vocabulary
- Discarded document context when offline (dead retrieval work, silently
  dropped result)
- Gemini model name updated to the current GA model, dated
  re-verification note attached
- pandas 3.x / sklearn incompatibility in the sign-language training script
- Sign-language UI status made unambiguous about experimental scope
- Closed a real test gap: the online/Gemini-success orchestrator branch
  had no test at all until Round 3
- `config.toml`'s language-code casing bug (a real silent-fallback bug,
  not cosmetic) — Round 4
- Benchmark's misleading confidence terminology and its unused
  `expected_lang` accuracy check — Round 4
- "Voice & Text Assistant" tagline corrected to match what the UI
  actually does — Round 4
- **TFLite's padding direction and class-order mapping — two compounding
  bugs, both found and fixed in Round 5.** `prefer_ml=True` is now the
  default, backed by 20/20 across three test sets (§5), not just a
  reversed guess. `get_language_detector()` was also missing
  `tokenizer_config_path` at both call sites in the app — even flipping
  `prefer_ml=True` alone, without that, would have failed to construct
  the detector at all.
- **The TFLite tests' skip condition and `evaluate.py`'s missing
  `tokenizer_config_path` — Round 6.** The "20/20" evidence from Round 5
  wasn't actually reproducible in a different environment because of
  these two bugs, not because the underlying finding was wrong. Both
  fixed; the skip condition now actually constructs the detector rather
  than checking any proxy for whether that should be possible (a
  proxy-based check was tried first and still wasn't robust enough — see
  §6 for exactly how it failed).

- **Keras-fallback backend misreporting — Round 7.** `detect()`
  hardcoded `backend="tflite"` regardless of which engine ran, which
  would have let a Keras fallback falsely satisfy the Round 6 assertions
  that exist to prove the TFLite path executed.
- **RAG measured for the first time — Round 7.** 50-query labelled set +
  threshold sweep (§7). Default threshold moved 0.15 -> 0.25 on evidence;
  the more important output is the documented finding that no threshold
  works well, which is now the measured case for semantic embeddings.
- **Misleading "25/25" benchmark framing and hardcoded test counts —
  Round 7.** Both fixed at the source rather than in the docs.

**Deliberately not fixed before the presentation:**
- Query resolution for follow-up questions against the offline KB
  (memory works for Gemini, not for retrieval) — documented limitation
- KB-fact translation costing a Gemini call — real API-dependency risk for
  a live demo; pre-translated static answers for a handful of key facts is
  a good Phase-2 idea, not attempted here
- A 3-tier confidence bucket instead of a binary threshold — reasonable,
  bigger than a quick fix
- Paraphrase and negative-retrieval evaluation metrics beyond
  self-consistency — needs a real hand-built eval set to mean anything
- `services/__init__.py`'s eager (but now safe) speech import
- Full reproduction of the 63,613-sentence NCHLT held-out test — the
  20/20 spot-check is strong evidence, not the same claim; reproducing
  the original number needs the actual test set, which isn't in this
  repository

## 12. Verification discipline, as a standing practice for this project

Four external rounds (2, 3, 4, 6) each found something the previous
round's "verified" state had missed — not fabricated, just
environment-dependent in a way that wasn't visible from inside the
environment that produced the claim (a masked bug in Round 3, a lookup
that only fails against real config data in Round 4, a claim that only
holds with a working ML runtime in Round 6). The practice going forward:
**run the actual failing scenario directly — uninstall the dependency
with its correct package name, break the thing on purpose — not the
convenient scenario already set up.** Round 4's own verification caught a
self-inflicted version of exactly this: an initial `pip uninstall
speech_recognition` silently no-op'd against the real package name
`SpeechRecognition`, so a "minimal environment" test wasn't actually
minimal until that was caught and corrected mid-check. Round 6 went a
layer deeper: even after fixing the skip condition to check whether
tensorflow was importable, that still wasn't robust — an uninstalled
tensorflow left a broken namespace-package stub behind, so `import
tensorflow` succeeded while `tf.lite` didn't exist. The lesson generalizes
past this one bug: **checking whether a dependency imports is not the
same as checking whether it works** — the only check that can't be fooled
is attempting the real operation and confirming the actual result (here,
constructing the detector and checking `result.backend == "tflite"`, not
any proxy for whether that should be possible). Even the checking process
needs checking.

Round 9 added the sharpest version of this lesson: the *same* masking bug
— an optional ML dependency silently falling back while the output still
claims the ML path ran — appeared in three separate places across three
rounds (the test suite, `evaluate.py`, and the benchmark). Fixing it in
one place did not fix it in the others, and each time it produced
plausible-looking wrong output rather than an error. It even caused an
external reviewer to report a model failure that wasn't real. **Any code
path that reports which backend produced a result must derive that label
from what actually ran, never from what was requested.**

A claim of "N tests passing" is only as trustworthy as
the environment it was measured in — every claim in this
document says which environment, for that reason.

## 13. What I would explicitly NOT do before the presentation

No FastAPI rewrite, no replacing Streamlit, no vector database for 32
rows, no agent framework, no intent router, no reviving sign language on
this timeline. The current architecture is sufficient for the
presentation; the risk this close to the deadline is overengineering, not
underengineering.

---

## Round 10 — the speech/accent round (2026-08-25)

Context: an external AI session was used to fix the TTS accent. Some of it
was right, some of it broke the app. This round separates the two.

### What that session got right, and is now applied

| Change | Verdict | Where it landed |
|---|---|---|
| TTS must be routed by the language of the **reply**, not the query | **Correct, and the actual cause of the silent turn** — "explain cloud computing in tsonga" is English, the reply is Xitsonga, and the reply was being handed to an English/Afrikaans voice | `orchestrator._speech_language()`, threshold-guarded so a weak detection on the reply can't override a confident one on the input |
| Long replies must be chunked before VITS | Correct in direction | Kept, but rewritten: terminal punctuation is now **preserved** (both backends use it for prosody; discarding it flattened long replies to monotone) and a word-boundary backstop handles text with no full stops at all |
| Browser mic input via `st.audio_input`, console playback of real MMS bytes instead of `pyttsx3` | Correct | Already in the tree; left alone |
| `speaking_rate` / `noise_scale` as config knobs | Correct — `VitsModel.forward` does read these off the instance | Kept; `speaking_rate` now also drives edge-tts via `_rate_percent()` so one knob controls both |

### What that session got wrong

| Change | Problem |
|---|---|
| Rewrote `core/security.py` and `knowledge/retriever.py` with a different API | Neither matched `orchestrator.py`, which the user (correctly) did not replace. Every turn died with `AttributeError`. The retriever rewrite also bypassed `knowledge/embeddings.py` and so silently **reverted the stop-word fix** — the "what is quantom physics" false positive was back. Both reverted in Round 10; see the warning comment at the top of `retriever.py` |
| The proposed `orchestrator.py` to pair with them | A downgrade: lost `force_offline`, the offline document-context branch, `translate_fact`, and the typed-exception handling. Not applied |
| `clean_text_for_speech` stripping the apostrophe | Would corrupt Xitsonga/isiZulu orthography (`ematshan'weni`). Not applied; now pinned by a test |
| `"English" = "afr"` framed as giving English "South African phonology" | **Half-true, and the important correction of this round.** It applies *Afrikaans* letter-to-sound rules to English words — "very" toward "ferry", "will" toward "vill". That is an Afrikaans speaker mispronouncing English, not South African English. MMS simply has no en-ZA model |

### The accent fix that actually addresses the complaint

Restored the two-backend design: `edge-tts` for the three languages with a
genuine Microsoft South African locale, MMS offline for the other eight.
Voice IDs (`en-ZA-LukeNeural`, `af-ZA-WillemNeural`, `zu-ZA-ThembaNeural`)
were checked against the published edge-tts voice list on 2026-08-25
rather than recalled — a wrong ID fails at request time with an unhelpful
error. `"English" = "afr"` is retained as the *offline* fallback and is
now documented as the trade-off it is.

### What is verified, and what is not

Verified here: 61 tests pass, `ruff check .` clean, Streamlit boots (HTTP
200), console runs a full turn, backend routing and text chunking are
covered by new tests in `tests/test_speech.py`, and the graceful-failure
path was exercised for real (edge-tts installed, endpoint unreachable →
one warning line, `None` returned, turn still answered).

**Not verified: any actual audio.** Neither backend can run in this
environment — MMS needs a Hugging Face download and edge-tts needs
`speech.platform.bing.com`. Backend *selection* is tested; voice quality
has to be judged by ear on a real machine before the presentation.

### Still open

- `stt_codes` in `config.toml` maps all eleven languages to `*-ZA` codes
  for `recognize_google`. Google's speech API is not believed to support
  most of them, so those lookups likely fail at request time rather than
  degrading. Not changed this round because it could not be verified from
  here — **test microphone input in a non-English language before relying
  on it in the demo.**

---

## Round 11 — latency (2026-08-25)

Reported symptom: replies take far too long. Measured before changing
anything, because the offline pipeline already benchmarks at **2.8 ms**
mean (`benchmark_results.json`) — so retrieval and detection were never
the problem, and any fix aimed at them would have been wasted work.

### What was actually costing the time

1. **Speech was inside the blocking path.** `handle(want_speech=True)`
   synthesized audio *before* returning, so the UI showed nothing until
   the voice was finished — even though the text had been ready for
   seconds. MMS runs one VITS forward pass per sentence on CPU, so a
   multi-paragraph answer is the worst case, and that is exactly the kind
   of answer this app produces.
2. **Generation was not streamed.** `generate_content` waits for the
   entire reply. Time-to-first-word equalled time-to-last-word.
3. **Every non-English knowledge-base hit paid a Gemini round trip.**
   `translate_fact` is called on each KB answer, uncached — so asking the
   same greeting three times in a demo bought three identical API calls.

### Changes

| Change | Effect |
|---|---|
| `Orchestrator.speak(result)` splits synthesis out of `handle()` | Text is returned as soon as it exists; the UI renders, then synthesizes |
| `GeminiClient.generate_stream()` + `handle(on_chunk=...)` | Words appear as they are generated. Total time unchanged; perceived time is not |
| Session cache in `services/translation.py` | Repeat KB answers skip the API entirely |
| "Speak replies" toggle in the sidebar | Skips the slowest stage outright when audio isn't wanted |
| `PipelineResult.stage_timings` + sidebar breakdown | Says *which* stage was slow instead of only that the turn was |

`handle()`'s signature is backward compatible: `on_chunk=None` uses the
original non-streaming call, and `want_speech=True` still synthesizes
inline (correct for the console, where nothing else is waiting). Pinned by
`test_without_a_callback_the_non_streaming_path_is_used`.

### Measured

Structural ordering verified with a deliberately slow (2 s) stub
synthesizer — the delay is synthetic, the ordering is real code:

```
BEFORE (speech inside handle): text visible after   2003.9 ms
AFTER  (speech after handle):  text visible after      2.4 ms
                               audio ready after    2003.1 ms
```

68 tests pass, `ruff check .` clean, Streamlit boots, console unchanged.

### Not measured here

Real Gemini and real MMS latency, for the usual reason: no API access from
this environment. The sidebar breakdown is there so those numbers can be
read off a real machine — **run one long cross-lingual query before the
presentation and read the `generation` and `speech` figures.** If `speech`
still dominates, the honest lever is the toggle, not more tuning: neural
TTS of a 400-word answer on a laptop CPU is slow because it is slow.

---

## Round 12 — sign language reach, and voice input placement (2026-08-25)

### Voice input moved into the chat box

`st.chat_input` supports a built-in microphone (`accept_audio=True`),
returning a value object with `.text` and `.audio`. The recorder now lives
there instead of in an expander above the transcript, which is where every
other chat product puts it and where people look for it.

The capability is **probed**, not version-compared
(`"accept_audio" in inspect.signature(st.chat_input).parameters`), because
this project's floor is `streamlit>=1.40` and the keyword arrived later.
Verified both branches: on 1.62 the probe is True and the in-chat mic is
used; with `st.chat_input` monkeypatched to a mic-less signature the probe
goes False and the old expander renders instead of raising `TypeError` on
an unsupported keyword.

### Sign language — what was actually blocking it

The recognizer was never the weak point. Three things downstream of it
were, and none needed ML:

| Problem | Fix |
|---|---|
| A recognized sign is a bare gloss (`help`), sent to chat as a one-word fragment | `data/sign_phrases.csv` + `SignPhraseBook` maps gloss → phrase ("I need help") |
| One photo = one message, but a single static handshape rarely says anything alone | Signs are captured into a sequence and composed into one turn |
| **Gloss labels are English words, so detection answered every signed turn in English** | `handle(language_override=...)` — the signer picks the reply language; it also wins for the voice, so a detector guess on the reply can't overrule an explicit choice |

The third is the one that mattered for the project's premise: sign input
was wired into an eleven-language assistant and could only ever get
English back.

### Scope, stated plainly

- **Sign → spoken language: implemented.** A signer can compose a request
  and get a reply, with audio, in any of the eleven.
- **Spoken language → sign: not attempted.** Needs a licensed SASL video
  corpus or a signing avatar; neither exists off the shelf, and both are
  separate projects.
- **Neither direction is translation.** `compose()` preserves capture
  order rather than reordering into spoken-language grammar, because SASL
  is not signed English and guessing at the mapping would be worse than
  not claiming it. Continuous recognition, non-manual markers, and grammar
  mapping are all listed in the guide as what real translation needs.

### Still the actual blocker

`models/sign_classifier.joblib` does not exist, so `is_available()` is
False and none of the above runs yet. That is a data-collection problem —
150-200 samples per sign, of the user's own hands — and no amount of code
changes it. Everything added this round is testable without it (8 new
tests in `tests/test_sign_phrases.py`, 3 in `test_orchestrator.py`),
which is why it was worth building now rather than after.

79 tests pass, `ruff check .` clean, Streamlit boots.

---

## Round 13 — external review, verified claim by claim (2026-08-25)

A 20-point external review. Same discipline as Round 2: every claim was
checked against the code before being accepted, and two turned out to be
wrong or stale.

### Verified and acted on

| # | Claim | Verified how | Outcome |
|---|---|---|---|
| 3 | TF-IDF returns confidently-wrong KB answers ("minimum wage in south africa" → currency fact, tagged verified) | Reproduced. Found **three** live cases, not one: minimum wage (0.666), population (0.666), register to vote (0.460) | **Confirmed, worse than stated, fixed** — see below |
| 9 | Uploaded documents reach the Gemini prompt unscreened; a PDF can carry prompt injection | Read `prompts.open_ended_prompt` — document text was pasted after a bare "Local document context:" heading with no boundary | **Confirmed, fixed** |
| 11 | Translation cache should be bounded | It was an unbounded dict | **Confirmed, fixed** (LRU, cap 512) |
| 17 | Keras model and tokenizer.pickle may not need to ship | Traced it: **neither UI passes `fallback_keras_path`**, so the Keras model is never loaded, and `tokenizer.pickle` is referenced *nowhere in the codebase*. 69.2 MB of 96.2 MB — **72% of model weight is unused at runtime** | **Confirmed, more definite than stated** — excluded from the Docker image |
| 19 | 100% self-retrieval is the wrong headline number | Agreed and already self-disclosed | **Fixed** — `evaluate.py` now runs the 70-query labelled evaluation by default, and the self-consistency figure prints its own caveat so it cannot be quoted alone |

### Corrected

- The review cites "airspeed velocity of an unladen swallow" as a live
  high-similarity false positive. It scores **0.000** — that class was
  fixed by the stop-word list in an earlier round, and
  `test_function_word_only_queries_are_rejected` pins it. The review was
  reading a docstring that describes the bug it fixed.
- "Basic prompt-injection screening" wording: already what
  `core/security.py` and the README say. No change needed.

### The retrieval fix, and why it isn't semantic embeddings

The review's Priority 1 is to benchmark `paraphrase-multilingual-MiniLM`
against TF-IDF. That is the right long-term move and
`compare_retrievers.py` exists for it — but it needs a ~470 MB Hugging
Face download, which is not possible in this environment. So the root
cause was diagnosed instead of the symptom being swapped out.

**A TF-IDF vectorizer silently drops terms outside its fitted
vocabulary.** "what is the minimum wage in south africa" therefore gets
scored as if the user had asked "south africa" — and duly matches the
currency fact. The similarity is not wrong; it is answering a different
question. The missing signal is not semantic, it is *how much of the
question survived*.

`KnowledgeRetriever.query_coverage()` measures that, weighted by IDF, with
an unknown word charged the maximum IDF (a word absent from the corpus is
maximally distinguishing). Both the weighting and the threshold were
chosen by measurement on `eval/rag_eval.csv`, not by argument:

| Gate | TP | FP | Precision | Recall | F1 |
|---|---|---|---|---|---|
| similarity only (before) | 26 | 22 | 0.542 | 0.929 | 0.684 |
| plain word-count coverage @ best | 22 | 10 | 0.688 | 0.759 | 0.721 |
| **IDF-weighted coverage @ 0.70 (shipped)** | **19** | **4** | **0.826** | **0.655** | **0.731** |

False positives 22 → 4. The recall cost is real and deliberate: the two
error types are not equally bad here. A false positive is returned tagged
`source="knowledge_base"` — *verified* — and bypasses generation entirely.
A false negative just falls through to Gemini, which usually answers
correctly anyway.

**Caveats, stated because they matter:**
- The threshold was selected on the same 70 queries it is reported on.
  There is no held-out split at this size. Treat 0.826 as "measured on the
  set it was tuned on," not as a generalization estimate.
- "how do i register to vote" scores exactly 0.70 and still slips through.
  That one is a *knowledge base gap*, not a threshold problem — it is a
  reasonable question about South African government services that the KB
  should answer. Adding the row fixes it properly.
- Coverage is a property of the corpus, so the gate loosens by itself as
  the KB grows. That is the correct direction.

### The document trust boundary

Three layers, since none is reliable alone: named `<untrusted_document>`
fencing; delimiter neutralization inside the content so a file cannot
close its own fence (the delimiter equivalent of SQL injection closing a
quote); and a system-instruction rule stating that fenced text is data,
never instructions. Six tests in `tests/test_prompt_injection.py`.

This is mitigation, not immunity, and the docs continue to say so.

### Declined, with reasons

- **Build an 1,100-sentence 11-language evaluation set.** The right idea;
  wrong author. Generating sentences in isiNdebele, Tshivenda and siSwati
  would produce plausible-looking text that is very likely wrong, and the
  classifier would then be measured against those errors — worse than
  having no set. Like the sign-language data, this needs native speakers
  or a real corpus (NCHLT is the obvious starting point).
- **Expand the KB from 32 to hundreds of records.** Same problem: these
  are emergency numbers and government service details. Unverified
  invented facts are worse than a small honest KB.
- **Split `streamlit_app.py` and friends into modules.** Correct advice,
  wrong week — churn across every UI path five days before a presentation,
  with no functional gain.
- **Persistent vector DB / FAISS.** Solving a scale problem this project
  does not have; already documented in `vector_store.py`.

96 tests pass (17 new), `ruff check .` clean, Streamlit boots, console
runs, `evaluate.py` reports the labelled metrics by default.

---

## Round 14 — greetings written in-language (2026-08-25)

Reported: the Xitsonga voice reads the Xitsonga part of a greeting
correctly and the English part as noise.

Correct diagnosis, and it is a DATA problem, not a speech one. The rows
were bilingual by construction:

```
Greeting,avuxeni,"Avuxeni! How can Bao assist you today?"
```

One row gets one MMS voice code, so `mms-tts-tso` applies Xitsonga
letter-to-sound rules to the whole string. There is no voice code that
reads both halves correctly — the fix has to be in the CSV.

### Changes

- The four existing non-English greetings rewritten with no English.
- Greetings added for the seven official languages that had none, so all
  eleven are covered.
- New optional `Language` column marking the language each curated answer
  is already written in. Defaults to English when absent, so older
  knowledge bases and test fixtures still load.
- `KnowledgeRetriever.lookup()` returns a `KnowledgeFact` carrying that
  language; `_respond_with_fact` now serves a fact **verbatim when it is
  already in the target language** instead of round-tripping it through
  the translator. That saves a Gemini call on the first turn of every
  demo, stops the model rewriting a human-checked string, and makes the
  online and offline paths agree.

### Two traps avoided

**Row indices.** `eval/rag_eval.csv` references knowledge-base rows by
position (5, 8, 9 … 29). Inserting the new greetings next to the existing
ones would have silently repointed every one of those at the wrong row.
The new rows are appended at indices 32-37 instead, and the labelled
metrics are byte-identical before and after: TP=19 FP=4 TN=37 FN=10,
precision 0.826, recall 0.655, F1 0.731.

**Sotho-Tswana trigger collision.** Sepedi, Sesotho and Setswana all greet
with "Dumela" — the exact ambiguity the language classifier exists to
resolve, and one retrieval cannot resolve from a shared trigger. Each row
gets a distinct trigger instead: `dumela` (Setswana), `lumela` (Sesotho),
`thobela` (Sepedi). Pinned by
`test_sotho_tswana_greetings_are_distinguishable`.

### A test that caught its own author

`test_non_english_greetings_contain_no_english` flagged the Afrikaans
greeting "Goeiedag! Hoe kan ek jou help?" for containing "help" — which is
a real Afrikaans word. Afrikaans is Germanic and genuinely shares
vocabulary with English, so a marker wordlist produces false positives
there rather than catching anything. Afrikaans is now exempt from that
check, with the reasoning recorded in the test.

### Not verified: the language itself

The greetings are short, high-frequency phrases, but they were written by
an AI and **no native speaker has checked them**. Two are known to carry
real complexity:

- **Tshivenda** `Ndaa` / `Aa` are gendered — conventionally `Ndaa` from
  men, `Aa` from women. The previous row said "Nda! Aa!", i.e. both at
  once. The new row uses `Ndaa` only, which is a choice, not a neutral
  default.
- **Sesotho** orthography differs between South Africa and Lesotho
  (`o thusa` vs `u thusa`, `Dumela` vs `Lumela`). The South African
  convention was used.

101 tests pass, `ruff check .` clean, both UIs verified, retrieval metrics
unchanged.

---

## Round 15 — three bugs from one live transcript (2026-08-25)

### 1. Curated answers were being corrupted by translation

`dumela` → detected **Sesotho** → matched the **Setswana** row → sent to
Gemini to be "translated", which returned:

```
"Lumela! Nka o me/thusa jwang?"
(Kapa ka mopeleto oa Lesotho: "Lumela! Nka o me/thusa joang?")
```

Quote marks, a `me/thusa` slash artifact, and a parenthetical note about
Lesotho orthography — all of which the voice then reads aloud. `sanibonani`
came back quoted the same way. Round 14 rewrote these rows in-language
specifically so they would not need translating, and this undid it.

Rule now: **a curated answer already written in a South African language
is served verbatim**, not just when the detected language happens to
match. Two reasons, both load-bearing:

- The text was written and human-checked in that language. Round-tripping
  it through a translator discards that verification and, as observed,
  decorates it.
- The detector is least reliable exactly where this fires. "dumela" is
  shared by Sepedi, Sesotho and Setswana — so the "translation" was
  between two languages, one of which was a coin flip.

English-tagged rows (the factual ones) still translate normally. Pinned by
`test_curated_non_english_answers_are_never_machine_translated` and
`test_english_facts_are_still_translated`.

### 2. `nda` detected as English

The Tshivenda keyword was `"ndaa"`; the knowledge-base trigger and the
user both typed `nda`. The word-boundary regex `\bndaa\b` doesn't match
`nda`, so nothing fired and the English baseline (0.20) won by default.

Added the missing greeting keywords — `nda` (Tshivenda), `lumela` /
`lumelang` (Sesotho, which had only the Setswana-style `dumela`),
`goeiedag` (Afrikaans) — and expanded the knowledge-base triggers to
include common variants. `Molweni` is simply the plural of `Molo` and was
missing entirely, which is why it fell through to Gemini instead of
answering offline.

Labelled retrieval metrics unchanged: TP=19 FP=4 TN=37 FN=10, F1 0.731.

### 3. A missing voice said nothing about why

Reported: "Xitsonga is the only one that can generate a voice."

This could NOT be diagnosed from here — huggingface.co is unreachable in
this environment, so a check of which `facebook/mms-tts-*` models exist
returns "missing" for all of them including `mms-tts-eng`, which certainly
exists. Guessing would have been worse than useless.

What was fixable is that the failure was invisible: no player appeared and
nothing said why. Four quite different causes were indistinguishable:

- the speech extras aren't installed at all;
- the voice is mid-download — **each MMS language is a separate ~145 MB
  fetch on first use**, so the first request in a new language routinely
  looks broken while the second is instant (and Xitsonga being the only
  one that works is exactly the shape of a warm cache);
- no `mms-tts-<code>` exists for that language;
- no network.

`synthesize_speech` now takes an `on_error` callback, `_explain_failure`
maps the exception onto those causes, and the reason is shown under the
message ("No audio: …"). The next report will name the cause instead of
the symptom.

103 tests pass, `ruff check .` clean, both UIs verified.

---

## Round 16 — external code review, verified claim by claim (2026-08-25)

An 11-point architecture review. Two claims were wrong; the rest were
checked before being acted on.

### Wrong, and worth recording

| Claim | Reality |
|---|---|
| `csv` files opened with `newline=" "` (a literal space) — "probable bug" | **False.** All three call sites use `newline=""` correctly. The reviewer hedged on this one, which was the right instinct |
| `download_sign_model.py`, `collect_sign_data.py`, `train_sign_classifier.py` and `sign_phrases.csv` "don't exist yet" | **False.** All four exist. The review then supplied replacements for scripts already in the tree |

### Verified and fixed

**1. Failed voice models were retried on every message.** The best catch in
the review. `load_tts_model` used `functools.cache`, and **`functools.cache`
does not cache exceptions** — only successful returns are stored. So a
language whose model could not be fetched re-ran `from_pretrained` on
*every single message*: a network round trip that 404s or times out,
before giving up, every turn, plus a repeating traceback.

Demonstrated: 5 requests → 5 download attempts. Now 1.

This is why "the voices don't work" and "messages are slow" were the same
bug. Replaced with explicit dicts that remember successes *and* failures,
plus `reset_tts_cache()` so a language that failed on a dead network can be
retried without restarting.

**2. Telemetry contradicted the latency claim.** `TelemetryFormatter.format`
called `psutil` on every log record. Measured before changing anything:

```
0.121 ms per log record x ~6 records per turn = 0.72 ms
   ... of a 2.62 ms "fast offline path"  -> 27% was the instrument
```

It also called `virtual_memory()` twice per record, and
`psutil.cpu_percent()` with no interval reports load since the previous
call — so sampling it microseconds apart inside one turn returned close to
noise. Cached on a 2-second wall-clock TTL: **0.121 ms → 0.0148 ms per
record**, and the reading is now more meaningful, not less.

**3. The orchestrator was constructed in two places.** `console_app` and
`streamlit_app` each hand-built the same six components. `orchestrator.py`
claims the two interfaces "can never again drift into implementing the
pipeline differently" — true of the pipeline, false of its construction.

New `bao/bootstrap.py` is the single composition root. The one legitimate
difference (the console has no upload mechanism) is a parameter, not a
second code path. Enforced by `tests/test_bootstrap.py`, which fails if
either UI constructs a pipeline component directly.

**4. Dead `GEMINI_API_KEY` constant.** Defined in `config.py`, imported by
nothing, while `client.py` read the environment itself. Removed, with a
test that keeps it gone.

**5. Loose `dict` annotations** tightened to `dict[str, float]` /
`dict[str, str]`.

### Sign language: the scripts exist, but one rule held only by luck

Audited against the three correctness rules the review named:

- **No train/serve skew** — `collect_sign_data.py` calls the same
  `landmarks_to_feature_vector()` inference uses. Correct.
- **No scaler** — RandomForest, scale-invariant, raw 63-dim vectors.
  Correct.
- **Class-order alignment** — `recognize()` indexes its label list with
  `argmax(predict_proba)`, whose columns follow `model.classes_`. The
  script saved `sorted(df["label"].unique())`, which *happens* to equal
  `classes_` because scikit-learn sorts string labels. Correct by
  coincidence, not construction — change how `labels` is derived and every
  prediction is silently mislabelled. **This is precisely the bug that hit
  the language detector.** Now saves `final_classifier.classes_` directly,
  with a test.

### New: `scripts/probe_tts.py`

"Only Xitsonga speaks" has at least four causes with four different fixes
(extras not installed / mid-download / no such model / no network).
Guessing between them from a description wastes time, so this measures and
prints a per-language verdict with the remedy for each. The sidebar now
also reports voices actually loaded rather than claiming availability
whenever any backend imports.

### Declined, with reasons

- **str-enums for `source` / `detection_backend`.** Reasonable, and it
  would touch every module five days before a presentation for no
  behavioural gain. The valid values are already documented at the
  dataclass and covered by tests.
- **Pydantic `BaseSettings`.** Same timing objection, plus a new runtime
  dependency in a project whose selling point is running on low-resource
  machines.
- **Splitting `streamlit_app.py`.** Correct advice, wrong week — churn
  across every UI path with no functional gain.
- **Pruning module docstrings that "carry a review log."** Partly
  disagree. Docstrings explaining *why* a rule exists ("do not simplify
  this — a previous revision did and reverted the stop-word fix") are what
  stopped these bugs recurring. Two rounds of regression came from exactly
  that kind of context being absent. They stay.
- **Auto-prewarming TTS at startup.** Would kick off a multi-hundred-MB
  download on a demo machine at the worst possible moment. `probe_tts.py`
  does it deliberately instead.
- **Streaming translated KB facts.** The translated string is one or two
  sentences and repeats are already cached; streaming would add a code
  path to save a fraction of a second on the first occurrence only.

109 tests pass (6 new), `ruff check .` clean, both UIs verified.

---

## Round 17 — self-review (2026-08-25)

No external prompt this round. Reviewed my own recent work, looking for
things I introduced or left behind. Found one real bug of my own.

### A bug I introduced in Round 12

`language_override` was recorded as **instance state on the Orchestrator**
(`self._language_was_overridden`), read later by `_speech_language()`. But
`speak()` is a separate call from `handle()` — that separation is itself a
Round 11 change — so the flag describes whichever turn ran *most recently*,
not the turn being spoken.

Reproduced:

```
signer explicitly chose : Sepedi
voice actually used     : English     <- after one unrelated turn intervened
```

A signer picks Sepedi, another turn happens before the audio is generated,
and their explicit choice is silently replaced by a detector guess — the
exact failure `language_override` exists to prevent. Not reachable through
today's UI (Streamlit calls `handle()` then `speak()` in one script run),
but latent, and `@st.cache_resource` shares one Orchestrator across reruns,
so any future async change makes it live.

Fixed by moving the flag onto `PipelineResult`, where it describes the turn
it belongs to. Two tests pin it, including one that interleaves turns.

**The general lesson, recorded because it will recur:** when a method is
split out so it can be called later, any state it reads must move with it.
Per-turn facts belong on the per-turn object.

### Tokenizer parity — closing the biggest unvalidated assumption

The reported 0.98 macro F1 was measured in the training notebook, with
Keras's own `Tokenizer` and the full Keras model. The app ships neither: a
hand-written `_KerasWordTokenizer` feeding a quantised TFLite model. If the
reimplementation disagrees with Keras anywhere, **the deployed accuracy is
not the measured accuracy, and nothing crashes to say so.**

`tests/test_tokenizer_parity.py` compares the two directly across ten
inputs chosen to hit the branches that differ between plausible
implementations (casing, filter characters, out-of-vocabulary words, words
above `num_words`, truncation, empty input). It skips without TensorFlow —
so it does nothing here and everything on a machine with
`requirements-ml.txt` installed.

**Run it on the demo machine.** It converts "our tokenizer should match"
into "our tokenizer provably matches", which is the difference between an
assumption and a result.

Six further tests need no TensorFlow and always run, pinning the
empirically-determined `post` padding and the OOV-not-dropped rule.

### The evaluation harness had no tests

`bao/evaluation.py` was at **0% coverage** — the one module whose output
becomes claims in a report. A defect there doesn't produce a visible
failure, it produces a confident wrong number in front of judges.

Testing it found a hole: a row whose retrieval returned `None` was skipped
by the loop but still counted in `total`, so it lowered the reported
accuracy while appearing in neither `correct` nor `misses`. The report
could not be reconciled with its own explanation. Now recorded as the miss
it is.

### Housekeeping

- **Stale numbers in the docs.** Adding six greeting rows left "32 rows"
  in the README and three places in `docs/TECHNICAL_REPORT.md` while the
  knowledge base held 38. Fixed, plus `tests/test_docs_match_code.py` so
  prose figures can't drift again. That guard fired twice on unrelated
  numbers ("2/40" from a threshold table, "20/20" from a detector
  spot-check) before being scoped to the two forms that unambiguously
  encode corpus size — a guard that cries wolf gets deleted.
- **Dead aliases removed.** `SecurityGuardrails.validate_query/validate/
  is_safe` and `KnowledgeRetriever.retrieve/find_match` had **zero call
  sites**. They were kept for "older call sites" that don't exist in a
  fresh submission.
- **`test_rag_edge.py` moved** to `scripts/smoke_test_rag.py`. It was
  misleading in both directions: pytest never collected it
  (`testpaths = ["tests"]`), so it looked like a test that silently never
  ran — while a bare `pytest` from another directory *would* have
  collected a script that prints rather than asserts.

126 tests pass (16 new), `ruff check .` clean, coverage 56% → 61%, both
UIs verified, `evaluate.py` reports 38/38 and F1 0.731.

---

## Round 18 — a scorecard for the propose/evaluate loop (2026-08-26)

A proposed workflow: Claude proposes a change, another model evaluates
independently, the score change decides accept or reject.

The loop is sound in shape. The risk is entirely in the word "score".

### What the current eval set can and cannot resolve

`eval/rag_eval.csv` has 70 queries — 40 negatives, 15 exact, 15
paraphrase. That means precision is measured over 23 items and recall over
29:

```
precision 19/23 = 0.826   95% CI [0.629, 0.930]   width 0.302
recall    19/29 = 0.655   95% CI [0.473, 0.801]   width 0.327
```

One query flipping moves precision 4.3 points. A change must flip roughly
**7-9 of the 70** before the movement is distinguishable from chance. A
loop that accepts on "F1 0.731 → 0.744" would be ratcheting on noise, and
would spend its iterations fitting the eval set rather than improving Bao.

Worse, `min_query_coverage` was already tuned on that same file (Round 13).
It is a development set, not a holdout. Repeated accept/reject cycles
against it will keep improving the number and stop improving the system,
with no signal that it has happened.

### `scripts/scorecard.py`

One reproducible artifact covering everything measurable, with Wilson
intervals so score changes come with error bars:

```
python scripts/scorecard.py --save baseline.json
# apply a change
python scripts/scorecard.py --compare baseline.json
```

`--compare` refuses to call a difference an improvement unless the
confidence intervals separate, and applies hard gates (failing tests,
failing lint, dropped self-consistency) that reject regardless of score.
It is built to say "no detectable change" often, because that is usually
the truth.

Verified against two realistic cases:

| Proposed change | Naive reading | Scorecard verdict |
|---|---|---|
| `min_query_coverage` 0.70 → 0.65 | "precision fell 6.6 points — regression" | **no evidence** — intervals overlap at n=23 |
| An unrelated edit that breaks lint | "score unchanged — accept" | **REJECT** — hard gate |

### Limits, stated so they are not assumed

- It cannot detect overfitting to `rag_eval.csv`. A genuine holdout must be
  written by someone who is not proposing the changes, and looked at once.
  **Claude must not author it** — authoring the set you are scored against
  defeats the purpose regardless of intent.
- It cannot score what has no ground truth here: whether the greetings are
  linguistically correct, whether the shipped tokenizer matches the one the
  model was trained with, whether a voice sounds right on the demo machine.
  Those are the project's largest open risks and this loop will not touch
  them.

The productive shape of cross-model review is the one already used in
Rounds 13 and 16: the other model as an adversary generating *claims*, each
verified against the code. That found a real bug Claude had missed
(`functools.cache` not caching exceptions) and two claims that were false.
Claims can be checked. Scores at n=23 mostly cannot.

127 tests pass, `ruff check .` clean.

---

## Round 19 — hard gates, and the rule that a skip is not a pass (2026-08-26)

The proposed workflow, revised: hard gates → adversarial review →
statistical evaluation → accept/reject. Correct ordering. Gates before
opinions, opinions before scores, scores last and with error bars.

`scripts/gates.py` implements the five gates. Exit codes: `0` pass, `1`
reject, `2` **blocked** — a third state that matters.

### The hole this closes

`pytest -q` prints `126 passed, 17 skipped`. Green. **Ten of those skips
are the tokenizer parity tests**, which skip when TensorFlow isn't
installed — so on a machine without the ML extras, the single most
important question in the project (does the shipped tokenizer agree with
the one the model was trained on) goes unanswered while the gate layer
reports success.

That is the same failure mode this project has hit twice: the heuristic
detector silently standing in for the ML classifier, and speech silently
returning no audio. Nothing failed; something quietly did less than it
appeared to. So a gate that skips reports `UNVERIFIED`, exit 2 — not
rejected, not cleared. `--allow-unverified` exists so accepting that risk
is always a deliberate act.

### Gates that earn their place

Verified each catches something the others miss:

| Simulated regression | Tests | Gates |
|---|---|---|
| Rename `PipelineResult.audio_mime` (Streamlit reads it by name) | **126 passed** | **FAIL** — API contracts |
| Similarity threshold 0.25 → 0.9 | 11 failed | FAIL — Runtime, with the branch named |
| Lint-breaking edit elsewhere | passed | FAIL — Lint |

The first row is the argument for the contract gate: Streamlit isn't unit
tested, so a rename it depends on surfaces during the demo and nowhere
earlier.

### A gate I had to fix after testing it honestly

The runtime gate originally probed three branches with exact trigger words.
Raising the live coverage threshold to 0.99 — which rejects most real
questions — **passed every check**, because exact triggers survive almost
any threshold. Branch coverage proved the paths execute and said nothing
about whether retrieval still retrieves.

Added a collapse floor: true positives on the labelled set must stay at or
above 14 (current 19). Deliberately a floor, not a metric — gates answer
"did something collapse", the statistical layer answers "is this better".
Conflating them turns a gate into a hill to climb, which is exactly what
the confidence intervals in `scorecard.py` exist to prevent.

Worth noting what the floor correctly does *not* do: coverage 0.99 drops
true positives 19 → 16, which is real degradation but not collapse. The
gate passes it and hands it to the statistical layer, which is the right
division of labour.

### Running order

```
python scripts/gates.py            # exit 0 -> proceed; 2 -> blocked, not rejected
#   adversarial review here — other models generate CLAIMS, each verified
python scripts/scorecard.py --compare baseline.json
```

126 tests pass, `ruff check .` clean.

---

## Round 20 — retrieval integrity as its own gate (2026-08-26)

The design converged. Gate 3 split out from gate 2, and the gate order now
reads cause-before-symptom: a broken contract explains a runtime failure,
and a runtime failure explains a retrieval failure — so the first FAIL from
the top is usually the cause rather than a consequence.

```
[  ok  ] API contracts        4 contracts intact
[  ok  ] Runtime integrity    4 branches exercised offline
[  ok  ] Retrieval integrity  self-consistency 38/38, 19 TP >= floor 14
[  ??  ] Tokenizer parity     skipped — TensorFlow not installed
[  ok  ] Tests                126 passed, 17 skipped
[  ok  ] Lint                 ruff check . clean
```

`Retrieval integrity` checks two things the others cannot: that every
knowledge-base row still retrieves its own question (a corpus or index
break), and that labelled questions still get answered above a floor.

### The split was earned, not cosmetic

Verified against two configurations, and the boundary between them is the
whole point:

| Change | True positives | Gate | Why that is correct |
|---|---|---|---|
| coverage 0.70 → 0.99 | 19 → 16 | **PASS** | real degradation, but not collapse — hand it to the statistical layer, which has error bars |
| similarity 0.25 → 0.9 | 19 → 0 | **FAIL** | collapse; no scorecard needed to know it |

A gate that failed the first row would be a metric, and a metric in a gate
is a hill to climb — the exact failure the confidence intervals in
`scorecard.py` exist to prevent. Gates answer "did something collapse".
The scorecard answers "is this better", and usually answers "no evidence",
which is usually the truth.

### The pipeline is now complete

```
python scripts/gates.py                          # 0 proceed / 2 blocked / 1 reject
#   adversarial review — models produce CLAIMS; each is verified here
python scripts/scorecard.py --compare baseline.json
```

Nothing further is needed on the process. What remains before the
presentation is not process work — see the standing list at the end of
Round 17 and the notes in `BAO_ARCHITECTURE_FOR_PRESENTATION.md`. Gate 4
going green on the demo machine is worth more than any further refinement
of this loop.

126 tests pass, `ruff check .` clean.

---

## Round 21 — two bugs from one live session (2026-08-26)

### 1. Badge and answer disagreed on screen

```
dumela
Language: Sesotho (keyword match) · source: knowledge_base
Dumela! Nka go thusa jang?          <- the SETSWANA row
```

The detector said Sesotho. Retrieval returned Setswana. Both were doing
their job: "Dumela" is the greeting in Sepedi, Sesotho *and* Setswana, so
similarity cannot separate them — the information needed is not in the
query. It is in the classifier's output, which retrieval never saw.

Round 14 had "solved" this by giving each row a distinct trigger
(`dumela` / `lumela` / `thobela`). That hid the ambiguity rather than
handling it, and this session is what it looked like from the outside.

Now all three rows answer to the shared greeting, and
`lookup(query, prefer_language=...)` selects among the candidates using the
detected language. The classifier does the disambiguation it exists for:

```
dumela   -> detected Sesotho   | Lumela! Nka o thusa jwang?
thobela  -> detected Sepedi    | Thobela! Nka go thuša ka eng?
```

This also improves the demo story, because the code now matches it: shared
vocabulary resolved by the LSTM, not by word overlap. Labelled retrieval
metrics unchanged (TP=19 FP=4, F1 0.731).

### 2. My own error classifier missed the error it was written for

The reported message:

```
Voice for Sesotho (mms-tts-sot) failed: facebook/mms-tts-sot could not be
loaded: facebook/mms-tts-sot is not a local folder and is not a valid
model identifier
```

That is transformers' message for a missing repository — and it contains
neither "404" nor "not found", which is all `_explain_failure` matched on.
So a **definitive** "this model does not exist" was reported as a generic
failure, leaving the user to wonder whether retrying would help. It would
not. Fixed, with the verbatim string pinned in a test.

### The finding itself: Sesotho has no offline voice

`facebook/mms-tts-sot` is not a published model. This is now established
fact rather than speculation, and it is the first hard data point on the
"only Xitsonga speaks" question: at least one of the eleven has **no
offline voice at all**, and no amount of code changes that.

Which of the remaining seven MMS languages exist could not be determined
from this environment — searching turned up no listing, and a direct check
is impossible here. `python scripts/probe_tts.py` answers it definitively
on the demo machine, per language, in one run.

For any language with no MMS voice, the options are: use edge-tts where a
locale exists (English, Afrikaans, isiZulu only), or state plainly in the
presentation that offline speech covers a subset. The second is a
perfectly good answer — MMS covers ~1100 languages and South Africa's
smaller ones are exactly where coverage thins out. That is a finding about
the state of speech technology for these languages, which is on-topic for
this project rather than an embarrassment.

130 tests pass, `ruff check .` clean, gates green except the unverified
tokenizer parity.

---

## Round 22 — the detector proposal, adjudicated by adding a slot (2026-08-26)

A teammate proposed replacing the TFLite LSTM with TF-IDF character
n-grams + a linear classifier. The objection raised: it replaces a
validated component with an unvalidated one, and requires changing code
around it. Both positions have merit; the claims were checked.

### Claims verified

| Claim | Verdict |
|---|---|
| "Zero source code rewrite" | **False as stated.** 14 files reference `TFLITE_CLASS_ORDER`, `_KerasWordTokenizer`, `has_tflite_support`, `classifier_model_path`, `tokenizer_config_path` or `detection_backend`, including `benchmark_bao.py`, `gates.py`, `evaluate.py` and three test files. It IS a clean drop-in for a *new backend* — `LanguageDetector` is a one-method ABC — but not for a *replacement* |
| ".pkl stays under 15 MB, avoiding GitHub's 25 MB limit" | **Partly right, wrong reason.** No current artifact exceeds 25 MB: the TFLite model is 12.7 MB and `tokenizer_config.json` is 14.3 MB. Total runtime footprint is 27 MB, which is a Streamlit Cloud memory argument, not a GitHub one. The 25 MB figure is the *web upload* limit; `git push` allows 100 MB and LFS is already configured |
| "Same dataset, same LabelEncoder" | **Credible and important** — unlike the 16-language model from Round 21's predecessor, this is not synthetic data |
| "Better on out-of-vocabulary input" | **Well-founded.** A live session had a user type `axuxeni` for `avuxeni`; the word-level path could only emit `<OOV>` and fell through to English |
| "Evaluated on held-out test sets with full metrics" | **Unverified here.** No artifact was provided. This is exactly the kind of claim the adversarial-review step exists to check rather than accept |

### Resolution: a third backend, not a replacement

`SklearnLanguageDetector` added alongside the heuristic and TFLite
backends. It satisfies the same one-method ABC, reports
`backend="sklearn"`, reads its labels off the fitted estimator
(`classes_`, for the same class-order reason as the sign classifier), and
degrades to `backend="none"` when no bundle is present — so adding it
changes nothing until a model is actually trained.

`scripts/compare_detectors.py` runs every available backend against the
same held-out CSV and reports macro F1, per-language F1, a confusion
matrix, and a robustness probe that corrupts every 7th character — testing
the specific claim made for character n-grams. Verified end to end with a
throwaway 12-sentence model (since deleted).

### Why not simply decide

The two models are good at different things and the ordering is not
obvious from argument:

- Character n-grams see inside words, so typos and unseen vocabulary
  survive.
- The LSTM reads word ORDER, which a bag of n-grams discards — and word
  order was the stated justification for choosing it over a bag-of-words
  model, specifically for the Sepedi / Sesotho / Setswana case that is the
  hardest in the set and the project's headline claim.

Which dominates on NCHLT is a measurement. The comparison script makes it
one, on the held-out 10% both models can share.

The script also prints a caution when the margin is under 0.01: on a
difference that small, prefer the component that is already validated and
already documented. Four days from a presentation, "we evaluated both and
kept the LSTM" and "we evaluated both and switched" are equally strong
answers — but "we switched" without the comparison is not.

**One consequence worth weighing separately:** the brief asks for graphs
generated during training. A linear classifier has no epochs and produces
no training curves. Replacing the LSTM would forfeit that deliverable,
which is not a modelling argument but is a marks argument.

130 tests pass, `ruff check .` clean, gates unchanged.

---

## Round 23 — the 14-language proposal is a data question, not an architecture one (2026-08-26)

Clarified: the TF-IDF / linear-classifier proposal is the *vehicle* for
adding 14 more African languages, not an end in itself. That reframes
Round 22 entirely — the architecture debate is a distraction from the
question that decides it.

**NCHLT supplies 636,123 real sentences for 11 South African languages.
What supplies real sentences for the other 14?**

That is the whole issue. Character n-grams versus an LSTM barely matters
next to it, and neither architecture can rescue a corpus that does not
contain the languages it claims to.

### Why this is urgent rather than theoretical

A 16-language classifier was already submitted to this project. Its
artifacts showed 2,400 documents across 16 classes (exactly 150 each),
each language's name appearing in every one of its own rows, and a total
vocabulary of 592 words. Training examples read:

```
sample text sentence for language classification in swahili sequence 42
```

It reported near-perfect validation accuracy and recognised **zero** words
of a real Swahili, Yoruba, Hausa or French sentence.

**The trap in merging.** Combine that with NCHLT and the fake classes are
trivially separable, so they score near 1.00 and pull macro F1 *up*. A
25-language model would report a HIGHER number than the current
11-language one while being substantially worse — and nothing anywhere in
the pipeline would warn about it. The metric improves as the system
degrades. That is the most dangerous shape a result can take in a graded
submission.

### `scripts/audit_corpus.py`

Run on any corpus before training, and on each language source separately
before merging. Six checks; three can reject:

| Check | Level | Catches |
|---|---|---|
| Label leakage | FAIL | a class's own name inside its own rows |
| Cross-class overlap | FAIL | one English scaffold reused for every language |
| Templating | FAIL | mass-identical sentence openings |
| Vocabulary variety | WARN | lots of text, little lexical range |
| Lexical richness | WARN | type-token ratio |
| Class balance | WARN | suspiciously exact per-class counts |

Verified both directions. On a reconstruction of the known-bad corpus all
three hard checks fire. On a varied control they stay silent and only the
soft checks comment.

**A design flaw the control exposed.** The vocabulary check started as an
absolute floor, which failed a small but genuine corpus. It is now
relative — a class is only flagged when it has plenty of text *and* almost
no variety, which is what templating actually produces. Testing a checker
against data it should pass is as necessary as testing it against data it
should fail.

The script cannot prove a corpus is genuine. It reliably catches the
specific ways generated data gives itself away, and that is the claim it
makes.

### The recommendation

Ask for the 14 languages' data and run the audit on it. If it passes, the
expansion is real and worth doing properly — after the presentation, with
a proper split. If it fails, that is settled without anyone having to
argue about tokenizers.

Four days out, 11 languages backed by NCHLT with a documented 0.98 macro
F1 is a stronger submission than 25 languages where 14 have unverifiable
provenance. The pan-African expansion is an excellent roadmap slide.

130 tests pass, `ruff check .` clean.

---

## Round 24 — pan-African detector integrated, switched off (2026-08-30)

The group's 14-language classifier (TF-IDF char n-grams + logistic
regression, disjoint from the 11 South African languages) is now wired in
as a **fallback backend behind a flag that defaults to false**.

### Why flag-off rather than simply on

Adding a second detector changes the detection path for EVERY message, not
just the fourteen new languages. Detection feeds greeting lookup, retrieval
language preference and voice selection — so a misrouted isiZulu message
takes the wrong greeting row and the wrong voice with it. Shipping it
disabled means the model and the code are present and testable while
behaviour is bit-for-bit unchanged until somebody enables it and checks.

`tests/test_pan_african.py::test_default_build_uses_a_plain_detector`
asserts the orchestrator does not even hold a composite by default — the
guarantee is structural, not behavioural coincidence.

### The arbitration rule, and why not "most confident wins"

The two models are trained on different corpora and are not calibrated
against each other, so their probabilities are not comparable. Measured on
the shipped bundle:

```
South African / English input  ->  16-23%  (wrong label)
genuine pan-African input      ->  37-95%  (right label)
```

Those bands do not overlap, so: the primary is always asked first and its
answer stands unless it falls below 0.5; a secondary answer is only
accepted above 0.35. Erring means falling back to current behaviour rather
than guessing. The margin comes from eleven probe inputs, not a held-out
set, and is documented as such.

Without this rule, "hello how can you help me" scores 22.6% as Nigerian
Pidgin — English would have been relabelled.

### A real bug caught while building it

`classifier.classes_` on the group's model is `[0, 1, 2, ...]`, not
language names — it was trained on **encoded** labels. Reading names off
`classes_` directly (which `SklearnLanguageDetector` previously did as a
fallback) yields integers, so the app would have reported languages called
"0" and "7". The correct name for column *i* is
`encoder.classes_[classifier.classes_[i]]`, composing both rather than
trusting either.

This is the third time this exact class-order assumption has bitten this
project (the TFLite detector, the sign classifier, now this). The loader
now refuses a bundle whose labels are all digits, and a test pins it.

### Verified

- Gates green with the flag off (tokenizer parity still UNVERIFIED here).
- With the flag temporarily on: all eleven SA languages still resolve
  through the primary detector, backend `heuristic`; pan-African input
  routes to `sklearn` and is labelled correctly.
- `test_every_sa_language_is_covered_by_the_probes` failed on first run —
  Setswana and siSwati had no probe. Worth recording that siSwati shares
  nearly its whole keyword set with isiZulu, so under the heuristic
  backend it is only reachable through distinctive words like "kantsi".
  A limitation of the keyword fallback; the ML classifier separates them.

### What integration does NOT provide

Detection and a Gemini reply in the right language. Nothing else — the new
fourteen have **zero** voice codes, speech-to-text codes, knowledge-base
rows and greetings. A Yoruba message gets detected, labelled and answered
online, with no audio, no verified answer and nothing offline. Real, and
not the same feature as the eleven. The report should say so rather than
let "25 languages" imply parity.

### To enable

Set `enable_pan_african = true` in `config.toml`, then run
`python scripts/gates.py` and check the eleven SA languages still detect
correctly before demoing.

150 tests pass (20 new), `ruff check .` clean.

---

## Round 25 — sign language removed (2026-08-30)

Removed at the group's request. Recorded here so the decision is
documented rather than looking like drift.

### What went

Nine files deleted:

```
bao/services/sign_language.py       data/sign_phrases.csv
scripts/collect_sign_data.py        scripts/train_sign_classifier.py
scripts/download_sign_model.py      docs/SIGN_LANGUAGE_GUIDE.md
tests/test_sign_language.py         tests/test_sign_phrases.py
requirements-sign.txt
```

Plus every reference: three `Settings` path properties, the
`services/__init__` export, the Streamlit sign panel and its sidebar
section (and the `numpy`/`PIL` imports only it used), `init_system`'s
return shape, and the sections in README.md and docs/ARCHITECTURE.md.

Test count 150 -> 136. No orphan references remain outside the
explanatory note in ARCHITECTURE.md.

### What stayed, and why

`Orchestrator.handle(language_override=...)` was introduced for sign input
and outlived it. Kept, with its docstring corrected so it no longer cites a
feature that does not exist: it is the mechanism a "reply in ..." selector
would use, and it is how the pan-African languages become reachable when
the detector is unsure. Currently exercised by tests and `gates.py`, not by
either interface — flagged in the docstring for removal if that is still
true at the next tidy-up.

### Documentation now states the real reason

The prototype was never trained, and training it was not the only obstacle:
most SASL vocabulary is *movement* rather than a static pose, and its
grammar is partly carried on the face through non-manual markers that hand
landmarks cannot see. Usable recognition needs continuous video and a
sequence model; usable translation additionally needs a SASL-to-spoken
grammar mapping, since SASL is not signed English.

SASL became South Africa's 12th official language in 2023, so it remains a
legitimate long-term direction — which is how README.md and
docs/ARCHITECTURE.md now frame it. Claiming it on the strength of an
untrained hand-shape classifier would have overstated the system.

### Also fixed while in here

`joblib` was imported by name in `language_detector.py` but declared in no
requirements file — it worked only because scikit-learn pulls it in.
Declared explicitly. `scikit-learn` floor raised to 1.6.0 so the
pan-African bundle (exported under 1.6.1) loads without a version warning.

### Verified

Gates green (tokenizer parity still UNVERIFIED here), 136 tests pass,
`ruff check .` clean, Streamlit boots, console runs a full turn.

---

## Round 26 — a wrong diagnosis, corrected (2026-08-30)

`scripts/probe_tts.py` was run on the demo machine and produced the
evidence that overturns Round 21's conclusion.

### What I got wrong

Round 21 reported, definitively, that `facebook/mms-tts-sot` "is not a
published model" and that this "cannot be fixed in code". That was stated
as a finding. It was a guess.

The probe output shows why:

```
mms-tts-afr, zul, xho, sot, tsn, nso, ven, ssw, nbl  ->  401 Unauthorized
mms-tts-tso                                          ->  307 -> 200 OK, loaded
```

**401 is not 404.** Nine repositories refused the request; a tenth served
it and loaded fine. The log even carries the explanation:

```
You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits.
```

`transformers` renders an absent repo, a gated repo and a refused request
with the identical sentence — "is not a local folder and is not a valid
model identifier". Round 21 read that sentence as proof of absence and
built a confident conclusion on it, then put that conclusion in the
project's presentation guidance.

### Root cause

The error classifier matched on the *message text* rather than the *status
code*, because the message was all it was shown. Both `speech.py` and
`probe_tts.py` now branch on 401 / 429 / 404 separately, and — where no
status code is present — say the cause is ambiguous instead of picking one.

Three tests replace the one that pinned the wrong behaviour, including
`test_ambiguous_message_admits_it_is_ambiguous`.

### The general lesson

An error message is evidence about what a library chose to print, not
about what happened. When a library collapses several causes into one
string, the honest report is "this could be either", and the fix is to
find a signal that discriminates — here, the HTTP status sitting in the
same exception the whole time.

This is the second time in this project that a confidently-worded
diagnosis proved wrong (the first was assuming the sign classifier's label
order was safe because it happened to agree). Both were cases of reading a
coincidence as a guarantee.

### Also surfaced by the probe

`edge-tts installed : False` — so the three languages with real South
African voices (en-ZA, af-ZA, zu-ZA) have no voice either, for a reason
entirely unrelated to Hugging Face. The probe's remedy section now says so
explicitly, since `pip install edge-tts` needs no account and fixes three
languages immediately.

Also worth noting: the probe tests the *configured* MMS code, not the
language. English maps to `afr` in config.toml, so `mms-tts-eng` — which
certainly exists — was never tried.

138 tests pass, `ruff check .` clean.

---

## Round 27 — voice coverage, finally established (2026-08-30)

`probe_tts.py` re-run with `HF_TOKEN` set. Every 401 became a **404**. The
question is now closed on evidence rather than inference.

### Verified coverage: 4 of 11

| Language | Backend | Voice | Offline |
|---|---|---|---|
| English | edge-tts | en-ZA-LukeNeural | no |
| isiZulu | edge-tts | zu-ZA-ThembaNeural | no |
| Afrikaans | edge-tts | af-ZA-WillemNeural | no |
| Xitsonga | MMS | mms-tts-tso | **yes** |

No voice, confirmed 404 with an authenticated request: isiXhosa (`xho`),
Sesotho (`sot`), Setswana (`tsn`), Sepedi (`nso`), Tshivenda (`ven`),
siSwati (`ssw`), isiNdebele (`nbl`).

Round 21's conclusion was correct in the end, but it was reached on
evidence that could not support it. A 401 was read as a 404 for nine days.
The finding is only sound now because a token turned an ambiguous refusal
into an unambiguous absence — which is exactly the discrimination Round 26
added.

### A dead mapping this exposed

`config.toml` routed English through `afr` to give it a local accent.
`mms-tts-afr` returns 404, so that mapping silently removed English from
the offline path as well — a language with a working model (`mms-tts-eng`)
had no offline voice because of a config line intended to improve it.
Changed to `eng`.

Afrikaans genuinely has no MMS model, so it is online-only via edge-tts.

### `scripts/probe_mms_codes.py`

Before "no voice exists" goes in a report, it is worth ruling out that MMS
simply uses a different code — its 1107 languages mostly follow ISO 639-3
but not always. This script tries macrolanguage codes, 639-2/B forms and
dialect variants per language and reports which resolve. It only asks
whether a repo exists; it downloads nothing.

If everything 404s, the finding is evidenced. If something resolves, it
prints the `config.toml` lines to change.

### What this is worth saying out loud

Seven of South Africa's eleven official languages have no open
text-to-speech model in the largest multilingual speech project that
exists. Xitsonga does; isiXhosa, with roughly 8 million speakers, does not.

That is a real, citable finding about the state of speech technology for
these languages, and it belongs in the report as a result rather than a
limitation. It is also precisely the digital exclusion this project was
built to address, measured rather than asserted.

138 tests pass, `ruff check .` clean.

---

## Round 28 — voice fallback tiers (2026-08-30)

A proposed four-tier fallback design, implemented — with three rows of it
removed after checking them against the probe results.

### Implemented

| Tier | Behaviour | Default |
|---|---|---|
| 1 | Native voice — English, isiZulu, Afrikaans (edge-tts); Xitsonga (MMS) | always |
| 2 | Closely-related language's voice | **off** |
| 3 | Speak in English | **off** |
| 4 | Text only, stated plainly | default |

`speech.resolve_voice_language()` returns `(language_to_speak, note)`. The
note is `None` for a native voice and a short user-facing sentence
otherwise, shown under the message. **A substitution the listener is told
about is a fallback; one they are not told about is a misrepresentation** —
which is why both fallbacks are opt-in and neither is ever silent.

The sidebar now lists per-language coverage: 4 native, 3 related-capable,
4 text-only.

### Three proposed substitutions rejected

The proposal's Tier 2 table had seven rows. Four survived checking:

| Proposed | Verdict |
|---|---|
| isiXhosa, siSwati, isiNdebele → isiZulu | **Kept.** All Nguni; shared orthography including the click letters c/q/x, so the isiZulu model has seen these characters |
| Sesotho → Setswana/Sepedi | **Dropped — unimplementable.** All three return 404. There is no voice anywhere in the Sotho-Tswana family to borrow |
| Setswana → Sesotho/Sepedi | **Dropped.** Same |
| Sepedi → Setswana/Sesotho | **Dropped.** Same |
| Tshivenda → Xitsonga | **Dropped — linguistically unfounded.** Venda is its own branch of Bantu; Tsonga is Tswa-Ronga. Venda orthography also uses dental diacritics (ṱ ḓ ṋ ḽ) absent from Tsonga, and a character-level VITS model mangles characters it never saw. "Ndi khou pfa vhuṱungu" contains one |

Three of seven rows would have substituted a voice into a family where no
member has one — a table written from linguistic intuition without checking
it against the probe output that was already available.

The rejections are pinned by tests
(`test_sotho_tswana_has_no_related_fallback`,
`test_tshivenda_does_not_borrow_the_xitsonga_voice`) with the reasoning in
the docstrings, because a rejected option that isn't recorded gets proposed
again.

### Consistency with an earlier decision

Round 10 rejected routing English through the Afrikaans MMS model because
it applies one language's letter-to-sound rules to another's words —
mispronunciation, not an accent. Tier 2 is the same operation, so it gets
the same treatment: allowed only within a closely-related family, opt-in,
and disclosed.

159 tests pass (21 new), `ruff check .` clean, gates green, Streamlit boots.

---

## Round 29 — the classifier could not be installed (2026-08-30)

`pip install -r requirements-ml.txt` failed:

```
ERROR: Could not find a version that satisfies the requirement
       tensorflow>=2.15.0 (from versions: none)
```

The machine runs **Python 3.14** (`C:\Python314`). TensorFlow publishes no
wheels for 3.14; support stops at 3.12/3.13. So every "install
requirements-ml.txt and the badge will say ML model" instruction given over
the past week was un-actionable, and the repeated `keyword match` was an
environment constraint rather than an oversight.

### The dependency was far larger than the need

`language_detector.py` uses exactly one thing from TensorFlow:
`tf.lite.Interpreter`. Full TensorFlow is roughly 600 MB for that.

`ai-edge-litert` is Google's standalone LiteRT runtime — the same
interpreter, a fraction of the size, and packaged for current Pythons. The
import chain is now:

1. `ai_edge_litert.interpreter.Interpreter` — preferred
2. `tensorflow.lite.Interpreter` — fallback where TF already exists
3. neither → heuristic backend, as before

`tflite_runtime_name()` reports which is in use, so a missing classifier is
diagnosable rather than mysterious, and `probe_tts.py` prints it.

`requirements-ml.txt` now installs `ai-edge-litert` and documents full
TensorFlow as optional — needed only for `convert_model.py` and
`tests/test_tokenizer_parity.py`, both of which genuinely require Keras.

### A latent crash this exposed

The Keras fallback path called `tf.keras.…` at line 307. With the
module-level `import tensorflow as tf` gone, that name is unbound in a
LiteRT-only environment — a `NameError` on a rarely-taken branch. Now
imported locally with a clear message.

### Note on requirements-ml.txt's own docstring

It already recorded that the pre-refactor `pyproject.toml` declared
`ai-edge-litert` while the code imported `tensorflow`, and treated the
declaration as the error. It was the better choice all along; the code was
what needed changing.

159 tests pass, `ruff check .` clean, gates green apart from the parity
gate, which now explains that it needs Keras specifically rather than
implying the classifier does.

---

## Round 30 — the edge backend never cleaned its input (2026-08-30)

Reported from a live session: a Xitsonga reply was spoken **with the
markdown asterisks read aloud**.

### Cause

A missing step, not a broken one. `clean_text_for_speech()` was called from
inside `split_into_sentences()` — and only the MMS branch calls that,
because edge-tts handles long text server-side and needs no chunking. So
the edge branch passed **raw Gemini markdown straight to the voice**:

```
edge receives : 'Yi katsa...\n\n1. **Differential Calculus (...):** Yi lavisisa'
should receive: 'Yi katsa... 1. Differential Calculus ... Yi lavisisa'
```

Introduced in Round 10 with the edge backend. Every English, Afrikaans and
isiZulu reply since has been spoken with its markdown intact — which is
every reply with a working voice on the demo machine, since Xitsonga is the
only MMS voice that loads.

The lesson is the same shape as Round 17's: **cleaning was coupled to
chunking**, so a backend that legitimately skipped chunking silently
skipped cleaning too. A transformation every consumer needs does not belong
inside a step only one of them takes.

### Also widened

The token list now covers `__`, `` ``` ``, `_`, `|`, `>`, `~~` and leading
list bullets (`-`, `•`, `–`) — all of which Gemini emits and none of which
were handled. The apostrophe is still deliberately preserved: it is part of
the orthography in Xitsonga (`ematshan'weni`) and isiZulu, so stripping it
changes the word rather than tidying it. Pinned by a test.

Nine new tests, including one asserting the edge path receives cleaned
text.

168 tests pass, `ruff check .` clean, gates green apart from the parity
gate.

### Still outstanding

The same reply was spoken by an **English** voice, because the heuristic
detector scores Xitsonga text as English at 0.90 — the English keyword
list contains `"hi"`, which is a very common Xitsonga preposition. That is
Round 29's problem, and it is fixed by installing `ai-edge-litert` so the
LSTM runs instead of the keyword table.

---

## Round 31 — the classifier runs; a weak guess must not steer the reply (2026-08-30)

`ai-edge-litert` installed. The badge reads **ML model** for the first time
— the LSTM is finally the thing answering, not the keyword table.

### What the first live session showed

| Confidence | Query | Detected | Reply language correct? |
|---:|---|---|---|
| 35% | `vuxeni` | siSwati | **no** (Xitsonga) |
| 37% | `explain calu=culus in ndebele` | English | yes |
| 42% | `what is car in xhosa` | Afrikaans | **no — and the answer came back in Afrikaans** |
| 64% | `what is car in swati` | English | yes |
| 72% | `what is car in ndebele` | English | yes |
| 83% | `explain calculus in xitsonga` | English | yes |
| 91% | `explain maths in zulu` | English | yes |

Everything at or above 64% correct; the misfires sat at 35-42%.

The classifier is not at fault. It was trained on NCHLT **news sentences**
and is being asked short English imperatives — out of distribution, exactly
as with the pan-African model, which scores 18% on "bonjour" and 54% on
"bonjour comment allez vous".

### The real damage, and the fix

Detection does not only label a badge; it chooses **the language Gemini
answers in**. So a 42% guess turned an English question *about* isiXhosa
into an Afrikaans answer. A wrong badge is cosmetic. A whole answer in the
wrong language is not.

`min_detection_confidence` (default 0.5): below it the detection is still
**reported honestly** and simply not acted on — the reply is generated in
English, and the badge says so ("Afrikaans (ML model, 42%) — too unsure to
use, replying in English").

`PipelineResult` now carries `detected_language` and `reply_language`
separately, because "what the model said" and "what we did about it" are
different claims and collapsing them is what hid this.

An explicit `language_override` bypasses the floor entirely — it is a
stated choice, not a guess, and has no confidence to be unsure about.

### Note

The threshold is set from seven live observations, not a held-out
calibration set. It is a floor chosen to sit above every observed misfire
and below every observed success, and it is one config line to change.

171 tests pass, `ruff check .` clean, gates green apart from the parity
gate.

---

## Round 32 — external review, verified claim by claim (2026-08-30)

A 15-point review. Every checkable claim was confirmed against the code
before acting.

### Confirmed and fixed

| Claim | Verified | Fix |
|---|---|---|
| `docs/TECHNICAL_REPORT.md` says threshold 0.15; config says 0.25 | line 65 | corrected, and the coverage gate documented alongside it |
| Report calls the heuristic the default and ML "experimental" — the reverse of `bootstrap.py`'s `prefer_ml=True` | line 25 | corrected to "TFLite LSTM, default; heuristic as fallback" |
| Raw user queries logged on error paths | 3 sites in `retriever.py` | now logs query **length**, with the reasoning recorded |
| `.keras` / `tokenizer.pickle` are training artifacts, not runtime | confirmed | README table separating runtime from training assets |
| 20/20 probe results should not be presented as reproducing 0.98 | agreed | report and README now say the 0.98 is the original training-corpus evaluation, independently unverified |
| 70-query set was used for tuning, so it is a dev set | already self-disclosed | stated in the new README limitations section |

The logging fix matters more than its size. A retrieval failure is a bug
report, and the text that triggered it is the user's actual question —
which in this app may be a health or government-services query typed by
someone unaware a log file exists. Length is enough to reproduce a crash
class; content is not ours to keep.

Two `!r` sites remain in `evaluation.py` and are correct: those log the
knowledge base's *own* questions and the test set's sentences, not user
input.

### Stale in the review itself

It describes graceful degradation for "sign model unavailable" and lists
sign-language files. That feature was removed in Round 25 — the review was
run against an earlier zip. Worth noting so the observation isn't
re-applied.

### The held-out set: harness built, questions deliberately not written

The review's strongest recommendation. `scripts/holdout_eval.py` scores
`eval/rag_holdout.csv` and prints precision with a Wilson interval next to
the dev-set numbers, so the generalisation gap is visible.

**The CSV ships empty on purpose.** The coverage threshold was tuned by an
AI assistant; that same assistant authoring the questions it is graded on
would make the exercise circular. The file documents what to write —
roughly a third paraphrases, half plausible South African questions the KB
does *not* cover — and notes that looking at the result and then adjusting
the threshold turns it back into a development set.

Same reasoning as declining to generate the 11-language detector
evaluation corpus: the mechanism is worth building, the answers are not
mine to supply.

### README restructured

Added a **Known limitations** section — speech covering 4 of 11, the 0.98
provenance, the dev-set caveat, out-of-distribution detection on short
queries, prompt-injection as mitigation, offline-first not offline-only.
Stated so a marker reads them rather than discovering them.

### Clean-environment check

Requested by the review and run: `requirements.txt` installed fresh, all
core imports succeed, 171 tests pass, `ruff check .` clean, gates green
apart from the parity gate.

### Not done

Wording changes to "prompt-injection protection" — the code, README and
report already say *mitigation* and *basic screening* throughout. The
review's concern was already satisfied.

---

## Round 33 — an incomplete fix, and a false positive coverage cannot catch (2026-08-30)

### 1. My Round 31 fix was half-applied

```
sawubona
Language: siSwati (ML model, 35%) — too unsure to use, replying in English
Sanibonani! Nginganisita njani?        <- the siSwati row
```

The confidence floor reached **generation** but not **retrieval**:
`lookup(prefer_language=detection.language)` still used the raw 35% guess
to choose which row came back. So the badge said "too unsure to use" while
the answer was served from the language it had just declined to trust.

One decision, one input — retrieval now uses `reply_language` too.

A partial fix that leaves the two halves disagreeing is worse than none,
because the interface now actively contradicts itself.

### 2. A false positive the coverage gate structurally cannot catch

```
calculus in xitsonga
Language: English (ML model, 63%) · source: knowledge_base
Advanced multivariable calculus tutorials are usually held in the main
sports complex hall or the library study rooms.
```

Similarity 0.303 at **100% coverage** — so the gate passed it. Both words
are in the knowledge base: "calculus" from a campus tutorial row,
"xitsonga" from a language row.

This is the *opposite* failure from the one coverage was built for. Round
13's gate detects questions the corpus **has no words for**. This question
uses only known words, but the combination asks for something the corpus
does not contain. No similarity threshold separates them either: the
genuine query for that row scores 0.643 and this scores 0.303, but
tightening to 0.4 would cost real matches elsewhere.

**Fix:** a query naming a target language ("… in Xitsonga", "… in
isiXhosa") is a *generation* request by construction — the knowledge base
holds no per-language content, so it cannot answer one. Those bypass
retrieval entirely.

Deliberately narrow: it requires the preposition immediately before a
known language name, so "how many official languages does South Africa
have" is unaffected. Pinned both ways by tests.

### Why this one matters beyond the bug

It is the sharpest illustration yet of the earlier warning about
institution-specific rows. A row about *where calculus tutorials are held*
acts as a magnet for any question containing "calculus", and the answer is
returned labelled **verified**. Two of the three transcript failures trace
back to that single row.

Labelled retrieval metrics unchanged: TP=19 FP=4 TN=37 FN=10, F1 0.731.

180 tests pass (9 new), `ruff check .` clean, gates green apart from the
parity gate.

---

## Round 34 — why the pan-African languages did nothing (2026-08-30)

Two reasons, one expected and one a bug.

### 1. The flag is off

`enable_pan_african = false`, as shipped in Round 24. Nothing routes
through the second classifier until someone turns it on.

### 2. Two thresholds gating the same decision on incompatible scales

With the flag on, a genuine detection was still being thrown away:

```
sannu   ->  detected hau 43% (sklearn)  ->  replied in English
```

`CompositeLanguageDetector` accepts a pan-African answer above **0.35**,
calibrated against that model: it scores 16-23% on input that is not its
language and 37-95% on input that is, so those bands do not overlap.

Round 31 then added `min_detection_confidence = 0.50` in the orchestrator,
calibrated against the **LSTM**, whose confidences are distributed
differently.

The result was a dead band between 0.35 and 0.50 in which a correct
detection was accepted by the detector and then silently discarded by the
orchestrator. Two gates, two calibrations, one decision.

**Fix:** the orchestrator does not re-judge a detection the composite has
already vouched for (`backend == "sklearn"`). The floor still applies to
the primary detector, which is what it was calibrated for. Both directions
tested.

### The general failure

A confidence number is only meaningful relative to the model that produced
it. Round 28 got this right for voices — the tier policy never compares
across backends. Round 31 got it wrong for detection by introducing a
single global floor over two models. The same mistake, one round apart.

Anything comparing confidences from different models needs to state which
model's scale it is using.

### After the fix, with the flag on

```
sannu                       hau  43%  (sklearn)   -> hau
jambo habari yako rafiki    swa  68%  (sklearn)   -> swa
bonjour comment allez vous  fra  54%  (sklearn)   -> fra
sawubona                    isiZulu 75% (heuristic) -> isiZulu
avuxeni                     Xitsonga 75% (heuristic) -> Xitsonga
```

Single-word greetings like "jambo" still fall through: they activate ~20
of 65,000 character features and score 31%, below the composite's own
floor. Full phrases work. That is the model being out of distribution on
one-word input, not a threshold problem — the same effect as the LSTM
scoring 35-42% on short English imperatives.

182 tests pass, `ruff check .` clean, gates green apart from the parity
gate.

---

## Round 35 — sidebar toggle for the extra 14 languages (2026-08-30)

Editing `config.toml` and restarting is a bad way to change a demo
mid-presentation. The pan-African detector is now a checkbox in the
sidebar.

### How it avoids a rebuild

`CompositeLanguageDetector` gained an `enabled` flag, and `bootstrap.py`
now constructs the composite **whenever a bundle exists** rather than only
when the setting is on. `config.toml` supplies the starting state; the
checkbox flips it live.

The alternative — rebuilding the pipeline on change — would mean
invalidating Streamlit's `@st.cache_resource` and reloading the TFLite
model on every flip. Slow, and exactly the kind of thing that fails in
front of judges.

Disabled, the composite returns the primary's result untouched, so the
default path is bit-for-bit unchanged rather than merely intended to be.
Verified:

```
                            enabled=False        enabled=True
sawubona                 -> isiZulu (heuristic)  isiZulu (heuristic)
avuxeni                  -> Xitsonga (heuristic) Xitsonga (heuristic)
sannu yaya kake          -> English (heuristic)  hau (sklearn)
jambo habari yako rafiki -> English (heuristic)  swa (sklearn)
```

The eleven are identical either way; only the extra fourteen appear.

### A test guarantee that had to change with the design

`test_default_build_uses_a_plain_detector` asserted the orchestrator did
not hold a composite at all. That is no longer true by construction, so
the guarantee moved from *"not wrapped"* to *"wrapped but inert"* — and
inertness is checked behaviourally (the composite's output equals its
primary's on every probe) rather than by type, because a type check would
have passed while the object misbehaved.

Mutating a cached object from the sidebar is safe in this deployment,
where Streamlit serves one session per process. Recorded in a comment
rather than left as an assumption.

183 tests pass, `ruff check .` clean, gates green apart from the parity
gate, Streamlit boots.

---

## Round 36 — Gemini failing, and an ISO code where a name belongs (2026-08-30)

```
jambo                    -> siSwati 35% -> replying in English -> offline_fallback
jambo habari yako rafiki -> swa                                -> offline_fallback
"I ran into a problem generating a response just now."
```

### The toggle worked; Gemini did not

Line 3 detected `swa` via the sidebar toggle, so the composite is
functioning. All three then returned `GENERATION_ERROR_MESSAGE`.

That message is specifically a caught `GenerationError` — distinct from
`OFFLINE_NO_KEY_MESSAGE`, which is what an absent API key produces. So the
key is present and loaded; the API call itself failed. Most likely a rate
limit or quota after heavy testing. The real cause is printed in the
terminal by `logger.error(f"Gemini API error: {e}")`.

Nothing to fix in the code — the failure path behaved correctly, degrading
to a clear message rather than a stack trace.

### The real bug: ISO codes leaking out of the model

The bundle's labels are ISO 639-3 codes, which is right for a label encoder
and wrong for everything downstream:

- the interface read **"Language: swa"**;
- the Gemini system prompt read **"Respond naturally in swa"** — an
  instruction in a code the model has to guess at rather than a language it
  knows by name.

The second matters more. If the pan-African path had reached Gemini
successfully, it would have been asking for output in "swa" rather than
Swahili, which is a quiet quality problem that looks like a model
weakness.

`PAN_AFRICAN_LANGUAGE_NAMES` maps all fourteen at the detector boundary,
where the model's labels become the application's. Unmapped codes fall
through unchanged, so a retrain with extra languages still works — it just
shows a code until a name is added. A test checks the mapping against the
bundle's actual labels rather than a hardcoded list, so that gap is caught
rather than assumed.

### A test that had to change with the fix

`test_bundle_labels_are_names_not_encoded_integers` asserted the detector
returned a raw ISO code. That was the correct assertion for the bug it was
written against (Round 24's encoded-integer labels) and the wrong one after
this fix. Rewritten to check the property it exists for — that the label
survived decoding — against the mapping's value set.

185 tests pass, `ruff check .` clean, gates green apart from the parity
gate.
