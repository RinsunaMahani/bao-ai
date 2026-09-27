# Bao AI — Architecture

## Request pipeline

Every user interaction — typed text, transcribed voice, or (future)
or transcribed speech — flows through the same sequence:

```
User
  │
  ▼
Security guardrails        core/security.py
  │  (blocked -> short-circuit, never reaches memory)
  ▼
Language detection         services/language_detector.py
  │
  ▼
Knowledge retrieval        knowledge/retriever.py
  │  (offline facts CSV + session-uploaded documents)
  ▼
Gemini generation          ai/client.py
  │  (skipped if a verified fact was found, or if offline)
  ▼
Translation (if needed)    services/translation.py
  │  (only for facts pulled from the KB; Gemini answers are
  │   already generated directly in the target language)
  ▼
Conversation memory        ai/memory.py
  │  (updated after generation, so the reply is available as
  │   context for the *next* turn)
  ▼
Speech output (optional)   services/speech.py
  │
  ▼
Response
```

This is implemented as one class, `ai/orchestrator.py::Orchestrator`, so
both UIs (`ui/streamlit_app.py`, `ui/console_app.py`) call the exact same
sequence. The pre-refactor codebase had two Streamlit apps that each
implemented this flow independently and had already drifted apart — one
used TF-IDF retrieval + a broken TFLite classifier, the other used
keyword-overlap retrieval + a different (working) heuristic classifier,
with two different security implementations between them. There is now
exactly one pipeline, and the two UIs are rendering layers on top of it.

## Module map

| Package       | Owns                                              | Must never do                          |
|----------------|---------------------------------------------------|-----------------------------------------|
| `core/`        | Config, logging, security, exception types        | Talk to Gemini, touch the knowledge base|
| `knowledge/`   | Loading, embedding, indexing, retrieving text      | Know about Gemini or Streamlit          |
| `ai/`          | The Gemini client, prompts, memory, orchestration  | Parse files or do similarity search itself |
| `services/`    | Language detection, translation, speech, offline policy | Own the pipeline sequencing (that's `ai/orchestrator.py`'s job) |
| `ui/`          | Streamlit and console rendering                    | Contain pipeline branching logic        |

The rule of thumb used throughout: if you can describe what a function does
without saying "Streamlit" or "Gemini," it belongs in `core/`, `knowledge/`,
or `services/`, not in `ai/` or `ui/`.

## Key design decisions

**Two retrieval surfaces, kept separate.**
`KnowledgeRetriever` (the curated, offline facts CSV) and
`DocumentRetriever` (documents uploaded mid-session) answer different
questions and have different lifecycles — one loaded once at startup, one
rebuilt per session. Collapsing them into one class would mean either the
curated facts get wiped every time someone uploads a PDF, or uploaded
documents get treated as permanent verified facts. Keeping them separate
costs one extra class and avoids that entire category of bug.

**A pluggable language detector, not two competing ones.**
The original code had a keyword-heuristic detector and a TFLite/Keras
classifier that were never actually interchangeable — the classifier
returned a numeric class index with no path back to a language string, and
the two UIs called incompatible methods on them. `LanguageDetector` is now
an interface with two implementations (`HeuristicLanguageDetector`,
`TFLiteLanguageDetector`) selected by one factory function,
`get_language_detector()`, so the orchestrator never needs to know which
one it's talking to.

The TFLite model is now the default (`prefer_ml=True`), not the
heuristic — a reversal from an earlier version of this document, worth
explaining because *how* that changed is more instructive than the
conclusion itself. Two real bugs were compounding: the model's
preprocessing didn't match its own trained tokenizer (fixed first), and —
found later, empirically, by feeding one clearly-monolingual sentence per
language through the real model — the class-index-to-language mapping
was also wrong, using a UI-ordering list instead of the model's actual
(alphabetical) training-time class order. A four-sentence spot-check after
the first fix alone still looked bad (1/4 correct), which is exactly why
it stayed disabled at that point. Only after finding and fixing the
second bug did the same detector reach 20/20 across three independent
test sets, including outperforming the heuristic on a case (a Sepedi
greeting sharing a marker word with Sesotho) the heuristic is documented
to get wrong. The heuristic remains available and is the automatic
fallback if the model or tokenizer files are ever missing — zero load
time, trivial to debug, and a legitimate choice on its own merits, just
not the more accurate one once the ML path was actually working. Full
account: `REVIEW.md` §5.

**Translation is an explicit pipeline stage.**
Previously "translate this verified fact" was an inline f-string built
directly inside an if/else branch, indistinguishable in the code from
"answer this open-ended question with Gemini." It's now
`services/translation.py::translate_fact()`, called from one place in the
orchestrator, matching the pipeline diagram literally rather than
approximately.

**Conversation memory is new, not renamed.**
Both original Streamlit apps kept `st.session_state["messages"]`, but only
to render chat bubbles — every Gemini call was still built from a single,
memory-less prompt. `ai/memory.py::ConversationMemory` is fed back into
`system_instruction` on every turn, so the assistant can now resolve a
follow-up like "and in Afrikaans?" that refers to the previous answer. This
is a genuinely new capability, not a rename of something that already
worked.

**A documented, known limitation: TF-IDF's threshold.**
Testing this refactor against the real dataset surfaced a real precision
issue: the query *"what is the capital of a fictional planet"* returns a
KB match (`South Africa has three capital cities...`) at a similarity
score of 0.186, just above the default 0.15 threshold — because both
strings share enough common words ("the", "capital") for TF-IDF's cosine
similarity to register a signal that has nothing to do with actual
relevance. A later live test surfaced a more dramatic instance of the same
issue: *"what is the airspeed velocity of an unladen swallow"* — a
question sharing no topic with anything in the knowledge base — scores
**0.443** against a fact about South Africa's currency, nearly 3x the
threshold. Both are pinned as regression tests
(`tests/test_knowledge.py`). This isn't a bug so much as the known ceiling
of a keyword-overlap retrieval method on a small (52-row) corpus, where
TF-IDF's inverse-document-frequency weighting has far less data to
down-weight common words than it would in a larger collection; it's the
concrete argument for the `knowledge/embeddings.py` interface being
separate from `knowledge/vector_store.py` in the first place — swapping in
a real sentence-embedding model later is a new class behind the same
interface,
not a rewrite.

## Sessions and concurrency

The web app serves every visitor from one Python process, each session on
its own thread. Two rules follow, and both were learned from bugs rather
than designed in up front.

**Shared by mutability, not by cost.** `@st.cache_resource` is shared
across all visitors. The read-only parts of the pipeline — the TFLite
classifier, the fitted knowledge base, the API client, the guardrails —
are identical for everyone and expensive to build, so they are cached and
shared. The parts that hold one visitor's data are not:
`bootstrap.for_session` gives each session its own `ConversationMemory`,
its own `DocumentRetriever`, and its own copy of the detector's on/off
toggle. Before this split, a second visitor's prompt carried the first
visitor's conversation, and a private file one visitor uploaded was
retrievable by the next — invisibly, because the chat transcript on screen
was always per-session.

`for_session` copies the orchestrator rather than re-listing its
constructor, so a tuning parameter added later cannot silently stop
reaching sessions; a test asserts every non-session field carries across.

**Shared objects must be safe to use concurrently.** TFLite inference is
three stateful calls on one interpreter (`set_tensor`, `invoke`,
`get_tensor`) and is not reentrant: with three threads detecting at once,
two raised inside LiteRT. It is serialised with a lock — inference takes
about a millisecond, while an interpreter per session would cost seconds
and ~13 MB each. The output is copied inside the lock, because
`get_tensor` returns a view onto the interpreter's own buffer. The voice
models (MMS and the Coqui VITS) were put under the same concurrent load
and did not need a lock, so they do not have one.

## Sign language — removed

A static hand-shape prototype (MediaPipe hand landmarks feeding a
scikit-learn classifier) previously lived in `bao/services/sign_language.py`
and was removed from this codebase.

It was never trained. Making it work required recording hand data that
cannot be downloaded, and even then it would only have recognised static
poses — while most South African Sign Language vocabulary is *movement*,
and its grammar is carried partly on the face through non-manual markers
that hand landmarks cannot see. Usable recognition needs continuous video
and a sequence model; usable translation additionally needs a mapping
between SASL grammar and spoken-language grammar, since SASL is not signed
English.

That is a separate project, not a feature of this one. SASL became South
Africa's 12th official language in 2023, so it remains the right long-term
direction — but claiming it on the strength of an untrained hand-shape
classifier would have overstated what the system does.

## Accessibility and SASL

South African Sign Language became the country's 12th official language
in 2023 (Constitution Eighteenth Amendment Act, signed July 2023). Bao
supports the 11 spoken official languages and **does not support SASL**.
Documentation that says "11 official languages" without qualification is
now inaccurate, and has been corrected to "11 spoken official languages".

### Why SASL is not a feature we can add

SASL is a distinct language, not a manual encoding of spoken South
African languages. Parliament's own committee report on the amendment
notes that sign language is not universal — countries have their own
sign languages, and regions have dialects within them. SASL has its own
grammar and lexicon, and non-manual markers (facial expression, mouth
shape, body shift, eye gaze) carry grammatical meaning that hand
position alone does not encode.

Three consequences for architecture:

1. **It is a different input modality, not a different label.** Every
   existing language in Bao arrives as text. SASL arrives as continuous
   video, and requires pose, hand and face tracking plus a temporal
   model — none of which the current pipeline has a place for.

2. **Continuous recognition is not isolated classification.** Recognising
   individual signs from still frames is a solved-ish toy problem;
   segmenting and recognising continuous signing is not, and the two are
   routinely confused.

3. **There is no public annotated SASL corpus at all.** The available
   sign language datasets are predominantly ASL and predominantly
   isolated signs. A SASL system would need a corpus built first, with
   Deaf community involvement in its design. This was checked rather than
   assumed — see "What exists today" below.

### What exists today (checked 2026-09-23)

The claim above was re-checked against the research literature, the
public dataset hosts and the open-source translation tools, because
"nothing exists" is exactly the sort of statement that ought to be
verified before it is repeated.

| Resource | What it is | Usable here? |
|---|---|---|
| UCT SASL glove dataset (2014)[^uct2014] | Data-glove sensor readings, 31 static gestures, 5 participants | No — sensor data, not video, so it cannot train anything that works from a camera. Promised under GPL; no location is given. The thesis itself states "there are no available comprehensive databases available online." |
| UCT SASL sentence corpus (2024)[^uct2024] | 5,047 sentences, about five hours of studio video, six interpreters, government and politics domain | No — a master's thesis corpus, not publicly released |
| SASL alphabet image set | 26 static letters, about 12,400 training images, described in published work | Not found on Kaggle, Roboflow or GitHub |
| sign.mt[^signmt] | Open-source bidirectional spoken↔signed translation | Works "reasonably well" for American, German and Brazilian sign languages only, and "much worse for all other language pairs". SASL is not mentioned. |
| Sign language processing dataset catalogue[^slp] | The field's own index of datasets by language | Zero SASL datasets, against roughly 8–10 for ASL |

The most informative single figure is the 2024 thesis's own result.
Neural sign-to-text translation on that five-hour corpus scored **BLEU-4
1.35**, which the author describes as "very poor and still very far from
practical"; the same approach scores 13.23 on the German weather benchmark.
So the limit is not that this project did not try — it is that the people
who hold the only SASL video corpus cannot yet translate it usably either.

Tooling is not the blocker. MediaPipe 1.0.1 installs cleanly on the
Python 3.14 this project runs on, adding only itself and OpenCV, so camera
hand-tracking would run. The blocker is data, and behind the data, a
partnership with the Deaf community to decide what should be collected and
how.

One technically feasible, narrow piece remains — SASL *fingerspelling*,
which uses a one-handed manual alphabet[^wiki] — but it would need
self-recorded data from several signers, and each letter checked by a SASL
signer: SASL draws on Irish, British and American signing, so borrowing
the plentiful ASL alphabet data could teach wrong letters. Why that alone
should not be presented as SASL support is the subject of the next
section.

[^uct2014]: University of Cape Town, *South African Sign Language Dataset Development and Translation: A Glove-based Approach* (2014). https://open.uct.ac.za/server/api/core/bitstreams/4e3ee7a7-cf46-4c29-9e76-ddbda00f5619/content
[^uct2024]: M. Setshekgamollo, *Vision-Based Automatic Translation for South African Sign Language (SASL)*, MSc thesis, University of Cape Town (2024). https://open.uct.ac.za/items/5c66b556-1f37-4b1c-b12d-cba33a6f5728
[^signmt]: A. Moryossef, *sign.mt: Real-Time Multilingual Sign Language Translation Application*, EMNLP 2024 system demonstrations. https://arxiv.org/abs/2310.05064
[^slp]: *Sign Language Processing* — survey and dataset list. https://sign-language-processing.github.io/
[^wiki]: *South African Sign Language*, Wikipedia (official language from 19 July 2023; one-handed manual alphabet). https://en.wikipedia.org/wiki/South_African_Sign_Language

### What would be dishonest

Shipping a fingerspelling classifier or a fixed-vocabulary handshape
recogniser and calling it sign language support. Fingerspelling is not
how Deaf people converse; a 26-handshape alphabet classifier spells words
letter by letter, which is slower and worse than typing for the user it
claims to serve. It would demo well and serve nobody.

### The honest architecture, if it were ever built

    Camera
      -> pose + hand + face landmark extraction
      -> temporal model over landmark sequences
      -> SASL linguistic representation (glosses, non-manual markers)
      -> Bao's existing language/intent layer
      -> text / speech output

That is a research programme with a corpus-building phase, not a feature.
It belongs on a roadmap with the blocker named — corpus availability and
Deaf community partnership — rather than in a demo.

## The 14 pan-African languages: what "25 languages" does and does not mean

The pan-African detector (MasakhaNEWS, 14 languages) extends what Bao can
IDENTIFY. It does not extend what Bao KNOWS: the curated knowledge base is
written for South Africa and has no rows in any of these languages.

What happens at each stage, in this deployment:

| Stage | 11 South African | 14 pan-African |
|---|---|---|
| Detection | yes | yes — 24 of the 25 route correctly end to end; Luganda is detected as Xitsonga |
| Verified knowledge-base answer | yes — 38 rows served, 14 more awaiting a first-language speaker's review | **none — zero rows** |
| Generated answer | yes | yes (Gemini, online only) |
| Voice | 11 of 11 (4 on the code defaults; the other 7 need the opt-in SA VITS model) | 12 of 14 — MMS, with Microsoft neural voices for Amharic, French, Somali and Swahili; **none for Igbo or Lingala** |

So "25 languages" is two different claims. State it as: **11 languages
Bao can answer from verified knowledge, and 14 more it can identify and
answer through generation.** A panel that finds this seam before you name
it will read it as overclaiming; naming it first reads as knowing your own
system.

### The voices for the 14

`scripts/verify_pan_african_voices.py` probes which MMS voices exist. Run
it where Hugging Face is reachable — do not assume. The South African MMS
table started at eleven entries and finished at two once it was actually
queried.

Macrolanguage and variety codes do not both exist, and the two languages
where it matters go opposite ways:

- Swahili: `swa` 404s, `swh` exists — Bao speaks Coastal Swahili
  specifically, and should not present it as "Swahili" in general.
- Oromo: `orm` exists, `gaz` 404s. An earlier version of this document and
  of the script said the reverse; anything built on that claim would have
  failed silently.

French is in the classifier because MasakhaNEWS includes it, not because
it is under-resourced. It has strong neural voices already. Do not present
a French voice as an under-resourced-language contribution.

**These twelve voices are verified to exist, not verified to be good.**
Nobody on this team speaks most of these languages, so a bad voice would
ship unnoticed. The eleven South African voices were listened to and
signed off; these have not been. That is the difference between "has a
voice" and "is spoken well", and it should be stated whenever the
pan-African voices are demonstrated.

