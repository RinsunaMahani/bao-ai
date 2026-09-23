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

3. **There is no large public annotated SASL corpus.** The available sign
   language datasets are predominantly ASL and predominantly isolated
   signs. A SASL system would need a corpus built first, with Deaf
   community involvement in its design.

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

## The 14 pan-African languages: detection without response

The pan-African detector (MasakhaNEWS, 14 languages) is a *detector*
only. Those languages have no knowledge-base rows and no speech path —
`voice_coverage()` iterates the 11 SA labels, so a pan-African language
is not merely uncovered, it is outside the coverage function's domain.

What actually happens when Bao detects Yoruba:

| Stage | 11 SA languages | 14 pan-African |
|---|---|---|
| Detection | yes | yes |
| Verified KB answer | yes (English + 2 pending review) | **none — zero rows** |
| Generated answer | yes | yes (Gemini, online only) |
| Voice | 5 native, 5 related, 1 text-only | **none** |

So "25 languages" is two different claims. State it as: **11 languages
Bao can respond in, and 14 more it can identify.** A panel that finds
this seam before you name it will read it as overclaiming; naming it
first reads as knowing your own system.

### Adding voices for the 14

`scripts/verify_pan_african_voices.py` checks MMS availability. Run it
where Hugging Face is reachable — do not assume. The SA MMS table started
at eleven entries and finished at two once it was actually queried.

Two code mismatches will cause silent 404s if the classifier's own labels
are reused as repository names:

- `swa` → MMS publishes `swh` (Coastal Swahili)
- `orm` → MMS publishes `gaz` (West Central Oromo)

These are varieties, not typos. If `gaz` resolves, Bao speaks West
Central Oromo specifically, and the UI should say so rather than claiming
"Oromo".

French is in the classifier because MasakhaNEWS includes it, not because
it is under-resourced. It has strong neural voices already. Do not
present a French voice as an under-resourced-language contribution.

A reachable repository means a voice exists, not that it is good. Nobody
on this team speaks most of these languages, which means a bad voice
would ship unnoticed. Availability is a precondition for shipping, not a
reason to.
