"""
Bao AI - Knowledge Retrieval.

Two retrievers, deliberately separate because they answer with different
levels of guarantee (see the note at the top of ai/orchestrator.py):

  - `KnowledgeRetriever` searches the curated Question/Answer CSV and
    returns a VERIFIED, human-written answer verbatim. A match here
    bypasses generation entirely — this is the offline path.
  - `DocumentRetriever` searches text the user uploaded this session and
    returns raw excerpts as UNVERIFIED context for a Gemini prompt.

Neither class implements vectorization or similarity itself. Both compose
the three layers that already exist:

    loader.py       CSV / file bytes -> clean rows and text chunks
    embeddings.py   text -> vectors (TF-IDF today, swappable)
    vector_store.py vectors -> nearest-match search

That is why `embedding_model` is injectable here: `compare_retrievers.py`
builds one KnowledgeRetriever with `TfidfEmbeddings()` and another with
`SentenceEmbeddings()` and scores them against the same evaluation set,
with no change to this file.

IMPORTANT — do not "simplify" this by calling `TfidfVectorizer` directly.
A previous revision did exactly that and, in bypassing embeddings.py, lost
its curated stop-word list. The consequence reached a live user: "what is
quantom physics" scored 0.352 against the South African currency fact on
the shared words "what is" alone, cleared the 0.25 threshold, and was
returned as a *verified* knowledge-base answer. See
`test_function_word_only_queries_are_rejected`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bao.core.exceptions import RetrievalError
from bao.core.logging import get_logger
from bao.core.security import strip_invisible
from bao.knowledge.embeddings import (
    KNOWLEDGE_BASE_STOP_WORDS,
    EmbeddingModel,
    TfidfEmbeddings,
    has_sklearn_support,
)
from bao.knowledge.loader import chunk_text, load_knowledge_csv
from bao.knowledge.vector_store import VectorStore

logger = get_logger(__name__)

# Error paths log the LENGTH of a query, never its text.
#
# A retrieval failure is a bug report, and the text that triggered it is
# the user's actual question — which in this app may be a health or
# government-services query typed by someone who has no idea a log file
# exists. The length is enough to reproduce a crash class; the content is
# not ours to keep. Same reasoning as the speech layer never logging
# synthesized text.

_STOP_WORDS = frozenset(KNOWLEDGE_BASE_STOP_WORDS)

# Sentinel: None is a real cached value here ("no discrete vocabulary").
_UNSET = object()


def has_retrieval_support() -> bool:
    """True when the offline retrieval path can run at all (scikit-learn
    present). The UIs check this and degrade to generation-only rather
    than crashing on a core-minus-sklearn install.
    """
    return has_sklearn_support()


@dataclass(frozen=True)
class KnowledgeFact:
    """A curated answer plus the language it is already written in."""

    answer: str
    language: str
    row: int
    score: float
    # Which rows are the same fact in other languages: a shared
    # Canonical_Id, or for greetings, the category (see counterpart()).
    category: str = ""
    canonical_id: str = ""


class KnowledgeRetriever:
    """Searches the curated facts CSV for a verified answer.

    Never raises on a missing or malformed knowledge base: it logs, leaves
    `is_initialized` False, and returns None from every query. The app is
    offline-first but must still start when the CSV is absent.
    """

    def __init__(
        self,
        data_path: str,
        threshold: float = 0.25,
        embedding_model: EmbeddingModel | None = None,
        min_coverage: float = 0.7,
        include_unreviewed: bool = False,
    ):
        self.data_path = data_path
        self.include_unreviewed = include_unreviewed
        self.threshold = threshold
        self.min_coverage = min_coverage
        self._df = None
        self._store: VectorStore | None = None
        self._weights_cache = _UNSET

        if not has_sklearn_support():
            logger.warning("scikit-learn unavailable — knowledge retrieval disabled.")
            return

        try:
            df = load_knowledge_csv(data_path)
        except RetrievalError as e:
            logger.warning(f"Knowledge base unavailable: {e}")
            return

        # Rows flagged `needs-review` are drafts awaiting a first-language
        # speaker's sign-off and are excluded from the index entirely.
        # Serving an unreviewed translation of an emergency number under a
        # "verified" label is strictly worse than returning nothing and
        # falling through to generation: the user cannot tell that the
        # answer was never checked, and the failure is silent. Set
        # include_unreviewed=True only to measure what the reviewed rows
        # WOULD do once approved.
        if not include_unreviewed and "Verified" in df.columns:
            pending = int((df["Verified"] == "needs-review").sum())
            if pending:
                df = df[df["Verified"] != "needs-review"].reset_index(drop=True)
                logger.info(
                    f"Excluded {pending} needs-review rows from retrieval. "
                    f"They serve once Verified is set."
                )

        corpus = df["search_corpus"].fillna("").astype(str).tolist()
        if not any(text.strip() for text in corpus):
            logger.warning(f"Knowledge base at {data_path} has no searchable text.")
            return

        try:
            store = VectorStore(embedding_model or TfidfEmbeddings())
            store.build(corpus)
        except (ValueError, ImportError) as e:
            logger.warning(f"Could not build the knowledge vector store: {e}")
            return

        self._df = df
        self._store = store
        logger.info(f"Knowledge retriever ready ({len(df)} facts, threshold {threshold}).")

    @property
    def is_initialized(self) -> bool:
        return self._store is not None and self._df is not None

    @property
    def dataframe(self):
        """The loaded knowledge base. `evaluation.py` iterates this to run
        retrieval self-consistency over every row.
        """
        if self._df is None:
            raise RetrievalError("Knowledge base was never initialized.")
        return self._df

    def best_match_index(self, query: str) -> tuple[int, float] | None:
        """Threshold-FREE nearest match: returns (row_index, similarity),
        or None if retrieval isn't available.

        Kept separate from `query_fact` so the threshold sweep in
        rag_evaluation.py can score every candidate threshold from one
        similarity computation per query, instead of rebuilding the
        retriever a dozen times. A score of 0.0 is a valid result (the
        query shared no content words with the corpus) — that is not the
        same as None, which means "retrieval is switched off."
        """
        if not self.is_initialized or not query or not query.strip():
            return None

        try:
            match = self._store.best_match(query.strip())
        except Exception as e:  # a malformed query must not take the app down
            logger.warning(f"Retrieval failed (query length {len(query)}): {e}")
            return None

        if match is None:
            return None
        return match.index, match.score

    _WORD_RE = re.compile(r"[a-z']+")

    def match_margin(self, query: str) -> float | None:
        """Gap between the best and second-best match, or None if
        retrieval is unavailable. A single candidate scores 1.0 (nothing
        competes with it).

        This is a SECOND, independent confidence signal to the absolute
        similarity score, and it exists for the semantic-embedding
        upgrade path rather than for TF-IDF. Dense sentence embeddings
        compress their similarity range — unrelated pairs sit well above
        zero rather than near it — so an absolute threshold that is
        correctly tuned for TF-IDF transfers badly. The margin does not
        depend on where the range sits: a query that genuinely matches
        one row beats the runner-up clearly, and an out-of-scope query
        scores several rows near-identically no matter how the backend
        scales.

        NOT yet wired into `query_fact` on purpose. Which of similarity,
        coverage and margin should gate the answer is exactly what the
        backend comparison is supposed to decide, and hard-coding a
        combination here before that experiment runs would prejudge it.
        `sweep_gates.py` reports all three.
        """
        if not self.is_initialized or not query or not query.strip():
            return None
        top = self._store.top_k(query, k=2)
        if not top:
            return 0.0
        if len(top) == 1:
            return 1.0
        return float(top[0].score - top[1].score)

    def query_coverage(self, query: str) -> float:
        """How much of the question the knowledge base can actually
        represent, weighted by how informative each word is. 1.0 means
        every meaningful word is known; low values mean the important
        parts of the question are invisible to the retriever.

        This exists because of a failure similarity alone cannot detect. A
        TF-IDF vectorizer silently DROPS terms outside its fitted
        vocabulary, so "what is the minimum wage in south africa" gets
        scored as if the user had asked "south africa" — matching the
        currency fact at 0.666, far above the 0.25 threshold, and
        returning it tagged `source="knowledge_base"`, i.e. verified. Two
        further queries did the same ("population of south africa", "how
        do i register to vote in south africa"). The similarity is not
        wrong; it is answering a different question from the one asked.

        Weighting matters and was measured, not assumed. Plain word-count
        coverage treats losing "population" the same as losing "south",
        which under-penalises exactly the case that hurts. Weighting each
        word by its IDF — and charging an unknown word the maximum IDF,
        since a word absent from the corpus is maximally distinguishing —
        cut false positives on the 70-query labelled set from 22 to 4,
        against 22 to 10 for the unweighted version at its own best
        threshold.

        Note this is a property of the CORPUS, not of English: "capital of
        south africa" scores 1.0 only because the knowledge base happens
        to have a capital-cities row. Coverage therefore improves on its
        own as the knowledge base grows, which is the right behaviour —
        the gate loosens exactly when the KB earns more trust.
        """
        if not self.is_initialized:
            return 0.0

        words = [w for w in self._WORD_RE.findall(query.lower()) if w not in _STOP_WORDS]
        if not words:
            return 0.0

        weights = self._term_weights()
        if weights is None:
            # Semantic backends have no discrete vocabulary and don't drop
            # unknown words, so this failure mode doesn't apply to them.
            return 1.0

        vocabulary, idf, max_idf = weights
        kept = sum(float(idf[vocabulary[w]]) for w in words if w in vocabulary)
        lost = sum(max_idf for w in words if w not in vocabulary)
        total = kept + lost
        return (kept / total) if total else 0.0

    def _term_weights(self):
        """(vocabulary, idf array, max idf) for the fitted vectorizer, or
        None if the embedding backend has no discrete vocabulary.
        """
        if self._weights_cache is _UNSET:
            vectorizer = getattr(self._store.embedding_model, "_vectorizer", None)
            vocabulary = getattr(vectorizer, "vocabulary_", None) if vectorizer else None
            idf = getattr(vectorizer, "idf_", None) if vectorizer else None
            if vocabulary is None or idf is None:
                self._weights_cache = None
            else:
                self._weights_cache = (vocabulary, idf, float(idf.max()))
        return self._weights_cache

    def query_fact(self, query: str) -> str | None:
        """Returns the curated Answer verbatim when the best match clears
        the configured threshold, otherwise None so the caller falls
        through to generation.
        """
        result = self.best_match_index(query)
        if result is None:
            return None

        index, score = result
        if score < self.threshold:
            return None

        # Similarity is necessary but not sufficient — see query_coverage.
        coverage = self.query_coverage(query)
        if coverage < self.min_coverage:
            logger.info(
                f"Rejecting KB match (similarity {score:.3f} but only "
                f"{coverage:.0%} of the query is in the knowledge base's "
                f"vocabulary); falling through to generation."
            )
            return None

        answer = str(self._df.iloc[index]["Answer"])
        logger.info(f"Knowledge-base hit (row {index}, similarity {score:.3f}).")
        return answer

    _TARGET_LANGUAGE_RE = None

    @classmethod
    def _names_a_target_language(cls, query: str) -> bool:
        """True for questions like "explain calculus in Xitsonga".

        These ask for content RENDERED IN a language. The knowledge base
        holds no per-language content, so by construction it cannot answer
        them — they belong to generation.

        This exists because the coverage gate cannot catch them. Coverage
        detects questions the corpus has no words for; this is the opposite
        failure — every word is known, but the combination means something
        the corpus does not contain. "calculus in xitsonga" scored 0.303
        with 100% coverage against a row about campus tutorial venues, and
        was returned as a VERIFIED answer, because "calculus" appears in
        that row and "xitsonga" appears in another.

        Deliberately narrow: it requires the preposition immediately before
        a known language name, so "how many official languages does South
        Africa have" is unaffected.
        """
        from bao.services.language_detector import named_target_language

        # Shared with the orchestrator, which applies the same rule to
        # decide what language to ANSWER in. Two copies of "does this name
        # a language" could disagree about one sentence, and then the
        # knowledge base and the reply language would be reading the same
        # question differently. It also widens this to the pan-African
        # names, which the local copy never knew about.
        return named_target_language(query) is not None

    def lookup(self, query: str, prefer_language: str | None = None) -> KnowledgeFact | None:
        """Finds a curated answer, preferring one written in the language
        the user is actually speaking.

        `prefer_language` exists because similarity alone cannot resolve
        genuine ambiguity. "Dumela" is the greeting in Sepedi, Sesotho AND
        Setswana, so three rows match it equally well and TF-IDF picks
        whichever happens to score highest. A live session showed the cost:
        the interface reported "Language: Sesotho" while answering with the
        Setswana row — badge and answer disagreeing on screen.

        Retrieval cannot fix that, because the information needed is not in
        the query. It is in the classifier's output. So the classifier's
        answer is passed in, and among candidates that clear the threshold,
        one written in that language wins.

        This is the language model doing the job it exists for. The demo
        story and the code now agree: shared vocabulary is disambiguated by
        the LSTM, not by word overlap.
        """
        if self._names_a_target_language(query):
            logger.info(
                "Query names a target language; this is a generation request, "
                "not a knowledge-base lookup."
            )
            return None

        candidates = self._candidates(query)
        if not candidates:
            return None

        if prefer_language:
            wanted = prefer_language.strip().lower()
            matching = [c for c in candidates if c.language.strip().lower() == wanted]
            if matching:
                return matching[0]

        return candidates[0]

    def _candidates(self, query: str, top_k: int = 5) -> list[KnowledgeFact]:
        """Rows clearing both the similarity threshold and the coverage
        gate, best first.
        """
        if not self.is_initialized or not query or not query.strip():
            return []
        if self.query_coverage(query) < self.min_coverage:
            return []

        try:
            matches = self._store.top_k(query.strip(), k=top_k)
        except Exception as e:
            logger.warning(f"Retrieval failed (query length {len(query)}): {e}")
            return []

        return [
            self._fact_at(match.index, match.score)
            for match in matches
            if match.score >= self.threshold
        ]

    def _fact_at(self, position: int, score: float) -> KnowledgeFact:
        row = self._df.iloc[position]

        def column(name: str, default: str = "") -> str:
            if name not in self._df.columns:
                return default
            value = row[name]
            return default if value is None or value != value else str(value).strip()

        return KnowledgeFact(
            answer=str(row["Answer"]),
            language=column("Language", "English") or "English",
            row=position,
            score=score,
            category=column("Category"),
            canonical_id=column("Canonical_Id"),
        )

    def counterpart(self, fact: KnowledgeFact, language: str) -> KnowledgeFact | None:
        """The same fact written in `language`, or None if there is none.

        "The same fact" is a row sharing its Canonical_Id - a reviewed
        translation - or, for a greeting, that language's own greeting.
        The greetings are parallel by design, one per language, but were
        never given shared ids, and a greeting is the commonest case: a
        visitor who has chosen Sesotho and types "avuxeni" should be greeted
        in Sesotho, not handed the Xitsonga greeting.

        Only served rows are searched, so a translation still marked
        needs-review is never returned by this route either.
        """
        if not self.is_initialized or not language or "Language" not in self._df.columns:
            return None
        df = self._df

        def normalised(name: str):
            return df[name].fillna("").astype(str).str.strip().str.lower()

        in_language = normalised("Language") == language.strip().lower()
        same_fact = None
        if fact.canonical_id and "Canonical_Id" in df.columns:
            same_fact = in_language & (normalised("Canonical_Id") == fact.canonical_id.lower())
        if (same_fact is None or not same_fact.any()) and fact.category.lower() == "greeting":
            if "Category" in df.columns:
                same_fact = in_language & (normalised("Category") == "greeting")
        if same_fact is None or not same_fact.any():
            return None

        position = int(same_fact.to_numpy().nonzero()[0][0])
        return self._fact_at(position, fact.score)

    def __len__(self) -> int:
        return 0 if self._df is None else len(self._df)


class DocumentRetriever:
    """In-memory chunk store for documents uploaded during a session.

    Session-scoped and never persisted: uploaded documents are the user's,
    and nothing here writes them to disk. `clear()` genuinely drops them.
    """

    # A ceiling on what one session can hold. Nothing bounded this: every
    # upload appended, so a large PDF re-indexed a few times grew the store
    # and the per-add refit cost without limit. 400 chunks is roughly 120
    # pages of text, far more than a session needs, and reaching it is
    # reported rather than silently truncated.
    MAX_CHUNKS = 400

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50,
                 max_chunks: int = MAX_CHUNKS):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
        self._chunks: list = []
        self._store: VectorStore | None = None

    def add_document(self, doc_id: str, content: str) -> int:
        """Chunks and indexes one document, REPLACING any previous version
        of the same one. Returns the number of chunks added (0 if the
        document had no extractable text).

        Replacing rather than appending because indexing the same file
        twice is a click away — the uploader keeps its files across reruns,
        so pressing the button again re-indexed everything. The visible
        cost was in the prompt: the same passage was handed to Gemini
        twice, paying for the tokens and inviting the model to treat a
        repetition as emphasis. It also distorts TF-IDF, which weights
        terms by how many chunks contain them.
        """
        # Again here, not only in the loader: not every caller comes through
        # an upload, and this text goes into a prompt. See strip_invisible.
        content = strip_invisible(content or "")
        if not has_sklearn_support() or not content.strip():
            return 0

        kept = [c for c in self._chunks if getattr(c, "source", "") != doc_id]
        replaced = len(self._chunks) - len(kept)
        # Where this document's chunks were, so a re-index puts the new
        # ones back in the same place. Appending instead would reorder the
        # "Documents Bao can read" list while the user is looking at it.
        # Every chunk before the first match is kept, so the index into
        # _chunks is also the index into `kept`.
        position = next(
            (i for i, c in enumerate(self._chunks)
             if getattr(c, "source", "") == doc_id),
            len(kept),
        )
        room = self.max_chunks - len(kept)
        if room <= 0:
            logger.warning(
                f"Document store is full ({self.max_chunks} chunks); "
                f"{doc_id} was not indexed. Clear the uploads to add more."
            )
            return 0

        # Cut to what fits BEFORE chunking. chunk_text builds every chunk of
        # the whole text, so a long file was fully chunked only for all but
        # `room` chunks to be thrown away - the work grew with the file,
        # not with what was kept.
        words = content.split()
        fits = (room - 1) * (self.chunk_size - self.chunk_overlap) + self.chunk_size
        if len(words) > fits:
            logger.warning(
                f"{doc_id} was truncated to its first {fits:,} of {len(words):,} "
                f"words — the store holds at most {self.max_chunks} chunks."
            )
            content = " ".join(words[:fits])

        new_chunks = chunk_text(
            content,
            source_name=doc_id,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )[:room]
        if not new_chunks:
            return 0
        if replaced:
            logger.info(f"Re-indexed {doc_id}: replaced {replaced} existing chunk(s).")

        self._chunks = kept[:position] + new_chunks + kept[position:]
        self._rebuild()
        return len(new_chunks)

    @property
    def sources(self) -> list[str]:
        """Which documents are currently indexed, for the interface to
        show. Without this a user cannot tell what the assistant can see.
        """
        seen: list[str] = []
        for chunk in self._chunks:
            source = getattr(chunk, "source", "")
            if source and source not in seen:
                seen.append(source)
        return seen

    def _rebuild(self) -> None:
        """Re-fits over ALL chunks, not just the new ones — TF-IDF weights
        depend on the whole corpus, so an incremental append would leave
        earlier chunks scored against a stale vocabulary.
        """
        try:
            store = VectorStore(TfidfEmbeddings())
            store.build([c.content for c in self._chunks])
            self._store = store
        except (ValueError, ImportError) as e:
            logger.warning(f"Could not index uploaded documents: {e}")
            self._store = None

    def search(self, query: str, top_k: int = 2, min_score: float = 0.15) -> str:
        """Returns matching excerpts as labeled text, or "" if nothing
        clears `min_score`. Returns a string rather than objects because
        every caller feeds it straight into a prompt or displays it.
        """
        if self._store is None or not query or not query.strip():
            return ""

        try:
            matches = self._store.top_k(query.strip(), k=top_k)
        except Exception as e:
            logger.warning(f"Document search failed (query length {len(query)}): {e}")
            return ""

        excerpts = [
            f"[Document: {self._chunks[m.index].title}]\n{self._chunks[m.index].content}"
            for m in matches
            if m.score >= min_score
        ]
        return "\n\n".join(excerpts)

    def clear(self) -> None:
        self._chunks = []
        self._store = None

    def __len__(self) -> int:
        return len(self._chunks)
