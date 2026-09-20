"""The composition root must actually be the only one.

`orchestrator.py` claims the two interfaces "can never again drift into
implementing the pipeline differently". That was true of the pipeline and
false of its construction: both UIs hand-built the same six components
with the same arguments, in two files. These tests make the claim真 by
enforcement rather than by comment.
"""

import inspect
import re

import pytest

from bao.bootstrap import build_orchestrator
from bao.ui import console_app, streamlit_app

# Components that define how Bao thinks. Constructing any of these outside
# bootstrap.py is how the two interfaces drift apart.
PIPELINE_COMPONENTS = [
    "SecurityGuardrails(",
    "KnowledgeRetriever(",
    "GeminiClient(",
    "ConversationMemory(",
    "get_language_detector(",
]


def _source(module):
    return inspect.getsource(module)


def test_neither_ui_constructs_pipeline_components_itself():
    for module in (console_app, streamlit_app):
        source = _source(module)
        offenders = [c for c in PIPELINE_COMPONENTS if c in source]
        assert not offenders, (
            f"{module.__name__} builds {offenders} directly instead of calling "
            "bao.bootstrap.build_orchestrator — that is how the console and the "
            "web app drift apart"
        )


def test_both_uis_call_the_shared_factory():
    for module in (console_app, streamlit_app):
        assert "build_orchestrator(" in _source(module), module.__name__


def test_factory_returns_a_usable_pipeline():
    settings, orchestrator = build_orchestrator()
    assert orchestrator.security is not None
    assert orchestrator.language_detector is not None
    assert orchestrator.document_retriever is not None
    # Settings come back with the orchestrator so callers don't re-parse
    # config.toml just to read a threshold.
    assert settings.similarity_threshold > 0


def test_console_profile_omits_the_document_retriever():
    """The one legitimate difference between the two interfaces: the web
    app accepts uploads, the console has no upload mechanism. Everything
    else must be identical by construction.
    """
    _, console = build_orchestrator(with_document_retriever=False)
    assert console.document_retriever is None


def test_settings_thresholds_actually_reach_the_retriever():
    """A factory that silently ignores config would be worse than the
    duplication it replaced.
    """
    settings, orchestrator = build_orchestrator()
    assert orchestrator.knowledge_retriever.threshold == settings.similarity_threshold
    assert orchestrator.knowledge_retriever.min_coverage == settings.min_query_coverage


def test_no_dead_api_key_constant():
    """config.py used to define GEMINI_API_KEY that nothing imported, while
    client.py read the environment itself — two sources of truth, one of
    them dead.
    """
    from bao.core import config

    source = inspect.getsource(config)
    assert not re.search(r"^GEMINI_API_KEY\s*=", source, re.MULTILINE), (
        "dead constant: client.py reads the environment directly"
    )


# --- one visitor's data must not reach another ---------------------------


def test_uploaded_documents_do_not_leak_between_sessions():
    """Demonstrated before the fix with a private results file: three
    differently-worded questions from a second session all returned it.

    Streamlit's @st.cache_resource caches across ALL users and sessions,
    and the cached orchestrator owned the uploaded-document store.
    """
    from bao.bootstrap import build_orchestrator, for_session

    _, shared = build_orchestrator()
    a, b = for_session(shared), for_session(shared)

    a.document_retriever.add_document(
        "results.txt",
        "Rinsuna Mahani failed MATH301 with 42 percent and is on academic probation.")

    assert a.document_retriever.search("MATH301 academic probation"), "A keeps its own"
    for query in ("MATH301 academic probation", "who failed MATH301",
                  "what percent did Rinsuna get"):
        assert not b.document_retriever.search(query), f"leaked via {query!r}"


def test_conversation_history_does_not_leak_between_sessions():
    """Memory is prepended to every prompt, so a shared one meant a second
    visitor's question arrived carrying the first visitor's conversation.

    Invisible from the screen: display_messages is per-session and was
    always correct, so each visitor saw only their own bubbles while the
    model received everybody's.
    """
    from bao.bootstrap import build_orchestrator, for_session

    _, shared = build_orchestrator()
    a, b = for_session(shared), for_session(shared)

    a.handle("my medical results are confidential", force_offline=True)

    assert a.memory.as_context(), "A keeps its own history"
    assert not b.memory.as_context(), "B must start with an empty context"


def test_the_language_toggle_is_per_session():
    """The sidebar mutates `enabled` on the composite detector. Shared,
    one visitor switching the pan-African languages on switched them on
    for everyone.
    """
    from bao.bootstrap import build_orchestrator, for_session
    from bao.services.language_detector import CompositeLanguageDetector

    _, shared = build_orchestrator()
    if not isinstance(shared.language_detector, CompositeLanguageDetector):
        pytest.skip("no pan-African bundle present")

    a, b = for_session(shared), for_session(shared)
    a.language_detector.enabled = not b.language_detector.enabled
    assert a.language_detector.enabled != b.language_detector.enabled


def test_the_expensive_components_are_still_shared():
    """The isolation must not become a per-visitor model reload. The
    classifier and the fitted knowledge base are read-only and identical
    for everyone, so they stay shared — that is what @st.cache_resource is
    for, and rebuilding them per session would cost seconds each time.
    """
    from bao.bootstrap import build_orchestrator, for_session

    _, shared = build_orchestrator()
    a, b = for_session(shared), for_session(shared)

    assert a.knowledge_retriever is b.knowledge_retriever
    assert a.gemini_client is b.gemini_client
    assert a.security is b.security


def test_session_orchestrators_inherit_every_tuning_value():
    """for_session copies rather than re-listing the constructor, so a
    parameter added to Orchestrator later cannot silently stop reaching
    session orchestrators. This fails if that ever changes.
    """
    from bao.bootstrap import build_orchestrator, for_session

    _, shared = build_orchestrator()
    session = for_session(shared)

    replaced = {"memory", "document_retriever", "language_detector"}
    for field, value in vars(shared).items():
        if field in replaced:
            continue
        assert vars(session)[field] == value, f"{field} did not carry across"
