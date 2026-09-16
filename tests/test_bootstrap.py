"""The composition root must actually be the only one.

`orchestrator.py` claims the two interfaces "can never again drift into
implementing the pipeline differently". That was true of the pipeline and
false of its construction: both UIs hand-built the same six components
with the same arguments, in two files. These tests make the claim真 by
enforcement rather than by comment.
"""

import inspect
import re

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
