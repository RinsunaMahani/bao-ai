"""Bao AI package.

Deliberately left with no re-exports (the previous version imported
ASSISTANT_AVATAR/GEMINI_MODEL/etc. from a flat `config.py` and
`SecurityGuardrails` from a flat `security.py` at this level — both moved
under `core/` as part of the clean-architecture restructure, so importing
them here would just reintroduce a second, indirect path to the same
names). Import what you need directly from its real home:

    from bao.core.config import Settings
    from bao.core.security import SecurityGuardrails
    from bao.ai.orchestrator import Orchestrator
"""
