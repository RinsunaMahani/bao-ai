# Contributing to Bao AI

Thank you for improving Bao AI! This project is organized to support
reproducible development and responsible research.

## Getting started

1. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install -r dev-requirements.txt
   ```
2. (Optional) install the pre-commit hook, so `ruff` runs automatically before each commit:
   ```bash
   pre-commit install
   ```
3. Run the test suite:
   ```bash
   pytest tests/ -v
   ```
4. Lint:
   ```bash
   ruff check .
   ```

## Code organization

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full pipeline
diagram and the module responsibility table. The short version:

- `bao/core/` — config, logging, security, exceptions. No module here may
  import from `ai/`, `knowledge/`, `services/`, or `ui/`.
- `bao/knowledge/` — loading, embedding, indexing, and retrieving text.
  Must never import Gemini or Streamlit.
- `bao/ai/` — the Gemini client, prompt templates, conversation memory,
  and `orchestrator.py`, which is the *only* place pipeline sequencing
  should live.
- `bao/services/` — language detection, translation, speech, sign
  language, offline-mode policy. Each service does one job and exposes it
  through a small interface; it should not know about the other services.
- `bao/ui/` — `streamlit_app.py` and `console_app.py` are rendering layers
  only. If you find yourself writing an `if`/`else` branch here that
  decides *what* the assistant should do (not just how to display it), it
  belongs in `ai/orchestrator.py` instead.

## Style and quality

- Keep business logic out of the UI layer — `bao/ui/*.py` should stay thin
  wrappers around `ai/orchestrator.py`.
- Add unit tests for new functionality in `tests/` — one test file per
  module is the convention (`test_security.py` mirrors `core/security.py`,
  etc.). Prefer mocking heavy external dependencies (the Gemini client,
  the TFLite model, torch/transformers) over requiring real model weights,
  a GPU, or an API key, so the suite stays fast (a few seconds end to end)
  and runs the same way in CI as it does locally.
- Guard optional/heavy imports with `try/except` + a `has_*_support()`
  flag, the way `knowledge/embeddings.py`, `services/language_detector.py`,
  and `services/speech.py` already do, rather than importing them
  unconditionally at module level — a missing or mismatched dependency
  should degrade a single feature, never crash the whole app on startup.
  (This project has direct, tested history of that going wrong: an earlier
  version imported the TTS module unconditionally and its two undefined
  names took the entire Streamlit app down before rendering a single
  page — see `docs/ARCHITECTURE.md` for the full account.)
- New prompt text goes in `ai/prompts.py`, not inline in `orchestrator.py`
  or a service module — it's the one file meant to be skimmed end-to-end
  to understand exactly what the app asks Gemini to do.
- Run `ruff check .` before committing; CI will reject a PR that fails it.
- Document any model or dataset changes in the README, and re-run
  `python evaluate.py` after changing `data/african_data.csv` or the
  retrieval logic — a drop in self-retrieval accuracy usually means
  something regressed.
