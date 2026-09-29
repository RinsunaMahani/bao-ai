from bao.core.config import LABELS, Settings


def test_mms_codes_cover_every_canonical_label():
    """Regression test for a real bug found via review: config.toml's
    mms_codes/stt_codes keys ("IsiZulu", "SiSwati", etc.) previously used
    different capitalization than LABELS ("isiZulu", "siSwati"). Since
    services/speech.py looks languages up with a plain `dict.get()`, that
    mismatch was a silent lookup miss — not a crash — that fell back to
    English audio for 5 of the 11 languages regardless of what was
    actually detected. Confirmed by testing the actual dict lookup before
    fixing, not just eyeballing the casing.
    """
    settings = Settings()
    mms_codes = settings.mms_codes
    stt_codes = settings.stt_codes

    missing_from_mms = [lang for lang in LABELS if lang not in mms_codes]
    missing_from_stt = [lang for lang in LABELS if lang not in stt_codes]

    assert not missing_from_mms, f"Languages missing a working MMS/TTS code: {missing_from_mms}"
    assert not missing_from_stt, f"Languages missing a working STT code: {missing_from_stt}"


def test_default_mms_and_stt_codes_also_cover_every_label():
    """Same check against the in-code fallback maps (used when config.toml
    is missing or doesn't define [languages]), so both sources of truth
    are pinned, not just the one that happens to be loaded right now.
    """
    from bao.core.config import DEFAULT_MMS_CODES, DEFAULT_STT_CODES

    assert all(lang in DEFAULT_MMS_CODES for lang in LABELS)
    assert all(lang in DEFAULT_STT_CODES for lang in LABELS)


def test_english_maps_to_a_voice_that_exists():
    """DEFAULT_MMS_CODES mapped English to "afr" to give it a local accent
    by routing English through Afrikaans letter-to-sound rules.

    facebook/mms-tts-afr does not exist, so that mapping silently removed
    English from the offline path entirely. config.toml was corrected and
    this default was not, which left every caller that does not pass an
    explicit table - and any install without a config.toml - fetching a
    404 and getting no audio.
    """
    from bao.core.config import DEFAULT_MMS_CODES
    from bao.services.speech import _KNOWN_MMS_VOICES

    assert DEFAULT_MMS_CODES["English"] == "eng"
    assert DEFAULT_MMS_CODES["English"] in _KNOWN_MMS_VOICES


def test_the_shipped_config_and_the_code_default_agree_on_english():
    """The two drifted once and nothing noticed. A mismatch here means one
    of them is wrong and the app behaves differently depending on whether
    config.toml is present.
    """
    from bao.core.config import DEFAULT_MMS_CODES, Settings

    assert Settings().mms_codes["English"] == DEFAULT_MMS_CODES["English"]


def test_the_declared_python_floor_is_the_one_ci_tests():
    """pyproject once declared Python 3.10 while the code needed 3.11 twice
    over: tomllib for config.toml, and scikit-learn 1.8+ for the pickled
    pan-African detector. On 3.10 the config was silently ignored and CI's
    own install log showed scikit-learn resolving to 1.7.2. A support claim
    is only true if CI tests it, so the floor, the lowest CI version and the
    README must agree.
    """
    import re
    import tomllib
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    floor = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"]["requires-python"]
    assert floor == ">=3.11"

    ci = (repo / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    versions = re.search(r"python-version:\s*\[([^\]]+)\]", ci).group(1)
    tested = sorted(tuple(map(int, v.strip(" \"'").split("."))) for v in versions.split(","))
    assert tested[0] == (3, 11), f"CI's lowest Python is {tested[0]}, not the declared floor"

    # Checks for a CLAIM of 3.10 support, not the number: the README
    # legitimately mentions 3.10 when explaining why it is not supported.
    readme = (repo / "README.md").read_text(encoding="utf-8")
    for claim in ("Python 3.10+", "Python-3.10", "on Python 3.10"):
        assert claim not in readme, f"README still claims support: {claim!r}"
    assert "Python 3.11+" in readme, "README should state the real floor"


def test_the_code_default_model_is_the_one_config_toml_chose():
    """If config.toml fails to load, the app should still use the model
    that was chosen for it - not silently fall onto a different model with
    a far smaller quota.
    """
    from bao.core.config import GEMINI_MODEL_DEFAULT, Settings

    assert Settings().gemini_model == GEMINI_MODEL_DEFAULT


def test_the_docker_image_contains_every_file_the_app_loads():
    """.dockerignore excluded the whole of docs/, and the web app loads its
    logo from docs/assets/. The image therefore could never render the page,
    and its health check - which reports the server, not the page - said
    nothing. A runtime file excluded from the build context is invisible
    until someone opens the app in a browser.

    Docker patterns are matched here with fnmatch against the path and each
    of its parent directories, which is enough for the simple patterns this
    project uses; `!` exceptions re-include.
    """
    import fnmatch
    from pathlib import Path

    from bao.core.config import ASSISTANT_LOGO_PATH, BASE_DIR, Settings

    repo = Path(BASE_DIR)
    s = Settings()
    runtime_files = [
        ASSISTANT_LOGO_PATH, s.knowledge_base_path, s.classifier_model_path,
        s.tokenizer_config_path, s.pan_african_model_path, str(repo / "config.toml"),
        str(repo / "bao" / "ui" / "streamlit_app.py"),
        # The server's security settings: without it the image would run on
        # Streamlit's defaults (tracebacks shown, usage statistics sent).
        str(repo / ".streamlit" / "config.toml"),
    ]
    patterns = [
        line.strip() for line in (repo / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    def excluded(rel: str) -> bool:
        parts = rel.split("/")
        candidates = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
        verdict = False
        for pattern in patterns:
            negate = pattern.startswith("!")
            body = pattern[1:] if negate else pattern
            body = body.rstrip("/")
            if any(fnmatch.fnmatch(c, body) for c in candidates):
                verdict = not negate
        return verdict

    missing = [
        rel for rel in (Path(f).resolve().relative_to(repo.resolve()).as_posix()
                        for f in runtime_files)
        if excluded(rel)
    ]
    assert not missing, f".dockerignore excludes files the app loads at runtime: {missing}"


def test_the_server_settings_stay_secure():
    """Each of these replaces a Streamlit default that is wrong for an app
    holding an API key and people's questions. See .streamlit/config.toml.
    """
    import tomllib
    from pathlib import Path

    from bao.knowledge.loader import MAX_UPLOAD_BYTES

    repo = Path(__file__).resolve().parent.parent
    cfg = tomllib.loads((repo / ".streamlit" / "config.toml").read_text(encoding="utf-8"))

    assert cfg["server"]["address"] == "localhost", "not on the network by default"
    assert cfg["server"]["enableXsrfProtection"] is True
    assert cfg["server"]["enableCORS"] is True
    assert cfg["browser"]["gatherUsageStats"] is False
    assert cfg["client"]["showErrorDetails"] == "none", "no tracebacks for visitors"
    assert cfg["server"]["maxUploadSize"] * 2**20 == MAX_UPLOAD_BYTES, (
        "the browser's upload limit and the loader's must agree"
    )


def test_the_container_still_listens_beyond_its_own_loopback():
    """Inside a container, localhost is the container itself, so the image
    must override the address or nothing outside could reach it. The host
    side is then limited by docker-compose.yml's 127.0.0.1 port binding.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    dockerfile = (repo / "docker" / "Dockerfile").read_text(encoding="utf-8")
    compose = (repo / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    assert "--server.address=0.0.0.0" in dockerfile
    assert '"127.0.0.1:8501:8501"' in compose, "published on the host's localhost only"

