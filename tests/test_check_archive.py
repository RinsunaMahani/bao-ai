"""The guard that stops live keys leaving in a shared archive.

It exists because it already happened once: a hand-built zip of this
project was shared with a live GEMINI_API_KEY and HF_TOKEN inside it.

Every fake secret below is assembled at runtime. Written out literally,
this file would contain a realistic token and fail the very scan it tests.
"""
from __future__ import annotations

import zipfile

from scripts.check_archive import findings

FAKE_HF = "hf_" + "Ab3" * 12
FAKE_GOOGLE = "AIza" + "B7c" * 11 + "de"
PRIVATE_KEY = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIE...\n-----END RSA " + "PRIVATE KEY-----\n"
CERTIFICATE = "-----BEGIN CERTIFICATE-----\nMIID...\n-----END CERTIFICATE-----\n"


def _reader(text: str):
    return lambda: text.encode("utf-8")


def test_credential_files_are_refused_by_name():
    for name in (".env", ".env.production", "credentials.json", "id_rsa", "cert.p12"):
        assert findings(name, _reader("")), name


def test_the_template_is_allowed():
    assert findings(".env.example", _reader("GEMINI_API_KEY=\nHF_TOKEN=\n")) == []


def test_a_public_certificate_bundle_is_not_a_secret():
    """pip's cacert.pem failed the old name-only check on every machine
    with a virtualenv. A check that always fails is one people ignore.
    """
    path = ".venv/Lib/site-packages/pip/_vendor/certifi/cacert.pem"
    assert findings(path, _reader(CERTIFICATE * 50)) == []


def test_a_private_key_is_one_whatever_it_is_called():
    assert findings("deploy/server.pem", _reader(PRIVATE_KEY)) == ["private key"]
    assert findings("tls.key", _reader(PRIVATE_KEY)) == ["private key"]


def test_a_key_pasted_into_an_ordinary_file_is_found():
    """The original incident was caught by name. The same key pasted into
    config.toml or a README would have sailed through a name-only check.
    """
    assert "Hugging Face token" in findings("config.toml", _reader(f'token = "{FAKE_HF}"'))
    assert "Google API key" in findings("notes.md", _reader(f"key: {FAKE_GOOGLE}"))


def test_an_assigned_credential_is_found_in_any_key_format():
    """Providers change key formats; this project's live Gemini key does
    not use the classic AIza shape at all. An assignment of the variable
    to a real-looking value is caught regardless of the format.
    """
    value = "AQ." + "x9Y" * 16
    assert "assigned credential" in findings("run.sh", _reader(f"GEMINI_API_KEY={value}"))


def test_third_party_code_is_checked_by_name_not_content():
    """Installed packages ship realistic fake keys as test fixtures."""
    path = "venv/lib/python3.14/site-packages/google/auth/tests/data.py"
    assert findings(path, _reader(f"FIXTURE = '{FAKE_GOOGLE}'")) == []


def test_git_internals_are_skipped_on_any_platform(tmp_path):
    """The old walk skipped paths containing "/.git/", a substring that
    never occurs on Windows, where the separator is a backslash - so all
    of .git was walked there. The walk now compares path PARTS.
    """
    from scripts.check_archive import _working_tree

    (tmp_path / ".git" / "objects").mkdir(parents=True)
    (tmp_path / ".git" / "objects" / "ab12").write_text(f"token {FAKE_HF}")
    (tmp_path / "bao").mkdir()
    (tmp_path / "bao" / "app.py").write_text("x = 1\n")

    walked = {path.replace("\\", "/") for path, _ in _working_tree(tmp_path)}
    assert walked == {"bao/app.py"}


def test_a_built_archive_is_checked(tmp_path):
    archive = tmp_path / "bao.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("bao/core/config.py", "x = 1\n")
        zf.writestr(".env", f"HF_TOKEN={FAKE_HF}\n")
    with zipfile.ZipFile(archive) as zf:
        flagged = {
            info.filename
            for info in zf.infolist()
            if findings(info.filename, lambda info=info: zf.read(info))
        }
    assert flagged == {".env"}
