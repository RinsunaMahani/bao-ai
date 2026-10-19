"""Sign-in: passwords stored only as salted hashes, and guessing slowed down."""

from __future__ import annotations

import pytest

from bao.core.auth import (
    USERNAME_RE,
    LoginGuard,
    hash_password,
    load_users,
    save_users,
    verify_password,
)


def test_a_password_is_never_stored_and_still_verifies():
    stored = hash_password("correct horse")
    assert "correct horse" not in stored
    assert verify_password("correct horse", stored)
    assert not verify_password("correct hors", stored)


def test_the_same_password_hashes_differently_each_time():
    """A random salt per account, so equal passwords are not visible as
    equal hashes and a precomputed table is useless."""
    assert hash_password("same password") != hash_password("same password")


@pytest.mark.parametrize("stored", ["", "plain-text", "md5$abc", "scrypt$x$8$1$AAAA$AAAA"])
def test_a_damaged_entry_refuses_rather_than_crashes(stored):
    assert not verify_password("anything", stored)


def test_accounts_survive_a_save_and_load(tmp_path):
    path = str(tmp_path / "users.toml")
    assert load_users(path) == {}, "no file yet means no accounts"
    users = {"rinsuna": hash_password("first password"), "guest.1": hash_password("second one")}
    save_users(path, users)
    assert load_users(path) == users
    assert "first password" not in (tmp_path / "users.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize("name,ok", [
    ("rinsuna", True), ("guest.1", True), ("a_b-c", True),
    ("", False), ("has space", False), ('quote"', False), ("x" * 33, False),
])
def test_usernames_are_plain(name, ok):
    """Plain names keep the accounts file trivially valid TOML."""
    assert bool(USERNAME_RE.fullmatch(name)) is ok


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def users():
    return {"rinsuna": hash_password("correct horse")}


def test_the_right_password_signs_in(users):
    assert LoginGuard().check(users, "rinsuna", "correct horse") == "rinsuna"


def test_a_wrong_password_and_an_unknown_name_look_the_same(users):
    guard = LoginGuard()
    assert guard.check(users, "rinsuna", "wrong") is None
    assert guard.check(users, "nobody", "correct horse") is None


def test_repeated_wrong_passwords_lock_the_name(users):
    """Even the right password is refused while locked, so guessing gains
    nothing by continuing; the lock lifts on its own."""
    clock = _Clock()
    guard = LoginGuard(max_failures=5, lockout_seconds=60, clock=clock)
    for _ in range(5):
        assert guard.check(users, "rinsuna", "guess") is None
    assert guard.seconds_locked("rinsuna") > 0
    assert guard.check(users, "rinsuna", "correct horse") is None

    clock.now += 61
    assert guard.seconds_locked("rinsuna") == 0
    assert guard.check(users, "rinsuna", "correct horse") == "rinsuna"


def test_a_lock_does_not_depend_on_letter_case(users):
    guard = LoginGuard(max_failures=2, clock=_Clock())
    guard.check(users, "Rinsuna", "guess")
    guard.check(users, "RINSUNA", "guess")
    assert guard.seconds_locked("rinsuna") > 0


def test_a_success_clears_earlier_failures(users):
    guard = LoginGuard(max_failures=3, clock=_Clock())
    guard.check(users, "rinsuna", "guess")
    guard.check(users, "rinsuna", "guess")
    assert guard.check(users, "rinsuna", "correct horse") == "rinsuna"
    guard.check(users, "rinsuna", "guess")
    assert guard.seconds_locked("rinsuna") == 0, "the count started again after the success"
