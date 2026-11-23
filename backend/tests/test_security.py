import hashlib

import app.core.security as security
from app.core.security import (
    create_access_token,
    get_access_token_subject,
    hash_password,
    verify_password,
)


def test_access_token_round_trips_subject(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret-key")

    token = create_access_token("admin-user-id")

    assert get_access_token_subject(token) == "admin-user-id"


def test_remembered_access_token_round_trips_subject(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret-key")

    token = create_access_token("admin-user-id", remember_me=True)

    assert get_access_token_subject(token) == "admin-user-id"


def test_invalid_access_token_has_no_subject(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret-key")

    assert get_access_token_subject("not-a-valid-token") is None


def test_expired_access_token_has_no_subject(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret-key")
    monkeypatch.setattr(security, "ACCESS_TOKEN_TTL_SECONDS", -1)
    token = create_access_token("admin-user-id")

    assert get_access_token_subject(token) is None


def test_expired_remembered_access_token_has_no_subject(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret-key")
    monkeypatch.setattr(security, "REMEMBER_ACCESS_TOKEN_TTL_SECONDS", -1)
    token = create_access_token("admin-user-id", remember_me=True)

    assert get_access_token_subject(token) is None


def test_password_verification_supports_argon2_and_legacy_sha256():
    password = "safe-example-password"
    argon2_hash = hash_password(password)
    legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()

    assert verify_password(argon2_hash, password)
    assert verify_password(legacy_hash, password)
    assert not verify_password(argon2_hash, "incorrect-password")
    assert not verify_password(legacy_hash, "incorrect-password")
