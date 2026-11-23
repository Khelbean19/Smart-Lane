from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Barrier
from time import sleep
from uuid import UUID

from argon2 import PasswordHasher
from fastapi.testclient import TestClient

import app.main as backend_main
from app.core.security import create_access_token


class StubCursor:
    def __init__(self, user):
        self.user = user

    def execute(self, query, params=None):
        self.query = query

    def fetchone(self):
        return self.user


class InterleavingCursor:
    def __init__(self, user):
        self.user = user
        self.has_result = False

    def execute(self, query, params=None):
        self.has_result = True
        sleep(0.02)

    def fetchone(self):
        if not self.has_result:
            return None
        self.has_result = False
        return self.user


def test_current_user_endpoint_returns_safe_profile_for_active_enforcer(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret-key")
    user_id = UUID("6ce4d9d4-4525-4fee-b324-e04d4feeb820")
    monkeypatch.setattr(backend_main, "connection", object())
    monkeypatch.setattr(backend_main, "cursor", StubCursor({
        "user_id": user_id,
        "full_name": "Test Enforcer",
        "username": "test.enforcer",
        "email": "enforcer@example.com",
        "phone": "639123456789",
        "role": "ENFORCER",
        "status": "ACTIVE",
        "last_login": None,
        "created_at": datetime(2026, 11, 18, 12, 0),
        "updated_at": None,
        "password_hash": "must-not-be-returned",
    }))
    token = create_access_token(str(user_id))

    response = TestClient(backend_main.app).get(
        "/api/v1/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["username"] == "test.enforcer"
    assert response.json()["role"] == "ENFORCER"
    assert "password_hash" not in response.json()


def test_concurrent_current_user_lookups_do_not_interleave_shared_cursor(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret-key")
    user_id = UUID("6ce4d9d4-4525-4fee-b324-e04d4feeb820")
    monkeypatch.setattr(backend_main, "connection", object())
    monkeypatch.setattr(backend_main, "cursor", InterleavingCursor({
        "user_id": user_id,
        "full_name": "Test Enforcer",
        "username": "test.enforcer",
        "email": "enforcer@example.com",
        "phone": "639123456789",
        "role": "ENFORCER",
        "status": "ACTIVE",
        "last_login": None,
        "created_at": datetime(2026, 11, 18, 12, 0),
        "updated_at": None,
    }))
    token = create_access_token(str(user_id))
    start = Barrier(2)

    def lookup_user():
        start.wait()
        return backend_main.get_current_user(token)

    with ThreadPoolExecutor(max_workers=2) as executor:
        users = list(executor.map(lambda _: lookup_user(), range(2)))

    assert [user["user_id"] for user in users] == [user_id, user_id]


def test_current_user_endpoint_requires_bearer_authentication():
    response = TestClient(backend_main.app).get("/api/v1/me")

    assert response.status_code == 401


def test_login_sets_remembered_token_lifetime(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret-key")
    user_id = UUID("6ce4d9d4-4525-4fee-b324-e04d4feeb820")
    monkeypatch.setattr(backend_main, "connection", object())
    monkeypatch.setattr(backend_main, "cursor", StubCursor({
        "user_id": user_id,
        "password_hash": PasswordHasher().hash("test-password"),
        "role": "ADMINISTRATOR",
        "status": "ACTIVE",
    }))

    response = TestClient(backend_main.app).post(
        "/api/v1/auth/",
        json={
            "username": "admin",
            "password": "test-password",
            "remember": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["expires_in"] == 7 * 24 * 60 * 60
