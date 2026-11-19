import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

import app.main as backend_main
from app.services.vision_worker_client import (
    VisionWorkerClient,
    VisionWorkerConfigurationError,
    VisionWorkerResponseError,
    VisionWorkerUnavailable,
)
from app.services.camera_repository import InvalidCameraAssignment


def test_worker_client_sends_shared_token_and_payload():
    def handle(request):
        assert request.url == "http://127.0.0.1:8001/api/v1/cameras/reconcile"
        assert request.headers["Authorization"] == "Bearer worker-secret"
        assert request.read() == b'{"active_camera_ids":[10]}'
        return httpx.Response(200, json=[])

    client = VisionWorkerClient(
        token="worker-secret",
        transport=httpx.MockTransport(handle),
    )

    result = asyncio.run(
        client.request_json(
            "PUT",
            "/api/v1/cameras/reconcile",
            {"active_camera_ids": [10]},
        )
    )

    assert result == []


def test_worker_client_restricts_url_to_loopback():
    with pytest.raises(VisionWorkerConfigurationError):
        VisionWorkerClient(base_url="http://example.com:8001", token="worker-secret")


def test_worker_client_surfaces_unavailable_and_worker_errors():
    def unavailable(_request):
        raise httpx.ConnectError("connection refused")

    offline_client = VisionWorkerClient(
        token="worker-secret",
        transport=httpx.MockTransport(unavailable),
    )
    with pytest.raises(VisionWorkerUnavailable):
        asyncio.run(offline_client.request_json("GET", "/api/v1/cameras"))

    error_client = VisionWorkerClient(
        token="worker-secret",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(409, json={"detail": "Camera busy."})
        ),
    )
    with pytest.raises(VisionWorkerResponseError) as error:
        asyncio.run(error_client.request_json("PUT", "/api/v1/cameras/reconcile", {}))
    assert error.value.status_code == 409


class FakeCameraRepository:
    def __init__(self, *, initially_assigned=False):
        self.cameras = [
            {
                "camera_id": 1,
                "camera_name": "North camera",
                "region": "North Road",
                "source": "0",
                "status": "OFFLINE",
            },
            {
                "camera_id": 2,
                "camera_name": "South camera",
                "region": "South Road",
                "source": "1",
                "status": "OFFLINE",
            },
        ]
        self.slot_regions = {1: "North Road", 2: "South Road"}
        self.assignments = {1: 1} if initially_assigned else {}
        self.updated_by = None

    def read_state(self):
        by_id = {camera["camera_id"]: camera for camera in self.cameras}
        slots = []
        for slot_id, region in self.slot_regions.items():
            camera = by_id.get(self.assignments.get(slot_id))
            slots.append(
                {
                    "slot_id": slot_id,
                    "region": region,
                    "camera_id": camera["camera_id"] if camera else None,
                    "camera_name": camera["camera_name"] if camera else None,
                    "source": camera["source"] if camera else None,
                    "status": camera["status"] if camera else None,
                }
            )
        return [dict(camera) for camera in self.cameras], slots

    def replace_assignments(self, assignments, updated_by):
        if {item["slot_id"] for item in assignments} != set(self.slot_regions):
            raise InvalidCameraAssignment(
                "The update must include every current camera slot exactly once."
            )
        selected_ids = [
            item["camera_id"]
            for item in assignments
            if item["camera_id"] is not None
        ]
        if len(set(selected_ids)) != len(selected_ids):
            raise InvalidCameraAssignment(
                "A camera cannot be assigned to more than one slot."
            )
        self.assignments = {
            assignment["slot_id"]: assignment["camera_id"]
            for assignment in assignments
            if assignment["camera_id"] is not None
        }
        self.updated_by = updated_by
        for camera in self.cameras:
            camera["status"] = "OFFLINE"

    def update_statuses(self, statuses):
        for camera in self.cameras:
            camera["status"] = statuses[camera["camera_id"]]


class FakeWorkerClient:
    def __init__(self, calls, *, unavailable=False):
        self.calls = calls
        self.unavailable = unavailable

    async def request_json(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if self.unavailable:
            raise VisionWorkerUnavailable("The vision worker is unavailable.")
        active = set(payload["active_camera_ids"])
        return [
            {
                "camera_id": camera["camera_id"],
                "source": camera["source"],
                "state": "running" if camera["camera_id"] in active else "stopped",
                "error": None,
            }
            for camera in payload["cameras"]
        ]


def install_backend_overrides(repository, user, worker_client):
    backend_main.app.dependency_overrides[
        backend_main.get_current_user
    ] = lambda: user
    backend_main.app.dependency_overrides[
        backend_main.get_admin_user
    ] = lambda: user
    backend_main.app.dependency_overrides[
        backend_main.get_camera_repository
    ] = lambda: repository
    backend_main.app.dependency_overrides[
        backend_main.get_vision_worker_client
    ] = lambda: worker_client


def test_admin_applies_full_unique_slot_assignment_and_worker_reconcile():
    calls = []
    repository = FakeCameraRepository()
    user = {"role": "ADMINISTRATOR", "user_id": "admin-123"}
    install_backend_overrides(repository, user, FakeWorkerClient(calls))
    try:
        with TestClient(backend_main.app) as client:
            response = client.put(
                "/api/v1/vision/camera-slots",
                json={
                    "assignments": [
                        {"slot_id": 1, "camera_id": 2},
                        {"slot_id": 2, "camera_id": None},
                    ]
                },
            )
    finally:
        backend_main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert repository.updated_by == "admin-123"
    assert response.json()["slots"][0]["source"] == "1"
    assert response.json()["cameras"][0]["status"] == "OFFLINE"
    assert response.json()["cameras"][1]["status"] == "ONLINE"
    assert calls == [
        (
            "PUT",
            "/api/v1/cameras/reconcile",
            {
                "cameras": [
                    {"camera_id": 1, "source": "0"},
                    {"camera_id": 2, "source": "1"},
                ],
                "active_camera_ids": [2],
                "retry_failed": True,
            },
        )
    ]


def test_enforcer_camera_slot_response_omits_all_camera_sources():
    calls = []
    repository = FakeCameraRepository(initially_assigned=True)
    user = {"role": "ENFORCER", "user_id": "enforcer-456"}
    install_backend_overrides(repository, user, FakeWorkerClient(calls))
    try:
        response = TestClient(backend_main.app).get(
            "/api/v1/vision/camera-slots"
        )
    finally:
        backend_main.app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert "source" not in body["slots"][0]
    assert body["cameras"] == []


def test_assignment_validation_rejects_duplicate_camera_and_incomplete_slots():
    calls = []
    repository = FakeCameraRepository()
    user = {"role": "ADMINISTRATOR", "user_id": "admin-123"}
    install_backend_overrides(repository, user, FakeWorkerClient(calls))
    try:
        with TestClient(backend_main.app) as client:
            duplicate = client.put(
                "/api/v1/vision/camera-slots",
                json={
                    "assignments": [
                        {"slot_id": 1, "camera_id": 1},
                        {"slot_id": 2, "camera_id": 1},
                    ]
                },
            )
            incomplete = client.put(
                "/api/v1/vision/camera-slots",
                json={"assignments": [{"slot_id": 1, "camera_id": 1}]},
            )
            duplicate_slot = client.put(
                "/api/v1/vision/camera-slots",
                json={
                    "assignments": [
                        {"slot_id": 1, "camera_id": 1},
                        {"slot_id": 1, "camera_id": 2},
                    ]
                },
            )
    finally:
        backend_main.app.dependency_overrides.clear()

    assert duplicate.status_code == 422
    assert incomplete.status_code == 422
    assert duplicate_slot.status_code == 422
    assert calls == []


def test_worker_unavailability_persists_assigned_camera_as_fault():
    calls = []
    repository = FakeCameraRepository(initially_assigned=True)
    user = {"role": "ADMINISTRATOR", "user_id": "admin-123"}
    install_backend_overrides(
        repository,
        user,
        FakeWorkerClient(calls, unavailable=True),
    )
    try:
        response = TestClient(backend_main.app).get(
            "/api/v1/vision/camera-slots"
        )
    finally:
        backend_main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["slots"][0]["status"] == "FAULT"
    assert response.json()["worker_available"] is False


def test_backend_requires_authentication_for_camera_slots():
    response = TestClient(backend_main.app).get("/api/v1/vision/camera-slots")

    assert response.status_code == 401
