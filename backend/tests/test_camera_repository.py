from threading import RLock

import pytest

from app.services.camera_repository import (
    CameraDatabaseUnavailable,
    CameraRepository,
    InvalidCameraAssignment,
)


class FakeConnection:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakeCursor:
    def __init__(self):
        self.query = ""
        self.parameters = None
        self.executed = []
        self.rowcount = 1

    def execute(self, query, parameters=None):
        self.query = query
        self.parameters = parameters
        self.executed.append((query, parameters))

    def fetchall(self):
        if self.query == "SELECT road_end_id FROM road_ends":
            return [{"road_end_id": 1}, {"road_end_id": 2}]
        if self.query == "SELECT camera_id FROM cameras":
            return [{"camera_id": 11}, {"camera_id": 22}]
        return []


def create_repository(connection=None, cursor=None):
    return CameraRepository(
        connection or FakeConnection(),
        cursor or FakeCursor(),
        RLock(),
    )


def test_replace_assignments_persists_unique_camera_selection_atomically():
    connection = FakeConnection()
    cursor = FakeCursor()
    repository = create_repository(connection, cursor)

    repository.replace_assignments(
        [
            {"slot_id": 1, "camera_id": 22},
            {"slot_id": 2, "camera_id": None},
        ],
        "admin-id",
    )

    inserts = [
        parameters
        for query, parameters in cursor.executed
        if query.startswith("INSERT INTO camera_assignments")
    ]
    assert inserts == [(1, 22, "admin-id")]
    assert connection.commits == 1
    assert connection.rollbacks == 0


def test_replace_assignments_rolls_back_invalid_slot_set():
    connection = FakeConnection()
    cursor = FakeCursor()
    repository = create_repository(connection, cursor)

    with pytest.raises(InvalidCameraAssignment, match="every current camera slot"):
        repository.replace_assignments(
            [{"slot_id": 1, "camera_id": 11}],
            "admin-id",
        )

    assert not any(
        query == "DELETE FROM camera_assignments"
        for query, _ in cursor.executed
    )
    assert connection.rollbacks == 1
    assert connection.commits == 0


def test_repository_reports_unavailable_database():
    repository = CameraRepository(None, None, RLock())

    with pytest.raises(CameraDatabaseUnavailable):
        repository.read_state()


def test_update_camera_preprocessing_persists_all_settings_atomically():
    connection = FakeConnection()
    cursor = FakeCursor()
    repository = create_repository(connection, cursor)

    repository.update_camera_preprocessing(
        11,
        {
            "detection_width": 640,
            "detection_height": 480,
            "preprocessing_mode": "crop",
            "roi_x": 0.25,
            "roi_y": 0.1,
            "roi_width": 0.5,
            "roi_height": 0.8,
        },
    )

    assert cursor.executed[-1] == (
        "UPDATE cameras SET detection_width = %s, "
        "detection_height = %s, preprocessing_mode = %s, "
        "roi_x = %s, roi_y = %s, roi_width = %s, roi_height = %s "
        "WHERE camera_id = %s",
        (640, 480, "crop", 0.25, 0.1, 0.5, 0.8, 11),
    )
    assert connection.commits == 1
    assert connection.rollbacks == 0
