"""Database access for registered cameras and desired live-slot assignments."""

from __future__ import annotations

from typing import Any, ContextManager


CAMERA_STATUSES = {"ONLINE", "OFFLINE", "FAULT"}


class CameraDatabaseUnavailable(RuntimeError):
    """Raised when the camera database connection is unavailable."""


class InvalidCameraAssignment(ValueError):
    """Raised when a requested slot assignment does not match the database."""

class InvalidCameraSettings(ValueError):
    """Raised when camera preprocessing settings target an unknown camera."""


class CameraRepository:
    GLOBAL_FRAME_CAP_SETTINGS = {
        "width": "vision_max_frame_width",
        "height": "vision_max_frame_height",
    }

    def __init__(
        self,
        connection: Any,
        cursor: Any,
        database_lock: ContextManager[Any],
    ) -> None:
        self._connection = connection
        self._cursor = cursor
        self._database_lock = database_lock

    def read_state(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._require_connection()
        with self._database_lock:
            try:
                self._cursor.execute(
                    "SELECT cameras.camera_id, cameras.camera_name, "
                    "cameras.road_end_id, road_ends.label AS region, "
                    "cameras.source, cameras.status, "
                    "cameras.detection_width, cameras.detection_height, "
                    "cameras.preprocessing_mode, cameras.roi_x, cameras.roi_y, "
                    "cameras.roi_width, cameras.roi_height "
                    "FROM cameras "
                    "JOIN road_ends ON road_ends.road_end_id = cameras.road_end_id "
                    "ORDER BY cameras.camera_id"
                )
                cameras = [dict(row) for row in self._cursor.fetchall()]
                self._cursor.execute(
                    "SELECT road_ends.road_end_id AS slot_id, "
                    "road_ends.label AS region, cameras.camera_id, "
                    "cameras.camera_name, cameras.source, cameras.status, "
                    "cameras.detection_width, cameras.detection_height, "
                    "cameras.preprocessing_mode, cameras.roi_x, cameras.roi_y, "
                    "cameras.roi_width, cameras.roi_height "
                    "FROM road_ends "
                    "LEFT JOIN camera_assignments "
                    "ON camera_assignments.road_end_id = road_ends.road_end_id "
                    "LEFT JOIN cameras "
                    "ON cameras.camera_id = camera_assignments.camera_id "
                    "ORDER BY road_ends.road_end_id"
                )
                slots = [dict(row) for row in self._cursor.fetchall()]
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return cameras, slots

    def update_camera_preprocessing(
        self,
        camera_id: int,
        settings: dict[str, Any],
    ) -> None:
        self._require_connection()
        with self._database_lock:
            try:
                self._cursor.execute(
                    "UPDATE cameras SET detection_width = %s, "
                    "detection_height = %s, preprocessing_mode = %s, "
                    "roi_x = %s, roi_y = %s, roi_width = %s, roi_height = %s "
                    "WHERE camera_id = %s",
                    (
                        settings["detection_width"],
                        settings["detection_height"],
                        settings["preprocessing_mode"],
                        settings["roi_x"],
                        settings["roi_y"],
                        settings["roi_width"],
                        settings["roi_height"],
                        camera_id,
                    ),
                )
                if self._cursor.rowcount == 0:
                    raise InvalidCameraSettings(
                        f"Unknown camera ID {camera_id}."
                    )
                self._connection.commit()
            except InvalidCameraSettings:
                self._connection.rollback()
                raise
            except Exception:
                self._connection.rollback()
                raise

    def reconcile_global_frame_cap(
        self,
        max_width: int,
        max_height: int,
    ) -> list[dict[str, int]]:
        """Adjust camera detection dimensions when the configured cap changes."""
        if max_width <= 0 or max_height <= 0:
            raise ValueError("Global frame cap dimensions must be positive.")

        self._require_connection()
        with self._database_lock:
            try:
                setting_keys = tuple(self.GLOBAL_FRAME_CAP_SETTINGS.values())
                self._cursor.execute(
                    "SELECT setting_key, setting_value FROM system_settings "
                    "WHERE setting_key IN (%s, %s)",
                    setting_keys,
                )
                previous_cap = {
                    row["setting_key"]: int(row["setting_value"])
                    for row in self._cursor.fetchall()
                }
                cap_changed = previous_cap != {
                    self.GLOBAL_FRAME_CAP_SETTINGS["width"]: max_width,
                    self.GLOBAL_FRAME_CAP_SETTINGS["height"]: max_height,
                }
                changes: list[dict[str, int]] = []

                if cap_changed:
                    self._cursor.execute(
                        "SELECT camera_id, detection_width, detection_height "
                        "FROM cameras ORDER BY camera_id"
                    )
                    for camera in self._cursor.fetchall():
                        width = camera["detection_width"]
                        height = camera["detection_height"]
                        scale = min(max_width / width, max_height / height, 1.0)
                        adjusted_width = max(1, min(max_width, round(width * scale)))
                        adjusted_height = max(1, min(max_height, round(height * scale)))
                        if (adjusted_width, adjusted_height) == (width, height):
                            continue

                        self._cursor.execute(
                            "UPDATE cameras SET detection_width = %s, "
                            "detection_height = %s WHERE camera_id = %s",
                            (adjusted_width, adjusted_height, camera["camera_id"]),
                        )
                        changes.append(
                            {
                                "camera_id": camera["camera_id"],
                                "previous_width": width,
                                "previous_height": height,
                                "detection_width": adjusted_width,
                                "detection_height": adjusted_height,
                            }
                        )

                    for dimension, value in (
                        ("width", max_width),
                        ("height", max_height),
                    ):
                        self._cursor.execute(
                            "INSERT INTO system_settings "
                            "(setting_key, setting_value, value_type) "
                            "VALUES (%s, %s, 'INT') "
                            "ON CONFLICT (setting_key) DO UPDATE SET "
                            "setting_value = EXCLUDED.setting_value, "
                            "value_type = EXCLUDED.value_type, "
                            "updated_at = CURRENT_TIMESTAMP",
                            (self.GLOBAL_FRAME_CAP_SETTINGS[dimension], str(value)),
                        )

                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return changes

    def replace_assignments(
        self,
        assignments: list[dict[str, int | None]],
        updated_by: Any,
    ) -> None:
        self._require_connection()
        with self._database_lock:
            try:
                self._cursor.execute("SELECT road_end_id FROM road_ends")
                known_slots = {row["road_end_id"] for row in self._cursor.fetchall()}
                self._cursor.execute("SELECT camera_id FROM cameras")
                known_cameras = {row["camera_id"] for row in self._cursor.fetchall()}

                requested_slots = {
                    assignment["slot_id"] for assignment in assignments
                }
                if (
                    len(requested_slots) != len(assignments)
                    or requested_slots != known_slots
                ):
                    raise InvalidCameraAssignment(
                        "The update must include every current camera slot exactly once."
                    )

                camera_ids = [
                    assignment["camera_id"]
                    for assignment in assignments
                    if assignment["camera_id"] is not None
                ]
                if len(set(camera_ids)) != len(camera_ids):
                    raise InvalidCameraAssignment(
                        "A camera cannot be assigned to more than one slot."
                    )
                unknown_cameras = set(camera_ids) - known_cameras
                if unknown_cameras:
                    raise InvalidCameraAssignment(
                        "Unknown camera IDs: "
                        f"{', '.join(map(str, sorted(unknown_cameras)))}."
                    )

                self._cursor.execute("DELETE FROM camera_assignments")
                for assignment in assignments:
                    camera_id = assignment["camera_id"]
                    if camera_id is not None:
                        self._cursor.execute(
                            "INSERT INTO camera_assignments "
                            "(road_end_id, camera_id, updated_by, updated_at) "
                            "VALUES (%s, %s, %s, CURRENT_TIMESTAMP)",
                            (assignment["slot_id"], camera_id, updated_by),
                        )
                self._cursor.execute(
                    "UPDATE cameras SET status = 'OFFLINE'::camera_status"
                )
                self._connection.commit()
            except InvalidCameraAssignment:
                self._connection.rollback()
                raise
            except Exception:
                self._connection.rollback()
                raise

    def update_statuses(self, statuses: dict[int, str]) -> None:
        self._require_connection()
        invalid_statuses = set(statuses.values()) - CAMERA_STATUSES
        if invalid_statuses:
            raise ValueError(f"Invalid camera statuses: {sorted(invalid_statuses)}.")

        with self._database_lock:
            try:
                for camera_id, status in statuses.items():
                    self._cursor.execute(
                        "UPDATE cameras SET status = %s::camera_status "
                        "WHERE camera_id = %s",
                        (status, camera_id),
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def _require_connection(self) -> None:
        if self._connection is None or self._cursor is None:
            raise CameraDatabaseUnavailable("Database unavailable.")
