"""Request and response models shared by the vision worker API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CameraConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int | None = Field(default=None, ge=160, le=7680)
    height: int | None = Field(default=None, ge=120, le=4320)
    fps: float | None = Field(default=None, gt=0, le=120)
    jpeg_quality: int | None = Field(default=None, ge=30, le=100)

    @model_validator(mode="after")
    def require_a_setting(self) -> "CameraConfigUpdate":
        if not self.model_fields_set:
            raise ValueError("At least one camera setting must be provided.")
        if any(getattr(self, name) is None for name in self.model_fields_set):
            raise ValueError("Camera settings cannot be null.")
        return self


class CameraConfigResponse(BaseModel):
    width: int | None
    height: int | None
    fps: float | None
    jpeg_quality: int


class CameraStatusResponse(BaseModel):
    camera_id: int
    source: str
    state: Literal["stopped", "starting", "running", "error"]
    config: CameraConfigResponse
    frame_sequence: int
    error: str | None


class CameraRegistryEntry(BaseModel):
    camera_id: int = Field(gt=0)
    source: str = Field(min_length=1)


class CameraReconcileRequest(BaseModel):
    cameras: list[CameraRegistryEntry]
    active_camera_ids: list[int]
    retry_failed: bool = False

    @model_validator(mode="after")
    def validate_registry_and_active_ids(self) -> "CameraReconcileRequest":
        camera_ids = [camera.camera_id for camera in self.cameras]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("Camera IDs in the registry must be unique.")
        if len(set(self.active_camera_ids)) != len(self.active_camera_ids):
            raise ValueError("Active camera IDs must be unique.")
        if not set(self.active_camera_ids).issubset(camera_ids):
            raise ValueError("Active camera IDs must exist in the registry.")
        return self
