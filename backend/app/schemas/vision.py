"""Request and response models shared by the vision worker API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CameraConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int | None = Field(default=None, ge=160, le=7680)
    height: int | None = Field(default=None, ge=120, le=4320)
    fps: float | None = Field(default=None, gt=0, le=120)
    jpeg_quality: int | None = Field(default=None, ge=30, le=100)
    detection_width: int | None = Field(default=None, ge=1, le=7680)
    detection_height: int | None = Field(default=None, ge=1, le=4320)
    preprocessing_mode: Literal["fit", "crop"] | None = None
    roi_x: float | None = Field(default=None, ge=0, le=1)
    roi_y: float | None = Field(default=None, ge=0, le=1)
    roi_width: float | None = Field(default=None, gt=0, le=1)
    roi_height: float | None = Field(default=None, gt=0, le=1)

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
    detection_width: int
    detection_height: int
    preprocessing_mode: Literal["fit", "crop"]
    roi_x: float
    roi_y: float
    roi_width: float
    roi_height: float


class CameraResolutionResponse(BaseModel):
    width: int
    height: int


class CameraPreprocessingSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detection_width: int = Field(ge=1, le=7680)
    detection_height: int = Field(ge=1, le=4320)
    preprocessing_mode: Literal["fit", "crop"]
    roi_x: float = Field(ge=0, le=1)
    roi_y: float = Field(ge=0, le=1)
    roi_width: float = Field(gt=0, le=1)
    roi_height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_roi_bounds(self) -> "CameraPreprocessingSettingsUpdate":
        if self.roi_x + self.roi_width > 1 or self.roi_y + self.roi_height > 1:
            raise ValueError("The normalized crop region must lie within the frame.")
        return self


class CameraStatusResponse(BaseModel):
    camera_id: int
    source: str
    state: Literal["stopped", "starting", "running", "error"]
    config: CameraConfigResponse
    max_frame_resolution: CameraResolutionResponse
    source_resolution: CameraResolutionResponse | None
    preview_resolution: CameraResolutionResponse | None
    frame_sequence: int
    error: str | None


class CameraRegistryEntry(BaseModel):
    camera_id: int = Field(gt=0)
    source: str = Field(min_length=1)
    detection_width: int = Field(default=640, ge=1, le=7680)
    detection_height: int = Field(default=480, ge=1, le=4320)
    preprocessing_mode: Literal["fit", "crop"] = "fit"
    roi_x: float = Field(default=0, ge=0, le=1)
    roi_y: float = Field(default=0, ge=0, le=1)
    roi_width: float = Field(default=1, gt=0, le=1)
    roi_height: float = Field(default=1, gt=0, le=1)

    @model_validator(mode="after")
    def validate_roi_bounds(self) -> "CameraRegistryEntry":
        if self.roi_x + self.roi_width > 1 or self.roi_y + self.roi_height > 1:
            raise ValueError("The normalized crop region must lie within the frame.")
        return self


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
