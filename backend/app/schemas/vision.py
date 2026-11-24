"""Request and response models shared by the vision worker API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CameraConfigUpdate(BaseModel):
    """Partial camera-worker configuration update.

    At least one setting is required, and explicitly supplied settings cannot
    be null.  Resolution, frame, and ROI fields are constrained to worker-safe
    ranges.
    """

    model_config = ConfigDict(extra="forbid")

    width: int | None = Field(default=None, ge=160, le=7680, description="Captured frame width in pixels.")
    height: int | None = Field(default=None, ge=120, le=4320, description="Captured frame height in pixels.")
    fps: float | None = Field(default=None, gt=0, le=120, description="Target capture rate in frames per second.")
    jpeg_quality: int | None = Field(default=None, ge=30, le=100, description="JPEG preview quality from 30 through 100.")
    detection_width: int | None = Field(default=None, ge=1, le=7680, description="Detection input width in pixels.")
    detection_height: int | None = Field(default=None, ge=1, le=4320, description="Detection input height in pixels.")
    preprocessing_mode: Literal["fit", "crop"] | None = Field(default=None, description="Fit the frame or crop it to the ROI.")
    roi_x: float | None = Field(default=None, ge=0, le=1, description="Normalized ROI left boundary.")
    roi_y: float | None = Field(default=None, ge=0, le=1, description="Normalized ROI top boundary.")
    roi_width: float | None = Field(default=None, gt=0, le=1, description="Normalized ROI width.")
    roi_height: float | None = Field(default=None, gt=0, le=1, description="Normalized ROI height.")

    @model_validator(mode="after")
    def require_a_setting(self) -> "CameraConfigUpdate":
        """Require at least one setting and reject explicitly null settings."""
        if not self.model_fields_set:
            raise ValueError("At least one camera setting must be provided.")
        if any(getattr(self, name) is None for name in self.model_fields_set):
            raise ValueError("Camera settings cannot be null.")
        return self


class CameraConfigResponse(BaseModel):
    """Effective worker camera configuration returned by status endpoints."""
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
    """Frame or image dimensions reported by the worker."""

    width: int
    height: int


class CameraPreprocessingSettingsUpdate(BaseModel):
    """Complete preprocessing configuration for a configured camera.

    The normalized ROI must remain inside the unit frame.  Both ROI origin
    coordinates and dimensions are expressed from 0.0 through 1.0.
    """

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
        """Reject ROI rectangles that extend past the right or bottom edge."""
        if self.roi_x + self.roi_width > 1 or self.roi_y + self.roi_height > 1:
            raise ValueError("The normalized crop region must lie within the frame.")
        return self


class CameraStatusResponse(BaseModel):
    """Worker runtime state for one camera, including active resolutions."""

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
    """One camera registration used when reconciling database state to the worker."""

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
        """Reject a registry ROI that extends outside the normalized frame."""
        if self.roi_x + self.roi_width > 1 or self.roi_y + self.roi_height > 1:
            raise ValueError("The normalized crop region must lie within the frame.")
        return self


class CameraReconcileRequest(BaseModel):
    """Desired camera registry and active-camera set for worker reconciliation."""

    cameras: list[CameraRegistryEntry]
    active_camera_ids: list[int]
    retry_failed: bool = False

    @model_validator(mode="after")
    def validate_registry_and_active_ids(self) -> "CameraReconcileRequest":
        """Require unique registry/active IDs and ensure active IDs are registered."""
        camera_ids = [camera.camera_id for camera in self.cameras]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("Camera IDs in the registry must be unique.")
        if len(set(self.active_camera_ids)) != len(self.active_camera_ids):
            raise ValueError("Active camera IDs must be unique.")
        if not set(self.active_camera_ids).issubset(camera_ids):
            raise ValueError("Active camera IDs must exist in the registry.")
        return self


class CameraActivationError(BaseModel):
    """Worker activation failure associated with a camera."""

    camera_id: int
    error: str


class CameraSlotResponse(BaseModel):
    """Role-aware camera-slot record returned to an API caller."""

    slot_id: int
    region: str | None
    camera_id: int | None
    camera_name: str | None
    status: str
    source: str | None = None
    source_resolution: CameraResolutionResponse | None = None
    preview_resolution: CameraResolutionResponse | None = None
    max_frame_resolution: CameraResolutionResponse | None = None
    detection_resolution: CameraResolutionResponse | None = None
    preprocessing_mode: Literal["fit", "crop"] | None = None
    roi: dict[str, float] | None = None


class CameraRegistryStatusResponse(BaseModel):
    """Administrative camera status and preprocessing details."""

    camera_id: int
    camera_name: str
    region: str
    status: str
    source: str
    detection_width: int
    detection_height: int
    preprocessing_mode: Literal["fit", "crop"]
    roi_x: float
    roi_y: float
    roi_width: float
    roi_height: float
    max_frame_resolution: CameraResolutionResponse
    source_resolution: CameraResolutionResponse | None
    preview_resolution: CameraResolutionResponse | None


class CameraSlotsResponse(BaseModel):
    """Camera-slot and worker status payload."""

    slots: list[CameraSlotResponse]
    cameras: list[CameraRegistryStatusResponse]
    global_max_resolution: CameraResolutionResponse | None
    worker_available: bool
    worker_error: str | None
    activation_errors: list[CameraActivationError]


class CameraAssignmentsResponse(CameraSlotsResponse):
    """Response after a camera-slot assignment update."""

    assignments_saved: bool
    pending_reconciliation: bool


class CameraPreprocessingResponse(CameraSlotsResponse):
    """Response after updating preprocessing settings for one camera."""
    camera: CameraRegistryStatusResponse
    settings_saved: bool
    pending_reconciliation: bool


class DetectionResult(BaseModel):
    """One normalized YOLO box and class prediction."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: dict[str, float]


class CameraInferenceResult(BaseModel):
    """Latest inference state for one camera."""

    camera_id: int
    frame_sequence: int | None
    timestamp: float | None
    status: Literal["running", "ok", "error"]
    detections: list[DetectionResult]
    error: str | None = None


class InferenceStatusResponse(BaseModel):
    """Worker-level detector and per-camera inference status."""

    detector: dict[str, Any]
    active_camera_ids: list[int]
    camera_results: dict[str, CameraInferenceResult]
