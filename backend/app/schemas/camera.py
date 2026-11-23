"""Request models for camera-slot assignment operations."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CameraSlotAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: int = Field(gt=0)
    camera_id: int | None = Field(default=None, gt=0)


class ApplyCameraAssignments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignments: list[CameraSlotAssignment]

    @model_validator(mode="after")
    def require_unique_slots_and_cameras(self) -> "ApplyCameraAssignments":
        slot_ids = [assignment.slot_id for assignment in self.assignments]
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("A camera slot may only be specified once.")

        camera_ids = [
            assignment.camera_id
            for assignment in self.assignments
            if assignment.camera_id is not None
        ]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("A camera cannot be assigned to more than one slot.")
        return self