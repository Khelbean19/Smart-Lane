"""Request models for camera-slot assignment operations."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CameraSlotAssignment(BaseModel):
    """Assignment of one physical camera slot to an optional camera.

    A ``null`` camera id unassigns the slot.  IDs must be positive, and the
    containing request rejects duplicate slot or camera assignments.
    """

    model_config = ConfigDict(extra="forbid")

    slot_id: int = Field(gt=0, description="Positive identifier of the camera slot.")
    camera_id: int | None = Field(default=None, gt=0, description="Positive camera identifier, or null to leave the slot unassigned.")


class ApplyCameraAssignments(BaseModel):
    """Request payload that replaces camera-to-slot assignments."""

    model_config = ConfigDict(extra="forbid")

    assignments: list[CameraSlotAssignment] = Field(
        description="New slot assignments. Each slot and assigned camera may appear once."
    )

    @model_validator(mode="after")
    def require_unique_slots_and_cameras(self) -> "ApplyCameraAssignments":
        """Reject duplicate slots and cameras before database reconciliation."""
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