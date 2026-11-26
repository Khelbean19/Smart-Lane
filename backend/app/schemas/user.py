"""Account profile models returned by user API endpoints."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Public account profile returned to clients.

    Password and password-hash fields are intentionally excluded.
    """

    user_id: UUID = Field(description="Unique account identifier.")
    full_name: str = Field(description="User's display name.")
    username: str = Field(description="Unique login username.")
    email: str | None = Field(description="Email address, if configured.")
    phone: str = Field(description="Account phone number.")
    role: str = Field(description="Current authorization role.")
    status: str = Field(description="Current account status.")
    last_login: datetime | None = Field(description="Most recent successful login, if available.")
    created_at: datetime = Field(description="Timestamp when the account was created.")
    updated_at: datetime | None = Field(description="Timestamp of the latest account update.")
