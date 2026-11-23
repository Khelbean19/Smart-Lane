from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class UserProfile(BaseModel):
    user_id: UUID
    full_name: str
    username: str
    email: str | None
    phone: str
    role: str
    status: str
    last_login: datetime | None
    created_at: datetime
    updated_at: datetime | None
