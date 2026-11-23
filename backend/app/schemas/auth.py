"""Authentication and account-operation response models."""

from uuid import UUID

from pydantic import BaseModel
from .user import UserProfile


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class AuthResponse(TokenResponse):
    message: str
    user_id: UUID | str


class MessageResponse(BaseModel):
    message: str


class UserListResponse(BaseModel):
    data: list[UserProfile]