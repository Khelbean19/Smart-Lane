"""Authentication and account-operation response models."""

from uuid import UUID

from pydantic import BaseModel, Field
from .user import UserProfile


class TokenResponse(BaseModel):
    """Bearer-token details returned by authentication endpoints.

    ``expires_in`` is the lifetime of ``access_token`` in seconds.  Clients
    should send the token as ``Authorization: Bearer <access_token>``.
    """

    access_token: str = Field(description="JWT access token.")
    token_type: str = Field(default="bearer", description="Authorization type used with the access token.")
    expires_in: int = Field(description="Access-token lifetime in seconds.")


class AuthResponse(TokenResponse):
    """Authentication result returned by the JSON login endpoint."""

    message: str = Field(description="Human-readable authentication result.")
    user_id: UUID | str = Field(description="Identifier of the authenticated account.")


class MessageResponse(BaseModel):
    """Generic operation result message."""

    message: str = Field(description="Human-readable result of the operation.")


class UserListResponse(BaseModel):
    """Paged-by-limit container for user profile records.

    The current user-list endpoint populates at most ten records.
    """

    data: list[UserProfile] = Field(description="User profiles returned by the request.")