from pydantic import BaseModel, Field

class UserIn(BaseModel):
    full_name: str = Field(description="User's display name.")
    username: str
    email: str | None = Field(
        description="Must a valid email address",
        default=None,
        pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    )
    phone: str = Field(
        description="Must be a valid Philippine phone number",
        min_length=12, max_length=12,
        pattern=r"63[0-9]{10}"
    )
    password: str = Field(description="Plain-text password supplied when creating the account.", min_length=8)
    role: str = Field(description="Account role, such as ADMINISTRATOR or ENFORCER.")
    status: str = Field(description="Account status, such as ACTIVE or DISABLED.")


class UserAuth(BaseModel):
    username: str = Field(description="Account username or email address.")
    password: str = Field(description="Account password.")
    remember: bool = Field(default=False, description="Extend the access-token lifetime.")
    
