from pydantic import BaseModel, Field

class UserIn(BaseModel):
    full_name: str
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
    password: str
    role: str
    status: str


class UserAuth(BaseModel):
    username: str
    password: str
    remember: bool
    
