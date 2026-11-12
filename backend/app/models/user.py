from pydantic import BaseModel

class User(BaseModel):
    full_name: str
    username: str
    email: str
    phone: str
    password: str
    role: str
    status: str