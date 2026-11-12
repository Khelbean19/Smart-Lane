from psycopg2 import Timestamp
from pydantic import BaseModel

class Camera(BaseModel):
    id: int
    name: str
    road_end_id: int
    source: str
    status: str
    last_frame_at: str