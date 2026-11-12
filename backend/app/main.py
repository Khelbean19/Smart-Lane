import hashlib
import sys
import uvicorn
from loguru import logger
from fastapi import FastAPI

from backend.app import database
from .models import User, Camera
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor


load_dotenv()
logger.remove()
logger.add(sys.stdout, level="DEBUG")

app = FastAPI()  # initialize backend
connection = database.main()  # initialize database connection

if connection:
    cursor = connection.cursor(cursor_factory=RealDictCursor)

# directory is relative to project root
# app.mount("/public", StaticFiles(directory="frontend/public"), name="public")
app.frontend("/", directory="frontend/public", fallback="404.html")

## LOGGING TEMPLATES FOR CONSOLE
# logger.debug("Received request")
# logger.info("Received request")
# logger.error("Received request")
# logger.critical("Received request")


# Server status
@app.get("/status")
async def root():
    return {"message": "SmartLane server is currently up."}


@app.post("/api/v1/users/")
async def create_user(user: User):
    if connection and cursor:
        cursor.execute("INSERT INTO users " \
                       "(full_name, username, email, phone, password_hash, role, status) " \
                       "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                       (user.full_name,
                        user.username,
                        user.email,
                        user.phone,
                        hashlib.sha256(user.password.encode("utf-8")).hexdigest(),  # dont fuck this up, for security reasons
                        user.role,
                        user.status))
        connection.commit()
    return {
        "message": "User created successfully",
        "user": user
    }


@app.get("/api/v1/users/")  # TODO (OPTIONAL): Add query parameters for pagination
async def get_users():
    if connection and cursor:  # db connection is open
        cursor.execute("SELECT " \
                       "user_id, full_name, username, email, phone, role, status, last_login, created_at, updated_at " \
                       "FROM users LIMIT 10")
        data = cursor.fetchall()
        return data
    else:
        return {
            "message": "Internal server error"
        }

# POST creates data
# GET reads data
# PUT updates data
# DELETE deletes data


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)