from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


app = FastAPI()

# directory is relative to project root
# app.mount("/public", StaticFiles(directory="frontend/public"), name="public")
app.frontend("/", directory="frontend/public", fallback="404.html")


# Server status
@app.get("/status")
async def root():
    return {"message": "SmartLane server is currently up."}

# POST creates data
# GET reads data
# PUT updates data
# DELETE deletes data
