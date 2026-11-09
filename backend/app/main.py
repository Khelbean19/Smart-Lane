from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World!"}

# POST creates data
# GET reads data
# PUT updates data
# DELETE deletes data
