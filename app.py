from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return {"status": "ok", "message": "Bot server is alive"}
