from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/webhook/telegram")
async def tg_webhook(request: Request):
    # пока просто читаем апдейт и возвращаем 200
    _ = await request.json()
    return {"ok": True}

