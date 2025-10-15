import os
from pathlib import Path
from fastapi import FastAPI, Request
import httpx
import traceback

app = FastAPI()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
if not BOT_TOKEN:
    print("!!! TELEGRAM_BOT_TOKEN is missing in env")

API_URL  = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_URL = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

DOWNLOAD_DIR = Path("/tmp/tg_photos")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

async def tg_api(method: str, payload: dict):
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"{API_URL}/{method}", json=payload)
        # Логи на случай ошибки
        if r.status_code != 200:
            print("TG API ERROR", r.status_code, r.text)
        r.raise_for_status()
        return r.json()

async def tg_send_message(chat_id: int | str, text: str):
    try:
        return await tg_api("sendMessage", {"chat_id": chat_id, "text": text})
    except Exception as e:
        print("sendMessage failed:", e)
        print(traceback.format_exc())

async def tg_get_file(file_id: str) -> str:
    data = await tg_api("getFile", {"file_id": file_id})
    return data["result"]["file_path"]

async def tg_download_file(file_path: str) -> Path:
    url = f"{FILE_URL}/{file_path}"
    local = DOWNLOAD_DIR / Path(file_path).name
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url)
        if r.status_code != 200:
            print("FILE GET ERROR", r.status_code, r.text)
        r.raise_for_status()
        local.write_bytes(r.content)
    return local

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/webhook/telegram")
async def tg_webhook(request: Request):
    try:
        update = await request.json()
        print("== Incoming update ==")
        print(update)  # Жёсткий лог входящего апдейта

        # 1) message / edited_message
        message = update.get("message") or update.get("edited_message")
        if message:
            chat_id = message["chat"]["id"]
            text = message.get("text") or ""
            photos = message.get("photo") or []

            # а) /start
            if text.startswith("/start"):
                await tg_send_message(
                    chat_id,
                    "Привет! Отправь фото домашнего (лучше по одному заданию на фото)."
                )
                return {"ok": True}

            # б) фото
            if photos:
                largest = photos[-1]
                file_id = largest["file_id"]
                try:
                    file_path = await tg_get_file(file_id)
                    local_path = await tg_download_file(file_path)
                    await tg_send_message(chat_id, "Фото получено ✅ Сейчас проверю…")
                    # тут позже добавим вызов OpenAI
                    print("Saved photo to:", str(local_path))
                except Exception as e:
                    await tg_send_message(chat_id, f"Не удалось скачать фото: {e}")
                return {"ok": True}

            # в) любой другой текст — эха-тест (чтобы точно увидеть ответ)
            if text:
                await tg_send_message(chat_id, f"Я получил: {text}")
                return {"ok": True}

            # г) иначе — подсказка
            await tg_send_message(chat_id, "Пришли /start или отправь фото.")
            return {"ok": True}

        # 2) callback_query (на будущее)
        cb = update.get("callback_query")
        if cb:
            chat_id = cb["message"]["chat"]["id"]
            await tg_send_message(chat_id, "Нажата кнопка.")
            return {"ok": True}

        # 3) другие типы апдейтов
        print("Unhandled update type.")
        return {"ok": True}

    except Exception as e:
        print("Webhook handler error:", e)
        print(traceback.format_exc())
        # Важно всегда отвечать 200, чтобы TG не заспамил ретраями
        return {"ok": True}
