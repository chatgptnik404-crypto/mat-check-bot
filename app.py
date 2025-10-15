import os
from pathlib import Path
from fastapi import FastAPI, Request
import httpx

app = FastAPI()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_URL = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

DOWNLOAD_DIR = Path("/tmp/tg_photos")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def tg_api(method: str, payload: dict):
    """Вызов Telegram Bot API."""
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"{API_URL}/{method}", json=payload)
        r.raise_for_status()
        return r.json()


async def tg_send_message(chat_id: int, text: str):
    return await tg_api("sendMessage", {"chat_id": chat_id, "text": text})


async def tg_get_file(file_id: str) -> str:
    """Возвращает путь файла на серверах Telegram (file_path)."""
    data = await tg_api("getFile", {"file_id": file_id})
    return data["result"]["file_path"]


async def tg_download_file(file_path: str) -> Path:
    """Качает файл по file_path в /tmp и возвращает локальный путь."""
    url = f"{FILE_URL}/{file_path}"
    local = DOWNLOAD_DIR / Path(file_path).name
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url)
        r.raise_for_status()
        local.write_bytes(r.content)
    return local


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/webhook/telegram")
async def tg_webhook(request: Request):
    update = await request.json()
    # Раскомментируй при отладке:
    # print("Incoming update:", update)

    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True}

    chat_id = message["chat"]["id"]

    # 1) Команда /start
    text = message.get("text", "")
    if text.startswith("/start"):
        hello = (
            "Привет! Я бот-проверяка по математике.\n\n"
            "Отправь фото домашнего задания (лучше по одному заданию на фото).\n"
            "После загрузки я пришлю краткий разбор и упражнения на тренировки."
        )
        await tg_send_message(chat_id, hello)
        return {"ok": True}

    # 2) Фото от пользователя
    if "photo" in message:
        # В массиве photo размеры по возрастанию — берём последнее (самое большое)
        largest = message["photo"][-1]
        file_id = largest["file_id"]

        try:
            file_path = await tg_get_file(file_id)
            local_path = await tg_download_file(file_path)
            # На следующем шаге сюда добавим вызов OpenAI (анализ изображения local_path)
            await tg_send_message(
                chat_id,
                "Фото получено ✅\nСейчас проверю и пришлю разбор (MVP-этап)."
            )
            # Для отладки можно оставить след:
            # await tg_send_message(chat_id, f"Сохранил: {local_path.name}")
        except Exception as e:
            await tg_send_message(chat_id, f"Не удалось обработать фото: {e}")

        return {"ok": True}

    # На всё остальное отвечаем подсказкой
    await tg_send_message(chat_id, "Пришли /start или отправь фото домашнего задания.")
    return {"ok": True}
