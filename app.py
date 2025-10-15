import os
import io
import json
import base64
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
import httpx
from PIL import Image

# ==== CONFIG ====
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # можно сменить на gpt-4o
MAX_SIDE = int(os.getenv("MAX_SIDE", "1600"))  # макс. длина стороны для сжатия

API_URL  = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_URL = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

# ==== APP ====
app = FastAPI()
DOWNLOAD_DIR = Path("/tmp/tg_photos")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Telegram helpers ----------
async def tg_api(method: str, payload: dict):
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(f"{API_URL}/{method}", json=payload)
        if r.status_code != 200:
            print("TG API ERROR", r.status_code, r.text)
        r.raise_for_status()
        return r.json()

async def tg_send_message(chat_id: int | str, text: str, reply_to: Optional[int] = None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    try:
        return await tg_api("sendMessage", payload)
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


# ---------- Image helpers ----------
def load_and_downscale(path: Path, max_side: int = MAX_SIDE) -> bytes:
    """
    Открывает картинку, мягко сжимает (длинная сторона <= max_side),
    сохраняет в JPEG и возвращает байты.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
    if scale > 1.0:
        img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()

def b64_jpeg(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


# ---------- OpenAI Vision ----------
async def analyze_math_image(image_path: Path, grade_label: str = "") -> dict:
    """
    Отправляет картинку в мультимодальную модель OpenAI и получает структурированный JSON.
    Возвращает dict.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")

    # готовим картинку
    img_bytes = load_and_downscale(image_path, MAX_SIDE)
    img_b64 = b64_jpeg(img_bytes)

    # промпт (уточняем краткость ответа и формат JSON)
    system_prompt = (
        "Ты — учитель математики 7–9 классов. Анализируй фото тетради: "
        "коротко распиши шаги решения, найди типовые ошибки, сформулируй вероятные пробелы "
        "и предложи 2–3 коротких упражнения. Пиши строго в JSON."
    )
    if grade_label:
        system_prompt += f" Класс/тема: {grade_label}."

    user_prompt = (
        "Верни строго такой JSON без лишнего текста:\n"
        "{\n"
        '  "steps": ["шаг 1", "шаг 2", "..."],\n'
        '  "mistakes": [{"where":"...", "type":"...", "why":"..." }],\n'
        '  "gaps": ["..."],\n'
        '  "drills": ["задача 1", "задача 2", "задача 3"],\n'
        '  "summary": "1–2 предложения: что подтянуть"\n'
        "}\n"
        "Если что-то не видно — укажи это в summary и всё равно верни JSON."
    )

    # вызов OpenAI Responses API
    # используем чистый httpx (SDK 1.x тоже ок, но так надёжнее в небольшом файле)
    # --- ВЫЗОВ OpenAI Chat API ---
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,  # например, gpt-4o-mini или gpt-4o
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{img_b64}",
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        if r.status_code != 200:
            print("OpenAI ERROR", r.status_code, r.text)
        r.raise_for_status()
        data = r.json()

    # --- Извлекаем текст ---
    try:
        raw = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        return parsed
    except Exception:
        try:
            fixed = raw.strip().strip("`").strip()
            return json.loads(fixed)
        except Exception:
            print("JSON parse failed. Raw:", raw[:500])
            return {
                "steps": [],
                "mistakes": [],
                "gaps": [],
                "drills": [],
                "summary": "Не удалось распарсить ответ модели. Попробуйте переснять фото."
            }

    # держим защитный парсинг на случай изменений
    def extract_text(res: dict) -> str:
        # попытка 1: новый формат
        try:
            return res["output"][0]["content"][0]["text"]
        except Exception:
            pass
        # попытка 2: candidates
        try:
            return res["choices"][0]["message"]["content"]
        except Exception:
            pass
        return "{}"

    raw = extract_text(data)
    try:
        parsed = json.loads(raw)
        return parsed
    except Exception:
        # если пришёл нестрогий JSON — пытаемся поправить
        try:
            fixed = raw.strip().strip("`").strip()
            return json.loads(fixed)
        except Exception:
            print("JSON parse failed. Raw:", raw[:500])
            return {
                "steps": [],
                "mistakes": [],
                "gaps": [],
                "drills": [],
                "summary": "Не удалось распарсить ответ модели. Попробуйте переснять фото."
            }


# ---------- Format helpers ----------
def format_report(j: dict) -> str:
    steps = j.get("steps") or []
    mistakes = j.get("mistakes") or []
    gaps = j.get("gaps") or []
    drills = j.get("drills") or []
    summary = j.get("summary") or ""

    lines = []
    if steps:
        lines.append("Шаги решения:")
        for i, s in enumerate(steps, 1):
            lines.append(f"{i}) {s}")
        lines.append("")

    if mistakes:
        lines.append("Ошибки:")
        for m in mistakes:
            where = m.get("where", "—")
            mtype = m.get("type", "—")
            why = m.get("why", "")
            lines.append(f"• {where}: {mtype}. {why}")
        lines.append("")
    else:
        lines.append("Ошибок не найдено или видимость низкая.")
        lines.append("")

    if gaps:
        lines.append("Вероятные пробелы:")
        for g in gaps:
            lines.append(f"• {g}")
        lines.append("")

    if drills:
        lines.append("Мини-тренировка (3 задания):")
        for d in drills:
            lines.append(f"• {d}")
        lines.append("")

    if summary:
        lines.append(f"Итог: {summary}")

    msg = "\n".join(lines).strip()
    # Telegram ограничение ~4096 символов
    return msg[:4000] if len(msg) > 4000 else msg


# ---------- Routes ----------
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/webhook/telegram")
async def tg_webhook(request: Request):
    try:
        update = await request.json()
        # print("== Incoming ==", update)

        message = update.get("message") or update.get("edited_message")
        if message:
            chat_id = message["chat"]["id"]
            message_id = message.get("message_id")
            text = message.get("text") or ""
            photos = message.get("photo") or []
            grade_label = ""  # на будущее: можно хранить выбор темы/класса

            # /start
            if text.startswith("/start"):
                hello = (
                    "Привет! Отправь фото задачи (лучше по одной на фото). "
                    "Я разберу решение, отмечу ошибки и дам мини-тренировку."
                )
                await tg_send_message(chat_id, hello, reply_to=message_id)
                return {"ok": True}

            # Фото
            if photos:
                largest = photos[-1]
                file_id = largest["file_id"]
                try:
                    # подтверждение сразу (UX)
                    await tg_send_message(chat_id, "Фото получено ✅ Анализирую…", reply_to=message_id)

                    file_path = await tg_get_file(file_id)
                    local_path = await tg_download_file(file_path)

                    report = await analyze_math_image(local_path, grade_label=grade_label)
                    text_report = format_report(report)

                    if not text_report:
                        text_report = "Не получилось сформировать отчёт. Попробуйте переснять фото крупнее/резче."

                    await tg_send_message(chat_id, text_report)
                except Exception as e:
                    await tg_send_message(chat_id, f"Ошибка анализа: {e}")
                    print("Analysis error:", e)
                    print(traceback.format_exc())
                return {"ok": True}

            # любой другой текст — эхо
            if text:
                await tg_send_message(chat_id, f"Я получил: {text}", reply_to=message_id)
                return {"ok": True}

            await tg_send_message(chat_id, "Пришли /start или отправь фото.")
            return {"ok": True}

        # callback_query (на будущее для кнопок)
        if update.get("callback_query"):
            chat_id = update["callback_query"]["message"]["chat"]["id"]
            await tg_send_message(chat_id, "Кнопка нажата.")
            return {"ok": True}

        return {"ok": True}

    except Exception as e:
        print("Webhook handler error:", e)
        print(traceback.format_exc())
        return {"ok": True}
