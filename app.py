# app.py
import os
import json
import traceback
from typing import Optional

from fastapi import FastAPI, Request
import httpx

# ================== CONFIG ==================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # или "gpt-4o"

API_URL  = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_URL = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

# ================== APP =====================
app = FastAPI()


# ------------- Telegram helpers -------------
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

async def tg_get_file_path(file_id: str) -> str:
    """Возвращает file_path на серверах Telegram (без скачивания)."""
    data = await tg_api("getFile", {"file_id": file_id})
    return data["result"]["file_path"]


# --------------- OpenAI Vision --------------
async def analyze_math_image_by_url(image_url: str, grade_label: str = "") -> dict:
    """
    Передаёт в OpenAI прямую **ссылку** на картинку (без base64) и получает JSON.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")

    system_prompt = (
        "Ты — учитель математики 7–9 классов. Тебе приходит фото тетради. "
        "Твоя задача: аккуратно прочитать запись, ПЕРЕрешить задачи самостоятельно и сравнить с тем, "
        "что написано учеником. Если видимость плохая — не угадывай цифры, а пиши, что неразборчиво. "
        "Никогда не выдумывай ошибки, если не уверен. Отвечай строго в JSON."
    )
    if grade_label:
        system_prompt += f" Класс/тема: {grade_label}."

    user_prompt = (
        "Верни JSON ровно такого вида:\n"
        "{\n"
        '  "confidence": 0.0..1.0,  // оценка уверенности в распознавании записи ученика\n'
        '  "steps": ["шаг 1", "шаг 2", "..."],\n'
        '  "mistakes": [{"where":"...", "type":"...", "why":"..."}],\n'
        '  "gaps": ["..."],\n'
        '  "drills": ["задача 1", "задача 2", "задача 3"],\n'
        '  "summary": "1–2 предложения: что подтянуть"\n'
        "}\n"
        "Правила: 1) Сначала реши задачу сам. 2) Сравни со строками ученика. "
        "3) Если запись цифры неуверенно читается, отметь в mistakes тип 'неразборчиво' и не утверждай ошибку. "
        "4) Если всё верно — укажи, что ошибок нет."
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # ВАЖНО: image_url — ОБЪЕКТ {"url": "..."}; отдаём ПРЯМОЙ URL Telegram (без base64)
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 350,  # держим коротко и дёшево
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

    try:
        raw = data["choices"][0]["message"]["content"]
        return json.loads(raw)
    except Exception:
        try:
            fixed = (raw or "").strip().strip("`").strip()
            return json.loads(fixed)
        except Exception:
            print("JSON parse failed. Raw:", (raw or "")[:500])
            return {
                "confidence": 0.0,
                "steps": [],
                "mistakes": [],
                "gaps": [],
                "drills": [],
                "summary": "Не удалось распарсить ответ модели. Попробуйте переснять фото."
            }


# --------------- Formatting -----------------
def format_report(j: dict) -> str:
    conf = j.get("confidence")
    steps = j.get("steps") or []
    mistakes = j.get("mistakes") or []
    gaps = j.get("gaps") or []
    drills = j.get("drills") or []
    summary = j.get("summary") or ""

    lines = []
    if isinstance(conf, (int, float)):
        lines.append(f"Уверенность распознавания: {round(float(conf)*100)}%")
        lines.append("")

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
        lines.append("Ошибок не найдено или запись плохо читается.")
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
    return msg[:4000] if len(msg) > 4000 else msg


# ----------------- Routes -------------------
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/webhook/telegram")
async def tg_webhook(request: Request):
    try:
        update = await request.json()

        message = update.get("message") or update.get("edited_message")
        if message:
            chat_id = message["chat"]["id"]
            message_id = message.get("message_id")
            text = message.get("text") or ""
            photos = message.get("photo") or []

            # /start
            if text.startswith("/start"):
                hello = (
                    "Привет! Отправь фото задачи (лучше по одной на фото). "
                    "Я разберу решение, отмечу ошибки и дам мини-тренировку.\n\n"
                    "Лайфхак: фотографируй крупно и при хорошем свете — так точнее."
                )
                await tg_send_message(chat_id, hello, reply_to=message_id)
                return {"ok": True}

            # Фото
            if photos:
                largest = photos[-1]
                file_id = largest["file_id"]
                try:
                    await tg_send_message(chat_id, "Фото получено ✅ Анализирую…", reply_to=message_id)

                    # берём ПРЯМОЙ url с серверов Telegram
                    file_path = await tg_get_file_path(file_id)
                    tg_file_url = f"{FILE_URL}/{file_path}"  # публичная ссылка по токену бота

                    report = await analyze_math_image_by_url(tg_file_url)
                    text_report = format_report(report) or \
                        "Не получилось сформировать отчёт. Попробуйте переснять фото крупнее/резче."

                    await tg_send_message(chat_id, text_report)
                except httpx.HTTPError as e:
                    print("HTTP error during analysis:", e)
                    await tg_send_message(
                        chat_id,
                        "Не удалось обратиться к сервису анализа. Попробуй ещё раз чуть позже."
                    )
                except Exception as e:
                    print("Analysis error:", e)
                    print(traceback.format_exc())
                    await tg_send_message(
                        chat_id,
                        "Не удалось проанализировать фото 😕\n"
                        "Сделай снимок ближе и чётче, по одному заданию на фото."
                    )
                return {"ok": True}

            # Эхо — для проверки, что бот жив
            if text:
                await tg_send_message(chat_id, f"Я получил: {text}", reply_to=message_id)
                return {"ok": True}

            await tg_send_message(chat_id, "Пришли /start или отправь фото.")
            return {"ok": True}

        # под кнопки на будущее
        if update.get("callback_query"):
            chat_id = update["callback_query"]["message"]["chat"]["id"]
            await tg_send_message(chat_id, "Кнопка нажата.")
            return {"ok": True}

        return {"ok": True}

    except Exception as e:
        print("Webhook handler error:", e)
        print(traceback.format_exc())
        return {"ok": True}
