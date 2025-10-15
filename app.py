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
    Передаёт прямой URL изображения в OpenAI и получает строгий JSON.
    Логика: прочитать финальный ответ ученика, решить задачу заново, сравнить,
    найти только реальные ошибки хода (если есть), решать неуверенность честно.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")

    system_prompt = (
        "Ты — строгий и доброжелательный учитель математики 7–9 классов. "
        "Правила:\n"
        "1) Аккуратно считай, что написал ученик: выпиши ЕГО финальный ответ (если виден).\n"
        "2) Независимо реши задачу сам и получи СВОЙ финальный ответ.\n"
        "3) Сравни: если ответы совпадают (учитывай разумную погрешность для десятичных: 1e-3 или 1%), то итог ВЕРНЫЙ.\n"
        "4) Отмечай только РЕАЛЬНЫЕ ошибки хода (например, пропуск шага, неверное преобразование). Если хода решения не видно — не выдумывай.\n"
        "5) Если итог верный и ошибок хода НЕТ — не предлагай тренировку и не придумывай ошибки.\n"
        "6) Если что-то неразборчиво — честно укажи это и не утверждай про ошибки.\n"
        "7) Пиши строго в JSON указанного формата."
    )
    if grade_label:
        system_prompt += f" Контекст: {grade_label}."

    user_prompt = (
        "Верни РОВНО такой JSON (без лишнего текста):\n"
        "{\n"
        '  "confidence": 0.0,                   // 0..1 — уверенность, что запись ученика прочитана верно\n'
        '  "student_final_answer": null,        // строка/число, финальный ответ ученика, если прочитал; иначе null\n'
        '  "model_final_answer": null,          // твой рассчитанный ответ (строка/число)\n'
        '  "is_final_answer_correct": null,     // true/false, а если не удалось прочитать — null\n'
        '  "steps_student": [],                 // краткая реконструкция шагов ученика, если видны\n'
        '  "step_issues": [                     // реальные ошибки/недочёты хода, если есть\n'
        '    {"step": "…", "type": "…", "why": "…"}\n'
        '  ],\n'
        '  "gaps": [],                          // вероятные пробелы только если есть ошибки\n'
        '  "need_drills": false,                // предлагать ли мини-тренировку\n'
        '  "drills": [],                        // 0–3 задания, только если need_drills=true\n'
        '  "summary": "…"                       // короткий вывод по делу\n'
        "}\n"
        "Уточнения:\n"
        "- Итог «верно», если твой ответ совпадает с ученическим (целые — строго, десятичные — погрешность до 1e-3 или 1%).\n"
        "- Если ответ верный и step_issues пуст — поставь need_drills=false и не пиши drills.\n"
        "- Если не уверен в чтении цифр — укажи это в summary, оставь is_final_answer_correct = null, не придумывай ошибки."
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,          # gpt-4o-mini или gpt-4o
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
        "max_tokens": 320,              # держим короче и дешевле
    }

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, json=payload)
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
                "student_final_answer": None,
                "model_final_answer": None,
                "is_final_answer_correct": None,
                "steps_student": [],
                "step_issues": [],
                "gaps": [],
                "need_drills": False,
                "drills": [],
                "summary": "Не удалось надёжно распознать запись. Переснимите крупнее/резче."
            }
# --------------- Formatting -----------------
def format_report(j: dict) -> str:
    conf  = j.get("confidence")
    s_ans = j.get("student_final_answer")
    m_ans = j.get("model_final_answer")
    ok    = j.get("is_final_answer_correct")
    steps = j.get("steps_student") or []
    issues = j.get("step_issues") or []
    gaps   = j.get("gaps") or []
    need   = bool(j.get("need_drills"))
    drills = j.get("drills") or []
    summary = j.get("summary") or ""

    out = []

    # заголовок: верно/неверно/неопределённо
    if ok is True:
        out.append("✅ Итоговый ответ: ВЕРНО.")
    elif ok is False:
        out.append("❌ Итоговый ответ: НЕВЕРНО.")
    else:
        out.append("⚠️ Не удалось надёжно прочитать финальный ответ ученика.")

    # финальные ответы
    if s_ans is not None:
        out.append(f"Ответ ученика: {s_ans}")
    if m_ans is not None:
        out.append(f"Проверочный ответ: {m_ans}")

    # уверенность OCR
    if isinstance(conf, (int, float)):
        out.append(f"Уверенность распознавания: {round(float(conf)*100)}%")
    out.append("")

    # шаги ученика (если есть)
    if steps:
        out.append("Шаги ученика (как читаются с фото):")
        for i, s in enumerate(steps, 1):
            out.append(f"{i}) {s}")
        out.append("")

    # реальные недочёты хода
    if issues:
        out.append("Ошибки/недочёты хода решения:")
        for m in issues:
            step = m.get("step", "—")
            mtype = m.get("type", "—")
            why = m.get("why", "")
            out.append(f"• {step}: {mtype}. {why}")
        out.append("")

    # пробелы — только если есть ошибки
    if issues and gaps:
        out.append("Вероятные пробелы:")
        for g in gaps:
            out.append(f"• {g}")
        out.append("")

    # мини-тренировка — только если need_drills == True
    if need and drills:
        out.append("Мини-тренировка:")
        for d in drills:
            out.append(f"• {d}")
        out.append("")

    if summary:
        out.append(f"Итог: {summary}")

    msg = "\n".join(out).strip()
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
