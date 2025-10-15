# app.py
import os
import io
import json
import base64
import time
import asyncio
import traceback
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, Request
import httpx
from PIL import Image, ImageOps, ImageFilter

# ================= CONFIG =================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # или "gpt-4o"

# Профили сжатия (дешёвый и «спасательный» дорогой)
LOW_MAX_SIDE = 640
LOW_JPEG_Q   = 60
LOW_DETAIL   = "low"

HIGH_MAX_SIDE = 768
HIGH_JPEG_Q   = 72
HIGH_DETAIL   = "high"

API_URL  = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_URL = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

APP_VERSION = "build-2025-10-16-03"
print(f"[BOOT] {APP_VERSION} | model={OPENAI_MODEL} | LOW={LOW_MAX_SIDE}/{LOW_JPEG_Q}/{LOW_DETAIL} | HIGH={HIGH_MAX_SIDE}/{HIGH_JPEG_Q}/{HIGH_DETAIL}")

# ================ APP ======================
app = FastAPI()
DOWNLOAD_DIR = Path("/tmp/tg_photos")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

SEEN: dict[int, float] = {}  # anti-dup

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

# ---------------- Image helpers -------------------
def _trim_whitespace(img_l: Image.Image, thresh: int = 245) -> Image.Image:
    bw = img_l.point(lambda p: 255 if p > thresh else 0, mode="L")
    bbox = bw.getbbox()
    if bbox:
        return img_l.crop(bbox)
    return img_l

def encode_image_b64(path: Path, max_side: int, quality: int, apply_median: bool) -> Tuple[str, str]:
    """
    Возвращает (base64, лог_строка). Для высокого профиля медиан-фильтр отключаем,
    чтобы не смазывать тонкие штрихи.
    """
    img = Image.open(path).convert("L")
    w0, h0 = img.size

    img = _trim_whitespace(img, thresh=245)

    w1, h1 = img.size
    if apply_median:
        img = img.filter(ImageFilter.MedianFilter(size=3))  # только в дешёвом прогоне

    scale = max(w1, h1) / max_side if max(w1, h1) > max_side else 1.0
    if scale > 1.0:
        img = img.resize((int(w1/scale), int(h1/scale)), Image.LANCZOS)

    img = ImageOps.autocontrast(img, cutoff=1)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    jpeg_bytes = buf.getvalue()
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")

    log = (f"original {w0}x{h0} -> trimmed {w1}x{h1} -> resized {img.size[0]}x{img.size[1]}, "
           f"jpeg={len(jpeg_bytes)/1024:.1f}KB, b64_len={len(b64)}, MAX_SIDE={max_side}, Q={quality}, "
           f"median={'on' if apply_median else 'off'}")
    return b64, log

# ---------------- OpenAI call ---------------------
async def call_openai(img_b64: str, detail: str) -> dict:
    system_prompt = """Ты — строгий и доброжелательный учитель математики 7–9 классов.
1) Считай финальный ответ ученика (если виден).
2) Реши задачу сам и получи свой финальный ответ.
3) Сравни: целые — строго; десятичные — с погрешностью 1e-3 или 1%.
4) Указывай ТОЛЬКО реальные ошибки хода (неразборчиво ≠ ошибка).
5) Если итог верный и ошибок хода нет — не предлагай тренировку.
6) Если видимость плохая — честно укажи это и не придумывай ошибки.
Ответ строго в JSON.
"""

    user_prompt = """Верни РОВНО такой JSON:
{
  "confidence": 0.0,
  "student_final_answer": null,
  "model_final_answer": null,
  "is_final_answer_correct": null,
  "steps_student": [],
  "step_issues": [],
  "gaps": [],
  "need_drills": false,
  "drills": [],
  "summary": "…"
}
Правила вывода:
- Если итог верный и нет ошибок хода — need_drills=false и drills пуст.
- Если запись неразборчива — укажи это в summary и не придумывай ошибки.
"""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": detail
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }

    start = time.time()
    print(f"[AI] model={OPENAI_MODEL}, detail={detail}, max_tokens=300, temp=0.0, image_b64_len={len(img_b64)}")

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, json=payload)
        if r.status_code != 200:
            print("OpenAI ERROR", r.status_code, r.text)
        r.raise_for_status()
        data = r.json()

    try:
        u = data.get("usage", {})
        print(f"[AI] usage: prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} total={u.get('total_tokens')} time={(time.time()-start):.2f}s")
    except Exception:
        pass

    # Парс ответа в JSON
    raw = data["choices"][0]["message"]["content"]
    try:
        return json.loads(raw)
    except Exception:
        try:
            return json.loads((raw or "").strip().strip("`").strip())
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

def is_unreadable(result: dict) -> bool:
    """Критерии, когда считаем, что модель не смогла прочитать."""
    conf = result.get("confidence")
    summary = (result.get("summary") or "").lower()
    no_content = not result.get("steps_student") and result.get("student_final_answer") in (None, "", "null")
    conf_bad = (isinstance(conf, (int, float)) and conf < 0.4)
    mentions_blurry = any(word in summary for word in ["неразборч", "плохо видно", "размыто", "не вижу"])
    return no_content or conf_bad or mentions_blurry

async def analyze_math_image(image_path: Path) -> dict:
    """
    Двухступенчатый анализ:
      Pass A (дешёвый): 640/60, detail=low, с медиан-фильтром.
      Если нечитабельно -> Pass B (дорогой): 768/72, detail=high, без медиан-фильтра.
    """
    # Pass A
    b64_low, log_low = encode_image_b64(image_path, LOW_MAX_SIDE, LOW_JPEG_Q, apply_median=True)
    print(f"[IMG/LOW] {log_low}")
    res_low = await call_openai(b64_low, LOW_DETAIL)

    if not is_unreadable(res_low):
        res_low["_profile"] = "low"
        return res_low

    print("[FALLBACK] low-profile unreadable → retry with HIGH profile")
    # Pass B
    b64_high, log_high = encode_image_b64(image_path, HIGH_MAX_SIDE, HIGH_JPEG_Q, apply_median=False)
    print(f"[IMG/HIGH] {log_high}")
    res_high = await call_openai(b64_high, HIGH_DETAIL)
    res_high["_profile"] = "high"
    return res_high

# ---------------- Formatting ---------------------
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
    profile = j.get("_profile", "low")

    out = [f"Профиль анализа: {('экономичный' if profile=='low' else 'повышенной точности')}."]
    if ok is True:
        out.append("✅ Итоговый ответ: ВЕРНО.")
    elif ok is False:
        out.append("❌ Итоговый ответ: НЕВЕРНО.")
    else:
        out.append("⚠️ Не удалось надёжно прочитать финальный ответ ученика.")

    if s_ans is not None:
        out.append(f"Ответ ученика: {s_ans}")
    if m_ans is not None:
        out.append(f"Проверочный ответ: {m_ans}")
    if isinstance(conf, (int, float)):
        try:
            out.append(f"Уверенность распознавания: {round(float(conf)*100)}%")
        except Exception:
            pass
    out.append("")

    if steps:
        out.append("Шаги ученика (как читаются с фото):")
        for i, s in enumerate(steps, 1):
            out.append(f"{i}) {s}")
        out.append("")

    if issues:
        out.append("Ошибки/недочёты хода решения:")
        for m in issues:
            step = m.get("step", "—")
            mtype = m.get("type", "—")
            why = m.get("why", "")
            out.append(f"• {step}: {mtype}. {why}")
        out.append("")

    if issues and gaps:
        out.append("Вероятные пробелы:")
        for g in gaps:
            out.append(f"• {g}")
        out.append("")

    if need and drills:
        out.append("Мини-тренировка:")
        for d in drills:
            out.append(f"• {d}")
        out.append("")

    if summary:
        out.append(f"Итог: {summary}")

    msg = "\n".join(out).strip()
    return msg[:4000] if len(msg) > 4000 else msg

# --------------- Background task ---------------
async def process_photo(chat_id: int, reply_to: Optional[int], file_id: str):
    try:
        file_path = await tg_get_file(file_id)
        local_path = await tg_download_file(file_path)

        report = await analyze_math_image(local_path)
        text_report = format_report(report) or \
            "Не получилось сформировать отчёт. Попробуйте переснять фото крупнее/резче."

        await tg_send_message(chat_id, text_report)
    except httpx.HTTPError as e:
        print("HTTP error during analysis:", e)
        await tg_send_message(chat_id, "Не удалось связаться с сервисом анализа. Попробуй позже.")
    except Exception as e:
        print("Analysis error:", e)
        print(traceback.format_exc())
        await tg_send_message(
            chat_id,
            "Не удалось проанализировать фото 😕\n"
            "Сделай снимок ближе и чётче, по одному заданию на фото."
        )

# --------------------- Routes ---------------------
@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/debug")
def debug():
    return {
        "version": APP_VERSION,
        "model": OPENAI_MODEL,
        "LOW": {"MAX_SIDE": LOW_MAX_SIDE, "Q": LOW_JPEG_Q, "detail": LOW_DETAIL},
        "HIGH": {"MAX_SIDE": HIGH_MAX_SIDE, "Q": HIGH_JPEG_Q, "detail": HIGH_DETAIL},
    }

@app.post("/webhook/telegram")
async def tg_webhook(request: Request):
    global SEEN
    try:
        update = await request.json()
        message = update.get("message") or update.get("edited_message")
        if not message:
            return {"ok": True}

        chat_id = message["chat"]["id"]
        message_id = message.get("message_id")
        text = message.get("text") or ""
        photos = message.get("photo") or []

        now = time.time()
        if message_id in SEEN and now - SEEN[message_id] < 60:
            print(f"[DEDUP] skip message_id {message_id}")
            return {"ok": True}
        SEEN[message_id] = now

        if text.startswith("/start"):
            asyncio.create_task(
                tg_send_message(
                    chat_id,
                    "Привет! Отправь фото задачи (лучше по одной на фото). "
                    "Я проверю итог, отмечу реальные недочёты и дам рекомендации.\n\n"
                    "Лайфхак: снимай крупно и при хорошем свете.",
                    reply_to=message_id,
                )
            )
            return {"ok": True}

        if photos:
            largest = photos[-1]
            file_id = largest["file_id"]
            asyncio.create_task(tg_send_message(chat_id, "Фото получено ✅ Анализирую…", reply_to=message_id))
            asyncio.create_task(process_photo(chat_id, message_id, file_id))
            return {"ok": True}

        if text:
            asyncio.create_task(tg_send_message(chat_id, f"Я получил: {text}", reply_to=message_id))
            return {"ok": True}

        asyncio.create_task(tg_send_message(chat_id, "Пришли /start или отправь фото."))
        return {"ok": True}

    except Exception as e:
        print("Webhook handler error:", e)
        print(traceback.format_exc())
        return {"ok": True}
