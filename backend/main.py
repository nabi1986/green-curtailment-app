from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import random
import os, requests

app = FastAPI()

# --- CORS (keep simple for dev/demo) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
#  Providers & Settings
# =======================

# --- Hugging Face (primary) ---
HF_TOKEN = os.getenv("HF_TOKEN")  # set in Render → Environment
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2").strip()
HF_URL_CHAT = "https://api-inference.huggingface.co/v1/chat/completions"   # OpenAI-compatible
HF_URL_MODELS = f"https://api-inference.huggingface.co/models/{HF_MODEL}"  # legacy
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "60"))

# --- OpenRouter (fallback) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OR_MODEL = os.getenv("OR_MODEL", "meta-llama/llama-3.1-8b-instruct").strip()
OR_HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"} if OPENROUTER_API_KEY else {}

SYSTEM_PROMPT = (
    "You are an assistant that helps minimize renewable energy curtailment. "
    "Always respond in English only, regardless of the user's language. "
    "Be concise and practical: battery charge/discharge, flexible load shifting, "
    "time-of-use strategies, ancillary markets, storage sizing, and bidding tactics. "
    "Limit answers to 4–6 sentences unless explicitly asked for more detail."
)

# =======================
#  Provider Callers
# =======================

def ask_openrouter(user_msg: str, max_new_tokens: int = 220) -> str:
    """Call OpenRouter (OpenAI-compatible)."""
    if not OPENROUTER_API_KEY:
        return "⚠️ OPENROUTER_API_KEY is not set."
    payload = {
        "model": OR_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.3,
    }
    try:
        r = requests.post(OPENROUTER_URL, headers=OR_HEADERS, json=payload, timeout=HF_TIMEOUT)
    except Exception as e:
        return f"❌ OpenRouter connection error: {e}"
    if r.status_code != 200:
        return f"❌ OpenRouter Error {r.status_code}: {r.text}"
    j = r.json()
    try:
        return (j["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return str(j)

def ask_hf(user_msg: str, max_new_tokens: int = 200) -> tuple[str, bool]:
    """
    Try Hugging Face (Chat, then Legacy).
    Returns (reply, ok_bool). If ok_bool=False, caller can try fallback.
    """
    if HF_TOKEN:
        # 1) Chat Completions API
        chat_payload = {
            "model": HF_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.3,
            "stream": False,
        }
        try:
            r = requests.post(HF_URL_CHAT, headers=HF_HEADERS, json=chat_payload, timeout=HF_TIMEOUT)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, dict) and "choices" in j and j["choices"]:
                    return (j["choices"][0]["message"]["content"] or "").strip(), True
            else:
                # اگر مدل روی chat endpoint در دسترس نبود، روی legacy می‌رویم
                if r.status_code not in (404, 422):
                    return f"❌ HF Chat Error {r.status_code}: {r.text}", False
        except Exception:
            # اگر شبکه مشکل داشت، به legacy می‌رویم
            pass

        # 2) Legacy /models endpoint
        legacy_payload = {
            "inputs": f"{SYSTEM_PROMPT}\n\nUser: {user_msg}\nAssistant:",
            "parameters": {"max_new_tokens": max_new_tokens, "return_full_text": False},
        }
        try:
            r2 = requests.post(HF_URL_MODELS, headers=HF_HEADERS, json=legacy_payload, timeout=HF_TIMEOUT)
        except Exception as e:
            return f"❌ Connection error to Hugging Face: {e}", False

        if r2.status_code != 200:
            # اجازه بدهیم fallback تصمیم بگیرد
            return f"❌ HF Error {r2.status_code} at {HF_URL_MODELS}: {r2.text}", False

        data = r2.json()
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return (data[0]["generated_text"] or "").strip(), True
        return str(data), True

    # اگر HF_TOKEN نداریم، مستقیماً می‌گوییم ok=False که fallback اجرا شود
    return "⚠️ HF token not set.", False

def ask_ai(user_msg: str, max_new_tokens: int = 200) -> str:
    """Main router: try HF first; if it fails, use OpenRouter."""
    hf_reply, hf_ok = ask_hf(user_msg, max_new_tokens=max_new_tokens)
    if hf_ok:
        return hf_reply

    # اگر HF جواب نداد/404 شد و OpenRouter داریم، از OpenRouter استفاده کن
    if OPENROUTER_API_KEY:
        or_reply = ask_openrouter(user_msg, max_new_tokens=max_new_tokens)
        return or_reply

    # اگر هیچ‌کدام در دسترس نبود
    return hf_reply  # خطای HF را برگردانیم که مشخص باشد چه شد

# =======================
#  API & WebSocket
# =======================

class BidRequest(BaseModel):
    site_id: str
    horizon_hours: int = 24
    features: dict | None = None  # grid/weather features

@app.get("/health")
def health():
    providers = {
        "huggingface": bool(HF_TOKEN),
        "openrouter": bool(OPENROUTER_API_KEY),
    }
    return {"ok": True, "model": HF_MODEL, "providers": providers}

@app.get("/forecast")
def forecast(site_id: str = "site-A", horizon_hours: int = 24):
    """Dummy time series for generation/price/risk."""
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    points = []
    base_gen = 42.0
    base_price = 70.0
    for i in range(horizon_hours):
        t = now + timedelta(hours=i)
        gen = max(0.0, base_gen + 15 * random.random() - 5)          # MW
        price = max(0.0, base_price + 20 * random.random() - 10)     # €/MWh
        curtailment_risk = max(0.0, min(1.0, 0.3 + 0.5 * random.random()))
        points.append({
            "timestamp": t.isoformat() + "Z",
            "gen_mw": round(gen, 2),
            "price_eur_mwh": round(price, 2),
            "curtailment_prob": round(curtailment_risk, 2),
        })
    return {"site_id": site_id, "points": points}

@app.post("/bid")
def bid(req: BidRequest):
    """Dummy bid output (replace with your real ML model)."""
    qty = round(30 + 10 * random.random(), 2)        # MW
    price = round(60 + 15 * random.random(), 2)      # €/MWh
    return {"site_id": req.site_id, "qty_mw": qty, "price_eur_mwh": price}

@app.websocket("/ws/chat")
async def chat_socket(ws: WebSocket):
    await ws.accept()
    welcome = "🤖 Connected. Primary: Hugging Face; Fallback: OpenRouter. (English only)"
    if not HF_TOKEN and not OPENROUTER_API_KEY:
        welcome += " ⚠️ No HF_TOKEN or OPENROUTER_API_KEY is set in Environment."
    await ws.send_text(welcome)

    try:
        while True:
            user_msg = await ws.receive_text()
            reply = ask_ai(user_msg, max_new_tokens=200)
            if not reply or reply.isspace():
                reply = "No response from providers. Check HF/OpenRouter credentials."
            await ws.send_text(reply)
    except WebSocketDisconnect:
        pass
