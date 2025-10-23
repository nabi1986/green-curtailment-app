from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import random

app = FastAPI()

# اجازه دسترسی فرانت‌اند (در توسعه ساده نگه داریم)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BidRequest(BaseModel):
    site_id: str
    horizon_hours: int = 24
    features: dict | None = None  # grid/weather features

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/forecast")
def forecast(site_id: str = "site-A", horizon_hours: int = 24):
    """خروجی ساختگی: سری زمانی تولید/قیمت برای نمونه."""
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    points = []
    base_gen = 42.0
    base_price = 70.0
    for i in range(horizon_hours):
        t = now + timedelta(hours=i)
        gen = max(0.0, base_gen + 15*random.random() - 5)           # MW
        price = max(0.0, base_price + 20*random.random() - 10)      # €/MWh
        curtailment_risk = max(0.0, min(1.0, 0.3 + 0.5*random.random()))
        points.append({
            "timestamp": t.isoformat() + "Z",
            "gen_mw": round(gen, 2),
            "price_eur_mwh": round(price, 2),
            "curtailment_prob": round(curtailment_risk, 2),
        })
    return {"site_id": site_id, "points": points}

@app.post("/bid")
def bid(req: BidRequest):
    """مدل بید نمونه: قیمت/مقدار با در نظر گرفتن ریسک curtailment ساختگی."""
    # معمولاً اینجا از مدل واقعی استفاده می‌کنید (ML)
    qty = round(30 + 10*random.random(), 2)        # MW
    price = round(60 + 15*random.random(), 2)      # €/MWh
    return {"site_id": req.site_id, "qty_mw": qty, "price_eur_mwh": price}

# --- WebSocket چت (مدل دوم) ---
@app.websocket("/ws/chat")
async def chat_socket(ws: WebSocket):
    await ws.accept()
    try:
        await ws.send_text("🤖 Hi! If curtailment risk is high, tell me and I'll suggest actions.")
# ...
        while True:
            msg = await ws.receive_text()
            # منطق ساده‌ی نمونه (به‌جای LLM)
            reply = "To reduce curtailment impact you can charge batteries, shift flexible loads, or perform peak shaving."
            if "60" in msg or "بالا" in msg or "high" in msg:
                reply = "Risk is high (>60%). Options: 1) charge batteries 2) increase flexible process consumption 3) sell flexibility to ancillary markets."
            await ws.send_text(reply)
    except WebSocketDisconnect:
        pass
