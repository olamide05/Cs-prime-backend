import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import uuid4
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

# Load .env (optional)
load_dotenv()

#  Logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#  Config 
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST = "free-chatgpt-api.p.rapidapi.com"
if not RAPIDAPI_KEY:
    raise RuntimeError("RAPIDAPI_KEY environment variable is missing!")

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a friendly AI assistant for the Maynooth University CS department."
)
SESSION_TIMEOUT = timedelta(hours=1)
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "10"))

# Default FRONTEND_ORIGIN points to your domain (can be overridden by env)
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "https://csprime.vercel.app")

#  FastAPI app 
app = FastAPI(title="Maynooth CS Chatbot (RapidAPI)", version="1.0")


if FRONTEND_ORIGIN == "*" or not FRONTEND_ORIGIN:
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in FRONTEND_ORIGIN.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Session management 
class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict[str, str]] = []
        self.last_active = datetime.now()

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.messages = self.messages[-MAX_HISTORY_MESSAGES:]
        self.last_active = datetime.now()

    def get_history_text(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)


sessions: Dict[str, ChatSession] = {}

def cleanup_sessions():
    now = datetime.now()
    expired = [sid for sid, s in sessions.items() if now - s.last_active > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]
        logger.info(f"Expired session removed: {sid}")

#  Request model (keeps original 'query') 
class Question(BaseModel):
    query: str
    model: Optional[str] = None   # kept for compatibility but ignored here
    stream: Optional[bool] = False  # ignored (RapidAPI has no streaming)

#  RapidAPI call helper 
async def call_rapidapi(prompt: str, timeout: float = 25.0) -> str:
    url = f"https://{RAPIDAPI_HOST}/chat-completion-one"
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    params = {"prompt": prompt}
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            # Attempt to parse JSON
            try:
                j = resp.json()
            except Exception:
                return resp.text
            # Try common keys for reply
            for k in ("text", "response", "message", "output"):
                if isinstance(j.get(k), str) and j.get(k).strip():
                    return j.get(k).strip()
            # fallback to raw json string
            return str(j)
        except httpx.HTTPStatusError as e:
            logger.error(f"RapidAPI status error: {e} - body: {getattr(e.response, 'text', '')}")
            raise HTTPException(status_code=502, detail="Upstream API error")
        except Exception:
            logger.exception("RapidAPI request failed")
            raise HTTPException(status_code=502, detail="Upstream API error")

#  /ask endpoint (same contract) 
@app.post("/ask")
async def ask_question(q: Question, request: Request):
    cleanup_sessions()
    session_id = request.headers.get("X-Session-ID") or str(uuid4())
    if session_id not in sessions:
        sessions[session_id] = ChatSession(session_id)
        logger.info(f"Created session: {session_id}")
    session = sessions[session_id]

    # Save user message
    session.add_message("user", q.query)

    # Build prompt that includes system prompt + short history + user query
    prompt = SYSTEM_PROMPT + "\n\nConversation history:\n" + session.get_history_text() + "\n\nUser: " + q.query + "\nAssistant:"
    try:
        reply = await call_rapidapi(prompt)
    except HTTPException:
        # Friendly fallback to frontend
        return {"answer": "Sorry, I couldn't contact the assistant service right now. Please try again later.", "session_id": session_id, "timestamp": datetime.now().isoformat()}

    # Save assistant reply to session
    session.add_message("assistant", reply)
    logger.info(f"[{session_id}] user -> assistant (rapidapi)")

    return {"answer": reply, "session_id": session_id, "timestamp": datetime.now().isoformat()}

# --- health & status ---
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/status")
async def status_check():
    # RapidAPI doesn't expose model lists—keep simple fallback
    fallback = ["rapidapi-chat"]
    return {"status": "operational", "models": fallback, "timestamp": datetime.now().isoformat()}
