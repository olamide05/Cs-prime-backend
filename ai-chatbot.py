import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request, requests
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Configuration validation
def validate_config() -> Dict[str, str]:
    required_keys = ["GEMINI_API_KEY", "SEARCH_API_KEY", "SEARCH_ENGINE_ID"]
    config = {}
    for key in required_keys:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        config[key] = value
    return config

class ChatSession:
    def __init__(self, max_messages: int = 50):
        self.messages: List[Dict[str, str]] = []
        self.max_messages = max_messages
        self.last_access = datetime.now()

    def add_message(self, role: str, text: str) -> None:
        if len(self.messages) >= self.max_messages:
            self.messages.pop(0)
        self.messages.append({"role": role, "text": text})
        self.last_access = datetime.now()

class SessionManager:
    def __init__(self, max_sessions: int = 1000):
        self.sessions: Dict[str, ChatSession] = {}
        self.max_sessions = max_sessions

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> None:
        current_time = datetime.now()
        sessions_to_remove = [
            sid for sid, session in self.sessions.items()
            if (current_time - session.last_access).total_seconds() > max_age_hours * 3600
        ]
        for sid in sessions_to_remove:
            del self.sessions[sid]

    def get_or_create_session(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            if len(self.sessions) >= self.max_sessions:
                self.cleanup_old_sessions()
            self.sessions[session_id] = ChatSession()
        return self.sessions[session_id]

load_dotenv()

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://csprime.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session memory store
chat_sessions = {}

# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(q: Question, request: Request):
    # Step 1: Search Google
    context = ""
    try:
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": SEARCH_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": q.query,
            "num": 3
        }
        search_res = requests.get(search_url, params=params)
        search_res.raise_for_status()
        search_data = search_res.json()

        if search_data.get("items"):
            snippets = [item.get("snippet", "") for item in search_data["items"][:3]]
            context = "\n\n".join(snippets)
        else:
            context = "No Google results found."
    except Exception as e:
        context = "Google Search failed."

    # Step 2: Load JSON knowledge (optional)
    json_snippet = ""
    json_path = Path("module_info.json")
    if json_path.exists():
        try:
            with json_path.open() as f:
                maynooth_data = json.load(f)
                json_snippet = json.dumps(maynooth_data, indent=2)
        except Exception:
            json_snippet = "Module data could not be loaded."
    else:
        json_snippet = "No structured module data available."

    # Step 3: Manage chat memory
    session_id = request.headers.get("X-Session-ID") or str(uuid4())
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    history = chat_sessions[session_id]
    history.append({"role": "user", "text": q.query})

    # Step 4: Construct the prompt
    prompt = (
        f"User asked: {q.query}\n\n"
        f"Here are some relevant search results:\n{context}\n\n"
        f"Here is structured data from Maynooth University:\n{json_snippet}\n\n"
        "Now, using the information above and your own reasoning ability, "
        "give a helpful, clear answer as if you're explaining to a 15-year-old.\n"
        "If something is not directly stated, try to infer it from the context.\n"
        "You are a friendly, smart AI assistant that likes to help students.\n"
        "Have normal conversations to the best of your ability even if the question isn't about computer science — "
        "but try to relate it to tech or learning if possible.\n"
        "Here's the chat so far:\n\n"
    )

    for msg in history:
        who = "User" if msg["role"] == "user" else "Assistant"
        prompt += f"{who}: {msg['text']}\n"

    prompt += "\nAssistant:"

    # Step 5: Call Gemini
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,
            "topK": 50,
            "topP": 0.95
        }
    }

    try:
        gemini_res = requests.post(gemini_url, headers=headers, json=payload)
        gemini_res.raise_for_status()
        gemini_data = gemini_res.json()
        reply = gemini_data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        reply = "I'm here to help, but I couldn't generate a full response right now. Try asking in a simpler way!"

    # Save reply to memory
    history.append({"role": "assistant", "text": reply})
    return {"answer": reply, "session_id": session_id}