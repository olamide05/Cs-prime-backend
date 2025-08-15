import os
from datetime import datetime
from typing import Dict, List
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv("keys.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
# Updated Configuration
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_API_KEY}"  # Updated model
TIMEOUT = 30  # seconds

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Keep last 5 messages
        self.messages = self.messages[-5:]

    def get_history(self):
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)

sessions = {}

def call_gemini(prompt: str):
    try:
        response = requests.post(
            API_URL,
            json={
                "contents": [{
                    "parts": [{"text": prompt}],
                    "role": "user"
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 2000
                }
            },
            timeout=TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"API Error: {str(e)}")
        raise

class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(q: Question, request: Request):
    session_id = request.headers.get("X-Session-ID") or str(uuid4())

    if session_id not in sessions:
        sessions[session_id] = ChatSession(session_id)
    session = sessions[session_id]

    session.add_message("user", q.query)

    try:
        prompt = f"""You are a helpful assistant for maynooth university but you are focused on the computer science department . Conversation history:
        
{session.get_history()}

User: {q.query}
Assistant:"""

        response = call_gemini(prompt)
        session.add_message("assistant", response)

        return {
            "answer": response,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception:
        return {
            "answer": "I'm currently experiencing technical difficulties. Please try again later.",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/status")
async def get_status():
    try:
        test_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        response = requests.get(test_url, timeout=10)
        return {
            "status": "operational" if response.status_code == 200 else "degraded",
            "models": [m["name"] for m in response.json().get("models", [])],
            "timestamp": datetime.now().isoformat()
        }
    except Exception:
        return {
            "status": "offline",
            "timestamp": datetime.now().isoformat()
        }
@app.get("/health")
async def health_check():
    logger.info("Health check triggered")
    return {"status": "healthy"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)