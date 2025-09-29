import os
from datetime import datetime
from typing import Dict, List
from uuid import uuid4
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx # For async HTTP requests

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Environment Variables
# load_dotenv("keys.env")  # Uncomment if using a .env file locally
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SEARCH_API_KEY = os.environ.get("SEARCH_API_KEY")  # ✅ fixed
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")  # ✅ fixed

# Configuration 
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"  # ✅ fixed endpoint
TIMEOUT = 30  # seconds for Gemini API calls
SEARCH_TIMEOUT = 10
SEARCH_RESULTS_COUNT = 3

# FastAPI App Setup 
app = FastAPI(
    title="Maynooth University CS Chatbot",
    description="An AI assistant focused on the Computer Science department at Maynooth University, powered by Gemini 1.5 Pro with search capabilities.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Chat Session Management 
class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.messages = self.messages[-10:]  # Keep last 10 messages

    def get_history(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)

sessions: Dict[str, ChatSession] = {}

# Gemini API Interaction 
async def call_gemini(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            response = await client.post(
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
                }
            )
            response.raise_for_status()
            data = response.json()

            if "candidates" in data and data["candidates"] and \
               "content" in data["candidates"][0] and \
               "parts" in data["candidates"][0]["content"] and \
               data["candidates"][0]["content"]["parts"]:
                return data["candidates"][0]["content"]["parts"][0]["text"]

            logger.error(f"Gemini API response missing expected structure: {data}")
            raise ValueError("Invalid Gemini API response format")

        except httpx.RequestError as e:
            logger.error(f"Gemini API network error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Could not connect to Gemini API.")
        except httpx.HTTPStatusError as e:
            logger.error(f"Gemini API HTTP error {e.response.status_code}: {e.response.text}", exc_info=True)
            raise HTTPException(status_code=e.response.status_code, detail=f"Gemini API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Unexpected Gemini error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Unexpected Gemini API error.")

# --- Google Custom Search ---
async def search_google(query: str) -> List[Dict[str, str]]:
    if not SEARCH_API_KEY or not SEARCH_ENGINE_ID:
        logger.warning("SEARCH_API_KEY or SEARCH_ENGINE_ID not set. Skipping search.")
        return []

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": SEARCH_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": SEARCH_RESULTS_COUNT
    }
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        try:
            response = await client.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            return [
                {"title": item.get("title"), "link": item.get("link"), "snippet": item.get("snippet")}
                for item in data.get("items", [])
            ]
        except Exception as e:
            logger.error(f"Google Search API error for query '{query}': {e}", exc_info=True)
            return []

# --- Pydantic Model ---
class Question(BaseModel):
    query: str

# --- API Endpoints ---
@app.post("/ask")
async def ask_question(q: Question, request: Request):
    session_id = request.headers.get("X-Session-ID") or str(uuid4())

    if session_id not in sessions:
        sessions[session_id] = ChatSession(session_id)
    session = sessions[session_id]

    session.add_message("user", q.query)

    try:
        initial_prompt = f"""You are an AI assistant for Maynooth University CS dept.
If the question is within scope, answer it. 
If external info is needed, respond ONLY with:
TOOL_CODE:SEARCH: <query>"""

        gemini_initial_response = await call_gemini(initial_prompt)
        final_response_text = ""

        if gemini_initial_response.strip().startswith("TOOL_CODE:SEARCH:"):
            search_query = gemini_initial_response.replace("TOOL_CODE:SEARCH:", "").strip()
            search_results = await search_google(search_query)

            if search_results:
                search_context = "\n\n".join([
                    f"Title: {r['title']}\nLink: {r['link']}\nSnippet: {r['snippet']}"
                    for r in search_results
                ])
                follow_up_prompt = f"""Use the following results to answer:
{search_context}

User question: {q.query}
Answer:"""
                final_response_text = await call_gemini(follow_up_prompt)
            else:
                final_response_text = "I couldn't find relevant info for your query."
        else:
            final_response_text = gemini_initial_response

        session.add_message("assistant", final_response_text)

        return {
            "answer": final_response_text,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /ask: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/status")
async def get_status():
    try:
        test_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(test_url)

        status_text = "operational" if response.status_code == 200 else "degraded"
        return {"status": status_text, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "error", "detail": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# --- Entry Point ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # ✅ dynamic port for Render
    uvicorn.run(app, host="0.0.0.0", port=port)
