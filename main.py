import os
from datetime import datetime
from typing import Dict, List
from uuid import uuid4
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx  # For Google Search
from google import genai  # Gemini API client

# --- Logging setup ---
# This helps us keep track of what's happening in the app, and any errors that pop up.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load environment variables ---
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
SEARCH_API_KEY = os.environ.get("SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")

# Make sure we have our main API key set, otherwise stop the app.
if not GENAI_API_KEY:
    raise RuntimeError("GENAI_API_KEY environment variable is missing!")

# --- Gemini client setup ---
# This is the official Google client for Gemini. We'll use the 'gemini-2.0-flash' model.
client = genai.Client(api_key=GENAI_API_KEY)
MODEL_NAME = "gemini-2.0-flash"

# --- FastAPI setup ---
# Create the app and configure CORS so browsers can connect to it.
app = FastAPI(
    title="Maynooth CS Chatbot",
    description="A friendly AI assistant for Maynooth University CS Department using Gemini 2.0 Flash.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you might want to restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chat session management ---
# We keep track of ongoing chats so the AI remembers the conversation.
class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        # Add a message to the chat history and keep only the last 10 messages to save memory.
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.messages = self.messages[-10:]

    def get_history(self) -> str:
        # Format the chat history nicely so we can feed it into the AI.
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)

# Store active sessions in memory.
sessions: Dict[str, ChatSession] = {}

# --- Google Search helper ---
# If the AI thinks a search is needed, we'll use Google Custom Search to fetch results.
async def search_google(query: str) -> List[Dict[str, str]]:
    if not SEARCH_API_KEY or not SEARCH_ENGINE_ID:
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": SEARCH_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": 3
    }

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            # Return only the essentials: title, link, snippet.
            return [{"title": i.get("title"), "link": i.get("link"), "snippet": i.get("snippet")} for i in items]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# --- Request model ---
# This defines what a user sends when asking a question.
class Question(BaseModel):
    query: str

# --- Gemini API call helper ---
# Sends a prompt to Gemini and gets a response.
async def call_gemini(prompt: str) -> str:
    try:
        response = client.generate(
            model=MODEL_NAME,
            prompt=prompt,
            temperature=0.7,
            max_output_tokens=2000
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

# --- /ask endpoint ---
# Users send their questions here. We'll decide if a search is needed and return an answer.
@app.post("/ask")
async def ask_question(q: Question, request: Request):
    session_id = request.headers.get("X-Session-ID") or str(uuid4())
    
    # Create a new session if this is the first message
    if session_id not in sessions:
        sessions[session_id] = ChatSession(session_id)
        logger.info(f"Created new session: {session_id}")

    session = sessions[session_id]
    session.add_message("user", q.query)

    # Prepare the prompt for Gemini
    initial_prompt = f"""
You are a friendly AI assistant for the Maynooth University CS department.

Conversation history:
{session.get_history()}

User: {q.query}
Assistant:"""

    gemini_response = await call_gemini(initial_prompt)

    # Check if Gemini wants us to do a search
    if gemini_response.strip().startswith("TOOL_CODE:SEARCH:"):
        search_query = gemini_response.replace("TOOL_CODE:SEARCH:", "").strip()
        search_results = await search_google(search_query)
        
        if search_results:
            search_context = "\n\n".join(
                f"Title: {r['title']}\nLink: {r['link']}\nSnippet: {r['snippet']}" 
                for r in search_results
            )
            follow_up_prompt = f"""
You are a friendly AI assistant for the Maynooth CS department.

Use the following search results to answer the user's question:
{search_context}

Conversation history:
{session.get_history()}

User's question: {q.query}
Assistant:"""
            final_response = await call_gemini(follow_up_prompt)
        else:
            final_response = "Sorry, I couldn't find relevant information via search."
    else:
        final_response = gemini_response

    session.add_message("assistant", final_response)

    return {
        "answer": final_response,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat()
    }

# --- Health & Status endpoints ---
@app.get("/health")
async def health_check():
    # Simple endpoint to make sure the app is running
    return {"status": "healthy"}

@app.get("/status")
async def status_check():
    # Check Gemini models and report status
    try:
        models = [m["name"] for m in client.models.list()]
        return {"status": "operational", "models": models, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"status": "error", "detail": str(e), "timestamp": datetime.now().isoformat()}
