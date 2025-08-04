import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# New imports for semantic fallback
from sentence_transformers import SentenceTransformer, util
import torch

# Load environment variables
load_dotenv()

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
    def __init__(self, max_messages: int = 20):
        self.messages: List[Dict[str, str]] = []
        self.max_messages = max_messages
        self.last_access = datetime.now()

    def add_message(self, role: str, text: str) -> None:
        if len(self.messages) >= self.max_messages:
            self.messages.pop(0)
        self.messages.append({"role": role, "text": text})
        self.last_access = datetime.now()

class SessionManager:
    def __init__(self, max_sessions: int = 500):
        self.sessions: Dict[str, ChatSession] = {}
        self.max_sessions = max_sessions

    def cleanup_old_sessions(self, max_age_hours: int = 12) -> None:
        current_time = datetime.now()
        sessions_to_remove = [
            sid for sid, session in self.sessions.items()
            if (current_time - session.last_access).total_seconds() > max_age_hours * 3600
        ]
        for sid in sessions_to_remove:
            del self.sessions[sid]

    def get_or_create_session(self, session_id: str) -> ChatSession:
        self.cleanup_old_sessions()
        if session_id not in self.sessions:
            if len(self.sessions) >= self.max_sessions:
                raise HTTPException(status_code=429, detail="Too many active sessions")
            self.sessions[session_id] = ChatSession()
        return self.sessions[session_id]

app = FastAPI()

# Allow CORS for the frontend
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

# Initialize session manager
session_manager = SessionManager()

# Validate config
try:
    config = validate_config()
    GEMINI_API_KEY = config["GEMINI_API_KEY"]
    SEARCH_API_KEY = config["SEARCH_API_KEY"]
    SEARCH_ENGINE_ID = config["SEARCH_ENGINE_ID"]
except ValueError as e:
    raise RuntimeError(f"Configuration error: {str(e)}")

# Load Maynooth University knowledge base
MAYNOOTH_KNOWLEDGE = ""
MODULE_DATA = []
try:
    json_path = Path("/app/data/module_info.json")  # Use an absolute container path
    if json_path.exists():
        with json_path.open() as f:
            MODULE_DATA = list(json.load(f).values())
            MAYNOOTH_KNOWLEDGE = "\n".join(
                f"Module {mod['scrapedModuleCodeFromPage']}: {mod['title']}\n"
                f"Credits: {mod['credits']}\n"
                f"Overview: {mod['overview']}\n"
                f"Learning Outcomes: {', '.join(mod['learningOutcomes'])}\n"
                for mod in MODULE_DATA
            )
except Exception as e:
    print(f"Warning: Could not load Maynooth knowledge base: {str(e)}")
    MAYNOOTH_KNOWLEDGE = "No module data available"

# Initialize sentence-transformers model & embeddings for fallback
print("Loading sentence-transformers model for fallback...")
model = SentenceTransformer("all-MiniLM-L6-v2")
module_texts = []
if MODULE_DATA:
    module_texts = [
        f"{mod['scrapedModuleCodeFromPage']}: {mod['title']}. Overview: {mod['overview']}. Learning Outcomes: {', '.join(mod['learningOutcomes'])}"
        for mod in MODULE_DATA
    ]
    module_embeddings = model.encode(module_texts, convert_to_tensor=True)
else:
    module_embeddings = None

def semantic_fallback_answer(user_query: str) -> str:
    if not module_embeddings:
        return "Sorry, I couldn’t find any relevant module information."
    query_emb = model.encode(user_query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_emb, module_embeddings)[0]
    top_idx = int(torch.argmax(cos_scores))
    score = cos_scores[top_idx].item()
    if score < 0.4:  # threshold for similarity; adjust if needed
        return "Sorry, I couldn’t find any relevant module information."
    return module_texts[top_idx]

class Question(BaseModel):
    query: str

def should_perform_search(response_text: str, query: str) -> bool:
    # ... same as before (unchanged) ...
    uncertainty_phrases = [
        "don't know", "not sure", "couldn't find", "no information",
        "need to look", "don't have", "unable to answer", "can't find",
        "doesn't appear", "not available in my knowledge"
    ]

    current_event_terms = [
        "current", "recent", "this year", str(datetime.now().year),
        "now", "latest", "upcoming", "newest"
    ]

    fact_based_terms = [
        "statistics", "numbers", "data", "research",
        "study", "survey", "percentage", "how many"
    ]

    location_terms = [
        "where is", "location of", "find the", "directions to",
        "map of", "how to get to"
    ]

    # Check if the response indicates uncertainty
    if any(phrase.lower() in response_text.lower() for phrase in uncertainty_phrases):
        return True

    # Check a query for terms that likely need fresh information
    query_lower = query.lower()
    if (any(term in query_lower for term in current_event_terms) or \
            (any(term in query_lower for term in fact_based_terms)) or \
            (any(term in query_lower for term in location_terms))):
        return True

    return False

def perform_google_search(query: str) -> str:
    # ... same as before (unchanged) ...
    try:
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": SEARCH_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": f"{query} site:maynoothuniversity.ie",
            "num": 3
        }
        search_res = requests.get(search_url, params=params)
        search_res.raise_for_status()
        search_data = search_res.json()

        if not search_data.get("items"):
            return "No relevant results found on Maynooth University website."

        results = []
        for item in search_data["items"][:3]:
            results.append(
                f"Title: {item.get('title', 'No title')}\n"
                f"URL: {item.get('link', 'No URL')}\n"
                f"Content: {item.get('snippet', 'No snippet available')}\n"
            )
        return "\n\n".join(results)
    except Exception as e:
        print(f"Search failed: {str(e)}")
        return "Unable to perform search at this time."

def call_gemini(prompt: str) -> str:
    # ... same as before (unchanged) ...
    try:
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}],
                "role": "user"
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.9,
                "maxOutputTokens": 1024
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]
        }

        response = requests.post(gemini_url, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()

        if "candidates" not in response_data or not response_data["candidates"]:
            raise ValueError("No candidates in response")

        return response_data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        raise

@app.post("/ask")
async def ask_question(q: Question, request: Request):
    sources_used = ["bot_knowledge"]
    try:
        session_id = request.headers.get("X-Session-ID") or str(uuid4())
        session = session_manager.get_or_create_session(session_id)

        session.add_message("user", q.query)

        initial_prompt = f"""You are MU Bot, the official AI assistant for Maynooth University.
Your knowledge includes:
1. General information about Maynooth University
2. Academic programs and modules
3. Campus facilities and services
4. Student life information
5. General knowledge about Ireland and education

Current Maynooth University module information:
{MAYNOOTH_KNOWLEDGE if MAYNOOTH_KNOWLEDGE else "No module data available"}

Conversation history:
"""
        for msg in session.messages[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            initial_prompt += f"{role}: {msg['text']}\n"

        initial_prompt += f"\nAssistant: Please respond to this query about '{q.query}' " \
                          "focusing on Maynooth University context where appropriate. " \
                          "If you're unsure or need current information, indicate that."

        try:
            initial_response = call_gemini(initial_prompt)

            needs_search = should_perform_search(initial_response, q.query)
            sources_used = ["bot_knowledge"]

            if needs_search:
                search_results = perform_google_search(q.query)
                enhanced_prompt = f"""{initial_prompt}
                
Additional Context from Maynooth University website:
{search_results}

Please revise your response incorporating this new information where relevant:"""
                final_response = call_gemini(enhanced_prompt)
                sources_used.append("web_search")
            else:
                final_response = initial_response

        except Exception as e:
            print(f"Response generation error: {str(e)}")
            # New fallback: use semantic fallback answer instead of generic message
            final_response = semantic_fallback_answer(q.query)

        session.add_message("assistant", final_response)

        return {
            "answer": final_response,
            "session_id": session_id,
            "sources_used": sources_used
        }

    except Exception as e:
        print(f"Error in ask endpoint: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
            "session_id": session_id if 'session_id' in locals() else str(uuid4()),
            "sources_used": ["error"]
        }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
