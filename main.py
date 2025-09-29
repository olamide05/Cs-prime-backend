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

# --- Load Environment Variables ---
# Ensure you have a 'keys.env' file in the same directory with your API keys.
# load_dotenv("keys.env")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SEARCH_API_KEY = os.environ("SEARCH_API_KEY") #  for search functionality
SEARCH_ENGINE_ID = os.environ("SEARCH_ENGINE_ID") #  for search functionality

# --- Configuration ---
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_API_KEY}"
TIMEOUT = 30  # seconds for Gemini API calls
SEARCH_TIMEOUT = 10 # seconds for Google Custom Search API calls
SEARCH_RESULTS_COUNT = 3 # Number of search results to fetch

# --- FastAPI App Setup ---
app = FastAPI(
    title="Maynooth University CS Chatbot",
    description="An AI assistant focused on the Computer Science department at Maynooth University, powered by Gemini 1.5 Pro with search capabilities.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production for specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chat Session Management (In-memory, consider persistent storage for production) ---
class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict[str, str]] = [] # Stores role and content

    def add_message(self, role: str, content: str):
        """Adds a message to the session history, keeping only the last 5 messages."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Keep last 5 messages to manage context window and memory.
        # This is a heuristic; a token-based approach would be more precise.
        self.messages = self.messages[-10:]
        logger.debug(f"Session {self.session_id}: Added message from {role}. Current history length: {len(self.messages)}")

    def get_history(self) -> str:
        """Formats the conversation history for the Gemini prompt."""
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)

sessions: Dict[str, ChatSession] = {} # In-memory storage for active sessions

# --- Gemini API Interaction ---
async def call_gemini(prompt: str) -> str:
    """Makes an asynchronous call to the Gemini API."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            response = await client.post(
                API_URL,
                json={
                    "contents": [{
                        "parts": [{"text": prompt}],
                        "role": "user" # Assuming all parts in 'contents' are from user for the prompt
                    }],
                    "generationConfig": {
                        "temperature": 0.7, # Controls randomness: 0.0 (deterministic) to 1.0 (creative)
                        "maxOutputTokens": 2000 # Max tokens in the generated response
                    }
                }
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()
            # Check for potential 'candidates' or 'parts' missing in the response
            if "candidates" in data and data["candidates"] and \
               "content" in data["candidates"][0] and \
               "parts" in data["candidates"][0]["content"] and \
               data["candidates"][0]["content"]["parts"]:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                logger.error(f"Gemini API response missing expected structure: {data}")
                raise ValueError("Invalid Gemini API response format")
        except httpx.RequestError as e:
            logger.error(f"Gemini API network error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Could not connect to Gemini API.")
        except httpx.HTTPStatusError as e:
            logger.error(f"Gemini API HTTP error (status {e.response.status_code}): {e.response.text}", exc_info=True)
            raise HTTPException(status_code=e.response.status_code, detail=f"Gemini API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Unexpected error during Gemini API call: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="An unexpected error occurred with the Gemini API.")

# --- Google Custom Search Integration ---
async def search_google(query: str) -> List[Dict[str, str]]:
    """Performs an asynchronous Google Custom Search."""
    if not SEARCH_API_KEY or not SEARCH_ENGINE_ID:
        logger.warning("SEARCH_API_KEY or SEARCH_ENGINE_ID not set. Skipping search and returning empty results.")
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
            search_results = data.get("items", [])
            return [{"title": item.get("title"), "link": item.get("link"), "snippet": item.get("snippet")} for item in search_results]
        except httpx.RequestError as e:
            logger.error(f"Google Search API network error for query '{query}': {e}", exc_info=True)
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"Google Search API HTTP error (status {e.response.status_code}) for query '{query}': {e.response.text}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error during Google Search API call for query '{query}': {e}", exc_info=True)
            return []

# --- Pydantic Model for Request Body ---
class Question(BaseModel):
    query: str

# --- API Endpoints ---
@app.post("/ask")
async def ask_question(q: Question, request: Request):
    """
    Receives a user question, manages session history, and provides an AI-generated answer.
    Integrates Google Custom Search if the AI determines a search is necessary.
    """
    session_id = request.headers.get("X-Session-ID") or str(uuid4())

    if session_id not in sessions:
        sessions[session_id] = ChatSession(session_id)
        logger.info(f"New session created: {session_id}")
    session = sessions[session_id]

    session.add_message("user", q.query)
    logger.info(f"Session {session_id}: User query: '{q.query}'")

    try:
        # Step 1: Initial Gemini call to decide if a search is needed
        # We explicitly instruct Gemini to use a specific tag if a search is required.
        initial_prompt = f"""You are a highly knowledgeable and friendly AI assistant for the Maynooth University Computer Science department.
        Your primary goal is to provide accurate and helpful information *strictly* related to the Computer Science department.
        
        If a user's question can be answered from common knowledge about the department or general CS topics, answer directly.
        
        If you suspect the question requires external, very specific, or up-to-date information that you might not possess (e.g., "What are the latest research papers from Dr. X?", "What's the current course schedule for YYYY-ZZZZ?"),
        you MUST indicate a need for a web search. To do this, output: `TOOL_CODE:SEARCH: [a concise, effective search query related to the user's question]`
        Do NOT include any other text when requesting a search.
        
        If the question is outside the scope of the Computer Science department, politely state that you can only assist with CS department-related queries at Maynooth University.
        Do NOT invent information. If you don't know, say you don't know within your scope.

        Conversation history (most recent last):
        {session.get_history()}

        User: {q.query}
        Assistant:"""
        
        gemini_initial_response = await call_gemini(initial_prompt)
        final_response_text = ""

        if gemini_initial_response.strip().startswith("TOOL_CODE:SEARCH:"):
            search_query = gemini_initial_response.replace("TOOL_CODE:SEARCH:", "").strip()
            logger.info(f"Session {session_id}: Gemini requested search with query: '{search_query}'")
            search_results = await search_google(search_query)
            
            if search_results:
                search_context = "\n\n".join([
                    f"Title: {r['title']}\nLink: {r['link']}\nSnippet: {r['snippet']}"
                    for r in search_results
                ])
                logger.debug(f"Session {session_id}: Search results obtained:\n{search_context}")

                # Step 2: Call Gemini again with search results for a refined answer
                follow_up_prompt = f"""You are a highly knowledgeable and friendly AI assistant for the Maynooth University Computer Science department.
                Based on the following search results, answer the user's question. Prioritize information from these results.
                If the search results do not contain the answer, state that you couldn't find the specific information.
                Maintain a helpful tone.

                Search Results (if available):
                {search_context}

                Conversation history (most recent last):
                {session.get_history()}

                User's original question: {q.query}
                Assistant:"""
                
                final_response_text = await call_gemini(follow_up_prompt)
            else:
                final_response_text = "I couldn't find any relevant information through search for your query. Please try rephrasing or asking a different question."
                logger.warning(f"Session {session_id}: No search results found for query: '{search_query}'")
        else:
            # If Gemini didn't request a search, its initial response is the final answer.
            final_response_text = gemini_initial_response

        session.add_message("assistant", final_response_text)
        logger.info(f"Session {session_id}: Assistant response generated.")

        return {
            "answer": final_response_text,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        # Re-raise HTTPExceptions directly to be handled by FastAPI
        raise
    except Exception as e:
        logger.exception(f"Session {session_id}: An unhandled error occurred during /ask request.")
        raise HTTPException(
            status_code=500,
            detail="I'm currently experiencing technical difficulties. Please try again later."
        )

@app.get("/status")
async def get_status():
    """
    Checks the status of the Gemini API connection.
    """
    try:
        test_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(test_url)
        
        status_text = "operational" if response.status_code == 200 else "degraded"
        models_list = [m["name"] for m in response.json().get("models", [])]
        logger.info(f"Status check: Gemini API {status_text}. Models: {', '.join(models_list)}")
        return {
            "status": status_text,
            "models": models_list,
            "timestamp": datetime.now().isoformat()
        }
    except httpx.RequestError as e:
        logger.error(f"Status check: Could not connect to Gemini API: {e}", exc_info=True)
        return {
            "status": "offline",
            "detail": f"Network error: {e}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Status check: Unexpected error: {e}", exc_info=True)
        return {
            "status": "error",
            "detail": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint, useful for load balancers/orchestrators.
    """
    logger.debug("Health check triggered")
    return {"status": "healthy"}

# --- Main entry point for Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    # To run this, save as e.g., main.py and run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
    logger.info("Starting FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
