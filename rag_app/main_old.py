# rag_app/main.py

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Union

import ollama
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models
from sentence_transformers import SentenceTransformer

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from.env file
load_dotenv()

# --- Pydantic Models for Ollama API Compatibility ---
# These models ensure that our API endpoint speaks the same language as Ollama,
# allowing seamless integration with clients like OpenWebUI.
# Based on the Ollama /api/chat documentation [4]

class Message(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = True
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None   # fixed as suggested
    keep_alive: Optional[Union[str, int]] = None

# --- Global Clients & Models ---
# These are initialized once at startup to avoid reloading on every request.

class AppState:
    def __init__(self):
        self.embedding_model = None
        self.qdrant_client = None
        self.llm_client = None
        self.llm_provider = None
        self.ollama_model = None
        self.openai_model = None
        self.qdrant_collection_name = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    logging.info("Application startup...")

    # Initialize Embedding Model
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    logging.info(f"Loading embedding model: {embedding_model_name}")
    app_state.embedding_model = SentenceTransformer(embedding_model_name)
    logging.info("Embedding model loaded successfully.")

    # Initialize Qdrant Client
    qdrant_host = os.getenv("QDRANT_HOST")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    app_state.qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    
    if not all([qdrant_host, app_state.qdrant_collection_name]):
        raise ValueError("QDRANT_HOST and QDRANT_COLLECTION_NAME must be set.")

    logging.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    app_state.qdrant_client = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)

    # Ensure Qdrant collection exists and has the correct vector size
    vector_size = app_state.embedding_model.get_sentence_embedding_dimension()
    try:
        collection_info = await app_state.qdrant_client.get_collection(collection_name=app_state.qdrant_collection_name)
        if collection_info.vectors_config.params.size!= vector_size:
            logging.error(f"Qdrant collection '{app_state.qdrant_collection_name}' has vector size "
                          f"{collection_info.vectors_config.params.size}, but embedding model "
                          f"'{embedding_model_name}' has size {vector_size}. Please recreate the collection.")
            # In a real application, you might want to handle this more gracefully
            raise RuntimeError("Mismatched vector sizes in Qdrant collection and embedding model.")
    except Exception as e:
        if "404" in str(e): # A simple way to check for Not Found error
            logging.warning(f"Qdrant collection '{app_state.qdrant_collection_name}' not found. Creating it...")
            await app_state.qdrant_client.create_collection(
                collection_name=app_state.qdrant_collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            logging.info(f"Collection '{app_state.qdrant_collection_name}' created successfully.")
        else:
            logging.error(f"Failed to connect to or configure Qdrant: {e}")
            raise

    # Initialize LLM Client based on provider
    app_state.llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    logging.info(f"Using LLM provider: {app_state.llm_provider}")

    if app_state.llm_provider == "ollama":
        ollama_host = os.getenv("OLLAMA_HOST")
        ollama_port = int(os.getenv("OLLAMA_PORT", 11434))
        app_state.ollama_model = os.getenv("OLLAMA_MODEL")
        if not all([ollama_host, app_state.ollama_model]):
            raise ValueError("OLLAMA_HOST and OLLAMA_MODEL must be set for the 'ollama' provider.")
        
        ollama_url = f"http://{ollama_host}:{ollama_port}"
        logging.info(f"Initializing Ollama client for host: {ollama_url}")
        app_state.llm_client = ollama.AsyncClient(host=ollama_url)
    
    elif app_state.llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        app_state.openai_model = os.getenv("OPENAI_MODEL")
        if not all([api_key, app_state.openai_model]):
            raise ValueError("OPENAI_API_KEY and OPENAI_MODEL must be set for the 'openai' provider.")
        
        logging.info("Initializing OpenAI client.")
        app_state.llm_client = AsyncOpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {app_state.llm_provider}. Must be 'ollama' or 'openai'.")

    logging.info("Application startup complete.")
    yield
    # --- Shutdown Logic ---
    logging.info("Application shutdown...")
    # Clean up resources if needed
    if app_state.qdrant_client:
        await app_state.qdrant_client.close()
    logging.info("Application shutdown complete.")


app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"status": "RAG API is running"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    This endpoint performs RAG and then streams a response from an LLM.
    It is designed to be a drop-in replacement for Ollama's /api/chat endpoint.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    # 1. Get the last user query
    user_query = request.messages[-1].content

    # 2. Generate embedding for the user query
    query_vector = app_state.embedding_model.encode(user_query).tolist()

    # 3. Search Qdrant for relevant context
    try:
        search_results = await app_state.qdrant_client.search(
            collection_name=app_state.qdrant_collection_name,
            query_vector=query_vector,
            limit=3,  # Retrieve top 3 most relevant documents
            with_payload=True
        )
        context = "\n\n---\n\n".join([result.payload['text'] for result in search_results])
    except Exception as e:
        logging.error(f"Error searching Qdrant: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve context from vector database.")

    # 4. Augment the prompt
    augmented_prompt = f"""
Based on the following context, please answer the user's query.
If the context does not contain the answer, state that you don't have enough information.

Context:
{context}

User Query:
{user_query}
"""
    
    # Replace the last user message with the augmented one
    rag_messages = request.messages[:-1] + [Message(role="user", content=augmented_prompt)]

    # 5. Stream response from the selected LLM provider
    if request.stream:
        return StreamingResponse(
            stream_llm_response(rag_messages, request.model),
            media_type="application/x-ndjson"
        )
    else:
        # Handle non-streaming case (though streaming is the primary use case)
        raise HTTPException(status_code=400, detail="Non-streaming responses are not fully implemented in this RAG proxy.")


async def stream_llm_response(messages: List[Message], original_model: str):
    """
    A generator function that streams responses from either Ollama or OpenAI
    and formats them into the Ollama-compatible JSON structure.
    """
    if app_state.llm_provider == "ollama":
        # Stream from Ollama
        stream = await app_state.llm_client.chat(
            model=app_state.ollama_model,
            messages=[msg.dict() for msg in messages],
            stream=True
        )
        async for chunk in stream:
            # Reformat the chunk to match the expected API output
            # This ensures compatibility with clients expecting the Ollama format.
            reformatted_chunk = {
                "model": original_model, # Report the model the user requested
                "created_at": chunk.get("created_at", ""),
                "message": {
                    "role": "assistant",
                    "content": chunk.get("message", {}).get("content", "")
                },
                "done": chunk.get("done", False)
            }
            if reformatted_chunk["done"]:
                 # Add final stats if available
                reformatted_chunk["total_duration"] = chunk.get("total_duration")
                reformatted_chunk["prompt_eval_count"] = chunk.get("prompt_eval_count")
                reformatted_chunk["eval_count"] = chunk.get("eval_count")

            yield json.dumps(reformatted_chunk) + "\n"

    elif app_state.llm_provider == "openai":
        # Stream from OpenAI
        stream = await app_state.llm_client.chat.completions.create(
            model=app_state.openai_model,
            messages=[msg.dict() for msg in messages],
            stream=True
        )
        async for chunk in stream:
            content = chunk.choices.delta.content or ""
            if content:
                # Create an Ollama-like streaming chunk
                response_chunk = {
                    "model": original_model,
                    "created_at": "N/A", # OpenAI doesn't provide this per chunk
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "done": False
                }
                yield json.dumps(response_chunk) + "\n"
        
        # Send the final "done" message
        final_chunk = {
            "model": original_model,
            "created_at": "N/A",
            "message": {"role": "assistant", "content": ""},
            "done": True
        }
        yield json.dumps(final_chunk) + "\n"