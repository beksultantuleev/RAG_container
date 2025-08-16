# rag_app/main.py
from typing import Optional, Dict, Union, List, Any
from fastapi import HTTPException, Request, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
import os
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import ollama
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from sentence_transformers import SentenceTransformer

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Pydantic Models for Ollama API Compatibility ---
class Message(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = True
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Union[str, int]] = None

    class Config:
        extra = "allow"  # Accept extra unexpected fields

# --- Global Clients & Models ---
class AppState:
    def __init__(self):
        self.embedding_model = None
        self.qdrant_client = None
        # Each client and model can be None if not configured
        self.ollama_client: Optional[ollama.AsyncClient] = None
        self.ollama_model: Optional[str] = None
        self.openai_client: Optional[AsyncOpenAI] = None
        self.openai_model: Optional[str] = None
        self.qdrant_collection_name = None
        # Lookup: model_name (no version) -> ("ollama"|"openai", full_model_name)
        self.model_map: Dict[str, tuple[str, str]] = {}

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    
    vector_size = app_state.embedding_model.get_sentence_embedding_dimension()
    try:
        collection_info = await app_state.qdrant_client.get_collection(collection_name=app_state.qdrant_collection_name)
        vectors_config = collection_info.config.params.vectors
        if isinstance(vectors_config, models.VectorParams):
            collection_vector_size = vectors_config.size
        else:
            collection_vector_size = vectors_config.get("").size
        if collection_vector_size != vector_size:
            logging.error(
                f"Qdrant collection '{app_state.qdrant_collection_name}' has vector size {collection_vector_size}, "
                f"but embedding model '{embedding_model_name}' has size {vector_size}. "
                "Please recreate the collection."
            )
            raise RuntimeError("Mismatched vector sizes in Qdrant collection and embedding model.")
    except Exception as e:
        if "Not found" in str(e) or "404" in str(e):
            logging.warning(f"Qdrant collection '{app_state.qdrant_collection_name}' not found. Creating it...")
            await app_state.qdrant_client.create_collection(
                collection_name=app_state.qdrant_collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            logging.info(f"Collection '{app_state.qdrant_collection_name}' created successfully.")
        else:
            logging.error(f"Failed to connect to or configure Qdrant: {e}")
            raise

    # Initialize Ollama client if configured
    ollama_host = os.getenv("OLLAMA_HOST")
    ollama_port = int(os.getenv("OLLAMA_PORT", 11434))
    ollama_model = os.getenv("OLLAMA_MODEL")
    if ollama_host and ollama_model:
        ollama_url = f"http://{ollama_host}:{ollama_port}"
        logging.info(f"Initializing Ollama client for host: {ollama_url} with model: {ollama_model}")
        app_state.ollama_client = ollama.AsyncClient(host=ollama_url)
        app_state.ollama_model = ollama_model
        # register in model map
        base_name = ollama_model.split(":")[0]
        app_state.model_map[base_name] = ("ollama", ollama_model)
    else:
        logging.info("Ollama environment variables not fully set, skipping Ollama client initialization.")

    # Initialize OpenAI client if configured
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    if openai_api_key and openai_model:
        logging.info(f"Initializing OpenAI client with model: {openai_model}")
        app_state.openai_client = AsyncOpenAI(api_key=openai_api_key)
        app_state.openai_model = openai_model
        base_name = openai_model.split(":")[0]
        app_state.model_map[base_name] = ("openai", openai_model)
    else:
        logging.info("OpenAI environment variables not fully set, skipping OpenAI client initialization.")
    
    logging.info(f"Available models: {app_state.model_map}")
    logging.info("Application startup complete.")
    yield
    logging.info("Application shutdown...")
    if app_state.qdrant_client:
        await app_state.qdrant_client.close()
    logging.info("Application shutdown complete.")

app = FastAPI(lifespan=lifespan)

# --- Helper to build model entries for /api/tags ---
def build_model_entry(model_name: str, version: str = "dsml", family: str = "rag") -> dict:
    full_name = f"{model_name}:{version}" if ":" not in model_name else model_name
    details: Dict[str, Any] = {
        "family": family,
        "families": [family],
        "parent_model": "",
        "parameter_size": "N/A",
        "quantization_level": "N/A",
    }
    size = 0
    digest = "rag-app-digest"
    if family == "ollama":
        size = 3338801804  # example realistic size, replace if needed
        digest = "a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a"
        details.update({
            "parameter_size": "over 9000B",
            "quantization_level": "Q4_K_M"
        })
    elif family == "openai":
        size = 0  # OpenAI doesn’t publish size, keep 0 or add info if you want
        details.update({
            "parameter_size": "unknown",
            "quantization_level": "N/A"
        })
    return {
        "name": full_name,
        "model": full_name,
        "modified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "size": size,
        "digest": digest,
        "details": details,
    }

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"status": "RAG API is running"}

@app.get("/api/version")
async def get_version():
    """Returns a mock version for Ollama compatibility."""
    return JSONResponse(content={"version": "0.1.32"})

@app.get("/api/tags")
async def get_tags():
    models = []
    if app_state.ollama_client and app_state.ollama_model:
        models.append(build_model_entry(
            model_name=app_state.ollama_model,
            family="ollama"
        ))
    if app_state.openai_client and app_state.openai_model:
        models.append(build_model_entry(
            model_name=app_state.openai_model,
            family="openai"
        ))
    # Provide default/fallback "rag" model entry if none available:
    if not models:
        models.append(build_model_entry("rag-model", family="rag"))
    return JSONResponse(content={"models": models})

@app.post("/api/chat")
async def chat(request: Request):
    payload = await request.json()
    logging.info(f"Chat request payload: {payload}")
    try:
        chat_request = ChatRequest.parse_obj(payload)
    except Exception as e:
        logging.error(f"Failed to parse chat request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
    
    if not chat_request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")
    
    # Normalize incoming model name (without version)
    requested_model_raw = chat_request.model
    requested_model = requested_model_raw.split(":")[0]

    # Check if requested model is configured
    if requested_model not in app_state.model_map:
        raise HTTPException(status_code=400, detail=f"Model '{requested_model_raw}' not found")
    
    provider, effective_model = app_state.model_map[requested_model]

    user_query = chat_request.messages[-1].content
    query_vector = app_state.embedding_model.encode(user_query).tolist()
    try:
        search_results = await app_state.qdrant_client.search(
            collection_name=app_state.qdrant_collection_name,
            query_vector=query_vector,
            limit=3,
            with_payload=True,
        )
        context = "\n\n---\n\n".join([result.payload.get("text", "") for result in search_results])
    except Exception as e:
        logging.error(f"Error searching Qdrant: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve context from vector database.")
    
    augmented_prompt = f"""Based on the following context, please answer the user's query.
If the context does not contain the answer, state that you don't have enough information.
Context:
{context}
User Query:
{user_query}
"""
    rag_messages = chat_request.messages[:-1] + [Message(role="user", content=augmented_prompt)]

    if chat_request.stream:
        # Streaming response
        return StreamingResponse(
            stream_llm_response(
                messages=rag_messages,
                provider=provider,
                request_model=effective_model,
                request_options=chat_request.options,
                request_format=chat_request.format,
                request_keep_alive=chat_request.keep_alive,
            ),
            media_type="application/x-ndjson",
        )
    else:
        # Non-streaming: collect all chunks and metadata before returning final JSON response
        content_parts = []
        final_metadata = {}
        async for chunk in stream_llm_response(
            messages=rag_messages,
            provider=provider,
            request_model=effective_model,
            request_options=chat_request.options,
            request_format=chat_request.format,
            request_keep_alive=chat_request.keep_alive,
        ):
            chunk_json = json.loads(chunk)
            content = chunk_json.get("message", {}).get("content", "")
            content_parts.append(content)
            if chunk_json.get("done", False):
                # Collect metadata except the message content itself
                final_metadata = {k: v for k, v in chunk_json.items() if k != "message"}
        full_content = "".join(content_parts)
        response = {
            "model": effective_model,
            "message": {"role": "assistant", "content": full_content},
            "done": True,
            **final_metadata,
        }
        return JSONResponse(content=response)

@app.post("/ollama/api/chat")
async def ollama_api_chat(request: Request):
    # Alias endpoint for compatibility; delegate to main chat endpoint
    return await chat(request)


async def stream_llm_response(
    messages: List[Message],
    provider: str,
    request_model: str,
    request_options: Optional[Dict] = None,
    request_format: Optional[str] = None,
    request_keep_alive: Optional[Union[str, int]] = None,
):
    if provider == "ollama":
        if app_state.ollama_client is None or app_state.ollama_model is None:
            raise RuntimeError("Ollama client or model is not configured.")

        try:
            ollama_args = {
                "model": request_model,
                "messages": [msg.dict() for msg in messages],
                "stream": True,
            }
            if request_options:
                ollama_args["options"] = request_options
            if request_format:
                ollama_args["format"] = request_format
            if request_keep_alive:
                ollama_args["keep_alive"] = request_keep_alive
            stream = await app_state.ollama_client.chat(**ollama_args)
            async for chunk in stream:
                chunk_dict = chunk.dict() if hasattr(chunk, "dict") else chunk
                response_chunk = {
                    "model": request_model,
                    "created_at": chunk_dict.get("created_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
                    "message": chunk_dict.get("message", {"role": "assistant", "content": ""}),
                    "done": chunk_dict.get("done", False),
                    "done_reason": chunk_dict.get("done_reason"),
                    "total_duration": chunk_dict.get("total_duration"),
                    "load_duration": chunk_dict.get("load_duration"),
                    "prompt_eval_count": chunk_dict.get("prompt_eval_count"),
                    "prompt_eval_duration": chunk_dict.get("prompt_eval_duration"),
                    "eval_count": chunk_dict.get("eval_count"),
                    "eval_duration": chunk_dict.get("eval_duration"),
                }
                yield json.dumps(response_chunk) + "\n"
        except Exception as e:
            logging.error(f"Ollama streaming failed: {e}", exc_info=True)
            raise
    elif provider == "openai":
        if app_state.openai_client is None or app_state.openai_model is None:
            raise RuntimeError("OpenAI client or model is not configured.")

        openai_args = {
            "model": request_model,
            "messages": [msg.dict() for msg in messages],
            "stream": True,
        }
        if request_options:
            if "temperature" in request_options:
                openai_args["temperature"] = request_options["temperature"]
            if "top_p" in request_options:
                openai_args["top_p"] = request_options["top_p"]
        stream = await app_state.openai_client.chat.completions.create(**openai_args)
        async for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                response_chunk = {
                    "model": request_model,
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "message": {"role": "assistant", "content": content},
                    "done": False,
                }
                yield json.dumps(response_chunk) + "\n"
        # Send final chunk indicating done
        final_chunk = {
            "model": request_model,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "eval_count": 0,
            "eval_duration": 0,
        }
        yield json.dumps(final_chunk) + "\n"
    else:
        raise RuntimeError(f"Unsupported LLM provider '{provider}' for model '{request_model}'")