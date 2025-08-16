# rag_app/main.py

from typing import Optional, Dict, Union, List
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import os
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone

import ollama
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
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
        self.llm_client = None
        self.llm_provider = None
        self.ollama_model = None
        self.openai_model = None
        self.qdrant_collection_name = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application startup...")

    # Initialize Embedding Model
    embedding_model_name = os.getenv(
        "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
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
    app_state.qdrant_client = AsyncQdrantClient(
        host=qdrant_host, port=qdrant_port)

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
            raise RuntimeError(
                "Mismatched vector sizes in Qdrant collection and embedding model.")

    except Exception as e:
        if "Not found" in str(e) or "404" in str(e):
            logging.warning(
                f"Qdrant collection '{app_state.qdrant_collection_name}' not found. Creating it...")
            await app_state.qdrant_client.create_collection(
                collection_name=app_state.qdrant_collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE),
            )
            logging.info(
                f"Collection '{app_state.qdrant_collection_name}' created successfully.")
        else:
            logging.error(f"Failed to connect to or configure Qdrant: {e}")
            raise

    # Initialize LLM Client
    app_state.llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    logging.info(f"Using LLM provider: {app_state.llm_provider}")

    if app_state.llm_provider == "ollama":
        ollama_host = os.getenv("OLLAMA_HOST")
        ollama_port = int(os.getenv("OLLAMA_PORT", 11434))
        app_state.ollama_model = os.getenv("OLLAMA_MODEL")

        if not all([ollama_host, app_state.ollama_model]):
            raise ValueError(
                "OLLAMA_HOST and OLLAMA_MODEL must be set for the 'ollama' provider.")

        ollama_url = f"http://{ollama_host}:{ollama_port}"
        logging.info(f"Initializing Ollama client for host: {ollama_url}")
        app_state.llm_client = ollama.AsyncClient(host=ollama_url)

    elif app_state.llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        app_state.openai_model = os.getenv("OPENAI_MODEL")

        if not all([api_key, app_state.openai_model]):
            raise ValueError(
                "OPENAI_API_KEY and OPENAI_MODEL must be set for the 'openai' provider.")

        logging.info("Initializing OpenAI client.")
        app_state.llm_client = AsyncOpenAI(api_key=api_key)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {app_state.llm_provider}. Must be 'ollama' or 'openai'.")

    logging.info("Application startup complete.")
    yield
    logging.info("Application shutdown...")

    if app_state.qdrant_client:
        await app_state.qdrant_client.close()

    logging.info("Application shutdown complete.")


app = FastAPI(lifespan=lifespan)


# --- Helper to build model entries for /api/tags ---


def build_model_entry(model_name: str, family: str = "rag") -> Dict[str, Any]:
    return {
        "name": model_name,
        "model": model_name,
        "modified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "size": 0,
        "digest": "rag-app-digest",
        "details": {
            "family": family,
            "format": "gguf",
            "parameter_size": "N/A",
            "quantization_level": "N/A",
        },
    }


# --- API Endpoints ---


@app.get("/")
async def root():
    return {"status": "RAG API is running"}


@app.get("/api/version")
async def get_version():
    """Returns a mock version for Ollama compatibility."""
    return JSONResponse(content={"version": "0.1.32"})

def build_model_entry(model_name: str, version: str = "latest", family: str = "ollama") -> dict:
    full_name = f"{model_name}:{version}"
    return {
        "name": full_name,
        "model": full_name,
        "modified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "size": 3338801804,  # Example realistic size, use actual if available
        "digest": "a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a",  # Actual SHA256 or similar
        "details": {
            "family": family,
            "families": [family],
            "parent_model": "",  # Provide if applicable, else empty string
            "parameter_size": "4.3B",              # Your model's actual parameter size
            "quantization_level": "Q4_K_M"        # Your model quantization level
        }
    }

@app.get("/api/tags")
async def get_tags():
    models = []
    if app_state.llm_provider == "ollama" and app_state.ollama_model:
        models.append(build_model_entry(
            model_name=app_state.ollama_model,
            version="latest",
            family="ollama"
        ))
    elif app_state.llm_provider == "openai" and app_state.openai_model:
        models.append(build_model_entry(
            model_name=app_state.openai_model,
            version="latest",
            family="openai"
        ))
    else:
        models.append(build_model_entry(
            model_name="rag-model",
            version="latest",
            family="rag"
        ))
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

    # Normalize incoming model name by stripping version suffix
    model_name_raw = chat_request.model
    model_name = model_name_raw.split(":")[0]  # e.g. "gemma3:latest" -> "gemma3"

    # Build list of valid base model names (without versions)
    configured_models = []
    if app_state.llm_provider == "ollama" and app_state.ollama_model:
        configured_models.append(app_state.ollama_model.split(":")[0])
    if app_state.llm_provider == "openai" and app_state.openai_model:
        configured_models.append(app_state.openai_model.split(":")[0])

    if model_name not in configured_models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name_raw}' not found")

    # Map effective model to full configured model name (with version suffix)
    if app_state.llm_provider == "ollama":
        effective_model = app_state.ollama_model
    else:
        effective_model = app_state.openai_model

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
            request_model=effective_model,
            request_options=chat_request.options,
            request_format=chat_request.format,
            request_keep_alive=chat_request.keep_alive,
        ):
            chunk_json = json.loads(chunk)
            content = chunk_json.get("message", {}).get("content", "")
            content_parts.append(content)
            if chunk_json.get("done", False):
                # Collect metadata from the last chunk (except the message content itself)
                final_metadata = {k: v for k, v in chunk_json.items() if k != "message"}

        full_content = "".join(content_parts)
        response = {
            "model": effective_model,
            "message": {"role": "assistant", "content": full_content},
            "done": True,
            **final_metadata,
        }
        return JSONResponse(content=response)

# Provide alias endpoint for OpenWebUI / Ollama compatibility


@app.post("/ollama/api/chat")
async def ollama_api_chat(request: Request):
    return await chat(request)


# async def stream_llm_response(
#     messages: List[Message],
#     request_model: str,
#     request_options: Optional[Dict] = None,
#     request_format: Optional[str] = None,
#     request_keep_alive: Optional[Union[str, int]] = None,
# ):
#     if app_state.llm_provider == "ollama":
#         try:
#             ollama_args = {
#                 "model": app_state.ollama_model,
#                 "messages": [msg.dict() for msg in messages],
#                 "stream": True,
#             }
#             if request_options:
#                 ollama_args["options"] = request_options
#             if request_format:
#                 ollama_args["format"] = request_format
#             if request_keep_alive:
#                 ollama_args["keep_alive"] = request_keep_alive

#             stream = await app_state.llm_client.chat(**ollama_args)

#             async for chunk in stream:
#                 # Serialize chunk after converting to dict
#                 yield json.dumps(chunk.dict()) + "\n"

#         except Exception as e:
#             logging.error(f"Ollama streaming failed: {e}", exc_info=True)
#             raise


async def stream_llm_response(
    messages: List[Message],
    request_model: str,
    request_options: Optional[Dict] = None,
    request_format: Optional[str] = None,
    request_keep_alive: Optional[Union[str, int]] = None,
):
    if app_state.llm_provider == "ollama":
        try:
            ollama_args = {
                "model": app_state.ollama_model,
                "messages": [msg.dict() for msg in messages],
                "stream": True,
            }
            if request_options:
                ollama_args["options"] = request_options
            if request_format:
                ollama_args["format"] = request_format
            if request_keep_alive:
                ollama_args["keep_alive"] = request_keep_alive

            stream = await app_state.llm_client.chat(**ollama_args)

            async for chunk in stream:
                # Convert Pydantic object to dict if needed
                chunk_dict = chunk.dict() if hasattr(chunk, "dict") else chunk

                # Build response chunk including Ollama metadata if present
                response_chunk = {
                    "model": request_model,
                    "created_at": chunk_dict.get("created_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
                    "message": chunk_dict.get("message", {"role": "assistant", "content": ""}),
                    "done": chunk_dict.get("done", False),
                    # Ollama includes this on done
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

    elif app_state.llm_provider == "openai":
        openai_args = {
            "model": app_state.openai_model,
            "messages": [msg.dict() for msg in messages],
            "stream": True,
        }
        if request_options:
            if "temperature" in request_options:
                openai_args["temperature"] = request_options["temperature"]
            if "top_p" in request_options:
                openai_args["top_p"] = request_options["top_p"]

        stream = await app_state.llm_client.chat.completions.create(**openai_args)

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
        raise RuntimeError(
            f"Unsupported LLM provider: {app_state.llm_provider}")
