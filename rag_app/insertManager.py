from langchain.text_splitter import TokenTextSplitter
import cx_Oracle
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import uuid
import json
import os
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my-rag-collection")
# 1. DB retrieval
def fetch_documents_from_oracle(dsn, user, password, query):
    connection = cx_Oracle.connect(user=user, password=password, dsn=dsn)
    cursor = connection.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    documents = []
    for row in rows:
        doc_id, text_content, metadata_json = row
        metadata = json.loads(metadata_json) if metadata_json else {}
        documents.append({
            "id": doc_id,
            "text": text_content,
            "metadata": metadata,
        })
    cursor.close()
    connection.close()
    return documents

# 2. Website text fetcher
def fetch_website_text(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join(line for line in lines if line)
    return clean_text

def ingest_documents(documents: list[dict], urls: list[str], chunk_size=500, chunk_overlap=50):
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Ensure collection (same as before) omitted for brevity
    
    # Fetch urls
    website_texts = [fetch_website_text(url) for url in urls]
    
    # Combine all texts with metadata
    all_docs = documents + [{"id": f"url-{i}", "text": t, "metadata": {"source": "url", "url": urls[i]}} for i, t in enumerate(website_texts)]
    
    # Token-based splitter
    token_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="gpt2"
    )
    
    chunks = []
    payloads = []
    for doc in all_docs:
        splitted = token_splitter.split_text(doc["text"])
        for chunk_text in splitted:
            chunks.append(chunk_text)
            # Store metadata with chunk
            meta = {
                "source_id": doc["id"],
                **doc.get("metadata", {})
            }
            payloads.append(meta)
    
    print(f"Chunks to embed: {len(chunks)}")
    
    embeddings = embedder.encode(chunks)
    points = []
    for idx, (chunk, embedding, meta) in enumerate(zip(chunks, embeddings, payloads)):
        points.append(
            models.PointStruct(
                id=uuid.uuid4().int >> 64,
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    **meta,
                }
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} points.")

if __name__ == "__main__":
    # Fetch from Oracle DB example
    oracle_docs = fetch_documents_from_oracle(
        dsn="your_dsn",
        user="your_user",
        password="your_password",
        query="SELECT id, text_column, metadata_column FROM your_table"
    )
    
    # Example URLs
    urls = ["https://en.wikipedia.org/wiki/OpenAI"]
    
    ingest_documents(oracle_docs, urls)