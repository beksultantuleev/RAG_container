from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Configuration
QDRANT_HOST = "localhost"      # Replace with your Qdrant host
QDRANT_PORT = 6333             # Replace with your Qdrant port
COLLECTION_NAME = "my-rag-collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    # Initialize Qdrant client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Initialize embedding model
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Loaded embedding model")

    # Sample documents to index
    documents = [
        "microdistrict 5 has best PC setup in bishkek.",
        "Kara oi has greatest beach to have fun.",
        "6700xt gpu is great price to value .",
    ]

    # Embed documents
    embeddings = embedder.encode(documents)

    # Prepare points for Qdrant upsert
    points = []
    for idx, (doc, emb) in enumerate(zip(documents, embeddings)):
        points.append(
            models.PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload={"text": doc}
            )
        )

    # Upsert points into collection
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"Upserted {len(points)} points into collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()