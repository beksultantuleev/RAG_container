import requests
import json
from deepdiff import DeepDiff  # Optional for better diff output

OLLAMA_URL = "http://localhost:11434/api/tags"  # Update host/port if needed
RAG_URL = "http://localhost:8000/api/tags"      # Update host/port if needed

def get_json(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def normalize_models(models):
    """Normalize model list for comparison: sort and prune non-essential fields if needed."""
    # Sort by model 'name' for consistent ordering
    return sorted(models, key=lambda m: m.get("name", "").lower())

def print_diff(dict1, dict2):
    try:
        from deepdiff import DeepDiff
        diff = DeepDiff(dict1, dict2, ignore_order=True)
        if diff:
            print("Differences found:")
            print(diff.pretty())
        else:
            print("No differences found. Both model lists match!")
    except ImportError:
        # If deepdiff not installed, do a naive print
        if dict1 != dict2:
            print("Differences found between model lists!")
            print("Ollama models:")
            print(json.dumps(dict1, indent=2))
            print("\nRAG app models:")
            print(json.dumps(dict2, indent=2))
        else:
            print("No differences found. Both model lists match!")

def main():
    print(f"Fetching Ollama models from {OLLAMA_URL}")
    ollama_data = get_json(OLLAMA_URL)
    print(f"Fetching RAG app models from {RAG_URL}")
    rag_data = get_json(RAG_URL)

    ollama_models = normalize_models(ollama_data.get("models", []))
    rag_models = normalize_models(rag_data.get("models", []))

    print(f"Ollama returned {len(ollama_models)} models")
    print(f"RAG app returned {len(rag_models)} models")

    print_diff(ollama_models, rag_models)

if __name__ == "__main__":
    main()