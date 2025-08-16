import requests
import json

API_URL = "http://localhost:8000/api/chat"  # Replace with your API base URL if different

def main():
    payload = {
        # "model": "gpt-4.1-mini",  # Use your configured model name here
        "model": "gemma3",  # Use your configured model name here
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": False  # Non-streaming, get full response at once
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        print("Response from RAG API:")
        print(response.json())
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()