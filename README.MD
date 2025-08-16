Project Configuration
This document outlines the necessary environment variables for configuring and running the project. These variables are stored in a .env file in the root directory.

Creating the .env File
The .env file is used to manage sensitive information and configuration settings. You must create this file and populate it with the required parameters before running the application.

Below are the configuration options, grouped by function.

1. Qdrant Configuration
These parameters define the connection to your Qdrant vector database instance.

QDRANT_HOST: The hostname of your Qdrant service. If running with Docker Compose, this should match the service name defined in your docker-compose.yml (e.g., qdrant).

QDRANT_PORT: The port for Qdrant's HTTP API.

QDRANT_COLLECTION_NAME: The name of the collection to be created in Qdrant for storing your vectors.

2. Embedding Model
This parameter specifies the model used to convert text into vector embeddings.

EMBEDDING_MODEL_NAME: The name of the sentence-transformers model. The default, all-MiniLM-L6-v2, is a good starting point for a fast and efficient model.

3. Ollama Configuration
These settings are used to connect to your local or remote Ollama instance.

OLLAMA_HOST: The IP address or hostname of your Ollama service. If Ollama is running on your host machine and you are using Docker, you may need to use host.docker.internal.

OLLAMA_PORT: The port of the Ollama service.

OLLAMA_MODEL: The name of the model you have downloaded and want to use (e.g., llama3, mistral, gemma3).

4. OpenAI Configuration
These settings are used to connect to the OpenAI API.

OPENAI_API_KEY: Your secret API key from OpenAI. Keep this key secure and do not commit it to your version control system.

OPENAI_MODEL: The name of the OpenAI model to use for generation (e.g., gpt-4o, gpt-4o-mini, gpt-4.1-mini).

Example .env File
Below is a single example .env file that includes all configuration options. Simply uncomment the section for your preferred LLM provider (Ollama or OpenAI) before running.

# .env

# -- Qdrant Configuration --
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=my-rag-collection

# -- Embedding Model Configuration --
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# Uncomment the section below for your chosen LLM provider

# -- Ollama Configuration --
# OLLAMA_HOST=host.docker.internal
# OLLAMA_PORT=11434
# OLLAMA_MODEL=gemma3

# -- OpenAI Configuration --
# OPENAI_API_KEY=sk-xxxxxx
# OPENAI_MODEL=gpt-4.1-mini
