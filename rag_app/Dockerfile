# rag_app/Dockerfile

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy only the requirements file first to leverage Docker's layer caching.
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . .

# Command to run the application using Uvicorn ASGI server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]