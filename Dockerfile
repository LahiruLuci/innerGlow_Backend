
FROM python:3.11-slim

WORKDIR /app

# Keep Hugging Face model downloads in a shared location
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code and knowledge files
COPY main.py chatbot.py ./
COPY emotion/ ./emotion/
COPY rag/ ./rag/
COPY knowledge/ ./knowledge/

# Build the ChromaDB knowledge index
RUN python -m rag.build_index

# Allow the Space runtime user to access application files
RUN chmod -R a+rX /app

# Hugging Face Spaces application port
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
