FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade packaging tools and install requirements
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download spaCy English model into the image
RUN python -m spacy download en_core_web_sm

COPY . .

RUN mkdir -p /app/models

EXPOSE 10000

# Note: Using Flask WSGI application format
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "ChatBot:app"]