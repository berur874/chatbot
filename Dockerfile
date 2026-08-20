FROM python:3.11-slim

WORKDIR /app

# Install system utilities & C/C++ compiler tools needed for native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade core packaging tools before installing requirements
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for models
RUN mkdir -p /app/models

EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "ChatBot:app"]