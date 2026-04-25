FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for torch and transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Start the server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
