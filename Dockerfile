
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing bytecode
ENV PYTHONDONTWRITEBYTECODE=1

# Display Python logs immediately
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Copy project configuration
COPY requirements.txt .

# Create document storage directory
RUN mkdir -p /app/data/raw

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
