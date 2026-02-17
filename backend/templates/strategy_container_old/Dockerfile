FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy strategy files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WEBHOOK_URL=""
ENV LICENSE_ID=""
ENV OCULUS_API_URL="https://api.oculusalgorithms.com"

# Run entrypoint
CMD ["python", "entrypoint.py"]

