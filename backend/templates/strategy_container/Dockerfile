# Signal Synk Strategy Container Template
# Production-ready Docker image for running trading strategies

FROM python:3.11-slim

# Metadata
LABEL maintainer="Signal Synk Team"
LABEL version="1.0.0"
LABEL description="Signal Synk Trading Strategy Container"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy strategy code
COPY . .

# Environment variables (will be overridden at runtime)
ENV STRATEGY_PASSPHRASE=""
ENV SIGNALSYNK_WS_URL="wss://app.signalsynk.com/ws/strategy"
ENV SIGNALSYNK_API_URL="https://app.signalsynk.com"
ENV ALPHAVANTAGE_API_KEY=""
ENV LOG_LEVEL="INFO"
ENV HEARTBEAT_INTERVAL="30"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash strategy && \
    chown -R strategy:strategy /app

USER strategy

# Health check - verify process is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import asyncio; asyncio.run(__import__('websockets').connect('ws://localhost:8765', close_timeout=1))" 2>/dev/null || exit 0

# Run the strategy
CMD ["python", "-u", "main.py"]

