# Oculus Backend Production Dockerfile
# Non-root user, HEALTHCHECK, start.sh entrypoint

FROM python:3.11-slim

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Install system dependencies (curl needed for health checks)
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies as non-root user
COPY --chown=appuser:appuser requirements.txt .
USER appuser
RUN pip install --no-cache-dir --user -r requirements.txt

# Add user site-packages to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"
ENV PYTHONPATH="/home/appuser/.local/lib/python3.11/site-packages:${PYTHONPATH}"

# Switch back to root to copy files
USER root

# Copy application code
COPY --chown=appuser:appuser . .

# Make startup script executable
RUN chmod +x start.sh

# Switch to non-root user for runtime
USER appuser

# Expose port 8080
EXPOSE 8080

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health').read()" || exit 1

# Run production entrypoint script
CMD ["./start.sh"]

