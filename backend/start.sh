#!/bin/bash
set -e

echo "========================================="
echo "Oculus Production Starting..."
echo "========================================="

# Verify DATABASE_URL is set (required for production)
if [ -z "$DATABASE_URL" ]; then
    echo "❌ ERROR: DATABASE_URL is not set!"
    echo "This application requires a PostgreSQL database connection."
    exit 1
fi
echo "✓ DATABASE_URL is configured"

# Run Alembic database migrations
echo "Running database migrations..."
cd /app
alembic upgrade head
echo "✓ Migrations complete!"

# Seed default admin user (idempotent - safe to run every start)
echo "Seeding admin user..."
python -m scripts.seed_admin
echo "✓ Admin seed complete!"

# Start the application
PORT=${PORT:-8080}
echo "========================================="
echo "Starting uvicorn on 0.0.0.0:$PORT..."
echo "Health check: /health"
echo "Workers: 4"
echo "========================================="

exec uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info --workers 4 --timeout-keep-alive 120

