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

# Grant CREATE permission on public schema (PostgreSQL 15+ revoked this by default)
echo "Ensuring database schema permissions..."
psql "${DATABASE_URL}" -c "GRANT ALL ON SCHEMA public TO CURRENT_USER;" 2>&1 || echo "⚠ Schema grant skipped (may already have permissions)"

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
echo "Workers: ${UVICORN_WORKERS:-12}"
echo "========================================="

exec uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info --workers ${UVICORN_WORKERS:-12} --timeout-keep-alive 120

