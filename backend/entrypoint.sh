#!/bin/bash
set -e

# Wait for database to be ready
echo "Waiting for database..."
while ! pg_isready -h postgres -U ${POSTGRES_USER:-oculus_user} -q; do
  sleep 1
done
echo "Database is ready!"

# Run Alembic migrations
echo "Running database migrations..."
cd /app
alembic upgrade head
echo "Migrations complete!"

# Seed default admin user
echo "Seeding admin user..."
python -m scripts.seed_admin
echo "Admin seed complete!"

# Note: README generation now runs in background after app startup (see main.py lifespan)

# Start the application
echo "Starting Oculus API..."
exec "$@"

