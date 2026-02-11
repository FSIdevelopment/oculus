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

# Start the application
echo "Starting Oculus API..."
exec "$@"

