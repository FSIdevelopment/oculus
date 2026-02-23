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

# Generate missing READMEs for existing builds
echo "Generating missing READMEs for existing builds..."
python scripts/generate_missing_readmes.py || echo "README generation skipped (may have failed or no builds need READMEs)"
echo "README generation complete!"

# Start the application
echo "Starting Oculus API..."
exec "$@"

