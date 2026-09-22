#!/usr/bin/env bash
# exit on error
set -o errexit

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo "==> Collecting static files..."
python manage.py collectstatic --no-input

echo "==> Checking DATABASE_URL..."
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL is not set!"
    exit 1
fi
echo "DATABASE_URL is set"

echo "==> Testing database connection..."
python manage.py check --database default

echo "==> Running database migrations..."
python manage.py migrate --noinput

echo "==> Checking if jobs table was created..."
python manage.py shell -c "from django.db import connection; cursor = connection.cursor(); cursor.execute('SELECT COUNT(*) FROM information_schema.tables WHERE table_name = %s', ['jobs']); print(f'Jobs table exists: {cursor.fetchone()[0] > 0}')"

# Jobs are loaded manually from a local scrape:
#   DATABASE_URL=<external db url> python manage.py load_jobs

echo "==> Build completed successfully!"

# Preload ML model to avoid timeout on first request (optional)
# python manage.py shell -c "from jobs.services.embeddings import EmbeddingService; EmbeddingService()._load_model()" || echo "Model preload skipped"
