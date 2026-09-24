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

echo "==> Warming embedding model cache and precomputing skill/title embeddings..."
python manage.py warm_embedding_model

echo "==> Checking if jobs table was created..."
python manage.py shell -c "from django.db import connection; cursor = connection.cursor(); cursor.execute('SELECT COUNT(*) FROM information_schema.tables WHERE table_name = %s', ['jobs']); print(f'Jobs table exists: {cursor.fetchone()[0] > 0}')"

# Jobs are loaded manually from a local scrape:
#   DATABASE_URL=<external db url> python manage.py load_jobs

echo "==> Build completed successfully!"

# Note: this script runs in a separate process from the gunicorn worker that
# actually serves requests, so the loaded model object itself can't be shared
# with it - the model still loads lazily into each worker's own memory on its
# first request. What warm_embedding_model above avoids is the *download*:
# it populates the on-disk fastembed cache (jobs/services/embeddings.py's
# FASTEMBED_CACHE_DIR) so that lazy load reads from local disk instead of
# fetching from the Hugging Face Hub inside a live request.
