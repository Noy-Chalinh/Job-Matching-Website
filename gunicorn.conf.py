"""Gunicorn configuration for Render deployment"""

import multiprocessing
import os

# Bind to the port Render provides
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Worker configuration
workers = 1  # Use only 1 worker to save memory (ML models are heavy)
worker_class = "sync"
# 1, not 2: two concurrent requests each doing model-load + 500-candidate
# scoring roughly doubles peak memory, which is what OOM-killed the worker
# on Render's free (512MB) tier. Serializing requests costs latency, not RAM.
threads = 1

# Timeout settings (ML model loading can take time)
timeout = 300  # 5 minutes for workers to respond (model loading on first request)
graceful_timeout = 120  # 2 minutes for graceful shutdown
keepalive = 5

# Memory management.
# Deliberately high: every worker restart re-loads the ~90MB ONNX embedding
# model (see post_worker_init below), and with a single sync worker that
# boot time is dead air for queued requests. Recycling every 100 requests
# meant paying that cost constantly. The L1 embedding cache is bounded by
# the skill vocabulary (~10k entries, ~15MB), so it isn't an unbounded leak.
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"

# Preload application to save memory (shared code before forking)
preload_app = False  # Set to False to avoid loading ML models during startup

# Worker lifecycle hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    print("Starting Gunicorn server...")

def when_ready(server):
    """Called just after the server is started."""
    print("Gunicorn server is ready. Waiting for requests...")

def post_worker_init(worker):
    """Load the embedding model into this worker before it accepts traffic.

    Runs after the worker has loaded the WSGI app (so Django is fully set
    up), but before it serves its first request. Without this the model
    loads lazily *inside* whichever request happens to need it first, which
    on a cold instance means that user waits for a ~90MB Hugging Face Hub
    download (observed live: 'Fetching 5 files' mid-request). build.sh's
    warm_embedding_model step only populates the on-disk cache, and only if
    the deploy actually runs build.sh - this hook is what guarantees the
    cost is paid at boot either way.

    Best-effort: a failure here must not stop the worker from booting, since
    the service still degrades gracefully to exact-only matching.
    """
    try:
        from jobs.services.embeddings import EmbeddingService

        EmbeddingService()._load_model()
        print(f"Worker {worker.pid}: embedding model preloaded.")
    except Exception as e:
        print(
            f"Worker {worker.pid}: could not preload embedding model ({e}); "
            "it will load lazily on first use instead."
        )


def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    print(f"Worker {worker.pid} received SIGINT/SIGQUIT")

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    print(f"Worker {worker.pid} received SIGABRT - likely out of memory!")
