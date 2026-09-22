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

# Memory management
max_requests = 100  # Restart workers after 100 requests to prevent memory leaks
max_requests_jitter = 20  # Add jitter to prevent all workers restarting at once

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

def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    print(f"Worker {worker.pid} received SIGINT/SIGQUIT")

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    print(f"Worker {worker.pid} received SIGABRT - likely out of memory!")
