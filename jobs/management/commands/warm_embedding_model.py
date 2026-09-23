"""
Download/extract the fastembed model into its persistent, project-local cache
directory (see jobs.services.embeddings.FASTEMBED_CACHE_DIR), so the first
live request on a fresh worker doesn't pay for a network fetch from the
Hugging Face Hub inside the request itself.

Run during build.sh, not on every deploy's first user request: a previous
production incident showed this download taking long enough, combined with
gunicorn's single sync worker and 300s timeout, to get the worker SIGKILLed
mid-request.

Failure here must not fail the build - the embedding service still falls
back to downloading at request time (slower, but functional) if this step
was skipped or failed.
"""
from django.core.management.base import BaseCommand

from jobs.services.embeddings import EmbeddingService


class Command(BaseCommand):
    help = "Warm the fastembed model's on-disk cache so it doesn't download during a live request."

    def handle(self, *args, **options):
        try:
            # Call _load_model() directly rather than embed_batch(): this only
            # needs to trigger the download/ONNX-session creation, not touch
            # the SkillEmbedding table with a fake "warmup" entry.
            EmbeddingService()._load_model()
            self.stdout.write(self.style.SUCCESS("Embedding model cache warmed."))
        except Exception as e:
            self.stdout.write(self.style.WARNING(
                f"Could not warm embedding model cache ({e}); "
                "it will be downloaded lazily on first use instead."
            ))
