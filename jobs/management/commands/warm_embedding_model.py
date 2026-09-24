"""
Download/extract the fastembed model into its persistent, project-local cache
directory (see jobs.services.embeddings.FASTEMBED_CACHE_DIR), so the first
live request on a fresh worker doesn't pay for a network fetch from the
Hugging Face Hub inside the request itself.

Then precompute embeddings for everything a search compares against that
isn't user input: every canonical skill in the vocabulary (see
jobs.services.skill_vocab) and every job title. Both go through
EmbeddingService.embed_batch, which persists them to the SkillEmbedding
table, so live searches only ever read them from there - model inference on
a small instance's CPU is by far the slowest part of a search, and this moves
it into the build, which has CPU to spare.

Run during build.sh, not on every deploy's first user request: a previous
production incident showed this download taking long enough, combined with
gunicorn's single sync worker and 300s timeout, to get the worker SIGKILLed
mid-request.

Failure here must not fail the build - the embedding service still falls
back to downloading/embedding at request time (slower, but functional) if
this step was skipped or failed.
"""
from django.core.management.base import BaseCommand

from jobs.services.embeddings import EmbeddingService


class Command(BaseCommand):
    help = "Warm the fastembed model cache and precompute skill/title embeddings."

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-only', action='store_true',
            help="Only download the model; don't touch the database.",
        )

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
            return

        if options['model_only']:
            return

        try:
            from jobs.models import Job
            from jobs.services.matcher import clean_title
            from jobs.services.skill_vocab import get_vocabulary

            texts = set(get_vocabulary().alias_to_canonical.values())
            texts.update(
                t for t in (clean_title(title) for title in Job.objects.values_list('job_title', flat=True)) if t
            )
            texts = sorted(texts)
            service = EmbeddingService()
            for i in range(0, len(texts), 256):
                service.embed_batch(texts[i:i + 256])
            self.stdout.write(self.style.SUCCESS(
                f"Skill/title embeddings ready ({len(texts)} texts)."
            ))
        except Exception as e:
            self.stdout.write(self.style.WARNING(
                f"Could not precompute skill/title embeddings ({e}); "
                "they will be computed on first use instead."
            ))
