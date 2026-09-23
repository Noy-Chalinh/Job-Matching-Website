"""
One-off/offline backfill of JobEmbedding rows via sentence-transformers
(torch). NOT for the production web dyno - run manually or in CI, in an
environment with requirements-backfill.txt installed. See that file and
jobs/services/embeddings.py for why torch is kept out of the deployed app.

Postgres-only: JobEmbedding.vector is a pgvector VectorField, and this
command writes real data to it, so it refuses to run against a non-Postgres
connection (see jobs/models.py's JobEmbedding docstring for why the table
itself can still exist on SQLite even though nothing populates or queries
it there).

Usage:
    pip install -r requirements-backfill.txt
    python manage.py backfill_job_embeddings              # fill gaps only
    python manage.py backfill_job_embeddings --force       # recompute all
    python manage.py backfill_job_embeddings --dry-run

Safe to re-run: only fills gaps by default, and upserts by job either way.
"""
from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from jobs.models import Job, JobEmbedding
from jobs.services.embeddings import EmbeddingService


class Command(BaseCommand):
    help = "Backfill JobEmbedding rows via sentence-transformers (offline only, not for prod)."

    def add_arguments(self, parser):
        parser.add_argument(
            '--force', action='store_true',
            help='Recompute even jobs that already have an embedding for this model.',
        )
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        if connection.vendor != 'postgresql':
            raise CommandError(
                "JobEmbedding.vector is a pgvector column - this command only "
                "makes sense against the Postgres (Supabase) database, not "
                f"'{connection.vendor}'. Point DATABASE_URL at Postgres to run it."
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.stderr.write(self.style.ERROR(
                "sentence-transformers is not installed. Run:\n"
                "  pip install -r requirements-backfill.txt"
            ))
            return

        model_name = EmbeddingService.MODEL_NAME

        if options['force']:
            todo_ids = list(Job.objects.values_list('job_id', flat=True))
        else:
            already_done = set(
                JobEmbedding.objects.filter(model_name=model_name)
                .values_list('job__job_id', flat=True)
            )
            todo_ids = list(
                Job.objects.exclude(job_id__in=already_done).values_list('job_id', flat=True)
            )
        self.stdout.write(
            f"{len(todo_ids)} jobs need embedding{' (forced)' if options['force'] else ''}."
        )

        if not todo_ids:
            return
        if options['dry_run']:
            self.stdout.write("Dry run, not computing or writing anything.")
            return

        model = SentenceTransformer(model_name)
        batch_size = options['batch_size']
        done = 0
        for i in range(0, len(todo_ids), batch_size):
            batch_ids = todo_ids[i:i + batch_size]
            jobs = list(
                Job.objects.filter(job_id__in=batch_ids).only('id', 'job_id', 'job_title', 'raw_text')
            )
            # job_title + raw_text: the full text of the posting, not just
            # the extracted skills list (see JobMatcher._prefilter_jobs for
            # why raw_text is normally kept out of the hot request path -
            # that concern doesn't apply here, this runs offline).
            texts = [f"{job.job_title}\n{job.raw_text}".strip() for job in jobs]
            # normalize_embeddings=True: must match fastembed's L2-normalized
            # output, since scoring treats the dot product/cosine distance
            # as cosine similarity. No BGE query-prefix here - these are the
            # passage side of asymmetric search, not the query side.
            vectors = model.encode(texts, normalize_embeddings=True)
            JobEmbedding.objects.bulk_create(
                [
                    JobEmbedding(
                        job=job, vector=vec.tolist(),
                        model_name=model_name, dim=len(vec),
                    )
                    for job, vec in zip(jobs, vectors)
                ],
                update_conflicts=True,
                unique_fields=['job'],
                update_fields=['vector', 'model_name', 'dim', 'updated_at'],
            )
            done += len(jobs)
            self.stdout.write(f"  {done}/{len(todo_ids)} embedded")

        self.stdout.write(self.style.SUCCESS(f"Done: {done} job embeddings written."))
