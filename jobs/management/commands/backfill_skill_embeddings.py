"""
One-off/offline backfill of SkillEmbedding rows via sentence-transformers
(torch). NOT for the production web dyno - run manually or in CI, in an
environment with requirements-backfill.txt installed. See that file and
jobs/services/embeddings.py for why torch is kept out of the deployed app.

Usage:
    pip install -r requirements-backfill.txt
    python manage.py backfill_skill_embeddings              # fill gaps only
    python manage.py backfill_skill_embeddings --force       # recompute all
    python manage.py backfill_skill_embeddings --dry-run

Safe to re-run: only fills gaps by default, and upserts by skill either way,
so it never conflicts with embeddings the live write-through path (see
EmbeddingService._write_through) may have already inserted for a skill.
"""
from django.core.management.base import BaseCommand

from jobs.models import Job, SkillEmbedding
from jobs.services.embeddings import EmbeddingService, normalize_skill


def discover_distinct_skills():
    """Flatten Job.skills across every row, in Python rather than a
    Postgres-specific jsonb function, to stay portable between Postgres
    (prod) and SQLite (local dev)."""
    seen = set()
    for skills in Job.objects.values_list('skills', flat=True).iterator(chunk_size=1000):
        for raw in (skills or []):
            normalized = normalize_skill(raw)
            if normalized:
                seen.add(normalized)
    return seen


class Command(BaseCommand):
    help = "Backfill SkillEmbedding rows via sentence-transformers (offline only, not for prod)."

    def add_arguments(self, parser):
        parser.add_argument(
            '--force', action='store_true',
            help='Recompute even skills that already have a row for this model.',
        )
        parser.add_argument('--batch-size', type=int, default=256)
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.stderr.write(self.style.ERROR(
                "sentence-transformers is not installed. Run:\n"
                "  pip install -r requirements-backfill.txt"
            ))
            return

        model_name = EmbeddingService.MODEL_NAME
        all_skills = discover_distinct_skills()
        self.stdout.write(f"Discovered {len(all_skills)} distinct normalized skills.")

        if options['force']:
            todo = sorted(all_skills)
        else:
            existing = set(
                SkillEmbedding.objects.filter(model_name=model_name).values_list('skill', flat=True)
            )
            todo = sorted(all_skills - existing)
        self.stdout.write(
            f"{len(todo)} skills need embedding{' (forced)' if options['force'] else ''}."
        )

        if not todo:
            return
        if options['dry_run']:
            self.stdout.write("Dry run, not computing or writing anything.")
            return

        model = SentenceTransformer(model_name)
        batch_size = options['batch_size']
        done = 0
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            # normalize_embeddings=True: must match fastembed's L2-normalized
            # output, since scoring treats the dot product as cosine similarity.
            vectors = model.encode(batch, normalize_embeddings=True)
            SkillEmbedding.objects.bulk_create(
                [
                    SkillEmbedding(
                        skill=skill, vector=vec.tolist(),
                        model_name=model_name, dim=len(vec),
                    )
                    for skill, vec in zip(batch, vectors)
                ],
                update_conflicts=True,
                unique_fields=['skill'],
                update_fields=['vector', 'model_name', 'dim', 'updated_at'],
            )
            done += len(batch)
            self.stdout.write(f"  {done}/{len(todo)} embedded")

        self.stdout.write(self.style.SUCCESS(f"Done: {done} skill embeddings written."))
