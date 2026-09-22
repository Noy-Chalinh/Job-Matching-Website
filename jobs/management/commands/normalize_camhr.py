"""
Extract skills/education/etc. from raw_jobs and upsert the result into jobs.

This replaces the old data/scraping_script/camhr_normalized.py, which read
a raw JSON file and wrote a normalized JSON file. Both ends are now the
database: read RawJob rows, write Job rows, no file in between.

Processing happens in small chunks, each written in its own short
transaction: a long-lived connection to a pooled Postgres (Supabase) can be
dropped mid-run, and one big all-or-nothing transaction would lose every
already-extracted job when that happens. Chunking bounds the loss to one
chunk and keeps each DB round trip short; --all/--limit/re-running afterward
still work exactly as before since already-normalized rows are skipped.

Usage:
    python manage.py normalize_camhr              # only rows not yet normalized
    python manage.py normalize_camhr --all         # re-normalize everything
    python manage.py normalize_camhr --limit 50    # test run
    python manage.py normalize_camhr --chunk-size 50
"""
import sys
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import close_old_connections, transaction
from django.db.utils import OperationalError, InterfaceError

sys.path.insert(0, str(Path(settings.BASE_DIR) / "data"))
sys.path.insert(0, str(Path(settings.BASE_DIR) / "data" / "scraping_script"))
from camhr_normalize_logic import normalize_job  # noqa: E402
from extraction.skills_extractor_v3 import EnhancedSkillsExtractor  # noqa: E402
from extraction.education_extractor_v3 import EnhancedEducationExtractor  # noqa: E402
from extraction.config import EnhancedExtractionConfig  # noqa: E402

from jobs.models import Job, RawJob
from jobs.services.job_ingest import UPDATE_FIELDS, build_job


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class Command(BaseCommand):
    help = 'Extract and normalize raw_jobs into jobs (skills, education, languages, ...)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--all', action='store_true',
            help='Re-normalize every raw job, including ones already normalized'
        )
        parser.add_argument(
            '--limit', type=int, default=None,
            help='Only process this many raw jobs (useful for a quick/test run)'
        )
        parser.add_argument(
            '--source', type=str, default='camhr',
            help='Only process raw jobs from this source (default: camhr)'
        )
        parser.add_argument(
            '--chunk-size', type=int, default=100,
            help='Extract+write this many jobs per DB transaction (default: 100)'
        )

    def handle(self, *args, **options):
        queryset = RawJob.objects.filter(source=options['source'])
        if not options['all']:
            queryset = queryset.filter(normalized_jobs__isnull=True)
        queryset = queryset.order_by('id')
        if options['limit']:
            queryset = queryset[:options['limit']]

        raw_jobs = list(queryset)
        if not raw_jobs:
            self.stdout.write(self.style.WARNING(
                'No raw jobs to normalize (use --all to re-process everything).'
            ))
            return

        self.stdout.write(f'Normalizing {len(raw_jobs)} raw job(s)...')
        self.stdout.write('Initializing extraction engine (spaCy + KeyBERT)...')
        config = EnhancedExtractionConfig()
        skills_extractor = EnhancedSkillsExtractor(config)
        education_extractor = EnhancedEducationExtractor(config)

        total_new = 0
        total_updated = 0
        total_errors = 0
        chunk_size = max(1, options['chunk_size'])
        num_chunks = (len(raw_jobs) + chunk_size - 1) // chunk_size

        for chunk_i, chunk in enumerate(chunked(raw_jobs, chunk_size), 1):
            jobs = {}
            for raw_job in chunk:
                try:
                    normalized = normalize_job(
                        raw_job.job_id, raw_job.raw_data, skills_extractor, education_extractor
                    )
                    jobs[raw_job.job_id] = build_job(
                        raw_job.job_id, normalized, source=raw_job.source, raw_job=raw_job
                    )
                except Exception as e:
                    self.stdout.write(self.style.ERROR(
                        f'Error normalizing raw job {raw_job.job_id}: {e}'
                    ))
                    total_errors += 1

            new_count, updated_count = self._write_chunk(jobs)
            total_new += new_count
            total_updated += updated_count
            self.stdout.write(
                f'  chunk {chunk_i}/{num_chunks}: +{new_count} new, {updated_count} updated '
                f'(running total: {total_new + total_updated}/{len(raw_jobs)})'
            )

        self.stdout.write(self.style.SUCCESS(
            f'\n✓ Normalized {total_new} new jobs, updated {total_updated} existing jobs'
        ))
        if total_errors:
            self.stdout.write(self.style.ERROR(f'✗ {total_errors} errors'))

    def _write_chunk(self, jobs, retries=2):
        """Upsert one chunk. Retries once on a dropped connection so a single
        flaky round trip doesn't abandon the whole (expensive) chunk."""
        if not jobs:
            return 0, 0
        for attempt in range(retries + 1):
            try:
                with transaction.atomic():
                    existing = Job.objects.filter(job_id__in=list(jobs)).count()
                    Job.objects.bulk_create(
                        list(jobs.values()),
                        update_conflicts=True,
                        unique_fields=['job_id'],
                        update_fields=UPDATE_FIELDS,
                    )
                return len(jobs) - existing, existing
            except (OperationalError, InterfaceError) as e:
                close_old_connections()
                if attempt == retries:
                    self.stdout.write(self.style.ERROR(
                        f'  Chunk failed after {retries} retries, skipping {len(jobs)} job(s): {e}'
                    ))
                    return 0, 0
                self.stdout.write(self.style.WARNING(
                    f'  DB connection dropped, retrying chunk ({attempt + 1}/{retries})...'
                ))
                time.sleep(2)
