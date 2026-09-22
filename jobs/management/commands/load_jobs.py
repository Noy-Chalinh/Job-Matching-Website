"""
Load job data from a normalized JSON file into the database.

This is a secondary, manual-import path (e.g. for a one-off migration or a
backup file). The primary pipeline is now:
    python manage.py scrape_camhr      # CamHR API -> raw_jobs table
    python manage.py normalize_camhr   # raw_jobs -> jobs table (extraction)
which stores everything directly in the database (Supabase/Postgres) and
never touches a JSON file.

Usage:
    python manage.py load_jobs --file path/to/file.json [--clear]

Re-running is safe: jobs are upserted by job_id, so existing jobs are
updated instead of duplicated.
"""
import json
from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import transaction

from jobs.models import Job
from jobs.services.job_ingest import UPDATE_FIELDS, build_job


class Command(BaseCommand):
    help = 'Load job data from a normalized JSON file into the database (manual/import use only)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            required=True,
            help='Path to JSON file containing normalized job data',
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing jobs before loading'
        )

    def handle(self, *args, **options):
        file_path = Path(options['file'])
        if not file_path.exists():
            self.stdout.write(self.style.ERROR(
                f'File not found: {file_path}'
            ))
            return

        self.stdout.write(f'Loading jobs from: {file_path}')

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                jobs_data = json.load(f)
        except json.JSONDecodeError as e:
            self.stdout.write(self.style.ERROR(
                f'Invalid JSON file: {e}'
            ))
            return
        except Exception as e:
            self.stdout.write(self.style.ERROR(
                f'Error reading file: {e}'
            ))
            return

        # Build Job objects (last record wins for duplicate job_ids)
        jobs = {}
        skipped = 0
        for job_data in jobs_data:
            job_id = job_data.get('job_id')
            if not job_id:
                skipped += 1
                continue
            job_id = str(job_id)[:50]
            jobs[job_id] = build_job(job_id, job_data, source=job_data.get('source', 'camhr'))

        with transaction.atomic():
            if options['clear']:
                deleted_count = Job.objects.count()
                Job.objects.all().delete()
                self.stdout.write(self.style.WARNING(
                    f'Cleared {deleted_count} existing jobs'
                ))

            existing = Job.objects.filter(job_id__in=list(jobs)).count()
            Job.objects.bulk_create(
                list(jobs.values()),
                batch_size=500,
                update_conflicts=True,
                unique_fields=['job_id'],
                # Never touch raw_job here: a JSON import has no RawJob to
                # link, and this must not null out a link normalize_camhr set.
                update_fields=[f for f in UPDATE_FIELDS if f != 'raw_job'],
            )

        self.stdout.write(self.style.SUCCESS(
            f'\n✓ Created {len(jobs) - existing} jobs, updated {existing} existing jobs'
        ))
        if skipped > 0:
            self.stdout.write(self.style.WARNING(
                f'⚠ Skipped {skipped} records without job_id'
            ))

        total = Job.objects.count()
        self.stdout.write(self.style.SUCCESS(
            f'\nTotal jobs in database: {total}'
        ))
