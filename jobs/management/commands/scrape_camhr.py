"""
Scrape CamHR job postings straight into the database (RawJob / raw_jobs table).

This replaces the old data/scraping_script/camhr.py, which wrote scraped
data only to a local JSON file (data/raw_data/*.json). Point DATABASE_URL at
Supabase (or any Postgres) and every run upserts here instead, so scraped
data survives across machines and redeploys.

Usage:
    python manage.py scrape_camhr
    python manage.py scrape_camhr --max-pages 2       # test run, gentle on the API
    python manage.py scrape_camhr --delay 1
"""
import sys
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

sys.path.insert(0, str(Path(settings.BASE_DIR) / "data" / "scraping_script"))
import camhr_client  # noqa: E402

from jobs.models import RawJob


class Command(BaseCommand):
    help = "Scrape CamHR job postings into the raw_jobs table"

    def add_arguments(self, parser):
        parser.add_argument(
            '--max-pages', type=int, default=None,
            help='Only scrape this many pages (useful for a quick/test run)'
        )
        parser.add_argument(
            '--delay', type=float, default=2.0,
            help='Seconds to wait between per-job requests (default: 2.0)'
        )
        parser.add_argument(
            '--page-size', type=int, default=10,
            help='Jobs per listing page (default: 10, matches the site)'
        )

    def handle(self, *args, **options):
        delay = options['delay']
        page_size = options['page_size']

        total_pages = camhr_client.get_total_pages(page_size=page_size)
        if options['max_pages']:
            total_pages = min(total_pages, options['max_pages'])
        self.stdout.write(f'Scraping {total_pages} page(s) of CamHR listings...')

        created = 0
        updated = 0
        failed = 0

        for page in range(1, total_pages + 1):
            self.stdout.write(f'Page {page}/{total_pages}')
            job_ids = camhr_client.get_job_ids(page, page_size=page_size)

            for job_id in job_ids:
                raw = camhr_client.get_job_raw(job_id)
                if raw is None:
                    self.stdout.write(self.style.WARNING(f'  Failed job {job_id}'))
                    failed += 1
                    time.sleep(delay)
                    continue

                _, was_created = RawJob.objects.update_or_create(
                    source='camhr',
                    job_id=str(job_id),
                    defaults={'page': page, 'raw_data': raw},
                )
                created += was_created
                updated += not was_created
                time.sleep(delay)

        self.stdout.write(self.style.SUCCESS(
            f'\n✓ Scraped: {created} new, {updated} updated, {failed} failed'
        ))
        self.stdout.write(f'Total raw jobs in database: {RawJob.objects.count()}')
