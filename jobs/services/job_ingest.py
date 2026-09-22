"""
Shared helpers for turning a normalized job dict into a Job model instance.

Used by both:
- jobs.management.commands.normalize_camhr (RawJob -> Job, the primary path)
- jobs.management.commands.load_jobs (a normalized JSON file -> Job, kept for
  one-off imports / migrating data that predates the raw_jobs/jobs tables)
"""
from datetime import datetime, timezone as dt_timezone

from django.conf import settings
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime

from jobs.models import Job

UPDATE_FIELDS = [
    'source', 'raw_job', 'job_title', 'company', 'location', 'industry',
    'min_years_experience', 'education_level', 'education_major', 'skills',
    'languages', 'raw_text', 'pubdate', 'expdate',
]


def parse_job_date(value):
    """Accept epoch seconds/ms (int or numeric string) or ISO strings."""
    if value in (None, ''):
        return None
    try:
        num = float(value)
        if num > 1e11:  # milliseconds
            num /= 1000
        dt = datetime.fromtimestamp(num, tz=dt_timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        dt = parse_datetime(str(value))
        if dt is None:
            d = parse_date(str(value))
            dt = datetime(d.year, d.month, d.day) if d else None
    if dt is None:
        return None
    if settings.USE_TZ and timezone.is_naive(dt):
        dt = timezone.make_aware(dt)
    elif not settings.USE_TZ and timezone.is_aware(dt):
        dt = timezone.make_naive(dt)
    return dt


def build_job(job_id, data, source='camhr', raw_job=None):
    """Map a normalized record (nested or flat experience/education schema)
    to an unsaved Job instance, ready for bulk_create(update_conflicts=True).
    """
    experience = data.get('experience') or {}
    education = data.get('education') or {}
    years = data.get('min_years_experience', experience.get('min_years'))
    try:
        years = int(years or 0)
    except (TypeError, ValueError):
        years = 0
    return Job(
        job_id=job_id,
        source=source,
        raw_job=raw_job,
        job_title=(data.get('job_title') or '')[:200],
        company=(data.get('company') or '')[:200],
        location=(data.get('location') or '')[:100],
        industry=(data.get('industry') or '')[:100],
        min_years_experience=years,
        education_level=(data.get('education_level') or education.get('level') or '')[:50],
        education_major=(data.get('education_major') or education.get('major') or '')[:100],
        skills=data.get('skills') or [],
        languages=data.get('languages') or [],
        raw_text=data.get('raw_text') or '',
        pubdate=parse_job_date(data.get('pubdate')),
        expdate=parse_job_date(data.get('expdate')),
    )
