from django.db import models
from pgvector.django import VectorField


class RawJob(models.Model):
    """Raw scraped job payload, exactly as returned by the source API.

    This is the durable replacement for data/raw_data/*.json: every job the
    scraper fetches is upserted here first, before any extraction runs.
    """
    source = models.CharField(max_length=50, default='camhr', db_index=True)
    job_id = models.CharField(max_length=50, db_index=True)
    page = models.IntegerField(null=True, blank=True)
    raw_data = models.JSONField()  # verbatim API response for this job
    scraped_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.source}:{self.job_id}"

    class Meta:
        db_table = 'raw_jobs'
        ordering = ['-scraped_at']
        constraints = [
            models.UniqueConstraint(fields=['source', 'job_id'], name='uniq_raw_job_source_job_id'),
        ]


class Job(models.Model):
    """Normalized job posting.

    This is the durable replacement for data/normalized_data/*.json: the
    extraction pipeline upserts directly here instead of writing a JSON file.
    """
    job_id = models.CharField(max_length=50, unique=True, db_index=True)
    source = models.CharField(max_length=50, default='camhr', db_index=True)
    raw_job = models.ForeignKey(
        RawJob, null=True, blank=True, on_delete=models.SET_NULL, related_name='normalized_jobs'
    )
    job_title = models.CharField(max_length=200, db_index=True)
    company = models.CharField(max_length=200, blank=True)
    location = models.CharField(max_length=100, blank=True, db_index=True)
    industry = models.CharField(max_length=100, blank=True)

    # Experience
    min_years_experience = models.IntegerField(default=0)

    # Education
    education_level = models.CharField(max_length=50, blank=True)
    education_major = models.CharField(max_length=100, blank=True)

    # JSON fields
    skills = models.JSONField(default=list)  # ["python", "django"]
    languages = models.JSONField(default=list)  # [{"name": "english", "level": "good"}]

    # Metadata
    raw_text = models.TextField(blank=True)
    pubdate = models.DateTimeField(null=True, blank=True)
    expdate = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    normalized_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.job_id}: {self.job_title}"

    class Meta:
        db_table = 'jobs'
        ordering = ['-pubdate']


class SkillEmbedding(models.Model):
    """Persistent cache of one embedding vector per distinct normalized skill
    string (see jobs.services.embeddings.normalize_skill). Skill vocabulary is
    far smaller than job count and shared across all jobs/users, so caching
    here - rather than per-job - is what lets a search skip model inference
    for any skill it has already seen, even across worker restarts/deploys.
    """
    skill = models.CharField(max_length=255, unique=True)
    vector = models.JSONField()  # list[float], length == dim, L2-normalized
    model_name = models.CharField(max_length=100, default='BAAI/bge-small-en-v1.5')
    dim = models.PositiveSmallIntegerField(default=384)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.skill

    class Meta:
        db_table = 'skill_embeddings'


class JobEmbedding(models.Model):
    """One full-text (job_title + raw_text) embedding per job, for semantic
    search via pgvector - see jobs.services.matcher for how this is queried
    and jobs/management/commands/backfill_job_embeddings.py for how it's
    populated offline. Unlike SkillEmbedding (cached per distinct skill
    string, shared across jobs), this is one row per job.

    Postgres-only in practice: `vector`'s db_type() is backend-agnostic so
    this table can still be created on SQLite local dev, but its HNSW index
    (added via a separate migration gated on connection.vendor - see
    jobs/migrations) and any ANN query against it only work on Postgres.
    Code that queries this table must check connection.vendor == 'postgresql'
    first and skip semantic job search otherwise (mirrors matcher.py's
    existing use_semantic fallback for skill scoring).
    """
    job = models.OneToOneField(Job, on_delete=models.CASCADE, related_name='embedding')
    vector = VectorField(dimensions=384)
    model_name = models.CharField(max_length=100, default='BAAI/bge-small-en-v1.5')
    dim = models.PositiveSmallIntegerField(default=384)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"embedding for {self.job_id}"

    class Meta:
        db_table = 'job_embeddings'


# Temporary models for matching (not stored in database)
class UserProfile:
    """Temporary user profile for matching.

    Holds every experience ({'title', 'years'}) and education ({'level',
    'major'}) entry from the search form. Either pass those lists, or the
    single-value arguments (a one-entry profile). The single-value
    attributes are always set as summaries: total years across all roles,
    the first job title, and the highest education level with its major.
    """

    # Ordered lowest to highest, matching jobs.forms EDUCATION_LEVEL_CHOICES.
    EDUCATION_RANK = {'high school': 1, 'associate': 2, "bachelor's degree": 3, "master's degree": 4, 'phd': 5}

    def __init__(self, years_of_experience=0, current_job_title='',
                 education_level='', education_major='',
                 preferred_location='', willing_to_relocate=False,
                 experiences=None, educations=None):
        if experiences is None:
            experiences = (
                [{'title': current_job_title, 'years': years_of_experience}]
                if current_job_title or years_of_experience else []
            )
        if educations is None:
            educations = (
                [{'level': education_level, 'major': education_major}]
                if education_level or education_major else []
            )
        self.experiences = experiences
        self.educations = educations

        self.years_of_experience = sum(e['years'] for e in experiences)
        self.current_job_title = next((e['title'] for e in experiences if e['title']), '')
        highest = max(
            educations, key=lambda e: self.EDUCATION_RANK.get(e['level'], 0), default=None
        )
        self.education_level = highest['level'] if highest else ''
        self.education_major = highest['major'] if highest else ''
        self.preferred_location = preferred_location
        self.willing_to_relocate = willing_to_relocate


class UserSkill:
    """Temporary user skill for matching"""
    def __init__(self, skill_name, proficiency='intermediate'):
        self.skill_name = skill_name
        self.proficiency = proficiency


class UserLanguage:
    """Temporary user language for matching"""
    def __init__(self, language_name, proficiency):
        self.language_name = language_name
        self.proficiency = proficiency
