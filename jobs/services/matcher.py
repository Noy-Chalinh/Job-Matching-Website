import re

from django.db import connection
from django.db.models import Q
from .embeddings import EmbeddingService, BGE_QUERY_PREFIX
from .scorers import (
    SkillScorer, TitleScorer, EducationScorer, ExperienceScorer, LanguageScorer,
    LocationScorer, soft_threshold,
)
from .skill_vocab import get_vocabulary, KHMER_CHARS
import logging

logger = logging.getLogger(__name__)

# Category weights used to combine per-job-category scores into a single
# match_score. When a job posting has no data for a category, that category
# is excluded from the weighted average (see match()) instead of being
# counted at its scorer's "no data" fallback value of 1.0 - a job that never
# mentions education, for example, should not be treated as a perfect
# education match just because it has nothing to disqualify a candidate on.
# job_semantic follows the same rule: excluded whenever a job has no
# JobEmbedding row (mid-backfill, embedding failure, or non-Postgres), never
# defaulted to 1.0 or 0.0. title is excluded when the user didn't give one.
CATEGORY_WEIGHTS = {
    'skill': 0.40,
    'title': 0.15,
    'job_semantic': 0.15,
    'education': 0.13,
    'experience': 0.12,
    'language': 0.03,
    'location': 0.02,
}

# Rescaling for pgvector's query-vs-job-text cosine similarity, for the same
# reason as scorers.soft_threshold: BGE's raw similarity between a short
# profile query and an unrelated full posting is still well above zero, so
# it's only meaningful relative to this range. Measured over the live jobs
# table with eight test profiles (accountant, developer, sales, designer, HR,
# site engineer, marketer, warehouse): the 150th-best candidate sits at
# ~0.55-0.68, the best at ~0.74-0.83.
JOB_SEMANTIC_LOW = 0.60
JOB_SEMANTIC_HIGH = 0.80

# Salary ranges, headcounts, etc. in titles ("assistant ceo 700$-900$").
_TITLE_NOISE = re.compile(r'\S*[\d$]\S*')


def clean_title(title):
    title = KHMER_CHARS.sub(' ', (title or '').lower())
    title = _TITLE_NOISE.sub(' ', title)
    return re.sub(r'[\s()\-–:/|,]+', ' ', title).strip()


class JobMatcher:
    """Database-only job matching service with exact + semantic matching"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vocabulary = get_vocabulary()
        self.skill_scorer = SkillScorer(self.embedding_service)
        self.title_scorer = TitleScorer()
        self.education_scorer = EducationScorer()
        self.experience_scorer = ExperienceScorer()
        self.language_scorer = LanguageScorer()
        self.location_scorer = LocationScorer()

    def _build_query_vector(self, user_profile, user_skills):
        """Embed a synthesized query string for ANN job search over
        JobEmbedding (full job_title + raw_text) via pgvector.

        There's no free-text resume field today (jobs/forms.py only has
        current_job_title, comma-separated skills, education_level/major) -
        this approximates one from those structured fields. Prefixed with
        BGE_QUERY_PREFIX since this is the asymmetric query side (a short
        query compared against long job passages), unlike the symmetric
        skill-vs-skill comparisons in SkillScorer.

        Returns None - meaning "skip semantic job search for this search" -
        on a non-Postgres connection, an empty profile, or any embedding
        failure, mirroring this app's existing graceful-degradation
        convention (e.g. SkillScorer.use_semantic).
        """
        if connection.vendor != 'postgresql':
            return None

        parts = []
        if user_profile.current_job_title:
            parts.append(user_profile.current_job_title)
        if user_skills:
            parts.append(f"Skills: {', '.join(user_skills)}")
        if user_profile.education_major:
            parts.append(user_profile.education_major)
        if not parts:
            return None

        try:
            # Deliberately not embed(): that goes through the skill cache and
            # would persist every distinct search query as a SkillEmbedding row.
            self.embedding_service._load_model()
            text = BGE_QUERY_PREFIX + '. '.join(parts)
            return next(iter(EmbeddingService._model.embed([text])))
        except Exception as e:
            logger.warning(f"Query embedding for semantic job search failed: {e}")
            return None

    def _prefilter_jobs(self, user_profile, max_candidates=150, query_vector=None):
        """
        Pre-filter jobs from database to reduce matching workload
        Returns: (list of job dicts, {job_id: raw cosine similarity} for jobs
        that were ANN-ranked; empty if query_vector is None or no
        JobEmbedding rows matched, in which case candidates fall back to the
        skill-overlap ordering below)
        """
        from jobs.models import Job

        user_location = user_profile.preferred_location.lower() if user_profile.preferred_location else None
        user_experience = user_profile.years_of_experience

        # No COUNT(*) queries here for logging: the database is a network
        # round trip away, and on a small instance those add up per search.
        query = Q()

        # Filter 1: Location match (if user not willing to relocate)
        if user_location and not user_profile.willing_to_relocate:
            query &= Q(location__icontains=user_location)

        # Filter 2: Experience requirement (user meets minimum)
        # Allow some flexibility: user_exp >= (job_min - 2)
        if user_experience > 0:
            query &= Q(min_years_experience__lte=user_experience + 2)

        jobs_qs = Job.objects.filter(query) if query else Job.objects.all()
        fields = (
            'job_id', 'job_title', 'company', 'location',
            'min_years_experience', 'education_level', 'education_major',
            'skills', 'languages'
        )

        # Semantic candidate selection: rank the DB-filtered jobs by real
        # relevance (cosine distance over full job_title + raw_text) instead
        # of the skill-overlap-then-arbitrary-order fallback below, so the
        # max_candidates cap isn't just "whatever the query returned first".
        if query_vector is not None:
            from pgvector.django import CosineDistance
            from jobs.models import JobEmbedding

            ann_rows = list(
                JobEmbedding.objects.filter(job__in=jobs_qs)
                .annotate(distance=CosineDistance('vector', query_vector))
                .order_by('distance')
                .values_list('job__job_id', 'distance')[:max_candidates]
            )
            if ann_rows:
                similarities = {job_id: 1.0 - float(distance) for job_id, distance in ann_rows}
                jobs_by_id = {
                    j['job_id']: j for j in Job.objects.filter(job_id__in=similarities).values(*fields)
                }
                # Preserve ANN rank order (values(job_id__in=...) doesn't).
                jobs = [jobs_by_id[jid] for jid, _ in ann_rows if jid in jobs_by_id]
                logger.info(f"Selected {len(jobs)} candidates via pgvector ANN ranking")
                return jobs, similarities

        # Fallback (no query vector / no embeddings): prefer jobs whose
        # canonical skills overlap the user's, but don't require it so
        # semantic matching can still find similar skills.
        jobs = list(jobs_qs.values(*fields))
        user_skills = set(self.vocabulary.user_skills(
            [s.skill_name for s in user_profile.skills.all()]
        ))
        if user_skills:
            jobs.sort(key=lambda job: not (
                user_skills & set(self.vocabulary.job_skills(job.get('skills'), job['job_title']))
            ))
        return jobs[:max_candidates], {}

    def match(self, user_profile, top_n=20):
        """
        Match user profile against jobs from database only
        Returns: List of job matches sorted by score
        """
        # Canonical skills on both sides (see skill_vocab): the user's typed
        # skills, and each job's stored phrases filtered down to real skills.
        user_skills = self.vocabulary.user_skills(
            [s.skill_name for s in user_profile.skills.all()]
        )
        user_title = clean_title(user_profile.current_job_title)

        # Get candidate jobs from database, ranked by pgvector ANN
        # similarity when available (see _build_query_vector/_prefilter_jobs),
        # else by the skill-overlap fallback.
        #
        # 150, not 500: ANN hands back the most semantically relevant jobs
        # first, so the extra 350 were the *least* relevant ones.
        query_vector = self._build_query_vector(user_profile, user_skills)
        jobs, semantic_similarities = self._prefilter_jobs(
            user_profile, max_candidates=150, query_vector=query_vector
        )

        user_language_data = [{
            'name': lang.language_name,
            'level': lang.proficiency
        } for lang in user_profile.languages.all()]

        # Pass 1: cheap exact-match pass over every job, no embedding calls yet.
        # Collects which jobs need semantic scoring and what skills they need it for.
        prepared = []
        pending_texts = set(user_skills)
        for job in jobs:
            job_skills = self.vocabulary.job_skills(job.get('skills'), job['job_title'])
            exact, unmatched = self.skill_scorer.prepare(user_skills, job_skills)
            job_title = clean_title(job['job_title'])
            prepared.append((job, job_skills, exact, unmatched, job_title))
            pending_texts.update(unmatched)
            if user_title and job_title:
                pending_texts.add(job_title)
        if user_title:
            pending_texts.add(user_title)

        # Pass 2: ONE embedding call for every skill and title across every
        # job in this batch, instead of up to len(jobs) separate small calls.
        # Canonical skills and job titles are a small, fixed vocabulary, so
        # after warm_embedding_model (build.sh) these are all cache hits.
        embeddings = {}
        if pending_texts:
            texts = sorted(pending_texts)
            try:
                embeddings = dict(zip(texts, self.embedding_service.embed_batch(texts)))
            except Exception as e:
                # If the model itself is unavailable, fall back to exact-only
                # scoring for the whole search rather than failing it, and
                # rather than retrying the same failure once per job below.
                logger.warning(f"Bulk embedding failed, falling back to exact-only matching: {e}")
                self.skill_scorer.use_semantic = False

        user_embeddings = None
        if user_skills and embeddings:
            import numpy as np
            user_embeddings = np.array([embeddings[s] for s in user_skills])
        user_title_embedding = embeddings.get(user_title) if user_title else None

        matches = []
        for job, job_skills, exact, unmatched, job_title in prepared:
            job_edu_level = job.get('education_level', '')
            job_edu_major = job.get('education_major', '')
            job_min_years = job.get('min_years_experience', 0)
            job_languages = job.get('languages', [])
            job_location = job.get('location', '')

            # Pass 3: combine exact + semantic score. The cache is already
            # warm from pass 2, so this does no new model inference.
            # A job with no recognizable skills provides zero evidence of
            # skill relevance, so it scores 0 at full weight rather than
            # being excluded - excluding it once let a job with no data but
            # an exact location match score 100%.
            skill_score, missing_skills = self.skill_scorer.finish(
                exact, unmatched, user_skills, job_skills, user_embeddings=user_embeddings
            )

            title_score = None
            if user_title_embedding is not None and job_title in embeddings:
                title_score = self.title_scorer.score(user_title_embedding, embeddings[job_title])

            job_semantic_score = None
            if job['job_id'] in semantic_similarities:
                job_semantic_score = soft_threshold(
                    semantic_similarities[job['job_id']], JOB_SEMANTIC_LOW, JOB_SEMANTIC_HIGH
                )

            education_score = self.education_scorer.score(
                user_level=user_profile.education_level,
                user_major=user_profile.education_major,
                job_level=job_edu_level,
                job_major=job_edu_major
            )

            experience_score = self.experience_scorer.score(
                user_years=user_profile.years_of_experience,
                job_min_years=job_min_years
            )

            language_score = self.language_scorer.score(
                user_languages=user_language_data,
                job_languages=job_languages
            )

            location_score = self.location_scorer.score(
                user_location=user_profile.preferred_location,
                job_location=job_location,
                willing_to_relocate=user_profile.willing_to_relocate
            )

            # The other categories: a job with no data for one of these is
            # excluded from the weighted average below (rather than counted
            # as a perfect match) - mirrors each scorer's own "no data"
            # shortcut exactly (see CATEGORY_WEIGHTS) so this check and the
            # scorer's internal 1.0 fallback can never disagree.
            has_data = {
                'title': title_score is not None,
                'education': bool(job_edu_level) or bool(job_edu_major),
                # job_min_years == 0 is ambiguous (explicit "0 years
                # required" vs. "not extracted" - job_ingest.build_job does
                # `years = int(years or 0)`), so it's treated as "no data",
                # consistent with ExperienceScorer's own 1.0 shortcut for
                # that same value.
                'experience': job_min_years > 0,
                'language': bool(job_languages),
                'location': bool(job_location),
                'job_semantic': job_semantic_score is not None,
            }
            category_scores = {
                'title': title_score or 0.0,
                'education': education_score,
                'experience': experience_score,
                'language': language_score,
                'location': location_score,
                'job_semantic': job_semantic_score or 0.0,
            }

            # Skill's weight is always included; the other categories are
            # only included when the job posting actually has data for them.
            total_weight = CATEGORY_WEIGHTS['skill'] + sum(
                CATEGORY_WEIGHTS[cat] for cat, present in has_data.items() if present
            )
            match_score = (
                skill_score * CATEGORY_WEIGHTS['skill'] + sum(
                    category_scores[cat] * CATEGORY_WEIGHTS[cat]
                    for cat, present in has_data.items() if present
                )
            ) / total_weight

            # Normalize job format for consistent output
            normalized_job = {
                'job_id': job['job_id'],
                'job_title': job['job_title'],
                'company': job.get('company', 'N/A'),
                'location': job.get('location', 'N/A'),
                'skills': job_skills,
                'education': {
                    'level': job_edu_level,
                    'major': job_edu_major
                },
                'experience': {
                    'min_years': job_min_years
                },
                'languages': job_languages
            }

            matches.append({
                'job': normalized_job,
                'match_score': match_score,
                'skill_score': skill_score,
                'title_score': title_score,
                'job_semantic_score': job_semantic_score,
                'education_score': education_score,
                'experience_score': experience_score,
                'language_score': language_score,
                'location_score': location_score,
                'missing_skills': missing_skills
            })

        # Sort by match score (descending)
        matches.sort(key=lambda x: x['match_score'], reverse=True)

        # The same posting is often re-listed under several job_ids (one
        # company had five identical "sales executive" rows), which filled
        # the whole top 5 with one job. Keep the best-scoring copy only.
        seen, unique = set(), []
        for m in matches:
            key = (clean_title(m['job']['job_title']), (m['job']['company'] or '').lower(), tuple(m['job']['skills']))
            if key not in seen:
                seen.add(key)
                unique.append(m)

        return unique[:top_n]
