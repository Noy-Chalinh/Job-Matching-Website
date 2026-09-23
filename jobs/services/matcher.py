from django.db import connection
from django.db.models import Q, Count
from .embeddings import EmbeddingService, BGE_QUERY_PREFIX
from .scorers import SkillScorer, EducationScorer, ExperienceScorer, LanguageScorer, LocationScorer
from .skill_gap import SkillGapAnalyzer
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
# defaulted to 1.0 or 0.0.
CATEGORY_WEIGHTS = {
    'skill': 0.50,
    'job_semantic': 0.15,
    'education': 0.18,
    'experience': 0.14,
    'language': 0.02,
    'location': 0.01,
}


class JobMatcher:
    """Database-only job matching service with exact + semantic matching"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.skill_scorer = SkillScorer(self.embedding_service)
        self.education_scorer = EducationScorer()
        self.experience_scorer = ExperienceScorer()
        self.language_scorer = LanguageScorer()
        self.location_scorer = LocationScorer()
        self.skill_gap_analyzer = SkillGapAnalyzer()

    def _build_query_vector(self, user_profile):
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

        skill_names = [s.skill_name for s in user_profile.skills.all()]
        parts = []
        if user_profile.current_job_title:
            parts.append(user_profile.current_job_title)
        if skill_names:
            parts.append(f"Skills: {', '.join(skill_names)}")
        if user_profile.education_major:
            parts.append(user_profile.education_major)
        if not parts:
            return None

        try:
            return self.embedding_service.embed(BGE_QUERY_PREFIX + '. '.join(parts))
        except Exception as e:
            logger.warning(f"Query embedding for semantic job search failed: {e}")
            return None

    def _prefilter_jobs(self, user_profile, max_candidates=150, query_vector=None):
        """
        Pre-filter jobs from database to reduce matching workload
        Returns: (list of job dicts, {job_id: job_semantic_score} for jobs
        that were ANN-ranked; empty if query_vector is None or no
        JobEmbedding rows matched, in which case candidates fall back to the
        skill-overlap ordering below)
        """
        from jobs.models import Job

        # Extract user data
        user_skills = [s.skill_name.lower() for s in user_profile.skills.all()]
        user_location = user_profile.preferred_location.lower() if user_profile.preferred_location else None
        user_experience = user_profile.years_of_experience

        total_jobs = Job.objects.count()
        logger.info(f"Starting prefilter with {total_jobs} total jobs")
        logger.info(f"User skills: {user_skills}")
        logger.info(f"User location: {user_location}, willing to relocate: {user_profile.willing_to_relocate}")
        logger.info(f"User experience: {user_experience} years")

        # Build filter query
        query = Q()

        # Filter 1: Location match (if user not willing to relocate)
        if user_location and not user_profile.willing_to_relocate:
            query &= Q(location__icontains=user_location)
            logger.info(f"Applied location filter: {user_location}")

        # Filter 2: Experience requirement (user meets minimum)
        # Allow some flexibility: user_exp >= (job_min - 2)
        if user_experience > 0:
            query &= Q(min_years_experience__lte=user_experience + 2)
            logger.info(f"Applied experience filter: <= {user_experience + 2} years")

        # Start with filtered base queryset
        jobs_qs = Job.objects.filter(query) if query else Job.objects.all()
        jobs_after_db_filter = jobs_qs.count()
        logger.info(f"Jobs after database filters: {jobs_after_db_filter}")

        # Semantic candidate selection: rank the DB-filtered jobs by real
        # relevance (cosine distance over full job_title + raw_text) instead
        # of the skill-overlap-then-arbitrary-order fallback below, so the
        # max_candidates cap isn't just "whatever the query returned first".
        # Falls through to the fallback if there's no query vector (non-
        # Postgres, empty profile, embedding failure) or no job has been
        # backfilled with an embedding yet.
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
                semantic_scores = {
                    job_id: max(0.0, 1.0 - float(distance)) for job_id, distance in ann_rows
                }
                jobs_by_id = {
                    j['job_id']: j for j in Job.objects.filter(job_id__in=semantic_scores).values(
                        'job_id', 'job_title', 'company', 'location',
                        'min_years_experience', 'education_level', 'education_major',
                        'skills', 'languages'
                    )
                }
                # Preserve ANN rank order (values(job_id__in=...) doesn't).
                jobs = [jobs_by_id[jid] for jid, _ in ann_rows if jid in jobs_by_id]
                logger.info(f"Selected {len(jobs)} candidates via pgvector ANN ranking")
                return jobs, semantic_scores

        # Filter 3: Skill overlap (using PostgreSQL JSONB containment)
        # This is more complex - we'll fetch all and filter in Python for now
        # Future optimization: Use PostgreSQL GIN index on skills JSONB field

        # Only the columns actually used below - raw_text/industry/pubdate/expdate
        # are never read again, and raw_text especially is expensive to carry
        # around for up to max_candidates rows on a memory-constrained instance.
        jobs = list(jobs_qs.values(
            'job_id', 'job_title', 'company', 'location',
            'min_years_experience', 'education_level', 'education_major',
            'skills', 'languages'
        ))

        # Filter 4: Prefer jobs with skill overlap, but don't require it
        # This allows semantic matching to find similar skills
        if user_skills:
            jobs_with_overlap = []
            jobs_without_overlap = []
            
            for job in jobs:
                job_skills = [s.lower() for s in job.get('skills', [])]
                if set(user_skills) & set(job_skills):  # At least 1 skill match
                    jobs_with_overlap.append(job)
                else:
                    jobs_without_overlap.append(job)
            
            logger.info(f"Jobs with skill overlap: {len(jobs_with_overlap)}")
            logger.info(f"Jobs without skill overlap: {len(jobs_without_overlap)}")
            
            # Prioritize jobs with overlap, but include others too (up to max_candidates)
            jobs = jobs_with_overlap + jobs_without_overlap
        
        logger.info(f"Total jobs after all filters: {len(jobs)}")

        # Limit to max_candidates. No ANN scores in this fallback path.
        return jobs[:max_candidates], {}

    def match(self, user_profile, top_n=20):
        """
        Match user profile against jobs from database only
        Returns: List of job matches sorted by score
        """
        # Get candidate jobs from database, ranked by pgvector ANN
        # similarity when available (see _build_query_vector/_prefilter_jobs),
        # else by the skill-overlap fallback.
        #
        # 150, not 500: ANN hands back the most semantically relevant jobs
        # first, so the extra 350 were the *least* relevant ones - scoring
        # them cost a 3x bigger Python loop (and 3x more skill lookups) per
        # search to influence a top-5 result set they'd never reach.
        query_vector = self._build_query_vector(user_profile)
        jobs, semantic_scores = self._prefilter_jobs(
            user_profile, max_candidates=150, query_vector=query_vector
        )

        matches = []

        # Evaluate these querysets once per search instead of once per candidate job
        user_skill_names = [s.skill_name for s in user_profile.skills.all()]
        user_language_data = [{
            'name': lang.language_name,
            'level': lang.proficiency
        } for lang in user_profile.languages.all()]

        # Precompute user skill embeddings once per search (SkillScorer falls back to
        # computing them itself if this is None, but that would mean re-embedding the
        # same user skills for every candidate job)
        user_embeddings = None
        if user_skill_names:
            user_embeddings = self.embedding_service.embed_batch(
                list(set(s.lower() for s in user_skill_names))
            )

        # Pass 1: cheap exact-match pass over every job, no embedding calls yet.
        # Collects which jobs need semantic scoring and what skills they need it for.
        prepared = []
        pending_skills = set()
        for job in jobs:
            job_skills = job.get('skills', [])
            exact_score, unmatched = self.skill_scorer.prepare(user_skill_names, job_skills)
            prepared.append((job, job_skills, exact_score, unmatched))
            if unmatched:
                pending_skills.update(unmatched)

        # Pass 2: ONE embedding call for every unmatched skill across every job
        # in this batch, instead of up to len(jobs) separate small calls - this
        # is what keeps a 500-candidate search from doing hundreds of round
        # trips through the model.
        if pending_skills:
            try:
                self.embedding_service.embed_batch(list(pending_skills))
            except Exception as e:
                # Same graceful degradation as before this batching change:
                # if the model itself is unavailable, fall back to exact-only
                # scoring for the whole search rather than failing it, and
                # rather than retrying the same failure once per job below.
                logger.warning(f"Bulk skill embedding failed, falling back to exact-only matching: {e}")
                self.skill_scorer.use_semantic = False

        for job, job_skills, exact_score, unmatched in prepared:
            job_edu_level = job.get('education_level', '')
            job_edu_major = job.get('education_major', '')
            job_min_years = job.get('min_years_experience', 0)
            job_languages = job.get('languages', [])
            job_location = job.get('location', '')

            # Pass 3: combine exact + semantic score. The cache is already
            # warm from pass 2, so this does no new model inference.
            skill_score = self.skill_scorer.finish(
                exact_score, unmatched, user_skill_names, user_embeddings=user_embeddings
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

            job_semantic_score = semantic_scores.get(job['job_id'])

            # Skill is the core signal for this product (50% weight) - unlike
            # the other categories, a job with no listed skills provides zero
            # evidence of relevance, and must not be treated as a perfect
            # match (SkillScorer.prepare's own 1.0 shortcut) nor excluded
            # from the weighted average. Excluding it would let a job with
            # zero skill data hit 100% purely off e.g. an exact location
            # match, out-ranking jobs with genuine (if partial) skill
            # overlap - observed live: a job with no data at all except a
            # location matching the user's exactly scored 100%, ahead of
            # real dev-skill postings at the same location. So: no evidence
            # of skill relevance is scored as no skill credit, always at
            # full weight, rather than "unknown, don't count it against you".
            if not job_skills:
                skill_score = 0.0

            # The other categories: a job with no data for one of these is
            # excluded from the weighted average below (rather than counted
            # as a perfect match) - mirrors each scorer's own "no data"
            # shortcut exactly (see CATEGORY_WEIGHTS) so this check and the
            # scorer's internal 1.0 fallback can never disagree. job_semantic
            # follows the same principle: excluded whenever there's no
            # JobEmbedding row for this job (mid-backfill, embedding
            # failure, or non-Postgres - see _build_query_vector).
            has_data = {
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

            # Skill gap analysis
            missing_skills = self.skill_gap_analyzer.analyze(
                user_skills=user_skill_names,
                job_skills=job_skills
            )

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
                'job_semantic_score': job_semantic_score,
                'education_score': education_score,
                'experience_score': experience_score,
                'language_score': language_score,
                'location_score': location_score,
                'missing_skills': missing_skills
            })

        # Sort by match score (descending)
        matches.sort(key=lambda x: x['match_score'], reverse=True)

        return matches[:top_n]
