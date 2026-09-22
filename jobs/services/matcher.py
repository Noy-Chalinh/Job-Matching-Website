from django.db.models import Q, Count
from .embeddings import EmbeddingService
from .scorers import SkillScorer, EducationScorer, ExperienceScorer, LanguageScorer, LocationScorer
from .skill_gap import SkillGapAnalyzer
import logging

logger = logging.getLogger(__name__)


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

    def _prefilter_jobs(self, user_profile, max_candidates=500):
        """
        Pre-filter jobs from database to reduce matching workload
        Returns: QuerySet of filtered Job objects
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

        # Limit to max_candidates
        return jobs[:max_candidates]

    def match(self, user_profile, top_n=20):
        """
        Match user profile against jobs from database only
        Returns: List of job matches sorted by score
        """
        # Get candidate jobs from database
        jobs = self._prefilter_jobs(user_profile, max_candidates=500)

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

        for job in jobs:
            # Database format: flat dictionary
            job_skills = job.get('skills', [])
            job_edu_level = job.get('education_level', '')
            job_edu_major = job.get('education_major', '')
            job_min_years = job.get('min_years_experience', 0)
            job_languages = job.get('languages', [])
            job_location = job.get('location', '')

            # Compute component scores
            skill_score = self.skill_scorer.score(
                user_skills=user_skill_names,
                job_skills=job_skills,
                user_embeddings=user_embeddings
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

            # Weighted aggregation - Skills prioritized
            match_score = (
                skill_score * 0.60 +  # Increased from 40% to 60%
                education_score * 0.20 +  # Reduced from 25% to 20%
                experience_score * 0.15 +  # Reduced from 20% to 15%
                language_score * 0.03 +  # Reduced from 10% to 3%
                location_score * 0.02  # Reduced from 5% to 2%
            )

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
                'education_score': education_score,
                'experience_score': experience_score,
                'language_score': language_score,
                'location_score': location_score,
                'missing_skills': missing_skills
            })

        # Sort by match score (descending)
        matches.sort(key=lambda x: x['match_score'], reverse=True)

        return matches[:top_n]
