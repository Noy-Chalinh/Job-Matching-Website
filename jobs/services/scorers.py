import numpy as np


class SkillScorer:
    """Hybrid exact + semantic skill matching.

    Scoring a batch of jobs is split into three steps so the embedding model
    is invoked once per search instead of once per job:
      1. prepare() - cheap exact-match pass per job, no embedding calls
      2. one embed_batch() call, over every job's unmatched skills combined
      3. finish() - combines exact + semantic score per job, using the now-
         warm embedding cache (near-instant, no new model inference)
    score() still exists as a single-job convenience wrapper (e.g. for tests),
    it just does all three steps itself.
    """

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.use_semantic = True  # Flag to disable semantic matching if it fails

    def prepare(self, user_skills, job_skills):
        """Exact-match pass only. Returns (exact_score, unmatched_job_skills)
        where unmatched_job_skills is a non-empty set if semantic scoring is
        needed, or None if not (good exact match, no requirements, or
        semantic matching already disabled)."""
        if not job_skills:
            return 1.0, None

        user_set = set(s.lower() for s in user_skills)
        job_set = set(s.lower() for s in job_skills)

        exact_matches = user_set & job_set
        exact_score = len(exact_matches) / len(job_set)

        if exact_score >= 0.7 or not self.use_semantic:
            return exact_score, None

        unmatched_job_skills = job_set - exact_matches
        return exact_score, (unmatched_job_skills or None)

    def finish(self, exact_score, unmatched_job_skills, user_skills, user_embeddings=None):
        """Combine exact_score with semantic similarity for unmatched skills.
        Cheap as long as embed_batch's cache was already warmed for
        unmatched_job_skills (and user_embeddings, if not passed in)."""
        if not unmatched_job_skills or not self.use_semantic:
            return exact_score

        try:
            if user_embeddings is None:
                user_embeddings = self.embedding_service.embed_batch(
                    list(set(s.lower() for s in user_skills))
                )
            job_embeddings = self.embedding_service.embed_batch(list(unmatched_job_skills))

            # Embeddings are L2-normalized, so cosine similarity is just the
            # dot product: one matrix multiply replaces the per-pair Python loop.
            sims = np.asarray(job_embeddings) @ np.asarray(user_embeddings).T
            semantic_score = sims.max(axis=1).mean()

            # Combine: 70% exact, 30% semantic
            final_score = (exact_score * 0.7) + (semantic_score * 0.3)
            return min(float(final_score), 1.0)
        except Exception as e:
            # If semantic matching fails (e.g., memory error), disable it and use exact match only
            print(f"Warning: Semantic matching failed ({str(e)}), falling back to exact matching only")
            self.use_semantic = False
            return exact_score

    def score(self, user_skills, job_skills, user_embeddings=None):
        """
        Hybrid scoring: exact match + semantic fallback, for a single job.
        Returns: Score between 0.0 and 1.0

        user_embeddings: optional precomputed embeddings for the user's skills
        (caller may compute this once per search instead of once per job)
        """
        exact_score, unmatched_job_skills = self.prepare(user_skills, job_skills)
        if unmatched_job_skills is None:
            return exact_score
        return self.finish(exact_score, unmatched_job_skills, user_skills, user_embeddings)


class EducationScorer:
    """Education level and major matching"""

    LEVEL_HIERARCHY = {
        'high school': 1,
        'associate': 2,
        "bachelor's degree": 3,
        "master's degree": 4,
        'phd': 5
    }

    def score(self, user_level, user_major, job_level, job_major):
        """
        Education scoring: level (60%) + major (40%)
        Returns: Score between 0.0 and 1.0
        """
        level_score = self._score_level(user_level, job_level)
        major_score = self._score_major(user_major, job_major)

        # If no education required, perfect match
        if not job_level and not job_major:
            return 1.0

        # Weighted combination
        if job_level and job_major:
            return (level_score * 0.6) + (major_score * 0.4)
        elif job_level:
            return level_score
        else:
            return major_score

    def _score_level(self, user_level, job_level):
        """Score education level match"""
        if not job_level:
            return 1.0

        user_rank = self.LEVEL_HIERARCHY.get(user_level.lower() if user_level else '', 0)
        job_rank = self.LEVEL_HIERARCHY.get(job_level.lower(), 0)

        # User meets or exceeds requirement
        if user_rank >= job_rank:
            return 1.0
        # User is one level below
        elif user_rank == job_rank - 1:
            return 0.7
        # Two levels below
        elif user_rank == job_rank - 2:
            return 0.4
        else:
            return 0.0

    def _score_major(self, user_major, job_major):
        """Score education major match"""
        if not job_major:
            return 1.0

        if not user_major:
            return 0.5  # Partial credit

        user_m = user_major.lower()
        job_m = job_major.lower()

        # Exact match
        if user_m == job_m:
            return 1.0

        # Partial match (substring)
        if user_m in job_m or job_m in user_m:
            return 0.8

        # Related fields (heuristic)
        related_groups = [
            {'computer science', 'information technology', 'software engineering'},
            {'engineering', 'mechanical engineering', 'civil engineering'},
            {'business', 'business administration', 'management'},
            {'design', 'graphic design', 'interior design', 'architecture'}
        ]

        for group in related_groups:
            if user_m in group and job_m in group:
                return 0.6

        return 0.0


class ExperienceScorer:
    """Experience years matching"""

    def score(self, user_years, job_min_years):
        """
        Experience scoring
        Returns: Score between 0.0 and 1.0
        """
        if job_min_years == 0:
            return 1.0

        if user_years >= job_min_years:
            return 1.0
        elif user_years >= job_min_years - 1:
            return 0.8  # 1 year short
        elif user_years >= job_min_years - 2:
            return 0.6  # 2 years short
        else:
            return 0.3  # Significantly short


class LanguageScorer:
    """Language requirements matching"""

    LEVEL_HIERARCHY = {
        'basic': 1,
        'good': 2,
        'fluent': 3,
        'native': 4
    }

    def score(self, user_languages, job_languages):
        """
        Language scoring
        Returns: Score between 0.0 and 1.0
        """
        if not job_languages:
            return 1.0

        user_lang_map = {
            lang['name'].lower(): self.LEVEL_HIERARCHY.get(lang['level'].lower(), 0)
            for lang in user_languages
        }

        matches = 0
        for job_lang in job_languages:
            job_name = job_lang['name'].lower()
            job_level = self.LEVEL_HIERARCHY.get(job_lang['level'].lower(), 0)

            user_level = user_lang_map.get(job_name, 0)

            if user_level >= job_level:
                matches += 1
            elif user_level == job_level - 1:
                matches += 0.7  # Partial credit

        return matches / len(job_languages)


class LocationScorer:
    """Location matching"""

    def score(self, user_location, job_location, willing_to_relocate):
        """
        Location scoring
        Returns: Score between 0.0 and 1.0
        """
        if not job_location:
            return 1.0

        if not user_location:
            return 0.5

        # Exact match
        if user_location.lower() == job_location.lower():
            return 1.0

        # Willing to relocate
        if willing_to_relocate:
            return 0.8

        return 0.0
