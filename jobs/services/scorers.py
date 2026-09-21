import numpy as np


class SkillScorer:
    """Hybrid exact + semantic skill matching"""

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.use_semantic = True  # Flag to disable semantic matching if it fails

    def score(self, user_skills, job_skills, user_embeddings=None):
        """
        Hybrid scoring: exact match + semantic fallback
        Returns: Score between 0.0 and 1.0

        user_embeddings: optional precomputed embeddings for the user's skills
        (caller may compute this once per search instead of once per job)
        """
        if not job_skills:
            return 1.0  # No requirements = perfect match

        # Step 1: Exact matching
        user_set = set(s.lower() for s in user_skills)
        job_set = set(s.lower() for s in job_skills)

        exact_matches = user_set & job_set
        exact_score = len(exact_matches) / len(job_set)

        # Step 2: Semantic matching (only if exact match is poor and semantic is enabled)
        if exact_score >= 0.7 or not self.use_semantic:
            return exact_score

        # Compute semantic similarity for unmatched skills
        unmatched_job_skills = job_set - exact_matches

        if not unmatched_job_skills:
            return exact_score

        try:
            # Embed user skills (unless already precomputed by the caller) and unmatched job skills
            if user_embeddings is None:
                user_embeddings = self.embedding_service.embed_batch(list(user_set))
            job_embeddings = self.embedding_service.embed_batch(list(unmatched_job_skills))

            # Compute similarity matrix
            similarities = []
            for job_emb in job_embeddings:
                max_sim = max(
                    self.embedding_service.cosine_similarity(job_emb, user_emb)
                    for user_emb in user_embeddings
                )
                similarities.append(max_sim)

            semantic_score = np.mean(similarities)

            # Combine: 70% exact, 30% semantic
            final_score = (exact_score * 0.7) + (semantic_score * 0.3)

            return min(final_score, 1.0)
        except Exception as e:
            # If semantic matching fails (e.g., memory error), disable it and use exact match only
            print(f"Warning: Semantic matching failed ({str(e)}), falling back to exact matching only")
            self.use_semantic = False
            return exact_score


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
