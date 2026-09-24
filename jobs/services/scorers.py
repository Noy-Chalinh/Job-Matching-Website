import numpy as np


def soft_threshold(similarity, low, high):
    """Map a cosine similarity onto 0..1 credit: nothing at or below `low`,
    full credit at or above `high`, linear in between.

    Raw BGE similarities can't be used as scores directly: measured with
    bge-small-en-v1.5 on this app's own vocabulary, clearly *unrelated*
    short phrases ("python" vs "accounting", "photoshop" vs "taxation")
    still score ~0.50-0.63, while related ones ("bookkeeping" vs
    "accounting", "django" vs "python") score ~0.60-0.84. Averaging raw
    similarities gave every job the same ~0.55 floor, drowning out the
    signal; thresholding keeps only the part of the range that means
    something.
    """
    return float(np.clip((np.asarray(similarity) - low) / (high - low), 0.0, 1.0))


class SkillScorer:
    """Exact + semantic skill matching over canonical skills
    (see jobs.services.skill_vocab).

    Every job skill earns credit 1.0 if the user has it exactly, otherwise
    partial credit from its closest user skill via soft_threshold (capped at
    RELATED_CREDIT, so having a related skill never counts as much as having
    the skill itself). The score blends:
      - coverage: how much of what the job asks for the user covers, and
      - relevance: how much of what the user offers this job actually uses,
    so a job that happens to list one of a user's ten skills doesn't rank
    alongside a job that needs most of them.

    Scoring a batch of jobs is split so the embedding model is invoked once
    per search instead of once per job:
      1. prepare() - cheap exact-match pass per job, no embedding calls
      2. one embed_batch() call, over every job's unmatched skills combined
      3. finish() - the per-job score, using the now-warm embedding cache
    score() still exists as a single-job convenience wrapper (e.g. for tests).
    """

    SIM_LOW = 0.62
    SIM_HIGH = 0.82
    RELATED_CREDIT = 0.8
    COVERAGE_WEIGHT = 0.7
    RELEVANCE_WEIGHT = 0.3
    # A job skill counts as "missing" for the skill-gap list below this credit.
    MISSING_BELOW = 0.5

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.use_semantic = True  # Flag to disable semantic matching if it fails

    def prepare(self, user_skills, job_skills):
        """Exact-match pass only. Returns (exact_matches, unmatched_job_skills)
        where unmatched_job_skills is the set that still needs semantic
        scoring (empty if none, or if semantic matching is disabled)."""
        user_set = set(user_skills)
        job_set = set(job_skills)
        exact = user_set & job_set
        unmatched = (job_set - exact) if self.use_semantic and user_set else set()
        return exact, unmatched

    def finish(self, exact, unmatched_job_skills, user_skills, job_skills, user_embeddings=None):
        """Returns (score, missing_skills). Cheap as long as embed_batch's
        cache was already warmed for unmatched_job_skills."""
        job_skills = sorted(set(job_skills))
        user_skills = list(dict.fromkeys(user_skills))
        if not job_skills or not user_skills:
            return 0.0, job_skills

        job_credit = {s: (1.0 if s in exact else 0.0) for s in job_skills}
        user_credit = {s: (1.0 if s in exact else 0.0) for s in user_skills}

        if unmatched_job_skills and self.use_semantic:
            try:
                if user_embeddings is None:
                    user_embeddings = self.embedding_service.embed_batch(user_skills)
                unmatched = sorted(unmatched_job_skills)
                job_embeddings = self.embedding_service.embed_batch(unmatched)
                # Embeddings are L2-normalized, so cosine similarity is just
                # the dot product: one matrix multiply for every pair.
                sims = np.asarray(job_embeddings) @ np.asarray(user_embeddings).T
                for i, skill in enumerate(unmatched):
                    job_credit[skill] = self.RELATED_CREDIT * soft_threshold(
                        sims[i].max(), self.SIM_LOW, self.SIM_HIGH
                    )
                for j, skill in enumerate(user_skills):
                    if user_credit[skill] < 1.0:
                        user_credit[skill] = self.RELATED_CREDIT * soft_threshold(
                            sims[:, j].max(), self.SIM_LOW, self.SIM_HIGH
                        )
            except Exception as e:
                # If semantic matching fails (e.g., memory error), disable it and use exact match only
                print(f"Warning: Semantic matching failed ({str(e)}), falling back to exact matching only")
                self.use_semantic = False

        coverage = sum(job_credit.values()) / len(job_credit)
        relevance = sum(user_credit.values()) / len(user_credit)
        score = self.COVERAGE_WEIGHT * coverage + self.RELEVANCE_WEIGHT * relevance
        missing = [s for s in job_skills if job_credit[s] < self.MISSING_BELOW]
        return min(float(score), 1.0), missing

    def score(self, user_skills, job_skills, user_embeddings=None):
        """Single-job convenience wrapper. Returns a score between 0.0 and 1.0."""
        exact, unmatched = self.prepare(user_skills, job_skills)
        score, _ = self.finish(exact, unmatched, user_skills, job_skills, user_embeddings)
        return score


class TitleScorer:
    """Similarity between the user's current job title and a job's title.

    Titles are the most direct relevance signal this job board has ("Senior
    Accountant" vs "Accountant"), but before this the user's title only
    reached scoring indirectly, blended into the full-text job_semantic query.
    Thresholds measured the same way as SkillScorer's: related titles
    ("accountant"/"chief accountant", "hr officer"/"recruitment officer")
    score ~0.66-0.84, unrelated ones ("cashier"/"civil engineer") ~0.52-0.60.
    """

    SIM_LOW = 0.60
    SIM_HIGH = 0.84

    def score(self, user_title_embedding, job_title_embedding):
        return soft_threshold(
            float(np.dot(user_title_embedding, job_title_embedding)), self.SIM_LOW, self.SIM_HIGH
        )


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

        # Related fields: same group in data/resources/major_taxonomy.json
        # (e.g. "management" and "business administration", the two most
        # common majors on this job board, used to score 0.0 against each
        # other).
        user_groups = _major_groups(user_m)
        if user_groups and user_groups & _major_groups(job_m):
            return 0.6

        # "any field" / "related field" style requirements.
        if _major_groups(job_m) & {'other:any'}:
            return 0.8

        return 0.0


_MAJOR_GROUPS = None
_ANY_MAJOR = {'any field', 'related field', 'any discipline', 'related discipline'}


def _major_groups(major):
    """Taxonomy groups a (lowercased) major belongs to, matching either the
    exact entry or an entry contained in it ("bachelor of accounting" ->
    business)."""
    global _MAJOR_GROUPS
    if _MAJOR_GROUPS is None:
        import json
        from django.conf import settings
        path = settings.BASE_DIR / 'data' / 'resources' / 'major_taxonomy.json'
        with open(path, encoding='utf-8') as f:
            _MAJOR_GROUPS = json.load(f)
    groups = set()
    for group, majors in _MAJOR_GROUPS.items():
        for m in majors:
            if m in _ANY_MAJOR:
                if m in major:
                    groups.add('other:any')
            elif m == major or (len(m) > 3 and m in major):
                groups.add(group)
    return groups


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
