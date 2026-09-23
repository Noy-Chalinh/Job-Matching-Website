from django.test import TestCase

from jobs.models import Job, UserLanguage, UserProfile, UserSkill
from jobs.services.matcher import JobMatcher


class _StubManager:
    """Mimics Django's related-manager .all() interface for the plain
    UserProfile/UserSkill/UserLanguage classes used during matching, the
    same way jobs.views.search_results builds a temporary profile."""

    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


def _make_profile(skills=None, languages=None, **kwargs):
    profile = UserProfile(**kwargs)
    profile.skills = _StubManager(skills or [])
    profile.languages = _StubManager(languages or [])
    return profile


class MatchScoreWeightingTests(TestCase):
    """Regression tests for the match_score weight renormalization: a job
    posting with no data for a category (e.g. empty skills) must not be
    scored as a perfect match on that category, since that made every
    sparse posting outrank real matches regardless of search input."""

    def setUp(self):
        self.matcher = JobMatcher()
        self.profile = _make_profile(
            years_of_experience=3,
            education_level="bachelor's degree",
            education_major='computer science',
            preferred_location='phnom penh',
            willing_to_relocate=False,
            skills=[UserSkill('python'), UserSkill('django')],
            languages=[UserLanguage('english', 'fluent')],
        )

    def test_fully_populated_job_keeps_full_weights(self):
        Job.objects.create(
            job_id='full-1', job_title='Backend Dev', location='phnom penh',
            min_years_experience=2, education_level="bachelor's degree",
            education_major='computer science',
            skills=['python', 'django'], languages=[{'name': 'english', 'level': 'good'}],
        )
        [match] = self.matcher.match(self.profile, top_n=1)
        expected = (
            match['skill_score'] * 0.60 + match['education_score'] * 0.20 +
            match['experience_score'] * 0.15 + match['language_score'] * 0.03 +
            match['location_score'] * 0.02
        )
        self.assertAlmostEqual(match['match_score'], expected, places=6)

    def test_only_location_populated_forces_zero_skill_credit(self):
        """A job with no listed skills must not be able to reach a high
        match_score purely via an unrelated category (e.g. exact location) -
        skill_score is forced to 0 and always counted at its full 60%
        weight, so even an exact location match can't push this above ~3%."""
        Job.objects.create(
            job_id='empty-1', job_title='Personal Driver', location='phnom penh',
            min_years_experience=0, education_level='', education_major='',
            skills=[], languages=[],
        )
        [match] = self.matcher.match(self.profile, top_n=1)
        self.assertEqual(match['skill_score'], 0.0)
        total_weight = 0.60 + 0.02  # skill (forced) + location only
        expected = (0.0 * 0.60 + match['location_score'] * 0.02) / total_weight
        self.assertAlmostEqual(match['match_score'], expected, places=6)
        self.assertLess(match['match_score'], 0.05)

    def test_partial_data_renormalizes_across_present_categories_only(self):
        Job.objects.create(
            job_id='partial-1', job_title='Junior Dev', location='phnom penh',
            min_years_experience=0, education_level='', education_major='',
            skills=['python', 'django'], languages=[],
        )
        [match] = self.matcher.match(self.profile, top_n=1)
        total_weight = 0.60 + 0.02  # skill + location only
        expected = (match['skill_score'] * 0.60 + match['location_score'] * 0.02) / total_weight
        self.assertAlmostEqual(match['match_score'], expected, places=6)

    def test_real_skill_overlap_outranks_empty_skill_location_match(self):
        """Regression test for the live bug: a job with no listed skills
        used to be able to tie/beat a job with genuine skill overlap just by
        matching the user's location exactly, since both would otherwise
        renormalize to ~100%. It must not, once skill is always weighted."""
        Job.objects.create(
            job_id='dev-1', job_title='Backend Dev', location='phnom penh',
            min_years_experience=2, education_level="bachelor's degree",
            education_major='computer science',
            skills=['python', 'django'], languages=[{'name': 'english', 'level': 'good'}],
        )
        Job.objects.create(
            job_id='driver-1', job_title='Personal Driver', location='phnom penh',
            min_years_experience=0, education_level='', education_major='',
            skills=[], languages=[],
        )
        matches = {m['job']['job_id']: m for m in self.matcher.match(self.profile, top_n=2)}
        self.assertGreater(matches['dev-1']['match_score'], matches['driver-1']['match_score'])
