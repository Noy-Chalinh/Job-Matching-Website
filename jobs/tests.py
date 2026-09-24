from django.test import TestCase

from jobs.models import Job, UserLanguage, UserProfile, UserSkill
from jobs.services.matcher import CATEGORY_WEIGHTS, JobMatcher


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
        # No JobEmbedding row exists for this job in these tests, so
        # job_semantic has no data and is excluded from the weighted
        # average regardless of backend - same renormalization rule as
        # every other category here. Likewise title: this profile has no
        # current_job_title to compare.
        total_weight = (
            sum(CATEGORY_WEIGHTS.values()) - CATEGORY_WEIGHTS['job_semantic'] - CATEGORY_WEIGHTS['title']
        )
        expected = (
            match['skill_score'] * CATEGORY_WEIGHTS['skill'] +
            match['education_score'] * CATEGORY_WEIGHTS['education'] +
            match['experience_score'] * CATEGORY_WEIGHTS['experience'] +
            match['language_score'] * CATEGORY_WEIGHTS['language'] +
            match['location_score'] * CATEGORY_WEIGHTS['location']
        ) / total_weight
        self.assertAlmostEqual(match['match_score'], expected, places=6)

    def test_only_location_populated_forces_zero_skill_credit(self):
        """A job with no listed skills must not be able to reach a high
        match_score purely via an unrelated category (e.g. exact location) -
        skill_score is forced to 0 and always counted at its full weight
        (see CATEGORY_WEIGHTS), so even an exact location match can't push
        this above ~5%."""
        Job.objects.create(
            job_id='empty-1', job_title='Personal Driver', location='phnom penh',
            min_years_experience=0, education_level='', education_major='',
            skills=[], languages=[],
        )
        [match] = self.matcher.match(self.profile, top_n=1)
        self.assertEqual(match['skill_score'], 0.0)
        total_weight = CATEGORY_WEIGHTS['skill'] + CATEGORY_WEIGHTS['location']  # skill (forced) + location only
        expected = (0.0 * CATEGORY_WEIGHTS['skill'] + match['location_score'] * CATEGORY_WEIGHTS['location']) / total_weight
        self.assertAlmostEqual(match['match_score'], expected, places=6)
        self.assertLess(match['match_score'], 0.05)

    def test_partial_data_renormalizes_across_present_categories_only(self):
        Job.objects.create(
            job_id='partial-1', job_title='Junior Dev', location='phnom penh',
            min_years_experience=0, education_level='', education_major='',
            skills=['python', 'django'], languages=[],
        )
        [match] = self.matcher.match(self.profile, top_n=1)
        total_weight = CATEGORY_WEIGHTS['skill'] + CATEGORY_WEIGHTS['location']  # skill + location only
        expected = (
            match['skill_score'] * CATEGORY_WEIGHTS['skill'] +
            match['location_score'] * CATEGORY_WEIGHTS['location']
        ) / total_weight
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


class SkillVocabularyTests(TestCase):
    """Job.skills from the extraction pipeline is mostly noise; the
    vocabulary must keep real skills, drop the rest, and canonicalize
    spelling variants on both sides."""

    def setUp(self):
        from jobs.services.skill_vocab import get_vocabulary
        self.vocab = get_vocabulary()

    def test_noise_is_dropped_and_variants_canonicalized(self):
        skills = self.vocab.job_skills(
            ['public holidays', 'st. 271', 'how to apply', 'excel & power point',
             'proficiency with micrsoft word', 'annual leave 18 days', 'negotiable'],
            'Admin Assistant',
        )
        self.assertEqual(skills, ['administration', 'ms excel', 'ms powerpoint', 'ms word'])

    def test_cv_false_positive_needs_title_evidence(self):
        self.assertNotIn('computer vision', self.vocab.job_skills(['computer vision'], 'Receptionist'))
        self.assertIn('computer vision', self.vocab.job_skills(['computer vision'], 'AI Engineer'))

    def test_degree_is_not_a_skill(self):
        self.assertEqual(self.vocab.job_skills(['business administration', 'bachelor'], ''), [])

    def test_khmer_fragments_are_stripped(self):
        self.assertEqual(self.vocab.job_skills(['microsoft office នទ', 'កកន ឬខ បញ'], ''), ['ms office'])

    def test_title_supplies_generic_skills(self):
        self.assertIn('sales', self.vocab.job_skills([], 'Sales Executive'))
        self.assertIn('accounting', self.vocab.job_skills([], 'Senior Accountant'))

    def test_user_skills_keep_unknown_terms(self):
        self.assertEqual(
            self.vocab.user_skills(['Excel', 'JS', 'Microsoft Word', 'underwater welding']),
            ['ms excel', 'javascript', 'ms word', 'underwater welding'],
        )


class SkillScorerTests(TestCase):
    def setUp(self):
        from jobs.services.embeddings import EmbeddingService
        from jobs.services.scorers import SkillScorer
        self.scorer = SkillScorer(EmbeddingService())

    def test_exact_full_match_is_perfect(self):
        self.assertAlmostEqual(self.scorer.score(['python', 'django'], ['python', 'django']), 1.0)

    def test_unrelated_skills_get_no_semantic_credit(self):
        """The old scorer gave ~0.17 to any job via raw similarity averaging."""
        self.assertLess(self.scorer.score(['python', 'django'], ['accounting', 'taxation']), 0.05)

    def test_related_skills_beat_unrelated(self):
        related = self.scorer.score(['bookkeeping'], ['accounting'])
        unrelated = self.scorer.score(['bookkeeping'], ['autocad'])
        self.assertGreater(related, unrelated)
        self.assertLess(related, 1.0)


class RepeatableSectionFormTests(TestCase):
    """Experience, education and languages are repeatable rows added and
    removed in the browser, so their numbering can have gaps."""

    def _form(self, **data):
        from jobs.forms import JobSearchForm
        form = JobSearchForm(data)
        self.assertTrue(form.is_valid(), form.errors)
        return form.cleaned_data

    def test_rows_after_a_removed_row_are_kept(self):
        data = self._form(
            experience_title_1='Accountant', experience_years_1='2',
            experience_title_3='Auditor', experience_years_3='1.5',
            language_1='English', proficiency_1='fluent',
            language_3='Khmer', proficiency_3='native',
        )
        self.assertEqual(data['experiences'], [
            {'title': 'Accountant', 'years': 2.0}, {'title': 'Auditor', 'years': 1.5},
        ])
        self.assertEqual([l['name'] for l in data['languages']], ['english', 'khmer'])

    def test_empty_rows_are_ignored(self):
        data = self._form(
            experience_title_1='', experience_years_1='',
            education_level_1='', education_major_1='',
            education_level_2="master's degree", education_major_2='Finance',
        )
        self.assertEqual(data['experiences'], [])
        self.assertEqual(data['educations'], [{'level': "master's degree", 'major': 'Finance'}])

    def test_invalid_years_are_rejected(self):
        from jobs.forms import JobSearchForm
        self.assertFalse(JobSearchForm({'experience_title_1': 'Dev', 'experience_years_1': '-3'}).is_valid())
        self.assertFalse(JobSearchForm({'experience_title_1': 'Dev', 'experience_years_1': 'abc'}).is_valid())

    def test_unknown_education_level_is_rejected(self):
        from jobs.forms import JobSearchForm
        self.assertFalse(JobSearchForm({'education_level_1': 'wizard'}).is_valid())


class MultipleEntryMatchingTests(TestCase):
    def setUp(self):
        self.matcher = JobMatcher()
        Job.objects.create(
            job_id='acct-1', job_title='Accountant', location='phnom penh',
            min_years_experience=3, education_level="bachelor's degree",
            education_major='accounting', skills=['accounting'], languages=[],
        )

    def test_years_are_summed_across_roles(self):
        profile = _make_profile(experiences=[
            {'title': 'Junior Accountant', 'years': 1.5}, {'title': 'Bookkeeper', 'years': 2},
        ])
        self.assertEqual(profile.years_of_experience, 3.5)
        [match] = self.matcher.match(profile, top_n=1)
        self.assertEqual(match['experience_score'], 1.0)

    def test_best_degree_is_used(self):
        profile = _make_profile(educations=[
            {'level': "master's degree", 'major': 'marketing'},
            {'level': "bachelor's degree", 'major': 'accounting'},
        ])
        self.assertEqual(profile.education_level, "master's degree")
        [match] = self.matcher.match(profile, top_n=1)
        self.assertEqual(match['education_score'], 1.0)

    def test_closest_past_title_is_used(self):
        one_title = _make_profile(experiences=[{'title': 'Graphic Designer', 'years': 2}])
        two_titles = _make_profile(experiences=[
            {'title': 'Graphic Designer', 'years': 2}, {'title': 'Accountant', 'years': 1},
        ])
        [only_designer] = self.matcher.match(one_title, top_n=1)
        [with_accountant] = self.matcher.match(two_titles, top_n=1)
        self.assertGreater(with_accountant['title_score'], only_designer['title_score'])
        self.assertAlmostEqual(with_accountant['title_score'], 1.0)
