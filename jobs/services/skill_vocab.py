"""Map noisy extracted skill phrases onto a curated canonical vocabulary.

Job.skills as produced by the extraction pipeline (KeyBERT + NER + regex) is
mostly noise - profiled on the live table: phrases like "public holidays",
"st. 271", "how to apply", Khmer text split into fragments, and 172 jobs
tagged "computer vision" because the regex matcher treated "CV" (resume) as
an alias for it. Scoring against that noise is what made match scores both
low and arbitrary: every junk phrase inflated a job's skill denominator, and
random junk occasionally "matched" a user's skill semantically.

Instead of trying to blocklist the noise, this keeps only what maps onto a
known skill (data/resources/skill_vocabulary.json, with technical_skills.json
and skill_synonyms.json merged underneath), and canonicalizes spelling
variants on both the job and the user side so "Excel", "microsoft excel" and
"excel & power point" all compare equal to "ms excel".

Runs at request time on the stored phrases, so it needs no DB migration or
re-extraction - it's a handful of regex scans per candidate job.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from django.conf import settings

from .embeddings import normalize_skill

RESOURCES_DIR = Path(settings.BASE_DIR) / 'data' / 'resources'

KHMER_CHARS = re.compile(r'[ក-៿᧠-᧿]+')

# Categories of technical_skills.json that aren't skills for matching:
# languages are scored separately by LanguageScorer from Job.languages.
SKIPPED_TECHNICAL_CATEGORIES = {'languages_spoken'}

# Terms from technical_skills.json that are too ambiguous to trust in
# scraped job text at all ("cv" is how "computer vision" got onto 172 jobs;
# "soap"/"zoom"/"spring"/... are far more often ordinary words).
IGNORED_TERMS = {
    'cv', 'soap', 'zoom', 'slack', 'echo', 'gin', 'express', 'spring', 'publisher',
    'inventor', 'backbone', 'ember', 'apache', 'hive', 'chai', 'mocha', 'cucumber',
    'koa', 'less', 'rails', 'xd', 'ts', 'vb', 'r', 'go', 'c', 'project', 'email',
    'telegram', 'insurance', 'nssf', 'production', 'access',
}

# Degree names are education requirements (scored by EducationScorer), not
# skills - without this, "bachelor of business administration" would count
# as the skill "administration" on 127 jobs.
NON_SKILL_PHRASES = re.compile(
    r"\b(?:business administration|business management|bachelor\w*|master\w*|degree|diploma)\b"
)

# Canonical skills that historically come from a known false-positive and
# only count when the job title backs them up.
TITLE_GATED = {
    'computer vision': re.compile(r'\b(ai|vision|machine learning|ml|data scien\w*|deep learning)\b'),
}


# "word"/"excel"/"outlook"/"office" are too ambiguous to trust inside an
# arbitrary phrase, but a phrase naming two or more of them ("excel & power
# point", "word excel powerpoint" - 50+ jobs each) is unmistakably about
# MS Office, so in that context they count.
OFFICE_CANONICALS = {'ms office', 'ms word', 'ms excel', 'ms powerpoint', 'ms outlook'}
OFFICE_TERMS = re.compile(r'\b(?:ms|microsoft|words?|excels?|power ?points?|outlook|office)\b')


def _alternation(aliases):
    # Longest first so "microsoft office word" wins over "microsoft office";
    # lookarounds instead of \b so aliases like "c++", "c#", ".net", "f&b"
    # still get whole-word boundaries.
    ordered = sorted(aliases, key=len, reverse=True)
    body = '|'.join(re.escape(a) for a in ordered)
    return re.compile(rf'(?<![\w.+#&])(?:{body})(?![\w+#&])')


class SkillVocabulary:
    def __init__(self):
        self.alias_to_canonical = {}
        self._load()
        ambiguous, generic = self._ambiguous, self._generic
        phrase_aliases = [a for a in self.alias_to_canonical if a not in ambiguous and a not in generic]
        title_aliases = [a for a in self.alias_to_canonical if a not in ambiguous]
        self._phrase_re = _alternation(phrase_aliases)
        self._title_re = _alternation(title_aliases)
        self._office_re = _alternation(
            [a for a, c in self.alias_to_canonical.items() if c in OFFICE_CANONICALS]
        )

    def _add(self, alias, canonical):
        alias = normalize_skill(alias)
        if alias and alias not in IGNORED_TERMS:
            self.alias_to_canonical[alias] = canonical

    def _load(self):
        with open(RESOURCES_DIR / 'technical_skills.json', encoding='utf-8') as f:
            technical = json.load(f)
        with open(RESOURCES_DIR / 'skill_synonyms.json', encoding='utf-8') as f:
            synonyms = json.load(f)
        with open(RESOURCES_DIR / 'skill_vocabulary.json', encoding='utf-8') as f:
            vocab = json.load(f)

        # Lowest precedence first; later layers overwrite earlier mappings.
        for category, terms in technical.items():
            if category in SKIPPED_TECHNICAL_CATEGORIES:
                continue
            for term in terms:
                self._add(term, normalize_skill(term))
        for canonical, aliases in synonyms.items():
            canonical = normalize_skill(canonical)
            self._add(canonical, canonical)
            for alias in aliases:
                self._add(alias, canonical)

        self._ambiguous = {normalize_skill(a) for a in vocab.get('_ambiguous', [])}
        self._generic = {normalize_skill(a) for a in vocab.get('_generic', [])}
        entries = {k: v for k, v in vocab.items() if not k.startswith('_')}
        for canonical, aliases in vocab.get('_title_aliases', {}).items():
            entries.setdefault(canonical, [])
            entries[canonical] = list(entries[canonical]) + list(aliases)
        for canonical, aliases in entries.items():
            canonical = normalize_skill(canonical)
            self._add(canonical, canonical)
            for alias in aliases:
                self._add(alias, canonical)

    @staticmethod
    def _clean(text):
        text = KHMER_CHARS.sub(' ', (text or '').lower())
        return normalize_skill(NON_SKILL_PHRASES.sub(' ', text))

    def _from_phrase(self, phrase, allow_generic):
        """Canonical skills found in one phrase. The whole phrase may be any
        alias; inside a longer phrase only unambiguous ones count."""
        phrase = self._clean(phrase)
        if not phrase:
            return set()
        if phrase in self.alias_to_canonical:
            return {self.alias_to_canonical[phrase]}
        pattern = self._title_re if allow_generic else self._phrase_re
        found = {self.alias_to_canonical[m.group(0)] for m in pattern.finditer(phrase)}
        if len(set(OFFICE_TERMS.findall(phrase))) >= 2:
            found |= {self.alias_to_canonical[m.group(0)] for m in self._office_re.finditer(phrase)}
        return found

    def job_skills(self, raw_skills, job_title=''):
        """Canonical skills for a job, from its stored (noisy) skill phrases
        plus its title (which is usually the single best skill signal on
        this job board, e.g. "Accountant", "Sales Executive")."""
        found = set()
        for phrase in raw_skills or []:
            found |= self._from_phrase(phrase, allow_generic=False)
        title = self._clean(job_title)
        if title:
            found |= self._from_phrase(title, allow_generic=True)
        for skill, evidence in TITLE_GATED.items():
            if skill in found and not evidence.search(title):
                found.discard(skill)
        return sorted(found)

    def user_skills(self, raw_skills):
        """Canonicalize what the user typed. Anything that isn't in the
        vocabulary is kept verbatim (normalized) so semantic matching can
        still relate it to job skills - a user's input is intentional, so
        unlike job text it isn't discarded just for being unknown."""
        result = []
        for phrase in raw_skills or []:
            hits = self._from_phrase(phrase, allow_generic=True)
            if hits:
                result.extend(sorted(hits))
            else:
                cleaned = self._clean(phrase)
                if cleaned:
                    result.append(cleaned)
        # De-duplicate, keeping the user's order.
        return list(dict.fromkeys(result))


@lru_cache(maxsize=1)
def get_vocabulary():
    return SkillVocabulary()
