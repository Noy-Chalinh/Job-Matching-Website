import re

from django import forms


def _repeatable_rows(data, *fields):
    """Rows of a repeatable form section whose inputs are named
    '<field>_<N>' (added/removed in the browser by search.html's JS), in N
    order. N can have gaps - removing an entry leaves a hole in the
    numbering - so every index present is collected rather than counting up
    from 1 and stopping at the first missing one.
    """
    pattern = re.compile(r'^(?:%s)_(\d+)$' % '|'.join(map(re.escape, fields)))
    indices = sorted({int(m.group(1)) for key in data for m in [pattern.match(key)] if m})
    return [{f: (data.get(f'{f}_{i}') or '').strip() for f in fields} for i in indices]


class JobSearchForm(forms.Form):
    """Form for user to input their profile and find matching jobs.

    Experience, education and languages are repeatable sections rendered and
    managed by search.html (inputs named e.g. experience_title_1,
    experience_years_1, experience_title_3, ...), so they aren't declared as
    fields here - clean() parses them into cleaned_data['experiences'],
    ['educations'] and ['languages'].
    """

    # Skills (comma-separated)
    skills = forms.CharField(
        label='Your Skills',
        widget=forms.Textarea(attrs={
            'rows': 3,
            'placeholder': 'Enter your skills separated by commas (e.g., Python, Django, React, MS Office)'
        }),
        help_text='Separate multiple skills with commas',
        required=False
    )

    EDUCATION_LEVEL_CHOICES = [
        ('', 'Select education level'),
        ('high school', 'High School'),
        ('associate', 'Associate Degree'),
        ("bachelor's degree", "Bachelor's Degree"),
        ("master's degree", "Master's Degree"),
        ('phd', 'PhD'),
    ]

    PROFICIENCY_CHOICES = [
        ('', 'Select level'),
        ('basic', 'Basic'),
        ('good', 'Good'),
        ('fluent', 'Fluent'),
        ('native', 'Native'),
    ]

    MAX_YEARS_PER_ROLE = 60

    # Location
    preferred_location = forms.CharField(
        label='Preferred Location',
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'e.g., Phnom Penh'})
    )

    willing_to_relocate = forms.BooleanField(
        label='Willing to Relocate',
        required=False,
        initial=False
    )

    def clean_skills(self):
        """Parse comma-separated skills"""
        skills_text = self.cleaned_data.get('skills', '')
        if not skills_text:
            return []

        # Split by comma and clean up
        skills = [s.strip() for s in skills_text.split(',') if s.strip()]
        return skills

    def clean(self):
        cleaned_data = super().clean()
        # These come from dynamic inputs, not declared fields, so Django
        # never calls their clean_*() methods on its own.
        for name, parse in (
            ('experiences', self.clean_experiences),
            ('educations', self.clean_educations),
            ('languages', self.clean_languages),
        ):
            try:
                cleaned_data[name] = parse()
            except forms.ValidationError as e:
                self.add_error(None, e)
        return cleaned_data

    def clean_experiences(self):
        """[{'title': str, 'years': float}], one per filled-in experience row."""
        experiences = []
        for row in _repeatable_rows(self.data, 'experience_title', 'experience_years'):
            title, years_text = row['experience_title'][:200], row['experience_years']
            if not title and not years_text:
                continue
            try:
                years = float(years_text) if years_text else 0.0
            except ValueError:
                raise forms.ValidationError(f'"{years_text}" is not a valid number of years.')
            if not 0 <= years <= self.MAX_YEARS_PER_ROLE:
                raise forms.ValidationError(
                    f'Years of experience must be between 0 and {self.MAX_YEARS_PER_ROLE}.'
                )
            experiences.append({'title': title, 'years': years})
        return experiences

    def clean_educations(self):
        """[{'level': str, 'major': str}], one per filled-in education row."""
        valid_levels = {value for value, _ in self.EDUCATION_LEVEL_CHOICES}
        educations = []
        for row in _repeatable_rows(self.data, 'education_level', 'education_major'):
            level, major = row['education_level'].lower(), row['education_major'][:100]
            if not level and not major:
                continue
            if level not in valid_levels:
                raise forms.ValidationError('Please choose an education level from the list.')
            educations.append({'level': level, 'major': major})
        return educations

    def clean_languages(self):
        """[{'name': str, 'level': str}], one per row with a language name."""
        valid_levels = {value for value, _ in self.PROFICIENCY_CHOICES if value}
        languages = []
        for row in _repeatable_rows(self.data, 'language', 'proficiency'):
            if not row['language']:
                continue
            level = row['proficiency'].lower()
            languages.append({
                'name': row['language'].lower(),
                'level': level if level in valid_levels else 'good'
            })
        return languages
