from django import forms
from .models import UserProfile, UserSkill, UserLanguage


class JobSearchForm(forms.Form):
    """Form for user to input their profile and find matching jobs"""

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

    # Experience
    years_of_experience = forms.IntegerField(
        label='Years of Experience',
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={'placeholder': '0'})
    )

    current_job_title = forms.CharField(
        label='Current Job Title',
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'e.g., Software Developer'})
    )

    # Education
    EDUCATION_LEVEL_CHOICES = [
        ('', 'Select education level'),
        ('high school', 'High School'),
        ('associate', 'Associate Degree'),
        ("bachelor's degree", "Bachelor's Degree"),
        ("master's degree", "Master's Degree"),
        ('phd', 'PhD'),
    ]

    education_level = forms.ChoiceField(
        label='Education Level',
        choices=EDUCATION_LEVEL_CHOICES,
        required=False
    )

    education_major = forms.CharField(
        label='Major / Field of Study',
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'e.g., Computer Science, Engineering'})
    )

    # Languages - will be handled by JavaScript in template
    # We'll process this in the view from POST data

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
        # Languages come from dynamic language_N/proficiency_N inputs, not a
        # declared field, so Django never calls clean_languages() on its own -
        # without this, every search matched as if the user spoke nothing.
        cleaned_data['languages'] = self.clean_languages()
        return cleaned_data

    def clean_languages(self):
        """Parse language input from multiple fields"""
        languages = []
        
        # Get all language and proficiency fields from POST data
        data = self.data
        i = 1
        while f'language_{i}' in data and data.get(f'language_{i}', '').strip():
            language = data.get(f'language_{i}', '').strip()
            proficiency = data.get(f'proficiency_{i}', '').strip()
            
            if language:
                languages.append({
                    'name': language.lower(),
                    'level': proficiency.lower() if proficiency else 'good'
                })
            i += 1
        
        return languages
