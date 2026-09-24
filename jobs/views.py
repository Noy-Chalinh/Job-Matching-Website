from django.core.cache import cache
from django.shortcuts import render
from django.http import HttpResponse
from .forms import JobSearchForm
from .services.matcher import JobMatcher
from .models import UserProfile, UserSkill, UserLanguage, Job
import logging
import traceback

logger = logging.getLogger(__name__)


def healthz(request):
    """Liveness check for Render's health checks and uptime pingers.
    Deliberately touches neither the database nor the embedding model, so
    pinging it every few minutes (to stop the free instance from spinning
    down) costs next to nothing."""
    return HttpResponse("ok", content_type="text/plain")


def get_job_count():
    """Job count for the search page, cached so each page view isn't a
    COUNT(*) round trip to the database. Jobs only change when a scrape is
    loaded, so a few minutes of staleness is harmless."""
    job_count = cache.get('job_count')
    if job_count is None:
        job_count = Job.objects.count()
        cache.set('job_count', job_count, 600)
    return job_count


def search(request):
    """Display job search form"""
    try:
        job_count = get_job_count()

        form = JobSearchForm()
        return render(request, 'search.html', {
            'form': form,
            'job_count': job_count
        })
    except Exception as e:
        logger.error(f"Error in search view: {str(e)}")
        logger.error(traceback.format_exc())
        return HttpResponse(f"Error loading search page: {str(e)}", status=500)


def search_results(request):
    """Process search and display matched jobs"""
    try:
        if request.method != 'POST':
            # If not POST, redirect to search
            form = JobSearchForm()
            return render(request, 'search.html', {'form': form})

        # Check if there are any jobs in database
        if get_job_count() == 0:
            return render(request, 'results.html', {
                'results': [],
                'total_matches': 0,
                'error': 'No jobs available in the database. Please contact the administrator.'
            })

        form = JobSearchForm(request.POST)

        if form.is_valid():
            logger.info("Form is valid, processing search...")
            
            # Log form data for debugging
            logger.info(f"Skills from form: {form.cleaned_data.get('skills', [])}")
            logger.info(f"Languages from form: {form.cleaned_data.get('languages', [])}")
            logger.info(f"Experience: {form.cleaned_data.get('years_of_experience', 0)} years")
            logger.info(f"Location: {form.cleaned_data.get('preferred_location', '')}, willing to relocate: {form.cleaned_data.get('willing_to_relocate', False)}")
            
            # Create temporary user profile (not saved to database)
            temp_profile = UserProfile(
                years_of_experience=form.cleaned_data.get('years_of_experience', 0),
                current_job_title=form.cleaned_data.get('current_job_title', ''),
                education_level=form.cleaned_data.get('education_level', ''),
                education_major=form.cleaned_data.get('education_major', ''),
                preferred_location=form.cleaned_data.get('preferred_location', ''),
                willing_to_relocate=form.cleaned_data.get('willing_to_relocate', False)
            )

            # Create temporary skill and language objects
            temp_skills = [
                UserSkill(skill_name=skill)
                for skill in form.cleaned_data.get('skills', [])
            ]

            temp_languages = [
                UserLanguage(
                    language_name=lang['name'],
                    proficiency=lang['level']
                )
                for lang in form.cleaned_data.get('languages', [])
            ]

            # Create a simple wrapper class to mimic Django's RelatedManager
            class TempRelatedManager:
                def __init__(self, items):
                    self._items = items

                def all(self):
                    return self._items

            # Add temp managers to profile
            temp_profile.skills = TempRelatedManager(temp_skills)
            temp_profile.languages = TempRelatedManager(temp_languages)

            # Initialize JobMatcher and find matches (top 5 results)
            # Database-only operation
            logger.info("Initializing JobMatcher...")
            try:
                matcher = JobMatcher()
                logger.info("JobMatcher initialized, starting matching process...")
                matches = matcher.match(temp_profile, top_n=5)
                logger.info(f"Matching completed. Found {len(matches)} matches.")
            except Exception as match_error:
                logger.error(f"Error during matching: {str(match_error)}")
                logger.error(traceback.format_exc())
                return render(request, 'results.html', {
                    'results': [],
                    'total_matches': 0,
                    'error': f'Error during job matching: {str(match_error)}'
                })

            # Format results for template
            results = []
            for match in matches:
                job = match['job']
                results.append({
                    'job_id': job['job_id'],
                    'job_title': job['job_title'],
                    'company': job.get('company', 'N/A'),
                    'location': job.get('location', 'N/A'),
                    'match_score': round(match['match_score'] * 100, 1),  # Convert to percentage
                    'skill_score': round(match['skill_score'] * 100, 1),
                    'title_score': None if match['title_score'] is None else round(match['title_score'] * 100, 1),
                    'education_score': round(match['education_score'] * 100, 1),
                    'experience_score': round(match['experience_score'] * 100, 1),
                    'language_score': round(match['language_score'] * 100, 1),
                    'location_score': round(match['location_score'] * 100, 1),
                    'missing_skills': match['missing_skills'],
                    'required_experience': job['experience']['min_years'],
                    'education_level': job['education'].get('level', 'N/A'),
                    'education_major': job['education'].get('major', 'N/A'),
                })

            return render(request, 'results.html', {
                'results': results,
                'total_matches': len(results)
            })
        else:
            # Form is invalid
            logger.warning(f"Form validation failed: {form.errors}")
            return render(request, 'search.html', {
                'form': form,
                'error': 'Please correct the errors in the form.'
            })

    except Exception as e:
        logger.error(f"Unexpected error in search_results: {str(e)}")
        logger.error(traceback.format_exc())
        return HttpResponse(f"Error processing search: {str(e)}<br><br>Traceback:<br><pre>{traceback.format_exc()}</pre>", status=500)