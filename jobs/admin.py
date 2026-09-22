from django.contrib import admin

from .models import Job, RawJob


@admin.register(RawJob)
class RawJobAdmin(admin.ModelAdmin):
    list_display = ('job_id', 'source', 'page', 'scraped_at')
    list_filter = ('source',)
    search_fields = ('job_id',)
    ordering = ('-scraped_at',)


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = (
        'job_id', 'job_title', 'company', 'location', 'source',
        'min_years_experience', 'education_level', 'pubdate',
    )
    list_filter = ('source', 'education_level', 'location')
    search_fields = ('job_id', 'job_title', 'company')
    ordering = ('-pubdate',)
