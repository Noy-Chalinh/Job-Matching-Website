from django.urls import path
from . import views

app_name = 'jobs'

urlpatterns = [
    path('', views.search, name='search'),
    path('results/', views.search_results, name='search_results'),
    path('healthz', views.healthz, name='healthz'),
]