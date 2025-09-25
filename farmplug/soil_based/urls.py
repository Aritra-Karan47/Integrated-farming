# soil_based/urls.py
from django.urls import path
from .views import soil_based_view

urlpatterns = [
    path('soilbased/', soil_based_view, name='soil_based'),
]