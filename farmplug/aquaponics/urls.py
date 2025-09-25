# aquaponics/urls.py
from django.urls import path
from .views import aquaponics_view

urlpatterns = [
    path('aquaponics/', aquaponics_view, name='aquaponics'),
]