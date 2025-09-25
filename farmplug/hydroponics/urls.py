# hydroponics/urls.py
from django.urls import path
from .views import hydroponics_view

urlpatterns = [
    path('hydroponics/', hydroponics_view, name='hydroponics'),
]