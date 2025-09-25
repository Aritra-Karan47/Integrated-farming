# vertical_farming/urls.py
from django.urls import path
from .views import vertical_farming_view

urlpatterns = [
    path('verticalfarming/', vertical_farming_view, name='vertical_farming'),
]