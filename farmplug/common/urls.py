# common/urls.py
from django.urls import path
from .views import profile, CustomLoginView

urlpatterns = [
    path('login/', CustomLoginView.as_view(), name='login'),
    path('profile/', profile, name='profile'),
]