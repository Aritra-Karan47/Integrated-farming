# hydroponics/views.py
from django.shortcuts import render

def hydroponics_view(request):
    return render(request, 'hydroponics.html')  # Optional custom template