# soil_based/views.py
from django.shortcuts import render

def soil_based_view(request):
    return render(request, 'soil_based.html')  # Optional custom template