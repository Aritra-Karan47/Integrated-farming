# aquaponics/views.py
from django.shortcuts import render

def aquaponics_view(request):
    return render(request, 'aquaponics.html')  # Optional custom template