# vertical_farming/views.py
from django.shortcuts import render

def vertical_farming_view(request):
    return render(request, 'vertical_farming.html')  # Optional custom template