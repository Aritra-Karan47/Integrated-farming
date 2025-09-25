# common/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required
from django import forms
from .models import FarmingType, UserProfile
from django.db import connection

class FarmingForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['farming_types']

@login_required
def profile(request):
    profile = UserProfile.objects.get(user=request.user)
    if request.method == 'POST':
        form = FarmingForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            # Create dynamic tables for new types with specific sensors
            for ft in form.cleaned_data['farming_types']:
                with connection.schema_editor() as schema_editor:
                    for sensor in get_sensors_for_type(ft.name):
                        table_name = f"{ft.name}_{sensor}"
                        schema_editor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (id SERIAL PRIMARY KEY, timestamp TIMESTAMP, raw_value FLOAT, processed_value FLOAT);")
            return redirect('dashboard')
    else:
        form = FarmingForm(instance=profile)
    return render(request, 'profile.html', {'form': form})

class CustomLoginView(LoginView):
    template_name = 'login.html'

# Helper function to define sensors per farming type
def get_sensors_for_type(farming_type):
    sensor_mapping = {
        'hydroponics': ['ph', 'water_level', 'temperature', 'humidity', 'ec'],
        'soil_based': ['moisture', 'temperature', 'npk', 'ph', 'light'],
        'aquaponics': ['water_quality', 'temperature', 'ph', 'oxygen_level', 'flow_rate'],
        'vertical_farming': ['light_intensity', 'temperature', 'humidity', 'co2_level', 'water_level'],
    }
    return sensor_mapping.get(farming_type.lower(), [])