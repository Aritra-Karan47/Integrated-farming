# dashboard/views.py
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .weather import get_weather
import pandas as pd
from django.db import connection
from common.views import get_sensors_for_type

@login_required
def dashboard(request):
    profile = request.user.userprofile
    farming_types = profile.farming_types.all()
    selected_type = request.GET.get('type', farming_types.first().name if farming_types.exists() else None)
    view = request.GET.get('view', 'graphs')

    weather_data = get_weather('your_city')  # Replace with user location

    if selected_type:
        sensors = get_sensors_for_type(selected_type)
        sensor_data = {}
        for sensor in sensors:
            table = f"{selected_type}_{sensor}"
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT timestamp, processed_value FROM {table} ORDER BY timestamp DESC LIMIT 100;")
                df = pd.DataFrame(cursor.fetchall(), columns=['timestamp', 'value'])
            sensor_data[sensor] = df.to_json(orient='records')

        tasks = FarmingType.objects.get(name=selected_type).tasks  # Assume tasks are stored as text or JSON

    context = {
        'farming_types': farming_types,
        'selected_type': selected_type,
        'view': view,
        'weather': weather_data,
        'sensor_data': sensor_data if view == 'graphs' else None,
        'tasks': tasks if view == 'notes' else None,
    }
    return render(request, 'dashboard.html', context)