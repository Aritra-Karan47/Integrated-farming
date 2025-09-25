# dashboard/weather.py
import requests
import os

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.getenv('OPENWEATHER_API_KEY')}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {'temp': data['main']['temp'], 'description': data['weather'][0]['description']}
    return {'temp': 'N/A', 'description': 'N/A'}