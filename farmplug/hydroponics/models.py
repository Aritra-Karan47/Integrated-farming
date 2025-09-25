# hydroponics/models.py
from django.db import models

class SensorData(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    raw_value = models.FloatField()
    processed_value = models.FloatField()