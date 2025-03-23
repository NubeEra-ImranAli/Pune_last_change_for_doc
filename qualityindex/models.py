from django.db import models

class AQIAttributes(models.Model):
    no2_max = models.FloatField()
    ozone_max = models.FloatField()
    pm10_max = models.FloatField()
    co2_min = models.FloatField(null=True, blank=True)  # Optional field
    air_pressure = models.FloatField()
    humidity = models.FloatField()
    temperature_max = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

class PredictionLog(models.Model):
    aqi_value = models.FloatField()
    predicted_at = models.DateTimeField(auto_now_add=True)
    user_input = models.ForeignKey(AQIAttributes, on_delete=models.CASCADE)

class CSVFile(models.Model):
    file = models.FileField(upload_to='csvs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
