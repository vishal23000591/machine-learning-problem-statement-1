from django.db import models

class MedicalAnalysis(models.Model):
    description = models.TextField()
    boolean_query = models.TextField()
    ai_analysis = models.TextField()
    image_analysis = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)