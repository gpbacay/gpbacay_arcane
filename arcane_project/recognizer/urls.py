from django.urls import path
from . import views

app_name = 'recognizer'

urlpatterns = [
    path('', views.index, name='index'),
    path('generate/', views.generate_text, name='generate_text'),
    path('model-info/', views.model_info, name='model_info'),
    path('health/', views.health_check, name='health_check'),
    path('reload-model/', views.reload_model, name='reload_model'),
]