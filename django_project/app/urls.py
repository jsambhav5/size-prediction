from django.contrib import admin
from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from app import views

urlpatterns = [
    path("", views.index, name='home'),
    path("home", views.index, name='home'),
    path("about", views.about, name='about'),
    path("contact", views.about, name='contact'),
    path("predict", csrf_exempt(views.predict), name='predict')
]
