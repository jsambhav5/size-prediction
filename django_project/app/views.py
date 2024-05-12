from django.shortcuts import render, HttpResponse
import pickle
import json
import os
from pathlib import Path

def index(request):
	context = {"name": "Sambhav Jain"}
	return render(request, 'index.html', context)

def about(request):
	context = {"name": "Sambhav Jain"}
	return render(request, 'about.html', context)

def contact(request):
	context = {"name": "Sambhav Jain"}
	return render(request, 'contact.html', context)

def predict(request):
	body = request.body
	body = json.loads(body)
	data = list(body.values())
	rf = pickle.load(open(Path(__file__).parents[2].joinpath('prediction models/rf.sav'), 'rb'))
	ans = rf.predict([data])
	return(HttpResponse(ans))