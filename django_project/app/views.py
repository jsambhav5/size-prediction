from django.shortcuts import render, HttpResponse
import pickle
import json
import os
from pathlib import Path

def index(request):
	return render(request, 'index.html')

def about(request):
	return render(request, 'about.html')

def predict(request):
	body = request.body
	body = json.loads(body)
	data = list(body.values())
	rf = pickle.load(open(Path(__file__).parents[2].joinpath('prediction models/rf.sav'), 'rb'))
	ans = rf.predict([data])
	return(HttpResponse(ans))