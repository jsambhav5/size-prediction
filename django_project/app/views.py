from django.shortcuts import render, HttpResponse
import pickle
import json
import os
from pathlib import Path

def index(request):
	return HttpResponse("This is HOME Page")

def about(request):
	return HttpResponse("This is ABOUT Page")

def predict(request):
	body = request.body
	body = json.loads(body)
	rf = pickle.load(open(Path(__file__).parents[2].joinpath('prediction models/rf.sav'), 'rb'))
	ans = rf.predict([body['data']])
	return(HttpResponse(ans))