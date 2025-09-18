#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 22:03:42 2025

@author: rossbrancati
"""
from fastapi import FastAPI 
from pydantic import BaseModel
import joblib
import numpy as np

# Load model at startup
model = joblib.load("model.pkl")

app = FastAPI(title="Sports Performance API", description="Predict performance score from wearable features.")

# Define input schema
class Features(BaseModel):
    HR: float
    Steps: float
    Accel_X: float
    Accel_Y: float

@app.post("/predict")
def predict(features: Features):
    # Convert to numpy array
    data = np.array([[features.HR, features.Steps, features.Accel_X, features.Accel_Y]])
    prediction = model.predict(data)[0]
    return {"predicted_performance_score": prediction}