#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 23:05:08 2025

@author: rossbrancati
"""
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Import data
df = pd.read_csv('iot_sports_training_dataset.csv')

# Assign columns from data investigation in Jupyter notebook
columns = ['HR', 'Steps', 'Accel_X', 'Accel_Y', 'Performance_Score']

# Assign subset of data for training linear regression model
df = df[columns]

# Split data into training and testing set (doing this before other steps maximizes generalizability to new, unseen data)
X = df.drop(['Performance_Score'], axis=1)
y = df['Performance_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Assign numeric features, knowing that all features are numeric 
numeric_feats = X.columns.tolist()

# Pipeline for standardizing numeric features
numeric_transformer = Pipeline(steps = [
    ('scaler', StandardScaler()),
    # If you want to use PCA for feature reduction
    # (('pca'), PCA(n_components=5))
])

# Single Preprocessor pipeline
preprocessor = ColumnTransformer([
    ('numeric transform', numeric_transformer, numeric_feats)
])

# Create a single model pipeline for pre-processing and fitting data
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit model parameters
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate RMSE and R^2
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics: 
print(f'RMSE: {rmse}')
print(f'R^2: {r2}')

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved to model.pkl")

