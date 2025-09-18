## Sports Performance ML API

This project demonstrates how to train, contanerize, and deploy a machine learning model as a production ready API. The API predicts an athlete's performance score from wearable sensor, heart rate, and step count data. The model was trained using the [iot-sports-performance-dataset](https://www.kaggle.com/datasets/ziya07/iot-driven-sports-training-dataset) provided by Ziya. 

It is built with **Fast API** for serving predictions and **Docker** for contanerized deployment. The API can be run locally or deployed to cloud platforms such as Render or Google Cloud Run. For this project, I deployed with Render. 

I have also included a Jupyter notebook file in this repo walking through feature selection, model selection, and training and testing. 

## Table of Contents:
- [Project overview](#project-overview)
- [Tech stack](#tech-stack)
- [Demo the model](#demo-model)
- [Getting Started](#getting-started)

## Project Overview:
- **Goal:** show end-to-end ML workflow from dataset → model training → API deployment.
- **Domain:** sports science and wearable sensors
- **Key Features:**
  - Heart rate (HR)
  - Step count
  - Accelerometer (X, Y axes)
- **Output:** predicted performance_score

In general, this project highlights how machine learning can be integrated into applications for sports analytics, biomechanics, and wearable sensors. Fair warning that the predictive performance is not excellent (RMSE: 6.53, $R^2$: 0.602) as the main purpose was demonstration of contanerizing and deploying a ML model. 

## Tech stack:
- **Python** (pandas, scikit-learn, numpy)
- **FastAPI** for API framework
- **Uvicorn** ASGI server
- **Docker** for contanerization
- **Joblib** for model serialization

## Demo model:
1. Go to https://sports-perf-api.onrender.com/docs
2. Click on the POST/predict endpoint (it will expand)
3. Click on the "Try it out" button (top right of dropdown)
4. Enter a JSON input that matches my schema, for example:

```JSON
{
  "HR": 137,
  "Steps": 5524,
  "Accel_X": 1.37,
  "Accel_Y": 0.49
}
```
5. Click Execute, and you should see a predicted performance_score in the Response Body:

![response_body](https://github.com/rossbrancati/sports_performance_ML_api/blob/main/assets/result.png)

## Getting started:

### 1. Clone the Repository
```bash
git clone https://github.com/rossbrancati/sports_performance_ML_api.git
cd sports_performance_ML_api
```

### 2. Retrain the model

You can use the saved (.pkl) model, or retrain it with your own selected featureset. Running this will generate ```model.pkl``` used by the API.
```
python train.py
```

### 3. Run locally with Docker

Build and run the container. The API will be available at http://localhost:8080/docs
```
docker build -t sports-perf-api .
docker run -p 8080:8080 sports-perf-api
```

### 4. Deploy the model to a cloud platform like [Render](https://render.com/)

I would recommend Render for starting out as they have a free tier and integrates with Github very easily. 



