import mediapipe as mp
import cv2
import numpy as np 
import os
import pandas as pd
import csv
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle5

df = pd.read_csv('sign13_v1.csv')

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
start_time = time.time()  # Start time
data_count = 0  # Counter for number of data points trained
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model
    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat)
    print(f"{algo} Accuracy:", accuracy)

    data_count += len(X_train)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds, Data count: {data_count}")

with open('handsign13.pkl', 'wb') as f:
    pickle5.dump(fit_models['rf'], f)
