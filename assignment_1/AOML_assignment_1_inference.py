import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

scaler = joblib.load('C:\Users\khushi shetty\OneDrive\Desktop\AOML\scaler.pkl')
ohe = joblib.load('C:\Users\khushi shetty\OneDrive\Desktop\AOML\ohe.pkl')
knn_model = joblib.load('C:\Users\khushi shetty\OneDrive\Desktop\AOML\knn.pkl')
dt_model = joblib.load('C:\Users\khushi shetty\OneDrive\Desktop\AOML\dt_classifier.pkl')

