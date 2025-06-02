import streamlit as st
import numpy as np
import joblib
from joblib import load

# Page config
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
#Title
st.title("Breast Cancer Prediction")

model = joblib.load('modelbreast.joblib')

feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Sidebar
st.sidebar.title("🔍 About This App")
st.sidebar.info("""
This model predicts whether a breast tumor is **benign or malignant** based on input features.

- Model: Logistic Regression  
- Dataset: Breast Cancer Wisconsin
""")
st.sidebar.write("✨ Developed using **Streamlit**")

# User input
st.header("Enter values for the following features:")
input_data = []

for feature in feature_names:
    value = st.number_input(feature, min_value=0.0, format="%.4f")
    input_data.append(value)

clicked=st.button("Predict")

# Predict button
if clicked==True:
    input_array = np.array(input_data).reshape(1, -1)
    pred = model.predict(input_array)
    if pred[0] == 1:
        st.write('Prediction: Benign')
    else:
        st.write("Prediction: Malignant")
