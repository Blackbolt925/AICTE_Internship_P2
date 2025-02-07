import pickle
import streamlit as st
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

# Define folder path
MODEL_SCALER_FOLDER = "models_scalers"

# Load models and scalers
models = {
    "Diabetes": os.path.join(MODEL_SCALER_FOLDER, "best_diabetes_model.pkl"),
    "Heart Disease": os.path.join(MODEL_SCALER_FOLDER, "best_heart_model.pkl"),
    "Parkinson's": os.path.join(MODEL_SCALER_FOLDER, "best_parkinsons_model.pkl")
}

scalers = {
    "Diabetes": os.path.join(MODEL_SCALER_FOLDER, "diabetes_scaler.pkl"),
    "Heart Disease": os.path.join(MODEL_SCALER_FOLDER, "heart_scaler.pkl"),
    "Parkinson's": os.path.join(MODEL_SCALER_FOLDER, "parkinsons_scaler.pkl")
}

# Load models and scalers into memory
loaded_models = {}
loaded_scalers = {}

for disease, model_path in models.items():
    with open(model_path, "rb") as file:
        loaded_models[disease] = pickle.load(file)

    with open(scalers[disease], "rb") as file:
        loaded_scalers[disease] = pickle.load(file)


input_features = {
    "Diabetes": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    "Heart Disease": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope","ca","thal"],
    "Parkinson's": [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", 
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", 
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", 
    "spread1", "spread2", "D2", "PPE"]

}

# Sidebar for disease selection
st.sidebar.title("Disease Prediction")
selected_disease = st.sidebar.radio("Select a disease to predict:", list(models.keys()))

st.title(f"{selected_disease} Prediction")

# Generate dynamic input fields
user_input = {}
for feature in input_features[selected_disease]:
    user_input[feature] = st.number_input(f"Enter {feature}", min_value=0.0, step=0.1)

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Apply scaling
scaler = loaded_scalers[selected_disease]
scaled_input = scaler.transform(input_df)

# Prediction button
if st.button("Predict"):
    model = loaded_models[selected_disease]
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] if hasattr(model, "predict_proba") else None

    # Display prediction
    if prediction == 1:
        st.success(f"The model predicts a high likelihood of {selected_disease}.")
    else:
        st.error(f"The model predicts a low likelihood of {selected_disease}.")

    if probability is not None:
        st.info(f"Prediction Confidence: {probability * 100:.2f}%")

    
    explainer = shap.Explainer(model, scaled_input)
    shap_values = explainer(scaled_input)

    # Get the raw SHAP values
    shap_importance = np.abs(shap_values.values[0])  

    # Get the indices of the top 3 contributing features
    top_3_indices = np.argsort(shap_importance)[-3:][::-1]  # Sort in descending order and get top 3

    # Map the indices to the corresponding feature names
    top_3_features = [input_features[selected_disease][i] for i in top_3_indices]

    # Display the explanation as text
    st.subheader("Why this prediction?")
    st.info(f"Top contributing factors: {', '.join(top_3_features)}")
