import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("disease_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.title("🩺 Smart Disease Prediction System")

st.write("Select symptoms below:")

# User inputs
fever = st.selectbox("Fever", [0, 1])
cough = st.selectbox("Cough", [0, 1])
headache = st.selectbox("Headache", [0, 1])
vomiting = st.selectbox("Vomiting", [0, 1])
fatigue = st.selectbox("Fatigue", [0, 1])

# Prediction
if st.button("Predict Disease"):

    symptoms = np.array([[fever, cough, headache, vomiting, fatigue]])

    prediction = model.predict(symptoms)

    disease = le.inverse_transform(prediction)

    st.success(f"Predicted Disease: {disease[0]}")