import streamlit as st
import numpy as np
import joblib

# =======================
# Load scaler and models
# =======================
scaler = joblib.load("scaler.pkl")

log_reg = joblib.load("logistic_regression.pkl")
svm_model = joblib.load("svm.pkl")
rf_model = joblib.load("random_forest.pkl")
knn_model = joblib.load("knn.pkl")

# =======================
# App Title & Description
# =======================
st.title("💓 Heart Attack Prediction App")
st.markdown("""
This app predicts whether a person is **at risk of heart attack** based on health parameters.  
Select a model from the sidebar and enter the patient details below.
""")

# =======================
# Sidebar: Model Selection
# =======================
model_choice = st.sidebar.selectbox(
    "🔍 Select a Model",
    ("Logistic Regression", "SVM", "Random Forest", "KNN")
)

# =======================
# User Inputs
# =======================
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
bp = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
hr = st.number_input("Heart Rate", min_value=40, max_value=200, value=80)
diabetes = st.selectbox("Diabetes (1 = Yes, 0 = No)", [0, 1])
family_history = st.selectbox("Family History of Heart Disease (1 = Yes, 0 = No)", [0, 1])
smoking = st.selectbox("Smoking (1 = Yes, 0 = No)", [0, 1])
obesity = st.selectbox("Obesity (1 = Yes, 0 = No)", [0, 1])
alcohol = st.selectbox("Alcohol Consumption (1 = Yes, 0 = No)", [0, 1])
previous_hp = st.selectbox("Previous Heart Problems (1 = Yes, 0 = No)", [0, 1])
stress = st.number_input("Stress Level (0–10)", min_value=0, max_value=10, value=5)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
sleep = st.number_input("Sleep Hours Per Day", min_value=0, max_value=24, value=7)

# =======================
# Prediction Logic
# =======================
features = np.array([[age, sex, bp, hr, diabetes, family_history, smoking,
                      obesity, alcohol, previous_hp, stress, bmi, sleep]])

# Apply scaling only to models that need it
if model_choice in ["Logistic Regression", "SVM", "KNN"]:
    features_scaled = scaler.transform(features)
else:
    features_scaled = features  # Random Forest doesn't need scaling

# Predict when button is clicked
if st.button("Predict"):
    if model_choice == "Logistic Regression":
        prediction = log_reg.predict(features_scaled)[0]
    elif model_choice == "SVM":
        prediction = svm_model.predict(features_scaled)[0]
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(features_scaled)[0]
    else:  # KNN
        prediction = knn_model.predict(features_scaled)[0]

    # Display result
    if prediction == 1:
        st.error("⚠️ The patient is at **RISK** of Heart Attack!")
    else:
        st.success("✅ The patient is **NOT at risk** of Heart Attack.")


