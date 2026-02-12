import streamlit as st
import pandas as pd
import pickle

# =========================
# 1. Load Trained Model
# =========================
# We use a function with caching so the model doesn't reload every time you click a button
@st.cache_resource
def load_model():
    try:
        with open("heart_disease_pipeline.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

model = load_model()

# =========================
# 2. Streamlit App UI
# =========================
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient clinical data to predict the risk of heart disease.")

if model is None:
    st.error("Error: 'heart_disease_pipeline.pkl' not found. Please upload it to your GitHub repository.")
    st.stop()

# Grouping inputs for better UI
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", 
                      ["typical angina", "asymptomatic", "non-anginal", "atypical angina"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=200, value=130)
    chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [Yes, No])
    restecg = st.selectbox("Resting Electrocardiographic Results", 
                           ["lv hypertrophy", "normal", "st-t wave abnormality"])

with col2:
    thalch = st.number_input("Maximum Heart Rate Achieved (thalch)", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", [Yes, No])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", 
                         ["downsloping", "flat", "upsloping"])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", 
                        ["fixed defect", "normal", "reversable defect"])

# =========================
# 3. Create Input DataFrame
# =========================
input_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalch": thalch,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])

# =========================
# 4. Make Prediction
# =========================
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    # Check if model has predict_proba (most classifiers do)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
        
        if prediction == 1:
            st.error(f"⚠️ Heart Disease Detected (Risk: {probability:.2%})")
        else:
            st.success(f"✅ No Heart Disease (Risk: {probability:.2%})")
    else:
        # If model doesn't support probability
        if prediction == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease")

