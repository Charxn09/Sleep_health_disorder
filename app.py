import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Title & Description
st.markdown('<div class="main-header">😴 Sleep Health & Lifestyle Disorder Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Machine Learning clinical decision support tool to screen for Sleep Apnea, Insomnia, or Normal Sleep patterns.</div>', unsafe_allow_html=True)

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load('sleep_disorder_model.pkl')

try:
    artifacts = load_model()
    model = artifacts['model']
    label_encoders = artifacts['label_encoders']
    feature_names = artifacts['feature_names']
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'sleep_disorder_model.pkl' is in the same directory.")
    st.stop()

# Sidebar for Info
with st.sidebar:
    st.header("📋 About the Project")
    st.markdown("""
    This application predicts potential sleep disorders using a **Random Forest Classifier** trained on biometric and lifestyle variables.
    
    **Predictable Categories:**
    - ✅ **None**: Healthy sleep habits
    - ⚠️ **Insomnia**: Difficulty falling or staying asleep
    - 🚨 **Sleep Apnea**: Breathing interruptions during sleep
    
    *Disclaimer: This tool is an academic project and should not replace professional medical diagnosis.*
    """)
    st.markdown("---")
    st.caption("Term Project | Data Science & ML")

# Input Form in 2 Columns
st.subheader("👤 Patient & Lifestyle Parameters")
col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🩺 Demographics & Vitals")
    gender = st.selectbox("Gender", options=label_encoders['Gender'].classes_)
    age = st.slider("Age (years)", min_value=18, max_value=80, value=32)
    occupation = st.selectbox("Occupation", options=label_encoders['Occupation'].classes_)
    bmi_category = st.selectbox("BMI Category", options=label_encoders['BMI Category'].classes_)
    
    st.markdown("##### ❤️ Cardiovascular Indicators")
    systolic_bp = st.number_input("Systolic Blood Pressure (upper, mmHg)", min_value=90, max_value=200, value=120, step=1)
    diastolic_bp = st.number_input("Diastolic Blood Pressure (lower, mmHg)", min_value=60, max_value=130, value=80, step=1)
    heart_rate = st.slider("Resting Heart Rate (bpm)", min_value=50, max_value=120, value=70)

with col2:
    st.markdown("##### 🌙 Sleep & Daily Routine")
    sleep_duration = st.slider("Daily Sleep Duration (hours)", min_value=3.0, max_value=12.0, value=7.0, step=0.1)
    quality_of_sleep = st.slider("Subjective Sleep Quality (1 = Poor, 10 = Excellent)", min_value=1, max_value=10, value=7)
    stress_level = st.slider("Daily Stress Level (1 = Calm, 10 = Extreme)", min_value=1, max_value=10, value=5)
    physical_activity = st.slider("Physical Activity (minutes/day)", min_value=10, max_value=120, value=45, step=5)
    daily_steps = st.number_input("Daily Steps", min_value=1000, max_value=25000, value=6500, step=500)

st.markdown("---")

# Predict Button
if st.button("🔍 Analyze & Predict Sleep Health", type="primary", use_container_width=True):
    # Prepare input data
    input_data = {
        'Gender': label_encoders['Gender'].transform([gender])[0],
        'Age': age,
        'Occupation': label_encoders['Occupation'].transform([occupation])[0],
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'BMI Category': label_encoders['BMI Category'].transform([bmi_category])[0],
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp
    }

    input_df = pd.DataFrame([input_data])[feature_names]
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    st.subheader("📊 Diagnostic Assessment")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 'None':
            st.success("### ✅ Status: Healthy Sleep (No Disorder)")
            st.write("The entered profile does not show clinical symptoms of Insomnia or Sleep Apnea. Keep up the consistent sleep schedule and healthy physical activity!")
        elif prediction == 'Insomnia':
            st.warning("### ⚠️ Status: Indicators of Insomnia")
            st.write("The model identified patterns commonly associated with **Insomnia** (e.g., elevated stress, reduced sleep quality, or irregular sleep duration).")
        else:
            st.error("### 🚨 Status: Indicators of Sleep Apnea")
            st.write("The model identified physiological markers correlated with **Sleep Apnea** (e.g., elevated blood pressure, higher BMI, or disrupted sleep patterns). Medical consultation is recommended.")

    with res_col2:
        st.markdown("##### 📈 Model Confidence")
        prob_df = pd.DataFrame({
            'Disorder': classes,
            'Confidence (%)': (probabilities * 100).round(1)
        }).sort_values(by='Confidence (%)', ascending=False)
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

    # Health Recommendations Section
    with st.expander("💡 Recommended Lifestyle Interventions"):
        if prediction == 'None':
            st.markdown("- Maintain regular bedtime and wake-up hours.\n- Continue moderate aerobic exercise daily.")
        elif prediction == 'Insomnia':
            st.markdown("- Practice sleep hygiene: avoid screens 1 hour before bed.\n- Integrate stress reduction techniques like guided meditation or breathwork.\n- Avoid high caffeine intake in the afternoon/evening.")
        else:
            st.markdown("- Monitor your blood pressure and BMI regularly.\n- Consult a sleep specialist or ENT for a polysomnography (sleep study).\n- Sleep on your side rather than flat on your back to reduce airway obstruction.")
