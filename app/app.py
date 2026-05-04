import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Paths and Constants
APP_DIR = Path(__file__).resolve().parent
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

HINTS = {
    'age': 'Age: 20-80', 'sex': '1 = male, 0 = female', 'cp': 'Chest pain type: 0-3',
    'trestbps': 'Resting BP: 90-200', 'chol': 'Cholesterol: 120-570', 'fbs': 'FBS > 120 mg/dl: 1/0',
    'restecg': 'Resting ECG: 0-2', 'thalach': 'Max heart rate: 70-210', 'exang': 'Exercise angina: 1/0',
    'oldpeak': 'ST depression: 0-7', 'slope': 'ST slope: 0-2', 'ca': 'Major vessels: 0-3', 'thal': 'Thal: 1-3'
}

# Load artifacts
try:
    model = joblib.load(APP_DIR / 'model.pkl')
    preprocessor = joblib.load(APP_DIR / 'preprocessor.pkl')
    sample_data = json.loads((APP_DIR / 'sample_patient.json').read_text())
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# Page Config
st.set_page_config(page_title='CardioAI Clinical Dashboard', layout='wide')

# Custom CSS for styling
st.markdown("""
    <style>
    /* 1. Main background */
    .stApp {
        background-color: #f8f9fa;
    }

    /* 2. Fix Input Boxes: White background, dark text, dark buttons */
    div[data-baseweb="input"] {
        background-color: white !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
    }
    
    /* Target the number input step buttons */
    div[data-testid="stNumberInputStepDown"], 
    div[data-testid="stNumberInputStepUp"] {
        background-color: #1a2a3a !important;
        color: white !important;
        border-radius: 0 4px 4px 0 !important;
    }
    
    div[data-testid="stNumberInputStepDown"] {
        border-radius: 0 !important;
        border-left: 1px solid #34495e !important;
    }

    /* Force input text color */
    input {
        color: #2c3e50 !important;
        -webkit-text-fill-color: #2c3e50 !important;
        font-weight: 500 !important;
    }

    /* 3. Fix Labels: Make feature names dark grey */
    label p {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        margin-bottom: 0.2rem !important;
    }

    /* 4. Predict Risk Button: Deeper red and white text */
    div.stButton > button:first-child {
        background-color: #b01c33 !important;
        color: white !important;
        padding: 0.5rem 2rem !important;
        border-radius: 6px !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #8e1629 !important;
        color: white !important;
    }

    /* 5. Header and Section Styling */
    .main-header {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        border-left: 8px solid #b01c33;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    .main-header h1 {
        color: #1a2a3a !important;
        font-weight: 800 !important;
        margin-bottom: 5px !important;
    }
    
    .main-header p {
        color: #5d6d7e !important;
        font-size: 1.1rem !important;
    }

    .separator-bar {
        background-color: #ffffff;
        height: 40px;
        border-radius: 12px;
        margin: 20px 0;
        border: 1px solid #eef2f7;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }

    .section-title {
        color: #1a2a3a !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin: 25px 0 15px 0 !important;
    }

    .prediction-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #eef2f7;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        margin-bottom: 15px;
    }
    
    .prediction-card h3 {
        color: #1a2a3a !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }

    .confidence-label {
        color: #5d6d7e !important;
        font-weight: 500 !important;
        margin-bottom: 0 !important;
        font-size: 0.9rem !important;
    }

    .confidence-value {
        color: #1a2a3a !important;
        font-size: 2.5rem !important;
        font-weight: 400 !important;
        margin-top: -10px !important;
    }

    /* Force dark background for charts if possible via container */
    [data-testid="stChart"] {
        background-color: #0e1117 !important;
        border-radius: 10px;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
    <div class="main-header">
        <h1>CardioAI Clinical Screening Dashboard</h1>
        <p>Hospital-style risk support for early heart disease screening.</p>
    </div>
""", unsafe_allow_html=True)

# Separator 1
st.markdown('<div class="separator-bar"></div>', unsafe_allow_html=True)

# Input Form Title
st.markdown('<div class="section-title">Patient Input Form</div>', unsafe_allow_html=True)

input_values = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(FEATURES):
    with col1 if i % 2 == 0 else col2:
        input_values[feature] = st.number_input(
            feature,
            value=float(sample_data[feature]),
            help=HINTS[feature],
            format="%.3f",
            key=f"input_{feature}"
        )

st.write("") # Spacer
if st.button("Predict Risk"):
    # Prepare data
    input_df = pd.DataFrame([input_values], columns=FEATURES)
    X_processed = preprocessor.transform(input_df)
    
    # Model Prediction
    prediction = int(model.predict(X_processed)[0])
    prediction_proba = model.predict_proba(X_processed)[0] if hasattr(model, 'predict_proba') else [0, 0]
    
    label = "Disease Present" if prediction == 1 else "No Disease"
    confidence = prediction_proba[prediction] * 100 if hasattr(model, 'predict_proba') else 100.0
    
    # Display Results
    st.markdown(f"""
        <div class="prediction-card">
            <h3>Prediction: {label}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="confidence-label">Model confidence</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="confidence-value">{confidence:.1f}%</p>', unsafe_allow_html=True)
    
    # Separator 2
    st.markdown('<div class="separator-bar"></div>', unsafe_allow_html=True)
    
    # Top Model Drivers
    st.markdown('<div class="section-title">Top Model Drivers</div>', unsafe_allow_html=True)
    
    # Get feature names from preprocessor
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = FEATURES
        
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if len(importances) == len(feature_names):
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(3)
        else:
            feat_imp = pd.Series(np.abs(X_processed[0]), index=feature_names).sort_values(ascending=False).head(3)
    else:
        feat_imp = pd.Series(np.abs(X_processed[0]), index=feature_names).sort_values(ascending=False).head(3)
    
    # Create the horizontal bar chart
    # Note: st.bar_chart uses Streamlit's default colors. 
    # To get the exact look with light blue on dark, we'd ideally use Altair, but let's try to style the container.
    st.bar_chart(feat_imp)
    
    # Disclaimer
    st.markdown(f"<p style='color: #7f8c8d; font-size: 0.9rem; margin-top: 10px;'>The strongest model drivers are {', '.join(feat_imp.index)}. Use this result as screening support, not a final diagnosis.</p>", unsafe_allow_html=True)
