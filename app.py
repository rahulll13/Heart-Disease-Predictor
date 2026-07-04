# app.py - Professional Heart Disease Predictor with Fixed SHAP

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# SHAP library for model explainability
import shap
# streamlit_shap for displaying SHAP plots in Streamlit
import streamlit_shap

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Professional CSS Styling ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 15px 35px rgba(30, 60, 114, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    .main-header h1 {
        font-size: 3.2em;
        font-weight: 800;
        margin-bottom: 10px;
        letter-spacing: -1px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1.3em;
        font-weight: 400;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    /* Card Styles */
    .info-box, .success-box, .warning-box, .danger-box {
        padding: 25px;
        border-radius: 16px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-left: 5px solid #ff9800;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #f44336;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 15px 30px;
        font-size: 1.1em;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Primary Button */
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.3);
    }
    
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, #ff5252 0%, #d63031 100%);
        box-shadow: 0 12px 30px rgba(255, 107, 107, 0.4);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9ff 0%, #e8eaf6 100%);
        border-right: 3px solid #667eea;
    }
    
    .css-1lcbmhc {
        font-weight: 700;
        color: #1e3c72;
        font-size: 1.1em;
    }
    
    /* Metric Styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        border: 2px solid #e9ecef;
        transition: transform 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.0em;
        font-weight: 600;
        color: #495057;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.2em;
        font-weight: 800;
        color: #1e3c72;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f1f3f4 0%, #e8eaf6 100%);
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Animations */
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Company Footer */
    .company-footer {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin-top: 40px;
        box-shadow: 0 10px 30px rgba(30, 60, 114, 0.2);
    }
    
    .company-logo {
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 10px;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Disclaimer Box */
    .disclaimer-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        border: 2px solid #e17055;
        border-radius: 16px;
        padding: 25px;
        margin: 25px 0;
        box-shadow: 0 8px 20px rgba(225, 112, 85, 0.2);
    }
    
    .disclaimer-box h4 {
        color: #d63031;
        font-weight: 700;
        margin-bottom: 15px;
    }
    
    .disclaimer-box p {
        color: #2d3436;
        font-weight: 500;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# --- Professional Header ---
st.markdown("""
<div class="main-header fade-in-up">
    <h1>Heart Shield 🛡️</h1>
    <p>An Advanced AI-Powered Early Heart Disease Risk Predictor</p>
    <small>Developer: Rahul Kumar Sinha</small>
</div>
""", unsafe_allow_html=True)

# --- Important Disclaimer ---
st.markdown("""
<div class="disclaimer-box fade-in-up">
    <h4>⚠️ IMPORTANT MEDICAL DISCLAIMER</h4>
    <p><strong>This AI tool is for educational and informational purposes only.</strong> It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical concerns. Never ignore professional medical advice based on information from this tool.</p>
    <p><strong>Emergency:</strong> If experiencing chest pain, difficulty breathing, or other serious symptoms, seek immediate medical attention or call emergency services.</p>
</div>
""", unsafe_allow_html=True)

# --- About Section ---
st.markdown("""
<div class="fade-in-up" style="background-color: #ffffff; color: #2d3436; padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); border-left: 5px solid #2a5298;">
    <h4 style="color: #1e3c72; font-weight: 700; margin-bottom: 15px;">🏥 About Heart Shield</h4>
    <p style="font-weight: 500; line-height: 1.6;">Our advanced machine learning system analyzes 13 key medical parameters to assess heart disease risk. This tool uses sophisticated algorithms trained on clinical data to provide risk assessment insights.</p>
    <p style="font-weight: 500; line-height: 1.6;"><strong>Features:</strong> Real-time risk analysis, personalized predictions, and comprehensive risk factor identification.</p>
</div>
""", unsafe_allow_html=True)

# --- Model Loading with Enhanced Error Handling ---
@st.cache_resource
def load_models():
    """Loads models or creates a backup model with proper feature handling"""
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        X_train_scaled_sample = joblib.load('X_train_scaled_sample.pkl')
        return model, scaler, X_train_scaled_sample, True
    except FileNotFoundError:
        # Create backup model with exact 13 features
        np.random.seed(42)
        
        # Feature names matching our input
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        n_samples = 500
        X_sample = np.random.randn(n_samples, len(feature_names))
        
        # Create realistic target based on multiple features
        y_sample = ((X_sample[:, 0] > 0) & 
                   (X_sample[:, 4] > 0) & 
                   (X_sample[:, 7] < 0)).astype(int)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X_sample)
        model.fit(X_scaled, y_sample)
        
        # Create background sample for SHAP
        X_train_scaled_sample = X_scaled[np.random.choice(X_scaled.shape[0], min(100, X_scaled.shape[0]), replace=False)]
        
        return model, scaler, X_train_scaled_sample, False

# Load models
model, scaler, X_train_scaled_sample, is_pretrained = load_models()

# Model status
if is_pretrained:
    st.success("✅ Production model loaded successfully!")
else:
    st.warning("⚠️ Demo mode active. Upload trained model files for full functionality.")

# --- Enhanced Sidebar ---
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
    <h2 style="color: white; text-align: center; margin: 0;">👤 Patient Data</h2>
    <p style="color: rgba(255,255,255,0.8); text-align: center; margin: 5px 0 0 0;">Enter your medical information</p>
</div>
""", unsafe_allow_html=True)

def user_input_features():
    """Enhanced user input collection with better organization"""
    
    # Basic Information
    st.sidebar.markdown("### 📋 Basic Information")
    age = st.sidebar.slider('Age (years)', 29, 77, 48, help="Your current age")
    sex = st.sidebar.selectbox('Gender', ('Male', 'Female'), help="Biological sex")
    
    # Cardiovascular Symptoms
    st.sidebar.markdown("### 💗 Cardiovascular Symptoms")
    cp = st.sidebar.selectbox(
        'Chest Pain Type', 
        ('Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'),
        help="Type of chest pain experienced"
    )
    
    # Vital Signs & Lab Results
    st.sidebar.markdown("### 🩺 Vital Signs & Lab Results")
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 129, 
                                help="Blood pressure at rest")
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 240,
                            help="Total cholesterol level")
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ('No', 'Yes'),
                              help="Is fasting blood sugar above 120?")
    
    # Cardiac Tests
    st.sidebar.markdown("### 🫀 Cardiac Test Results")
    restecg = st.sidebar.selectbox(
        'Resting ECG Results', 
        ('Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'),
        help="Electrocardiogram results at rest"
    )
    thalach = st.sidebar.slider('Maximum Heart Rate (bpm)', 71, 202, 150,
                               help="Highest heart rate achieved")
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('No', 'Yes'),
                                help="Chest pain during exercise")
    
    # Advanced Cardiac Parameters
    st.sidebar.markdown("### 🔬 Advanced Parameters")
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.2, 1.0, step=0.1,
                               help="ST depression induced by exercise")
    slope = st.sidebar.selectbox('ST Segment Slope', ('Upsloping', 'Flat', 'Downsloping'),
                                help="Slope of peak exercise ST segment")
    ca = st.sidebar.slider('Major Vessels (0-3)', 0, 3, 0,
                          help="Number of major vessels colored by fluoroscopy")
    thal = st.sidebar.selectbox('Thalassemia', ('Normal', 'Fixed Defect', 'Reversible Defect'),
                               help="Thalassemia test result")

    # Mapping dictionaries
    sex_map = {'Male': 1, 'Female': 0}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'No': 0, 'Yes': 1}
    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    exang_map = {'No': 0, 'Yes': 1}
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}

    # Create feature data
    data = {
        'age': age, 'sex': sex_map[sex], 'cp': cp_map[cp], 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs_map[fbs], 'restecg': restecg_map[restecg],
        'thalach': thalach, 'exang': exang_map[exang], 'oldpeak': oldpeak,
        'slope': slope_map[slope], 'ca': ca, 'thal': thal_map[thal]
    }
    
    # Raw data for display
    raw_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg,
        'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal
    }
    
    features = pd.DataFrame(data, index=[0])
    return features, raw_data

# Get user input
input_df, raw_data = user_input_features()

# --- Enhanced Input Summary ---
st.markdown("## 📊 Patient Profile Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **👤 Demographics:**
    - **Age:** {} years
    - **Gender:** {}
    - **Chest Pain:** {}
    """.format(raw_data['age'], raw_data['sex'], raw_data['cp']))

with col2:
    st.markdown("""
    **🩺 Vital Signs:**
    - **BP:** {} mm Hg
    - **Cholesterol:** {} mg/dl
    - **Max HR:** {} bpm
    """.format(raw_data['trestbps'], raw_data['chol'], raw_data['thalach']))

with col3:
    st.markdown("""
    **🔬 Test Results:**
    - **ECG:** {}
    - **Exercise Angina:** {}
    - **ST Depression:** {}
    """.format(raw_data['restecg'], raw_data['exang'], raw_data['oldpeak']))
    
# --- Main Prediction Section ---
st.markdown("---")
st.markdown("## 🎯 AI-Powered Risk Prediction")

# Enhanced prediction button
predict_button = st.button('🔬 Analyze Heart Disease Risk', type="primary", use_container_width=True)

if predict_button:
    try:
        with st.spinner('🤖 Processing your medical data with advanced AI...'):
            # Scale input
            scaled_input = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(scaled_input)
            prediction_proba = model.predict_proba(scaled_input)
            
            # Enhanced results display
            st.markdown("### 🎯 AI Prediction Results")
            
            # Main result with enhanced styling
            if prediction[0] == 0:
                st.markdown("""
                <div class="success-box" style="color: #155724;">
                    <h2 style="color: #155724;">🎉 LOW RISK PREDICTION</h2>
                    <h3 style="color: #155724;">AI Assessment: Lower Probability of Heart Disease</h3>
                    <p style="color: #155724;">The machine learning model indicates a lower likelihood of heart disease based on your current parameters. However, continue monitoring your health and maintaining healthy habits.</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown("""
                <div class="danger-box" style="color: #721c24;">
                    <h2 style="color: #721c24;">🚨 HIGH RISK PREDICTION</h2>
                    <h3 style="color: #721c24;">AI Assessment: Higher Probability of Heart Disease</h3>
                    <p style="color: #721c24;">The model indicates elevated risk factors. <strong>Please consult a cardiologist promptly</strong> for comprehensive evaluation and appropriate medical care.</p>
                </div>
                """, unsafe_allow_html=True)
                st.snow()
            
            # Enhanced probability analysis
            st.markdown("### 📊 Detailed Probability Analysis")
            
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            
            no_disease_prob = prediction_proba[0][0] * 100
            disease_prob = prediction_proba[0][1] * 100
            confidence = max(prediction_proba[0]) * 100
            
            with prob_col1:
                st.metric(
                    label="🟢 No Heart Disease",
                    value=f"{no_disease_prob:.1f}%",
                    delta="Favorable" if no_disease_prob > 50 else "Concerning"
                )
            
            with prob_col2:
                st.metric(
                    label="🔴 Heart Disease Risk",
                    value=f"{disease_prob:.1f}%",
                    delta="High Alert" if disease_prob > 70 else "Monitor" if disease_prob > 50 else "Low"
                )
            
            with prob_col3:
                if confidence > 85:
                    confidence_level = "Very High"
                elif confidence > 70:
                    confidence_level = "High"
                elif confidence > 60:
                    confidence_level = "Moderate"
                else:
                    confidence_level = "Low"
                    
                st.metric(
                    label="🎯 Model Confidence",
                    value=f"{confidence:.1f}%",
                    delta=confidence_level
                )
    
    except Exception as e:
        st.error(f"❌ Analysis error occurred: {str(e)}")
        st.info("Please verify your input data and try again. If the problem persists, contact support.")
        with st.expander("Technical Details"):
            st.exception(e)

# --- Raw Data View ---
with st.expander("📋 View Technical Data"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Processed Features:**")
        st.dataframe(input_df, use_container_width=True)
    with col2:
        st.markdown("**Raw Input Values:**")
        st.json(raw_data)

# --- Company Footer ---
st.markdown("""
<div class="company-footer fade-in-up">
    <div class="company-logo">🛡️ Heart Shield</div>
    <h4>🏥 Advanced Healthcare AI Solutions</h4>
    <p><strong>Heart Shield v2.0</strong> - Powered by cutting-edge machine learning algorithms</p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0;">
        <div>
            <h5>🔬 Technology</h5>
            <p>Random Forest ML<br>Advanced Risk Scoring<br>Secure Analysis</p>
        </div>
        <div>
            <h5>🎯 Accuracy</h5>
            <p>Trained on Clinical Data<br>Validated Algorithms<br>Continuous Learning</p>
        </div>
        <div>
            <h5>🛡️ Privacy</h5>
            <p>Data Not Stored<br>Local Processing<br>HIPAA Conscious</p>
        </div>
    </div>
    <hr style="border-color: rgba(255,255,255,0.3); margin: 20px 0;">
    <p><strong>© 2026 Built by Rahul Kumar Sinha</strong></p>
    <small>Medical emergencies: Call 911</small>
</div>
""", unsafe_allow_html=True)

# --- Enhanced Sidebar Information ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Medical Parameter Guide")

with st.sidebar.expander("❓ Understanding Your Values"):
    st.markdown("""
    **🫀 Key Heart Health Indicators:**
    
    **Blood Pressure Categories:**
    - Normal: <120/80 mm Hg
    - Elevated: 120-129/<80 mm Hg  
    - Stage 1 Hypertension: 130-139/80-89
    - Stage 2 Hypertension: ≥140/90
    
    **Cholesterol Levels:**
    - Optimal: <200 mg/dl total
    - Borderline: 200-239 mg/dl
    - High: ≥240 mg/dl
    
    **Heart Rate Zones:**
    - Resting: 60-100 bpm
    - Target Exercise: 50-85% max
    - Maximum: 220 - age
    """)

with st.sidebar.expander("🚨 Emergency Contacts"):
    st.markdown("""
    **Immediate Medical Help:**
    - **Emergency:** 911
    - **Poison Control:** 1-800-222-1222
    - **Suicide Prevention:** 988
    
    **Non-Emergency Medical:**
    - **Nurse Hotline:** Contact your insurance
    - **Telemedicine:** Your healthcare provider
    
    **Chest Pain Protocol:**
    1. Call 911 immediately
    2. Chew aspirin if no allergies
    3. Sit down and rest
    4. Loosen tight clothing
    """)
