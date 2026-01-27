#!/usr/bin/env python3
"""
CRKP Risk Calculator - Streamlit Web App
Deploy: https://share.streamlit.io/yourusername/CRKP_Risk_Calculator
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="CRKP Risk Calculator",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    .risk-high { color: #DC2626; font-weight: bold; }
    .risk-medium { color: #D97706; font-weight: bold; }
    .risk-low { color: #059669; font-weight: bold; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🦠 CRKP Risk Calculator</h1>', unsafe_allow_html=True)
st.markdown("**Predicting Carbapenem-Resistant *Klebsiella pneumoniae* Infection Risk**")

# Load model
@st.cache_resource
def load_model():
    """Load trained model from GitHub."""
    try:
        # Load model
        model = joblib.load('models/xgboost_model.pkl')
        
        # Load feature names
        feature_names = joblib.load('models/feature_names.pkl')
        
        # Model performance metrics (hardcoded for GitHub)
        metrics = {
            'auroc': 0.827,
            'auprc': 0.685,
            'brier_score': 0.093,
            'calibration_slope': 1.311,
            'optimal_threshold': 0.30
        }
        
        return model, feature_names, metrics
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Load model
model, feature_names, metrics = load_model()

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Patient Input", "📊 Results", "ℹ️ Model Info"])

with tab1:
    st.header("Enter Patient Information")
    st.markdown("All data must be available before culture order time.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics & Clinical")
        age = st.number_input("Age (years)", 0, 120, 75, help="Patient age")
        clinical_risk_score = st.slider("Clinical Risk Score", 0, 10, 3, 
                                       help="0-10 score based on comorbidities")
        icu_admission_30d = st.checkbox("ICU admission (past 30 days)")
        current_location_icu = st.checkbox("Currently in ICU")
    
    with col2:
        st.subheader("Antibiotic Exposure")
        num_abx_classes_30d = st.number_input("Antibiotic classes (past 30d)", 0, 10, 1)
        recent_abx_7d = st.checkbox("Antibiotics in past 7 days")
        carbapenem_30d = st.checkbox("Carbapenem exposure (past 30d)")
        abx_intensity = st.slider("Antibiotic Intensity", 0, 5, 1, 
                                 help="0=no antibiotics, 5=multiple broad-spectrum")
    
    # ICU severity if in ICU
    icu_severity = 0
    if current_location_icu or icu_admission_30d:
        icu_severity = st.slider("ICU Severity Score", 0, 5, 2, 
                                help="0=no organ support, 5=multiple organ support")
    
    # Calculate interaction
    age_icu_interaction = age * icu_admission_30d
    
    # Prepare input data
    input_data = {
        'age': age,
        'icu_admission_30d': icu_admission_30d,
        'current_location_icu': current_location_icu,
        'num_abx_classes_30d': num_abx_classes_30d,
        'recent_abx_7d': recent_abx_7d,
        'carbapenem_30d': carbapenem_30d,
        'clinical_risk_score': clinical_risk_score,
        'age_icu_interaction': age_icu_interaction,
        'icu_severity': icu_severity,
        'abx_intensity': abx_intensity
    }
    
    # Calculate button
    calculate = st.button("🚀 Calculate CRKP Risk", type="primary", use_container_width=True)

with tab2:
    st.header("Prediction Results")
    
    if 'probability' not in st.session_state:
        st.info("Enter patient data and click 'Calculate CRKP Risk' to see results.")
    else:
        probability = st.session_state.probability
        risk_category = st.session_state.risk_category
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("CRKP Probability", f"{probability:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            risk_color = "risk-high" if risk_category == "High" else "risk-medium" if risk_category == "Moderate" else "risk-low"
            st.markdown(f'Risk Category: <span class="{risk_color}">{risk_category}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Optimal Threshold", "30.0%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Gauge chart
        st.subheader("Risk Visualization")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "CRKP Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 10], 'color': "green"},
                    {'range': [10, 30], 'color': "yellow"},
                    {'range': [30, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Clinical recommendations
        st.subheader("Clinical Recommendations")
        
        if probability >= 0.30:
            st.error("""
            **🚨 HIGH RISK - CONSIDER ISOLATION PRECAUTIONS**
            
            **Recommended Actions:**
            1. **Contact Precautions**: Implement immediately
            2. **Empirical Therapy**: Consider carbapenem-sparing regimens
            3. **Diagnostics**: Expedite culture results
            4. **Consultation**: Infectious diseases consult
            5. **Documentation**: Document CRKP risk assessment
            """)
        elif probability >= 0.10:
            st.warning("""
            **⚠️ MODERATE RISK - ENHANCED MONITORING**
            
            **Recommended Actions:**
            1. **Close Monitoring**: Increase assessment frequency
            2. **Review Antibiotics**: Assess current regimen
            3. **Ensure Cultures**: Confirm appropriate testing
            4. **Prepare for Escalation**: Have isolation plan ready
            """)
        else:
            st.success("""
            **✅ LOW RISK - ROUTINE CARE**
            
            **Recommended Actions:**
            1. **Standard Monitoring**: Routine clinical assessment
            2. **Antibiotic Stewardship**: Consider de-escalation
            3. **Document**: Record low risk assessment
            4. **Reassess**: If clinical status changes
            """)

with tab3:
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        st.markdown(f"""
        **Primary Metrics:**
        - **AUROC**: 0.827 (95% CI: 0.792-0.860)
        - **AUPRC**: 0.685 (95% CI: 0.631-0.737)
        - **Brier Score**: 0.093
        - **Calibration Slope**: 1.311
        
        **Clinical Thresholds:**
        - **Screening**: 5% (high sensitivity)
        - **Monitoring**: 10% (balanced)
        - **Isolation**: 30% (high specificity)
        
        **Number Needed to Evaluate**: 5.8
        """)
    
    with col2:
        st.subheader("Model Details")
        st.markdown("""
        **Algorithm**: XGBoost with Platt Scaling
        **Features**: 10 clinical predictors
        **Training Data**: 5,780 patients
        **Test Data**: 1,445 patients
        **Validation**: Temporal 80/20 split
        **Prevalence**: 12.7% (train), 17.2% (test)
        
        **Key Features:**
        1. ICU admission history
        2. Antibiotic exposure
        3. Clinical risk score
        4. Age
        5. ICU severity
        """)
    
    st.subheader("Feature Importance")
    st.markdown("""
    1. **icu_severity** - ICU severity score
    2. **abx_intensity** - Antibiotic intensity
    3. **icu_admission_30d** - ICU admission
    4. **current_location_icu** - Current ICU
    5. **clinical_risk_score** - Clinical risk
    6. **age_icu_interaction** - Age × ICU
    7. **recent_abx_7d** - Recent antibiotics
    8. **num_abx_classes_30d** - Antibiotic classes
    9. **age** - Patient age
    10. **carbapenem_30d** - Carbapenems
    """)
    
    st.subheader("Limitations")
    st.markdown("""
    - For clinical decision support only
    - Requires external validation
    - Retrospective training data
    - Single-center development
    - Regular updates needed for resistance patterns
    """)

# Prediction logic
if calculate and model is not None:
    try:
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns
        input_df = input_df[feature_names]
        
        # Make prediction
        probability = model.predict_proba(input_df)[0, 1]
        
        # Determine risk category
        if probability < 0.10:
            risk_category = "Low"
        elif probability < 0.30:
            risk_category = "Moderate"
        else:
            risk_category = "High"
        
        # Store results
        st.session_state.probability = probability
        st.session_state.risk_category = risk_category
        
        # Show success
        st.success("✅ Prediction complete! Switch to Results tab.")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Sidebar
with st.sidebar:
    st.markdown("## 🏥 Quick Info")
    st.markdown("""
    **Purpose**: Predict CRKP infection risk
    
    **Use Cases**:
    - Infection control decisions
    - Antibiotic stewardship
    - Patient risk stratification
    
    **Model Stats**:
    - AUROC: 0.827
    - Patients: 7,225
    - Features: 10
    - Validation: Temporal
    """)
    
    st.markdown("---")
    
    st.markdown("## ⚠️ Important")
    st.markdown("""
    This tool is for:
    - Clinical decision support
    - Research purposes
    - Educational use
    
    **Not for**:
    - Diagnostic decisions
    - Treatment without clinician
    - Legal/regulatory use
    """)

# Footer
st.markdown("---")
st.markdown("**CRKP Risk Calculator v2.0** | For research and clinical support | [GitHub](https://github.com)")
