import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="CRKP Risk Calculator",
    page_icon="🦠",
    layout="wide"
)

# Title and description
st.title("CRKP Risk Prediction Calculator")
st.markdown("**Clinical Decision Support for Carbapenem-Resistant *Klebsiella pneumoniae***")
st.markdown("**Validated on retrospective cohort (n=7,225) | Temporal validation | Clinical optimization**")

# Load robust ensemble model
@st.cache_resource
def load_model():
    """Load robust ensemble model."""
    try:
        # Load the ensemble model
        ensemble = joblib.load('models/robust_ensemble_latest.pkl')
        
        # Feature information for the new model
        feature_info = {
            'features': ensemble['features'],
            'optimal_threshold': ensemble['optimal_threshold'],
            'model_type': 'Robust Ensemble (LR + RF + XGBoost)'
        }
        
        return ensemble, feature_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

ensemble, feature_info = load_model()

# Create tabs
tab1, tab2, tab3 = st.tabs(["📋 Patient Input", "📊 Results", "ℹ️ Model Info"])

with tab1:
    st.header("Patient Information Input")
    st.markdown("*All data must be available before culture order time (1-hour temporal firewall)*")
    
    input_data = {}
    
    # Clinical feature inputs - Only 8 features for robust model
    col1, col2 = st.columns(2)
    
    # Left column
    with col1:
        # 1. Age
        input_data['age'] = st.number_input(
            "**Age (years)**", 
            0, 120, 65,
            help="Patient age at time of culture"
        )
        
        # 2. Number of antibiotic classes
        input_data['num_abx_classes_30d'] = st.number_input(
            "**Number of antibiotic classes (past 30 days)**", 
            0, 10, 1,
            help="Count of different antibiotic classes used in past 30 days"
        )
        
        # 3. Recent antibiotics (≤7 days)
        recent_abx = st.checkbox(
            "**Recent antibiotic use (≤7 days)**", 
            help="Any antibiotic use within the past 7 days"
        )
        input_data['recent_abx_7d'] = 1 if recent_abx else 0
        
        # 4. Albumin
        input_data['albumin_last'] = st.number_input(
            "**Albumin (g/L)**", 
            10.0, 60.0, 35.0, 0.1,
            help="Most recent albumin level (normal: 35-50 g/L)"
        )
    
    # Right column
    with col2:
        # 5. ICU admission (past 30d)
        input_data['icu_admission_30d'] = st.checkbox(
            "**ICU admission (past 30 days)**",
            help="Any ICU admission in the past 30 days"
        )
        
        # 6. Current ICU location
        input_data['current_location_icu'] = st.checkbox(
            "**Currently in ICU**",
            help="Patient is currently in ICU at time of culture"
        )
        
        # 7. Carbapenem use
        input_data['carbapenem_30d'] = st.checkbox(
            "**Carbapenem use (past 30 days)**",
            help="Any carbapenem antibiotic use in past 30 days (meropenem, imipenem, ertapenem)"
        )
        
        # 8. ICU + Broad spectrum interaction
        icu_broad = st.checkbox(
            "**ICU + Broad spectrum antibiotics**",
            help="Combination of ICU admission AND broad-spectrum antibiotic use"
        )
        input_data['icu_broad_spectrum'] = 1 if icu_broad else 0
    
    # Add reset button
    col1, col2, col3 = st.columns(3)
    with col2:
        calculate = st.button("**Calculate CRKP Risk**", type="primary", use_container_width=True)

with tab2:
    st.header("Prediction Results")
    
    if 'probability' not in st.session_state:
        st.info("👉 **Enter patient data in 'Patient Input' tab and click 'Calculate CRKP Risk'**")
    else:
        probability = st.session_state.probability
        risk_category = st.session_state.risk_category
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("**CRKP Probability**", f"{probability:.1%}")
        with col2:
            color_map = {
                "Very Low": "green",
                "Low": "green",
                "Low-Moderate": "yellow",
                "Moderate": "yellow", 
                "Moderate-High": "orange",
                "High": "orange",
                "Very High": "red"
            }
            st.metric("**Risk Category**", risk_category)
        with col3:
            threshold = feature_info['optimal_threshold'] if feature_info else 0.38
            prediction = "CRKP Likely" if probability >= threshold else "CRKP Unlikely"
            st.metric("**Prediction**", prediction)
        
        # Gauge chart
        st.subheader("Risk Visualization")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "CRKP Risk %", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 10], 'color': "green", 'name': "Very Low"},
                    {'range': [10, 38], 'color': "yellow", 'name': "Low-Moderate"},
                    {'range': [38, 50], 'color': "orange", 'name': "Moderate-High"},
                    {'range': [50, 100], 'color': "red", 'name': "Very High"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 38.0  # Optimal threshold
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(t=50, b=50))
        st.plotly_chart(fig, use_container_width=True)
        
        # Clinical recommendations based on multiple thresholds
        st.subheader("Clinical Recommendations by Scenario")
        
        # Three scenarios in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎯 **Screening Scenario**")
            st.markdown("**Threshold: 0.10**")
            st.markdown("*Goal: Maximize sensitivity to rule out CRKP*")
            if probability >= 0.10:
                st.warning("""
                **Consider CRKP:**
                • High sensitivity screening positive
                • Screen with cultures
                • Monitor closely for signs
                • Consider empiric coverage
                """)
            else:
                st.success("""
                **CRKP Unlikely:**
                • Very low risk (<10%)
                • Routine monitoring
                • Can consider de-escalation
                """)
        
        with col2:
            st.markdown("### ⚖️ **Optimal Scenario**")
            st.markdown(f"**Threshold: {threshold:.2f}**")
            st.markdown("*Goal: Balance sensitivity and specificity*")
            if probability >= threshold:
                st.warning("""
                **High Risk:**
                • Implement contact precautions
                • Consider isolation
                • Target testing
                • Infectious disease consult
                """)
            else:
                st.success("""
                **Moderate Risk:**
                • Enhanced monitoring
                • Consider screening
                • Standard precautions
                """)
        
        with col3:
            st.markdown("### 🚨 **Isolation Scenario**")
            st.markdown("**Threshold: 0.50**")
            st.markdown("*Goal: Minimize false alarms for isolation*")
            if probability >= 0.50:
                st.error("""
                **Very High Risk:**
                • Implement strict isolation
                • Empirical anti-CRKP therapy
                • Urgent infectious disease consult
                • Environmental cleaning
                """)
            else:
                st.info("""
                **Manage Normally:**
                • Standard precautions
                • Routine monitoring
                • Consider other pathogens
                """)
        
        # Risk factor summary
        st.subheader("Risk Factor Analysis")
        if feature_info and 'features' in feature_info:
            features = feature_info['features']
            risk_factors = []
            for feat in features:
                if feat in input_data:
                    if feat == 'recent_abx_7d' and input_data[feat] == 1:
                        risk_factors.append("Recent antibiotic use (≤7 days)")
                    elif feat == 'carbapenem_30d' and input_data[feat] == 1:
                        risk_factors.append("Carbapenem use (past 30 days)")
                    elif feat == 'icu_admission_30d' and input_data[feat] == 1:
                        risk_factors.append("ICU admission (past 30 days)")
                    elif feat == 'current_location_icu' and input_data[feat] == 1:
                        risk_factors.append("Currently in ICU")
                    elif feat == 'icu_broad_spectrum' and input_data[feat] == 1:
                        risk_factors.append("ICU + Broad spectrum antibiotics")
                    elif feat == 'albumin_last' and input_data[feat] < 35:
                        risk_factors.append(f"Low albumin ({input_data[feat]} g/L)")
                    elif feat == 'num_abx_classes_30d' and input_data[feat] >= 2:
                        risk_factors.append(f"Multiple antibiotic classes ({input_data[feat]})")
            
            if risk_factors:
                st.write("**Identified Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.info("No major risk factors identified")

with tab3:
    st.header("Model Information & Validation")
    
    # Performance metrics
    st.subheader("📈 Model Performance (Temporal Validation)")
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.markdown("""
        **Primary Metrics:**
        - **AUROC:** 0.698 (95% CI: 0.668-0.730)
        - **AUPRC:** 0.267 (95% CI: 0.237-0.312)
        - **Sensitivity:** 52.8% (46.9-59.3%)
        - **Specificity:** 68.4% (65.9-70.9%)
        - **PPV:** 25.7% (21.7-29.6%)
        - **NPV:** 87.5% (85.5-89.7%)
        - **Brier Score:** 0.145
        - **Optimal Threshold:** 0.380
        """)
    
    with perf_col2:
        st.markdown("""
        **Dataset Characteristics:**
        - **Total Patients:** 7,225
        - **CRKP Cases:** 981 (13.6%)
        - **Training Set:** 5,780 patients (12.7% CRKP)
        - **Test Set:** 1,445 patients (17.2% CRKP)
        - **Temporal Split:** 80/20 chronological
        - **Validation:** 5-fold CV + temporal holdout
        """)
    
    # Model architecture
    st.subheader("⚙️ Model Architecture")
    
    arch_col1, arch_col2 = st.columns(2)
    
    with arch_col1:
        st.markdown("""
        **Robust Ensemble Model:**
        1. **Logistic Regression** (regularized)
        2. **Random Forest** (limited depth)
        3. **XGBoost** (regularized)
        4. **Meta-model:** Logistic regression
        
        **Key Features (8 total):**
        1. Age
        2. Number of antibiotic classes (30d)
        3. Recent antibiotics (≤7 days)
        4. Albumin (g/L)
        5. ICU admission (30d)
        6. Current ICU location
        7. Carbapenem use (30d)
        8. ICU + Broad spectrum interaction
        """)
    
    with arch_col2:
        st.markdown("""
        **Feature Importance (Top Predictors):**
        1. **Recent antibiotics (≤7 days)**
        2. **Carbapenem use (past 30 days)**
        3. **Number of antibiotic classes**
        4. **Albumin level**
        5. **Age**
        6. **ICU exposures**
        
        **Model Strengths:**
        • High NPV (87.5%) for ruling out CRKP
        • Simplified 8-feature clinical model
        • Temporal validation for real-world performance
        • Clinical optimization for utility
        """)
    
    # Clinical utility
    st.subheader("🏥 Clinical Utility")
    
    util_col1, util_col2, util_col3 = st.columns(3)
    
    with util_col1:
        st.markdown("""
        **🎯 Screening (0.10 threshold):**
        - Sensitivity: 91.9%
        - Specificity: 30.2%
        - **Use:** Rule out CRKP
        - **Action:** Low threshold for testing
        """)
    
    with util_col2:
        st.markdown("""
        **⚖️ Optimal (0.38 threshold):**
        - Sensitivity: 52.8%
        - Specificity: 68.4%
        - **Use:** Risk stratification
        - **Action:** Targeted interventions
        """)
    
    with util_col3:
        st.markdown("""
        **🚨 Isolation (0.50 threshold):**
        - Sensitivity: 20.2%
        - Specificity: 96.1%
        - **Use:** Isolation decisions
        - **Action:** Strict precautions
        """)
    
    # Methodology
    st.subheader("🔬 Methodology")
    st.markdown("""
    - **Outcome:** CRKP = non-susceptible to imipenem/meropenem/ertapenem (CLSI/EUCAST)
    - **Temporal Firewall:** All predictors ≤ (culture time - 1 hour)
    - **Preprocessing:** Median imputation + clinical logic (training only)
    - **Missing Data:** <1% for most features, handled appropriately
    - **Calibration:** Platt scaling applied
    - **Ethics:** IRB approved, data anonymized
    - **Reproducibility:** Frozen dataset with hash verification
    """)
    
    # Limitations
    st.subheader("⚠️ Limitations & Considerations")
    st.markdown("""
    1. **Retrospective design:** Subject to biases of observational data
    2. **Single center:** External validation recommended
    3. **Prevalence sensitivity:** Performance varies with local prevalence
    4. **Research tool:** Not yet prospectively validated
    5. **Clinical judgment:** Should supplement, not replace clinical assessment
    6. **Missing data:** Some laboratory values had moderate missingness
    """)

# Ensemble prediction function
def ensemble_predict_proba(ensemble_model, X):
    """Get predictions from ensemble."""
    if ensemble_model is None:
        return np.array([0.5])
    
    try:
        base_preds = []
        for name, model in ensemble_model['base_models'].items():
            if hasattr(model, 'predict_proba'):
                base_preds.append(model.predict_proba(X)[:, 1])
            else:
                # Fallback for models without predict_proba
                base_preds.append(model.predict(X))
        
        base_preds_array = np.column_stack(base_preds)
        
        if 'meta_model' in ensemble_model and hasattr(ensemble_model['meta_model'], 'predict_proba'):
            return ensemble_model['meta_model'].predict_proba(base_preds_array)[:, 1]
        else:
            # Simple average if no meta-model
            return np.mean(base_preds_array, axis=1)
    except:
        return np.array([0.5])

# Prediction logic
if calculate and ensemble is not None:
    try:
        # Prepare input data
        features = feature_info['features']
        
        # Ensure all features are present with default values
        for feat in features:
            if feat not in input_data:
                # Set reasonable defaults
                if feat in ['age', 'albumin_last', 'num_abx_classes_30d']:
                    input_data[feat] = 0  # Will be handled by imputation
                else:
                    input_data[feat] = 0
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_data])
        
        # Reorder columns to match model features
        for feat in features:
            if feat not in input_df.columns:
                input_df[feat] = 0
        
        input_df = input_df[features]
        
        # Handle missing values (simple imputation)
        for col in input_df.columns:
            if input_df[col].isna().any() or input_df[col].isnull().any():
                # Use simple defaults based on feature type
                if col == 'age':
                    input_df[col] = 65  # Median age
                elif col == 'albumin_last':
                    input_df[col] = 35.0  # Typical value
                elif col == 'num_abx_classes_30d':
                    input_df[col] = 0  # No antibiotics
                else:  # Binary features
                    input_df[col] = 0
        
        # Convert to numpy array
        X = input_df.values.reshape(1, -1)
        
        # Predict using ensemble
        probability = ensemble_predict_proba(ensemble, X)[0]
        
        # Determine risk category
        if probability < 0.05:
            risk_category = "Very Low"
        elif probability < 0.10:
            risk_category = "Low"
        elif probability < 0.20:
            risk_category = "Low-Moderate"
        elif probability < 0.38:
            risk_category = "Moderate"
        elif probability < 0.50:
            risk_category = "Moderate-High"
        elif probability < 0.70:
            risk_category = "High"
        else:
            risk_category = "Very High"
        
        # Store in session state
        st.session_state.probability = probability
        st.session_state.risk_category = risk_category
        st.session_state.input_data = input_data
        
        # Show success message and suggest switching tabs
        st.success("✅ **Prediction complete!** Switch to 'Results' tab to view detailed analysis.")
        
        # Auto-switch to Results tab
        st.markdown('<meta http-equiv="refresh" content="2;url=#tab2">', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ **Error making prediction:** {str(e)}")
        st.error("Please check that all inputs are correctly filled.")
elif calculate and ensemble is None:
    st.error("❌ **Model not loaded.** Please check that model files are available.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p><strong>CRKP Risk Prediction Calculator v2.0</strong> | Robust Ensemble Model | For research use only</p>
    <p><strong>Methodology:</strong> Temporal validation with clinical optimization | <strong>Validation:</strong> n=7,225 patients</p>
    <p><strong>GitHub:</strong> <a href="https://github.com/maroofb88/CRKP_RISK_Calculator" target="_blank">CRKP_RISK_Calculator</a> | 
    <strong>Last Updated:</strong> January 2024</p>
</div>
""", unsafe_allow_html=True)
