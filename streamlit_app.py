import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
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

# Load NEW robust ensemble model
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
tab1, tab2, tab3 = st.tabs(["Patient Input", "Results", "Model Info"])

with tab1:
    st.header("Patient Information Input")
    st.markdown("*All data must be available before culture order time (1-hour temporal firewall)*")
    
    input_data = {}
    
    # Demographics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Demographics")
        input_data['age'] = st.number_input("Age (years)", 0, 120, 65)
    
    # Antibiotic exposure - UPDATED FOR NEW MODEL
    with col2:
        st.subheader("Antibiotic Exposure")
        input_data['num_abx_classes_30d'] = st.number_input(
            "Number of antibiotic classes (past 30 days)", 
            0, 10, 1
        )
    
    # Medication details - NEW FEATURES
    col1, col2 = st.columns(2)
    with col1:
        recent_abx = st.checkbox("Recent antibiotic use (≤7 days)")
        input_data['recent_abx_7d'] = 1 if recent_abx else 0
        
    with col2:
        input_data['carbapenem_30d'] = st.checkbox("Carbapenem use (past 30 days)")
    
    # Hospitalization - UPDATED
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hospitalization")
        input_data['icu_admission_30d'] = st.checkbox("ICU admission (past 30d)")
        input_data['current_location_icu'] = st.checkbox("Currently in ICU")
    
    with col2:
        icu_broad = st.checkbox("ICU + Broad spectrum antibiotics")
        input_data['icu_broad_spectrum'] = 1 if icu_broad else 0
    
    # Laboratory values - SIMPLIFIED
    st.subheader("Laboratory Values (Most Recent)")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data['albumin_last'] = st.number_input("Albumin (g/L)", 10.0, 60.0, 35.0, step=0.1)
    
    # Remove unused features to avoid confusion
    with col2:
        # Placeholder for alignment
        st.write(" ")
    
    with col3:
        # Placeholder for alignment
        st.write(" ")
    
    # Calculate button
    calculate = st.button("Calculate CRKP Risk", type="primary", use_container_width=True)
    
    # Store input data in session state for use in Results tab
    if calculate:
        st.session_state.input_data = input_data
        st.session_state.calculate_clicked = True

with tab2:
    st.header("Prediction Results")
    
    # Check if calculation was performed
    if 'calculate_clicked' not in st.session_state or not st.session_state.calculate_clicked:
        st.info("Enter patient data in 'Patient Input' tab and click 'Calculate CRKP Risk' to see results")
    else:
        # Get input data from session state
        input_data = st.session_state.input_data
        
        # Perform prediction
        if ensemble is not None:
            try:
                # Prepare input data
                features = feature_info['features']
                
                # Ensure all features are present
                for feat in features:
                    if feat not in input_data:
                        # Set reasonable defaults
                        if feat in ['age', 'albumin_last', 'num_abx_classes_30d']:
                            input_data[feat] = 0
                        else:
                            input_data[feat] = 0
                
                # Create DataFrame with correct feature order
                input_df = pd.DataFrame([input_data])
                
                # Reorder columns to match model features
                for feat in features:
                    if feat not in input_df.columns:
                        input_df[feat] = 0
                
                input_df = input_df[features]
                
                # Handle missing values
                for col in input_df.columns:
                    if input_df[col].isna().any() or input_df[col].isnull().any():
                        if col == 'age':
                            input_df[col] = 65
                        elif col == 'albumin_last':
                            input_df[col] = 35.0
                        elif col == 'num_abx_classes_30d':
                            input_df[col] = 0
                        else:
                            input_df[col] = 0
                
                # Convert to numpy array
                X = input_df.values.reshape(1, -1)
                
                # Ensemble prediction function
                def ensemble_predict_proba(ensemble_model, X):
                    """Get predictions from ensemble."""
                    base_preds = []
                    for name, model in ensemble_model['base_models'].items():
                        if hasattr(model, 'predict_proba'):
                            base_preds.append(model.predict_proba(X)[:, 1])
                        else:
                            base_preds.append(model.predict(X))
                    
                    base_preds_array = np.column_stack(base_preds)
                    
                    if 'meta_model' in ensemble_model and hasattr(ensemble_model['meta_model'], 'predict_proba'):
                        return ensemble_model['meta_model'].predict_proba(base_preds_array)[:, 1]
                    else:
                        return np.mean(base_preds_array, axis=1)
                
                # Predict
                probability = ensemble_predict_proba(ensemble, X)[0]
                
                # Store in session state for this tab
                st.session_state.probability = probability
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                probability = 0.5  # Default if error
        else:
            st.error("Model not loaded")
            probability = 0.5
        
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
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CRKP Probability", f"{probability:.1%}")
        with col2:
            st.metric("Risk Category", risk_category)
        with col3:
            threshold = feature_info['optimal_threshold'] if feature_info else 0.38
            prediction = "CRKP Likely" if probability >= threshold else "CRKP Unlikely"
            st.metric("Prediction", prediction)
        
        # Gauge chart - UPDATED
        st.subheader("Risk Visualization")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "CRKP Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 10], 'color': "green", 'name': "Low"},
                    {'range': [10, 38], 'color': "yellow", 'name': "Moderate"},
                    {'range': [38, 100], 'color': "red", 'name': "High"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 38.0  # NEW optimal threshold
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Clinical recommendations - UPDATED WITH MULTIPLE SCENARIOS
        st.subheader("Clinical Recommendations")
        
        # Three scenarios
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Screening (Threshold: 0.10)**")
            if probability >= 0.10:
                st.warning("""
                **Consider CRKP:**
                • High sensitivity screening
                • Screen with cultures
                • Monitor closely
                """)
            else:
                st.success("""
                **CRKP Unlikely:**
                • Very low risk
                • Routine monitoring
                """)
        
        with col2:
            st.markdown(f"**Optimal (Threshold: {threshold:.2f})**")
            if probability >= threshold:
                st.warning("""
                **High Risk:**
                • Contact precautions
                • Consider isolation
                • Target testing
                """)
            else:
                st.success("""
                **Moderate Risk:**
                • Enhanced monitoring
                • Consider screening
                """)
        
        with col3:
            st.markdown("**Isolation (Threshold: 0.50)**")
            if probability >= 0.50:
                st.error("""
                **Very High Risk:**
                • Strict isolation
                • Empirical anti-CRKP
                • Urgent ID consult
                """)
            else:
                st.info("""
                **Manage Normally:**
                • Standard precautions
                • Routine monitoring
                """)

with tab3:
    st.header("Model Information & Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance (Updated)")
        st.markdown("""
        **Primary Metrics (Temporal Validation):**
        - AUROC: 0.698 (95% CI: 0.668-0.730)
        - AUPRC: 0.267 (95% CI: 0.237-0.312)
        - Sensitivity: 52.8% (46.9-59.3%)
        - Specificity: 68.4% (65.9-70.9%)
        - PPV: 25.7% (21.7-29.6%)
        - NPV: 87.5% (85.5-89.7%)
        - Brier Score: 0.145
        - Optimal Threshold: 0.380
        
        **Test Set (n=1,445):**
        - CRKP Prevalence: 17.2%
        - Temporal Split: 80/20 chronological
        """)
    
    with col2:
        st.subheader("Model Architecture (Updated)")
        st.markdown("""
        **Algorithm:** Robust Ensemble
        - Logistic Regression (regularized)
        - Random Forest (limited depth)
        - XGBoost (regularized)
        - Meta-model: Logistic Regression
        
        **Features:** 8 clinically meaningful predictors
        1. Age
        2. Number of antibiotic classes (30d)
        3. Recent antibiotics (≤7 days)
        4. Albumin
        5. ICU admission (30d)
        6. Current ICU location
        7. Carbapenem use (30d)
        8. ICU + Broad spectrum interaction
        
        **Training:** 5,780 patients (12.7% CRKP)
        **Testing:** 1,445 patients (17.2% CRKP)
        """)
    
    st.subheader("Feature Importance")
    st.markdown("""
    1. **Recent antibiotics (≤7 days)** - Most important predictor
    2. **Carbapenem use (past 30 days)** - Key antibiotic risk factor
    3. **Number of antibiotic classes** - Antibiotic burden
    4. **Albumin level** - Nutritional/clinical status
    5. **Age** - Demographic factor
    6. **ICU exposures** - Healthcare setting risks
    """)
    
    st.subheader("Clinical Utility")
    st.markdown("""
    **Three Operating Points:**
    1. **Screening (0.10 threshold):** Sensitivity 91.9% - Rule out CRKP
    2. **Optimal (0.38 threshold):** Balanced approach - Risk stratification
    3. **Isolation (0.50 threshold):** Specificity 96.1% - Isolation decisions
    
    **Key Strength:** High NPV (87.5%) for ruling out CRKP
    """)
    
    st.subheader("Methodology")
    st.markdown("""
    - **Temporal Validation:** 80/20 chronological split
    - **Preprocessing:** Pipeline fitted on training only
    - **Missing Data:** Median imputation + clinical logic
    - **Calibration:** Platt scaling applied
    - **Reproducibility:** Frozen dataset with hash verification
    - **Ethics:** IRB approved, data anonymized
    """)
    
    st.subheader("Limitations")
    st.markdown("""
    1. Retrospective design: Subject to biases of observational data
    2. Single center: External validation needed
    3. Prevalence sensitivity: Performance varies with local prevalence
    4. Research tool: Not yet prospectively validated
    5. Clinical judgment: Should supplement, not replace clinical assessment
    """)

# Footer
st.markdown("---")
st.markdown("**CRKP Prediction Calculator v2.0** | Robust Ensemble Model | For research use only")
st.markdown("**GitHub Repository:** https://github.com/maroofb88/CRKP_RISK_Calculator")
st.markdown("**Methodology:** Temporal validation with clinical optimization | **Last Updated:** January 2024")
