# CRKP Risk Calculator - Robust Ensemble Model

## 📋 Overview
Web-based calculator for predicting Carbapenem-Resistant *Klebsiella pneumoniae* (CRKP) risk using a robust ensemble machine learning model validated on 7,225 patient episodes.

## 🚀 Live Deployment
**https://crkpriskcalculator-imxxo3pe7fv3cmuxjyrnet.streamlit.app/**

## 🎯 Key Features
- **Robust Ensemble Model**: Logistic Regression + Random Forest + XGBoost
- **8 Clinically Meaningful Features**: Simplified for clinical usability
- **Three Clinical Scenarios**: Screening (0.10), Optimal (0.38), Isolation (0.50) thresholds
- **Temporal Validation**: 80/20 chronological split (real-world performance)
- **High NPV**: 87.5% negative predictive value for ruling out CRKP
- **Clinical Optimization**: Model optimized for clinical utility, not just statistical performance

## 📊 Model Performance
| Metric | Value | 95% Confidence Interval |
|--------|-------|-------------------------|
| **AUROC** | 0.698 | 0.668 - 0.730 |
| **Sensitivity** | 52.8% | 46.9% - 59.3% |
| **Specificity** | 68.4% | 65.9% - 70.9% |
| **PPV** | 25.7% | 21.7% - 29.6% |
| **NPV** | 87.5% | 85.5% - 89.7% |
| **Brier Score** | 0.145 | - |
| **Optimal Threshold** | 0.380 | - |

## 🏥 Input Features (8 Total)
1. **Age** (years)
2. **Number of antibiotic classes** (past 30 days)
3. **Recent antibiotics** (≤7 days) - *Most important predictor*
4. **Albumin** (g/L)
5. **ICU admission** (past 30 days)
6. **Current ICU location**
7. **Carbapenem use** (past 30 days) - *Key antibiotic risk factor*
8. **ICU + Broad spectrum interaction**

## 🎯 Clinical Scenarios
### **1. Screening Scenario (Threshold: 0.10)**
- **Sensitivity**: 91.9%
- **Specificity**: 30.2%
- **Goal**: Maximize sensitivity to rule out CRKP
- **Use**: Initial screening, minimize missed cases

### **2. Optimal Scenario (Threshold: 0.38)**
- **Sensitivity**: 52.8%
- **Specificity**: 68.4%
- **Goal**: Balance sensitivity and specificity
- **Use**: General risk stratification, targeted testing

### **3. Isolation Scenario (Threshold: 0.50)**
- **Sensitivity**: 20.2%
- **Specificity**: 96.1%
- **Goal**: Minimize false alarms for isolation
- **Use**: Isolation decisions, resource allocation

## 🚀 Quick Start
### Local Deployment
```bash
# 1. Clone repository
git clone https://github.com/maroofb88/CRKP_RISK_Calculator.git
cd CRKP_RISK_Calculator/web_calculator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run streamlit_app.py
