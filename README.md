# CRKP Risk Calculator

> A Machine Learning-Powered Clinical Decision Support Tool for Predicting Carbapenem-Resistant *Klebsiella pneumoniae* (CRKP) Infection Risk.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://crkpriskcalculator-e6jqlvfwytuc87z5o4g8u6.streamlit.app/)

**Live Application:** [https://crkpriskcalculator.streamlit.app/](https://crkpriskcalculator-e6jqlvfwytuc87z5o4g8u6.streamlit.app/)

---

## Project Overview

The **CRKP Risk Calculator** is an interactive web application that uses a validated XGBoost machine learning model to provide real-time, individual patient risk scores for CRKP infection. Built with **Streamlit**, it serves as a clinical decision support tool to aid healthcare professionals in risk stratification, antibiotic stewardship, and infection control planning.

This project addresses the critical need for rapid, evidence-based risk assessment in the management of antimicrobial resistance (AMR), specifically for difficult-to-treat carbapenem-resistant infections.

###  **Model Performance Highlights**
- **AUROC (Area Under the ROC Curve):** 0.827 (95% CI: 0.792–0.860)
- **AUPRC (Area Under the Precision-Recall Curve):** 0.685 (95% CI: 0.631–0.737)
- **Validation:** Temporal 80/20 split on a cohort of 7,225 patient episodes.
- **Key Predictors:** ICU admission history, antibiotic exposure intensity, clinical risk score, and patient demographics.

---

##  Key Features

*   ** Interactive Patient Assessment:** Intuitive form to input key clinical variables available before culture order time.
*   ** Real-Time Risk Prediction:** Instant calculation of CRKP probability and classification into **Low (<10%)**, **Moderate (10-30%)**, or **High (>30%)** risk categories.
*   ** Visual Risk Dashboard:** Interactive gauge chart and feature contribution plots to visualize the prediction and its driving factors.
*   ** Evidence-Based Recommendations:** Contextual clinical guidance and infection control suggestions tailored to the calculated risk level.
*   ** Full Model Transparency:** A dedicated section detailing the model's architecture, validation methodology, performance metrics, and limitations for clinical interpretability.

---

##  Quick Start & Deployment

### **Option 1: Use the Live Web App**
The easiest way to use the tool is via the hosted Streamlit Cloud application:
 **[Open the CRKP Risk Calculator](https://crkpriskcalculator-e6jqlvfwytuc87z5o4g8u6.streamlit.app/)**

### **Option 2: Run Locally**
Follow these steps to run the application on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MirzaMaroof/CRKP_RISK_Calculator.git
    cd CRKP_RISK_Calculator
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (e.g., `venv` or `conda`).
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```
4.  The application will open automatically in your default web browser (typically at `http://localhost:8501`).

---

##  Project Structure
CRKP_RISK_Calculator/
├── streamlit_app.py # Main Streamlit application script
├── requirements.txt # Python package dependencies
├── README.md # This documentation file
├── models/ # Directory for serialized ML models
│ ├── xgboost_model.pkl # Trained XGBoost model (Git LFS)
│ └── feature_names.pkl # List of features for the model

---

##  Technical Implementation

- **Frontend & Framework:** [Streamlit](https://streamlit.io/) for rapid development of the interactive web interface.
- **Core Machine Learning:** [XGBoost](https://xgboost.ai/) algorithm, chosen for its performance with structured clinical data.
- **Data Processing:** [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/).
- **Visualization:** [Plotly](https://plotly.com/python/) for interactive, publication-quality charts.
- **Model Validation:** Rigorous temporal validation strategy to ensure clinical realism and prevent data leakage.

For a complete methodological breakdown, including cohort definition, feature engineering, and statistical analysis, please refer to the `Model Information` tab within the application.

---

##  Important Disclaimer

**This tool is intended for RESEARCH and CLINICAL DECISION SUPPORT purposes only.**

- It is **NOT** a diagnostic device.
- Predictions **MUST** be interpreted by a qualified healthcare professional within the full clinical context of the patient.
- The model was developed on retrospective, single-center data. **Local validation is strongly recommended** before integration into clinical workflows.
- Always follow institutional guidelines and protocols for infection control and antimicrobial therapy.

---

## 👥 Contributing

Contributions, suggestions, and bug reports are welcome! Please feel free to open an [Issue](https://github.com/MirzaMaroof/CRKP_RISK_Calculator/issues) or submit a Pull Request.

---
