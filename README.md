#  CRKP Risk Calculator

A web-based clinical decision support tool for predicting Carbapenem-Resistant *Klebsiella pneumoniae* (CRKP) infection risk using machine learning.

## Live Demo
**[Click here to use the app](https://your-app-name.streamlit.app/)**

## Model Performance
- **AUROC**: 0.827 (95% CI: 0.792-0.860)
- **AUPRC**: 0.685 (95% CI: 0.631-0.737)
- **Validation**: Temporal 80/20 split
- **Patients**: 7,225 total (5,780 train, 1,445 test)

## Features
- **Risk Prediction**: Calculate CRKP probability in real-time
- **Clinical Guidance**: Evidence-based recommendations
- **Visual Analytics**: Interactive risk gauge chart
- **Model Transparency**: Complete performance metrics
- **Responsive Design**: Works on all devices

## Clinical Use
### Risk Categories:
- **Low Risk (<10%)**: Routine clinical monitoring
- **Moderate Risk (10-30%)**: Enhanced monitoring
- **High Risk (>30%)**: Consider isolation precautions

### Input Features:
1. Age
2. ICU admission history
3. Current ICU location
4. Antibiotic exposure
5. Clinical risk score
6. ICU severity
7. Antibiotic intensity

## Local Installation

1. **Clone repository**:
```bash
git clone https://github.com/yourusername/CRKP_Risk_Calculator.git
cd CRKP_Risk_Calculator
