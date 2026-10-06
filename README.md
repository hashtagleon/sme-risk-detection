# SME Credit Risk Detection & Underwriting System

An automated machine learning system for SME credit risk assessment, default prediction, and underwriting decision guidance.

## 🔗 Google Colab Notebook
You can run and retrain the machine learning model directly on Google Colab:
- **Colab Link:** [https://colab.research.google.com/drive/15VWt3_JMnwcNj3xKY0xRSH4c-5-Jr0DC](https://colab.research.google.com/drive/15VWt3_JMnwcNj3xKY0xRSH4c-5-Jr0DC)
- **Local Notebook File:** [`sme_risk_model_training.ipynb`](file:///c:/Users/LEON/Downloads/sme-risk-deployment/sme_risk_model_training.ipynb)

---

## 🛠️ Project Structure
- [app.py](file:///c:/Users/LEON/Downloads/sme-risk-deployment/app.py) — Flask Web Application, REST API & Vectorized Batch CSV Inference Engine
- [templates/index.html](file:///c:/Users/LEON/Downloads/sme-risk-deployment/templates/index.html) — Interactive Web UI, Credit Memo & Batch Testing Dashboard
- [sme_risk_model_training.ipynb](file:///c:/Users/LEON/Downloads/sme-risk-deployment/sme_risk_model_training.ipynb) — Model Training & Evaluation Notebook (Google Colab Ready)
- `sme_risk_model.pkl` — Trained Random Forest Classifier
- `feature_columns.pkl` — One-Hot Encoded Feature Column Alignments
- `districts.pkl`, `location_types.pkl`, `sectors.pkl` — Metadata lists for categorical variables
