# SomnoMetrics: Clinical Sleep Health Intelligence System 🩺

An end-to-end clinical decision support platform and population health intelligence system designed to screen, predict, and triage sleep disorders (**Healthy**, **Psychophysiological Insomnia**, and **Obstructive Sleep Apnea**) using biometric and behavioral indicators.

---

## 🎓 4-Month Academic Term Project Scope & Curriculum Alignment

| Curriculum Course | Implementation in Project |
| :--- | :--- |
| **Machine Learning Fundamentals** | Benchmarking 5 distinct ML families: Random Forest (Ensemble), Gradient Boosting, Decision Tree (CART), K-Nearest Neighbors (Instance-Based), and Logistic Regression (Linear). |
| **Statistical Inferencing & Optimization** | Stratified 5-Fold Cross Validation ($k=5$), Macro Recall (Clinical Sensitivity prioritization to minimize false negatives), and Confusion Matrix diagnostic analysis. |
| **Data Modeling & Visualization** | Multivariate Pearson correlation heatmaps, circadian stress-duration gradients, hemodynamic distribution curves, and occupational pathology stratification. |
| **Ethics in AI & Healthcare** | Algorithmic fairness analysis, class imbalance mitigation, clinical explainability (XAI), and medical decision-support governance disclaimers. |
| **Python Programming & Systems** | Object-oriented modular data pipeline, model serialization (`joblib`), and Streamlit clinical deployment. |

---

## 🌟 Advanced Platform Capabilities

1. **🩺 Patient Diagnostic & Explainable AI (XAI):**
   - Individual patient clinical intake across demographics, hemodynamics, and sleep hygiene.
   - Algorithmic switcher allowing evaluation across any of the 5 trained models.
   - Individualized biomarker attribution explaining *why* a diagnosis was assigned (e.g. Stage 1 hypertension, high BMI, sleep fragmentation).
   - Clinical intervention protocols (CBT-I recommendations, polysomnography referrals, CPAP triage).

2. **🏥 Batch Clinical Cohort Screening:**
   - Ingest hospital intake or enterprise employee wellness records (`.csv`).
   - Triage hundreds of patients in seconds with interactive cohort statistics.
   - High-acuity patient filtering (e.g., isolating high-risk Obstructive Sleep Apnea patients).
   - One-click export of triaged results as CSV. Includes a sample cohort generator for demonstration.

3. **📊 Population Health & Epidemiological Analytics (EDA):**
   - Circadian stress-duration gradient analysis.
   - Blood pressure distributions vs. hypertension diagnostic thresholds.
   - Occupational risk stratification (comparing nurses, doctors, engineers, and teachers).
   - Correlation matrices of physiological and behavioral markers.

4. **⚖️ Empirical Multi-Model Benchmark Matrix:**
   - Comprehensive comparative matrix across 5 models with Test Accuracy, 5-Fold Cross Validation ($\mu \pm \sigma$), Macro Precision, Macro Recall, and Macro F1-Score.
   - Feature Importance ranking isolating top physiological risk drivers (BMI, Blood Pressure, Sleep Duration).
   - Interactive Confusion Matrix inspector.

5. **📜 Academic AI Ethics & Clinical Governance:**
   - Documentation on demographic generalizability, majority class bias mitigation, and human-in-the-loop clinical boundary guidelines.

---

## 📊 Benchmark Results

| Model Architecture | Test Accuracy | 5-Fold Cross-Val | Macro Recall (Sensitivity) | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Ensemble)** | **97.33%** | **88.29% ± 3.48%** | **95.69%** | **95.59%** |
| **K-Nearest Neighbors (KNN)** | **94.67%** | **87.28% ± 4.12%** | **92.71%** | **92.65%** |
| **Decision Tree (CART)** | **93.33%** | **86.28% ± 3.85%** | **90.77%** | **90.74%** |
| **Gradient Boosting** | **93.33%** | **88.29% ± 3.10%** | **90.77%** | **90.80%** |
| **Logistic Regression (Standardized)** | **93.33%** | **87.60% ± 4.25%** | **90.49%** | **90.38%** |

---

## 🚀 Repository Structure
```
├── app.py                                 # 5-Tab clinical dashboard & decision support UI
├── train_model.py                         # Multi-model benchmarking engine & artifact pipeline
├── sleep_disorder_model.pkl               # Serialized multi-model artifacts & EDA dataset
├── Sleep_health_and_lifestyle_dataset.csv  # Standardized clinical dataset
├── requirements.txt                       # Application dependencies
├── Sleep_Health_Disorder_Prediction.ipynb # Google Colab / Jupyter demonstration notebook
└── README.md                              # Comprehensive project documentation
```

---

## 🔄 Updating Your Live Streamlit Cloud Deployment
Simply upload and overwrite the updated files in your GitHub repository:
1. `app.py`
2. `sleep_disorder_model.pkl`
3. `requirements.txt`
4. `README.md`

Streamlit Community Cloud automatically re-detects the commit and reloads the live app within 60 seconds!
