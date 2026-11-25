import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io

# ---------------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="SomnoMetrics | Clinical Sleep Disorder Intelligence System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic, clinical presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-term {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .card-risk-none {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 16px;
        border-radius: 8px;
    }
    .card-risk-insomnia {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 16px;
        border-radius: 8px;
    }
    .card-risk-apnea {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 16px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model & Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_clinical_artifacts():
    return joblib.load('sleep_disorder_model.pkl')

try:
    artifacts = load_clinical_artifacts()
    all_models = artifacts.get('all_models', {'Random Forest': artifacts.get('best_model', artifacts.get('model'))})
    best_model = artifacts.get('best_model', artifacts.get('model'))
    benchmark_df = artifacts.get('benchmark_df', pd.DataFrame())
    feature_importance_df = artifacts.get('feature_importance_df', pd.DataFrame())
    confusion_matrices = artifacts.get('confusion_matrices', {})
    label_encoders = artifacts['label_encoders']
    feature_names = artifacts['feature_names']
    target_classes = artifacts.get('target_classes', ['Insomnia', 'None', 'Sleep Apnea'])
    cleaned_df = artifacts.get('cleaned_dataset', pd.DataFrame())
except Exception as e:
    st.error(f"Error loading system artifacts: {e}. Please ensure 'sleep_disorder_model.pkl' is generated.")
    st.stop()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/medical-doctor.png", width=120)
    st.markdown("### **SomnoMetrics 2.0**")
    st.markdown('<span class="badge-term">Academic Term Project | 4-Month Scope</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("#### ⚙️ Inference Engine")
    selected_model_name = st.selectbox(
        "Active ML Algorithm",
        options=list(all_models.keys()),
        index=0,
        help="Select which trained algorithm executes predictions across individual and batch modules."
    )
    current_model = all_models[selected_model_name]

    st.markdown("---")
    st.markdown("#### 📚 Curriculum Alignment")
    st.markdown("""
    - **Machine Learning Fundamentals**
    - **Data Modeling & Visualization**
    - **Statistical Inferencing**
    - **Ethics in AI & Healthcare**
    - **Python Data Systems**
    """)
    st.markdown("---")
    st.caption("Clinical Decision Support Prototype • v2.4")

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown('<div class="main-title">🩺 SomnoMetrics: Clinical Sleep Health Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A multi-model diagnostic decision support and population health analytics system for sleep disorders (Insomnia & Sleep Apnea).</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs (5-Pillar Architecture)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Patient Diagnostic & XAI",
    "🏥 Batch Clinical Screening",
    "📊 Population Health & EDA",
    "⚖️ Multi-Model Benchmarks",
    "📜 AI Ethics & Methodology"
])

# =========================================================
# TAB 1: INDIVIDUAL PATIENT DIAGNOSIS & EXPLAINABLE AI
# =========================================================
with tab1:
    st.subheader("👤 Individual Patient Clinical Intake")
    st.caption(f"Currently evaluated using: **{selected_model_name}**")

    with st.form("patient_intake_form"):
        col_dem, col_cardio, col_sleep = st.columns(3)

        with col_dem:
            st.markdown("##### 🧬 Patient Demographics")
            gender = st.selectbox("Biological Sex", options=label_encoders['Gender'].classes_)
            age = st.slider("Patient Age (Years)", min_value=18, max_value=85, value=35)
            occupation = st.selectbox("Primary Occupation", options=label_encoders['Occupation'].classes_)
            bmi_cat = st.selectbox("BMI Classification", options=label_encoders['BMI Category'].classes_)

        with col_cardio:
            st.markdown("##### ❤️ Hemodynamic Metrics")
            sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=90, max_value=200, value=124, step=1)
            dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=55, max_value=130, value=82, step=1)
            resting_hr = st.slider("Resting Heart Rate (BPM)", min_value=50, max_value=120, value=72)
            daily_steps = st.number_input("Average Daily Step Count", min_value=1000, max_value=25000, value=6500, step=500)

        with col_sleep:
            st.markdown("##### 🌙 Sleep & Neuro-Behavioral")
            sleep_hours = st.slider("Daily Sleep Duration (Hours)", min_value=3.0, max_value=12.0, value=6.8, step=0.1)
            sleep_quality = st.slider("Subjective Sleep Quality (1-10)", min_value=1, max_value=10, value=6)
            stress = st.slider("Perceived Daily Stress Level (1-10)", min_value=1, max_value=10, value=6)
            physical_mins = st.slider("Physical Activity (Minutes/Day)", min_value=10, max_value=120, value=40, step=5)

        submit_btn = st.form_submit_button("⚡ Run Clinical Assessment", type="primary", use_container_width=True)

    if submit_btn:
        patient_vector = {
            'Gender': label_encoders['Gender'].transform([gender])[0],
            'Age': age,
            'Occupation': label_encoders['Occupation'].transform([occupation])[0],
            'Sleep Duration': sleep_hours,
            'Quality of Sleep': sleep_quality,
            'Physical Activity Level': physical_mins,
            'Stress Level': stress,
            'BMI Category': label_encoders['BMI Category'].transform([bmi_cat])[0],
            'Heart Rate': resting_hr,
            'Daily Steps': daily_steps,
            'Systolic_BP': sys_bp,
            'Diastolic_BP': dia_bp
        }

        input_df = pd.DataFrame([patient_vector])[feature_names]
        predicted_class = current_model.predict(input_df)[0]
        
        # Probabilities if available
        has_proba = hasattr(current_model, "predict_proba")
        if has_proba:
            probabilities = current_model.predict_proba(input_df)[0]
            prob_dict = dict(zip(current_model.classes_, (probabilities * 100).round(1)))
        else:
            prob_dict = {c: "N/A" for c in target_classes}

        st.markdown("---")
        st.subheader("📋 Diagnostic Output & Clinical Decision Support")

        res_left, res_right = st.columns([1.2, 0.8])

        with res_left:
            if predicted_class == 'None':
                st.markdown(f"""
                <div class="card-risk-none">
                    <h3 style="color:#065F46; margin:0 0 8px 0;">✅ Diagnostic Impression: Healthy Sleep Profile (No Disorder)</h3>
                    <p style="margin:0; color:#047857;">The patient exhibits normative cardiorespiratory and neuro-behavioral parameters. No high-acuity indicators of Obstructive Sleep Apnea or Psychophysiological Insomnia were detected.</p>
                </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Insomnia':
                st.markdown(f"""
                <div class="card-risk-insomnia">
                    <h3 style="color:#92400E; margin:0 0 8px 0;">⚠️ Diagnostic Impression: Suspected Psychophysiological Insomnia</h3>
                    <p style="margin:0; color:#B45309;">The predictive profile indicates elevated sleep latency and fragmentation. Primary correlative factors include high daily stress and suboptimal subjective sleep duration.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card-risk-apnea">
                    <h3 style="color:#991B1B; margin:0 0 8px 0;">🚨 Diagnostic Impression: High Risk for Obstructive Sleep Apnea (OSA)</h3>
                    <p style="margin:0; color:#B91C1C;">Elevated risk markers detected. Elevated systolic/diastolic blood pressure coupled with BMI metrics strongly correlate with nocturnal respiratory resistance and hypoxic events.</p>
                </div>
                """, unsafe_allow_html=True)

            # Explainable AI (XAI) Rule-Based Attribution
            st.markdown("#### 🔍 Explainable AI (XAI): Individual Patient Biomarkers")
            risk_factors = []
            positive_factors = []

            if sys_bp >= 130 or dia_bp >= 85:
                risk_factors.append(f"Elevated Blood Pressure ({sys_bp}/{dia_bp} mmHg exceeds AHA normal thresholds).")
            else:
                positive_factors.append(f"Normotensive Blood Pressure ({sys_bp}/{dia_bp} mmHg).")

            if bmi_cat in ['Overweight', 'Obese']:
                risk_factors.append(f"Elevated BMI classification ({bmi_cat}) increases pharyngeal airway collapse risk.")
            else:
                positive_factors.append(f"Normal BMI status ({bmi_cat}).")

            if stress >= 7:
                risk_factors.append(f"High perceived stress index ({stress}/10) linked to sympathetic nervous system hyperarousal.")
            
            if sleep_hours < 6.0:
                risk_factors.append(f"Insufficient sleep duration ({sleep_hours} hrs/day) correlates with chronic sleep debt.")
            elif sleep_hours >= 7.0:
                positive_factors.append(f"Adequate nocturnal sleep window ({sleep_hours} hrs/day).")

            if risk_factors:
                st.markdown("**Identified Clinical Risk Contributors:**")
                for rf in risk_factors:
                    st.markdown(f"- 🔴 {rf}")
            if positive_factors:
                st.markdown("**Protective / Healthy Factors:**")
                for pf in positive_factors:
                    st.markdown(f"- 🟢 {pf}")

        with res_right:
            st.markdown("#### 📊 Posterior Probability Distribution")
            if has_proba:
                prob_chart_df = pd.DataFrame(list(prob_dict.items()), columns=['Class', 'Probability (%)'])
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.barplot(data=prob_chart_df, x='Probability (%)', y='Class', palette='viridis', ax=ax)
                ax.set_xlim(0, 100)
                for p in ax.patches:
                    width = p.get_width()
                    ax.annotate(f"{width:.1f}%", (width + 1.5, p.get_y() + p.get_height() / 2.), va='center')
                sns.despine(top=True, right=True)
                st.pyplot(fig)
            else:
                st.info("Class probabilities not supported by the selected kernel.")

            # Clinical recommendations
            st.markdown("#### 💡 Clinical Recommendation Protocol")
            if predicted_class == 'None':
                st.write("1. Maintain standard circadian hygiene.\n2. Continue routine aerobic cardiovascular exercise.")
            elif predicted_class == 'Insomnia':
                st.write("1. Initiate Cognitive Behavioral Therapy for Insomnia (CBT-I).\n2. Restrict blue-light emission 90 minutes before sleep.\n3. Integrate mindfulness or progressive muscle relaxation.")
            else:
                st.write("1. Urgent referral for overnight Polysomnography (Sleep Study).\n2. Continuous Positive Airway Pressure (CPAP) clinical evaluation.\n3. Nutritional consultation for weight management.")

# =========================================================
# TAB 2: BATCH HOSPITAL SCREENING (CSV PIPELINE)
# =========================================================
with tab2:
    st.subheader("🏥 Multi-Patient Clinical Batch Screening")
    st.markdown("Upload hospital intake or enterprise employee wellness records (`.csv`) to screen cohorts simultaneously.")

    # Sample batch download helper
    if not cleaned_df.empty:
        sample_batch = cleaned_df.head(15).drop(columns=['Sleep Disorder', 'Systolic_BP', 'Diastolic_BP'], errors='ignore')
        # recreate Blood Pressure format
        if 'Blood Pressure' not in sample_batch.columns:
            sample_batch['Blood Pressure'] = cleaned_df.head(15)['Systolic_BP'].astype(str) + "/" + cleaned_df.head(15)['Diastolic_BP'].astype(str)
        csv_buffer = io.StringIO()
        sample_batch.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Sample Patient Cohort CSV (For Testing)",
            data=csv_buffer.getvalue(),
            file_name="sample_patient_cohort.csv",
            mime="text/csv",
            help="Download a ready-to-test CSV with 15 realistic patient records."
        )

    uploaded_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df)}** records for cohort triage.")

            # Preprocessing Pipeline
            proc_df = batch_df.copy()
            if 'Blood Pressure' in proc_df.columns and ('Systolic_BP' not in proc_df.columns):
                proc_df[['Systolic_BP', 'Diastolic_BP']] = proc_df['Blood Pressure'].str.split('/', expand=True).astype(int)
            
            if 'BMI Category' in proc_df.columns:
                proc_df['BMI Category'] = proc_df['BMI Category'].replace('Normal Weight', 'Normal')

            # Encode
            for col in ['Gender', 'Occupation', 'BMI Category']:
                if col in proc_df.columns:
                    le = label_encoders[col]
                    # handle unseen categories safely
                    proc_df[col] = proc_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)

            # Extract features
            X_batch = proc_df[feature_names]
            predictions = current_model.predict(X_batch)
            batch_df['Predicted_Diagnosis'] = predictions

            st.markdown("---")
            st.markdown("### 📊 Cohort Screening Summary")

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            total_pts = len(batch_df)
            none_count = (predictions == 'None').sum()
            insomnia_count = (predictions == 'Insomnia').sum()
            apnea_count = (predictions == 'Sleep Apnea').sum()

            col_m1.metric("Total Patients Screened", total_pts)
            col_m2.metric("Healthy Patterns", f"{none_count} ({none_count/total_pts*100:.1f}%)")
            col_m3.metric("Insomnia Cohort", f"{insomnia_count} ({insomnia_count/total_pts*100:.1f}%)")
            col_m4.metric("Sleep Apnea Cohort (High Priority)", f"{apnea_count} ({apnea_count/total_pts*100:.1f}%)")

            # Cohort distribution chart
            fig_cohort, ax_c = plt.subplots(figsize=(6, 2.5))
            diag_counts = pd.Series(predictions).value_counts()
            sns.barplot(x=diag_counts.values, y=diag_counts.index, palette='Set2', ax=ax_c)
            ax_c.set_title("Cohort Diagnosis Proportions")
            sns.despine(top=True, right=True)
            st.pyplot(fig_cohort)

            # Filterable Table
            st.markdown("#### 🔍 Filter Triaged Patients")
            filter_choice = st.selectbox("Filter by Clinical Category", options=['All Patients', 'Sleep Apnea (High Acuity)', 'Insomnia', 'Healthy'])
            if filter_choice == 'Sleep Apnea (High Acuity)':
                filtered_df = batch_df[batch_df['Predicted_Diagnosis'] == 'Sleep Apnea']
            elif filter_choice == 'Insomnia':
                filtered_df = batch_df[batch_df['Predicted_Diagnosis'] == 'Insomnia']
            elif filter_choice == 'Healthy':
                filtered_df = batch_df[batch_df['Predicted_Diagnosis'] == 'None']
            else:
                filtered_df = batch_df

            st.dataframe(filtered_df, use_container_width=True)

            # Export Button
            out_csv = io.StringIO()
            batch_df.to_csv(out_csv, index=False)
            st.download_button(
                label="📥 Export Triage Report as CSV",
                data=out_csv.getvalue(),
                file_name="cohort_triage_results.csv",
                mime="text/csv",
                type="primary"
            )
        except Exception as e:
            st.error(f"Error processing batch file: {e}. Please ensure the CSV contains standard columns matching the training dataset.")

# =========================================================
# TAB 3: POPULATION HEALTH & INTERACTIVE EDA
# =========================================================
with tab3:
    st.subheader("📊 Population Health & Epidemiological Analytics")
    st.caption("Investigating demographic and physiological determinants of sleep pathology across 374 clinical cohort samples.")

    if not cleaned_df.empty:
        col_eda1, col_eda2 = st.columns(2)

        with col_eda1:
            st.markdown("##### 1. Stress vs. Sleep Duration Correlation")
            fig1, ax1 = plt.subplots(figsize=(7, 4.5))
            sns.scatterplot(
                data=cleaned_df,
                x='Sleep Duration',
                y='Stress Level',
                hue='Sleep Disorder',
                palette={'None': '#10B981', 'Insomnia': '#F59E0B', 'Sleep Apnea': '#EF4444'},
                s=70,
                alpha=0.8,
                ax=ax1
            )
            ax1.set_title("Circadian Stress-Duration Gradient", fontweight='bold')
            sns.despine(top=True, right=True)
            st.pyplot(fig1)
            st.caption("Patients with Insomnia cluster tightly in low sleep duration (<6.5h) and high stress (6-8).")

        with col_eda2:
            st.markdown("##### 2. Hemodynamic Impact: Blood Pressure by Disorder")
            fig2, ax2 = plt.subplots(figsize=(7, 4.5))
            sns.boxplot(
                data=cleaned_df,
                x='Sleep Disorder',
                y='Systolic_BP',
                palette='Set2',
                ax=ax2
            )
            ax2.axhline(130, color='red', linestyle='--', label='Hypertension Threshold (130 mmHg)')
            ax2.set_title("Systolic Blood Pressure Distribution", fontweight='bold')
            ax2.legend(loc='upper right')
            sns.despine(top=True, right=True)
            st.pyplot(fig2)
            st.caption("Patients with Sleep Apnea consistently manifest systolic pressures exceeding clinical hypertension thresholds.")

        col_eda3, col_eda4 = st.columns(2)

        with col_eda3:
            st.markdown("##### 3. Disorder Prevalence Across Occupational Strata")
            fig3, ax3 = plt.subplots(figsize=(8, 4.5))
            occ_ct = pd.crosstab(cleaned_df['Occupation'], cleaned_df['Sleep Disorder'])
            occ_ct.plot(kind='barh', stacked=True, color=['#F59E0B', '#10B981', '#EF4444'], ax=ax3)
            ax3.set_title("Occupational Sleep Pathology Stratification", fontweight='bold')
            ax3.set_xlabel("Patient Count")
            sns.despine(top=True, right=True)
            st.pyplot(fig3)
            st.caption("Shift workers and high-stress careers (Nurses, Sales Representatives) exhibit high susceptibility to Apnea and Insomnia.")

        with col_eda4:
            st.markdown("##### 4. Hemodynamic & Lifestyle Correlation Matrix")
            fig4, ax4 = plt.subplots(figsize=(7, 4.5))
            num_cols = cleaned_df.select_dtypes(include=np.number).columns
            corr_matrix = cleaned_df[num_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='Blues', ax=ax4, cbar=False)
            ax4.set_title("Multivariate Pearson Correlation Heatmap", fontweight='bold')
            st.pyplot(fig4)
            st.caption("Strong negative correlation between Stress Level and Quality of Sleep (-0.90).")

# =========================================================
# TAB 4: MULTI-MODEL BENCHMARKS & EVALUATION
# =========================================================
with tab4:
    st.subheader("⚖️ Comparative Machine Learning Benchmark Matrix")
    st.markdown("""
    Rigorous empirical evaluation comparing 5 candidate classification algorithms across **Accuracy, 5-Fold Stratified Cross-Validation, Macro Precision, Macro Recall (Clinical Sensitivity), and Macro F1-Score**.
    """)

    if not benchmark_df.empty:
        st.dataframe(benchmark_df, use_container_width=True)

        col_bm1, col_bm2 = st.columns(2)

        with col_bm1:
            st.markdown("##### 🏆 Model Metric Comparison")
            fig_bm, ax_bm = plt.subplots(figsize=(7, 4))
            bm_plot_df = benchmark_df.set_index('Model')[['Test Accuracy (%)', 'Macro Recall (%)', 'Macro F1-Score (%)']]
            bm_plot_df.plot(kind='bar', ax=ax_bm, colormap='Spectral')
            ax_bm.set_ylim(80, 105)
            ax_bm.set_ylabel("Metric Score (%)")
            plt.xticks(rotation=25, ha='right')
            ax_bm.set_title("Cross-Algorithm Clinical Performance", fontweight='bold')
            sns.despine(top=True, right=True)
            st.pyplot(fig_bm)

        with col_bm2:
            st.markdown("##### 🌲 Random Forest Top Feature Importance")
            if not feature_importance_df.empty:
                fig_feat, ax_feat = plt.subplots(figsize=(7, 4))
                sns.barplot(
                    data=feature_importance_df.head(8),
                    x='Importance (%)',
                    y='Feature',
                    palette='mako',
                    ax=ax_feat
                )
                ax_feat.set_title("Top Physiological & Behavioral Predictors", fontweight='bold')
                sns.despine(top=True, right=True)
                st.pyplot(fig_feat)

        st.markdown("---")
        st.markdown("##### 🔬 Confusion Matrix Inspector")
        sel_cm_model = st.selectbox("Inspect Confusion Matrix for Model:", options=list(confusion_matrices.keys()))
        cm_data = confusion_matrices.get(sel_cm_model)
        if cm_data is not None:
            fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5))
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_classes, yticklabels=target_classes, ax=ax_cm)
            ax_cm.set_xlabel("Predicted Clinical Category")
            ax_cm.set_ylabel("True Gold Standard Category")
            ax_cm.set_title(f"Confusion Matrix: {sel_cm_model}", fontweight='bold')
            st.pyplot(fig_cm)

        st.markdown("""
        > **Clinical Insight on Metrics:** In medical diagnostics, **Macro Recall (Sensitivity)** is prioritized over raw accuracy. A False Negative (missing a Sleep Apnea patient) exposes the patient to severe unmanaged cardiovascular risks, whereas a False Positive simply results in secondary polysomnography screening.
        """)

# =========================================================
# TAB 5: AI ETHICS & METHODOLOGY
# =========================================================
with tab5:
    st.subheader("📜 Academic Methodology, Data Governance & AI Ethics")
    st.markdown("""
    This section aligns with the **Ethics in AI** and **Data Science Governance** subjects in the curriculum.
    """)

    st.markdown("""
    ### 1. Dataset Provenance & Cohort Composition
    - **Cohort Volume:** 374 adult patient entries with comprehensive biometric tracking.
    - **Demographic Split:** Balanced biological sex distribution (Male/Female).
    - **Multivariate Vitals:** Combines ambulatory step activity, wearable heart rate monitors, cuff sphygmomanometer blood pressure, and clinical sleep logs.

    ---

    ### 2. Algorithmic Fairness & Bias Mitigation
    - **Stratified Sampling:** All cross-validation and train-test splits employed Stratified K-Fold logic to preserve clinical prevalence ratios and mitigate majority-class skew.
    - **Occupational Generalizability:** The dataset exhibits clustering around specific professions (e.g., Doctors, Nurses, Engineers). In production deployment, domain adaptation techniques should be applied to prevent occupational profiling.

    ---

    ### 3. Explainability & Human-in-the-Loop AI
    - Automated machine learning outputs must function strictly as **Decision Support Systems (DSS)**, never autonomous clinical diagnostic agents.
    - Feature attribution (systolic/diastolic pressure, BMI classification, sleep duration) is exposed transparently to the attending physician to facilitate clinical validation.

    ---

    ### 4. Regulatory & Clinical Disclaimers
    > **Academic Project Notice:** This software is developed for academic demonstration and term evaluation purposes. It is not cleared by the FDA or CE as software-as-a-medical-device (SaMD). Patients exhibiting chronic daytime sleepiness, nocturnal gasping, or persistent sleep onset latency must consult a board-certified sleep specialist.
    """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center><small>SomnoMetrics Academic Term Project | Machine Learning & Clinical Decision Support Systems</small></center>",
    unsafe_allow_html=True
)
