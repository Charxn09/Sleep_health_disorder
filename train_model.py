import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def main():
    print("=" * 65)
    print("SOMNOMETRICS: CLINICAL ML BENCHMARKING ENGINE")
    print("=" * 65)

    # 1. Load Data
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    print(f"Loaded {len(df)} patient clinical records.")

    # 2. Data Cleaning & Standardization
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)

    if 'Person ID' in df.columns:
        df.drop(columns=['Person ID'], inplace=True)
    df.drop(columns=['Blood Pressure'], inplace=True)

    # Cleaned dataframe preserved for EDA dashboard
    cleaned_df = df.copy()

    # 3. Categorical Encoders
    categorical_cols = ['Gender', 'Occupation', 'BMI Category']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 4. Features & Target
    X = df.drop(columns=['Sleep Disorder'])
    y = df['Sleep Disorder']
    feature_names = list(X.columns)
    target_classes = sorted(list(y.unique()))

    # 5. Train-Test Stratified Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 6. Multi-Model Benchmark Suite (5 Diverse ML Families)
    models = {
        'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'Gradient Boosting (Boosting)': GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42),
        'Decision Tree (CART)': DecisionTreeClassifier(max_depth=6, random_state=42),
        'K-Nearest Neighbors (KNN)': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        'Logistic Regression (Linear)': make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
    }

    results = []
    trained_models = {}
    confusion_matrices = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nBenchmarking Algorithm Architectures:")
    print("-" * 65)

    for name, model in models.items():
        # Stratified 5-Fold Cross Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_mean = cv_scores.mean() * 100
        cv_std = cv_scores.std() * 100

        # Fit on training partition
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

        cm = confusion_matrix(y_test, y_pred, labels=target_classes)
        confusion_matrices[name] = cm
        trained_models[name] = model

        results.append({
            'Model': name,
            'Test Accuracy (%)': round(acc, 2),
            '5-Fold CV Accuracy (%)': f"{cv_mean:.2f} ± {cv_std:.2f}",
            'Macro Precision (%)': round(prec, 2),
            'Macro Recall (Sensitivity) (%)': round(rec, 2),
            'Macro F1-Score (%)': round(f1, 2)
        })

        print(f"✓ {name:30} | Acc: {acc:.2f}% | CV: {cv_mean:.2f}% | Rec: {rec:.2f}%")

    benchmark_df = pd.DataFrame(results).sort_values(by='Test Accuracy (%)', ascending=False).reset_index(drop=True)

    # 7. Feature Importance Analysis from Random Forest
    rf_best = trained_models['Random Forest (Ensemble)']
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance (%)': (rf_best.feature_importances_ * 100).round(2)
    }).sort_values(by='Importance (%)', ascending=False).reset_index(drop=True)

    print("\nTop Predictors (Feature Importance):")
    print(feature_importance_df.head(5).to_string(index=False))

    # 8. Save Comprehensive Clinical Artifacts
    artifacts = {
        'best_model': rf_best,
        'all_models': trained_models,
        'benchmark_df': benchmark_df,
        'feature_importance_df': feature_importance_df,
        'confusion_matrices': confusion_matrices,
        'label_encoders': label_encoders,
        'feature_names': feature_names,
        'target_classes': target_classes,
        'cleaned_dataset': cleaned_df,
        'test_data': {'X_test': X_test, 'y_test': y_test}
    }

    joblib.dump(artifacts, 'sleep_disorder_model.pkl')
    print("\nSaved comprehensive multi-model artifact to 'sleep_disorder_model.pkl'!")
    print("=" * 65)

if __name__ == '__main__':
    main()
