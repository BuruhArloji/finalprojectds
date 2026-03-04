import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Test Result Predictor",
    page_icon="🏥",
    layout="wide",
)

# ─── Constants ───────────────────────────────────────────────────────────────────
MODEL_PATH = "model.joblib"
DATA_PATH = "healthcare_dataset.csv"

SCORING_MAP = {
    'Cancer': 10,
    'Obesity': 7,
    'Diabetes': 6,
    'Hypertension': 4,
    'Asthma': 3,
    'Arthritis': 2,
}

CAT_FEATURES = [
    'Gender', 'Blood Type', 'Medical Condition', 'Doctor', 'Hospital',
    'Insurance Provider', 'Admission Type', 'Medication',
]

# ─── Helper Functions ─────────────────────────────────────────────────────────────
def categorize_los(days):
    if 1 <= days <= 3:
        return 'A (1-3 days)'
    elif 4 <= days <= 14:
        return 'B (4-14 days)'
    elif days >= 15:
        return 'C (>=15 days)'
    return 'B (4-14 days)'  # fallback


def preprocess_df(df):
    df = df.copy()
    df = df[df['Billing Amount'] >= 0]
    df.drop_duplicates(inplace=True)

    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Days_In_Hospital'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
    df['Cost_Per_Day'] = df['Billing Amount'] / df['Days_In_Hospital'].replace(0, 1)
    df['LOS Category'] = df['Days_In_Hospital'].apply(categorize_los)
    df['Surgical_Risk_Score'] = df['Medical Condition'].map(SCORING_MAP)
    df['Length_of_Stay'] = df['Days_In_Hospital']
    df['Cost_Intensity'] = df['Surgical_Risk_Score'] * df['Length_of_Stay']

    bins = [0, 14, 24, 65, df['Age'].max() + 1]
    labels = ['Children (00-14 years)', 'Youth (15-24 years)',
              'Adults (25-64 years)', 'Seniors (65 years and over)']
    df['Age Category'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
    return df


@st.cache_resource(show_spinner="⏳ Training model, please wait...")
def load_or_train_model():
    """Train model from CSV and cache it in session."""
    df_raw = pd.read_csv(DATA_PATH)
    df = preprocess_df(df_raw)

    target = 'Test Results'
    le = LabelEncoder()
    y = le.fit_transform(df[target])
    X = df.drop(columns=[target, 'Name', 'Date of Admission', 'Discharge Date', 'Room Number'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Encode categoricals
    t_enc = TargetEncoder(target_type='multiclass', random_state=42)
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    X_train_cat = t_enc.fit_transform(X_train[CAT_FEATURES], y_train)
    X_test_cat = t_enc.transform(X_test[CAT_FEATURES])

    n_classes = len(le.classes_)
    encoded_col_names = [f'{f}_encoded_class_{i}' for f in CAT_FEATURES for i in range(n_classes)]

    X_train_enc = pd.concat([
        X_train_enc.drop(columns=CAT_FEATURES),
        pd.DataFrame(X_train_cat, columns=encoded_col_names, index=X_train_enc.index)
    ], axis=1)
    X_test_enc = pd.concat([
        X_test_enc.drop(columns=CAT_FEATURES),
        pd.DataFrame(X_test_cat, columns=encoded_col_names, index=X_test_enc.index)
    ], axis=1)

    le_los = LabelEncoder()
    X_train_enc['LOS Category'] = le_los.fit_transform(X_train_enc['LOS Category'])
    X_test_enc['LOS Category'] = le_los.transform(X_test_enc['LOS Category'])

    le_age = LabelEncoder()
    X_train_enc['Age Category'] = le_age.fit_transform(X_train_enc['Age Category'])
    X_test_enc['Age Category'] = le_age.transform(X_test_enc['Age Category'])

    # Tuned best params from notebook
    tuned_stack = StackingClassifier(
        estimators=[
            ('xgb', XGBClassifier(learning_rate=0.05, max_depth=5, n_estimators=50, random_state=42)),
            ('rf', RandomForestClassifier(max_depth=15, min_samples_split=2, n_estimators=100, random_state=42)),
            ('hgb', HistGradientBoostingClassifier(learning_rate=0.01, max_depth=5, max_iter=100, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=False,
    )

    tuned_stack.fit(X_train_enc, y_train)

    return {
        'model': tuned_stack,
        'le': le,
        'le_los': le_los,
        'le_age': le_age,
        't_enc': t_enc,
        'feature_columns': X_train_enc.columns.tolist(),
        'encoded_col_names': encoded_col_names,
        'n_classes': n_classes,
    }


def predict_single(artifacts, input_dict):
    model     = artifacts['model']
    le        = artifacts['le']
    le_los    = artifacts['le_los']
    le_age    = artifacts['le_age']
    t_enc     = artifacts['t_enc']
    feat_cols = artifacts['feature_columns']
    enc_cols  = artifacts['encoded_col_names']

    sample = pd.DataFrame([input_dict])
    sample['Date of Admission'] = pd.to_datetime(sample['Date of Admission'])
    sample['Discharge Date'] = pd.to_datetime(sample['Discharge Date'])
    sample['Days_In_Hospital'] = (sample['Discharge Date'] - sample['Date of Admission']).dt.days
    sample['Cost_Per_Day'] = sample['Billing Amount'] / sample['Days_In_Hospital'].replace(0, 1)
    sample['LOS Category'] = sample['Days_In_Hospital'].apply(categorize_los)
    sample['LOS Category'] = le_los.transform(sample['LOS Category'])
    sample['Surgical_Risk_Score'] = sample['Medical Condition'].map(SCORING_MAP).fillna(0)
    sample['Length_of_Stay'] = sample['Days_In_Hospital']
    sample['Cost_Intensity'] = sample['Surgical_Risk_Score'] * sample['Length_of_Stay']

    bins   = [0, 14, 24, 65, 120]
    labels = ['Children (00-14 years)', 'Youth (15-24 years)',
              'Adults (25-64 years)', 'Seniors (65 years and over)']
    sample['Age Category'] = pd.cut(sample['Age'], bins=bins, labels=labels, include_lowest=True)
    sample['Age Category'] = le_age.transform(sample['Age Category'])

    cat_enc = t_enc.transform(sample[CAT_FEATURES])
    cat_df  = pd.DataFrame(cat_enc, columns=enc_cols, index=sample.index)

    numerical_cols = [
        'Age', 'Billing Amount', 'Days_In_Hospital', 'Cost_Per_Day',
        'LOS Category', 'Surgical_Risk_Score', 'Length_of_Stay',
        'Cost_Intensity', 'Age Category',
    ]

    final = pd.DataFrame(index=sample.index, columns=feat_cols)
    for col in numerical_cols:
        final[col] = sample[col].values
    for col in enc_cols:
        final[col] = cat_df[col].values

    final = final.astype(float)

    pred_encoded = model.predict(final)
    pred_proba   = model.predict_proba(final)
    pred_label   = le.inverse_transform(pred_encoded)[0]

    return pred_label, pred_proba[0], le.classes_


# ─── UI ──────────────────────────────────────────────────────────────────────────
st.title("🏥 Healthcare Test Result Predictor")
st.markdown(
    "Predict patient **Test Results** (Normal / Abnormal / Inconclusive) "
    "using a tuned Stacking Classifier (XGBoost + RandomForest + HistGradientBoosting)."
)

# Check data file
if not os.path.exists(DATA_PATH):
    st.error(
        f"Dataset `{DATA_PATH}` not found. "
        "Please place **healthcare_dataset.csv** in the same directory as `app.py`."
    )
    st.stop()

# Load / train model
artifacts = load_or_train_model()
st.success("✅ Model ready!")

# ─── Input Form ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    medical_condition = st.selectbox(
        "Medical Condition",
        ["Cancer", "Obesity", "Diabetes", "Hypertension", "Asthma", "Arthritis"]
    )
    admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])

with col2:
    date_admission = st.date_input("Date of Admission")
    date_discharge = st.date_input("Discharge Date")
    billing_amount = st.number_input("Billing Amount (USD)", min_value=0.0, value=25000.0, step=100.0)
    room_number = st.number_input("Room Number", min_value=1, max_value=1000, value=101)

with col3:
    insurance_provider = st.selectbox(
        "Insurance Provider",
        ["Medicare", "Aetna", "Blue Cross", "Cigna", "UnitedHealthcare"]
    )
    medication = st.selectbox(
        "Medication",
        ["Aspirin", "Ibuprofen", "Paracetamol", "Lipitor", "Metformin", "Penicillin"]
    )
    doctor = st.text_input("Doctor Name", value="Dr. Smith")
    hospital = st.text_input("Hospital Name", value="City General Hospital")

# ─── Predict ─────────────────────────────────────────────────────────────────────
st.divider()

if st.button("🔍 Predict Test Result", use_container_width=True, type="primary"):
    if date_discharge < date_admission:
        st.error("❌ Discharge Date cannot be before Date of Admission.")
    else:
        input_data = {
            'Age': age,
            'Gender': gender,
            'Blood Type': blood_type,
            'Medical Condition': medical_condition,
            'Date of Admission': str(date_admission),
            'Doctor': doctor,
            'Hospital': hospital,
            'Insurance Provider': insurance_provider,
            'Billing Amount': billing_amount,
            'Room Number': room_number,
            'Admission Type': admission_type,
            'Discharge Date': str(date_discharge),
            'Medication': medication,
        }

        with st.spinner("Predicting..."):
            label, probas, classes = predict_single(artifacts, input_data)

        # Result display
        color_map = {"Normal": "green", "Abnormal": "red", "Inconclusive": "orange"}
        color = color_map.get(label, "blue")

        st.markdown(f"""
        <div style="
            background-color: {'#d4edda' if color == 'green' else '#f8d7da' if color == 'red' else '#fff3cd'};
            border-left: 6px solid {'#28a745' if color == 'green' else '#dc3545' if color == 'red' else '#ffc107'};
            padding: 20px; border-radius: 8px; margin-top: 10px;">
            <h2 style="color: {'#155724' if color == 'green' else '#721c24' if color == 'red' else '#856404'};">
                🩺 Predicted Test Result: {label}
            </h2>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Test Result": classes,
            "Probability": [f"{p:.2%}" for p in probas],
            "Score": probas,
        }).sort_values("Score", ascending=False)

        for _, row in prob_df.iterrows():
            st.progress(float(row["Score"]), text=f"{row['Test Result']}: {row['Probability']}")

        # Summary
        days = (date_discharge - date_admission).days
        st.divider()
        st.subheader("📊 Input Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric("Days in Hospital", days)
        summary_col2.metric("Cost Per Day", f"${billing_amount / max(days, 1):,.2f}")
        summary_col3.metric("Surgical Risk Score", SCORING_MAP.get(medical_condition, 0))

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Model: Stacking Classifier (XGB + RF + HGB) · Dataset: healthcare_dataset.csv")
