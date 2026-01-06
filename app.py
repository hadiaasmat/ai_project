import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# 2. CLEAN ACADEMIC CSS
# --------------------------------------------------
<style>
    /* App Background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: "Inter", sans-serif;
    }

    /* Header */
    .main-header {
        background-color: #ffffff;
        padding: 28px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
        text-align: center;
        border-left: 6px solid #4f46e5;
    }

    .main-header h1 {
        color: #1e293b;
        font-weight: 700;
    }

    /* Prediction Card */
    .prediction-card {
        background-color: #ffffff;
        border-left: 6px solid #4f46e5;
        border-radius: 12px;
        padding: 22px;
        margin: 16px 0;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    /* Score Display */
    .score-display {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1e293b;
    }

    /* Buttons */
    .stButton > button {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
        padding: 12px 28px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #4338ca !important;
        transform: translateY(-1px);
    }

    /* Inputs */
    input, textarea, select {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 6px !important;
        border: 1px solid #cbd5f5 !important;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background-color: #e5e7eb;
        margin: 24px 0;
    }
</style>

# --------------------------------------------------
# 3. DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data(uploaded=None):
    if uploaded:
        return pd.read_csv(uploaded)
    elif os.path.exists("cleaned dataset.csv"):
        return pd.read_csv("cleaned dataset.csv")
    return None

with st.sidebar:
    st.header("📂 Dataset")
    uploaded_file = st.file_uploader("Upload cleaned dataset (CSV)", type="csv")

df = load_data(uploaded_file)

if df is None:
    st.error("Dataset not found. Upload `cleaned dataset.csv`.")
    st.stop()

# --------------------------------------------------
# 4. MODEL TRAINING
# --------------------------------------------------
@st.cache_resource
def train_model(df):
    X = df.drop("final_score", axis=1)
    y = df["final_score"]

    label_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds))
    }

    return model, scaler, label_encoders, X.columns.tolist(), metrics, X_test, y_test, preds

model, scaler, encoders, features, metrics, X_test, y_test, y_pred = train_model(df)

# --------------------------------------------------
# 5. INPUT PROCESSING
# --------------------------------------------------
def preprocess_input(data):
    for col, le in encoders.items():
        if col in data:
            data[col] = le.transform(
                data[col].astype(str)
                .where(data[col].isin(le.classes_), le.classes_[0])
            )

    for col in features:
        if col not in data:
            data[col] = 0

    data = data[features]
    return pd.DataFrame(scaler.transform(data), columns=features)

# --------------------------------------------------
# 6. TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Prediction", "📊 Model Analytics", "📚 Dataset", "ℹ️ About"]
)

# --------------------------------------------------
# TAB 1: PREDICTION
# --------------------------------------------------
with tab1:
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 15, 35, 20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        grade = st.selectbox("Grade Level", ["10th", "11th", "12th", "1st Year", "2nd Year", "3rd Year"])
        study = st.slider("Study Hours / Day", 0.0, 10.0, 3.5)

    with c2:
        sleep = st.slider("Sleep Hours", 0.0, 10.0, 7.0)
        social = st.slider("Social Media Hours", 0.0, 8.0, 2.5)
        concept = st.slider("Concept Understanding", 1, 10, 6)
        consistency = st.slider("Study Consistency Index", 1.0, 10.0, 5.5)

    with c3:
        ai_use = st.selectbox("Uses AI?", ["Yes", "No"])
        ai_time = st.slider("AI Usage (min/day)", 0, 300, 60)
        ai_dep = st.slider("AI Dependency", 1, 10, 5)
        ai_pct = st.slider("AI Content %", 0, 100, 30)

    if st.button("Predict Final Score"):
        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "grade_level": grade,
            "study_hours_per_day": study,
            "uses_ai": 1 if ai_use == "Yes" else 0,
            "ai_usage_time_minutes": ai_time,
            "ai_tools_used": "ChatGPT",
            "ai_usage_purpose": "Study",
            "ai_dependency_score": ai_dep,
            "ai_generated_content_percentage": ai_pct,
            "ai_prompts_per_week": 40,
            "ai_ethics_score": 6,
            "concept_understanding_score": concept,
            "study_consistency_index": consistency,
            "improvement_rate": 10,
            "sleep_hours": sleep,
            "social_media_hours": social,
            "tutoring_hours": 2,
            "class_participation_score": 6
        }])

        pred = model.predict(preprocess_input(input_df))[0]
        pred = np.clip(pred, 0, 100)

        st.markdown(f"""
        <div class="predict-card">
        <h2>Predicted Final Score</h2>
        <h1>{pred:.1f} / 100</h1>
        <p>Status: {"✅ PASS" if pred >= 50 else "⚠️ AT RISK"}</p>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# TAB 2: ANALYTICS
# --------------------------------------------------
with tab2:
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{metrics['r2']:.3f}")
    m2.metric("MAE", f"{metrics['mae']:.2f}")
    m3.metric("RMSE", f"{metrics['rmse']:.2f}")

    fig = go.Figure()
    fig.add_scatter(x=y_test, y=y_pred, mode="markers")
    fig.add_scatter(x=[0,100], y=[0,100], mode="lines", name="Perfect Fit")
    fig.update_layout(template="plotly_dark",
                      xaxis_title="Actual",
                      yaxis_title="Predicted")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# TAB 3: DATASET
# --------------------------------------------------
with tab3:
    st.dataframe(df.head(20), use_container_width=True)
    st.write(df.describe())

# --------------------------------------------------
# TAB 4: ABOUT
# --------------------------------------------------
with tab4:
    st.markdown("""
    **Model**: Random Forest Regressor  
    **Target**: Final Score (0–100)  
    **Features**: Academic habits, AI usage, lifestyle  
    **Purpose**: Educational analytics & performance prediction  
    """)


