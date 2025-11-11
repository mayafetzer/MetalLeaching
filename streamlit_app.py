import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --------------------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------------------
st.set_page_config(page_title="Metal Leaching Prediction App", layout="centered")

st.title("🔮 Metal Leaching Efficiency Predictor")
st.write("""
This app uses machine learning models trained on your leaching dataset to predict the 
recovery percentages of **Li, Co, Mn, Ni**.  
Models automatically preprocess numeric + categorical values, so **raw inputs are fine**.
""")

# --------------------------------------------------------------
# Model Loader (ABSOLUTE PATH + DEBUG)
# --------------------------------------------------------------
@st.cache_resource
def load_models():
    models_loaded = {}
    
    base_path = os.path.join(os.getcwd(), "models")

    st.write("📁 **Model directory detected at:**", base_path)

    if not os.path.exists(base_path):
        st.error("❌ ERROR: 'models' folder does not exist. Upload models to /models.")
        return models_loaded

    st.write("📄 **Files found in /models:**", os.listdir(base_path))

    for metal in ["Li", "Co", "Mn", "Ni"]:
        for label in ["withcat", "nocat"]:
            filename = f"best_tuned_{label}_{metal}.pkl"
            filepath = os.path.join(base_path, filename)

            st.write(f"Checking for: `{filename}`")

            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    models_loaded[f"{label}_{metal}"] = pickle.load(f)
                st.success(f"✅ Loaded: {filename}")
            else:
                st.warning(f"⚠️ Missing model: {filename}")

    return models_loaded

models = load_models()

# --------------------------------------------------------------
# Input Column Definitions
# --------------------------------------------------------------
numeric_cols = [
    "Li in feed %",
    "Co in feed %",
    "Mn in feed %",
    "Ni in feed %",
    "Concentration, M",
    "Concentration %",
    "Time,min",
    "Temperature, C"
]

categorical_cols = [
    "Leaching agent",
    "Type of reducing agent"
]

all_cols = numeric_cols + categorical_cols

# --------------------------------------------------------------
# Input Method Selector
# --------------------------------------------------------------
st.header("📥 Provide Input Data")
method = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

# --------------------------------------------------------------
# Manual Input Form
# --------------------------------------------------------------
def manual_input():
    st.subheader("📝 Manual Input")

    data = {}

    for col in numeric_cols:
        data[col] = st.number_input(col, step=0.01, value=0.0)

    data["Leaching agent"] = st.selectbox(
        "Leaching agent",
        ["ORGANIC_ACID", "INORGANIC_ACID", "BASE", "UNKNOWN"]
    )

    data["Type of reducing agent"] = st.selectbox(
        "Type of reducing agent",
        ["YES", "NO", "UNKNOWN"]
    )

    return pd.DataFrame([data])

# --------------------------------------------------------------
# CSV Upload
# --------------------------------------------------------------
def upload_input():
    st.subheader("📤 Upload Input CSV File")
    file = st.file_uploader("Upload input CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
            return None
        st.write("✅ Preview:")
        st.dataframe(df.head())
        return df
    return None

# Determine which pipeline to use
def detect_pipeline(df):
    if all(c in df.columns for c in categorical_cols):
        return "withcat"
    else:
        return "nocat"

# Run predictions
def run_predictions(df):
    pipeline_type = detect_pipeline(df)
    st.info(f"📌 Using **{pipeline_type.upper()}** models.")

    results = pd.DataFrame(index=df.index)

    for metal in ["Li", "Co", "Mn", "Ni"]:
        key = f"{pipeline_type}_{metal}"

        if key not in models:
            st.error(f"❌ Model not found: {key}")
            continue

        model = models[key]
        try:
            results[f"pred_{metal}"] = model.predict(df)
        except Exception as e:
            st.error(f"Prediction failed for {metal}: {e}")

    return results

# --------------------------------------------------------------
# Main Logic
# --------------------------------------------------------------
if method == "Manual Input":
    df_input = manual_input()
else:
    df_input = upload_input()

if df_input is not None:
    if st.button("🔮 Predict"):
        preds = run_predictions(df_input)

        st.subheader("✅ Predictions")
        st.dataframe(preds)

        # Save Excel output
        output = pd.concat([df_input, preds], axis=1)
        out_path = "predictions.xlsx"
        output.to_excel(out_path, index=False)

        st.download_button(
            "📥 Download Results",
            data=open(out_path, "rb").read(),
            file_name="predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
