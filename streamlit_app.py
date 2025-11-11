import streamlit as st
import pandas as pd
import pickle
import os

st.write("Working directory:", os.getcwd())
st.write("Files in working directory:", os.listdir("."))
st.write("Files in /models:", os.listdir("./models") if os.path.exists("./models") else "NO MODELS FOLDER FOUND")

st.set_page_config(page_title="Leaching Efficiency Predictor", layout="centered")

st.title("🔮 Leaching Efficiency Predictor (Li, Co, Mn, Ni)")
st.write("""
This tool uses **trained machine learning models** (with and without categorical variables) 
to predict the leaching recovery of **Li, Co, Mn, Ni**.

Enter input values manually **or** upload a CSV file containing the same columns.
""")

# -------------------------------------------
# Load models
# -------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    base_path = "models"

    for label in ["withcat", "nocat"]:
        for metal in ["Li", "Co", "Mn", "Ni"]:
            fname = f"{base_path}/best_tuned_{label}_{metal}.pkl"
            if os.path.exists(fname):
                with open(fname, "rb") as f:
                    models[f"{label}_{metal}"] = pickle.load(f)
    return models

models = load_models()

st.success(f"✅ Loaded {len(models)} trained models.")


# -------------------------------------------
# Input columns (raw, non-normalized)
# -------------------------------------------
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


# -------------------------------------------
# Input Method Selection
# -------------------------------------------
st.header("📥 Choose Input Method")
method = st.radio("How do you want to provide data?", ["Manual Input", "Upload CSV"])


# -------------------------------------------
# Manual Input Form
# -------------------------------------------
def manual_input():
    st.subheader("📝 Enter Input Values")

    data = {}

    # numeric inputs
    for col in numeric_cols:
        data[col] = st.number_input(col, value=0.0, format="%.4f")

    # categorical inputs
    data["Leaching agent"] = st.selectbox(
        "Leaching agent",
        ["ORGANIC_ACID", "INORGANIC_ACID", "BASE", "UNKNOWN"]
    )

    data["Type of reducing agent"] = st.selectbox(
        "Type of reducing agent",
        ["YES", "NO", "UNKNOWN"]
    )

    df = pd.DataFrame([data])
    return df


# -------------------------------------------
# CSV Upload
# -------------------------------------------
def upload_input():
    st.subheader("📤 Upload CSV File")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
        else:
            st.write("✅ Input Preview")
            st.dataframe(df.head())
            return df
    return None


# -------------------------------------------
# Select pipeline depending on data
# -------------------------------------------
def detect_pipeline(df):
    if all(c in df.columns for c in categorical_cols):
        return "withcat"
    else:
        return "nocat"


# -------------------------------------------
# Predict
# -------------------------------------------
def run_predictions(df):
    pipeline_type = detect_pipeline(df)

    st.info(f"Using **{pipeline_type.upper()}** models based on available columns.")

    preds = pd.DataFrame()
    preds["index"] = df.index

    for metal in ["Li", "Co", "Mn", "Ni"]:
        key = f"{pipeline_type}_{metal}"
        if key not in models:
            st.error(f"Model not found: {key}")
            continue

        model = models[key]

        try:
            preds[f"pred_{metal}"] = model.predict(df)
        except Exception as e:
            st.error(f"Prediction failed for {metal}: {e}")

    preds = preds.set_index("index")
    return preds


# -------------------------------------------
# MAIN LOGIC
# -------------------------------------------
if method == "Manual Input":
    df_input = manual_input()
else:
    df_input = upload_input()

if df_input is not None:
    if st.button("🔮 Predict"):
        results = run_predictions(df_input)

        st.subheader("✅ Predictions")
        st.dataframe(results)

        # Download
        out = pd.concat([df_input, results], axis=1)
        out_name = "predictions.xlsx"
        out.to_excel(out_name, index=False)

        st.download_button("📥 Download Predictions", data=open(out_name, "rb"),
                           file_name=out_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
