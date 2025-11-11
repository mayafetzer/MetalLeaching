
import streamlit as st
import pandas as pd
import pickle
import os

st.title("Predict Li/Co/Mn/Ni - Tuned Models")
st.write("Upload a CSV with the input columns (same names as in training).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:")
    st.dataframe(df.head())

    # Load models
    models = {}
    for label in ["withcat", "nocat"]:
        for metal in ["Li","Co","Mn","Ni"]:
            fname = f"models/best_tuned_{label}_{metal}.pkl"
            if os.path.exists(fname):
                models[f"{label}_{metal}"] = pickle.load(open(fname,"rb"))

    st.write("Loaded models:", list(models.keys()))

    if st.button("Run predictions"):
        # Decide pipeline type by presence of categorical columns
        need_cat = "Leaching agent" in df.columns and "Type of reducing agent" in df.columns

        results = df.copy()
        for key, model in models.items():
            # skip models that don't match input features (withcat vs nocat)
            if ("withcat" in key and not need_cat) or ("nocat" in key and need_cat and False):
                # only run nocat if we only have numeric inputs; run both otherwise
                pass
            try:
                preds = model.predict(df)
                results[f"pred_{key}"] = preds
            except Exception as e:
                st.write("Failed to predict with model", key, ":", e)

        st.write("Predictions sample:")
        st.dataframe(results.head())

        outname = "predictions_from_streamlit.xlsx"
        results.to_excel(outname, index=False)
        st.write(f"Saved predictions to {outname}")
        with open(outname, "rb") as f:
            st.download_button("Download predictions", f, file_name=outname)
