import streamlit as st
import pandas as pd
from engine import full_pipeline

st.title("Stock Direction Predictor 📈")

# --- User inputs ---
ticker = st.text_input("Ticker symbol", value="AAPL")
start_date = st.date_input("Start date", value=pd.to_datetime("2015-01-01"))
threshold = st.slider("Prediction threshold", 0.1, 0.9, 0.5, 0.01)

if st.button("Run model"):
    # Convert date to string 'YYYY-MM-DD'
    start_str = start_date.strftime("%Y-%m-%d")

    with st.spinner("Running model..."):
        output = full_pipeline(ticker=ticker, start=start_str, threshold=threshold)

    metrics = output["metrics"]
    st.subheader("Metrics")
    st.write(f"Accuracy: **{metrics['accuracy']:.3f}**")
    st.write(f"Baseline accuracy (always predict {metrics['baseline_class']}): "
             f"**{metrics['baseline_accuracy']:.3f}**")

    st.subheader("Confusion matrix")
    st.write(metrics["confusion_matrix"])

    # Optional: show some predictions
    st.subheader("Sample predictions (last 20 rows)")
    y_test = output["y_test"]
    y_pred = output["y_pred"]
    y_proba = output["y_proba"]

    df_results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "Prob_UP": y_proba,
    })

    st.dataframe(df_results.tail(20))