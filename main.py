import streamlit as st
import joblib
import pandas as pd
from legal_preprocessing_py import clean_legal_text

@st.cache_resource
def load_model():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/contract_risk_model.pkl")
    return vectorizer, model

vectorizer, model = load_model()

st.title("Contract Risk Analyzer")
st.markdown("Paste or type contract text below to check if it's risky.")

text = st.text_area(
    "Contract Text",
    height=200,
    placeholder="Paste the contract text here..."
)

if st.button("Analyze Risk"):

    if text.strip() == "":
        st.warning("Please enter contract text.")
    else:
        cleaned = clean_legal_text(text)
        features = vectorizer.transform([cleaned])

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)
        risk_score = probabilities[0][1] * 100

        st.subheader("Analysis Result")

        if prediction == 1:
            st.error("Risky Contract")
        else:
            st.success("Low Risk Contract")

        st.write(f"Risk Score: {risk_score:.2f} %")

st.markdown("---")
st.header("Model Insights (Learned During Training)")

if st.checkbox("Show Top Words Learned by the Model"):

    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]

    df = pd.DataFrame({
        "Word": feature_names,
        "Weight": weights
    })

    high_risk = df.sort_values(by="Weight", ascending=False).head(10)
    low_risk = df.sort_values(by="Weight", ascending=True).head(10)

    st.subheader("🔴 Top 10 High-Risk Words (Positive Weights)")
    st.table(high_risk)

    st.subheader("🟢 Top 10 Low-Risk Words (Negative Weights)")
    st.table(low_risk)