import streamlit as st
import joblib
from legal_preprocessing_py import clean_legal_text

@st.cache_resource
def load_model():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/contract_risk_model.pkl")
    return vectorizer, model

vectorizer, model = load_model()

st.title("Contract Risk Analyzer")
st.markdown("Paste or type contract text below to check if it's risky.")

text = st.text_area("Contract Text", height=200, placeholder="Paste the contract text here...")

if st.button("Analyze Risk"):

    if text.strip() == "":
        st.warning("Please enter contract text.")
    else:
        cleaned = clean_legal_text(text)
        features = vectorizer.transform([cleaned])

        # Prediction
        prediction = model.predict(features)[0]

        # Probability (Risk Score)
        probabilities = model.predict_proba(features)
        risk_score = (probabilities[0][1])*100   # assuming class 1 = risky

        st.subheader("Analysis Result")

        if prediction == 1:
            st.error("Risky Contract")
        else:
            st.success("Low Risk Contract")

        st.write(f"Risk Score: {risk_score:.2f} %")