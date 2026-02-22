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

    cleaned = clean_legal_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("Risky Contract")
    else:
        st.success("Low Risk Contract")
