**Contract Risk Analyzer**

It's an Streamlit app that classifies pasted contract text as risky or low risk using a TF‑IDF vectorizer and a trained model.

**Requirements**
- Python `3.14` (see `.python-version`)
- `pip` or `uv`

**Quick Start**
```bash
chmod +x start.sh
./start.sh
```

**Manual Start**
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install streamlit scikit-learn nltk joblib
streamlit run main.py
```

**What It Does**
- Cleans legal text with `legal_preprocessing_py.py`
- Vectorizes with `models/tfidf_vectorizer.pkl`
- Predicts risk with `models/contract_risk_model.pkl`
