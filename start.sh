#!/bin/bash
source .venv/bin/activate
python -m pip install streamlit scikit-learn nltk joblib
streamlit run main.py
