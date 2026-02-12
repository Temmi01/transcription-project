import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack


@st.cache_resource
def load_models():
    model = joblib.load("forest.pkl")
    vect = joblib.load("vectorizer.pkl")
    
    return model, vect

model, vect = load_models()

def extract_features(text):
    # A. Get TF-IDF features
    tfidf_part = vect.transform([text])

    # B. Calculate Manual Features
    word_count = len(text.split())
    char_count = len(text)
    average_word_length = char_count / (word_count + 1e-6)  # prevent div by zero

    # C. Combine them into one row
    # We use hstack because TF-IDF is a 'sparse' matrix and manual features are 'dense'
    manual_part = np.array([[word_count, char_count, average_word_length]])
    combined_features = hstack([tfidf_part, manual_part])

    return combined_features


st.title("Technical Support Sentiment AI")
user_input = st.text_area("Paste call transcription here:")

if st.button("Analyze Sentiment"):
    if user_input:
        # Prepare the data
        final_features = extract_features(user_input)

        # Predict
        prediction = model.predict(final_features)

        st.success(f"Predicted Sentiment: {prediction[0]}")
    else:
        st.warning("Please enter some text first.")
