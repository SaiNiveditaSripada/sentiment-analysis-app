import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

MODEL_PATH = 'final_best_model.keras'
TOKENIZER_PATH = 'tokenizer.pickle'
LABEL_ENCODER_PATH = 'label_encoder.pickle'
MAX_LEN = 120

@st.cache_resource
def load_all_resources():
    print("--- Loading resources for the first time... ---")
    try:
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print("--- Resources loaded successfully! ---")
        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"Error loading model or preprocessors: {e}")
        return None, None, None

model, tokenizer, label_encoder = load_all_resources()

st.title("Sentiment Predictor")
st.write("Enter a sentence to see if it's Positive or Negative.")
user_input = st.text_input("Your sentence:", "I am very happy today!")

if st.button("Analyze Sentiment", type="primary"):
    if model and tokenizer and label_encoder:
        if user_input.strip():
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
            
            prediction_score = model.predict(padded_sequence)[0][0]
            predicted_class_index = (prediction_score > 0.5).astype(int)
            sentiment = label_encoder.inverse_transform([predicted_class_index])[0]
            

            if sentiment.lower() == 'positive':
                st.success(f"Sentiment: {sentiment} (Score: {prediction_score:.2f})")
            else:
                st.error(f"Sentiment: {sentiment} (Score: {prediction_score:.2f})")
        else:
            st.warning("Please enter some text.")
    else:
        st.error("Model assets are not loaded. Please check the terminal for errors.")