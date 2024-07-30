import re
import string

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np


# Define the custom standardization function
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=3000,
    output_mode="int",
    output_sequence_length=300,
)


# Load the TensorFlow model (assuming the model is saved at 'NLP-keras.keras')
@st.cache_resource
def load_model():
    try:
        return keras.models.load_model(
            "./models/NLP-model.keras",
            custom_objects={"custom_standardization": custom_standardization},
        )

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


loaded_model = load_model()


# Ensure the model is loaded before proceeding
if loaded_model is not None:
    # Define the predictor function
    @st.cache_data
    def predictor(input_text):
        labels = ["Csharp", "Java", "Javascript", "Python"]
        input = tf.constant([input_text])
        answers = loaded_model.predict(input)
        label = np.argmax(answers)
        return labels[label]

    # Streamlit app interface
    st.title("Text Predictor")
    text_input = st.text_area("Enter your question here:", height=150)
    if st.button("Submit"):
        if text_input.strip() == "":
            st.error("Please enter a question!")
        else:
            processed_text = predictor(text_input)
            st.write(
                f"Model predicted the question is for the following language: {processed_text}"
            )
else:
    st.error("Model could not be loaded.")
