import streamlit as st
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import re
import string
from tensorflow import keras
from tensorflow.keras import layers


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
    output_mode='int',
    output_sequence_length=300)

# Load the model with the custom standardization function
loaded_model = keras.models.load_model('./notebooks/NLP-model.keras')


class TextInput(BaseModel):
    text: str


@st.cache_data
def predictor(input):
    labels = ["Csharp", "Java", "Javascript", "Python"]
    examples = input
    examples = vectorize_layer(examples)
    answers = loaded_model.predict(examples)
    label = np.argmax(answers)
    return labels[label]


st.title("Text Predictor")
text_input = st.text_area("Enter your question here:", height=150)
if st.button("Submit"):
    if text_input == "":
        st.error(
            "Please enter a question!"
        )  # Display error message for empty input
    else:
        processed_text = predictor(text_input)
        st.write(
            f"Model predicted the question is for the following language: {processed_text}"
        )
