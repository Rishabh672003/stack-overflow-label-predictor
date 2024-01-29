import streamlit as st
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import re
import string


# Define the custom standardization function
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


# Load the model with the custom standardization function
loaded_model = tf.keras.models.load_model(
    "./NLP-model",
    custom_objects={"custom_standardization": custom_standardization},
)


class TextInput(BaseModel):
    text: str


@st.cache_data
def predictor(input):
    labels = ["Csharp", "Java", "Javascript", "Python"]
    examples = [input]

    answers = loaded_model.predict(examples)
    label = np.argmax(answers)
    return labels[label]


st.title("Text Predictor")
text_input = st.text_input("Enter your text here:")
if st.button("Submit"):
    processed_text = predictor(text_input)
    st.write("Processed Text: ", processed_text)
