from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import re
import os
import re
import shutil
import string
import tensorflow as tf
import numpy as np

import numpy as np
import tensorflow as tf

from pydantic import BaseModel

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.post("/submit", response_class=HTMLResponse)
async def submit(request: Request, text: str = Form(...)):
    # Process the input text
    processed_text = predictor(text)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "processed_text": processed_text, "text": text},
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class TextInput(BaseModel):
    text: str


@app.post("/echo")
async def echo(input: TextInput):
    print(input)
    return {"text": input.text}


@app.get("/ping")
async def ping():
    return "Server Up and running"


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


def predictor(input):
    labels = ["Csharp", "Java", "Javascript", "Python"]
    examples = [input]

    answers = loaded_model.predict(examples)
    label = np.argmax(answers)
    return labels[label]


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
