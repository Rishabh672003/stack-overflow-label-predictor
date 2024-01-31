# Stack Overflow label predictor

A simple NLP model that predicts the language of the question user has asked, The model has been trained on stack-overflow dataset which has 16,000 questions from 4 languages and the language the question its labelled for, See the Jupyter Notebook in the notebooks directory for more information and see the implementation of the model.

before this was written with FastAPI and Jinja templates (see in `src/fastapi.py`)

But for deployement I switched to **Streamlit** 
Now the Website is deployed here - https://stack-overflow-label-predictor-rishabh.streamlit.app/

Now models only works for following language:
- Csharp
- Javascript
- Java
- Python

# Models analysis

## Loss and Validation Loss

![image](https://github.com/Rishabh672003/stack-overflow-label-predictor/assets/53911515/b11fb448-58cc-46cc-9607-1df4dc030222)

## Accuracy and Validation Accuracy

![image](https://github.com/Rishabh672003/stack-overflow-label-predictor/assets/53911515/a41e6d9b-5007-4650-a03c-135ecbdc2f83)

## Accuracy and Loss on the Test set

![image](https://github.com/Rishabh672003/stack-overflow-label-predictor/assets/53911515/99e7f5f8-fa99-4282-8e4f-cd7f92546b96)


# Preview of the website

![image](https://github.com/Rishabh672003/stack-overflow-label-predictor/assets/53911515/1f8c91df-3f5d-44f4-9789-ec3bd0d205a9)
