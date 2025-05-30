import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from pydantic import BaseModel
from src import predict_text

app = FastAPI(title="Fake News Detection API")

# Input schema
class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Fake News Detection API is running!"}

@app.post("/predict")
def predict(input: TextInput):
    try:
        prediction = predict_text.predict_text(input.text)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}
