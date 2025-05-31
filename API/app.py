import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from pydantic import BaseModel
from src import predict_text
from fastapi.responses import JSONResponse
from Pipelines import run_prediction_on_file
import pandas as pd

app = FastAPI(title="Fake News Detection API")
PATH = "/home/ec2-user/NewsBucketMount/news_data/trending_news.csv"

# Input schema
class TextInput(BaseModel):
    text: str

@app.get("/working-dir")
def get_working_directory():
    try:
        cwd = os.getcwd()
        return {"working_directory": cwd}
    except Exception as e:
        return {"error": str(e)}

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
    
@app.get("/predict_local_csv")
def predict_from_local_csv():
    try:
        # Read the CSV file (update the path as needed)
        df = pd.read_csv(PATH)

        # Check if required column exists
        if 'text' not in df.columns:
            return JSONResponse(status_code=400, content={"error": "'text' column not found in CSV"})

        if 'pred' not in df.colums:
            print('Predictions on data not found.')
            run_prediction_on_file()
            print('predictions done on data')

        # Return as JSON
        return {"results": df.to_dict(orient="records")}

    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "CSV file not found"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

