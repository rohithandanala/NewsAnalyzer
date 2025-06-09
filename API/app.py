import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src import predict_text
from fastapi.responses import JSONResponse
from Pipelines import run_prediction_on_file
import pandas as pd
import traceback
import numpy as np
import yaml

with open('Configs' + '/configs.yaml','r') as f:
    data_configs = yaml.safe_load(f)

app = FastAPI(title="Fake News Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your Vite dev server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PATH = data_configs['bucket_path']

# Input schema
class TextInput(BaseModel):
    text: str



@app.post("/predict")
def predict(input: TextInput):
    try:
        prediction = predict_text.predict_text(input.text)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/get_data")
def get_data():
    try:
        df = pd.read_csv(data_configs['data_path'])
        # Return as JSON
        safe_df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        return JSONResponse(content=safe_df.to_dict(orient="records"))

    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "CSV file not found"})

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "error": str(e),
            "traceback": tb
        }

@app.post('/run_predictions')
def run_predictions(input: TextInput):
    try:
        if input.text != data_configs['prediction_key']:
            return JSONResponse(status_code=401, content={"error": "Unable to authorize request"})

        print('prediction key is authorized.')
        # Read the CSV file (update the path as needed)
        df = pd.read_csv(PATH)

        # Check if required column exists
        if 'summary' not in df.columns:
            return JSONResponse(status_code=400, content={"error": "'text' column not found in CSV"})

        if 'pred' not in df.columns:
            print('Predictions on data not found.')
            df = run_prediction_on_file.run_prediction(PATH, data_configs['data_path'])
            print('predictions done on data')
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "CSV file not found"})

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "error": str(e),
            "traceback": tb
        }



from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

FRONTEND_DIR = os.path.join("frontend", "dist")
ASSETS_DIR = os.path.join(FRONTEND_DIR, "assets")

# ✅ Mount static assets correctly
app.mount("/assets", StaticFiles(directory=ASSETS_DIR, html=False), name="assets")

# ✅ Serve index.html for all routes (catch-all)
@app.get("/")
@app.get("/{full_path:path}")
def serve_spa(full_path: str = ""):
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_file, media_type='text/html')