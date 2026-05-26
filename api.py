from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    distance_scaler = pickle.load(f)
with open('y_scaler.pkl', 'rb') as f:
    y_scaler = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
with open('smearing_factor.pkl', 'rb') as f:
    smearing_factor = pickle.load(f)

class PredictionRequest(BaseModel):
    ship_type: str
    fuel_type: str
    distance: float
    actual_fuel: float = None

def predict_fuel(ship_type: str, fuel_type: str, distance: float) -> float:
    row = {col: 0 for col in feature_columns}
    dist_log = np.log(distance)
    dist_scaled = distance_scaler.transform(pd.DataFrame([[dist_log]], columns=['distance']))[0][0]
    row['distance'] = dist_scaled
    ship_col = f'ship_type_{ship_type}'
    if ship_col in row:
        row[ship_col] = 1
    fuel_col = f'fuel_type_{fuel_type}'
    if fuel_col in row:
        row[fuel_col] = 1

    input_df = pd.DataFrame([row])
    pred_scaled = model.predict(input_df)[0]
    pred_log = y_scaler.inverse_transform([[pred_scaled]])[0][0]
    pred_original = np.exp(pred_log) * smearing_factor
    return round(pred_original, 2)

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        pred = predict_fuel(request.ship_type, request.fuel_type, request.distance)
        response = {
            "predicted_fuel": pred,
            "ship_type": request.ship_type,
            "fuel_type": request.fuel_type,
            "distance": request.distance
        }
        if request.actual_fuel is not None:
            response["actual_fuel"] = request.actual_fuel
            response["difference"] = round(request.actual_fuel - pred, 2)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")