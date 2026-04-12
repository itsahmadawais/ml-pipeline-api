from fastapi import FastAPI
import pandas as pd
import joblib
from app.schemas import HouseInput

model = joblib.load("models/house_model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Fast API is working!"}

@app.post("/predict")
def predict(data: HouseInput):
    input_data = pd.DataFrame(
        [[data.house_size, data.bedrooms, data.car_space]],
        columns=["house_size", "bedrooms", "car_space"]
    )

    prediction = model.predict(input_data)[0]
    return {"predicted_price": prediction}