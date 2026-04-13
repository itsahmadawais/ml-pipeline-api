from fastapi import FastAPI
import pandas as pd
import joblib
from app.schemas import HouseInput, EmailInput

house_model = joblib.load("models/house_model.pkl")
email_spam_model = joblib.load("models/email_spam_detector_model.pkl")

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

    prediction = house_model.predict(input_data)[0]
    return {"predicted_price": prediction}

@app.post("/predict_spam")
def predict_spam(data: EmailInput):
    # prediction = email_spam_model.predict([data.email])[0]
    prob = email_spam_model.predict_proba([data.email])[0] # Probability of being spam
    
    prediction = int(prob[1] > 0.5) # Classify as spam if probability > 0.5

    return {
        "is_spam": bool(prediction),
        "spam_probability": prob[1]
    }