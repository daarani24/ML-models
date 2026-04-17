from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("AddCropML.pkl")
le_crop = joblib.load("le_crop.pkl")
le_soil = joblib.load("le_soil.pkl")
le_action = joblib.load("le_action.pkl")


def predict_action(crop, soil, land, days):
    crop_val = le_crop.transform([crop])[0]
    soil_val = le_soil.transform([soil])[0]

    pred = model.predict([[crop_val, soil_val, land, days]])

    return le_action.inverse_transform(pred)[0]


def generate_notification(crop_name, soil, land, days, rainfall):
    ml_action = predict_action(crop_name, soil, land, days)

    if rainfall > 20:
        final_action = "Skip Irrigation"
    else:
        final_action = ml_action

    return f"{crop_name}: {final_action} now"


@app.get("/")
def home():
    return {"message": "ML API Running 🚀"}


@app.post("/predict")
def predict(data: dict):
    try:
        msg = generate_notification(
            crop_name=data["crop"],
            soil=data["soil"],
            land=data["land"],
            days=data["days"],
            rainfall=data["rainfall"]
        )

        return {"notification": msg}

    except Exception as e:
        return {"error": str(e)}