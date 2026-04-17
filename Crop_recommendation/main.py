from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("AddCropML.pkl")
le_crop = joblib.load("le_crop.pkl")
le_soil = joblib.load("le_soil.pkl")
le_action = joblib.load("le_action.pkl")

@app.get("/")
def home():
    return {"message": "ML API Running 🚀"}

@app.post("/predict")
def predict(data: dict):
    try:
       
        crop = data["crop"]
        soil = data["soil"]
        land = data["land"]
        days = data["days"]

        crop_enc = le_crop.transform([crop])[0]
        soil_enc = le_soil.transform([soil])[0]

        input_data = np.array([[crop_enc, soil_enc, land, days]])

        pred = model.predict(input_data)[0]

        action = le_action.inverse_transform([pred])[0]

        return {"prediction": action}

    except Exception as e:
        return {"error": str(e)}