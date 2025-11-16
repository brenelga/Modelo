from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import numpy as np

app = FastAPI()

# === CARGAR MODELO, SCALER Y DATASET ===
clf = joblib.load("risk_zone_classifier.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv("municipal_features_and_risk.csv", encoding="latin-1")

# Normalizar nombre como lo hiciste en el entrenamiento
import unicodedata
def normalize_name(x):
    if pd.isna(x): return x
    s = str(x).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = " ".join(s.split())
    return s

df["Entidad"] = df["Entidad"].apply(normalize_name)
df["Municipio"] = df["Municipio"].apply(normalize_name)


@app.get("/")
def root():
    return {"status": "ok", "message": "API de clasificación de riesgo activa"}


@app.post("/predict-by-location")
def predict_by_location(payload: dict):
    entidad = normalize_name(payload["Entidad"])
    municipio = normalize_name(payload["Municipio"])

    # Buscar el municipio dentro del CSV
    row = df[(df["Entidad"] == entidad) & (df["Municipio"] == municipio)]

    if row.empty:
        raise HTTPException(status_code=404, detail="Entidad o Municipio no encontrado")

    # Extraer los features usados en el entrenamiento
    features = row[["mean_ann", "std_ann", "trend_slope", "sum_all", "growth_last"]].values

    # Escalar
    scaled = scaler.transform(features)

    # Predecir
    risk_prediction = clf.predict(scaled)[0]

    return {
        "Entidad": entidad,
        "Municipio": municipio,
        "risk_zone": int(risk_prediction)
    }