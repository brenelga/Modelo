import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import unicodedata

app = FastAPI()

def normalize_name(x):
    if x is None:
        return None
    s = str(x).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = " ".join(s.split())
    return s

df = pd.read_csv("municipal_features_and_risk.csv")

df["Entidad"] = df["Entidad"].apply(normalize_name)
df["Municipio"] = df["Municipio"].apply(normalize_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/predict-by-location")
def predict(payload: dict):
    try:
        ent = normalize_name(payload.get("Entidad"))
        mun = normalize_name(payload.get("Municipio"))
    except:
        return {"error": "Formato inválido"}

    if not ent or not mun:
        return {"error": "Se requiere Entidad y Municipio"}

    row = df[(df["Entidad"] == ent) & (df["Municipio"] == mun)]

    if row.empty:
        return {
            "error": "Municipio/Entidad no encontrado",
            "Entidad": ent,
            "Municipio": mun
        }

    riesgo = int(row["risk_zone"].iloc[0])

    return {
        "Entidad": ent,
        "Municipio": mun,
        "Riesgo": riesgo
    }

@app.get("/")
def root():
    return {"status": "ML Lookup API Online"}


import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)