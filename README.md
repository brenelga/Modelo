# 🚓 Sistema de Clasificación de Riesgo por Municipio  
### FastAPI + Machine Learning + Railway Deployment

Este proyecto implementa un modelo de **Machine Learning** para clasificar el **nivel de riesgo** por municipio en base al comportamiento histórico de delitos relacionados con vehículos.  
El sistema está compuesto por:

- 🧠 **Modelo ML** (KMeans + Random Forest)  
- 🧪 **Scaler** para estandarización de features  
- 🌎 **API en FastAPI** para consultas  
- ☁️ **Despliegue en Railway**  
- 📦 **Dataset procesado** (`municipal_features_and_risk.csv`)  
- 📱 **Consumo desde Vue.js / Laravel / Apps móviles**

---

## 🚀 Características del proyecto

- Clasifica municipios en **zonas de riesgo (1–5)**.
- Normaliza nombres de entidad y municipio para búsquedas consistentes.
- Devuelve predicciones usando solo:
  - **Entidad**  
  - **Municipio**

- No requiere que el frontend envíe cálculos numéricos.
- Es compatible con:
  - Vue.js  
  - Laravel  
  - React Native  
  - Python
  - Postman / Thunder Client  

---

## 📁 Estructura del proyecto

/project
│── app.py # FastAPI backend
│── requirements.txt # Dependencias
│── risk_zone_classifier.pkl # Modelo Random Forest
│── scaler.pkl # Scaler usado en el entrenamiento
│── municipal_features_and_risk.csv # Dataset procesado
│── LICENSE # MIT License
│── README.md


---

## ⚙️ Tecnologías utilizadas

- Python 3.10+
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas / Numpy
- Railway (deployment)
- Joblib

---

## 🧠 Entrenamiento del modelo

El modelo se entrenó aplicando:

- Limpieza y normalización de texto  
- Cálculo de características como:
  - `mean_ann`
  - `std_ann`
  - `trend_slope`
  - `sum_all`
  - `growth_last`
- Clustering con **KMeans**
- Clasificación final con **RandomForestClassifier**

El archivo `municipal_features_and_risk.csv` contiene los features por municipio ya procesados.

---

## 🌐 API – Endpoints

### **GET /**
Verifica que la API esté activa.

**Respuesta:**
```json
{
  "status": "ok",
  "message": "API de clasificación de riesgo activa"
}
---
POST /predict-by-location

Permite obtener la predicción del nivel de riesgo basado en:

Entidad

Municipio

Body JSON:
```json
{
  "Entidad": "ESTADO DE MÉXICO",
  "Municipio": "NEZAHUALCÓYOTL"
}
```
Respuesta:
```json
{
  "Entidad": "ESTADO DE MÉXICO",
  "Municipio": "NEZAHUALCÓYOTL",
  "risk_zone": 4
}

Instalación local

Clonar el repositorio:

```bash
git clone https://github.com/brenelga/modelo.git
cd modelo

Crear entorno virtual:

```bash
python -m venv venv
source venv/bin/activate

Instalar dependencias:
```bash
pip install -r requirements.txt

# Licencia
Este proyecto está licenciado bajo MIT License.
Puedes usarlo, modificarlo y distribuirlo libremente, siempre que mantengas el aviso de copyright.

# Autor

JESUS BRENEL GALICIA AGUILAR
