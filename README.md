# Rain Prediction Project (Scaffold)

Starter scaffold for Rain Prediction using FastAPI (backend) and React + Vite (frontend).

Structure:
- `backend/` — FastAPI app with modular layout and placeholders for ML models.
- `frontend/` — Vite + React app using Redux Toolkit; single-page UI with placeholders for model visualization.

Quick start (Windows):

1. Backend

```
cd backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

2. Frontend

```
cd frontend
npm install
npm run dev
```

The frontend expects the backend at `http://localhost:8000`.
# 🌧 Rainfall & Weather Analysis Project

## 📌 Project Motivation
Rainfall is a complex phenomenon influenced by multiple atmospheric factors such as temperature, humidity, wind, pressure, and cloud formation. The motivation of this project is to collect, analyze, and understand rainfall patterns at a specific geographic location using open, scientifically credible weather data.

### Objectives:
- Study daily rainfall variability
- Understand how meteorological parameters affect rain
- Support water resource planning, climate analysis, and research
- Provide a transparent and reproducible dataset for analysis or modeling

### Suitable For:
- Academic projects
- Environmental and hydrological studies
- Monsoon analysis
- Data science and machine learning experiments

---

## 📍 Location & Time Period
- **Latitude**: 13.555074
- **Longitude**: 80.027227
- **Region**: Coastal South India (near Bay of Bengal)
- **Time Period**: Historical daily data (user-defined range)

---

## 📊 Weather Parameters Used (Daily)

### 🌧 Precipitation
- **precipitation_sum (mm)**: Total precipitation in a day (primary variable representing daily rainfall).
- **rain_sum (mm)**: Liquid rain only (excludes snow). In tropical regions, nearly identical to `precipitation_sum`.

### 🌡 Temperature
- **temperature_2m_max (°C)**: Maximum air temperature at 2 m height. High temperatures enhance convection, increasing rain potential.
- **temperature_2m_min (°C)**: Minimum daily temperature. Influences nighttime cooling and condensation.
- **temperature_2m_mean (°C)**: Average daily temperature. Helps assess atmospheric stability.

### 💧 Humidity
- **relative_humidity_2m_mean (%)**: Average moisture content of air relative to saturation. High values (>75%) indicate rain-favorable conditions.
- **dewpoint_2m_mean (°C)**: Temperature at which air becomes saturated. Smaller gap between air temperature and dew point → higher rain likelihood.

### 🌬 Wind
- **wind_speed_10m_mean (km/h)**: Average wind speed at 10 m height. Helps transport moisture from the sea.
- **wind_direction_10m_dominant (°)**: Dominant wind direction. Indicates monsoon flow and moisture source.
- **wind_gusts_10m_max (km/h)**: Maximum wind gusts. Often associated with storms and heavy rainfall.

### 🔽 Pressure
- **surface_pressure_mean (hPa)**: Average surface air pressure. Lower pressure systems favor cloud formation and rain.

### ☁ Cloud Cover
- **cloud_cover_mean (%)**: Average sky coverage by clouds. High cloud cover strongly correlates with rainfall.

### ☀ Solar Radiation
- **shortwave_radiation_sum (MJ/m²)**: Total incoming solar energy. Lower values usually indicate cloudy or rainy days.

---

## 🧠 Meteorological Context
The study area is influenced by the North-East (NE) Monsoon during October–December, which brings moisture-laden winds from the Bay of Bengal, making this period critical for rainfall analysis.

---

## 📚 Data Source & Scientific Credibility

### 🌍 Data Provider
- **Open-Meteo Historical Weather API**: Free and open weather data service.
- Open-Meteo aggregates data from globally recognized scientific datasets and provides standardized access via API.

### 🔬 Primary Data Source
- **ERA5 Reanalysis Dataset**
  - Produced by: European Centre for Medium-Range Weather Forecasts (ECMWF)
  - ERA5 is a state-of-the-art global atmospheric reanalysis, created by assimilating:
    - Satellite observations
    - Ground weather stations
    - Radiosondes
    - Aircraft and ocean buoy data
    - Into physics-based numerical weather models.

---

## 🎯 Accuracy & Limitations

### ✔ Strengths
- High accuracy for:
  - Temperature
  - Wind
  - Humidity
  - Pressure
- Reliable for:
  - Daily rainfall trends
  - Monthly and seasonal rainfall totals
  - Monsoon analysis

### ⚠ Limitations
- Rainfall is model-estimated, not a single rain-gauge measurement.
- Extreme short-duration rainfall may be smoothed.
- Spatial resolution ≈ 9–25 km grid.
- These limitations are standard for all global reanalysis datasets.

---

## 🔗 API Endpoints Used

### Example Parameters:
```plaintext
latitude=13.555074
longitude=80.027227
daily=precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean,...
timezone=Asia/Kolkata
```

Server-side model storage
-------------------------
The backend provides a simple file-based model storage for development. New model files can be registered via a multipart form upload to:

- `POST /api/models/register` — fields: `file` (model file), `name` (string), optional `algorithm`, `version`.
- `GET /api/models/registry` — list stored model metadata.
- `GET /api/models/download/{model_name}` — download the saved model file.

Uploaded models are kept under `backend_data/models/` and a `registry.json` keeps metadata. This is a placeholder for later integration with model versioning or cloud storage.

---

## 📖 References & Proof of Accuracy
- [Open-Meteo Documentation](https://open-meteo.com/en/docs)
- [ERA5 Reanalysis (ECMWF)](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- [Copernicus Climate Data Store](https://cds.climate.copernicus.eu)