# Datathon-2026

Global CO2 change modeling pipeline for SBU Datathon 2026.

## What this model does

The model predicts **change in CO2 emissions** from **GDP sector/industry changes**.

- Uses CSV emissions data with these rules:
  - Combines `Buildings` + `Manufacturing and construction`
  - Excludes `Fugitive emissions` and `Other fuel combustion`
- Uses Excel indicators that contain `(ISIC)` in the name
- Trains on **year-over-year change ratios** at the country-year level
- Is **country-agnostic** during learning (country identity is not used as a feature), so it captures common trends across countries

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python train_predictive_model.py
```

This writes `artifacts/co2_change_model.joblib`.

## Test prediction from CLI

1) Create an input JSON (example file `scenario.json`):

```json
{
  "gdp_change_Agriculture, hunting, forestry, fishing (ISIC A-B)": 0.03,
  "gdp_change_Construction (ISIC F)": 0.05,
  "gdp_change_Manufacturing (ISIC D)": 0.02
}
```

2) Run prediction:

```bash
python predict_co2_change.py --input-json scenario.json --baseline-co2 1000000
```

If a feature is missing in JSON, it is treated as `0.0`.

## Run simple frontend

```bash
streamlit run app.py
```

The UI allows entering sector GDP change assumptions and returns:
- predicted CO2 change %
- predicted absolute CO2 delta from your baseline value
