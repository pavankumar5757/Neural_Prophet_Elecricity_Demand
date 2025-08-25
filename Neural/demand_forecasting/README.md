# Demand Forecasting for India using Neural Prophet

This project implements a four-phase plan to build regional and national electricity demand forecasts for India with Neural Prophet.

## Structure

- `requirements.txt` — pinned dependencies
- `main.py` — CLI to run phases
- `config.py` — constants and configuration
- `data/` — place source CSVs here (gitignored)
- `src/`
  - `ingest.py` — load_and_clean_demand_data
  - `holidays_events.py` — public holidays and lockdown events
  - `weather.py` — Open-Meteo fetching and feature engineering
  - `merge.py` — merge long-format demand, holidays, and weather
  - `models.py` — baseline/advanced/global Neural Prophet models
  - `validation.py` — rolling-origin CV and metrics
  - `tuning.py` — hyperparameter grid search

## Quick start

1. Create a virtual environment and install deps:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
2. Put your source CSV in `data/` (see `config.py`).
3. Run baseline end-to-end:
```bash
python main.py run --phase all_baseline
```

See `python main.py --help` for usage.
