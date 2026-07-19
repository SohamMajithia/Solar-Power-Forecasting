# Solar Power Forecasting & Performance Monitoring

This project provides a complete pipeline for solar power generation forecasting and performance monitoring for two PV plants.

## Project Structure

- `dashboard/app.py` - Streamlit dashboard for project overview, historical analytics, prediction, monitoring, and model performance.
- `src/` - Core Python modules for configuration, data loading, cleaning, feature engineering, modeling, monitoring, and EDA.
- `data/` - Raw CSV datasets.
- `data/processed/` - Cleaned and merged datasets used for training and dashboard visualization.
- `models/` - Serialized model payloads and scalers consumed by the dashboard.
- `reports/` - Generated reports and figures from exploratory data analysis.
- `notebooks/` - Project notebook placeholder.

## Setup

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

### Run the Streamlit dashboard

```powershell
streamlit run dashboard/app.py
```

### Rebuild the models

```powershell
python src/data_cleaner.py
python src/model.py
```

### Run tests

```powershell
pytest
```

## Notes

- The notebook file in `notebooks/` was previously empty. It now contains a placeholder explaining the missing original analysis.
- `requirements.txt` was empty; it now includes the packages required by the project code.
