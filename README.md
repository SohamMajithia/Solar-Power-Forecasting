# Solar-Power-Forecasting
A complete end-to-end machine learning project for solar PV generation forecasting and performance monitoring.

## Project Overview

This repository contains a data pipeline, feature engineering, model training, monitoring logic, and a Streamlit dashboard for interactive analysis and prediction for two solar plants.

## Files and Modules

### `dashboard/app.py`

- Main Streamlit application.
- Loads cleaned plant data and trained model payloads.
- Provides these pages:
  - Home: project overview, KPIs, and best model summary.
  - Historical Dashboard: interactive trends, correlations, and inverters filtering.
  - Power Prediction: user inputs weather/time values and predicts AC power.
  - Performance Monitor: compares predicted output against actual AC power and flags anomalies.
  - Model Performance: displays metrics, feature importance, and residuals.
- Uses `src.config` for paths and `src.monitor` for prediction and anomaly logic.

### `src/config.py`

- Defines project-level paths and constants.
- Paths for raw data, processed data, and trained models.
- Training configuration values like `RANDOM_STATE` and `TEST_SIZE`.
- Feature lists used by the model.
- Anomaly detection threshold.

### `src/data_loader.py`

- Loads raw CSV data for both plants.
- Parses `DATE_TIME` values using the correct timestamp formats for each file.
- Provides separate functions for generation and weather files per plant.

### `src/data_cleaner.py`

- Loads raw plant data via `src.data_loader`.
- Standardizes unit scales (Plant 1 `DC_POWER` is divided by 10).
- Merges generation and weather data on `DATE_TIME` and `PLANT_ID`.
- Cleans merged weather values using linear interpolation and forward/backward fill.
- Verifies that no missing values remain.
- Saves processed outputs to `data/processed/plant_1_merged_clean.csv` and `data/processed/plant_2_merged_clean.csv`.

### `src/feature_engineering.py`

- Creates time-based features from `DATE_TIME`:
  - `Hour`, `Minute`, `Day`, `Month`, `Weekday`, `Weekend`
  - `sin_time`, `cos_time` for cyclical time encoding
- Ensures safe datetime parsing.

### `src/model.py`

- Loads cleaned datasets from `data/processed/`.
- Engineers features using `src.feature_engineering.engineer_features`.
- Filters out nighttime/outage rows and severe zero-power records during high irradiation.
- Performs a chronological train/test split to avoid data leakage.
- Scales features with `StandardScaler` fitted only on training data.
- Trains three models:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- Evaluates each model on the test set.
- Saves the best model payload and scaler to `models/`.
- The saved payload includes the model object, metrics, test predictions, and metadata.

### `src/monitor.py`

- Contains anomaly monitoring logic and prediction helper functions.
- `predict_ac_power()`: builds a feature vector from weather and time inputs, scales it, and returns a model prediction.
- `monitor_performance()`: compares predicted and actual AC power and flags underperformance when actual power drops below the threshold.
- Returns structured results in `MonitorResult`.

### `src/eda.py`

- Exploratory data analysis visualizations and summary report generation.
- Loads cleaned processed data and generates plots:
  - distribution histograms
  - correlation heatmaps
  - hourly and daily generation profiles
  - weather vs AC power scatter plots
  - cumulative yield growth visuals
- Writes summary statistics to `reports/plant_statistics.txt`.

### `tests/test_monitor.py`

- Basic regression tests for `src.monitor` behavior.
- Confirms anomaly and normal status decisions on sample inputs.

## Data Files

- `data/Plant_1_Generation_Data.csv`
- `data/Plant_1_Weather_Sensor_Data.csv`
- `data/Plant_2_Generation_Data.csv`
- `data/Plant_2_Weather_Sensor_Data.csv`
- `data/processed/plant_1_merged_clean.csv`
- `data/processed/plant_2_merged_clean.csv`

## Models

- `models/plant_1_best_model.joblib`
- `models/plant_1_scaler.joblib`
- `models/plant_2_best_model.joblib`
- `models/plant_2_scaler.joblib`

## Reports

- `reports/eda_report.md`
- `reports/feature_engineering_report.md`
- `reports/model_report.md`
- `reports/plant_statistics.txt`
- `reports/figures/` contains image outputs from EDA.

## Setup and Run Instructions

### 1. Create and activate the environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Prepare the data pipeline

```powershell
python src/data_cleaner.py
```

This loads the raw plant CSVs, standardizes and merges generation/weather data, and writes cleaned output to `data/processed/`.

### 4. Train the models

```powershell
python src/model.py
```

This trains Linear Regression, Random Forest, and XGBoost models, selects the best performing model for each plant, and saves the model payloads and scalers to `models/`.

### 5. Run the Streamlit dashboard

```powershell
streamlit run dashboard/app.py
```

### 6. Run tests

```powershell
pytest
```
