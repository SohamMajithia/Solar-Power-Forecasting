import os
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import (
    PLANT_1_CLEAN_PATH,
    PLANT_2_CLEAN_PATH,
    MODELS_DIR,
    RANDOM_STATE,
)
from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Features to use for ML models
FEATURES = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "Hour",
    "Minute",
    "sin_time",
    "cos_time",
]
TARGET = "AC_POWER"

def prepare_data(df: pd.DataFrame, plant_name: str):
    """
    Prepares data for machine learning:
    1. Extracts engineered features.
    2. Filters to daytime data (IRRADIATION > 0) to focus training on active generation and prevent metrics skew.
    3. Performs a chronological train-test split (80% train, 20% test) to prevent future data leakage.
    4. Scales features using standard scaler fitted ONLY on training data.
    """
    logger.info(f"[{plant_name}] Engineering features for preprocessing...")
    df_feat = engineer_features(df)
    
    # Filter to daytime hours only and exclude clear outages (IRRADIATION > 0.1 but AC_POWER == 0)
    logger.info(f"[{plant_name}] Original shape: {df_feat.shape}")
    
    # Exclude rows where there is high solar irradiation but zero power generation (offline inverters)
    daytime_df = df_feat[(df_feat["IRRADIATION"] > 0) & ~((df_feat["IRRADIATION"] > 0.1) & (df_feat["AC_POWER"] == 0))].copy()
    removed_outages = len(df_feat[df_feat["IRRADIATION"] > 0]) - len(daytime_df)
    
    logger.info(f"[{plant_name}] Daytime shape (Irradiation > 0 and outages removed): {daytime_df.shape} (Removed {removed_outages} outage records)")
    
    # Chronological Split
    daytime_df = daytime_df.sort_values("DATE_TIME")
    split_idx = int(len(daytime_df) * 0.8)
    
    train_df = daytime_df.iloc[:split_idx]
    test_df = daytime_df.iloc[split_idx:]
    
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    
    # Save test timestamps and details for residual plots & dashboard validation
    test_meta = test_df[["DATE_TIME", "SOURCE_KEY", TARGET]].copy()
    
    logger.info(f"[{plant_name}] Chronological split completed:")
    logger.info(f"  Train Range: {train_df['DATE_TIME'].min()} to {train_df['DATE_TIME'].max()} (Rows: {len(train_df)})")
    logger.info(f"  Test Range:  {test_df['DATE_TIME'].min()} to {test_df['DATE_TIME'].max()} (Rows: {len(test_df)})")
    
    # Scaling to prevent data leakage (fit only on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames with column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURES, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURES, index=X_test.index)
    
    # Save the scaler
    scaler_path = MODELS_DIR / f"{plant_name.lower().replace(' ', '_')}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"[{plant_name}] Saved scaler to {scaler_path}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, test_meta

def train_and_evaluate_models(plant_name: str, X_train, X_test, y_train, y_test):
    """Trains Linear Regression, Random Forest, and XGBoost, and evaluates performance."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, learning_rate=0.1)
    }
    
    results = []
    trained_models = {}
    
    logger.info(f"[{plant_name}] Training and evaluating models...")
    for name, model in models.items():
        logger.info(f"[{plant_name}] Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        preds = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results.append({
            "Model": name,
            "MAE (kW)": round(mae, 4),
            "RMSE (kW)": round(rmse, 4),
            "R2 Score": round(r2, 4)
        })
        trained_models[name] = {
            "model": model,
            "predictions": preds,
            "metrics": {"mae": mae, "rmse": rmse, "r2": r2}
        }
        
    results_df = pd.DataFrame(results)
    return results_df, trained_models

def run_pipeline():
    # Load clean data
    p1 = pd.read_csv(PLANT_1_CLEAN_PATH)
    p2 = pd.read_csv(PLANT_2_CLEAN_PATH)
    
    for df, name in [(p1, "Plant 1"), (p2, "Plant 2")]:
        print("=" * 60)
        print(f"PIPELINE FOR {name.upper()}")
        print("=" * 60)
        
        # Preprocessing & Split
        X_train, X_test, y_train, y_test, test_meta = prepare_data(df, name)
        
        # Train & Eval
        results_table, trained_dict = train_and_evaluate_models(name, X_train, X_test, y_train, y_test)
        
        print(f"\nModel Performance Table for {name}:")
        print(results_table.to_markdown(index=False))
        print()
        
        # Select best model based on lowest RMSE / highest R2
        best_row = results_table.sort_values(by="RMSE (kW)").iloc[0]
        best_model_name = best_row["Model"]
        print(f"Selected Best Model for {name}: {best_model_name}")
        
        # Save best model
        best_model = trained_dict[best_model_name]["model"]
        best_model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}_best_model.joblib"
        
        # Save also a dictionary containing the model, its name, the features, and metrics
        model_payload = {
            "model": best_model,
            "model_name": best_model_name,
            "features": FEATURES,
            "metrics": trained_dict[best_model_name]["metrics"],
            "predictions": trained_dict[best_model_name]["predictions"],
            "y_test": y_test.values,
            "test_meta": test_meta
        }
        joblib.dump(model_payload, best_model_path)
        print(f"Saved best model payload to {best_model_path}\n")

if __name__ == "__main__":
    run_pipeline()
