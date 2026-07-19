import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering to the solar plant dataframe.
    
    Features engineered:
    - Hour (0-23): Captures diurnal cycle.
    - Minute (0-59): Captures fine time intervals.
    - Day (1-31): Captures daily trends.
    - Month (1-12): Captures monthly/seasonal trends.
    - Weekday (0-6): Captures weekly patterns.
    - Weekend (0 or 1): Captures weekend operational variations (if any).
    - sin_time & cos_time: Cyclical time-of-day encoding to capture circular nature of daily patterns.
    """
    # Create copy to prevent modifying in-place (avoid SideEffect/Warnings)
    df = df.copy()
    
    # Ensure DATE_TIME is parsed
    if not pd.api.types.is_datetime64_any_dtype(df["DATE_TIME"]):
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
        
    logger.info("Extracting time-based calendar features...")
    df["Hour"] = df["DATE_TIME"].dt.hour
    df["Minute"] = df["DATE_TIME"].dt.minute
    df["Day"] = df["DATE_TIME"].dt.day
    df["Month"] = df["DATE_TIME"].dt.month
    df["Weekday"] = df["DATE_TIME"].dt.weekday
    df["Weekend"] = df["Weekday"].apply(lambda x: 1 if x >= 5 else 0)
    
    logger.info("Creating cyclical time-of-day encodings...")
    # Time of day in decimal hours (e.g. 13:30 -> 13.5)
    time_of_day = df["Hour"] + df["Minute"] / 60.0
    df["sin_time"] = np.sin(2 * np.pi * time_of_day / 24.0)
    df["cos_time"] = np.cos(2 * np.pi * time_of_day / 24.0)
    
    return df

if __name__ == "__main__":
    from src.config import PLANT_1_CLEAN_PATH
    df = pd.read_csv(PLANT_1_CLEAN_PATH)
    logger.info(f"Loaded dataset with shape {df.shape}")
    df_feat = engineer_features(df)
    logger.info(f"Engineered features shape {df_feat.shape}")
    logger.info(f"New columns: {df_feat.columns.tolist()}")
