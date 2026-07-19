import pandas as pd
import logging
from src.config import (
    PLANT_1_CLEAN_PATH,
    PLANT_2_CLEAN_PATH,
)
from src.data_loader import (
    load_plant_1_generation,
    load_plant_1_weather,
    load_plant_2_generation,
    load_plant_2_weather,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def clean_and_merge_plant_1() -> pd.DataFrame:
    """Loads, cleans, standardizes and merges Plant 1 datasets."""
    logger.info("Starting cleaning and merging for Plant 1...")
    
    # Load raw
    gen = load_plant_1_generation()
    wea = load_plant_1_weather()
    
    # 1. Scaling DC_POWER by 10 in Plant 1 to match Plant 2 (standard kW units)
    logger.info("Standardizing Plant 1 DC_POWER (scaling down by 10)...")
    gen["DC_POWER"] = gen["DC_POWER"] / 10.0
    
    # 2. Merge weather and generation data
    # We do a left join from generation to weather on DATE_TIME and PLANT_ID
    logger.info("Merging Generation and Weather datasets for Plant 1...")
    merged = pd.merge(gen, wea, on=["DATE_TIME", "PLANT_ID"], how="left", suffixes=("_gen", "_wea"))
    
    # 3. Check for missing values in weather variables after left-join
    # Weather SOURCE_KEY might be NaN since weather source key is not in gen
    # Let's drop weather SOURCE_KEY or rename it
    if "SOURCE_KEY_wea" in merged.columns:
        merged = merged.drop(columns=["SOURCE_KEY_wea"])
    if "SOURCE_KEY_gen" in merged.columns:
        merged = merged.rename(columns={"SOURCE_KEY_gen": "SOURCE_KEY"})
        
    nan_counts = merged[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].isnull().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Found missing weather values after merge in Plant 1:\n{nan_counts}")
        logger.info("Interpolating missing weather values...")
        # Since weather data is continuous, linear interpolation is suitable
        merged[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]] = \
            merged[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].interpolate(method="linear")
            
    # Double check if any NaNs remain (e.g. at the edges) and fill backward/forward
    if merged[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].isnull().sum().sum() > 0:
        merged[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]] = \
            merged[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].bfill().ffill()
            
    # Verify final integrity
    assert merged.isnull().sum().sum() == 0, "Error: Missing values remain in Plant 1 clean data!"
    
    # Save to processed folder
    merged.to_csv(PLANT_1_CLEAN_PATH, index=False)
    logger.info(f"Saved Plant 1 cleaned merged data to {PLANT_1_CLEAN_PATH} (Shape: {merged.shape})")
    
    return merged

def clean_and_merge_plant_2() -> pd.DataFrame:
    """Loads, cleans, standardizes and merges Plant 2 datasets."""
    logger.info("Starting cleaning and merging for Plant 2...")
    
    # Load raw
    gen = load_plant_2_generation()
    wea = load_plant_2_weather()
    
    # Plant 2 DC_POWER is already on the correct scale (kW)
    
    # Merge weather and generation data
    logger.info("Merging Generation and Weather datasets for Plant 2...")
    merged = pd.merge(gen, wea, on=["DATE_TIME", "PLANT_ID"], how="left", suffixes=("_gen", "_wea"))
    
    # Handle weather columns and source key
    if "SOURCE_KEY_wea" in merged.columns:
        merged = merged.drop(columns=["SOURCE_KEY_wea"])
    if "SOURCE_KEY_gen" in merged.columns:
        merged = merged.rename(columns={"SOURCE_KEY_gen": "SOURCE_KEY"})
        
    nan_counts = merged[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].isnull().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Found missing weather values after merge in Plant 2:\n{nan_counts}")
        logger.info("Interpolating missing weather values...")
        merged[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]] = \
            merged[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].interpolate(method="linear")
            
    if merged[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].isnull().sum().sum() > 0:
        merged[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]] = \
            merged[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].bfill().ffill()
            
    # Verify final integrity
    assert merged.isnull().sum().sum() == 0, "Error: Missing values remain in Plant 2 clean data!"
    
    # Save to processed folder
    merged.to_csv(PLANT_2_CLEAN_PATH, index=False)
    logger.info(f"Saved Plant 2 cleaned merged data to {PLANT_2_CLEAN_PATH} (Shape: {merged.shape})")
    
    return merged

if __name__ == "__main__":
    clean_and_merge_plant_1()
    clean_and_merge_plant_2()
