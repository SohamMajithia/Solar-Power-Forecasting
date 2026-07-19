import os
from pathlib import Path

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Raw Data Paths
PLANT_1_GEN_PATH = DATA_DIR / "Plant_1_Generation_Data.csv"
PLANT_1_WEA_PATH = DATA_DIR / "Plant_1_Weather_Sensor_Data.csv"
PLANT_2_GEN_PATH = DATA_DIR / "Plant_2_Generation_Data.csv"
PLANT_2_WEA_PATH = DATA_DIR / "Plant_2_Weather_Sensor_Data.csv"

# Processed/Clean Data Paths
PLANT_1_CLEAN_PATH = PROCESSED_DATA_DIR / "plant_1_merged_clean.csv"
PLANT_2_CLEAN_PATH = PROCESSED_DATA_DIR / "plant_2_merged_clean.csv"

# Modeling Configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% test split

# Weather features used for training models (excluding DC_POWER)
WEATHER_FEATURES = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
]

TIME_FEATURES = [
    "Hour",
    "Minute",
]

# All features list for modeling
MODEL_FEATURES = WEATHER_FEATURES + TIME_FEATURES
TARGET_COLUMN = "AC_POWER"

# Anomaly Thresholds
# Default absolute deviation limit (kW) or percentage loss
ANOMALY_PERCENTAGE_THRESHOLD = 0.15  # 15% threshold for warning
