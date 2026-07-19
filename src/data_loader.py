import pandas as pd
import logging
from src.config import (
    PLANT_1_GEN_PATH,
    PLANT_1_WEA_PATH,
    PLANT_2_GEN_PATH,
    PLANT_2_WEA_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_plant_1_generation() -> pd.DataFrame:
    """Loads and parses Plant 1 Generation Data."""
    logger.info("Loading Plant 1 Generation Data...")
    df = pd.read_csv(PLANT_1_GEN_PATH)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%d-%m-%Y %H:%M")
    return df

def load_plant_1_weather() -> pd.DataFrame:
    """Loads and parses Plant 1 Weather Sensor Data."""
    logger.info("Loading Plant 1 Weather Sensor Data...")
    df = pd.read_csv(PLANT_1_WEA_PATH)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")
    return df

def load_plant_2_generation() -> pd.DataFrame:
    """Loads and parses Plant 2 Generation Data."""
    logger.info("Loading Plant 2 Generation Data...")
    df = pd.read_csv(PLANT_2_GEN_PATH)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")
    return df

def load_plant_2_weather() -> pd.DataFrame:
    """Loads and parses Plant 2 Weather Sensor Data."""
    logger.info("Loading Plant 2 Weather Sensor Data...")
    df = pd.read_csv(PLANT_2_WEA_PATH)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")
    return df
