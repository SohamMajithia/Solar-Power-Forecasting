"""
Performance Monitoring System
==============================
Compares expected AC Power (from the ML model) against actual AC Power
(from the plant sensor) and flags significant deviations as
"Potential Performance Anomaly".

This module intentionally does NOT diagnose the exact cause.
It only alerts operators to inspect the plant when a significant
deviation is detected. Possible causes may include:
  - Panel soiling
  - Inverter issues
  - Shading (permanent or temporary)
  - Sensor malfunction
  - Equipment degradation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from src.config import ANOMALY_PERCENTAGE_THRESHOLD


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class MonitorResult:
    """Structured result returned by the monitoring system."""
    predicted_ac_power: float          # kW — expected by the model
    actual_ac_power: float             # kW — measured by the sensor
    absolute_error: float              # kW — |actual - predicted|
    percentage_difference: float       # % — how much actual deviates from predicted
    status: str                        # "Normal" | "Potential Performance Anomaly"
    threshold_used: float              # fractional threshold applied
    recommendation: Optional[str] = field(default=None)


# ---------------------------------------------------------------------------
# Core Monitoring Function
# ---------------------------------------------------------------------------

def monitor_performance(
    predicted_ac_power: float,
    actual_ac_power: float,
    threshold: float = ANOMALY_PERCENTAGE_THRESHOLD,
) -> MonitorResult:
    """
    Compares predicted and actual AC Power, flags anomalies.

    Anomaly Rule
    ------------
    If the actual power is lower than the predicted power by more than
    ``threshold * 100``% of the predicted value, the record is flagged
    as a "Potential Performance Anomaly".

    Parameters
    ----------
    predicted_ac_power : float
        Expected AC Power output predicted by the ML model (kW).
    actual_ac_power : float
        Measured AC Power output from the inverter sensor (kW).
    threshold : float
        Fractional deviation threshold (default 0.15 → 15%).
        Configurable via ``ANOMALY_PERCENTAGE_THRESHOLD`` in config.py.

    Returns
    -------
    MonitorResult
        Dataclass containing all computed metrics and the anomaly status.
    """
    absolute_error = abs(actual_ac_power - predicted_ac_power)

    # Avoid division by zero: if predicted is 0 (night/no sun), we skip
    if predicted_ac_power <= 0:
        return MonitorResult(
            predicted_ac_power=predicted_ac_power,
            actual_ac_power=actual_ac_power,
            absolute_error=0.0,
            percentage_difference=0.0,
            status="Normal",
            threshold_used=threshold,
            recommendation="No generation expected at this time (irradiation = 0).",
        )

    # Percentage by which actual lags behind predicted
    # Negative means actual < predicted (underperformance)
    # Positive means actual > predicted (overperformance, rare but possible)
    percentage_difference = ((actual_ac_power - predicted_ac_power) / predicted_ac_power) * 100.0

    # Anomaly: actual is significantly below predicted (underperformance only)
    is_anomaly = (predicted_ac_power - actual_ac_power) / predicted_ac_power > threshold

    if is_anomaly:
        status = "Potential Performance Anomaly"
        recommendation = (
            "Significant underperformance detected. "
            "Recommend immediate inspection of the solar plant. "
            "Possible reasons may include: panel soiling, inverter issues, "
            "shading, sensor malfunction, or equipment degradation. "
            "Do NOT rely solely on this alert for diagnosis."
        )
    else:
        status = "Normal"
        recommendation = "Plant is operating within expected performance range."

    return MonitorResult(
        predicted_ac_power=round(predicted_ac_power, 4),
        actual_ac_power=round(actual_ac_power, 4),
        absolute_error=round(absolute_error, 4),
        percentage_difference=round(percentage_difference, 2),
        status=status,
        threshold_used=threshold,
        recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# Predict helper: turns user inputs into a model prediction
# ---------------------------------------------------------------------------

def predict_ac_power(
    model,
    scaler,
    ambient_temp: float,
    module_temp: float,
    irradiation: float,
    hour: int,
    minute: int,
) -> float:
    """
    Produces a single AC Power prediction from user-supplied weather inputs.

    Parameters
    ----------
    model       : trained sklearn/XGBoost model loaded from joblib payload.
    scaler      : fitted StandardScaler loaded from joblib.
    ambient_temp: Ambient temperature (°C).
    module_temp : Module/panel temperature (°C).
    irradiation : Solar irradiation index.
    hour        : Hour of day (0-23).
    minute      : Minute (0-59).

    Returns
    -------
    float : Predicted AC Power (kW). Clipped to ≥ 0.
    """

    time_of_day = hour + minute / 60.0
    sin_time = np.sin(2 * np.pi * time_of_day / 24.0)
    cos_time = np.cos(2 * np.pi * time_of_day / 24.0)

    feature_names = [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "Hour",
        "Minute",
        "sin_time",
        "cos_time",
    ]

    raw = pd.DataFrame(
        [[ambient_temp, module_temp, irradiation, hour, minute, sin_time, cos_time]],
        columns=feature_names,
    )

    scaled = scaler.transform(raw)
    prediction = float(model.predict(scaled)[0])
    return max(0.0, round(prediction, 4))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Monitor Self-Test ===\n")

    # Scenario 1: Normal operation
    r1 = monitor_performance(predicted_ac_power=900.0, actual_ac_power=880.0)
    print("Scenario 1 — Normal:")
    print(f"  Status             : {r1.status}")
    print(f"  Absolute Error     : {r1.absolute_error:.2f} kW")
    print(f"  Percentage Diff    : {r1.percentage_difference:.1f}%")
    print(f"  Recommendation     : {r1.recommendation}\n")

    # Scenario 2: Anomaly (30% drop)
    r2 = monitor_performance(predicted_ac_power=900.0, actual_ac_power=600.0)
    print("Scenario 2 — Anomaly (33% underperformance):")
    print(f"  Status             : {r2.status}")
    print(f"  Absolute Error     : {r2.absolute_error:.2f} kW")
    print(f"  Percentage Diff    : {r2.percentage_difference:.1f}%")
    print(f"  Recommendation     : {r2.recommendation}\n")

    # Scenario 3: No sun (predicted 0)
    r3 = monitor_performance(predicted_ac_power=0.0, actual_ac_power=0.0)
    print("Scenario 3 — Night-time (no irradiation):")
    print(f"  Status             : {r3.status}")
    print(f"  Recommendation     : {r3.recommendation}")
