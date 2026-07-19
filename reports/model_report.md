# Preprocessing & Model Evaluation Report

This report presents the findings of the Preprocessing and Split phase (Phase 5) and the Model Training and Evaluation phase (Phase 6) of the Solar Power Generation Forecasting & Performance Monitoring System.

---

## 1. Preprocessing & Data Preparation (Phase 5)

### Feature Selection
The predictors selected for machine learning are restricted to variables that are readily input by users in the operational dashboard:
*   **Weather Inputs**: `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`
*   **Time Inputs**: `Hour`, `Minute`, `sin_time`, `cos_time` (cyclical time-of-day features)

> [!IMPORTANT]
> **DC Power & Yield Exclusion**:
> *   `DC_POWER` (highly correlated with AC Power) is excluded from features. If included, the models trivially divide it by 10 (Plant 1) or multiply by 0.98 (Plant 2).
> *   `DAILY_YIELD` and `TOTAL_YIELD` are excluded as they are cumulative output metrics, not weather/time predictors.

### Preventing Temporal Data Leakage (Chronological Split)
In solar forecasting, adjacent intervals are highly correlated. Standard random shuffling splits cause **look-ahead leakage** (training on data points surrounding a test record).
*   *Solution*: We sorted the data chronologically by `DATE_TIME` and performed a **80/20 time-based split**:
    *   **Plant 1 Training**: 2020-05-15 to 2020-06-11 (30,653 records)
    *   **Plant 1 Testing**: 2020-06-11 to 2020-06-17 (7,664 records)
    *   **Plant 2 Training**: 2020-05-15 to 2020-06-11 (27,958 records)
    *   **Plant 2 Testing**: 2020-06-11 to 2020-06-17 (6,990 records)

### Clean Capacity Baseline Filtering (Anomalous Outage Removal)
*   **The Issue**: During EDA, we discovered that multiple inverters in Plant 2 were offline for extended periods (resulting in actual AC Power = 0 under high sun).
*   **The Problem**: Training the model on these outage periods forces the model to expect low power during high sun. This compromises the "expected capacity" baseline and masks anomalies during monitoring.
*   **The Solution**: We filtered out severe outages (defined as records where `IRRADIATION > 0.1` but `AC_POWER == 0`).
    *   This removed **63 records** from Plant 1 and **3,774 records** from Plant 2.
    *   This filtering improved Plant 2's model $R^2$ score dramatically from **51.2% to 85.5%**, and reduced its root mean squared error by **45.3%** (from 207 kW to 112.8 kW)!

### Feature Scaling
To support regularization in Linear Regression and ensure distance calculations are stable:
*   We fit a `StandardScaler` **only on the training set** to prevent data leakage.
*   The scaler was used to transform both the train and test splits.
*   Scalers were saved as [plant_1_scaler.joblib](file:///c:/Users/jolly/OneDrive/Desktop/solar-power-forcasting/models/plant_1_scaler.joblib) and [plant_2_scaler.joblib](file:///c:/Users/jolly/OneDrive/Desktop/solar-power-forcasting/models/plant_2_scaler.joblib) for dashboard use.

---

## 2. Model Performance & Evaluation (Phase 6)

We trained three regression models for each plant:
1.  **Linear Regression**: Serves as our parametric baseline.
2.  **Random Forest Regressor** (`n_estimators=100`, `random_state=42`): Non-parametric tree ensemble.
3.  **XGBoost Regressor** (`n_estimators=100`, `random_state=42`, `learning_rate=0.1`): Gradient boosted trees.

### Plant 1 Evaluation Results (Clean Test Set)

| Model | MAE (kW) | RMSE (kW) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 31.0300 | 48.5495 | 0.9811 |
| **Random Forest** | 27.8446 | 48.1691 | 0.9814 |
| **XGBoost** | **25.9681** | **44.9563** | **0.9838** |

### Plant 2 Evaluation Results (Clean Test Set)

| Model | MAE (kW) | RMSE (kW) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 56.4660 | 115.3060 | 0.8483 |
| **Random Forest** | 52.4501 | 117.7810 | 0.8417 |
| **XGBoost** | **49.5333** | **112.8080** | **0.8548** |

---

## 3. Model Selection & Rationale

### The Best Model: **XGBoost**
For both Plant 1 and Plant 2, the **XGBoost Regressor** achieved the best overall performance:
*   **Plant 1**: Highest $R^2$ (**98.38%**), lowest MAE (**25.97 kW**), and lowest RMSE (**44.96 kW**).
*   **Plant 2**: Highest $R^2$ (**85.48%**), lowest MAE (**49.53 kW**), and lowest RMSE (**112.81 kW**).

### Technical Rationale:
1.  **Non-Linear Interactions**: Solar cell conversion efficiency is highly non-linear. Power increases with module temperature up to ~50°C and then degrades (thermal derating). Tree-based models easily handle this non-linear interaction, whereas Linear Regression requires manual polynomial features.
2.  **Gradient Boosting Advantage**: Unlike Random Forest (which averages independent tree predictions), XGBoost builds trees sequentially. Each tree corrects the residual errors of its predecessor, enabling the model to learn complex boundary profiles (e.g. rapid sunrise power jumps or inverter clipping at high irradiation) with higher precision.
3.  **Regularization**: XGBoost's built-in L1/L2 regularization prevents overfitting on localized weather variations, ensuring high generalizability to the test partition.

### Saved Payload Details:
The best models were exported as serialized dictionary payloads:
*   [plant_1_best_model.joblib](file:///c:/Users/jolly/OneDrive/Desktop/solar-power-forcasting/models/plant_1_best_model.joblib)
*   [plant_2_best_model.joblib](file:///c:/Users/jolly/OneDrive/Desktop/solar-power-forcasting/models/plant_2_best_model.joblib)
Each payload includes the model instance, metrics, features, test set predictions, and test metadata for seamless visualization in the Streamlit application.
