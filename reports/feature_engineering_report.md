# Feature Engineering Report

This report documents the feature engineering phase (Phase 4) of the Solar Power Generation Forecasting & Performance Monitoring System. It explains the purpose and rationale behind each newly created feature.

---

## 1. Engineered Features List & Descriptions

| Feature Name | Source Column | Data Type | Formula / Encoding | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **`Hour`** | `DATE_TIME` | Integer (0-23) | `dt.hour` | Captures the macro diurnal cycle (day vs. night solar dynamics). |
| **`Minute`** | `DATE_TIME` | Integer (0-59) | `dt.minute` | Captures micro-intervals within the hour (15-minute resolution). |
| **`Day`** | `DATE_TIME` | Integer (1-31) | `dt.day` | Captures changes in solar path/weather over the 34-day monitoring window. |
| **`Month`** | `DATE_TIME` | Integer (1-12) | `dt.month` | Captures seasonal variation (e.g. solstice, solar elevation angles). |
| **`Weekday`** | `DATE_TIME` | Integer (0-6) | `dt.weekday` | Captures weekly cycles (e.g. routine weekly maintenance or grid patterns). |
| **`Weekend`** | `DATE_TIME` | Binary (0 or 1) | `1` if weekday $\ge 5$ else `0` | Identifies weekend operation periods. |
| **`sin_time`** | `DATE_TIME` | Float (-1.0 to 1.0) | $\sin\left(\frac{2\pi \cdot t}{24}\right)$ | Cyclical representation of the time of day (Y-coordinate). |
| **`cos_time`** | `DATE_TIME` | Float (-1.0 to 1.0) | $\cos\left(\frac{2\pi \cdot t}{24}\right)$ | Cyclical representation of the time of day (X-coordinate). |

---

## 2. Deep Dive: Cyclical Time-of-Day Encoding

### The Problem with Linear Time
Standard time representations like hours ($0, 1, 2, \dots, 23$) treat time as a linear numerical scale. However, time is circular:
*   `23:45` is physically adjacent to `00:00`.
*   A linear model (such as Linear Regression) treats $23.75$ and $0.0$ as opposite ends of a spectrum. The model fails to recognize that temperatures and solar output at midnight are continuously connected to late night.
*   Tree-based models (Random Forest, XGBoost) can partition linear time using vertical splits, but they require more splits to approximate a circular boundary.

### The Cyclical Solution
By converting the time of day (represented as decimal hours $t = \text{Hour} + \frac{\text{Minute}}{60}$) into sine and cosine components:
$$\text{sin\_time} = \sin\left(\frac{2\pi \cdot t}{24}\right)$$
$$\text{cos\_time} = \cos\left(\frac{2\pi \cdot t}{24}\right)$$

This maps each time of day onto the coordinates of a unit circle:
*   `12:00` (Noon) maps to $(\sin(\pi), \cos(\pi)) = (0.0, -1.0)$.
*   `00:00` (Midnight) maps to $(\sin(0), \cos(0)) = (0.0, 1.0)$.
*   `06:00` (Sunrise) maps to $(\sin(\pi/2), \cos(\pi/2)) = (1.0, 0.0)$.
*   `18:00` (Sunset) maps to $(\sin(3\pi/2), \cos(3\pi/2)) = (-1.0, 0.0)$.

This continuous representation allows all machine learning models (especially Linear Regression) to capture the smooth transitions of the diurnal cycle.

---

## 3. Engineering Decisions: Lags & Rolling Averages

### Decision: Exclusion of Lag Features and Rolling Averages
In time-series forecasting, lag features (e.g. AC Power or Irradiation at $t-1$, $t-2$) and rolling averages are often used to capture trends and temporal dependencies. However, for this project, we have explicitly excluded them from the ML feature set.

#### Rationale:
1.  **State Constraints in Production Dashboard**: The Streamlit application's pages (Page 3: *Power Prediction* and Page 4: *Performance Monitor*) ask the user to input **instantaneous weather readings** (Ambient Temperature, Module Temperature, and Irradiation) for a specific Hour and Minute. 
2.  **Operational Usability**: If the model required lag features (e.g., "Irradiation 15 minutes ago" or "AC Power 30 minutes ago"), the user would be forced to look up and input historical sequences on the dashboard. This would make the prediction and monitoring features cumbersome and impractical for quick checks.
3.  **Stateless Inference**: Excluding lags allows the model to remain **stateless**, performing predictions purely based on the current meteorological conditions. The model learns the direct physical function of mapping solar irradiation and module temperature to AC electrical output.
4.  **Preventing Data Leakage**: In active performance monitoring, using lag features of the target variable (`AC_POWER`) can mask anomalies. If an inverter fails, its AC Power drops to 0. A model using lag features of AC Power would see that the previous power was low/zero and predict that the current power should also be low/zero, thereby failing to highlight the performance anomaly!
