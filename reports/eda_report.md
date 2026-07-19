# Exploratory Data Analysis (EDA) Report

This report presents the findings of the Exploratory Data Analysis (EDA) for the Solar Power Generation Forecasting & Performance Monitoring System. It compares Plant 1 and Plant 2 across their generation patterns, meteorological relationships, and individual inverter performance.

---

## 1. Comparative Plant Statistics

Below is a high-level statistical comparison of Plant 1 and Plant 2 over the 34-day monitoring period (May 15, 2020 – June 17, 2020).

| Metric | Plant 1 | Plant 2 |
| :--- | :--- | :--- |
| **Total Generation Records** | 68,778 | 67,698 |
| **Unique Inverters** | 22 | 22 |
| **Mean AC Power Output** | 307.80 kW (Max: 1410.95 kW) | 241.28 kW (Max: 1385.42 kW) |
| **Mean DC Power Input** | 314.74 kW (Max: 1447.11 kW) | 246.70 kW (Max: 1420.93 kW) |
| **Average Ambient Temperature** | 25.56°C (Range: 20.40°C – 35.25°C) | 27.99°C (Range: 20.94°C – 39.18°C) |
| **Average Module Temperature** | 31.25°C (Range: 18.14°C – 65.55°C) | 32.61°C (Range: 20.27°C – 66.64°C) |
| **Mean Solar Irradiation** | 0.2323 W/m² (Max: 1.2217 W/m²) | 0.2292 W/m² (Max: 1.0988 W/m²) |
| **Mean Daily Yield per Inverter** | 7,193.58 kWh (Max: 9,163.00 kWh) | 6,256.38 kWh (Max: 9,873.00 kWh) |

### Key Business Takeaways:
*   **Plant 1 is more efficient**: Despite having slightly lower ambient temperatures (average 25.56°C vs 27.99°C), Plant 1 achieved a **15% higher average daily yield** (7,193 kWh vs 6,256 kWh) and a higher average AC power output (307.8 kW vs 241.3 kW).
*   **Thermal Derating in Plant 2**: Plant 2 experiences significantly higher ambient temperatures (up to 39.18°C). Solar panels lose efficiency as temperatures rise (thermal coefficient losses), which explains why Plant 2 produces less energy despite having similar module temperatures.

---

## 2. Feature Distribution Analysis

![Feature Distributions](file:///c:/Users/jolly/OneDrive/Desktop/solar-power-forcasting/reports/figures/distributions.png)

### Business Insights:
*   **Zero-Generation Dominance**: The distributions for `AC_POWER`, `DC_POWER`, and `IRRADIATION` are heavily skewed towards zero. This is a natural consequence of the 24-hour cycle, as solar plants do not generate power during night-time hours (~50% of the data).
*   **Temperature Ranges**: Ambient temperature is normally distributed, centering around 25°C for Plant 1 and 28°C for Plant 2. Module temperatures, however, show a bimodal distribution: a peak around 20-25°C (night hours when modules cool to ambient levels) and a broad peak between 35-55°C (active day hours when direct sunlight heats the panels far above ambient air temperature).
*   **Inverter Efficiency Uniformity**: Standardizing Plant 1's DC power allows us to see that the shape of DC and AC distributions are almost identical, indicating high inverter conversion efficiency (97-98%) across both plants under normal operating conditions.

---

## 3. Correlation Matrix

![Correlation Heatmap](file:///c:/Users/jolly/OneDrive/Desktop/solar-power-forcasting/reports/figures/correlation_heatmap.png)

### Business Insights:
*   **Perfect Linear Relationship (DC to AC)**: In both plants, `DC_POWER` and `AC_POWER` are perfectly correlated ($r = 1.00$). This indicates that the inverters convert DC power to AC power with constant linear efficiency.
*   **Strong Weather Correlates**: `AC_POWER` is most strongly correlated with `IRRADIATION` ($r = 0.97$ in Plant 1, $0.73$ in Plant 2) and `MODULE_TEMPERATURE` ($r = 0.84$ in Plant 1, $0.80$ in Plant 2). This confirms that irradiation is the primary driver of solar generation, and module temperature is a strong proxy for solar intensity.
*   **The Temperature Paradox**: There is a strong correlation between ambient temperature and module temperature ($r = 0.88$ for P1, $0.92$ for P2), but module temperature correlates much higher with AC power than ambient temperature does. This is because module temperature rises rapidly due to direct solar absorption, making it a better indicator of actual sunlight intensity.

---

## 4. Hourly Generation Profiles

![Hourly Generation Profiles](file:///c:/Users/jolly/OneDrive/Desktop/solar-power-forcasting/reports/figures/hourly_generation.png)

### Business Insights:
*   **Standard Solar Bell Curve**: Generation begins around 06:00, peaks between 11:30 and 13:00, and returns to zero by 18:30.
*   **Plant 1 Peak Output**: Plant 1 peaks at an average AC output of ~830 kW around 12:30.
*   **Plant 2 Peak Output**: Plant 2 peaks at an average AC output of ~650 kW around 12:30.
*   **Operational Validation**: The clear envelope of the hourly generation curve shows that any substantial power drop during peak hours (10:00 - 15:00) when irradiation is high is an operational anomaly that warrants inspection.

---

## 5. Daily Generation Trends

![Daily Generation](file:///c:/Users/jolly/OneDrive/Desktop/solar-power-forcasting/reports/figures/daily_generation.png)

### Business Insights:
*   **Weather-Driven Volatility**: Daily yields vary dramatically from day to day (e.g., dropping significantly around May 18th and June 4th). These drops represent overcast or rainy days, which are normal occurrences.
*   **Plant-Level Synchronization**: Both plants experience yield drops on the same days (e.g., June 11th and June 14th). This indicates they are located in close geographical proximity, sharing similar regional weather fronts.
*   **Divergent Trends**: On several days (e.g., May 25th - May 30th), Plant 1 significantly outperforms Plant 2, confirming localized shading, soiling, or system downtime in Plant 2.

---

## 6. Weather Variables vs. AC Power Generation

![Weather vs. AC Power](file:///c:/Users/jolly/OneDrive/Desktop/solar-power-forcasting/reports/figures/weather_vs_ac.png)

### Business Insights:
*   **Irradiation vs. AC Power**: Shows a highly linear relationship up to ~0.8 irradiation. Beyond this point, the curves start to bend (clipping), representing inverter capacity limits.
*   **Module Temperature vs. AC Power**: AC power increases with module temperature up to ~50°C. Above 50°C, the power output flattens or decreases slightly. This is physical proof of solar cell efficiency degradation due to high temperatures.
*   **Ambient Temperature vs. AC Power**: Shows a wider scatter. High ambient temperature does not guarantee high generation unless accompanied by high irradiation.

---

## 7. Cumulative Yield Growth

![Yield Growth](file:///c:/Users/jolly/OneDrive/Desktop/solar-power-forcasting/reports/figures/yield_growth.png)

### Business Insights:
*   **Consistent Growth**: Both plants show steady, linear cumulative yield growth, indicating that overall, both assets are generating substantial energy.
*   **Slope Discrepancy**: The slope of Plant 1 is steeper than Plant 2, representing a higher rate of daily energy accumulation. Over time, this cumulative gap expands, highlighting the financial impact of Plant 2's lower operational efficiency.

---

## 8. Inverter-Level Performance Analysis

### Plant 1 Inverter Rankings:
*   **Top Performer**: `adLQvlD726eNBSB` (Average Daily Yield: **7,443.84 kWh**)
*   **Bottom Performers**:
    *   `bvBOhCH3iADSZry` (Average Daily Yield: **6,535.47 kWh**) - **12.2% below top**
    *   `1BY6WEcLGh8j5v7` (Average Daily Yield: **6,639.87 kWh**) - **10.8% below top**
*   **Analysis**: Inverters `bvBOhCH3iADSZry` and `1BY6WEcLGh8j5v7` have lower max AC power capacity (1265 kW and 1300 kW vs ~1400 kW for others). This suggests either smaller solar array sizing (fewer panels connected) or severe permanent shading/degradation issues.

### Plant 2 Inverter Rankings:
*   **Top Performer**: `4UPUqMRk7TRMgml` (Average Daily Yield: **7,703.90 kWh**)
*   **Bottom Performers**:
    *   `Et9kgGMDl729KT4` (Average Daily Yield: **4,579.88 kWh**) - **40.5% below top**
    *   `Quc1TzYxW2pYoWX` (Average Daily Yield: **4,685.32 kWh**) - **39.2% below top**
    *   `LYwnQax7tkwH5Cb` (Average Daily Yield: **4,850.17 kWh**) - **37.0% below top**
*   **Analysis**: The bottom inverters in Plant 2 show extremely low average daily yields (under 5,000 kWh). These large deficits (up to 40% loss compared to the top inverter) indicate significant performance anomalies such as panel soiling, hardware faults, or local shading that require immediate onsite maintenance.
