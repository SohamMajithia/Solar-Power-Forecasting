"""
Solar Power Generation Forecasting & Performance Monitoring System
===================================================================
AHA Solar Technologies Pvt. Ltd.  |  Internship Project Dashboard

Pages
-----
1. Home                — Project overview, KPIs, model summary
2. Historical Dashboard — Interactive charts with filters
3. Power Prediction     — Predict AC Power from weather inputs
4. Performance Monitor  — Detect performance anomalies
5. Model Performance    — Comparison tables, residuals, feature importance
"""

import sys
import os

# Ensure the project root is on the path so src.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.config import (
    PLANT_1_CLEAN_PATH,
    PLANT_2_CLEAN_PATH,
    MODELS_DIR,
    ANOMALY_PERCENTAGE_THRESHOLD,
)
from src.monitor import monitor_performance, predict_ac_power

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AHA Solar Power Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Theme / CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #f2f2f7; }

.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #f2f2f7; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.12);
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 12px;
    padding: 16px 20px;
    backdrop-filter: blur(10px);
}

.card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
    color: #f7f7ff;
}

.card, .anomaly-box, .normal-box {
    color: #f7f7ff;
}

.card p,
.card h2,
.card h3,
.card ul,
.card li,
.card a,
.card span {
    color: #f7f7ff !important;
}

.anomaly-box {
    background: rgba(255, 80, 80, 0.18);
    border: 1px solid rgba(255, 80, 80, 0.75);
    border-radius: 12px;
    padding: 18px 22px;
    color: #ffe6e6;
}

.normal-box {
    background: rgba(50, 220, 130, 0.18);
    border: 1px solid rgba(50, 220, 130, 0.75);
    border-radius: 12px;
    padding: 18px 22px;
    color: #e9fff0;
}

h1, h2, h3 { color: #ffffff !important; }
h1 { font-weight: 700 !important; font-size: 2.1rem !important; }
h2 { font-weight: 600 !important; font-size: 1.5rem !important; }

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Cached Data Loaders
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_plant_data(plant: str) -> pd.DataFrame:
    path = PLANT_1_CLEAN_PATH if plant == "Plant 1" else PLANT_2_CLEAN_PATH
    df = pd.read_csv(path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    df["Hour"] = df["DATE_TIME"].dt.hour
    df["Date"] = df["DATE_TIME"].dt.date
    return df


@st.cache_resource
def load_model_payload(plant: str):
    key = "plant_1" if plant == "Plant 1" else "plant_2"
    model_path = MODELS_DIR / f"{key}_best_model.joblib"
    scaler_path = MODELS_DIR / f"{key}_scaler.joblib"
    payload = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return payload, scaler


def mpl_style():
    """Consistent dark matplotlib style for all charts."""
    return {
        "figure.facecolor": "#1e1e2e",
        "axes.facecolor":   "#1e1e2e",
        "axes.edgecolor":   "#444",
        "axes.labelcolor":  "#ccc",
        "xtick.color":      "#999",
        "ytick.color":      "#999",
        "text.color":       "#eee",
        "grid.color":       "#333",
        "grid.linestyle":   ":",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ AHA Solar")
    st.markdown("**Solar Power Forecasting**  \n*Performance Monitoring System*")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📊 Historical Dashboard",
            "⚡ Power Prediction",
            "🔍 Performance Monitor",
            "🤖 Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    plant = st.selectbox("Select Plant", ["Plant 1", "Plant 2"])
    st.caption(f"Plant ID: {'4135001' if plant == 'Plant 1' else '4136001'}")
    st.divider()
    st.caption("AHA Solar Technologies Pvt. Ltd.")
    st.caption("Internship Project · 2026")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1 — HOME
# ──────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("# ☀️ Solar Power Generation Forecasting")
    st.markdown("### Performance Monitoring System — AHA Solar Technologies Pvt. Ltd.")
    st.divider()

    df = load_plant_data(plant)
    payload, _ = load_model_payload(plant)

    total_inverters = df["SOURCE_KEY"].nunique()
    peak_ac = df["AC_POWER"].max()
    mean_daily = (
        df.groupby(["Date", "SOURCE_KEY"])["DAILY_YIELD"]
        .max()
        .groupby(level=0)
        .sum()
        .mean()
    )
    days_monitored = (df["DATE_TIME"].max() - df["DATE_TIME"].min()).days

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monitoring Period", f"{days_monitored} days", "May 15 – Jun 17, 2020")
    c2.metric("Active Inverters", total_inverters, "per plant")
    c3.metric("Peak AC Power", f"{peak_ac:,.1f} kW", "per inverter")
    c4.metric("Avg Daily Plant Yield", f"{mean_daily:,.0f} kWh", f"all {total_inverters} inverters")

    st.divider()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### Project Overview")
        st.markdown("""
<div class="card">
<p>This system was developed as an internship project for <strong>AHA Solar Technologies Pvt. Ltd.</strong>
It provides end-to-end solar power analytics — from raw sensor data ingestion to ML-driven
performance anomaly detection.</p>

<b>Core Capabilities:</b>
<ul>
  <li>Predict expected AC Power from weather inputs</li>
  <li>Compare predicted vs actual power in real-time</li>
  <li>Flag significant deviations as <em>Potential Performance Anomalies</em></li>
  <li>Interactive historical data exploration with filters</li>
</ul>

<b>Data Sources:</b>
<ul>
  <li>Generation Data: 15-min intervals per inverter</li>
  <li>Weather Sensor Data: Irradiation, Ambient &amp; Module Temperature</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("### Best Model Summary")
        m = payload["metrics"]
        model_name = payload["model_name"]
        st.markdown(f"""
<div class="card">
<p><b>Selected Model:</b> {model_name}</p>
<p><b>Plant:</b> {plant}</p>
<hr style="border-color:rgba(255,255,255,0.1)"/>
<p><b>MAE:</b> {m['mae']:.2f} kW</p>
<p><b>RMSE:</b> {m['rmse']:.2f} kW</p>
<p><b>R² Score:</b> {m['r2']:.4f}</p>
<hr style="border-color:rgba(255,255,255,0.1)"/>
<p>XGBoost was selected for its superior ability to model <b>non-linear weather-to-power
relationships</b> and its built-in regularization, outperforming both Linear Regression
and Random Forest across all metrics.</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2 — HISTORICAL DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📊 Historical Dashboard":
    st.markdown(f"# 📊 Historical Dashboard — {plant}")
    st.divider()

    df = load_plant_data(plant)

    with st.sidebar:
        st.markdown("### Filters")
        dates = sorted(df["Date"].unique())
        date_range = st.select_slider(
            "Date Range",
            options=dates,
            value=(dates[0], dates[-1]),
        )
        inverters = ["All"] + sorted(df["SOURCE_KEY"].unique().tolist())
        selected_inv = st.selectbox("Inverter", inverters)

    mask = (df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])
    if selected_inv != "All":
        mask &= df["SOURCE_KEY"] == selected_inv
    fdf = df[mask].copy()

    # Row 1: AC Power trend
    st.markdown("### Power Generation Trend")
    daily_ac = fdf.groupby("Date")["AC_POWER"].mean()
    with plt.rc_context(mpl_style()):
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.fill_between(daily_ac.index, daily_ac.values, alpha=0.3, color="#667eea")
        ax.plot(daily_ac.index, daily_ac.values, color="#667eea", linewidth=2)
        ax.set_title(f"Average Daily AC Power — {plant}", color="#eee")
        ax.set_ylabel("AC Power (kW)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Row 2: Temperature & Irradiation
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Temperature Trends")
        daily_temp = fdf.groupby("Date")[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]].mean()
        with plt.rc_context(mpl_style()):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(daily_temp.index, daily_temp["AMBIENT_TEMPERATURE"],
                    label="Ambient", color="#f39c12", linewidth=2)
            ax.plot(daily_temp.index, daily_temp["MODULE_TEMPERATURE"],
                    label="Module", color="#e74c3c", linewidth=2, linestyle="--")
            ax.set_ylabel("Temperature (°C)")
            ax.legend()
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with c2:
        st.markdown("### Solar Irradiation Trend")
        daily_irr = fdf.groupby("Date")["IRRADIATION"].mean()
        with plt.rc_context(mpl_style()):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.fill_between(daily_irr.index, daily_irr.values, alpha=0.4, color="#f1c40f")
            ax.plot(daily_irr.index, daily_irr.values, color="#f1c40f", linewidth=2)
            ax.set_ylabel("Irradiation Index")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Row 3: Daily Yield & Correlation
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### Daily Yield per Inverter")
        daily_yield = (
            fdf.groupby(["Date", "SOURCE_KEY"])["DAILY_YIELD"]
            .max()
            .groupby("Date")
            .mean()
        )
        with plt.rc_context(mpl_style()):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(daily_yield.index, daily_yield.values, color="#2ecc71", alpha=0.8, width=0.6)
            ax.set_ylabel("Avg Daily Yield (kWh)")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with c4:
        st.markdown("### Correlation Heatmap")
        cols = ["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
        corr = fdf[cols].corr()
        with plt.rc_context(mpl_style()):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                        ax=ax, vmin=-1, vmax=1, linewidths=0.5,
                        cbar_kws={"shrink": 0.8})
            ax.tick_params(colors="#ccc")
            fig.tight_layout()
        st.pyplot(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3 — POWER PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
elif page == "⚡ Power Prediction":
    st.markdown(f"# ⚡ Power Prediction — {plant}")
    st.markdown("Enter current weather readings to predict the expected AC Power output.")
    st.divider()

    payload, scaler = load_model_payload(plant)
    model = payload["model"]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Weather Inputs")
        amb_temp = st.slider("Ambient Temperature (°C)", 10.0, 50.0, 28.0, 0.1)
        mod_temp = st.slider("Module Temperature (°C)", 10.0, 80.0, 38.0, 0.1)
        irr = st.slider("Irradiation Index", 0.0, 1.5, 0.5, 0.01)
        hour = st.slider("Hour of Day", 0, 23, 12)
        minute = st.selectbox("Minute", [0, 15, 30, 45], index=0)
        predict_btn = st.button("Predict AC Power", use_container_width=True)

    with col2:
        st.markdown("### Prediction Result")
        if predict_btn:
            pred = predict_ac_power(model, scaler, amb_temp, mod_temp, irr, hour, minute)
            st.markdown(f"""
<div class="card">
<h2 style="font-size:2.8rem; color:#a78bfa; margin-bottom:4px;">{pred:,.2f} kW</h2>
<p style="color:#999; margin:0;">Predicted AC Power Output</p>
<hr style="border-color:rgba(255,255,255,0.1); margin:16px 0;"/>
<p><b>Plant:</b> {plant}</p>
<p><b>Model:</b> {payload['model_name']}</p>
<p><b>Time:</b> {hour:02d}:{minute:02d}</p>
<p><b>R² Score:</b> {payload['metrics']['r2']:.4f}</p>
</div>
""", unsafe_allow_html=True)

            with plt.rc_context(mpl_style()):
                fig, ax = plt.subplots(figsize=(6, 3))
                max_cap = 1450.0
                pct = min(pred / max_cap, 1.0)
                bar_color = "#667eea" if pct > 0.3 else "#e74c3c"
                ax.barh(["Predicted Output"], [pred], color=bar_color, height=0.4)
                ax.barh(["Max Capacity"], [max_cap], color="#333", height=0.4)
                ax.set_xlim(0, max_cap * 1.05)
                ax.set_xlabel("AC Power (kW)")
                ax.axvline(pred, color="#a78bfa", linestyle="--", linewidth=1.5)
                ax.set_title(f"Predicted: {pred:,.1f} kW / Max: {max_cap:,.0f} kW")
                fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Adjust the sliders and click **Predict AC Power** to see the result.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4 — PERFORMANCE MONITOR
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Performance Monitor":
    st.markdown(f"# 🔍 Performance Monitor — {plant}")
    st.markdown("Compare predicted vs actual power to detect potential performance anomalies.")
    st.divider()

    payload, scaler = load_model_payload(plant)
    model = payload["model"]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Inputs")
        amb_temp = st.slider("Ambient Temperature (°C)", 10.0, 50.0, 28.0, 0.1, key="mon_amb")
        mod_temp = st.slider("Module Temperature (°C)", 10.0, 80.0, 38.0, 0.1, key="mon_mod")
        irr = st.slider("Irradiation Index", 0.0, 1.5, 0.5, 0.01, key="mon_irr")
        hour = st.slider("Hour of Day", 0, 23, 12, key="mon_hr")
        minute = st.selectbox("Minute", [0, 15, 30, 45], index=0, key="mon_min")
        actual_ac = st.number_input(
            "Actual AC Power (kW) — from sensor",
            min_value=0.0, max_value=2000.0, value=400.0, step=10.0,
        )
        threshold = st.slider(
            "Anomaly Threshold (%)", 5, 40,
            int(ANOMALY_PERCENTAGE_THRESHOLD * 100), 1,
        )
        check_btn = st.button("Check Performance", use_container_width=True)

    with col2:
        st.markdown("### Analysis Result")
        if check_btn:
            predicted = predict_ac_power(model, scaler, amb_temp, mod_temp, irr, hour, minute)
            result = monitor_performance(predicted, actual_ac, threshold / 100.0)

            is_anomaly = result.status == "Potential Performance Anomaly"
            css_class = "anomaly-box" if is_anomaly else "normal-box"
            heading = "Potential Performance Anomaly" if is_anomaly else "Normal Operation"
            st.markdown(f"""
<div class="{css_class}">
<h3>{heading}</h3>
<p>{result.recommendation}</p>
</div>
""", unsafe_allow_html=True)

            st.divider()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted (kW)", f"{result.predicted_ac_power:,.2f}")
            m2.metric("Actual (kW)", f"{result.actual_ac_power:,.2f}")
            m3.metric("Absolute Error", f"{result.absolute_error:,.2f} kW")
            m4.metric("% Difference", f"{result.percentage_difference:.1f}%",
                      delta_color="inverse")

            with plt.rc_context(mpl_style()):
                fig, ax = plt.subplots(figsize=(6, 3))
                labels = ["Predicted (Expected)", "Actual (Measured)"]
                vals = [result.predicted_ac_power, result.actual_ac_power]
                bar_colors = ["#667eea", "#e74c3c" if is_anomaly else "#2ecc71"]
                bars = ax.bar(labels, vals, color=bar_colors, width=0.4)
                ax.bar_label(bars, fmt="%.1f kW", color="#eee", padding=4)
                ax.set_ylabel("AC Power (kW)")
                ax.set_title("Expected vs Actual AC Power")
                ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 10)
                fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Set the weather readings and actual power, then click **Check Performance**.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 5 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.markdown(f"# 🤖 Model Performance — {plant}")
    st.divider()

    payload, _ = load_model_payload(plant)
    model      = payload["model"]
    m          = payload["metrics"]
    preds      = payload["predictions"]
    y_test     = payload["y_test"]
    features   = payload["features"]
    model_name = payload["model_name"]

    # Metrics row
    st.markdown("### Best Model Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", model_name)
    c2.metric("MAE", f"{m['mae']:.4f} kW")
    c3.metric("RMSE", f"{m['rmse']:.4f} kW")
    c4.metric("R² Score", f"{m['r2']:.4f}")

    st.divider()

    # Model comparison table
    st.markdown("### All Models Comparison")
    comparison_data = {
        "Plant 1": {
            "Linear Regression": {"MAE": 31.03, "RMSE": 48.55, "R2": 0.9811},
            "Random Forest":     {"MAE": 27.84, "RMSE": 48.17, "R2": 0.9814},
            "XGBoost":           {"MAE": 25.97, "RMSE": 44.96, "R2": 0.9838},
        },
        "Plant 2": {
            "Linear Regression": {"MAE": 56.47, "RMSE": 115.31, "R2": 0.8483},
            "Random Forest":     {"MAE": 52.45, "RMSE": 117.78, "R2": 0.8417},
            "XGBoost":           {"MAE": 49.53, "RMSE": 112.81, "R2": 0.8548},
        },
    }

    cmp_df = pd.DataFrame(comparison_data[plant]).T.reset_index()
    cmp_df.columns = ["Model", "MAE (kW)", "RMSE (kW)", "R2 Score"]
    cmp_df["Best"] = cmp_df["Model"] == model_name
    st.dataframe(
        cmp_df.drop(columns=["Best"]).style.apply(
            lambda row: ["background: rgba(103,126,234,0.3)"] * len(row)
            if cmp_df.loc[row.name, "Best"] else [""] * len(row),
            axis=1,
        ).format({"MAE (kW)": "{:.2f}", "RMSE (kW)": "{:.2f}", "R2 Score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    col1, col2 = st.columns(2)

    # Feature importance
    with col1:
        st.markdown("### Feature Importance (XGBoost)")
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
            with plt.rc_context(mpl_style()):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(fi.index, fi.values, color="#667eea", alpha=0.85)
                ax.set_xlabel("Importance Score")
                ax.set_title("XGBoost Feature Importances")
                fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

    # Residuals plot
    with col2:
        st.markdown("### Residuals vs Predicted (Test Set)")
        residuals = y_test - preds
        with plt.rc_context(mpl_style()):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(preds, residuals, alpha=0.12, s=6, color="#a78bfa")
            ax.axhline(0, color="#e74c3c", linewidth=1.5, linestyle="--")
            ax.set_xlabel("Predicted AC Power (kW)")
            ax.set_ylabel("Residual (kW)")
            ax.set_title("Residuals vs Predicted (Test Set)")
            fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Actual vs Predicted
    st.markdown("### Actual vs Predicted (first 500 test samples)")
    with plt.rc_context(mpl_style()):
        fig, ax = plt.subplots(figsize=(14, 4))
        n = min(500, len(y_test))
        ax.plot(range(n), y_test[:n], label="Actual", color="#2ecc71", linewidth=1.2)
        ax.plot(range(n), preds[:n], label="Predicted", color="#a78bfa",
                linewidth=1.2, linestyle="--")
        ax.set_xlabel("Sample Index (Test Set)")
        ax.set_ylabel("AC Power (kW)")
        ax.set_title("Actual vs Predicted AC Power — Test Set")
        ax.legend()
        fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
