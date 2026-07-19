import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import (
    PLANT_1_CLEAN_PATH,
    PLANT_2_CLEAN_PATH,
    PROJECT_ROOT,
)

# Setup aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 150,
})

# Color palette
PALETTE_P1 = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Teal/Blue/Orange
PALETTE_P2 = ["#9467bd", "#d62728", "#bcbd22"]  # Purple/Red/Yellow-Green

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data():
    p1 = pd.read_csv(PLANT_1_CLEAN_PATH)
    p2 = pd.read_csv(PLANT_2_CLEAN_PATH)
    p1["DATE_TIME"] = pd.to_datetime(p1["DATE_TIME"])
    p2["DATE_TIME"] = pd.to_datetime(p2["DATE_TIME"])
    return p1, p2

def plot_distributions(p1, p2):
    """Generates distribution plots for numerical features."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    # AC Power
    sns.histplot(p1["AC_POWER"], ax=axes[0, 0], kde=True, color="#1f77b4", label="Plant 1")
    sns.histplot(p2["AC_POWER"], ax=axes[0, 0], kde=True, color="#9467bd", label="Plant 2")
    axes[0, 0].set_title("AC Power Output Distribution")
    axes[0, 0].set_xlabel("AC Power (kW)")
    axes[0, 0].legend()
    
    # DC Power
    sns.histplot(p1["DC_POWER"], ax=axes[0, 1], kde=True, color="#1f77b4", label="Plant 1")
    sns.histplot(p2["DC_POWER"], ax=axes[0, 1], kde=True, color="#9467bd", label="Plant 2")
    axes[0, 1].set_title("DC Power Input Distribution (Standardized Scale)")
    axes[0, 1].set_xlabel("DC Power (kW)")
    axes[0, 1].legend()
    
    # Ambient Temp
    sns.histplot(p1["AMBIENT_TEMPERATURE"], ax=axes[1, 0], kde=True, color="#1f77b4", label="Plant 1")
    sns.histplot(p2["AMBIENT_TEMPERATURE"], ax=axes[1, 0], kde=True, color="#9467bd", label="Plant 2")
    axes[1, 0].set_title("Ambient Temperature Distribution")
    axes[1, 0].set_xlabel("Temperature (°C)")
    axes[1, 0].legend()
    
    # Module Temp
    sns.histplot(p1["MODULE_TEMPERATURE"], ax=axes[1, 1], kde=True, color="#1f77b4", label="Plant 1")
    sns.histplot(p2["MODULE_TEMPERATURE"], ax=axes[1, 1], kde=True, color="#9467bd", label="Plant 2")
    axes[1, 1].set_title("Module Temperature Distribution")
    axes[1, 1].set_xlabel("Temperature (°C)")
    axes[1, 1].legend()
    
    # Irradiation
    sns.histplot(p1["IRRADIATION"], ax=axes[2, 0], kde=True, color="#1f77b4", label="Plant 1")
    sns.histplot(p2["IRRADIATION"], ax=axes[2, 0], kde=True, color="#9467bd", label="Plant 2")
    axes[2, 0].set_title("Solar Irradiation Distribution")
    axes[2, 0].set_xlabel("Irradiation Index")
    axes[2, 0].legend()
    
    # Daily Yield
    sns.histplot(p1["DAILY_YIELD"], ax=axes[2, 1], kde=True, color="#1f77b4", label="Plant 1")
    sns.histplot(p2["DAILY_YIELD"], ax=axes[2, 1], kde=True, color="#9467bd", label="Plant 2")
    axes[2, 1].set_title("Daily Yield Distribution")
    axes[2, 1].set_xlabel("Daily Yield (kWh)")
    axes[2, 1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distributions.png", dpi=300)
    plt.close()

def plot_correlation_heatmap(p1, p2):
    """Generates correlation matrices heatmaps."""
    cols = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    corr_p1 = p1[cols].corr()
    sns.heatmap(corr_p1, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title("Plant 1 Correlation Heatmap")
    
    corr_p2 = p2[cols].corr()
    sns.heatmap(corr_p2, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title("Plant 2 Correlation Heatmap")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

def plot_hourly_generation(p1, p2):
    """Plots average power generation by hour of the day."""
    p1["Hour"] = p1["DATE_TIME"].dt.hour
    p2["Hour"] = p2["DATE_TIME"].dt.hour
    
    hourly_p1 = p1.groupby("Hour")[["AC_POWER", "DC_POWER"]].mean()
    hourly_p2 = p2.groupby("Hour")[["AC_POWER", "DC_POWER"]].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plant 1
    axes[0].plot(hourly_p1.index, hourly_p1["AC_POWER"], label="AC Power", color="#1f77b4", linewidth=2.5)
    axes[0].plot(hourly_p1.index, hourly_p1["DC_POWER"], label="DC Power (Scaled)", color="#ff7f0e", linestyle="--", linewidth=2.5)
    axes[0].set_title("Plant 1 Average Hourly Power Generation Profile")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Power (kW)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # Plant 2
    axes[1].plot(hourly_p2.index, hourly_p2["AC_POWER"], label="AC Power", color="#9467bd", linewidth=2.5)
    axes[1].plot(hourly_p2.index, hourly_p2["DC_POWER"], label="DC Power", color="#2ca02c", linestyle="--", linewidth=2.5)
    axes[1].set_title("Plant 2 Average Hourly Power Generation Profile")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Power (kW)")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hourly_generation.png", dpi=300)
    plt.close()

def plot_daily_generation(p1, p2):
    """Plots total plant daily energy yield over time."""
    p1["Date"] = p1["DATE_TIME"].dt.date
    p2["Date"] = p2["DATE_TIME"].dt.date
    
    # Inverters aggregate daily yield per day
    # Max daily yield per inverter per day represents its final yield for that day
    daily_yield_p1 = p1.groupby(["Date", "SOURCE_KEY"])["DAILY_YIELD"].max().groupby("Date").sum()
    daily_yield_p2 = p2.groupby(["Date", "SOURCE_KEY"])["DAILY_YIELD"].max().groupby("Date").sum()
    
    plt.figure(figsize=(14, 6))
    plt.plot(daily_yield_p1.index, daily_yield_p1.values, label="Plant 1 Total Daily Yield", color="#1f77b4", marker="o", linewidth=2)
    plt.plot(daily_yield_p2.index, daily_yield_p2.values, label="Plant 2 Total Daily Yield", color="#9467bd", marker="s", linewidth=2)
    plt.title("Total Plant Daily Energy Yield (May 15 - June 17, 2020)")
    plt.xlabel("Date")
    plt.ylabel("Daily Energy Yield (kWh)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "daily_generation.png", dpi=300)
    plt.close()

def plot_weather_vs_ac(p1, p2):
    """Plots relationships between weather features and AC Power."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. Ambient Temp vs AC Power
    sns.scatterplot(data=p1, x="AMBIENT_TEMPERATURE", y="AC_POWER", ax=axes[0, 0], color="#1f77b4", alpha=0.1, s=10)
    axes[0, 0].set_title("Plant 1: Ambient Temperature vs AC Power")
    axes[0, 0].set_xlabel("Ambient Temp (°C)")
    axes[0, 0].set_ylabel("AC Power (kW)")
    
    sns.scatterplot(data=p2, x="AMBIENT_TEMPERATURE", y="AC_POWER", ax=axes[0, 1], color="#9467bd", alpha=0.1, s=10)
    axes[0, 1].set_title("Plant 2: Ambient Temperature vs AC Power")
    axes[0, 1].set_xlabel("Ambient Temp (°C)")
    axes[0, 1].set_ylabel("AC Power (kW)")
    
    # 2. Module Temp vs AC Power
    sns.scatterplot(data=p1, x="MODULE_TEMPERATURE", y="AC_POWER", ax=axes[1, 0], color="#1f77b4", alpha=0.1, s=10)
    axes[1, 0].set_title("Plant 1: Module Temperature vs AC Power")
    axes[1, 0].set_xlabel("Module Temp (°C)")
    axes[1, 0].set_ylabel("AC Power (kW)")
    
    sns.scatterplot(data=p2, x="MODULE_TEMPERATURE", y="AC_POWER", ax=axes[1, 1], color="#9467bd", alpha=0.1, s=10)
    axes[1, 1].set_title("Plant 2: Module Temperature vs AC Power")
    axes[1, 1].set_xlabel("Module Temp (°C)")
    axes[1, 1].set_ylabel("AC Power (kW)")
    
    # 3. Irradiation vs AC Power
    sns.scatterplot(data=p1, x="IRRADIATION", y="AC_POWER", ax=axes[2, 0], color="#1f77b4", alpha=0.1, s=10)
    axes[2, 0].set_title("Plant 1: Irradiation vs AC Power")
    axes[2, 0].set_xlabel("Irradiation Index")
    axes[2, 0].set_ylabel("AC Power (kW)")
    
    sns.scatterplot(data=p2, x="IRRADIATION", y="AC_POWER", ax=axes[2, 1], color="#9467bd", alpha=0.1, s=10)
    axes[2, 1].set_title("Plant 2: Irradiation vs AC Power")
    axes[2, 1].set_xlabel("Irradiation Index")
    axes[2, 1].set_ylabel("AC Power (kW)")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "weather_vs_ac.png", dpi=300)
    plt.close()

def plot_yields_over_time(p1, p2):
    """Plots daily and cumulative yields for sample inverters."""
    p1["Date"] = p1["DATE_TIME"].dt.date
    p2["Date"] = p2["DATE_TIME"].dt.date
    
    # Total yield tracks cumulative life energy
    # We plot the mean total yield per day for each plant to see growth over the 34 days
    total_yield_p1 = p1.groupby("Date")["TOTAL_YIELD"].mean()
    total_yield_p2 = p2.groupby("Date")["TOTAL_YIELD"].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].plot(total_yield_p1.index, total_yield_p1.values, label="Plant 1 Avg Total Yield", color="#1f77b4", linewidth=2.5)
    axes[0].set_title("Plant 1: Mean Total Yield Growth")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Total Cumulative Yield (kWh)")
    axes[0].tick_params(axis="x", rotation=45)
    
    axes[1].plot(total_yield_p2.index, total_yield_p2.values, label="Plant 2 Avg Total Yield", color="#9467bd", linewidth=2.5)
    axes[1].set_title("Plant 2: Mean Total Yield Growth")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Total Cumulative Yield (kWh)")
    axes[1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "yield_growth.png", dpi=300)
    plt.close()

def compute_statistics(p1, p2):
    """Computes descriptive and inverter-specific statistics."""
    stats_path = PROJECT_ROOT / "reports" / "plant_statistics.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("PLANT COMPARATIVE STATISTICS\n")
        f.write("============================================================\n\n")
        
        for name, df in [("Plant 1", p1), ("Plant 2", p2)]:
            f.write(f"--- {name} ---\n")
            f.write(f"Total Rows: {len(df)}\n")
            f.write(f"Unique Inverters: {df['SOURCE_KEY'].nunique()}\n")
            f.write(f"Mean AC Power: {df['AC_POWER'].mean():.2f} kW (Max: {df['AC_POWER'].max():.2f} kW)\n")
            f.write(f"Mean DC Power: {df['DC_POWER'].mean():.2f} kW (Max: {df['DC_POWER'].max():.2f} kW)\n")
            f.write(f"Mean Ambient Temp: {df['AMBIENT_TEMPERATURE'].mean():.2f}°C (Range: {df['AMBIENT_TEMPERATURE'].min():.2f}°C to {df['AMBIENT_TEMPERATURE'].max():.2f}°C)\n")
            f.write(f"Mean Module Temp: {df['MODULE_TEMPERATURE'].mean():.2f}°C (Range: {df['MODULE_TEMPERATURE'].min():.2f}°C to {df['MODULE_TEMPERATURE'].max():.2f}°C)\n")
            f.write(f"Mean Irradiation: {df['IRRADIATION'].mean():.4f} (Max: {df['IRRADIATION'].max():.4f})\n")
            
            # Daily yield stats
            daily_yields = df.groupby(["Date", "SOURCE_KEY"])["DAILY_YIELD"].max()
            f.write(f"Mean Daily Yield per Inverter: {daily_yields.mean():.2f} kWh (Max: {daily_yields.max():.2f} kWh)\n\n")
            
        f.write("============================================================\n")
        f.write("INVERTER DETAILED YIELDS\n")
        f.write("============================================================\n\n")
        
        # Inverter yields
        for name, df in [("Plant 1", p1), ("Plant 2", p2)]:
            f.write(f"--- {name} Inverter Rankings ---\n")
            inv_stats = df.groupby("SOURCE_KEY").agg(
                max_ac=("AC_POWER", "max"),
                mean_daily_yield=("DAILY_YIELD", lambda x: df.loc[x.index].groupby(df.loc[x.index, "DATE_TIME"].dt.date)["DAILY_YIELD"].max().mean()),
                total_yield_diff=("TOTAL_YIELD", lambda x: x.max() - x.min())
            ).sort_values("mean_daily_yield", ascending=False)
            f.write(inv_stats.to_string())
            f.write("\n\n")

    print(f"Statistics written successfully to: {stats_path}")

def main():
    p1, p2 = load_data()
    
    print("Generating distribution plots...")
    plot_distributions(p1, p2)
    
    print("Generating correlation heatmaps...")
    plot_correlation_heatmap(p1, p2)
    
    print("Generating hourly generation profiles...")
    plot_hourly_generation(p1, p2)
    
    print("Generating daily generation profiles...")
    plot_daily_generation(p1, p2)
    
    print("Generating weather vs AC power scatter plots...")
    plot_weather_vs_ac(p1, p2)
    
    print("Generating yield growth plots...")
    plot_yields_over_time(p1, p2)
    
    print("Computing comparative statistics...")
    compute_statistics(p1, p2)
    
    print("EDA graphs and stats generation complete!")

if __name__ == "__main__":
    main()
