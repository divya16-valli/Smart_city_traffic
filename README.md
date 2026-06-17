# 🏙️ Smart City — Traffic Pattern Forecasting
## Project 9 | Data Science for Urban Traffic Management

---

## ⚡ QUICK START (3 steps only)

### Step 1 — Download the dataset
Go to this link and download `traffic.csv`:
👉 https://drive.google.com/file/d/1y61cDyuO9Zrp1fSchWcAmCxk0B6SMx7X/view

Place `traffic.csv` in this project folder (same location as the scripts).

---

### Step 2 — Install dependencies

Open terminal / command prompt in this folder and run:

```
pip install -r requirements.txt
```

---

### Step 3 — Run the project

**Option A: Jupyter Notebook (recommended — step by step)**
```
jupyter notebook Smart_City_Traffic_Forecasting.ipynb
```

**Option B: Python Script (runs everything at once)**
```
python traffic_forecasting.py
```

**Option C: One-click on Windows**
```
Double-click run.bat
```

**Option D: One-click on Mac/Linux**
```
bash run.sh
```

**Option E: Google Colab (no install needed)**
1. Go to https://colab.research.google.com
2. Upload `Smart_City_Traffic_Forecasting.ipynb`
3. Upload `traffic.csv` via the Files panel (left sidebar)
4. Run cells with Shift+Enter

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `traffic.csv` | Dataset (you download this) |
| `traffic_forecasting.py` | Full Python script — runs all 7 steps |
| `Smart_City_Traffic_Forecasting.ipynb` | Jupyter Notebook — step-by-step |
| `requirements.txt` | All required Python libraries |
| `run.bat` | One-click launcher for Windows |
| `run.sh` | One-click launcher for Mac/Linux |
| `plots/` | Auto-created folder with all 12 output charts |

---

## 🔬 Pipeline Overview

| Step | What it does |
|------|-------------|
| 1 | Load & clean data |
| 2 | Feature engineering (hour, day, holiday flags, cyclical encoding) |
| 3 | EDA — 6 charts (time-series, hourly, weekly, monthly, holiday, correlation) |
| 4 | Seasonal decomposition (Trend + Seasonality + Residual) |
| 5 | ML models — Random Forest, Gradient Boost, XGBoost with metrics |
| 6 | SARIMA 7-day forecast for all 4 junctions |
| 7 | Peak hour heatmaps + infrastructure recommendations |

---

## 📊 Output Charts

```
plots/
├── 01_traffic_volume.png        ← Time-series for all junctions
├── 02_hourly_pattern.png        ← Average vehicles by hour
├── 03_weekly_pattern.png        ← Average vehicles by weekday
├── 04_monthly_trend.png         ← Monthly trend lines
├── 05_holiday_comparison.png    ← Holiday vs normal boxplot
├── 06_correlation.png           ← Feature correlation heatmap
├── 07_decomposition.png         ← Trend / seasonal / residual
├── 08_model_comparison.png      ← MAE / RMSE / R² bar chart
├── 09_actual_vs_predicted.png   ← XGBoost predictions vs actual
├── 10_feature_importance.png    ← XGBoost feature ranking
├── 11_sarima_forecast.png       ← 7-day forecast all junctions
└── 12_peak_heatmap.png          ← Hour × Day intensity heatmap
```
