"""
=================================================================
 Smart City Traffic Pattern Forecasting  |  Project 9
 Run: python traffic_forecasting.py
 Requirement: traffic.csv in the same folder
=================================================================
"""

import warnings, os
warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

# ── Imports ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

HOLIDAYS = [
    "2019-01-01","2019-01-14","2019-01-26","2019-03-04",
    "2019-03-21","2019-04-14","2019-04-17","2019-04-18",
    "2019-04-19","2019-05-18","2019-06-05","2019-08-15",
    "2019-10-02","2019-10-27","2019-11-10","2019-12-25",
]

FEATURE_COLS = [
    "hour","day_of_week","month","quarter","week",
    "is_weekend","is_holiday","is_peak",
    "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos","junction"
]

# ─────────────────────────────────────────────────────────────
def load_data(path="traffic.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        df.drop(["date","time"], axis=1, inplace=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["vehicles"] = df["vehicles"].fillna(method="ffill")
    print(f"Loaded: {len(df):,} rows | {df.datetime.min().date()} → {df.datetime.max().date()}")
    return df

def engineer_features(df):
    dt = df["datetime"]
    df["hour"]        = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["day_name"]    = dt.dt.day_name()
    df["month"]       = dt.dt.month
    df["month_name"]  = dt.dt.month_name()
    df["quarter"]     = dt.dt.quarter
    df["week"]        = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"]  = (dt.dt.dayofweek >= 5).astype(int)
    df["is_holiday"]  = dt.dt.normalize().isin(pd.to_datetime(HOLIDAYS)).astype(int)
    df["is_peak"]     = (dt.dt.hour.between(7,9) | dt.dt.hour.between(17,19)).astype(int)
    df["hour_sin"]    = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]    = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]     = np.sin(2*np.pi*df["day_of_week"]/7)
    df["dow_cos"]     = np.cos(2*np.pi*df["day_of_week"]/7)
    df["month_sin"]   = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]   = np.cos(2*np.pi*df["month"]/12)
    df["day_type"]    = df["is_holiday"].map({0:"Normal Day",1:"Holiday"})
    print("✅ Features engineered")
    return df

def run_eda(df, junctions):
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    # 1. Volume over time
    fig, axes = plt.subplots(2,2,figsize=(18,10))
    fig.suptitle("Hourly Traffic Volume per Junction", fontsize=16, fontweight="bold")
    for ax,junc,col in zip(axes.flatten(),junctions,COLORS):
        sub = df[df["junction"]==junc]
        ax.plot(sub["datetime"], sub["vehicles"], color=col, linewidth=0.5, alpha=0.7)
        ax.set_title(f"Junction {junc}", fontsize=13)
        ax.set_xlabel("Date"); ax.set_ylabel("Vehicles/hour")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout(); plt.savefig("plots/01_traffic_volume.png",dpi=150,bbox_inches="tight"); plt.show()

    # 2. Hourly pattern
    fig, axes = plt.subplots(2,2,figsize=(16,10))
    fig.suptitle("Average Vehicles by Hour of Day", fontsize=15, fontweight="bold")
    for ax,junc,col in zip(axes.flatten(),junctions,COLORS):
        sub = df[df["junction"]==junc]
        h = sub.groupby("hour")["vehicles"].mean()
        ax.bar(h.index, h.values, color=col, alpha=0.85)
        ax.axvspan(7,9,alpha=0.12,color="red",label="Morning peak")
        ax.axvspan(17,19,alpha=0.12,color="orange",label="Evening peak")
        ax.set_title(f"Junction {junc}"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("plots/02_hourly_pattern.png",dpi=150,bbox_inches="tight"); plt.show()

    # 3. Weekly pattern
    fig, axes = plt.subplots(2,2,figsize=(16,10))
    fig.suptitle("Average Vehicles by Day of Week", fontsize=15, fontweight="bold")
    for ax,junc,col in zip(axes.flatten(),junctions,COLORS):
        sub = df[df["junction"]==junc]
        d = sub.groupby("day_of_week")["vehicles"].mean()
        bars = ax.bar(days, d.values, color=col, alpha=0.85)
        for b in bars[5:]: b.set_color("#aaaaaa")
        ax.set_title(f"Junction {junc}")
    plt.tight_layout(); plt.savefig("plots/03_weekly_pattern.png",dpi=150,bbox_inches="tight"); plt.show()

    # 4. Monthly trend
    fig, ax = plt.subplots(figsize=(14,6))
    for junc,col in zip(junctions,COLORS):
        sub = df[df["junction"]==junc]
        m = sub.groupby(["month","month_name"])["vehicles"].mean().reset_index().sort_values("month")
        ax.plot(m["month_name"], m["vehicles"], marker="o", label=f"Junction {junc}", color=col, linewidth=2)
    ax.set_title("Monthly Average Traffic", fontsize=14, fontweight="bold")
    ax.legend(); plt.xticks(rotation=30)
    plt.tight_layout(); plt.savefig("plots/04_monthly_trend.png",dpi=150,bbox_inches="tight"); plt.show()

    # 5. Holiday vs Normal
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data=df, x="junction", y="vehicles", hue="day_type",
                palette={"Normal Day":"#1f77b4","Holiday":"#d62728"}, ax=ax)
    ax.set_title("Holiday vs Normal Traffic", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig("plots/05_holiday_comparison.png",dpi=150,bbox_inches="tight"); plt.show()

    # 6. Correlation
    num_cols = ["vehicles","hour","day_of_week","month","is_weekend","is_holiday","is_peak"]
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig("plots/06_correlation.png",dpi=150,bbox_inches="tight"); plt.show()

    print("✅ EDA complete — 6 plots saved")

def decompose_all(df, junctions):
    fig, big_axes = plt.subplots(4,4,figsize=(22,20))
    fig.suptitle("Seasonal Decomposition — All Junctions", fontsize=16, fontweight="bold")
    titles = ["Observed","Trend","Seasonality","Residual"]
    for j_idx, junc in enumerate(junctions):
        sub   = df[df["junction"]==junc]
        daily = sub.set_index("datetime")["vehicles"].resample("D").mean().dropna()
        res   = seasonal_decompose(daily, model="additive", period=7)
        for c_idx,(comp,title) in enumerate(zip([daily,res.trend,res.seasonal,res.resid],titles)):
            ax = big_axes[j_idx][c_idx]
            ax.plot(comp, linewidth=0.9, color=COLORS[j_idx])
            if j_idx==0: ax.set_title(title, fontsize=12, fontweight="bold")
            if c_idx==0: ax.set_ylabel(f"Junction {junc}", fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=30, labelsize=7)
    plt.tight_layout(); plt.savefig("plots/07_decomposition.png",dpi=150,bbox_inches="tight"); plt.show()
    print("✅ Decomposition saved")

def train_models(df):
    X = df[FEATURE_COLS]; y = df["vehicles"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    mods = {
        "Random Forest" : RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
        "Gradient Boost": GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
        "XGBoost"       : xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
    }
    results, preds = [], {}
    for name, m in mods.items():
        m.fit(X_tr, y_tr); p = m.predict(X_te); preds[name] = p
        results.append({"Model":name,"MAE":round(mean_absolute_error(y_te,p),2),
                        "RMSE":round(np.sqrt(mean_squared_error(y_te,p)),2),"R²":round(r2_score(y_te,p),4)})
        print(f"  {name:<16} MAE={results[-1]['MAE']}  RMSE={results[-1]['RMSE']}  R²={results[-1]['R²']}")

    res_df = pd.DataFrame(results)

    # Comparison bar chart
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    for ax,metric,col in zip(axes,["MAE","RMSE","R²"],["#e74c3c","#e67e22","#27ae60"]):
        bars = ax.bar(res_df["Model"], res_df[metric], color=col, alpha=0.85)
        ax.set_title(metric)
        for b,v in zip(bars, res_df[metric]):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*0.97, str(v),
                    ha="center", va="top", fontsize=10, color="white", fontweight="bold")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15)
    plt.tight_layout(); plt.savefig("plots/08_model_comparison.png",dpi=150,bbox_inches="tight"); plt.show()

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(y_te.values[:400], label="Actual", linewidth=1.5)
    ax.plot(preds["XGBoost"][:400], label="XGBoost", linestyle="--", linewidth=1.5)
    ax.set_title("Actual vs Predicted (XGBoost)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout(); plt.savefig("plots/09_actual_vs_predicted.png",dpi=150,bbox_inches="tight"); plt.show()

    # Feature importance
    imp = pd.Series(mods["XGBoost"].feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    imp.plot(kind="barh", figsize=(10,7), color="#3498db")
    plt.title("XGBoost Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig("plots/10_feature_importance.png",dpi=150,bbox_inches="tight"); plt.show()

    print("✅ Models trained — 3 plots saved")
    return mods, res_df

def sarima_forecast(df, junctions, forecast_days=7):
    fig, axes = plt.subplots(2,2,figsize=(18,12))
    fig.suptitle(f"SARIMA {forecast_days}-Day Forecast — All Junctions", fontsize=15, fontweight="bold")

    for ax,junc,col in zip(axes.flatten(),junctions,COLORS):
        sub   = df[df["junction"]==junc]
        daily = sub.set_index("datetime")["vehicles"].resample("D").mean().dropna()
        train = daily[:-forecast_days]; test = daily[-forecast_days:]
        fit   = SARIMAX(train,order=(1,1,1),seasonal_order=(1,1,1,7),
                        enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
        fc    = fit.forecast(steps=forecast_days)
        fd    = pd.date_range(train.index[-1]+timedelta(days=1), periods=forecast_days, freq="D")
        fc_s  = pd.Series(fc.values, index=fd)
        mae   = mean_absolute_error(test.values, fc.values)
        print(f"  Junction {junc} → MAE={mae:.2f}")
        ax.plot(daily[-45:], label="Historical", color=col, linewidth=2)
        ax.plot(fc_s, label="Forecast", color="#d62728", linestyle="--", linewidth=2, marker="o")
        ax.plot(test, label="Actual", color="#2ca02c", linewidth=1.5, alpha=0.8)
        ax.set_title(f"Junction {junc} | MAE={mae:.1f}"); ax.legend(fontsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout(); plt.savefig("plots/11_sarima_forecast.png",dpi=150,bbox_inches="tight"); plt.show()
    print("✅ SARIMA forecasts done")

def peak_analysis(df, junctions):
    fig, axes = plt.subplots(2,2,figsize=(18,12))
    fig.suptitle("Traffic Intensity — Hour × Day Heatmap", fontsize=15, fontweight="bold")
    for ax,junc in zip(axes.flatten(),junctions):
        sub = df[df["junction"]==junc]
        pivot = sub.pivot_table(values="vehicles", index="hour", columns="day_of_week", aggfunc="mean")
        pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3)
        ax.set_title(f"Junction {junc}", fontsize=13)
    plt.tight_layout(); plt.savefig("plots/12_peak_heatmap.png",dpi=150,bbox_inches="tight"); plt.show()

    print("\n" + "="*65)
    print("         INFRASTRUCTURE RECOMMENDATIONS")
    print("="*65)
    for junc in junctions:
        sub = df[df["junction"]==junc]
        bh = sub.groupby("hour")["vehicles"].mean().idxmax()
        bd = sub.groupby("day_name")["vehicles"].mean().idxmax()
        qh = sub.groupby("hour")["vehicles"].mean().idxmin()
        pv = sub.groupby("hour")["vehicles"].mean().max()
        hd = sub.groupby("is_holiday")["vehicles"].mean().diff().iloc[-1]
        wa = sub[sub["is_weekend"]==0]["vehicles"].mean()
        we = sub[sub["is_weekend"]==1]["vehicles"].mean()
        print(f"\n🔷 Junction {junc}")
        print(f"   Busiest hour   : {bh:02d}:00")
        print(f"   Busiest day    : {bd}")
        print(f"   Peak volume    : {pv:.0f} vehicles/hr")
        print(f"   Weekday avg    : {wa:.1f}  |  Weekend avg: {we:.1f}")
        print(f"   Holiday effect : {hd:+.1f} vehicles/hr vs normal")
        print(f"   ✔ Extend signal timing at {bh:02d}:00–{bh+1:02d}:00")
        print(f"   ✔ Schedule maintenance at {qh:02d}:00")
    print("\n" + "="*65)

# ─── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🏙️  SMART CITY TRAFFIC FORECASTING  🏙️\n" + "="*45)

    df        = load_data("traffic.csv")
    df        = engineer_features(df)
    junctions = sorted(df["junction"].unique())

    print("\n[STEP 3] EDA ─────────────────────────────")
    run_eda(df, junctions)

    print("\n[STEP 4] Seasonal Decomposition ──────────")
    decompose_all(df, junctions)

    print("\n[STEP 5] ML Models ───────────────────────")
    models, results = train_models(df)

    print("\n[STEP 6] SARIMA Forecast ─────────────────")
    sarima_forecast(df, junctions, forecast_days=7)

    print("\n[STEP 7] Peak Analysis ───────────────────")
    peak_analysis(df, junctions)

    print("\n✅  ALL DONE! Check the /plots/ folder.\n")
