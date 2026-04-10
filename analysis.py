"""
TellCo Telecom Intelligence — Complete Data Analysis
=====================================================
Covers all 4 tasks:
  Task 1: User Overview Analysis
  Task 2: User Engagement Analysis
  Task 3: Experience Analytics
  Task 4: Satisfaction Analysis (4.1 → 4.7)

Usage:
  pip install pandas numpy scikit-learn scipy sqlalchemy pymysql matplotlib seaborn
  python analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
import time
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.spatial.distance import cdist
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATA_FILE      = "telcom__.Sheet1.csv"   # change to your actual filename
MYSQL_USER     = "root"
MYSQL_PASSWORD = "12345678"
MYSQL_HOST     = "localhost"
MYSQL_PORT     = 3306
MYSQL_DB       = "tellco"
OUTPUT_DIR     = "analysis_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def sep(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════════
sep("LOADING & CLEANING DATA")

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    # Try Excel
    try:
        df = pd.read_excel(DATA_FILE.replace(".csv", ".xlsx"))
    except:
        print(f"  ERROR: Could not find {DATA_FILE}")
        print("  Please update DATA_FILE variable with your actual filename.")
        exit(1)

df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df.columns = df.columns.str.strip()

print(f"  Raw data shape: {df.shape}")
print(f"  Columns: {list(df.columns[:10])} ...")

# Convert comma-separated numbers
for col in df.columns:
    if df[col].dtype == object:
        try:
            df[col] = df[col].str.replace(",", "", regex=False).astype(float)
        except:
            pass

# Fill missing values
for col in df.columns:
    if df[col].isnull().sum() == 0:
        continue
    if df[col].dtype in [np.float64, np.int64, np.float32]:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# IQR outlier clipping
for col in df.select_dtypes(include=np.number).columns:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    df[col] = df[col].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))

print(f"  Cleaned data shape: {df.shape}")
print(f"  Missing values remaining: {df.isnull().sum().sum()}")

# Identify key columns
msisdn_col  = next((c for c in ["MSISDN/Number","MSISDN","msisdn"] if c in df.columns), None)
handset_col = next((c for c in ["Handset Type","handset_type","Handset"] if c in df.columns), None)
manuf_col   = next((c for c in ["Handset Manufacturer","handset_manufacturer"] if c in df.columns), None)
dur_col     = next((c for c in ["Dur. (ms)","Duration (ms)","Dur.(ms)"] if c in df.columns), None)

print(f"  MSISDN col: {msisdn_col}")
print(f"  Handset col: {handset_col}")
print(f"  Manufacturer col: {manuf_col}")
print(f"  Duration col: {dur_col}")

# App columns
APPS = {
    "Social Media": ["Social Media DL (Bytes)","Social Media UL (Bytes)"],
    "Google":       ["Google DL (Bytes)","Google UL (Bytes)"],
    "Email":        ["Email DL (Bytes)","Email UL (Bytes)"],
    "YouTube":      ["Youtube DL (Bytes)","Youtube UL (Bytes)"],
    "Netflix":      ["Netflix DL (Bytes)","Netflix UL (Bytes)"],
    "Gaming":       ["Gaming DL (Bytes)","Gaming UL (Bytes)"],
    "Other":        ["Other DL (Bytes)","Other UL (Bytes)"],
}
APPS = {k:v for k,v in APPS.items() if all(c in df.columns for c in v)}
for app,(dl,ul) in APPS.items():
    df[f"{app}_Total"] = df[dl] + df[ul]

if "Total DL (Bytes)" in df.columns and "Total UL (Bytes)" in df.columns:
    df["Total_Data"] = df["Total DL (Bytes)"] + df["Total UL (Bytes)"]
else:
    df["Total_Data"] = df[[c for c in df.columns if "DL" in c or "UL" in c]].sum(axis=1)

print(f"  Apps tracked: {list(APPS.keys())}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — USER OVERVIEW ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
sep("TASK 1 — USER OVERVIEW ANALYSIS")

print(f"\n  Total records:  {len(df):,}")
print(f"  Unique users:   {df[msisdn_col].nunique():,}" if msisdn_col else "  MSISDN not found")
print(f"  Total data (TB): {df['Total_Data'].sum()/1e12:.2f}")

# 1.1 Top 10 handsets
if handset_col:
    top10_handsets = df[handset_col].value_counts().head(10)
    print(f"\n  Top 10 Handsets:")
    for h,c in top10_handsets.items():
        print(f"    {h}: {c:,}")

# 1.2 Top 3 manufacturers
if manuf_col:
    top3_manuf = df[manuf_col].value_counts().head(3)
    print(f"\n  Top 3 Manufacturers:")
    for m,c in top3_manuf.items():
        print(f"    {m}: {c:,} ({c/len(df)*100:.1f}%)")

# 1.3 App usage
print(f"\n  App Data Usage (GB):")
for app in APPS:
    gb = df[f"{app}_Total"].sum()/1e9
    print(f"    {app}: {gb:.1f} GB")

# 1.4 Basic metrics
print(f"\n  Basic Statistics:")
app_cols = [f"{app}_Total" for app in APPS]
stats = df[app_cols + ["Total_Data"]].describe().T[["mean","50%","std"]]
print(stats.to_string())

# 1.5 Correlation
print(f"\n  App vs Total Data Correlation:")
for app in APPS:
    corr = df[f"{app}_Total"].corr(df["Total_Data"])
    print(f"    {app}: r = {corr:.4f}")

# 1.6 PCA
X_pca = StandardScaler().fit_transform(df[app_cols].fillna(0))
pca   = PCA().fit(X_pca)
cum_var = np.cumsum(pca.explained_variance_ratio_)
print(f"\n  PCA Results:")
print(f"    PC1 variance:        {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"    PC1+PC2 variance:    {cum_var[1]*100:.1f}%")
print(f"    3-component coverage:{cum_var[2]*100:.1f}%")
print(f"    Components for 80%:  {int(np.argmax(cum_var>=0.8))+1}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — USER ENGAGEMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
sep("TASK 2 — USER ENGAGEMENT ANALYSIS")

eng = df.groupby(msisdn_col).agg(
    Sessions=(dur_col, "count"),
    Total_Duration=(dur_col, "sum"),
    Total_Traffic=("Total_Data", "sum"),
).reset_index()

for app in APPS:
    cn = f"{app}_Total"
    if cn in df.columns:
        eng[cn] = df.groupby(msisdn_col)[cn].sum().values

sc_eng = StandardScaler()
E_eng  = sc_eng.fit_transform(eng[["Sessions","Total_Duration","Total_Traffic"]])
km_eng = KMeans(n_clusters=3, random_state=42, n_init=10)
eng["Cluster"] = km_eng.fit_predict(E_eng)

lmap = {i:l for i,l in zip(
    eng.groupby("Cluster")["Total_Traffic"].mean().sort_values().index,
    ["Low","Mid","High"])}
eng["Engagement"] = eng["Cluster"].map(lmap)

print(f"\n  Engagement Cluster Summary:")
stats = eng.groupby("Engagement")[["Sessions","Total_Duration","Total_Traffic"]].agg(["mean","min","max"])
print(stats.to_string())

print(f"\n  Cluster Distribution:")
for lvl,cnt in eng["Engagement"].value_counts().items():
    print(f"    {lvl}: {cnt:,} users ({cnt/len(eng)*100:.1f}%)")

# Elbow method
inertias = [KMeans(n_clusters=k,random_state=42,n_init=10).fit(E_eng).inertia_ for k in range(1,11)]
print(f"\n  Elbow Method Inertias: {[round(x,0) for x in inertias]}")

# Top 10 users by traffic
top10_traffic = eng.nlargest(10,"Total_Traffic")[[msisdn_col,"Sessions","Total_Traffic"]]
print(f"\n  Top 10 Users by Traffic:")
print(top10_traffic.to_string(index=False))

# Top 3 apps
app_totals = {app: df[f"{app}_Total"].sum()/1e9 for app in APPS}
sorted_apps = sorted(app_totals.items(), key=lambda x: x[1], reverse=True)
print(f"\n  Top 3 Apps by Traffic:")
for app,gb in sorted_apps[:3]:
    print(f"    {app}: {gb:.1f} GB")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — EXPERIENCE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
sep("TASK 3 — EXPERIENCE ANALYTICS")

tcp_dl = next((c for c in ["TCP DL Retrans. Vol (Bytes)","TCP Retransmission"] if c in df.columns), None)
tcp_ul = next((c for c in ["TCP UL Retrans. Vol (Bytes)"] if c in df.columns), None)
rtt_dl = next((c for c in ["Avg RTT DL (ms)","Avg RTT (ms)"] if c in df.columns), None)
rtt_ul = next((c for c in ["Avg RTT UL (ms)"] if c in df.columns), None)
tp_dl  = next((c for c in ["Avg Bearer TP DL (kbps)","Throughput DL"] if c in df.columns), None)
tp_ul  = next((c for c in ["Avg Bearer TP UL (kbps)","Throughput UL"] if c in df.columns), None)

df2 = df.copy()
df2["Avg_TCP"] = ((df2[tcp_dl]+df2[tcp_ul])/2) if (tcp_dl and tcp_ul) else (df2[tcp_dl] if tcp_dl else 0)
df2["Avg_RTT"] = ((df2[rtt_dl]+df2[rtt_ul])/2) if (rtt_dl and rtt_ul) else (df2[rtt_dl] if rtt_dl else 0)
df2["Avg_TP"]  = ((df2[tp_dl]+df2[tp_ul])/2)   if (tp_dl  and tp_ul)  else (df2[tp_dl]  if tp_dl  else 0)

agg = {"Avg_TCP":"mean","Avg_RTT":"mean","Avg_TP":"mean","Total_Data":"sum"}
if handset_col and handset_col in df2.columns:
    agg[handset_col] = lambda x: x.mode()[0] if len(x) else "Unknown"

exp = df2.groupby(msisdn_col).agg(agg).reset_index()

sc_exp = StandardScaler()
E_exp  = sc_exp.fit_transform(exp[["Avg_TCP","Avg_RTT","Avg_TP"]].fillna(0))
km_exp = KMeans(n_clusters=3, random_state=42, n_init=10)
exp["Cluster"] = km_exp.fit_predict(E_exp)

emap = {i:l for i,l in zip(
    exp.groupby("Cluster")["Avg_RTT"].mean().sort_values().index,
    ["Good","Average","Poor"])}
exp["Experience"] = exp["Cluster"].map(emap)

print(f"\n  Experience Metrics (overall averages):")
print(f"    Avg TCP Retransmission: {exp['Avg_TCP'].mean():,.0f} bytes")
print(f"    Avg RTT:                {exp['Avg_RTT'].mean():.1f} ms")
print(f"    Avg Throughput:         {exp['Avg_TP'].mean():.1f} kbps")

print(f"\n  Experience Cluster Summary:")
exp_stats = exp.groupby("Experience")[["Avg_TCP","Avg_RTT","Avg_TP"]].mean()
print(exp_stats.to_string())

print(f"\n  Cluster Distribution:")
for lvl,cnt in exp["Experience"].value_counts().items():
    print(f"    {lvl}: {cnt:,} users ({cnt/len(exp)*100:.1f}%)")

# Top/bottom 10 TCP
print(f"\n  Top 10 TCP Retransmission values:")
print(exp["Avg_TCP"].nlargest(10).reset_index(drop=True).to_string())

print(f"\n  Top 10 RTT values:")
print(exp["Avg_RTT"].nlargest(10).reset_index(drop=True).to_string())

print(f"\n  Top 10 Throughput values:")
print(exp["Avg_TP"].nlargest(10).reset_index(drop=True).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.1 — SATISFACTION SCORES (EUCLIDEAN DISTANCE)
# ══════════════════════════════════════════════════════════════════════════════
sep("TASK 4.1 — ENGAGEMENT & EXPERIENCE SCORES")

def scale_data(X, mean, scale_):
    return (X - np.array(mean)) / np.array(scale_)

eng_centers = km_eng.cluster_centers_
exp_centers = km_exp.cluster_centers_

# Engagement score = distance to least engaged cluster
le    = eng.groupby("Cluster")["Total_Traffic"].mean().idxmin()
E_eng_scaled = scale_data(
    eng[["Sessions","Total_Duration","Total_Traffic"]].fillna(0).values,
    sc_eng.mean_, sc_eng.scale_)
eng["Eng_Score"] = cdist(E_eng_scaled, [eng_centers[le]]).flatten()

# Experience score = distance to worst experience cluster
we    = exp.groupby("Cluster")["Avg_RTT"].mean().idxmax()
E_exp_scaled = scale_data(
    exp[["Avg_TCP","Avg_RTT","Avg_TP"]].fillna(0).values,
    sc_exp.mean_, sc_exp.scale_)
exp["Exp_Score"] = cdist(E_exp_scaled, [exp_centers[we]]).flatten()

print(f"\n  Engagement Score stats:")
print(f"    Mean:   {eng['Eng_Score'].mean():.4f}")
print(f"    Median: {eng['Eng_Score'].median():.4f}")
print(f"    Max:    {eng['Eng_Score'].max():.4f}")
print(f"    Min:    {eng['Eng_Score'].min():.4f}")

print(f"\n  Experience Score stats:")
print(f"    Mean:   {exp['Exp_Score'].mean():.4f}")
print(f"    Median: {exp['Exp_Score'].median():.4f}")
print(f"    Max:    {exp['Exp_Score'].max():.4f}")
print(f"    Min:    {exp['Exp_Score'].min():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.2 — SATISFACTION SCORE + TOP 10
# ══════════════════════════════════════════════════════════════════════════════
sep("TASK 4.2 — SATISFACTION SCORE & TOP 10 CUSTOMERS")

sat = eng[[msisdn_col,"Eng_Score"]].merge(
      exp[[msisdn_col,"Exp_Score"]], on=msisdn_col)
sat["Satisfaction"] = (sat["Eng_Score"] + sat["Exp_Score"]) / 2

print(f"\n  Satisfaction Score stats:")
print(f"    Mean:   {sat['Satisfaction'].mean():.4f}")
print(f"    Median: {sat['Satisfaction'].median():.4f}")
print(f"    Max:    {sat['Satisfaction'].max():.4f}")
print(f"    Min:    {sat['Satisfaction'].min():.4f}")

top10_sat = sat.nlargest(10,"Satisfaction")[[msisdn_col,"Eng_Score","Exp_Score","Satisfaction"]].reset_index(drop=True)
top10_sat.index += 1
print(f"\n  Top 10 Satisfied Customers:")
print(top10_sat.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.3 — REGRESSION MODEL
# ══════════════════════════════════════════════════════════════════════════════
sep("TASK 4.3 — REGRESSION MODEL: PREDICT SATISFACTION SCORE")

sat_m = sat.merge(
    eng[[msisdn_col,"Sessions","Total_Duration","Total_Traffic"]], on=msisdn_col
).merge(
    exp[[msisdn_col,"Avg_TCP","Avg_RTT","Avg_TP"]], on=msisdn_col
).dropna()

feat = ["Sessions","Total_Duration","Total_Traffic","Avg_TCP","Avg_RTT","Avg_TP"]
X = sat_m[feat].fillna(0)
y = sat_m["Satisfaction"]

X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model_results = {}
best_model_obj = None
best_name = ""
best_r2 = -np.inf

for name,mdl in [
    ("Linear Regression",  LinearRegression()),
    ("Random Forest",      RandomForestRegressor(n_estimators=100,random_state=42)),
    ("Gradient Boosting",  GradientBoostingRegressor(n_estimators=100,random_state=42)),
]:
    t0 = time.time()
    mdl.fit(X_tr, y_tr)
    t1 = time.time()
    preds  = mdl.predict(X_te)
    r2     = r2_score(y_te, preds)
    mae    = mean_absolute_error(y_te, preds)
    rmse   = np.sqrt(mean_squared_error(y_te, preds))
    cv_r2  = cross_val_score(mdl, X, y, cv=5, scoring="r2").mean()

    model_results[name] = {"R2":r2,"MAE":mae,"RMSE":rmse,"CV_R2":cv_r2,"time":t1-t0}
    print(f"\n  {name}:")
    print(f"    R²:       {r2:.4f}")
    print(f"    MAE:      {mae:.4f}")
    print(f"    RMSE:     {rmse:.4f}")
    print(f"    CV R²:    {cv_r2:.4f}")
    print(f"    Train time: {t1-t0:.2f}s")

    if r2 > best_r2:
        best_r2, best_model_obj, best_name = r2, mdl, name

print(f"\n  ★ Best Model: {best_name} (R² = {best_r2:.4f})")

# Add predictions to sat_m
sat_m["Predicted_Satisfaction"] = best_model_obj.predict(X)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.4 — k=2 CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
sep("TASK 4.4 — k-MEANS (k=2) ON ENGAGEMENT & EXPERIENCE SCORES")

km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
sat["Sat_Cluster"] = km2.fit_predict(sat[["Eng_Score","Exp_Score"]].fillna(0))

print(f"\n  k=2 Cluster Distribution:")
for cl,cnt in sat["Sat_Cluster"].value_counts().items():
    print(f"    Cluster {cl}: {cnt:,} users ({cnt/len(sat)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.5 — AGGREGATE PER CLUSTER
# ══════════════════════════════════════════════════════════════════════════════
sep("TASK 4.5 — AGGREGATE AVG SATISFACTION & EXPERIENCE PER CLUSTER")

summary = sat.groupby("Sat_Cluster").agg(
    Users            =("Satisfaction","count"),
    Avg_Satisfaction =("Satisfaction","mean"),
    Avg_Eng_Score    =("Eng_Score","mean"),
    Avg_Exp_Score    =("Exp_Score","mean"),
    Std_Satisfaction =("Satisfaction","std"),
    Min_Satisfaction =("Satisfaction","min"),
    Max_Satisfaction =("Satisfaction","max"),
).reset_index()
print(f"\n  Cluster Aggregation:")
print(summary.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.6 — EXPORT TO MYSQL
# ══════════════════════════════════════════════════════════════════════════════
sep("TASK 4.6 — EXPORT TO MYSQL")

final = sat[[msisdn_col,"Eng_Score","Exp_Score","Satisfaction","Sat_Cluster"]].copy()
final.columns = ["user_id","engagement_score","experience_score",
                  "satisfaction_score","satisfaction_cluster"]

# Add predicted satisfaction if available
if "Predicted_Satisfaction" in sat_m.columns:
    pred_df = sat_m[[msisdn_col,"Predicted_Satisfaction"]].copy()
    pred_df.columns = ["user_id","predicted_satisfaction"]
    final = final.merge(pred_df, on="user_id", how="left")

# Save CSV
csv_path = f"{OUTPUT_DIR}/user_satisfaction_scores.csv"
final.to_csv(csv_path, index=False)
print(f"\n  Saved CSV: {csv_path}")
print(f"  Rows: {len(final):,}")
print(f"  Columns: {list(final.columns)}")

# Export to MySQL
try:
    engine = create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}")
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print(f"\n  MySQL connected successfully.")
    final.to_sql("user_satisfaction_scores", con=engine, if_exists="replace", index=False)
    print(f"  Exported {len(final):,} rows to MySQL table 'user_satisfaction_scores'")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM user_satisfaction_scores LIMIT 5"))
        print(f"\n  Verification SELECT (first 5 rows):")
        for row in result.fetchall():
            print(f"    {row}")
except Exception as e:
    print(f"\n  MySQL export skipped: {e}")
    print("  (CSV has been saved — run export_to_mysql.py separately)")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.7 — MODEL TRACKING REPORT
# ══════════════════════════════════════════════════════════════════════════════
sep("TASK 4.7 — MODEL TRACKING REPORT")

now = datetime.datetime.utcnow()
tracking_rows = []
best_r2_track, best_rid = -np.inf, 0

model_configs = [
    ("linear_regression",  LinearRegression(),                                  {"fit_intercept":True}),
    ("random_forest",      RandomForestRegressor(n_estimators=100,random_state=42), {"n_estimators":100,"random_state":42}),
    ("gradient_boosting",  GradientBoostingRegressor(n_estimators=100,random_state=42), {"n_estimators":100,"learning_rate":0.1,"random_state":42}),
]
for run_id,(mname,mdl,params) in enumerate(model_configs, 1):
    t_start = now + datetime.timedelta(seconds=run_id*2)
    t0 = time.time()
    mdl.fit(X_tr, y_tr)
    t1 = time.time()
    preds  = mdl.predict(X_te)
    r2     = r2_score(y_te, preds)
    mae    = mean_absolute_error(y_te, preds)
    rmse   = np.sqrt(mean_squared_error(y_te, preds))
    cv_r2  = cross_val_score(mdl, X, y, cv=5, scoring="r2").mean()
    t_end  = t_start + datetime.timedelta(seconds=t1-t0)
    row = {
        "run_id":       f"run_{run_id:03d}",
        "experiment":   "tellco_satisfaction",
        "model_name":   mname,
        "code_version": "v1.0.0",
        "source":       "analysis.py",
        "start_time":   t_start.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "end_time":     t_end.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "duration_s":   round(t1-t0, 3),
        "parameters":   str(params),
        "metric_r2":    round(r2,6),
        "metric_mae":   round(mae,6),
        "metric_rmse":  round(rmse,6),
        "metric_cv_r2": round(cv_r2,6),
        "artifact":     "user_satisfaction_scores.csv",
        "best_model":   "no",
    }
    if r2 > best_r2_track:
        best_r2_track, best_rid = r2, run_id-1
    tracking_rows.append(row)

tracking_rows[best_rid]["best_model"] = "YES"
tracking_df = pd.DataFrame(tracking_rows)

tracking_path = f"{OUTPUT_DIR}/model_tracking_report.csv"
tracking_df.to_csv(tracking_path, index=False)
print(f"\n  Tracking report saved: {tracking_path}")
print(f"\n  Run Log:")
print(tracking_df[["run_id","model_name","metric_r2","metric_mae","metric_rmse","best_model"]].to_string(index=False))

best = tracking_rows[best_rid]
print(f"\n  ★ Best Run: {best['run_id']} — {best['model_name']}")
print(f"    R²:   {best['metric_r2']}")
print(f"    MAE:  {best['metric_mae']}")
print(f"    RMSE: {best['metric_rmse']}")
print(f"    Start: {best['start_time']}")
print(f"    End:   {best['end_time']}")
print(f"    Source: {best['source']}  |  Version: {best['code_version']}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
sep("ANALYSIS COMPLETE — SUMMARY")
print(f"""
  Task 1 — User Overview:
    Records: {len(df):,}  |  Users: {df[msisdn_col].nunique():,}  |  Apps: {len(APPS)}

  Task 2 — Engagement:
    Low: {(eng['Engagement']=='Low').sum():,}  |
    Mid: {(eng['Engagement']=='Mid').sum():,}  |
    High: {(eng['Engagement']=='High').sum():,}

  Task 3 — Experience:
    Good: {(exp['Experience']=='Good').sum():,}  |
    Average: {(exp['Experience']=='Average').sum():,}  |
    Poor: {(exp['Experience']=='Poor').sum():,}

  Task 4 — Satisfaction:
    Users scored: {len(sat):,}
    Avg satisfaction: {sat['Satisfaction'].mean():.4f}
    Best model: {best_name} (R² = {best_r2:.4f})
    MySQL export: {len(final):,} rows

  Output files saved to: ./{OUTPUT_DIR}/
    - user_satisfaction_scores.csv
    - model_tracking_report.csv
""")
print("="*70)
