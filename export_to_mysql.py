"""
TellCo — Task 4.6 MySQL Export + Task 4.3 Satisfaction Prediction
Run this ONCE after downloading user_satisfaction_scores.csv from the dashboard.

Steps:
  1. pip install sqlalchemy pymysql scikit-learn pandas numpy
  2. Make sure MySQL is running and database 'tellco' exists
  3. python export_to_mysql.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — change only these if needed
# ══════════════════════════════════════════════════════════════════════════════
MYSQL_USER     = "root"
MYSQL_PASSWORD = "12345678"
MYSQL_HOST     = "localhost"
MYSQL_PORT     = 3306
MYSQL_DB       = "tellco"
CSV_FILE       = "user_satisfaction_scores.csv"   # downloaded from dashboard

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load the satisfaction scores CSV
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1: Loading satisfaction scores CSV")
print("="*60)

try:
    df = pd.read_csv(CSV_FILE)
    print(f"  Loaded {len(df):,} rows from '{CSV_FILE}'")
    print(f"  Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"\n  ERROR: '{CSV_FILE}' not found!")
    print("  Please download it from the dashboard:")
    print("  Task 4 → 4.6 MySQL Export → click the download button")
    exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Satisfaction Prediction Model (Task 4.3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2: Building Satisfaction Prediction Model (Task 4.3)")
print("="*60)

# Use engagement_score and experience_score as features
feature_cols = [c for c in ["engagement_score","experience_score",
                              "satisfaction_cluster"] if c in df.columns]
target_col   = "satisfaction_score"

if target_col not in df.columns:
    # try alternate column name
    target_col = "Satisfaction" if "Satisfaction" in df.columns else None

if target_col is None or len(feature_cols) < 2:
    print("  WARNING: Expected columns not found, skipping model training.")
    print(f"  Available columns: {list(df.columns)}")
else:
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression":  LinearRegression(),
        "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100,
                                  learning_rate=0.1, random_state=42),
    }

    best_model  = None
    best_name   = ""
    best_r2     = -np.inf
    model_results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds  = model.predict(X_test)
        r2     = r2_score(y_test, preds)
        mae    = mean_absolute_error(y_test, preds)
        rmse   = np.sqrt(mean_squared_error(y_test, preds))
        cv_r2  = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

        model_results.append({
            "Model": name, "R2": round(r2,4),
            "MAE":   round(mae,4), "RMSE": round(rmse,4),
            "CV_R2": round(cv_r2,4)
        })
        print(f"\n  {name}")
        print(f"    R²:    {r2:.4f}")
        print(f"    MAE:   {mae:.4f}")
        print(f"    RMSE:  {rmse:.4f}")
        print(f"    CV R²: {cv_r2:.4f}")

        if r2 > best_r2:
            best_r2    = r2
            best_model = model
            best_name  = name
            best_preds = preds

    print(f"\n  ★ Best model: {best_name} (R² = {best_r2:.4f})")

    # Add predicted satisfaction score to the dataframe
    all_preds = best_model.predict(X)
    df["predicted_satisfaction"] = np.round(all_preds, 6)

    print(f"\n  Predicted satisfaction added to dataframe.")
    print(f"  Sample predictions:")
    print(df[[c for c in ["msisdn", target_col, "predicted_satisfaction"]
              if c in df.columns]].head(10).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Export to MySQL (Task 4.6)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3: Exporting to MySQL (Task 4.6)")
print("="*60)

try:
    engine = create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}",
        echo=False
    )

    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("  MySQL connection successful.")

    # Export full table
    df.to_sql(
        "user_satisfaction_scores",
        con=engine,
        if_exists="replace",
        index=False
    )
    print(f"  Exported {len(df):,} rows to table 'user_satisfaction_scores'")

    # ── Verify with SELECT ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Verification — SELECT query output (Task 4.6 screenshot)")
    print("="*60)
    print("\n  Run this in MySQL Workbench for your screenshot:")
    print("  SELECT * FROM user_satisfaction_scores LIMIT 10;\n")

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM user_satisfaction_scores LIMIT 10"))
        rows = result.fetchall()
        cols = result.keys()

        # Print as formatted table
        header = " | ".join(str(c) for c in cols)
        print("  " + header)
        print("  " + "-"*len(header))
        for row in rows:
            print("  " + " | ".join(str(round(v,4)) if isinstance(v,float)
                                    else str(v) for v in row))

    print(f"\n  ✓ Table 'user_satisfaction_scores' exported successfully.")
    print(f"  ✓ Take a screenshot of the table above for Task 4.6 report.")

except Exception as e:
    print(f"\n  ERROR connecting to MySQL: {e}")
    print("\n  Troubleshooting:")
    print("  1. Make sure MySQL is running")
    print("  2. Open MySQL and run: CREATE DATABASE IF NOT EXISTS tellco;")
    print("  3. Check your password is correct (currently set to: 12345678)")
    print("  4. Try: pip install sqlalchemy pymysql")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Save final table as CSV artifact
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5: Saving final CSV artifact")
print("="*60)

output_file = "user_satisfaction_final.csv"
df.to_csv(output_file, index=False)
print(f"  Saved to '{output_file}'")
print(f"  Columns: {list(df.columns)}")
print(f"  Rows: {len(df):,}")
print("\n  Done! All tasks complete.")
print("="*60)
