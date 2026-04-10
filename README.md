# TellCo Intelligence Dashboard

A Streamlit multi-page dashboard for the TellCo telecom analysis — covering all 4 tasks:
User Overview, Engagement, Experience, and Satisfaction Analysis.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` in your browser.

## Usage

1. Use the **sidebar file uploader** to upload your TellCo CSV or Excel file.
2. Navigate between the 4 analysis pages using the sidebar radio buttons.

## Pages

| Page | Task | Contents |
|------|------|----------|
| Overview | — | KPIs, app usage chart, manufacturer share |
| Task 1 · User Overview | 1, 1.1, 1.2 | Handsets, manufacturers, EDA, bivariate, decile, correlation, PCA |
| Task 2 · Engagement | 2, 2.1 | k-means k=3, elbow, top users, app usage |
| Task 3 · Experience | 3.1–3.4 | TCP/RTT/throughput dists, handset analysis, k=3 clusters |
| Task 4 · Satisfaction | 4.1–4.6 | Scores, regression (3 models), k=2 clusters, CSV export |

## Project structure

```
tellco_dashboard/
├── app.py           # Main Streamlit application
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Requirements

- Python ≥ 3.9
- The TellCo xDR dataset in CSV or Excel format

## MySQL export (Task 4.6)

After downloading `user_satisfaction_scores.csv` from the dashboard, import to MySQL:

```python
from sqlalchemy import create_engine
import pandas as pd

df = pd.read_csv("user_satisfaction_scores.csv")
engine = create_engine("mysql+pymysql://user:password@localhost/tellco_db")
df.to_sql("user_satisfaction_scores", engine, if_exists="replace", index=False)
```
