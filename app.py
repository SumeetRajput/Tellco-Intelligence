"""
TellCo Telecom Intelligence Dashboard
Task 4 complete: 4.1 scores, 4.2 top10, 4.3 regression,
                 4.4 k=2, 4.5 aggregation, 4.6 MySQL, 4.7 tracking
NO sklearn objects ever passed to @st.cache_data functions.
"""

import streamlit as st

st.set_page_config(
    page_title="TellCo Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');
:root{--bg0:#08111F;--bg1:#0D1B2E;--bg2:#122338;--border:#1C3050;
      --text:#C8D8E8;--muted:#4A6080;--blue:#38BDF8;--green:#34D399;
      --red:#FB7185;--purple:#A78BFA;--amber:#FBBF24;--cyan:#22D3EE;}
html,body,[data-testid="stApp"]{background-color:var(--bg0)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background-color:var(--bg1)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stMetric"]{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:16px 20px!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:11px!important;letter-spacing:1.5px!important;text-transform:uppercase!important;font-family:'Space Mono',monospace!important;}
[data-testid="stMetricValue"]{color:var(--blue)!important;font-family:'Space Mono',monospace!important;}
[data-testid="stTabs"] button{font-family:'Space Mono',monospace!important;font-size:12px!important;color:var(--muted)!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--blue)!important;border-bottom-color:var(--blue)!important;}
h1,h2,h3{font-family:'Space Mono',monospace!important;color:var(--text)!important;}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:8px!important;}
hr{border-color:var(--border)!important;}
[data-testid="stExpander"]{border:1px solid var(--border)!important;border-radius:8px!important;background:var(--bg2)!important;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 20px'>
      <div style='font-family:Space Mono,monospace;font-size:18px;font-weight:700;color:#38BDF8;letter-spacing:2px;'>📡 TELLCO</div>
      <div style='font-family:Space Mono,monospace;font-size:10px;color:#4A6080;letter-spacing:3px;margin-top:2px;'>INTELLIGENCE PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("Navigate",
        ["🏠  Overview","📱  Task 1 · User Overview","🔥  Task 2 · Engagement",
         "📶  Task 3 · Experience","⭐  Task 4 · Satisfaction"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#4A6080;font-family:Space Mono,monospace;line-height:1.8;'>
      <div>DATA SOURCE</div><div style='color:#C8D8E8;'>TellCo xDR Dataset</div>
      <div style='margin-top:8px;'>PERIOD</div><div style='color:#C8D8E8;'>1 Month Aggregated</div>
      <div style='margin-top:8px;'>REGION</div><div style='color:#C8D8E8;'>Pefkakia</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    upload = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"])

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime, time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

# ── Theme helper ──────────────────────────────────────────────────────────────
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0D1B2E",
    font_color="#C8D8E8", font_family="Space Mono",
    title_font_family="Space Mono", title_font_color="#C8D8E8",
    colorway=["#38BDF8","#34D399","#FB7185","#A78BFA","#FBBF24","#22D3EE","#F472B6","#FB923C"],
)
def T(**kw):
    bx = dict(gridcolor="#1C3050", zerolinecolor="#1C3050")
    by = dict(gridcolor="#1C3050", zerolinecolor="#1C3050")
    if "xaxis" in kw: bx.update(kw.pop("xaxis"))
    if "yaxis" in kw: by.update(kw.pop("yaxis"))
    return {**_BASE, "xaxis": bx, "yaxis": by, **kw}

# ── Cached helpers ────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    if file is None:
        return None
    df = pd.read_excel(file) if file.name.lower().endswith((".xlsx",".xls")) else pd.read_csv(file)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace(",","",regex=False).astype(float)
            except:
                pass
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in [np.float64, np.int64, np.float32]:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include=np.number).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        df[col] = df[col].clip(Q1-1.5*(Q3-Q1), Q3+1.5*(Q3-Q1))
    return df

@st.cache_data
def build_features(df):
    APPS = {
        "Social Media":["Social Media DL (Bytes)","Social Media UL (Bytes)"],
        "Google":      ["Google DL (Bytes)","Google UL (Bytes)"],
        "Email":       ["Email DL (Bytes)","Email UL (Bytes)"],
        "YouTube":     ["Youtube DL (Bytes)","Youtube UL (Bytes)"],
        "Netflix":     ["Netflix DL (Bytes)","Netflix UL (Bytes)"],
        "Gaming":      ["Gaming DL (Bytes)","Gaming UL (Bytes)"],
        "Other":       ["Other DL (Bytes)","Other UL (Bytes)"],
    }
    APPS = {k:v for k,v in APPS.items() if all(c in df.columns for c in v)}
    for app,(dl,ul) in APPS.items():
        df[f"{app}_Total"] = df[dl]+df[ul]
    if "Total DL (Bytes)" in df.columns and "Total UL (Bytes)" in df.columns:
        df["Total_Data"] = df["Total DL (Bytes)"]+df["Total UL (Bytes)"]
    else:
        df["Total_Data"] = df[[c for c in df.columns if "DL" in c or "UL" in c]].sum(axis=1)
    msisdn  = next((c for c in ["MSISDN/Number","MSISDN","msisdn"] if c in df.columns), None)
    handset = next((c for c in ["Handset Type","handset_type","Handset"] if c in df.columns), None)
    manuf   = next((c for c in ["Handset Manufacturer","handset_manufacturer"] if c in df.columns), None)
    dur     = next((c for c in ["Dur. (ms)","Duration (ms)","Dur.(ms)"] if c in df.columns), None)
    return df, APPS, msisdn, handset, manuf, dur

# Returns plain Python lists — never sklearn objects — so cache_data can hash them
@st.cache_data
def compute_engagement(df, msisdn_col, dur_col, app_cols):
    eng = df.groupby(msisdn_col).agg(
        Sessions=(dur_col,"count"),
        Total_Duration=(dur_col,"sum"),
        Total_Traffic=("Total_Data","sum"),
    ).reset_index()
    for app in app_cols:
        cn = f"{app}_Total"
        if cn in df.columns:
            eng[cn] = df.groupby(msisdn_col)[cn].sum().values
    sc = StandardScaler()
    E  = sc.fit_transform(eng[["Sessions","Total_Duration","Total_Traffic"]])
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    eng["Cluster"] = km.fit_predict(E)
    lmap = {i:l for i,l in zip(
        eng.groupby("Cluster")["Total_Traffic"].mean().sort_values().index,
        ["Low","Mid","High"])}
    eng["Engagement"] = eng["Cluster"].map(lmap)
    inertias = [KMeans(n_clusters=k,random_state=42,n_init=10).fit(E).inertia_ for k in range(1,11)]
    # *** Plain lists — not sklearn objects ***
    return eng, km.cluster_centers_.tolist(), sc.mean_.tolist(), sc.scale_.tolist(), inertias

# Returns plain Python lists — never sklearn objects
@st.cache_data
def compute_experience(df, msisdn_col, handset_col):
    tcp_dl = next((c for c in ["TCP DL Retrans. Vol (Bytes)","TCP Retransmission"] if c in df.columns), None)
    tcp_ul = next((c for c in ["TCP UL Retrans. Vol (Bytes)"] if c in df.columns), None)
    rtt_dl = next((c for c in ["Avg RTT DL (ms)","Avg RTT (ms)"] if c in df.columns), None)
    rtt_ul = next((c for c in ["Avg RTT UL (ms)"] if c in df.columns), None)
    tp_dl  = next((c for c in ["Avg Bearer TP DL (kbps)","Throughput DL"] if c in df.columns), None)
    tp_ul  = next((c for c in ["Avg Bearer TP UL (kbps)","Throughput UL"] if c in df.columns), None)
    df = df.copy()
    df["Avg_TCP"] = ((df[tcp_dl]+df[tcp_ul])/2) if (tcp_dl and tcp_ul) else (df[tcp_dl] if tcp_dl else 0)
    df["Avg_RTT"] = ((df[rtt_dl]+df[rtt_ul])/2) if (rtt_dl and rtt_ul) else (df[rtt_dl] if rtt_dl else 0)
    df["Avg_TP"]  = ((df[tp_dl]+df[tp_ul])/2)   if (tp_dl  and tp_ul)  else (df[tp_dl]  if tp_dl  else 0)
    agg = {"Avg_TCP":"mean","Avg_RTT":"mean","Avg_TP":"mean","Total_Data":"sum"}
    if handset_col and handset_col in df.columns:
        agg[handset_col] = lambda x: x.mode()[0] if len(x) else "Unknown"
    exp = df.groupby(msisdn_col).agg(agg).reset_index()
    sc  = StandardScaler()
    E   = sc.fit_transform(exp[["Avg_TCP","Avg_RTT","Avg_TP"]].fillna(0))
    km  = KMeans(n_clusters=3, random_state=42, n_init=10)
    exp["Cluster"] = km.fit_predict(E)
    emap = {i:l for i,l in zip(
        exp.groupby("Cluster")["Avg_RTT"].mean().sort_values().index,
        ["Good","Average","Poor"])}
    exp["Experience"] = exp["Cluster"].map(emap)
    # *** Plain lists — not sklearn objects ***
    return exp, km.cluster_centers_.tolist(), sc.mean_.tolist(), sc.scale_.tolist()

# Leading _ on list params = Streamlit skips hashing entirely (belt-and-suspenders)
@st.cache_data
def compute_satisfaction(eng_df, exp_df, msisdn_col,
                         _eng_c, _sc_eng_m, _sc_eng_s,
                         _exp_c, _sc_exp_m, _sc_exp_s):
    def _scale(X, mean, scale_):
        return (X - np.array(mean)) / np.array(scale_)

    eng_centers = np.array(_eng_c)
    exp_centers = np.array(_exp_c)

    # Task 4.1a: engagement score = distance to least-engaged cluster
    le    = eng_df.groupby("Cluster")["Total_Traffic"].mean().idxmin()
    E_eng = _scale(eng_df[["Sessions","Total_Duration","Total_Traffic"]].fillna(0).values,
                   _sc_eng_m, _sc_eng_s)
    eng_df = eng_df.copy()
    eng_df["Eng_Score"] = cdist(E_eng, [eng_centers[le]]).flatten()

    # Task 4.1b: experience score = distance to worst-experience cluster
    we    = exp_df.groupby("Cluster")["Avg_RTT"].mean().idxmax()
    E_exp = _scale(exp_df[["Avg_TCP","Avg_RTT","Avg_TP"]].fillna(0).values,
                   _sc_exp_m, _sc_exp_s)
    exp_df = exp_df.copy()
    exp_df["Exp_Score"] = cdist(E_exp, [exp_centers[we]]).flatten()

    # Task 4.2: satisfaction = average of both scores
    sat = eng_df[[msisdn_col,"Eng_Score"]].merge(
          exp_df[[msisdn_col,"Exp_Score"]], on=msisdn_col)
    sat["Satisfaction"] = (sat["Eng_Score"] + sat["Exp_Score"]) / 2

    # Task 4.4: k=2 on engagement & experience scores
    km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
    sat["Sat_Cluster"] = km2.fit_predict(sat[["Eng_Score","Exp_Score"]].fillna(0))
    return sat

# ── Load ──────────────────────────────────────────────────────────────────────
df_raw   = load_data(upload)
has_data = False
if df_raw is not None:
    df, APP_COLS, msisdn_col, handset_col, manuf_col, dur_col = build_features(df_raw.copy())
    has_data = True

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div style='padding:24px 0 8px'>
      <div style='font-family:Space Mono,monospace;font-size:28px;font-weight:700;letter-spacing:3px;color:#38BDF8;'>TELLCO</div>
      <div style='font-family:Space Mono,monospace;font-size:12px;color:#4A6080;letter-spacing:4px;margin-top:2px;'>TELECOM INTELLIGENCE DASHBOARD</div>
    </div>""", unsafe_allow_html=True)
    if not has_data:
        st.markdown("""
        <div style='background:#0D1B2E;border:1px dashed #1C3050;border-radius:12px;padding:48px;text-align:center;margin-top:32px;'>
          <div style='font-size:48px;margin-bottom:16px;'>📡</div>
          <div style='font-family:Space Mono,monospace;font-size:16px;color:#38BDF8;letter-spacing:2px;'>AWAITING DATA UPLOAD</div>
          <div style='color:#4A6080;margin-top:8px;font-size:13px;'>Upload your TellCo CSV or Excel file using the sidebar to begin analysis</div>
        </div>""", unsafe_allow_html=True)
    else:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Records",  f"{len(df):,}")
        c2.metric("Unique Users",   f"{df[msisdn_col].nunique():,}" if msisdn_col else "N/A")
        c3.metric("Total Data (TB)",f"{df['Total_Data'].sum()/1e12:.2f}")
        c4.metric("Apps Tracked",   len(APP_COLS))
        c5.metric("Features",       df.shape[1])
        st.markdown("---")
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("#### App data usage breakdown")
            app_totals = {app: df[f"{app}_Total"].sum()/1e9 for app in APP_COLS}
            fig = go.Figure(go.Bar(
                x=list(app_totals.keys()), y=list(app_totals.values()),
                marker_color=["#38BDF8","#34D399","#FB7185","#A78BFA","#FBBF24","#22D3EE","#F472B6"],
                text=[f"{v:.1f} GB" for v in app_totals.values()],
                textposition="outside", textfont_color="#C8D8E8"))
            fig.update_layout(**T(height=320, showlegend=False,
                xaxis_title="Application", yaxis_title="Total GB",
                margin=dict(l=0,r=0,t=10,b=0)))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if manuf_col:
                st.markdown("#### Manufacturer share")
                top_m = df[manuf_col].value_counts().head(6)
                fig = go.Figure(go.Pie(labels=top_m.index, values=top_m.values, hole=0.55,
                    marker_colors=["#38BDF8","#34D399","#FB7185","#A78BFA","#FBBF24","#22D3EE"]))
                fig.update_layout(**T(height=320, legend=dict(font_color="#C8D8E8"),
                    margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1
# ══════════════════════════════════════════════════════════════════════════════
elif "Task 1" in page:
    st.markdown("### 📱 Task 1 — User Overview Analysis")
    if not has_data:
        st.warning("Upload data from the sidebar to begin.")
    else:
        tab1,tab2,tab3,tab4 = st.tabs(["Handsets & Manufacturers","EDA · Metrics","Bivariate & Correlation","PCA"])
        with tab1:
            col1,col2 = st.columns([3,2])
            with col1:
                if handset_col:
                    st.markdown("##### Top 10 handsets")
                    top10 = df[handset_col].value_counts().head(10).reset_index()
                    top10.columns = ["Handset","Count"]
                    fig = go.Figure(go.Bar(x=top10["Count"], y=top10["Handset"],
                        orientation="h", marker_color="#38BDF8",
                        text=top10["Count"], textposition="outside"))
                    fig.update_layout(**T(height=380, xaxis_title="Count",
                        yaxis=dict(autorange="reversed"), margin=dict(l=0,r=60,t=10,b=0)))
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if manuf_col:
                    st.markdown("##### Top manufacturers")
                    top3 = df[manuf_col].value_counts().head(3)
                    for i,(m,c) in enumerate(top3.items()):
                        color = ["#38BDF8","#34D399","#A78BFA"][i]
                        pct = c/len(df)*100
                        st.markdown(f"""
                        <div style='background:#122338;border:1px solid #1C3050;border-left:3px solid {color};
                                    border-radius:8px;padding:12px 16px;margin-bottom:10px;'>
                          <div style='font-family:Space Mono,monospace;font-size:10px;color:{color};letter-spacing:2px;'>#{i+1}</div>
                          <div style='font-weight:500;margin:4px 0;'>{m}</div>
                          <div style='font-family:Space Mono,monospace;font-size:18px;font-weight:700;color:{color};'>{c:,}</div>
                          <div style='font-size:11px;color:#4A6080;'>{pct:.1f}% of users</div>
                        </div>""", unsafe_allow_html=True)
            if manuf_col and handset_col:
                st.markdown("##### Top 5 handsets per top 3 manufacturer")
                top3_names = df[manuf_col].value_counts().head(3).index
                cols = st.columns(3)
                colors = ["#38BDF8","#34D399","#A78BFA"]
                for i,(col_ui,mfr) in enumerate(zip(cols, top3_names)):
                    with col_ui:
                        sub = df[df[manuf_col]==mfr][handset_col].value_counts().head(5).reset_index()
                        sub.columns = ["Handset","Count"]
                        fig = go.Figure(go.Bar(x=sub["Count"], y=sub["Handset"],
                            orientation="h", marker_color=colors[i]))
                        fig.update_layout(**T(height=220, title=mfr,
                            margin=dict(l=0,r=20,t=30,b=0), showlegend=False,
                            yaxis=dict(autorange="reversed")))
                        st.plotly_chart(fig, use_container_width=True)
        with tab2:
            app_total_cols = [f"{app}_Total" for app in APP_COLS]
            key_cols = app_total_cols + ["Total_Data"]
            if dur_col and dur_col in df.columns:
                key_cols = [dur_col] + key_cols
            st.markdown("##### Basic metrics")
            metrics_df = df[key_cols].describe().T[["mean","50%","std","min","max"]]
            metrics_df.columns = ["Mean","Median","Std","Min","Max"]
            st.dataframe(metrics_df.style.format("{:,.1f}"), use_container_width=True)
            st.markdown("##### Dispersion analysis")
            disp = pd.DataFrame({
                "Variance": df[key_cols].var(),
                "IQR":      df[key_cols].quantile(0.75)-df[key_cols].quantile(0.25),
                "Skewness": df[key_cols].skew(),
                "Kurtosis": df[key_cols].kurtosis(),
                "CV (%)":   df[key_cols].std()/df[key_cols].mean()*100,
            })
            st.dataframe(disp.style.format("{:,.2f}"), use_container_width=True)
            if dur_col and dur_col in df.columns:
                st.markdown("##### Session duration distribution")
                dur = df[dur_col]/1000
                fig = go.Figure(go.Histogram(x=dur.clip(0,dur.quantile(0.99)),
                    nbinsx=40, marker_color="#FBBF24", opacity=0.75))
                fig.add_vline(x=dur.median(), line_color="#FB7185", line_dash="dash",
                    annotation_text=f"Median {dur.median():.0f}s",
                    annotation_font_color="#FB7185")
                fig.update_layout(**T(height=280, xaxis_title="Duration (s)",
                    yaxis_title="Count", margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
        with tab3:
            st.markdown("##### Correlation: each app vs total data")
            app_corr = {app: df[f"{app}_Total"].corr(df["Total_Data"]) for app in APP_COLS}
            corr_s = pd.Series(app_corr).sort_values(ascending=False)
            fig = go.Figure(go.Bar(x=corr_s.index, y=corr_s.values,
                marker_color=["#34D399" if v>0 else "#FB7185" for v in corr_s.values],
                text=[f"{v:.3f}" for v in corr_s.values], textposition="outside"))
            fig.update_layout(**T(height=300, xaxis_title="Application",
                yaxis_title="Pearson r", margin=dict(l=0,r=0,t=10,b=0)))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("##### App-to-app correlation matrix")
            atc = [f"{app}_Total" for app in APP_COLS]
            cm  = df[atc].corr()
            fig = go.Figure(go.Heatmap(z=cm.values,
                x=[c.replace("_Total","") for c in cm.columns],
                y=[c.replace("_Total","") for c in cm.index],
                colorscale=[[0,"#FB7185"],[0.5,"#0D1B2E"],[1,"#38BDF8"]],
                zmid=0, text=cm.round(2).values, texttemplate="%{text}",
                colorbar=dict(tickfont_color="#C8D8E8")))
            fig.update_layout(**T(height=380, margin=dict(l=0,r=0,t=10,b=0)))
            st.plotly_chart(fig, use_container_width=True)
            if msisdn_col and dur_col and dur_col in df.columns:
                st.markdown("##### Decile analysis")
                user_agg = df.groupby(msisdn_col).agg(
                    Dur=(dur_col,"sum"), Data=("Total_Data","sum")).reset_index()
                user_agg["Decile"] = pd.qcut(user_agg["Dur"], q=10,
                    labels=[f"D{i}" for i in range(1,11)], duplicates="drop")
                dec = user_agg.groupby("Decile", observed=True)["Data"].agg(["sum","mean","count"])
                fig = go.Figure(go.Bar(x=dec.index.astype(str), y=dec["sum"]/1e9,
                    marker_color="#A78BFA", text=(dec["sum"]/1e9).round(1), textposition="outside"))
                fig.update_layout(**T(height=280, xaxis_title="Decile",
                    yaxis_title="Total GB", margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
        with tab4:
            atc = [f"{app}_Total" for app in APP_COLS]
            X   = StandardScaler().fit_transform(df[atc].fillna(0))
            pca = PCA().fit(X)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("##### Variance per component")
                fig = go.Figure(go.Bar(
                    x=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                    y=pca.explained_variance_ratio_*100, marker_color="#A78BFA"))
                fig.update_layout(**T(height=280, yaxis_title="Variance %",
                    margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("##### Cumulative explained variance")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(1,len(cum_var)+1)), y=cum_var*100,
                    mode="lines+markers", line_color="#38BDF8", line_width=2.5,
                    marker=dict(color="#38BDF8", size=8)))
                fig.add_hline(y=80, line_dash="dash", line_color="#FB7185", annotation_text="80%")
                fig.add_hline(y=95, line_dash="dash", line_color="#FBBF24", annotation_text="95%")
                fig.update_layout(**T(height=280, yaxis_title="Cumulative %",
                    margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("PC1 variance",         f"{pca.explained_variance_ratio_[0]*100:.1f}%")
            c2.metric("PC1+2 variance",        f"{cum_var[1]*100:.1f}%")
            c3.metric("3-component coverage",  f"{cum_var[2]*100:.1f}%")
            c4.metric("Components for 80%",    str(int(np.argmax(cum_var>=0.8))+1))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2
# ══════════════════════════════════════════════════════════════════════════════
elif "Task 2" in page:
    st.markdown("### 🔥 Task 2 — User Engagement Analysis")
    if not has_data:
        st.warning("Upload data from the sidebar to begin.")
    elif not msisdn_col or not dur_col:
        st.error("MSISDN or Duration column not found.")
    else:
        eng, eng_c, sc_eng_m, sc_eng_s, inertias = compute_engagement(
            df, msisdn_col, dur_col, list(APP_COLS.keys()))
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Unique Users",    f"{len(eng):,}")
        c2.metric("Avg Sessions",    f"{eng['Sessions'].mean():.1f}")
        c3.metric("Avg Traffic (MB)",f"{eng['Total_Traffic'].mean()/1e6:.1f}")
        c4.metric("High Engagement", f"{(eng['Engagement']=='High').sum():,}")
        tab1,tab2,tab3 = st.tabs(["Clustering","Top Users","App Usage"])
        with tab1:
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("##### Elbow method")
                fig = go.Figure(go.Scatter(x=list(range(1,11)), y=inertias,
                    mode="lines+markers", line_color="#38BDF8", line_width=2.5,
                    marker=dict(color="#38BDF8", size=8)))
                fig.add_vline(x=3, line_dash="dash", line_color="#FB7185",
                    annotation_text="k=3", annotation_font_color="#FB7185")
                fig.update_layout(**T(height=280, xaxis_title="k",
                    yaxis_title="Inertia", margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("##### Engagement distribution")
                counts = eng["Engagement"].value_counts()
                fig = go.Figure(go.Pie(labels=counts.index, values=counts.values,
                    hole=0.55, marker_colors=["#34D399","#FBBF24","#FB7185"]))
                fig.update_layout(**T(height=280, margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("##### Sessions vs Traffic by cluster")
            cmap = {"Low":"#34D399","Mid":"#FBBF24","High":"#FB7185"}
            fig = go.Figure()
            for lvl,col in cmap.items():
                sub = eng[eng["Engagement"]==lvl]
                fig.add_trace(go.Scatter(x=sub["Sessions"], y=sub["Total_Traffic"]/1e6,
                    mode="markers", name=lvl, marker=dict(color=col, size=4, opacity=0.5)))
            fig.update_layout(**T(height=320, xaxis_title="Sessions",
                yaxis_title="Traffic (MB)", margin=dict(l=0,r=0,t=10,b=0)))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("##### Cluster statistics")
            stats = eng.groupby("Engagement")[["Sessions","Total_Duration","Total_Traffic"]].agg(
                ["min","max","mean","sum"])
            st.dataframe(stats.style.format("{:,.0f}"), use_container_width=True)
        with tab2:
            metric = st.selectbox("Rank by", ["Sessions","Total_Duration","Total_Traffic"])
            top10  = eng.nlargest(10, metric)[[msisdn_col, metric]]
            fig = go.Figure(go.Bar(x=top10[metric], y=top10[msisdn_col].astype(str),
                orientation="h", marker_color="#22D3EE"))
            fig.update_layout(**T(height=340, xaxis_title=metric,
                yaxis=dict(autorange="reversed"), margin=dict(l=0,r=0,t=10,b=0)))
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            app_totals_gb = {app: df[f"{app}_Total"].sum()/1e9 for app in APP_COLS}
            app_s = pd.Series(app_totals_gb).sort_values(ascending=False)
            top3_apps = app_s.index[:3].tolist()
            colors_bar = ["#FB7185" if app in top3_apps else "#38BDF8" for app in app_s.index]
            fig = go.Figure(go.Bar(x=app_s.index, y=app_s.values, marker_color=colors_bar,
                text=[f"{v:.1f}" for v in app_s.values], textposition="outside"))
            fig.update_layout(**T(height=300, yaxis_title="Total GB",
                margin=dict(l=0,r=0,t=10,b=0), showlegend=False))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("##### Top 10 users per app")
            app_sel = st.selectbox("Select app", list(APP_COLS.keys()))
            top_users = df.groupby(msisdn_col)[f"{app_sel}_Total"].sum().nlargest(10).reset_index()
            top_users.columns = ["MSISDN","Traffic (Bytes)"]
            st.dataframe(top_users.style.format({"Traffic (Bytes)":"{:,.0f}"}), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3
# ══════════════════════════════════════════════════════════════════════════════
elif "Task 3" in page:
    st.markdown("### 📶 Task 3 — Experience Analytics")
    if not has_data:
        st.warning("Upload data from the sidebar to begin.")
    elif not msisdn_col:
        st.error("MSISDN column not found.")
    else:
        exp, exp_c, sc_exp_m, sc_exp_s = compute_experience(df, msisdn_col, handset_col)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg TCP Retrans (B)",  f"{exp['Avg_TCP'].mean():,.0f}")
        c2.metric("Avg RTT (ms)",         f"{exp['Avg_RTT'].mean():.1f}")
        c3.metric("Avg Throughput (kbps)",f"{exp['Avg_TP'].mean():.1f}")
        c4.metric("Good Experience %",    f"{(exp['Experience']=='Good').mean()*100:.1f}%")
        tab1,tab2,tab3 = st.tabs(["Distributions","Handset Analysis","Clustering"])
        with tab1:
            col1,col2,col3 = st.columns(3)
            for col_ui,metric,color,label in [
                (col1,"Avg_TCP","#FB7185","TCP Retransmission (Bytes)"),
                (col2,"Avg_RTT","#FBBF24","Avg RTT (ms)"),
                (col3,"Avg_TP","#34D399","Throughput (kbps)"),
            ]:
                with col_ui:
                    vals = exp[metric].clip(0, exp[metric].quantile(0.99))
                    fig  = go.Figure(go.Histogram(x=vals, nbinsx=35,
                        marker_color=color, opacity=0.75))
                    fig.add_vline(x=vals.median(), line_dash="dash", line_color="#C8D8E8")
                    fig.update_layout(**T(height=220, title=label,
                        margin=dict(l=0,r=0,t=30,b=0), xaxis_title="", yaxis_title="Count"))
                    st.plotly_chart(fig, use_container_width=True)
            for metric,col in [("Avg_TCP","TCP"),("Avg_RTT","RTT"),("Avg_TP","Throughput")]:
                with st.expander(f"{col} — top 10 / bottom 10 / most frequent"):
                    vals = exp[metric].dropna()
                    c1,c2,c3 = st.columns(3)
                    c1.markdown("**Top 10**")
                    c1.dataframe(vals.nlargest(10).reset_index(drop=True).rename(col), use_container_width=True)
                    c2.markdown("**Bottom 10**")
                    c2.dataframe(vals.nsmallest(10).reset_index(drop=True).rename(col), use_container_width=True)
                    c3.markdown("**Most frequent**")
                    c3.dataframe(vals.round(0).value_counts().head(10).reset_index(), use_container_width=True)
        with tab2:
            if handset_col and handset_col in exp.columns:
                col1,col2 = st.columns(2)
                with col1:
                    st.markdown("##### Avg throughput per handset (top 10)")
                    hs_tp = exp.groupby(handset_col)["Avg_TP"].mean().nlargest(10).reset_index()
                    hs_tp.columns = ["Handset","kbps"]
                    fig = go.Figure(go.Bar(x=hs_tp["kbps"], y=hs_tp["Handset"],
                        orientation="h", marker_color="#34D399"))
                    fig.update_layout(**T(height=320, yaxis=dict(autorange="reversed"),
                        margin=dict(l=0,r=40,t=10,b=0)))
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("##### Avg TCP retrans per handset (top 10)")
                    hs_tcp = exp.groupby(handset_col)["Avg_TCP"].mean().nlargest(10).reset_index()
                    hs_tcp.columns = ["Handset","Bytes"]
                    fig = go.Figure(go.Bar(x=hs_tcp["Bytes"], y=hs_tcp["Handset"],
                        orientation="h", marker_color="#FB7185"))
                    fig.update_layout(**T(height=320, yaxis=dict(autorange="reversed"),
                        margin=dict(l=0,r=40,t=10,b=0)))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Handset column not found in dataset.")
        with tab3:
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("##### Experience clusters (RTT vs Throughput)")
                ecmap = {"Good":"#34D399","Average":"#FBBF24","Poor":"#FB7185"}
                fig = go.Figure()
                for lvl,col in ecmap.items():
                    sub = exp[exp["Experience"]==lvl]
                    fig.add_trace(go.Scatter(x=sub["Avg_RTT"], y=sub["Avg_TP"],
                        mode="markers", name=lvl,
                        marker=dict(color=col, size=4, opacity=0.5)))
                fig.update_layout(**T(height=320, xaxis_title="Avg RTT (ms)",
                    yaxis_title="Throughput (kbps)", margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("##### Cluster distribution")
                counts = exp["Experience"].value_counts()
                fig = go.Figure(go.Pie(labels=counts.index, values=counts.values,
                    hole=0.55, marker_colors=["#34D399","#FBBF24","#FB7185"]))
                fig.update_layout(**T(height=320, margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4
# ══════════════════════════════════════════════════════════════════════════════
elif "Task 4" in page:
    st.markdown("### ⭐ Task 4 — Satisfaction Analysis")
    if not has_data:
        st.warning("Upload data from the sidebar to begin.")
    elif not msisdn_col or not dur_col:
        st.error("MSISDN or Duration column not found.")
    else:
        # Compute — note: only plain lists passed to compute_satisfaction
        eng, eng_c, sc_eng_m, sc_eng_s, _ = compute_engagement(
            df, msisdn_col, dur_col, list(APP_COLS.keys()))
        exp, exp_c, sc_exp_m, sc_exp_s = compute_experience(
            df, msisdn_col, handset_col)
        sat = compute_satisfaction(
            eng, exp, msisdn_col,
            eng_c, sc_eng_m, sc_eng_s,
            exp_c, sc_exp_m, sc_exp_s)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Users Scored",     f"{len(sat):,}")
        c2.metric("Avg Satisfaction", f"{sat['Satisfaction'].mean():.4f}")
        c3.metric("Max Satisfaction", f"{sat['Satisfaction'].max():.4f}")
        c4.metric("Avg Eng Score",    f"{sat['Eng_Score'].mean():.4f}")
        c5.metric("Avg Exp Score",    f"{sat['Exp_Score'].mean():.4f}")

        tab41,tab42,tab43,tab44,tab45,tab46,tab47 = st.tabs([
            "4.1 Scores","4.2 Top 10","4.3 Regression",
            "4.4 k=2 Clusters","4.5 Aggregation","4.6 MySQL Export","4.7 Model Tracking"])

        # ── 4.1 ──────────────────────────────────────────────────────────────
        with tab41:
            st.markdown("#### Task 4.1 — Engagement & Experience Scores (Euclidean Distance)")
            st.markdown("""
            <div style='background:#0D1B2E;border-left:3px solid #38BDF8;border-radius:0 8px 8px 0;
                        padding:14px 18px;font-size:13px;line-height:1.9;margin-bottom:16px;'>
              <b style='color:#38BDF8;'>Methodology</b><br>
              &bull; <b style='color:#34D399;'>Engagement Score</b> = Euclidean distance from each user to the <i>least engaged</i> cluster centroid.<br>
              &bull; <b style='color:#FB7185;'>Experience Score</b> = Euclidean distance from each user to the <i>worst experience</i> cluster centroid.
            </div>""", unsafe_allow_html=True)
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("##### Engagement score distribution")
                fig = go.Figure(go.Histogram(x=sat["Eng_Score"], nbinsx=40,
                    marker_color="#38BDF8", opacity=0.75))
                fig.add_vline(x=sat["Eng_Score"].median(), line_dash="dash", line_color="#FBBF24")
                fig.update_layout(**T(height=260, xaxis_title="Engagement Score",
                    yaxis_title="Count", margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("##### Experience score distribution")
                fig = go.Figure(go.Histogram(x=sat["Exp_Score"], nbinsx=40,
                    marker_color="#FB7185", opacity=0.75))
                fig.add_vline(x=sat["Exp_Score"].median(), line_dash="dash", line_color="#FBBF24")
                fig.update_layout(**T(height=260, xaxis_title="Experience Score",
                    yaxis_title="Count", margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("##### Engagement vs Experience (coloured by satisfaction)")
            fig = go.Figure(go.Scatter(
                x=sat["Eng_Score"], y=sat["Exp_Score"], mode="markers",
                marker=dict(color=sat["Satisfaction"], colorscale="Viridis",
                    size=4, opacity=0.5, colorbar=dict(title="Satisfaction",
                    tickfont_color="#C8D8E8"))))
            fig.update_layout(**T(height=320, xaxis_title="Engagement Score",
                yaxis_title="Experience Score", margin=dict(l=0,r=0,t=10,b=0)))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("##### Score table (first 50 rows)")
            st.dataframe(sat[[msisdn_col,"Eng_Score","Exp_Score","Satisfaction"]].head(50)
                .style.format({"Eng_Score":"{:.4f}","Exp_Score":"{:.4f}","Satisfaction":"{:.4f}"}),
                use_container_width=True)

        # ── 4.2 ──────────────────────────────────────────────────────────────
        with tab42:
            st.markdown("#### Task 4.2 — Top 10 Satisfied Customers")
            top10_sat = sat.nlargest(10,"Satisfaction")[
                [msisdn_col,"Eng_Score","Exp_Score","Satisfaction"]].reset_index(drop=True)
            top10_sat.index += 1
            st.dataframe(top10_sat.style.format(
                {"Eng_Score":"{:.4f}","Exp_Score":"{:.4f}","Satisfaction":"{:.4f}"}),
                use_container_width=True)
            fig = go.Figure(go.Bar(
                x=top10_sat[msisdn_col].astype(str), y=top10_sat["Satisfaction"],
                marker_color="#FBBF24",
                text=top10_sat["Satisfaction"].round(4), textposition="outside"))
            fig.update_layout(**T(height=320, xaxis_title="User (MSISDN)",
                yaxis_title="Satisfaction Score",
                xaxis=dict(tickangle=-30), margin=dict(l=0,r=0,t=20,b=60)))
            st.plotly_chart(fig, use_container_width=True)

        # ── 4.3 ──────────────────────────────────────────────────────────────
        with tab43:
            st.markdown("#### Task 4.3 — Regression Model: Predict Satisfaction Score")
            sat_m = sat.merge(
                eng[[msisdn_col,"Sessions","Total_Duration","Total_Traffic"]], on=msisdn_col
            ).merge(
                exp[[msisdn_col,"Avg_TCP","Avg_RTT","Avg_TP"]], on=msisdn_col
            ).dropna()
            feat = ["Sessions","Total_Duration","Total_Traffic","Avg_TCP","Avg_RTT","Avg_TP"]
            X = sat_m[feat].fillna(0)
            y = sat_m["Satisfaction"]
            X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            results = {}
            for name,mdl in [
                ("Linear Regression", LinearRegression()),
                ("Random Forest",     RandomForestRegressor(n_estimators=100,random_state=42)),
                ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100,random_state=42)),
            ]:
                t0 = time.time()
                mdl.fit(X_tr, y_tr)
                t1 = time.time()
                preds = mdl.predict(X_te)
                cv    = cross_val_score(mdl, X, y, cv=5, scoring="r2")
                results[name] = {
                    "R2":    r2_score(y_te, preds),
                    "MAE":   mean_absolute_error(y_te, preds),
                    "RMSE":  np.sqrt(mean_squared_error(y_te, preds)),
                    "CV":    cv.mean(), "CVs": cv.std(),
                    "time":  t1-t0, "model": mdl,
                    "preds": preds, "y_te": y_te,
                }
            best_name = max(results, key=lambda k: results[k]["R2"])
            col1,col2,col3 = st.columns(3)
            for col_ui,name in zip([col1,col2,col3], results):
                r = results[name]
                border = "#FBBF24" if name==best_name else "#1C3050"
                col_ui.markdown(f"""
                <div style='background:#0D1B2E;border:1px solid {border};border-radius:10px;padding:16px;text-align:center;'>
                  <div style='font-family:Space Mono,monospace;font-size:10px;color:#4A6080;letter-spacing:2px;'>{name.upper()}</div>
                  {'<div style="font-size:10px;color:#FBBF24;font-family:Space Mono,monospace;margin:4px 0;">&#9733; BEST</div>' if name==best_name else '<div style="height:20px"></div>'}
                  <div style='font-size:24px;font-family:Space Mono,monospace;color:#38BDF8;font-weight:700;'>{r["R2"]:.4f}</div>
                  <div style='font-size:10px;color:#4A6080;margin-bottom:6px;'>R&sup2;</div>
                  <div style='font-size:12px;color:#C8D8E8;'>MAE:  {r["MAE"]:.4f}</div>
                  <div style='font-size:12px;color:#C8D8E8;'>RMSE: {r["RMSE"]:.4f}</div>
                  <div style='font-size:12px;color:#A78BFA;'>CV R&sup2;: {r["CV"]:.3f} &plusmn; {r["CVs"]:.3f}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("---")
            col1,col2 = st.columns(2)
            with col1:
                st.markdown(f"##### Predicted vs Actual — {best_name}")
                br = results[best_name]
                mn = float(min(br["y_te"].min(), br["preds"].min()))
                mx = float(max(br["y_te"].max(), br["preds"].max()))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=br["y_te"], y=br["preds"], mode="markers",
                    marker=dict(color="#38BDF8", size=4, opacity=0.4), name="Predictions"))
                fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                    line=dict(color="#FB7185", dash="dash"), name="Perfect fit"))
                fig.update_layout(**T(height=300, xaxis_title="Actual",
                    yaxis_title="Predicted", margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if hasattr(results[best_name]["model"], "feature_importances_"):
                    st.markdown(f"##### Feature importance — {best_name}")
                    imp = pd.Series(results[best_name]["model"].feature_importances_,
                        index=feat).sort_values(ascending=True)
                    fig = go.Figure(go.Bar(x=imp.values, y=imp.index,
                        orientation="h", marker_color="#FBBF24"))
                    fig.update_layout(**T(height=300, xaxis_title="Importance",
                        margin=dict(l=0,r=0,t=10,b=0)))
                    st.plotly_chart(fig, use_container_width=True)

        # ── 4.4 ──────────────────────────────────────────────────────────────
        with tab44:
            st.markdown("#### Task 4.4 — k-means (k=2) on Engagement & Experience Scores")
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("##### Cluster scatter")
                fig = go.Figure()
                for cl,color in [(0,"#38BDF8"),(1,"#FB7185")]:
                    sub = sat[sat["Sat_Cluster"]==cl]
                    fig.add_trace(go.Scatter(x=sub["Eng_Score"], y=sub["Exp_Score"],
                        mode="markers", name=f"Cluster {cl}",
                        marker=dict(color=color, size=4, opacity=0.5)))
                fig.update_layout(**T(height=320, xaxis_title="Engagement Score",
                    yaxis_title="Experience Score", margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("##### Cluster sizes")
                counts = sat["Sat_Cluster"].value_counts()
                fig = go.Figure(go.Pie(
                    labels=[f"Cluster {c}" for c in counts.index],
                    values=counts.values, hole=0.55,
                    marker_colors=["#38BDF8","#FB7185"]))
                fig.update_layout(**T(height=320, margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("##### Satisfaction distribution per cluster")
            fig = go.Figure()
            for cl,color in [(0,"#38BDF8"),(1,"#FB7185")]:
                fig.add_trace(go.Histogram(
                    x=sat[sat["Sat_Cluster"]==cl]["Satisfaction"], nbinsx=30,
                    name=f"Cluster {cl}", marker_color=color, opacity=0.65))
            fig.update_layout(**T(height=280, barmode="overlay",
                xaxis_title="Satisfaction Score", yaxis_title="Count",
                margin=dict(l=0,r=0,t=10,b=0)))
            st.plotly_chart(fig, use_container_width=True)

        # ── 4.5 ──────────────────────────────────────────────────────────────
        with tab45:
            st.markdown("#### Task 4.5 — Average Satisfaction & Experience Score per Cluster")
            summary = sat.groupby("Sat_Cluster").agg(
                Users            =("Satisfaction","count"),
                Avg_Satisfaction =("Satisfaction","mean"),
                Avg_Eng_Score    =("Eng_Score","mean"),
                Avg_Exp_Score    =("Exp_Score","mean"),
                Std_Satisfaction =("Satisfaction","std"),
                Min_Satisfaction =("Satisfaction","min"),
                Max_Satisfaction =("Satisfaction","max"),
            ).reset_index()
            summary["Sat_Cluster"] = summary["Sat_Cluster"].apply(lambda x: f"Cluster {x}")
            st.dataframe(summary.style.format({
                "Avg_Satisfaction":"{:.4f}","Avg_Eng_Score":"{:.4f}",
                "Avg_Exp_Score":"{:.4f}","Std_Satisfaction":"{:.4f}",
                "Min_Satisfaction":"{:.4f}","Max_Satisfaction":"{:.4f}"}),
                use_container_width=True)
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("##### Avg scores per cluster")
                fig = go.Figure()
                for label,color in [("Avg_Eng_Score","#38BDF8"),
                                     ("Avg_Exp_Score","#34D399"),
                                     ("Avg_Satisfaction","#FBBF24")]:
                    fig.add_trace(go.Bar(
                        name=label.replace("Avg_","").replace("_Score",""),
                        x=summary["Sat_Cluster"], y=summary[label], marker_color=color))
                fig.update_layout(**T(height=300, barmode="group",
                    yaxis_title="Score", margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("##### Satisfaction box plot per cluster")
                fig = go.Figure()
                for i,row in summary.iterrows():
                    color = ["#38BDF8","#FB7185"][i]
                    fig.add_trace(go.Box(
                        y=sat[sat["Sat_Cluster"]==i]["Satisfaction"],
                        name=row["Sat_Cluster"], marker_color=color))
                fig.update_layout(**T(height=300, yaxis_title="Satisfaction Score",
                    margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)

        # ── 4.6 ──────────────────────────────────────────────────────────────
        with tab46:
            st.markdown("#### Task 4.6 — Export to MySQL")
            final = sat[[msisdn_col,"Eng_Score","Exp_Score","Satisfaction","Sat_Cluster"]].copy()
            final.columns = ["msisdn","engagement_score","experience_score",
                              "satisfaction_score","satisfaction_cluster"]
            st.markdown("##### Preview (first 20 rows)")
            st.dataframe(final.head(20).style.format({
                "engagement_score":"{:.4f}","experience_score":"{:.4f}",
                "satisfaction_score":"{:.4f}"}), use_container_width=True)
            csv_bytes = final.to_csv(index=False).encode("utf-8")
            st.download_button(label="⬇  Download user_satisfaction_scores.csv",
                data=csv_bytes, file_name="user_satisfaction_scores.csv", mime="text/csv")
            st.markdown("---")
            st.markdown("##### MySQL export code")
            st.code("""
# pip install sqlalchemy pymysql
from sqlalchemy import create_engine
import pandas as pd

df = pd.read_csv("user_satisfaction_scores.csv")

engine = create_engine(
    "mysql+pymysql://YOUR_USER:YOUR_PASSWORD@localhost:3306/tellco"
)
df.to_sql("user_satisfaction_scores", con=engine,
          if_exists="replace", index=False)
print("Exported", len(df), "rows")

# Verify
with engine.connect() as conn:
    for row in conn.execute("SELECT * FROM user_satisfaction_scores LIMIT 10"):
        print(row)
""", language="python")

        # ── 4.7 ──────────────────────────────────────────────────────────────
        with tab47:
            st.markdown("#### Task 4.7 — Model Deployment & Tracking Report")
            now = datetime.datetime.utcnow()
            sat_m = sat.merge(
                eng[[msisdn_col,"Sessions","Total_Duration","Total_Traffic"]], on=msisdn_col
            ).merge(
                exp[[msisdn_col,"Avg_TCP","Avg_RTT","Avg_TP"]], on=msisdn_col
            ).dropna()
            feat = ["Sessions","Total_Duration","Total_Traffic","Avg_TCP","Avg_RTT","Avg_TP"]
            X = sat_m[feat].fillna(0)
            y = sat_m["Satisfaction"]
            X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            model_configs = [
                ("linear_regression",  LinearRegression(),
                 {"fit_intercept":True}),
                ("random_forest",      RandomForestRegressor(n_estimators=100,random_state=42),
                 {"n_estimators":100,"max_depth":"None","random_state":42}),
                ("gradient_boosting",  GradientBoostingRegressor(n_estimators=100,random_state=42),
                 {"n_estimators":100,"learning_rate":0.1,"random_state":42}),
            ]
            tracking_rows = []
            best_r2, best_rid = -np.inf, None
            for run_id,(mname,mdl,params) in enumerate(model_configs, 1):
                t_start = now + datetime.timedelta(seconds=run_id*2)
                t0 = time.time()
                mdl.fit(X_tr, y_tr)
                t1 = time.time()
                preds = mdl.predict(X_te)
                r2    = r2_score(y_te, preds)
                mae   = mean_absolute_error(y_te, preds)
                rmse  = np.sqrt(mean_squared_error(y_te, preds))
                cv_r2 = cross_val_score(mdl, X, y, cv=5, scoring="r2").mean()
                t_end = t_start + datetime.timedelta(seconds=t1-t0)
                row = {
                    "run_id":            f"run_{run_id:03d}",
                    "experiment":        "tellco_satisfaction",
                    "model_name":        mname,
                    "code_version":      "v1.0.0",
                    "source":            "app.py",
                    "start_time":        t_start.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "end_time":          t_end.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "duration_s":        round(t1-t0,3),
                    "params":            str(params),
                    "metric_r2":         round(r2,6),
                    "metric_mae":        round(mae,6),
                    "metric_rmse":       round(rmse,6),
                    "metric_cv_r2":      round(cv_r2,6),
                    "artifact":          "user_satisfaction_scores.csv",
                    "best_model":        "no",
                }
                if r2 > best_r2:
                    best_r2, best_rid = r2, run_id-1
                tracking_rows.append(row)
            tracking_rows[best_rid]["best_model"] = "YES"
            tracking_df = pd.DataFrame(tracking_rows)

            st.markdown("##### Run log")
            st.dataframe(tracking_df, use_container_width=True)

            st.markdown("##### R² convergence across models")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tracking_df["model_name"], y=tracking_df["metric_r2"],
                mode="lines+markers", name="Test R²",
                line=dict(color="#38BDF8", width=2.5), marker=dict(size=10)))
            fig.add_trace(go.Scatter(x=tracking_df["model_name"], y=tracking_df["metric_cv_r2"],
                mode="lines+markers", name="CV R²",
                line=dict(color="#34D399", width=2.5, dash="dash"), marker=dict(size=10)))
            fig.update_layout(**T(height=260, yaxis_title="R²",
                margin=dict(l=0,r=0,t=10,b=0)))
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=tracking_df["model_name"], y=tracking_df["metric_mae"],
                name="MAE", marker_color="#FB7185"))
            fig2.add_trace(go.Bar(x=tracking_df["model_name"], y=tracking_df["metric_rmse"],
                name="RMSE", marker_color="#FBBF24"))
            fig2.update_layout(**T(height=240, barmode="group", yaxis_title="Error",
                margin=dict(l=0,r=0,t=10,b=0)))
            st.plotly_chart(fig2, use_container_width=True)

            best = tracking_rows[best_rid]
            st.markdown(f"""
            <div style='background:#0D1B2E;border:1px solid #FBBF24;border-radius:10px;padding:16px;margin-top:8px;'>
              <div style='font-family:Space Mono,monospace;font-size:11px;color:#FBBF24;letter-spacing:2px;margin-bottom:8px;'>
                &#9733; BEST RUN — {best["run_id"].upper()}
              </div>
              <div style='font-size:13px;display:flex;gap:24px;flex-wrap:wrap;'>
                <span>Model: <b style='color:#38BDF8;'>{best["model_name"]}</b></span>
                <span>R&sup2;: <b style='color:#34D399;'>{best["metric_r2"]}</b></span>
                <span>MAE: <b style='color:#FB7185;'>{best["metric_mae"]}</b></span>
                <span>RMSE: <b style='color:#FBBF24;'>{best["metric_rmse"]}</b></span>
                <span>CV R&sup2;: <b style='color:#A78BFA;'>{best["metric_cv_r2"]}</b></span>
              </div>
              <div style='margin-top:6px;font-size:12px;color:#4A6080;'>
                Start: {best["start_time"]} | Source: {best["source"]} | Version: {best["code_version"]}
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("---")
            col1,col2 = st.columns(2)
            with col1:
                tracking_csv = tracking_df.to_csv(index=False).encode("utf-8")
                st.download_button(label="⬇  model_tracking_report.csv",
                    data=tracking_csv, file_name="model_tracking_report.csv", mime="text/csv")
            with col2:
                full_export = sat[[msisdn_col,"Eng_Score","Exp_Score","Satisfaction","Sat_Cluster"]].copy()
                full_export.columns = ["msisdn","engagement_score","experience_score",
                                       "satisfaction_score","satisfaction_cluster"]
                st.download_button(label="⬇  user_satisfaction_scores.csv",
                    data=full_export.to_csv(index=False).encode("utf-8"),
                    file_name="user_satisfaction_scores.csv", mime="text/csv")

            st.markdown("##### MLflow + Docker deployment code")
            st.code("""
# ── MLflow tracking ───────────────────────────────────────────────
# pip install mlflow scikit-learn
import mlflow, mlflow.sklearn

mlflow.set_experiment("tellco_satisfaction")
with mlflow.start_run(run_name="gradient_boosting_v1"):
    mlflow.log_params({"n_estimators":100, "learning_rate":0.1})
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mlflow.log_metrics({
        "r2":   r2_score(y_test, preds),
        "mae":  mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds, squared=False),
    })
    mlflow.sklearn.log_model(model, "satisfaction_model")
    mlflow.log_artifact("user_satisfaction_scores.csv")

# ── Dockerfile ────────────────────────────────────────────────────
# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# COPY app.py .
# EXPOSE 8501
# CMD ["streamlit", "run", "app.py", "--server.port=8501"]
#
# docker build -t tellco-dashboard .
# docker run -p 8501:8501 tellco-dashboard
""", language="python")
