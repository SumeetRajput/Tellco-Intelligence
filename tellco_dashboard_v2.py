"""
TellCo Telecom Intelligence Dashboard
All bugs fixed:
  1. No **PLOTLY_THEME in update_layout — uses theme() helper instead
  2. yaxis conflict resolved via theme() merge
  3. KMeans/Scaler objects never passed to @st.cache_data — plain lists used
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
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,700;1,300&display=swap');
:root {
    --bg0:#08111F;--bg1:#0D1B2E;--bg2:#122338;--border:#1C3050;
    --text:#C8D8E8;--muted:#4A6080;--blue:#38BDF8;--green:#34D399;
    --red:#FB7185;--purple:#A78BFA;--amber:#FBBF24;--cyan:#22D3EE;
}
html,body,[data-testid="stApp"]{background-color:var(--bg0)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background-color:var(--bg1)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stMetric"]{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:16px 20px!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:11px!important;letter-spacing:1.5px!important;text-transform:uppercase!important;font-family:'Space Mono',monospace!important;}
[data-testid="stMetricValue"]{color:var(--blue)!important;font-family:'Space Mono',monospace!important;}
[data-testid="stTabs"] button{font-family:'Space Mono',monospace!important;font-size:12px!important;letter-spacing:1px!important;color:var(--muted)!important;}
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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

# ── Theme helper — NEVER put xaxis/yaxis in the base dict ────────────────────
_BASE_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0D1B2E",
    font_color="#C8D8E8",
    font_family="Space Mono",
    title_font_family="Space Mono",
    title_font_color="#C8D8E8",
    colorway=["#38BDF8","#34D399","#FB7185","#A78BFA","#FBBF24","#22D3EE","#F472B6","#FB923C"],
)

def T(**kw):
    """
    Build a layout dict: base theme + grid-aware axis defaults,
    with caller kwargs merged on top (so yaxis overrides work cleanly).
    """
    bx = dict(gridcolor="#1C3050", zerolinecolor="#1C3050")
    by = dict(gridcolor="#1C3050", zerolinecolor="#1C3050")
    if "xaxis" in kw:
        bx.update(kw.pop("xaxis"))
    if "yaxis" in kw:
        by.update(kw.pop("yaxis"))
    return {**_BASE_THEME, "xaxis": bx, "yaxis": by, **kw}

# ── Data helpers ──────────────────────────────────────────────────────────────
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
        if df[col].dtype in [np.float64,np.int64,np.float32]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    for col in df.select_dtypes(include=np.number).columns:
        Q1,Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        df[col] = df[col].clip(Q1-1.5*(Q3-Q1), Q3+1.5*(Q3-Q1))
    return df

@st.cache_data
def build_features(df):
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
    # Return plain Python lists — cache_data can hash these; sklearn objects cannot
    return eng, km.cluster_centers_.tolist(), sc.mean_.tolist(), sc.scale_.tolist(), inertias

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
    df["Avg_TP"]  = ((df[tp_dl] +df[tp_ul] )/2) if (tp_dl  and tp_ul)  else (df[tp_dl]  if tp_dl  else 0)
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
    # Return plain Python lists — cache_data can hash these; sklearn objects cannot
    return exp, km.cluster_centers_.tolist(), sc.mean_.tolist(), sc.scale_.tolist()

@st.cache_data
def compute_satisfaction(eng_df, exp_df, msisdn_col,
                         km_eng_c, sc_eng_m, sc_eng_s,
                         km_exp_c, sc_exp_m, sc_exp_s):
    """Reconstruct scaling math from plain lists — no sklearn objects needed."""
    def _scale(X, mean, scale_):
        return (X - np.array(mean)) / np.array(scale_)

    eng_centers = np.array(km_eng_c)
    exp_centers = np.array(km_exp_c)

    le = eng_df.groupby("Cluster")["Total_Traffic"].mean().idxmin()
    E_eng = _scale(eng_df[["Sessions","Total_Duration","Total_Traffic"]].fillna(0).values, sc_eng_m, sc_eng_s)
    eng_df = eng_df.copy()
    eng_df["Eng_Score"] = cdist(E_eng, [eng_centers[le]]).flatten()

    we = exp_df.groupby("Cluster")["Avg_RTT"].mean().idxmax()
    E_exp = _scale(exp_df[["Avg_TCP","Avg_RTT","Avg_TP"]].fillna(0).values, sc_exp_m, sc_exp_s)
    exp_df = exp_df.copy()
    exp_df["Exp_Score"] = cdist(E_exp, [exp_centers[we]]).flatten()

    sat = eng_df[[msisdn_col,"Eng_Score"]].merge(exp_df[[msisdn_col,"Exp_Score"]], on=msisdn_col)
    sat["Satisfaction"] = (sat["Eng_Score"] + sat["Exp_Score"]) / 2
    km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
    sat["Sat_Cluster"] = km2.fit_predict(sat[["Eng_Score","Exp_Score"]].fillna(0))
    return sat

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = load_data(upload)
if df_raw is not None:
    df, APP_COLS, msisdn_col, handset_col, manuf_col, dur_col = build_features(df_raw.copy())
    has_data = True
else:
    has_data = False

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div style='padding:24px 0 8px'>
      <div style='font-family:Space Mono,monospace;font-size:28px;font-weight:700;letter-spacing:3px;color:#38BDF8;'>TELLCO</div>
      <div style='font-family:Space Mono,monospace;font-size:12px;color:#4A6080;letter-spacing:4px;margin-top:2px;'>TELECOM INTELLIGENCE DASHBOARD</div>
    </div>
    """, unsafe_allow_html=True)
    if not has_data:
        st.markdown("""
        <div style='background:#0D1B2E;border:1px dashed #1C3050;border-radius:12px;padding:48px;text-align:center;margin-top:32px;'>
          <div style='font-size:48px;margin-bottom:16px;'>📡</div>
          <div style='font-family:Space Mono,monospace;font-size:16px;color:#38BDF8;letter-spacing:2px;'>AWAITING DATA UPLOAD</div>
          <div style='color:#4A6080;margin-top:8px;font-size:13px;'>Upload your TellCo CSV or Excel file using the sidebar to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Unique Users", f"{df[msisdn_col].nunique():,}" if msisdn_col else "N/A")
        c3.metric("Total Data (TB)", f"{df['Total_Data'].sum()/1e12:.2f}")
        c4.metric("Apps Tracked", len(APP_COLS))
        c5.metric("Features", df.shape[1])
        st.markdown("---")
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("#### App data usage breakdown")
            app_totals = {app: df[f"{app}_Total"].sum()/1e9 for app in APP_COLS}
            fig = go.Figure(go.Bar(
                x=list(app_totals.keys()), y=list(app_totals.values()),
                marker_color=["#38BDF8","#34D399","#FB7185","#A78BFA","#FBBF24","#22D3EE","#F472B6"],
                text=[f"{v:.1f} GB" for v in app_totals.values()],
                textposition="outside", textfont_color="#C8D8E8",
            ))
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
# TASK 1 — USER OVERVIEW
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
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=0,r=60,t=10,b=0)))
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
                        </div>
                        """, unsafe_allow_html=True)
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
                fig.update_layout(**T(height=280, xaxis_title="Duration (seconds)",
                    yaxis_title="Count", margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("##### Correlation: each app vs total data")
            app_corr = {app: df[f"{app}_Total"].corr(df["Total_Data"]) for app in APP_COLS}
            corr_s = pd.Series(app_corr).sort_values(ascending=False)
            fig = go.Figure(go.Bar(
                x=corr_s.index, y=corr_s.values,
                marker_color=["#34D399" if v>0 else "#FB7185" for v in corr_s.values],
                text=[f"{v:.3f}" for v in corr_s.values], textposition="outside"))
            fig.update_layout(**T(height=300, xaxis_title="Application",
                yaxis_title="Pearson r", margin=dict(l=0,r=0,t=10,b=0)))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### App-to-app correlation matrix")
            atc = [f"{app}_Total" for app in APP_COLS]
            cm  = df[atc].corr()
            fig = go.Figure(go.Heatmap(
                z=cm.values,
                x=[c.replace("_Total","") for c in cm.columns],
                y=[c.replace("_Total","") for c in cm.index],
                colorscale=[[0,"#FB7185"],[0.5,"#0D1B2E"],[1,"#38BDF8"]],
                zmid=0, text=cm.round(2).values, texttemplate="%{text}",
                colorbar=dict(tickfont_color="#C8D8E8")))
            fig.update_layout(**T(height=380, margin=dict(l=0,r=0,t=10,b=0)))
            st.plotly_chart(fig, use_container_width=True)

            if msisdn_col and dur_col and dur_col in df.columns:
                st.markdown("##### Decile analysis — total data by session duration decile")
                user_agg = df.groupby(msisdn_col).agg(
                    Dur=(dur_col,"sum"), Data=("Total_Data","sum")).reset_index()
                user_agg["Decile"] = pd.qcut(user_agg["Dur"], q=10,
                    labels=[f"D{i}" for i in range(1,11)], duplicates="drop")
                dec = user_agg.groupby("Decile", observed=True)["Data"].agg(["sum","mean","count"])
                fig = go.Figure(go.Bar(x=dec.index.astype(str), y=dec["sum"]/1e9,
                    marker_color="#A78BFA",
                    text=(dec["sum"]/1e9).round(1), textposition="outside"))
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
            c1.metric("PC1 variance", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
            c2.metric("PC1+2 variance", f"{cum_var[1]*100:.1f}%")
            c3.metric("3-component coverage", f"{cum_var[2]*100:.1f}%")
            c4.metric("Components for 80%", str(int(np.argmax(cum_var>=0.8))+1))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — ENGAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
elif "Task 2" in page:
    st.markdown("### 🔥 Task 2 — User Engagement Analysis")
    if not has_data:
        st.warning("Upload data from the sidebar to begin.")
    elif not msisdn_col or not dur_col:
        st.error("MSISDN or Duration column not found in dataset.")
    else:
        eng, km_eng_c, sc_eng_m, sc_eng_s, inertias = compute_engagement(
            df, msisdn_col, dur_col, list(APP_COLS.keys()))

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Unique Users", f"{len(eng):,}")
        c2.metric("Avg Sessions", f"{eng['Sessions'].mean():.1f}")
        c3.metric("Avg Traffic (MB)", f"{eng['Total_Traffic'].mean()/1e6:.1f}")
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
                    annotation_text="k=3 chosen", annotation_font_color="#FB7185")
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
                    mode="markers", name=lvl,
                    marker=dict(color=col, size=4, opacity=0.5)))
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
            # FIX: yaxis passed through T() — no duplicate keyword
            fig.update_layout(**T(height=340, xaxis_title=metric,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0,r=0,t=10,b=0)))
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
# TASK 3 — EXPERIENCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Task 3" in page:
    st.markdown("### 📶 Task 3 — Experience Analytics")
    if not has_data:
        st.warning("Upload data from the sidebar to begin.")
    elif not msisdn_col:
        st.error("MSISDN column not found.")
    else:
        exp, km_exp_c, sc_exp_m, sc_exp_s = compute_experience(df, msisdn_col, handset_col)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg TCP Retrans (B)", f"{exp['Avg_TCP'].mean():,.0f}")
        c2.metric("Avg RTT (ms)", f"{exp['Avg_RTT'].mean():.1f}")
        c3.metric("Avg Throughput (kbps)", f"{exp['Avg_TP'].mean():.1f}")
        c4.metric("Good Experience %", f"{(exp['Experience']=='Good').mean()*100:.1f}%")

        tab1,tab2,tab3 = st.tabs(["Distributions","Handset Analysis","Clustering"])

        with tab1:
            col1,col2,col3 = st.columns(3)
            for col_ui,metric,color,label in [
                (col1,"Avg_TCP","#FB7185","TCP Retransmission (Bytes)"),
                (col2,"Avg_RTT","#FBBF24","Avg RTT (ms)"),
                (col3,"Avg_TP", "#34D399","Throughput (kbps)"),
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
                    # FIX: yaxis passed through T() — no duplicate keyword
                    fig.update_layout(**T(height=320,
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=0,r=40,t=10,b=0)))
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("##### Avg TCP retrans per handset (top 10)")
                    hs_tcp = exp.groupby(handset_col)["Avg_TCP"].mean().nlargest(10).reset_index()
                    hs_tcp.columns = ["Handset","Bytes"]
                    fig = go.Figure(go.Bar(x=hs_tcp["Bytes"], y=hs_tcp["Handset"],
                        orientation="h", marker_color="#FB7185"))
                    fig.update_layout(**T(height=320,
                        yaxis=dict(autorange="reversed"),
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
            st.markdown("""
            <div style='background:#0D1B2E;border-left:3px solid #A78BFA;border-radius:0 8px 8px 0;
                        padding:14px 18px;font-size:13px;line-height:1.9;'>
              <b style='color:#A78BFA;'>Cluster descriptions</b><br>
              • <b style='color:#34D399;'>Good:</b> Low RTT, low TCP retransmission, high throughput.<br>
              • <b style='color:#FBBF24;'>Average:</b> Moderate values. Acceptable but improvable.<br>
              • <b style='color:#FB7185;'>Poor:</b> High RTT, high packet loss, low throughput.
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — SATISFACTION
# ══════════════════════════════════════════════════════════════════════════════
elif "Task 4" in page:
    st.markdown("### ⭐ Task 4 — Satisfaction Analysis")
    if not has_data:
        st.warning("Upload data from the sidebar to begin.")
    elif not msisdn_col or not dur_col:
        st.error("MSISDN or Duration column not found.")
    else:
        eng, km_eng_c, sc_eng_m, sc_eng_s, _ = compute_engagement(
            df, msisdn_col, dur_col, list(APP_COLS.keys()))
        exp, km_exp_c, sc_exp_m, sc_exp_s = compute_experience(df, msisdn_col, handset_col)
        # FIX: plain lists passed — no sklearn objects, no UnhashableParamError
        sat = compute_satisfaction(eng, exp, msisdn_col,
            km_eng_c, sc_eng_m, sc_eng_s,
            km_exp_c, sc_exp_m, sc_exp_s)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Users Scored", f"{len(sat):,}")
        c2.metric("Avg Satisfaction", f"{sat['Satisfaction'].mean():.3f}")
        c3.metric("Max Satisfaction", f"{sat['Satisfaction'].max():.3f}")
        c4.metric("Cluster 0 size", f"{(sat['Sat_Cluster']==0).sum():,}")

        tab1,tab2,tab3,tab4 = st.tabs(["Scores","Regression","k=2 Clusters","Export"])

        with tab1:
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("##### Satisfaction score distribution")
                fig = go.Figure(go.Histogram(x=sat["Satisfaction"], nbinsx=35,
                    marker_color="#A78BFA", opacity=0.75))
                fig.add_vline(x=sat["Satisfaction"].median(), line_dash="dash",
                    line_color="#FB7185")
                fig.update_layout(**T(height=280, xaxis_title="Score",
                    yaxis_title="Count", margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("##### Engagement vs Experience score")
                fig = go.Figure()
                for cl,col in [(0,"#38BDF8"),(1,"#FB7185")]:
                    sub = sat[sat["Sat_Cluster"]==cl]
                    fig.add_trace(go.Scatter(x=sub["Eng_Score"], y=sub["Exp_Score"],
                        mode="markers", name=f"Cluster {cl}",
                        marker=dict(color=col, size=4, opacity=0.5)))
                fig.update_layout(**T(height=280, xaxis_title="Engagement Score",
                    yaxis_title="Experience Score", margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("##### Top 10 most satisfied customers")
            top10 = sat.nlargest(10,"Satisfaction")[[msisdn_col,"Eng_Score","Exp_Score","Satisfaction"]]
            st.dataframe(top10.style.format(
                {"Eng_Score":"{:.4f}","Exp_Score":"{:.4f}","Satisfaction":"{:.4f}"}),
                use_container_width=True)

        with tab2:
            st.markdown("##### Regression models — predict satisfaction score")
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
                mdl.fit(X_tr, y_tr)
                preds = mdl.predict(X_te)
                results[name] = {
                    "R2":   r2_score(y_te, preds),
                    "MAE":  mean_absolute_error(y_te, preds),
                    "RMSE": np.sqrt(mean_squared_error(y_te, preds)),
                    "model": mdl,
                }
            best_name = max(results, key=lambda k: results[k]["R2"])
            col1,col2,col3 = st.columns(3)
            for col_ui,name in zip([col1,col2,col3], results):
                r = results[name]
                border = "#FBBF24" if name==best_name else "#1C3050"
                col_ui.markdown(f"""
                <div style='background:#0D1B2E;border:1px solid {border};border-radius:10px;
                            padding:16px;text-align:center;'>
                  <div style='font-family:Space Mono,monospace;font-size:10px;color:#4A6080;
                              letter-spacing:2px;'>{name.upper()}</div>
                  {'<div style="font-size:10px;color:#FBBF24;font-family:Space Mono,monospace;margin:4px 0;">★ BEST</div>' if name==best_name else '<div style="height:20px"></div>'}
                  <div style='font-size:24px;font-family:Space Mono,monospace;color:#38BDF8;
                              font-weight:700;'>{r["R2"]:.4f}</div>
                  <div style='font-size:10px;color:#4A6080;margin-bottom:8px;'>R²</div>
                  <div style='font-size:12px;color:#C8D8E8;'>MAE: {r["MAE"]:.4f}</div>
                  <div style='font-size:12px;color:#C8D8E8;'>RMSE: {r["RMSE"]:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            if hasattr(results[best_name]["model"], "feature_importances_"):
                st.markdown(f"##### Feature importance — {best_name}")
                imp = pd.Series(results[best_name]["model"].feature_importances_,
                    index=feat).sort_values(ascending=True)
                fig = go.Figure(go.Bar(x=imp.values, y=imp.index,
                    orientation="h", marker_color="#FBBF24"))
                fig.update_layout(**T(height=260, xaxis_title="Importance",
                    margin=dict(l=0,r=0,t=10,b=0)))
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("##### k=2 cluster summary")
            summary = sat.groupby("Sat_Cluster").agg(
                Avg_Satisfaction=("Satisfaction","mean"),
                Avg_Engagement=("Eng_Score","mean"),
                Avg_Experience=("Exp_Score","mean"),
                Count=("Satisfaction","count"),
            ).reset_index()
            st.dataframe(summary.style.format({
                "Avg_Satisfaction":"{:.4f}","Avg_Engagement":"{:.4f}","Avg_Experience":"{:.4f}"}),
                use_container_width=True)
            fig = go.Figure()
            for label,col in [("Engagement","#38BDF8"),("Experience","#34D399"),("Satisfaction","#FBBF24")]:
                fig.add_trace(go.Bar(name=label, x=["Cluster 0","Cluster 1"],
                    y=summary[f"Avg_{label}"], marker_color=col))
            fig.update_layout(**T(height=300, barmode="group",
                yaxis_title="Score", margin=dict(l=0,r=0,t=10,b=0)))
            st.plotly_chart(fig, use_container_width=True)

        # ── Build final table with predicted satisfaction ─────────────────────
        sat_m2 = sat.merge(
            eng[[msisdn_col,"Sessions","Total_Duration","Total_Traffic"]], on=msisdn_col
        ).merge(
            exp[[msisdn_col,"Avg_TCP","Avg_RTT","Avg_TP"]], on=msisdn_col
        ).dropna()
        feat2 = ["Sessions","Total_Duration","Total_Traffic","Avg_TCP","Avg_RTT","Avg_TP"]
        X2 = sat_m2[feat2].fillna(0)
        y2 = sat_m2["Satisfaction"]
        from sklearn.ensemble import RandomForestRegressor as RFR2
        best_mdl2 = RFR2(n_estimators=100, random_state=42)
        best_mdl2.fit(X2, y2)
        sat_m2["Predicted_Satisfaction"] = best_mdl2.predict(X2)

        final_table = sat_m2[[
            msisdn_col,
            "Eng_Score", "Exp_Score",
            "Satisfaction", "Predicted_Satisfaction",
            "Sat_Cluster",
            "Sessions", "Total_Duration", "Total_Traffic",
            "Avg_TCP", "Avg_RTT", "Avg_TP",
        ]].copy()
        final_table.columns = [
            "user_id",
            "engagement_score", "experience_score",
            "satisfaction_score", "predicted_satisfaction",
            "satisfaction_cluster",
            "sessions", "total_duration_ms", "total_traffic_bytes",
            "avg_tcp_retrans", "avg_rtt_ms", "avg_throughput_kbps",
        ]

        with tab4:
            st.markdown("### 📦 Task 4.6 — Export to MySQL")

            # ── Preview table ────────────────────────────────────────────────
            st.markdown("##### Final table preview (user_id + all scores + predicted)")
            st.dataframe(
                final_table.head(20).style.format({
                    "engagement_score":    "{:.4f}",
                    "experience_score":    "{:.4f}",
                    "satisfaction_score":  "{:.4f}",
                    "predicted_satisfaction": "{:.4f}",
                }),
                use_container_width=True,
            )
            st.caption(f"Total rows: {len(final_table):,}  |  Columns: {list(final_table.columns)}")

            # ── CSV download ─────────────────────────────────────────────────
            csv_bytes = final_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇  Download user_satisfaction_scores.csv",
                data=csv_bytes,
                file_name="user_satisfaction_scores.csv",
                mime="text/csv",
            )

            st.markdown("---")

            # ── MySQL live export ─────────────────────────────────────────────
            st.markdown("##### Connect & push to MySQL (Task 4.6)")
            st.markdown("""
            <div style='background:#0D1B2E;border-left:3px solid #38BDF8;border-radius:0 8px 8px 0;
                        padding:12px 16px;font-size:12px;color:#4A6080;margin-bottom:12px;'>
              Fill in your local MySQL credentials below, then click <b style='color:#38BDF8;'>Export to MySQL</b>.
              The table <code>user_satisfaction_scores</code> will be created (or replaced) in your database.
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b, col_c = st.columns(3)
            db_host = col_a.text_input("Host",     value="localhost", key="db_host")
            db_port = col_b.text_input("Port",     value="3306",      key="db_port")
            db_name = col_c.text_input("Database", value="tellco_db", key="db_name")
            col_d, col_e = st.columns(2)
            db_user = col_d.text_input("Username", value="root",  key="db_user")
            db_pass = col_e.text_input("Password", value="",      key="db_pass", type="password")

            if st.button("🚀  Export to MySQL", key="mysql_export_btn"):
                try:
                    from sqlalchemy import create_engine, text
                    conn_str = (
                        f"mysql+pymysql://{db_user}:{db_pass}"
                        f"@{db_host}:{db_port}/{db_name}"
                    )
                    engine = create_engine(conn_str)
                    with engine.connect() as conn:
                        final_table.to_sql(
                            "user_satisfaction_scores",
                            con=conn,
                            if_exists="replace",
                            index=False,
                        )
                        result = conn.execute(
                            text("SELECT COUNT(*) as total_rows FROM user_satisfaction_scores")
                        )
                        row_count = result.fetchone()[0]

                    st.success(f"✅ Exported {len(final_table):,} rows to `{db_name}.user_satisfaction_scores`")
                    st.markdown(f"""
                    <div style='background:#0D1B2E;border:1px solid #34D399;border-radius:8px;
                                padding:16px;font-family:Space Mono,monospace;font-size:12px;margin-top:8px;'>
                      <div style='color:#34D399;margin-bottom:8px;'>SELECT query output</div>
                      <div style='color:#C8D8E8;'>
                        SELECT COUNT(*) as total_rows FROM user_satisfaction_scores;<br>
                        <span style='color:#FBBF24;'>→ {row_count:,} rows</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show SELECT TOP 5 preview
                    with engine.connect() as conn:
                        preview = pd.read_sql(
                            "SELECT user_id, satisfaction_score, predicted_satisfaction, "
                            "engagement_score, experience_score FROM user_satisfaction_scores LIMIT 5",
                            conn,
                        )
                    st.markdown("**SELECT preview from MySQL:**")
                    st.dataframe(preview, use_container_width=True)

                except ImportError:
                    st.error("pymysql not installed. Run: pip install pymysql sqlalchemy")
                except Exception as e:
                    st.error(f"MySQL connection failed: {e}")
                    st.info("Make sure MySQL is running and credentials are correct.")

            st.markdown("---")

            # ══════════════════════════════════════════════════════════════════
            # TASK 4.7 — MLflow model tracking panel
            # ══════════════════════════════════════════════════════════════════
            st.markdown("### 📊 Task 4.7 — MLflow Model Deployment Tracking")

            st.markdown("""
            <div style='background:#0D1B2E;border-left:3px solid #A78BFA;border-radius:0 8px 8px 0;
                        padding:12px 16px;font-size:12px;color:#C8D8E8;margin-bottom:16px;line-height:1.8;'>
              Clicking <b style='color:#A78BFA;'>Run MLflow Tracking</b> will train all 3 models,
              log code version, start/end time, parameters, metrics and artifacts to MLflow,
              and save a CSV artifact. View results in the MLflow UI by running
              <code style='color:#38BDF8;'>mlflow ui</code> in your terminal.
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶  Run MLflow Tracking", key="mlflow_btn"):
                import mlflow
                import mlflow.sklearn
                import datetime, os, tempfile

                mlflow.set_experiment("TellCo_Satisfaction_Model")

                feat_mlf = ["Sessions","Total_Duration","Total_Traffic","Avg_TCP","Avg_RTT","Avg_TP"]
                X_mlf = sat_m2[feat_mlf].fillna(0)
                y_mlf = sat_m2["Satisfaction"]
                from sklearn.model_selection import train_test_split as tts
                X_tr, X_te, y_tr, y_te = tts(X_mlf, y_mlf, test_size=0.2, random_state=42)

                models_mlf = {
                    "Linear Regression":  LinearRegression(),
                    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
                    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
                }

                mlf_results = []
                progress = st.progress(0, text="Training models...")

                for idx, (name, mdl) in enumerate(models_mlf.items()):
                    start_time = datetime.datetime.now()
                    with mlflow.start_run(run_name=name):
                        # Tags
                        mlflow.set_tag("code_version",  "v2.0.0")
                        mlflow.set_tag("source",        "TellCo Dashboard — Task 4.7")
                        mlflow.set_tag("model_type",    name)
                        mlflow.set_tag("start_time",    str(start_time))

                        # Params
                        mlflow.log_param("test_size",   0.2)
                        mlflow.log_param("random_state", 42)
                        mlflow.log_param("n_train",     len(X_tr))
                        mlflow.log_param("n_test",      len(X_te))
                        mlflow.log_param("features",    feat_mlf)
                        mlflow.log_param("target",      "Satisfaction_Score")
                        if hasattr(mdl, "n_estimators"):
                            mlflow.log_param("n_estimators", mdl.n_estimators)

                        # Train
                        mdl.fit(X_tr, y_tr)
                        preds = mdl.predict(X_te)

                        # Metrics
                        r2   = r2_score(y_te, preds)
                        mae  = mean_absolute_error(y_te, preds)
                        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
                        mlflow.log_metric("r2",   r2)
                        mlflow.log_metric("mae",  mae)
                        mlflow.log_metric("rmse", rmse)

                        # Log loss convergence for GBM
                        if hasattr(mdl, "train_score_"):
                            for step, loss in enumerate(mdl.train_score_):
                                mlflow.log_metric("train_loss", loss, step=step)

                        # Artifact: CSV
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".csv", delete=False, prefix=f"{name.replace(' ','_')}_"
                        ) as tmp:
                            final_table.to_csv(tmp.name, index=False)
                            mlflow.log_artifact(tmp.name, artifact_path="outputs")
                            os.unlink(tmp.name)

                        # Log model
                        mlflow.sklearn.log_model(mdl, "model")

                        end_time = datetime.datetime.now()
                        mlflow.set_tag("end_time", str(end_time))
                        duration = (end_time - start_time).total_seconds()

                    mlf_results.append({
                        "Model": name,
                        "R²": round(r2, 4),
                        "MAE": round(mae, 4),
                        "RMSE": round(rmse, 4),
                        "Duration (s)": round(duration, 2),
                        "Start": start_time.strftime("%H:%M:%S"),
                        "End": end_time.strftime("%H:%M:%S"),
                        "Code Version": "v2.0.0",
                        "Artifact": "user_satisfaction_scores.csv",
                    })
                    progress.progress((idx + 1) / 3, text=f"Logged: {name}")

                progress.empty()
                st.success("✅ MLflow tracking complete for all 3 models!")

                # Display tracking report table
                mlf_df = pd.DataFrame(mlf_results)
                best_idx = mlf_df["R²"].idxmax()

                st.markdown("##### Model tracking report")
                st.dataframe(
                    mlf_df.style
                        .highlight_max(subset=["R²"], color="#1a3a1a")
                        .highlight_min(subset=["MAE","RMSE"], color="#1a3a1a")
                        .format({"R²":"{:.4f}","MAE":"{:.4f}","RMSE":"{:.4f}"}),
                    use_container_width=True,
                )

                best_row = mlf_df.iloc[best_idx]
                st.markdown(f"""
                <div style='background:#0D1B2E;border:1px solid #FBBF24;border-radius:10px;
                            padding:20px;margin-top:12px;font-family:Space Mono,monospace;font-size:12px;'>
                  <div style='color:#FBBF24;font-size:14px;font-weight:700;margin-bottom:14px;'>
                    ★ BEST MODEL — {best_row["Model"].upper()}
                  </div>
                  <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;'>
                    <div><div style='color:#4A6080;'>R²</div>
                         <div style='color:#38BDF8;font-size:18px;font-weight:700;'>{best_row["R²"]}</div></div>
                    <div><div style='color:#4A6080;'>MAE</div>
                         <div style='color:#34D399;font-size:18px;font-weight:700;'>{best_row["MAE"]}</div></div>
                    <div><div style='color:#4A6080;'>RMSE</div>
                         <div style='color:#FB7185;font-size:18px;font-weight:700;'>{best_row["RMSE"]}</div></div>
                    <div><div style='color:#4A6080;'>Duration</div>
                         <div style='color:#A78BFA;font-size:18px;font-weight:700;'>{best_row["Duration (s)"]}s</div></div>
                  </div>
                  <div style='margin-top:14px;border-top:1px solid #1C3050;padding-top:12px;color:#C8D8E8;line-height:2;'>
                    Code version: <span style='color:#38BDF8;'>v2.0.0</span> &nbsp;|&nbsp;
                    Start: <span style='color:#38BDF8;'>{best_row["Start"]}</span> &nbsp;|&nbsp;
                    End: <span style='color:#38BDF8;'>{best_row["End"]}</span> &nbsp;|&nbsp;
                    Artifact: <span style='color:#38BDF8;'>user_satisfaction_scores.csv</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # GBM loss convergence chart
                st.markdown("##### Gradient Boosting — loss convergence")
                from sklearn.ensemble import GradientBoostingRegressor as GBR2
                gbm_check = GradientBoostingRegressor(n_estimators=100, random_state=42)
                gbm_check.fit(X_tr, y_tr)
                fig_loss = go.Figure(go.Scatter(
                    x=list(range(1, 101)),
                    y=gbm_check.train_score_,
                    mode="lines",
                    line=dict(color="#A78BFA", width=2),
                    name="Train score",
                ))
                fig_loss.update_layout(**T(
                    height=280,
                    title="GBM training score per estimator (loss convergence)",
                    xaxis_title="Estimator (iteration)",
                    yaxis_title="Train score (R²)",
                    margin=dict(l=0, r=0, t=40, b=0),
                ))
                st.plotly_chart(fig_loss, use_container_width=True)

                st.markdown("""
                <div style='background:#0D1B2E;border-left:3px solid #22D3EE;border-radius:0 8px 8px 0;
                            padding:12px 16px;font-size:12px;color:#C8D8E8;margin-top:8px;'>
                  <b style='color:#22D3EE;'>View full MLflow UI</b><br>
                  Open a new terminal and run: <code style='color:#38BDF8;'>mlflow ui</code><br>
                  Then open: <a href='http://localhost:5000' style='color:#38BDF8;'>http://localhost:5000</a>
                  to see all runs, parameters, metrics and artifacts.
                </div>
                """, unsafe_allow_html=True)

                # Download tracking report as CSV for submission
                mlf_csv = mlf_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇  Download mlflow_tracking_report.csv",
                    data=mlf_csv,
                    file_name="mlflow_tracking_report.csv",
                    mime="text/csv",
                )
