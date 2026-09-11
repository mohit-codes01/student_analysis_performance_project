"""
🎓 Student Performance — Interactive Dashboard
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Student Performance Dashboard", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.kpi {background:linear-gradient(135deg,#6C63FF,#8E7CFF);padding:18px;border-radius:16px;
      color:white;text-align:center;box-shadow:0 4px 12px rgba(0,0,0,.15);}
.kpi h2{margin:0;font-size:28px;}
.kpi p{margin:0;opacity:.85;}
.block-container{padding-top:1.5rem;}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Performance — Interactive Dashboard")
st.caption("Upload your `student_performance.csv`, or explore instantly with generated sample data.")

# ---------- LOAD DATA ----------
@st.cache_data
def make_sample(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["male", "female"], n)
    parent_edu = rng.choice(["high school", "some college", "associate's degree",
                              "bachelor's degree", "master's degree"], n)
    study_time = rng.uniform(0, 10, n).round(1)
    base = rng.normal(65, 12, n) + study_time * 1.8
    math = np.clip(base + rng.normal(0, 8, n), 0, 100).round(1)
    reading = np.clip(base + rng.normal(2, 8, n) + (gender == "female") * 4, 0, 100).round(1)
    writing = np.clip(reading + rng.normal(0, 5, n), 0, 100).round(1)
    attendance = np.clip(rng.normal(88, 8, n), 40, 100).round(1)
    avg = ((math + reading + writing) / 3).round(1)
    passfail = np.where(avg >= 50, "pass", "fail")
    return pd.DataFrame({
        "gender": gender, "parent_education": parent_edu, "study_time_hrs": study_time,
        "math_score": math, "reading_score": reading, "writing_score": writing,
        "attendance_pct": attendance, "average_score": avg, "pass_fail": passfail
    })

uploaded = st.file_uploader("📁 Upload CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = make_sample()
    st.info("Using generated sample data (same schema as the notebook). Upload your own CSV to replace it.")

# ---------- CLEAN ----------
df.columns = [c.strip() for c in df.columns]
if "gender" in df.columns:
    df["gender"] = df["gender"].astype(str).str.lower().str.strip().replace({"f": "female", "m": "male"})
if "pass_fail" in df.columns:
    df["pass_fail"] = df["pass_fail"].astype(str).str.lower().str.strip().replace({"f": "fail", "p": "pass"})
df = df.drop_duplicates()
for col in ["math_score", "reading_score", "writing_score", "attendance_pct", "study_time_hrs"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
if "average_score" not in df.columns and {"math_score", "reading_score", "writing_score"} <= set(df.columns):
    df["average_score"] = (df["math_score"] + df["reading_score"] + df["writing_score"]) / 3

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("🔎 Filters")
genders = st.sidebar.multiselect("Gender", sorted(df["gender"].dropna().unique()),
                                  default=list(df["gender"].dropna().unique()))
edu_col = "parent_education" if "parent_education" in df.columns else None
if edu_col:
    edus = st.sidebar.multiselect("Parent education", sorted(df[edu_col].dropna().unique()),
                                   default=list(df[edu_col].dropna().unique()))
study_min, study_max = float(df["study_time_hrs"].min()), float(df["study_time_hrs"].max())
study_range = st.sidebar.slider("Study time (hrs)", study_min, study_max, (study_min, study_max))

f = df[df["gender"].isin(genders) & df["study_time_hrs"].between(*study_range)]
if edu_col:
    f = f[f[edu_col].isin(edus)]

# ---------- KPI CARDS ----------
c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip(
    [c1, c2, c3, c4],
    ["Students", "Avg Score", "Pass Rate", "Avg Attendance"],
    [len(f), f"{f['average_score'].mean():.1f}",
     f"{(f['pass_fail'].eq('pass').mean()*100):.0f}%" if "pass_fail" in f.columns else "—",
     f"{f['attendance_pct'].mean():.1f}%" if "attendance_pct" in f.columns else "—"]
):
    col.markdown(f'<div class="kpi"><h2>{val}</h2><p>{label}</p></div>', unsafe_allow_html=True)

st.write("")

# ---------- TABS ----------
tabs = st.tabs(["👥 Gender", "📊 Distribution", "📦 Variation", "🔗 Reading↔Writing",
                "⏱️ Study Time", "🎓 Parent Education"])

with tabs[0]:
    g = f.groupby("gender")[["math_score", "reading_score", "writing_score"]].mean().reset_index()
    g_melt = g.melt(id_vars="gender", var_name="subject", value_name="score")
    fig = px.bar(g_melt, x="subject", y="score", color="gender", barmode="group",
                 title="Average Score by Gender", text_auto=".1f")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    subject = st.selectbox("Subject", ["math_score", "reading_score", "writing_score", "average_score"])
    fig = px.histogram(f, x=subject, nbins=25, color="gender", marginal="box",
                        title=f"Distribution of {subject}")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    subs = f[["math_score", "reading_score", "writing_score"]].melt(var_name="subject", value_name="score")
    fig = px.box(subs, x="subject", y="score", color="subject", points="outliers",
                 title="Score Spread & Outliers by Subject")
    st.plotly_chart(fig, use_container_width=True)
    var = f[["math_score", "reading_score", "writing_score"]].std().sort_values(ascending=False)
    st.markdown(f"**Highest variation:** `{var.index[0]}` (std = {var.iloc[0]:.2f})")

with tabs[3]:
    fig = px.scatter(f, x="reading_score", y="writing_score", color="gender", trendline="ols",
                      title="Reading vs Writing Score")
    st.plotly_chart(fig, use_container_width=True)
    corr = f["reading_score"].corr(f["writing_score"])
    st.markdown(f"**Correlation coefficient:** `{corr:.2f}`")

with tabs[4]:
    fig = px.scatter(f, x="study_time_hrs", y="average_score", color="gender", trendline="ols",
                      title="Study Time vs Average Score")
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    if edu_col:
        e = f.groupby(edu_col)["average_score"].mean().sort_values().reset_index()
        fig = px.bar(e, x="average_score", y=edu_col, orientation="h",
                     title="Average Score by Parent Education", text_auto=".1f", color="average_score",
                     color_continuous_scale="Purples")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No `parent_education` column found in this dataset.")

st.caption(f"Showing {len(f)} of {len(df)} students after filters.")