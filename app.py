import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
    from scipy.spatial.distance import pdist
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="Eurovision Voting Explorer", layout="wide")
st.title("Eurovision Voting Explorer")
st.caption("Interactive dashboard for Eurovision directed voting patterns using NVS.")

# Professional UI Color Palettes
# NVS_SCALE: Perceptually uniform teal sequence for density and strength.
NVS_SCALE = [
    [0.00, "#f8f9fa"], 
    [0.15, "#e0f2f1"], 
    [0.30, "#80cbc4"],
    [0.50, "#26a69a"], 
    [0.70, "#00897b"],
    [0.85, "#00695c"],
    [1.00, "#004d40"], 
]

# CORR_SCALE: Diverging palette with a pure white midpoint for neutral correlation.
CORR_SCALE = [
    [0.00, "#005d5d"], 
    [0.25, "#97d4d4"], 
    [0.50, "#ffffff"], 
    [0.75, "#ffb3b3"], 
    [1.00, "#a2191f"], 
]

# Standard Eurovision Point Caps
# 1975-2015: Max 12 points. 2016-Present: Max 24 points (Jury + Televote combined).
ERA_MAX = {y: 12 for y in range(1975, 2016)}
ERA_MAX.update({y: 24 for y in range(2016, 2026)})

def find_col(cols, exact, fuzzy):
    """Robust column matching for varying CSV schemas."""
    for c in cols:
        if c.lower() in exact: return c
    for c in cols:
        if any(k in c.lower() for k in fuzzy): return c
    raise ValueError(f"Column not found. exact={exact}, fuzzy={fuzzy}")

def linear_trend_slope(years, values):
    """Calculates the rate of change in voting strength over time."""
    if len(years) < 2: return 0.0
    x, y = np.asarray(years, dtype=float), np.asarray(values, dtype=float)
    x = x - x.mean()
    denom = np.sum(x * x)
    return float(np.sum(x * y) / denom) if denom != 0 else 0.0

@st.cache_data
def load_data(nodes_file, edges_file):
    """Loads and cleans Eurovision dataset, filtering for Grand Final totals."""
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)
    nodes.columns = [c.strip().lower() for c in nodes.columns]
    edges.columns = [c.strip().lower() for c in edges.columns]
    
    id2label = nodes.set_index("id")["label"].to_dict()
    
    for col in ["score_type", "round"]:
        if col in edges.columns:
            edges[col] = edges[col].astype(str).str.strip().str.lower()
            val = "total" if col == "score_type" else "final"
            if val in edges[col].unique():
                edges = edges[edges[col] == val]

    src_col = find_col(edges.columns, {"source", "from"}, ["source", "from", "voter"])
    tgt_col = find_col(edges.columns, {"target", "to"}, ["target", "to", "recip"])
    pts_col = [c for c in edges.columns if any(k in c.lower() for k in ("weight", "point", "score")) and c.lower() != "year"][0]
    
    return nodes, edges, id2label, src_col, tgt_col, pts_col

nodes, edges, id2label, src_col, tgt_col, pts_col = load_data("nodes_with_coordinates.csv", "eurovision_senior.csv")
all_years = sorted(edges["year"].unique())

with st.sidebar:
    st.header("Analysis Controls")
    start_year = st.selectbox("Start", all_years, index=0)
    end_year = st.selectbox("End", all_years, index=len(all_years)-1)
    top_n_choice = st.selectbox("Limit Countries", ["All", 10, 20, 30, 40], index=0)
    order_mode = st.selectbox("Sort By", ["strength", "cluster", "alphabetical"])
    view_mode = st.selectbox("Metric", ["nvs", "raw", "correlation"])
    threshold = st.slider("Min NVS Visibility", 0.0, 6.0, 0.0)

@st.cache_data
def compute_period_data(start_year, end_year):
    """
    Normalizes Eurovision points into an NVS (Normalized Voting Strength) score.
    Points are divided by the era's maximum possible points to allow longitudinal comparison.
    """
    df = edges[(edges["year"] >= start_year) & (edges["year"] <= end_year)].copy()
    if df.empty: return None
    
    actual = df.groupby(["year", src_col, tgt_col])[pts_col].sum().reset_index()
    actual.columns = ["year", "source", "target", "points"]

    # Calculate eligibility: determining which countries could have voted for each other in each year.
    pair_rows = []
    for yr in sorted(df["year"].unique()):
        p = sorted(set(df[df["year"]==yr][src_col]) | set(df[df["year"]==yr][tgt_col]))
        for s in p:
            for t in p:
                if s != t: pair_rows.append((yr, s, t))

    eligible = pd.DataFrame(pair_rows, columns=["year", "source", "target"])
    yr_agg = eligible.merge(actual, on=["year", "source", "target"], how="left").fillna(0)
    yr_agg["nvs_year"] = (yr_agg["points"] / yr_agg["year"].map(ERA_MAX).fillna(12)).clip(0, 1)

    # Average NVS score across the selected period, scaled to a 12-point visual.
    agg = yr_agg.groupby(["source", "target"]).agg(nvs_sum=("nvs_year", "sum"), years_eligible=("year", "nunique")).reset_index()
    raw_total = actual.groupby(["source", "target"])["points"].sum().reset_index(name="total_votes")
    
    agg = agg.merge(raw_total, on=["source", "target"], how="left").fillna(0)
    agg["nvs_score"] = (agg["nvs_sum"] / agg["years_eligible"]) * 12
    agg["src_label"] = agg["source"].map(id2label).fillna(agg["source"])
    agg["tgt_label"] = agg["target"].map(id2label).fillna(agg["target"])
    
    return {"df": df, "yr_agg": yr_agg, "agg": agg}

pdata = compute_period_data(start_year, end_year)
if not pdata: st.stop()

# Sorting Logic
agg = pdata["agg"]
all_labels = sorted(set(agg["src_label"]) | set(agg["tgt_label"]))
if order_mode == "strength":
    s = agg.groupby("src_label")["nvs_score"].sum().add(agg.groupby("tgt_label")["nvs_score"].sum(), fill_value=0)
    order = sorted(all_labels, key=lambda x: s.get(x, 0), reverse=True)
elif order_mode == "cluster" and SCIPY_OK:
    m = agg.pivot(index="src_label", columns="tgt_label", values="nvs_score").fillna(0).reindex(index=all_labels, columns=all_labels, fill_value=0)
    order_idx = leaves_list(optimal_leaf_ordering(linkage(pdist(m.values), "average"), pdist(m.values)))
    order = [all_labels[i] for i in order_idx]
else:
    order = sorted(all_labels)

if top_n_choice != "All": order = order[:int(top_n_choice)]

# Data Matrix Construction
m_nvs = agg.pivot(index="src_label", columns="tgt_label", values="nvs_score").fillna(0).reindex(index=order, columns=order, fill_value=0)
z_nvs = np.where(m_nvs.values >= threshold, m_nvs.values, np.nan)
z_corr = m_nvs.T.corr().fillna(0).values

# Visualization Setup
fig = go.Figure()

if view_mode == "nvs":
    z_data, scale, z_min, z_max, title_cb, fmt = z_nvs, NVS_SCALE, 0, 12, "NVS Score", ".1f"
elif view_mode == "correlation":
    z_data, scale, z_min, z_max, title_cb, fmt = z_corr, CORR_SCALE, -1, 1, "Correlation", ".2f"

fig.add_trace(go.Heatmap(
    z=z_data, x=order, y=order,
    colorscale=scale, zmin=z_min, zmax=z_max,
    text=[[f"{val:{fmt}}" if pd.notna(val) else "" for val in row] for row in z_data],
    texttemplate="%{text}",
    textfont=dict(size=8, color="#222"),
    xgap=1, ygap=1,
    colorbar=dict(title=dict(text=title_cb, font_size=12), thickness=15)
))

# High-Contrast Axis Styling (Black)
fig.update_layout(
    title=f"Eurovision Voting Explorer ({start_year}–{end_year})",
    xaxis=dict(
        tickangle=-45, tickfont=dict(size=9, color="black"),
        linecolor="black", mirror=True, showline=True, tickcolor="black"
    ),
    yaxis=dict(
        autorange="reversed", tickfont=dict(size=9, color="black"),
        linecolor="black", mirror=True, showline=True, tickcolor="black"
    ),
    height=max(800, len(order) * 20 + 200),
    margin=dict(l=150, r=50, t=100, b=150),
    paper_bgcolor="white", plot_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Top Directed Pairs (Selection)")
st.dataframe(agg.sort_values("nvs_score", ascending=False).head(25), use_container_width=True)