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

# =============================================================================
# PAGE
# =============================================================================
st.set_page_config(page_title="Eurovision Voting Explorer", layout="wide")
st.title("Eurovision Voting Explorer")
st.caption("Interactive dashboard for Eurovision directed voting patterns using NVS.")

# =============================================================================
# CONFIG
# =============================================================================
NODES_FILE = "nodes_with_coordinates.csv"
EDGES_FILE = "eurovision_senior.csv"

ROOT_START = 1975
ROOT_END = 2025

ERA_MAX = {y: 12 for y in range(1975, 2016)}
ERA_MAX.update({y: 24 for y in range(2016, 2026)})

# Better thesis-friendly blue scale
NVS_SCALE = [
    [0.00, "#f8fbff"],
    [0.08, "#edf5fc"],
    [0.18, "#dbe9f6"],
    [0.32, "#bfd7ee"],
    [0.48, "#93c3e5"],
    [0.64, "#5ea7d8"],
    [0.78, "#2f7fbe"],
    [0.90, "#165a9c"],
    [1.00, "#0b3c6f"],
]

# Better diverging correlation scale
CORR_SCALE = [
    [0.00, "#2b6cb0"],
    [0.15, "#5b9bd5"],
    [0.30, "#9ecae1"],
    [0.45, "#dceef7"],
    [0.50, "#f8f8f8"],
    [0.55, "#fde0dd"],
    [0.70, "#f4a6a6"],
    [0.85, "#d6604d"],
    [1.00, "#8b1e1e"],
]


# =============================================================================
# HELPERS
# =============================================================================
def find_col(cols, exact, fuzzy):
    for c in cols:
        if c.lower() in exact:
            return c
    for c in cols:
        if any(k in c.lower() for k in fuzzy):
            return c
    raise ValueError(f"Column not found. exact={exact}, fuzzy={fuzzy}")

def linear_trend_slope(years: np.ndarray, values: np.ndarray) -> float:
    if len(years) < 2:
        return 0.0
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return 0.0
    return float(np.sum(x * y) / denom)

def compute_simple_change_point(values: np.ndarray):
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 6:
        return None, 0.0

    best_t = None
    best_score = -1.0
    for t in range(2, n - 2):
        left = x[:t]
        right = x[t:]
        score = abs(left.mean() - right.mean()) * math.sqrt((t * (n - t)) / n)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score

def fig_to_html(fig, filename="plot"):
    return fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": filename,
                "scale": 2,
            },
        },
    )

# =============================================================================
# LOAD DATA
# =============================================================================
@st.cache_data
def load_data(nodes_file: str, edges_file: str):
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)

    nodes.columns = [c.strip().lower() for c in nodes.columns]
    edges.columns = [c.strip().lower() for c in edges.columns]

    if not {"id", "label"}.issubset(nodes.columns):
        raise ValueError("nodes file must contain at least columns: id, label")

    id2label = nodes.set_index("id")["label"].to_dict()

    for col in ["score_type", "round", "category", "source", "target"]:
        if col in edges.columns:
            edges[col] = edges[col].astype(str).str.strip()

    if "score_type" in edges.columns:
        edges["score_type"] = edges["score_type"].str.lower()
        if "total" in edges["score_type"].unique():
            edges = edges[edges["score_type"] == "total"]

    if "round" in edges.columns:
        edges["round"] = edges["round"].str.lower()
        if "final" in edges["round"].unique():
            edges = edges[edges["round"] == "final"]

    if edges.empty:
        raise ValueError("No rows remain after applying score_type/round filters.")

    src_col = find_col(edges.columns, {"source", "from", "from_country"}, ["source", "from", "voter"])
    tgt_col = find_col(edges.columns, {"target", "to", "to_country"}, ["target", "to", "recip"])

    numeric_cols = [c for c in edges.columns if pd.api.types.is_numeric_dtype(edges[c])]
    weight_candidates = [
        c for c in numeric_cols
        if any(k in c.lower() for k in ("weight", "point", "score", "pts", "value"))
        and c.lower() != "year"
    ]
    if not weight_candidates:
        raise ValueError("No numeric points column found.")
    pts_col = weight_candidates[0]

    return nodes, edges, id2label, src_col, tgt_col, pts_col

nodes, edges, id2label, src_col, tgt_col, pts_col = load_data(NODES_FILE, EDGES_FILE)

all_years_available = sorted(edges["year"].unique())
ROOT_START = max(ROOT_START, min(all_years_available))
ROOT_END = min(ROOT_END, max(all_years_available))

# =============================================================================
# COMPUTATION
# =============================================================================
@st.cache_data
def compute_period_data(start_year: int, end_year: int):
    df = edges[(edges["year"] >= start_year) & (edges["year"] <= end_year)].copy()
    if df.empty:
        return None

    years_with_data = sorted(df["year"].unique())

    actual = (
        df.groupby(["year", src_col, tgt_col], as_index=False)[pts_col]
          .sum()
          .rename(columns={src_col: "source", tgt_col: "target", pts_col: "points"})
    )
    actual["points"] = pd.to_numeric(actual["points"], errors="coerce").fillna(0)
    actual = actual[actual["points"] > 0].copy()

    pair_rows = []
    for yr in years_with_data:
        yr_df = df[df["year"] == yr]
        participants = sorted(
            set(yr_df[src_col].dropna().astype(str).unique()) |
            set(yr_df[tgt_col].dropna().astype(str).unique())
        )
        for s in participants:
            for t in participants:
                if s != t:
                    pair_rows.append((yr, s, t))

    eligible = pd.DataFrame(pair_rows, columns=["year", "source", "target"])

    yr_agg = eligible.merge(actual, on=["year", "source", "target"], how="left")
    yr_agg["points"] = yr_agg["points"].fillna(0)
    yr_agg["era_max"] = yr_agg["year"].map(ERA_MAX).fillna(12)
    yr_agg["nvs_year"] = (yr_agg["points"] / yr_agg["era_max"]).clip(lower=0, upper=1)

    raw_total_df = (
        actual.groupby(["source", "target"], as_index=False)["points"]
              .sum()
              .rename(columns={"points": "total_votes"})
    )

    agg = (
        yr_agg.groupby(["source", "target"], as_index=False)
              .agg(
                  nvs_sum=("nvs_year", "sum"),
                  years_eligible=("year", "nunique")
              )
    )

    agg = agg.merge(raw_total_df, on=["source", "target"], how="left")
    agg["total_votes"] = agg["total_votes"].fillna(0)

    agg["nvs_mean"] = agg["nvs_sum"] / agg["years_eligible"]
    agg["nvs_score"] = agg["nvs_mean"] * 12
    agg["raw_avg_per_year"] = agg["total_votes"] / agg["years_eligible"]

    agg["src_label"] = agg["source"].map(id2label).fillna(agg["source"])
    agg["tgt_label"] = agg["target"].map(id2label).fillna(agg["target"])

    return {
        "df": df,
        "years": years_with_data,
        "yr_agg": yr_agg,
        "agg": agg,
    }

def build_matrix(agg, value_col, order):
    m = agg.pivot(index="src_label", columns="tgt_label", values=value_col).fillna(0)
    return m.reindex(index=order, columns=order, fill_value=0)

def get_country_order(period_data, order_mode="strength"):
    agg = period_data["agg"]
    all_country_ids = sorted(
        set(edges[src_col].dropna().astype(str).unique()) |
        set(edges[tgt_col].dropna().astype(str).unique())
    )
    all_country_labels = [id2label.get(c, c) for c in all_country_ids]

    if order_mode == "strength":
        out_strength = agg.groupby("src_label")["nvs_score"].sum()
        in_strength = agg.groupby("tgt_label")["nvs_score"].sum()
        strength = out_strength.add(in_strength, fill_value=0).to_dict()
        for c in all_country_labels:
            strength.setdefault(c, 0.0)
        return sorted(all_country_labels, key=lambda x: strength.get(x, 0.0), reverse=True)

    if order_mode == "cluster" and SCIPY_OK:
        m = build_matrix(agg, "nvs_score", all_country_labels)
        X = m.values.astype(float)
        if len(all_country_labels) > 2:
            dist = pdist(X, metric="euclidean")
            Z = linkage(dist, method="average")
            Z = optimal_leaf_ordering(Z, dist)
            order_idx = leaves_list(Z)
            return [all_country_labels[i] for i in order_idx]

    return sorted(all_country_labels)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("Controls")

    start_year = st.selectbox("Start year", list(range(ROOT_START, ROOT_END + 1)), index=0)
    end_year = st.selectbox("End year", list(range(ROOT_START, ROOT_END + 1)), index=len(list(range(ROOT_START, ROOT_END + 1))) - 1)

    if start_year > end_year:
        st.error("Start year must be <= end year.")
        st.stop()

    top_n_choice = st.selectbox("TOP N countries", ["All", 10, 15, 20, 25, 30, 40, 50], index=0)
    TOP_N = None if top_n_choice == "All" else int(top_n_choice)

    order_mode = st.selectbox("Country order", ["strength", "cluster", "alphabetical"], index=0)
    view_mode = st.selectbox("View", ["both", "nvs", "raw", "correlation"], index=0)

    threshold = st.slider(
        "Hide weak NVS cells below",
        min_value=0.0,
        max_value=6.0,
        value=0.0,
        step=0.1
    )

# =============================================================================
# COMPUTE SELECTED PERIOD
# =============================================================================
pdata = compute_period_data(start_year, end_year)
if pdata is None:
    st.error("No data for the selected range.")
    st.stop()

year_label = f"{start_year}" if start_year == end_year else f"{start_year}–{end_year}"

if order_mode == "alphabetical":
    order = sorted(get_country_order(pdata, order_mode="strength"))
else:
    order = get_country_order(pdata, order_mode=order_mode)

if TOP_N is not None:
    order = order[:TOP_N]

agg = pdata["agg"]

m_nvs = build_matrix(agg, "nvs_score", order)
m_total = build_matrix(agg, "total_votes", order)
m_years = build_matrix(agg, "years_eligible", order)
m_avg = build_matrix(agg, "raw_avg_per_year", order)

z_nvs = m_nvs.values.astype(float)
z_total = m_total.values.astype(float)
z_years = m_years.values.astype(float)
z_avg = m_avg.values.astype(float)

# True hiding using NaN
z_nvs_plot = np.where(z_nvs >= threshold, z_nvs, np.nan) if threshold > 0 else z_nvs.copy()
z_total_plot = np.where(z_nvs >= threshold, z_total, np.nan) if threshold > 0 else z_total.copy()

m_corr = m_nvs.T.corr(method="pearson").fillna(0)
m_corr = m_corr.reindex(index=order, columns=order, fill_value=0)
z_corr = m_corr.values.astype(float)

# =============================================================================
# HOVERS
# =============================================================================
def make_hover_nvs():
    n = len(order)
    hover = []
    for r in range(n):
        row = []
        for c in range(n):
            if z_years[r, c] <= 0:
                row.append("")
            elif threshold > 0 and z_nvs[r, c] < threshold:
                row.append("")
            else:
                row.append(
                    f"<b>{order[r]}</b> → <b>{order[c]}</b><br>"
                    f"Total votes: <b>{int(z_total[r,c])}</b><br>"
                    f"Eligible years: <b>{int(z_years[r,c])}</b><br>"
                    f"NVS score: <b>{z_nvs[r,c]:.2f}</b> / 12<br>"
                    f"Raw avg/year: <b>{z_avg[r,c]:.2f}</b>"
                )
        hover.append(row)
    return hover

def make_hover_corr():
    row_strength = m_nvs.sum(axis=1).to_dict()
    n = len(order)
    return [[
        (
            f"<b>{order[r]}</b> ↔ <b>{order[c]}</b><br>"
            f"Correlation: <b>{z_corr[r,c]:.3f}</b><br>"
            f"{order[r]} profile strength: <b>{row_strength.get(order[r], 0):.2f}</b><br>"
            f"{order[c]} profile strength: <b>{row_strength.get(order[c], 0):.2f}</b>"
        )
        for c in range(n)] for r in range(n)]

def make_text(z, fmt):
    n = z.shape[0]
    return [[fmt.format(z[r, c]) if pd.notna(z[r, c]) else "" for c in range(n)] for r in range(n)]

hover_nvs = make_hover_nvs()
hover_corr = make_hover_corr()

# =============================================================================
# FIGURES
# =============================================================================
fig = go.Figure()

if view_mode in ("both", "nvs"):
    fig.add_trace(go.Heatmap(
        z=z_nvs_plot,
        x=order, y=order,
        text=make_text(z_nvs_plot, "{:.1f}"),
        texttemplate="%{text}",
        textfont=dict(size=7, color="#f0faf4"),
        hovertext=hover_nvs,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale=NVS_SCALE,
        zmin=0, zmax=12,
        xgap=2, ygap=2,
        visible=(view_mode != "both" or True),
        name="NVS Matrix",
        colorbar=dict(
            title=dict(text="NVS Score<br>(0–12)", font_size=11),
            thickness=14, len=0.5
        ),
    ))

if view_mode in ("both", "raw"):
    fig.add_trace(go.Heatmap(
        z=z_total_plot,
        x=order, y=order,
        text=make_text(z_total_plot, "{:.0f}"),
        texttemplate="%{text}",
        textfont=dict(size=7, color="#f0faf4"),
        hovertext=hover_nvs,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale=NVS_SCALE,
        zmin=0,
        xgap=2, ygap=2,
        visible=(view_mode == "raw"),
        name="Raw Total",
        colorbar=dict(
            title=dict(text="Total Votes", font_size=11),
            thickness=14, len=0.5
        ),
    ))

if view_mode in ("both", "correlation"):
    fig.add_trace(go.Heatmap(
        z=z_corr,
        x=order, y=order,
        text=make_text(z_corr, "{:.2f}"),
        texttemplate="%{text}",
        textfont=dict(size=7, color="#111111"),
        hovertext=hover_corr,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale=CORR_SCALE,
        zmid=0,
        zmin=-1, zmax=1,
        xgap=1, ygap=1,
        visible=(view_mode == "correlation"),
        name="Correlation",
        colorbar=dict(
            title=dict(text="Correlation", font_size=11),
            thickness=14, len=0.5
        ),
    ))

if view_mode == "both":
    labels = [tr.name for tr in fig.data]
    buttons = []
    for label in labels:
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[{"visible": [name == label for name in labels]}]
            )
        )

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5, xanchor="center",
            y=1.08, yanchor="top",
            buttons=buttons
        )]
    )

fig.update_layout(
    title=dict(
        text=(
            f"Eurovision Voting Explorer · {year_label}"
            f"<br><span style='font-size:12px; color:#4b5563;'>"
            f"Dynamic range selection | Order: {order_mode} | Countries shown: {len(order)}"
            f"</span>"
        ),
        x=0.5, xanchor="center"
    ),
    xaxis=dict(
    title=dict(
            text="Receives points →",
            font=dict(size=14, color="#1f2937")
        ),
        type="category",
        categoryorder="array",
        categoryarray=order,
        tickmode="array",
        tickvals=order,
        ticktext=order,
        tickangle=-45,
        tickfont=dict(size=8),
        side="bottom"
    ),
    yaxis=dict(
        title=dict(
            text="Source country",
            font=dict(size=14, color="#1f2937")
        ),
        type="category",
        categoryorder="array",
        categoryarray=order,
        tickmode="array",
        tickvals=order,
        ticktext=order,
        autorange="reversed",
        tickfont=dict(size=8)
    ),
    height=max(900, len(order) * 22 + 250),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=170, r=120, t=130, b=170),
)

st.plotly_chart(fig, use_container_width=True)

html_bytes = fig_to_html(fig, filename=f"eurovision_{view_mode}_{year_label}").encode("utf-8")
st.download_button(
    "Download current chart as HTML",
    data=html_bytes,
    file_name=f"eurovision_{view_mode}_{year_label.replace('–','-')}.html",
    mime="text/html"
)

# =============================================================================
# PAIR ANALYSIS
# =============================================================================
st.subheader("Pair analysis")

pair_col1, pair_col2 = st.columns(2)
with pair_col1:
    source_country = st.selectbox("Source country", order, index=0)
with pair_col2:
    target_country = st.selectbox("Target country", order, index=min(1, len(order) - 1))

pair_df = pdata["yr_agg"].copy()
pair_df["src_label"] = pair_df["source"].map(id2label).fillna(pair_df["source"])
pair_df["tgt_label"] = pair_df["target"].map(id2label).fillna(pair_df["target"])

pair_rows = pair_df[
    (pair_df["src_label"] == source_country) &
    (pair_df["tgt_label"] == target_country)
].copy()

if pair_rows.empty:
    st.warning("No eligible pair-years found for this pair.")
else:
    pair_rows = pair_rows.sort_values("year")
    years = pair_rows["year"].to_numpy()
    values = pair_rows["nvs_year"].to_numpy() * 12

    mean_v = float(np.mean(values))
    std_v = float(np.std(values))
    cv_v = float(std_v / (mean_v + 1e-6))
    slope_v = linear_trend_slope(years, values)
    stability = float(max(0.0, 1.0 - cv_v))
    cp_idx, cp_score = compute_simple_change_point(values)
    cp_year = int(years[cp_idx]) if cp_idx is not None else None

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean NVS", f"{mean_v:.2f}")
    m2.metric("Std dev", f"{std_v:.2f}")
    m3.metric("CV", f"{cv_v:.2f}")
    m4.metric("Trend slope", f"{slope_v:.3f}")
    m5.metric("Stability", f"{stability:.2f}")

    if cp_year is not None:
        st.info(f"Most likely change point around {cp_year} (heuristic score {cp_score:.2f})")

    roll = pd.Series(values).rolling(window=5, min_periods=1).mean().to_numpy()

    fig_pair = go.Figure()
    fig_pair.add_trace(go.Scatter(
        x=years, y=values,
        mode="lines+markers",
        name="Yearly NVS",
        hovertemplate="Year %{x}<br>NVS %{y:.2f}<extra></extra>"
    ))
    fig_pair.add_trace(go.Scatter(
        x=years, y=roll,
        mode="lines",
        name="5-year rolling mean",
        hovertemplate="Year %{x}<br>Rolling %{y:.2f}<extra></extra>"
    ))

    if cp_year is not None:
        fig_pair.add_vline(x=cp_year, line_width=2, line_dash="dash", line_color="red")

    fig_pair.update_layout(
        title=f"Pair trend: {source_country} → {target_country}",
        xaxis_title="Year",
        yaxis_title="NVS (0–12)",
        height=500,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig_pair, use_container_width=True)

# =============================================================================
# TOP PAIRS
# =============================================================================
st.subheader("Top directed pairs in selected period")

top_pairs = (
    pdata["agg"]
    .sort_values("nvs_score", ascending=False)
    [["src_label", "tgt_label", "nvs_score", "total_votes", "years_eligible", "raw_avg_per_year"]]
    .head(25)
    .rename(columns={
        "src_label": "Voter",
        "tgt_label": "Recipient",
        "nvs_score": "NVS (0–12)",
        "total_votes": "Total votes",
        "years_eligible": "Eligible years",
        "raw_avg_per_year": "Avg/year"
    })
)

st.dataframe(top_pairs, use_container_width=True)