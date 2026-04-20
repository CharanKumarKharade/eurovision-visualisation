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
ROOT_END   = 2025

ERA_MAX = {y: 12 for y in range(1975, 2016)}
ERA_MAX.update({y: 24 for y in range(2016, 2026)})

# Better thesis-friendly blue scale
NVS_SCALE = [
    [0.00, "#f8fbff"], [0.08, "#edf5fc"], [0.18, "#dbe9f6"],
    [0.32, "#bfd7ee"], [0.48, "#93c3e5"], [0.64, "#5ea7d8"],
    [0.78, "#2f7fbe"], [0.90, "#165a9c"], [1.00, "#0b3c6f"],
]

CORR_SCALE = [
    [0.00, "#2b6cb0"], [0.15, "#5b9bd5"], [0.30, "#9ecae1"],
    [0.45, "#dceef7"], [0.50, "#f8f8f8"], [0.55, "#fde0dd"],
    [0.70, "#f4a6a6"], [0.85, "#d6604d"], [1.00, "#8b1e1e"],
]

# =============================================================================
# HELPERS
# =============================================================================
def find_col(cols, exact, fuzzy):
    for c in cols:
        if c.lower() in exact: return c
    for c in cols:
        if any(k in c.lower() for k in fuzzy): return c
    raise ValueError(f"Column not found. exact={exact}, fuzzy={fuzzy}")

def linear_trend_slope(years: np.ndarray, values: np.ndarray) -> float:
    if len(years) < 2: return 0.0
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    x = x - x.mean()
    denom = np.sum(x * x)
    return 0.0 if denom == 0 else float(np.sum(x * y) / denom)

def compute_simple_change_point(values: np.ndarray):
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 6: return None, 0.0
    best_t, best_score = None, -1.0
    for t in range(2, n - 2):
        score = abs(x[:t].mean() - x[t:].mean()) * math.sqrt((t * (n - t)) / n)
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score

def fig_to_html(fig, filename="plot"):
    return fig.to_html(
        full_html=True, include_plotlyjs="cdn",
        config={
            "scrollZoom": True, "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 2},
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

    src_col = find_col(edges.columns, {"source","from","from_country"}, ["source","from","voter"])
    tgt_col = find_col(edges.columns, {"target","to","to_country"}, ["target","to","recip"])

    numeric_cols = [c for c in edges.columns if pd.api.types.is_numeric_dtype(edges[c])]
    weight_candidates = [
        c for c in numeric_cols
        if any(k in c.lower() for k in ("weight","point","score","pts","value"))
        and c.lower() != "year"
    ]
    if not weight_candidates:
        raise ValueError("No numeric points column found.")
    pts_col = weight_candidates[0]

    return nodes, edges, id2label, src_col, tgt_col, pts_col

nodes, edges, id2label, src_col, tgt_col, pts_col = load_data(NODES_FILE, EDGES_FILE)

all_years_available = sorted(edges["year"].unique())
ROOT_START = max(ROOT_START, min(all_years_available))
ROOT_END   = min(ROOT_END,   max(all_years_available))

# =============================================================================
# COUNTRY PARTICIPATION COUNTS (across all available data)
# =============================================================================
@st.cache_data
def compute_participation_counts():
    """
    For each country, count how many distinct years they appeared
    as either a source (voter) or target (recipient) in the full dataset.
    This is used for the min-participation filter.
    """
    src_years = edges.groupby(src_col)["year"].nunique().rename("years")
    tgt_years = edges.groupby(tgt_col)["year"].nunique().rename("years")
    combined  = pd.concat([src_years, tgt_years]).groupby(level=0).max()
    return combined.to_dict()   # {country_id: n_years}

participation_counts = compute_participation_counts()

def participation_years_for_label(label: str) -> int:
    """Return participation year count for a display label."""
    for cid, lbl in id2label.items():
        if lbl == label:
            return participation_counts.get(cid, 0)
    return participation_counts.get(label, 0)

# =============================================================================
# COMPUTATION
# =============================================================================
@st.cache_data
def compute_period_data(start_year: int, end_year: int):
    df = edges[(edges["year"] >= start_year) & (edges["year"] <= end_year)].copy()
    if df.empty: return None

    years_with_data = sorted(df["year"].unique())

    actual = (
        df.groupby(["year", src_col, tgt_col], as_index=False)[pts_col]
          .sum()
          .rename(columns={src_col:"source", tgt_col:"target", pts_col:"points"})
    )
    actual["points"] = pd.to_numeric(actual["points"], errors="coerce").fillna(0)
    actual = actual[actual["points"] > 0].copy()

    # ── TRACK ACTUAL PARTICIPANTS PER YEAR ────────────────────────────────────
    # A country "participated" in a year if it appears as source OR target
    participants_by_year = {}
    for yr in years_with_data:
        yr_df = df[df["year"] == yr]
        participants_by_year[yr] = set(
            yr_df[src_col].dropna().astype(str).unique()
        ) | set(yr_df[tgt_col].dropna().astype(str).unique())

    pair_rows = []
    for yr in years_with_data:
        participants = sorted(participants_by_year[yr])
        for s in participants:
            for t in participants:
                if s != t:
                    pair_rows.append((yr, s, t))

    eligible = pd.DataFrame(pair_rows, columns=["year","source","target"])

    yr_agg = eligible.merge(actual, on=["year","source","target"], how="left")
    yr_agg["points"]   = yr_agg["points"].fillna(0)
    yr_agg["era_max"]  = yr_agg["year"].map(ERA_MAX).fillna(12)
    yr_agg["nvs_year"] = (yr_agg["points"] / yr_agg["era_max"]).clip(0, 1)

    # ── FLAG: was the data truly missing or did they just give 0? ─────────────
    # "participated_as_voter" = source country appeared as a voter that year
    voter_years = set(zip(df["year"], df[src_col].astype(str)))
    yr_agg["voter_participated"] = [
        (row.year, row.source) in voter_years
        for row in yr_agg.itertuples()
    ]
    # "voted_for_target" = any vote row exists for this pair this year
    voted_pairs = set(zip(actual["year"], actual["source"], actual["target"]))
    yr_agg["gave_points"] = [
        (row.year, row.source, row.target) in voted_pairs
        for row in yr_agg.itertuples()
    ]
    # Classification per year-pair row:
    # "voted"    → gave_points = True
    # "abstained"→ voter_participated = True but gave 0 (chose not to give points)
    # "absent"   → voter_participated = False (country wasn't in contest that year)
    def classify(row):
        if row.gave_points:      return "voted"
        if row.voter_participated: return "abstained"
        return "absent"
    yr_agg["status"] = [classify(r) for r in yr_agg.itertuples()]

    raw_total_df = (
        actual.groupby(["source","target"], as_index=False)["points"]
              .sum().rename(columns={"points":"total_votes"})
    )
    agg = (
        yr_agg.groupby(["source","target"], as_index=False)
              .agg(nvs_sum=("nvs_year","sum"), years_eligible=("year","nunique"))
    )
    agg = agg.merge(raw_total_df, on=["source","target"], how="left")
    agg["total_votes"]      = agg["total_votes"].fillna(0)
    agg["nvs_mean"]         = agg["nvs_sum"] / agg["years_eligible"]
    agg["nvs_score"]        = agg["nvs_mean"] * 12
    agg["raw_avg_per_year"] = agg["total_votes"] / agg["years_eligible"]
    agg["src_label"]        = agg["source"].map(id2label).fillna(agg["source"])
    agg["tgt_label"]        = agg["target"].map(id2label).fillna(agg["target"])

    return {
        "df": df,
        "years": years_with_data,
        "yr_agg": yr_agg,
        "agg": agg,
        "participants_by_year": participants_by_year,
    }

def build_matrix(agg, value_col, order):
    m = agg.pivot(index="src_label", columns="tgt_label", values=value_col).fillna(0)
    return m.reindex(index=order, columns=order, fill_value=0)

def get_country_order(period_data, all_labels, order_mode="strength"):
    agg = period_data["agg"]
    if order_mode == "strength":
        out_s = agg.groupby("src_label")["nvs_score"].sum()
        in_s  = agg.groupby("tgt_label")["nvs_score"].sum()
        strength = out_s.add(in_s, fill_value=0).to_dict()
        for c in all_labels: strength.setdefault(c, 0.0)
        return sorted(all_labels, key=lambda x: strength.get(x, 0.0), reverse=True)

    if order_mode == "cluster" and SCIPY_OK:
        m = build_matrix(agg, "nvs_score", all_labels)
        X = m.values.astype(float)
        if len(all_labels) > 2:
            dist = pdist(X, metric="euclidean")
            Z    = linkage(dist, method="average")
            Z    = optimal_leaf_ordering(Z, dist)
            return [all_labels[i] for i in leaves_list(Z)]

    return sorted(all_labels)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("Controls")

    start_year = st.selectbox(
        "Start year", list(range(ROOT_START, ROOT_END + 1)), index=0)
    end_year = st.selectbox(
        "End year",   list(range(ROOT_START, ROOT_END + 1)),
        index=len(list(range(ROOT_START, ROOT_END + 1))) - 1)

    if start_year > end_year:
        st.error("Start year must be <= end year.")
        st.stop()

    st.markdown("---")

    # ── NEW: Minimum participation filter ─────────────────────────────────────
    st.markdown("**Country participation filter**")
    min_participation = st.slider(
        "Min years a country participated (full dataset)",
        min_value=1,
        max_value=int(max(participation_counts.values(), default=1)),
        value=5,
        step=1,
        help=(
            "Excludes countries that appeared in fewer than N years across the "
            "entire dataset. Set to 1 to include all."
        )
    )

    # Show how many countries pass the filter
    all_country_ids = sorted(
        set(edges[src_col].dropna().astype(str).unique()) |
        set(edges[tgt_col].dropna().astype(str).unique())
    )
    all_labels_unfiltered = [id2label.get(c, c) for c in all_country_ids]

    passing = [
        lbl for lbl in all_labels_unfiltered
        if participation_years_for_label(lbl) >= min_participation
    ]
    st.caption(f"{len(passing)} / {len(all_labels_unfiltered)} countries pass this filter")

    st.markdown("---")

    top_n_choice = st.selectbox(
        "TOP N countries (by NVS strength)",
        ["All", 10, 15, 20, 25, 30, 40, 50], index=0)
    TOP_N = None if top_n_choice == "All" else int(top_n_choice)

    order_mode = st.selectbox(
        "Country order", ["strength","cluster","alphabetical"], index=0)
    view_mode  = st.selectbox(
        "View", ["both","nvs","raw","correlation"], index=0)

    threshold = st.slider(
        "Hide weak NVS cells below",
        min_value=0.0, max_value=6.0, value=0.0, step=0.1)

# =============================================================================
# COMPUTE
# =============================================================================
pdata = compute_period_data(start_year, end_year)
if pdata is None:
    st.error("No data for the selected range.")
    st.stop()

year_label = str(start_year) if start_year == end_year else f"{start_year}–{end_year}"

# Apply min-participation filter to label list
filtered_labels = [
    lbl for lbl in all_labels_unfiltered
    if participation_years_for_label(lbl) >= min_participation
]

order = get_country_order(pdata, filtered_labels, order_mode=order_mode)
if order_mode == "alphabetical":
    order = sorted(filtered_labels)

if TOP_N is not None:
    order = order[:TOP_N]

agg = pdata["agg"]

m_nvs   = build_matrix(agg, "nvs_score",        order)
m_total = build_matrix(agg, "total_votes",       order)
m_years = build_matrix(agg, "years_eligible",    order)
m_avg   = build_matrix(agg, "raw_avg_per_year",  order)

z_nvs   = m_nvs.values.astype(float)
z_total = m_total.values.astype(float)
z_years = m_years.values.astype(float)
z_avg   = m_avg.values.astype(float)

z_nvs_plot   = np.where(z_nvs >= threshold, z_nvs,   np.nan) if threshold > 0 else z_nvs.copy()
z_total_plot = np.where(z_nvs >= threshold, z_total, np.nan) if threshold > 0 else z_total.copy()

m_corr = m_nvs.T.corr(method="pearson").fillna(0).reindex(index=order, columns=order, fill_value=0)
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
            if z_years[r,c] <= 0 or (threshold > 0 and z_nvs[r,c] < threshold):
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
        (f"<b>{order[r]}</b> ↔ <b>{order[c]}</b><br>"
         f"Correlation: <b>{z_corr[r,c]:.3f}</b><br>"
         f"{order[r]} strength: <b>{row_strength.get(order[r],0):.2f}</b><br>"
         f"{order[c]} strength: <b>{row_strength.get(order[c],0):.2f}</b>")
        for c in range(n)] for r in range(n)]

def make_text(z, fmt):
    n = z.shape[0]
    return [[fmt.format(z[r,c]) if pd.notna(z[r,c]) else "" for c in range(n)] for r in range(n)]

hover_nvs  = make_hover_nvs()
hover_corr = make_hover_corr()

# =============================================================================
# MAIN FIGURE
# =============================================================================
fig = go.Figure()

if view_mode in ("both","nvs"):
    fig.add_trace(go.Heatmap(
        z=z_nvs_plot, x=order, y=order,
        text=make_text(z_nvs_plot, "{:.1f}"), texttemplate="%{text}",
        textfont=dict(size=7, color="#f0faf4"),
        hovertext=hover_nvs, hovertemplate="%{hovertext}<extra></extra>",
        colorscale=NVS_SCALE, zmin=0, zmax=12, xgap=2, ygap=2,
        visible=True, name="NVS Matrix",
        colorbar=dict(title=dict(text="NVS Score<br>(0–12)", font_size=11), thickness=14, len=0.5),
    ))

if view_mode in ("both","raw"):
    fig.add_trace(go.Heatmap(
        z=z_total_plot, x=order, y=order,
        text=make_text(z_total_plot, "{:.0f}"), texttemplate="%{text}",
        textfont=dict(size=7, color="#f0faf4"),
        hovertext=hover_nvs, hovertemplate="%{hovertext}<extra></extra>",
        colorscale=NVS_SCALE, zmin=0, xgap=2, ygap=2,
        visible=(view_mode == "raw"), name="Raw Total",
        colorbar=dict(title=dict(text="Total Votes", font_size=11), thickness=14, len=0.5),
    ))

if view_mode in ("both","correlation"):
    fig.add_trace(go.Heatmap(
        z=z_corr, x=order, y=order,
        text=make_text(z_corr, "{:.2f}"), texttemplate="%{text}",
        textfont=dict(size=7, color="#111111"),
        hovertext=hover_corr, hovertemplate="%{hovertext}<extra></extra>",
        colorscale=CORR_SCALE, zmid=0, zmin=-1, zmax=1, xgap=1, ygap=1,
        visible=(view_mode == "correlation"), name="Correlation",
        colorbar=dict(title=dict(text="Correlation", font_size=11), thickness=14, len=0.5),
    ))

if view_mode == "both":
    labels = [tr.name for tr in fig.data]
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="right",
        x=0.5, xanchor="center", y=1.08, yanchor="top",
        buttons=[dict(label=l, method="update",
                      args=[{"visible":[n==l for n in labels]}])
                 for l in labels]
    )])

axis_common = dict(
    type="category", tickangle=-45, tickfont=dict(size=8),
    showgrid=False
)
fig.update_layout(
    title=dict(
        text=(
            f"Eurovision Voting Explorer · {year_label}"
            f"<br><span style='font-size:12px; color:#4b5563;'>"
            f"Order: {order_mode} | Min participation: {min_participation}+ yrs | "
            f"Countries: {len(order)}"
            f"</span>"
        ),
        x=0.5, xanchor="center"
    ),
    xaxis=dict(title="Receives points →",
               categoryorder="array", categoryarray=order,
               tickvals=order, ticktext=order, side="bottom", **axis_common),
    yaxis=dict(title="Gives points ←",
               categoryorder="array", categoryarray=order,
               tickvals=order, ticktext=order, autorange="reversed", **axis_common),
    height=max(900, len(order) * 22 + 250),
    paper_bgcolor="white", plot_bgcolor="white",
    margin=dict(l=170, r=120, t=130, b=170),
)

st.plotly_chart(fig, use_container_width=True)

html_bytes = fig_to_html(fig, f"eurovision_{view_mode}_{year_label}").encode("utf-8")
st.download_button(
    "Download current chart as HTML",
    data=html_bytes,
    file_name=f"eurovision_{view_mode}_{year_label.replace('–','-')}.html",
    mime="text/html"
)

# =============================================================================
# PAIR ANALYSIS  — with voted / abstained / absent distinction
# =============================================================================
st.subheader("Pair analysis")

pair_col1, pair_col2 = st.columns(2)
with pair_col1:
    source_country = st.selectbox("Source country", order, index=0)
with pair_col2:
    target_country = st.selectbox(
        "Target country", order, index=min(1, len(order) - 1))

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

    # ── STATUS SUMMARY ────────────────────────────────────────────────────────
    status_counts = pair_rows["status"].value_counts()
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Years gave points",   status_counts.get("voted",     0))
    sc2.metric("Years gave 0 (abstained)", status_counts.get("abstained", 0),
               help="Both countries were in the contest but the voter gave no points to this recipient")
    sc3.metric("Years not in contest",    status_counts.get("absent",    0),
               help="The voter country was not present in the contest that year — data is N/A, not 0")

    st.caption(
        "A cell showing 0 NVS can mean two different things: "
        "the country **chose not to give points** (abstained) or "
        "they **were not in the contest** (absent = N/A). "
        "The chart below uses different markers to distinguish these."
    )

    years_all  = pair_rows["year"].to_numpy()
    nvs_all    = pair_rows["nvs_year"].to_numpy() * 12
    status_all = pair_rows["status"].to_numpy()

    # Split into series
    voted_mask     = status_all == "voted"
    abstained_mask = status_all == "abstained"
    absent_mask    = status_all == "absent"

    # Rolling mean only over years both were in contest (voted + abstained)
    present_mask = voted_mask | abstained_mask
    years_present = years_all[present_mask]
    nvs_present   = nvs_all[present_mask]
    roll = (pd.Series(nvs_present, index=years_present)
              .rolling(window=5, min_periods=1).mean())

    # Stats on present years only
    if nvs_present.size > 0:
        mean_v  = float(np.mean(nvs_present))
        std_v   = float(np.std(nvs_present))
        cv_v    = float(std_v / (mean_v + 1e-6))
        slope_v = linear_trend_slope(years_present, nvs_present)
        stability = float(max(0.0, 1.0 - cv_v))
        cp_idx, cp_score = compute_simple_change_point(nvs_present)
        cp_year = int(years_present[cp_idx]) if cp_idx is not None else None
    else:
        mean_v = std_v = cv_v = slope_v = stability = 0.0
        cp_year = None

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean NVS",     f"{mean_v:.2f}",  help="Computed only on years both countries were present")
    m2.metric("Std dev",      f"{std_v:.2f}")
    m3.metric("CV",           f"{cv_v:.2f}")
    m4.metric("Trend slope",  f"{slope_v:.3f}", help="+ve = improving affinity over time")
    m5.metric("Stability",    f"{stability:.2f}")

    if cp_year:
        st.info(
            f"Possible relationship change around **{cp_year}** "
            f"(heuristic score {cp_score:.2f}). "
            "Check if a geopolitical or contest-rule event occurred near this year."
        )

    fig_pair = go.Figure()

    # Voted years — solid circles
    fig_pair.add_trace(go.Scatter(
        x=years_all[voted_mask], y=nvs_all[voted_mask],
        mode="markers", name="Gave points",
        marker=dict(size=9, color="#2f7fbe", symbol="circle"),
        hovertemplate="Year %{x}<br>NVS %{y:.2f}<br>Status: gave points<extra></extra>",
    ))

    # Abstained years — hollow circles at 0
    fig_pair.add_trace(go.Scatter(
        x=years_all[abstained_mask], y=nvs_all[abstained_mask],
        mode="markers", name="Abstained (0 pts, both present)",
        marker=dict(size=9, color="#d6604d", symbol="circle-open", line=dict(width=2)),
        hovertemplate="Year %{x}<br>NVS 0 — both in contest, no points given<extra></extra>",
    ))

    # Absent years — X marker with N/A label
    fig_pair.add_trace(go.Scatter(
        x=years_all[absent_mask], y=[None] * absent_mask.sum(),
        mode="markers", name="Absent (N/A — not in contest)",
        marker=dict(size=10, color="#888780", symbol="x", line=dict(width=2)),
        hovertemplate="Year %{x}<br>N/A — voter not in contest this year<extra></extra>",
    ))

    # 5-year rolling mean (present years only)
    fig_pair.add_trace(go.Scatter(
        x=roll.index, y=roll.values,
        mode="lines", name="5-yr rolling mean (present only)",
        line=dict(color="#0b3c6f", width=2.5, dash="solid"),
        hovertemplate="Year %{x}<br>Rolling NVS %{y:.2f}<extra></extra>",
    ))

    # Mean line
    if nvs_present.size > 0:
        fig_pair.add_hline(
            y=mean_v, line_dash="dot", line_color="#5ea7d8",
            annotation_text=f"mean {mean_v:.2f}",
            annotation_position="top right"
        )

    # Change point
    if cp_year:
        fig_pair.add_vline(
            x=cp_year, line_width=2, line_dash="dash", line_color="red",
            annotation_text=f"change ~{cp_year}",
            annotation_position="top left"
        )

    fig_pair.update_layout(
        title=f"Pair trend: {source_country} → {target_country}",
        xaxis_title="Year",
        yaxis_title="NVS (0–12)",
        yaxis=dict(range=[-0.5, 13]),
        legend=dict(orientation="h", y=-0.2),
        height=520,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    st.plotly_chart(fig_pair, use_container_width=True)

# =============================================================================
# TOP PAIRS TABLE
# =============================================================================
st.subheader("Top directed pairs in selected period")

top_pairs = (
    pdata["agg"]
    .sort_values("nvs_score", ascending=False)
    [["src_label","tgt_label","nvs_score","total_votes","years_eligible","raw_avg_per_year"]]
    .head(25)
    .rename(columns={
        "src_label":        "Voter",
        "tgt_label":        "Recipient",
        "nvs_score":        "NVS (0–12)",
        "total_votes":      "Total votes",
        "years_eligible":   "Eligible years",
        "raw_avg_per_year": "Avg/year",
    })
)
st.dataframe(top_pairs, use_container_width=True)
