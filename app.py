"""
Eurovision Voting Explorer

Final-stage analytical dashboard for exploring dynamic Eurovision voting behaviour.

Core features:
1. Directed NVS matrix
2. Raw total matrix
3. Voting-profile correlation matrix
4. Dynamic period comparison
5. Most strengthened / weakened relationships
6. Pair trend analysis
7. Pair behaviour classification
8. Community / bloc detection
9. Minimum participation filter applied consistently across all views
10. Selected year range applied consistently to all dynamic relationship tables
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale, sequential
import streamlit as st

try:
    from matplotlib import colormaps as mpl_colormaps

    def get_matplotlib_cmap(name):
        return mpl_colormaps.get_cmap(name)
except Exception:
    import matplotlib.cm as cm

    def get_matplotlib_cmap(name):
        return cm.get_cmap(name)

try:
    import networkx as nx
    NETWORKX_OK = True
except Exception:
    NETWORKX_OK = False

try:
    from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
    from scipy.spatial.distance import pdist
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import draft_visualizations as draftviz
    DRAFTVIZ_OK = True
except Exception as _draftviz_err:
    DRAFTVIZ_OK = False
    _draftviz_import_error = _draftviz_err


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(page_title="Eurovision Voting Explorer", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #fcfcfd;
        }
        [data-testid="stSidebar"] {
            background-color: #f8fafc;
        }
        h1, h2, h3 {
            color: #111827;
        }
        .stMetric {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Eurovision Voting Explorer")
st.caption(
    "Interactive dashboard for exploring directed, dynamic, and normalised Eurovision voting patterns."
)


# =============================================================================
# CONFIGURATION
# =============================================================================

NODES_FILE = "nodes_with_coordinates.csv"
EDGES_FILE = "eurovision_senior.csv"

ROOT_START = 1975
ROOT_END = 2025

ERA_MAX = {y: 12 for y in range(1975, 2016)}
ERA_MAX.update({y: 24 for y in range(2016, 2026)})

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

DIFF_SCALE = [
    [0.00, "#2166ac"],
    [0.25, "#92c5de"],
    [0.50, "#f7f7f7"],
    [0.75, "#f4a582"],
    [1.00, "#b2182b"],
]


# =============================================================================
# UTILITY FUNCTIONS
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


def classify_relationship(mean_v, std_v, cv_v, slope_v, stability):
    if mean_v >= 6 and stability >= 0.60 and abs(slope_v) < 0.05:
        return "Strong stable alliance"
    if mean_v >= 6 and stability < 0.40:
        return "Strong but volatile"
    if slope_v >= 0.05:
        return "Emerging relationship"
    if slope_v <= -0.05:
        return "Declining relationship"
    if stability < 0.35:
        return "Volatile relationship"
    if mean_v < 2 and stability >= 0.60:
        return "Weak stable relationship"
    return "Moderately stable / mixed"


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


def make_text(z, fmt):
    n = z.shape[0]
    return [
        [
            fmt.format(z[r, c]) if pd.notna(z[r, c]) else ""
            for c in range(n)
        ]
        for r in range(n)
    ]


def filter_agg_by_order(agg: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    return agg[
        agg["src_label"].isin(order)
        & agg["tgt_label"].isin(order)
    ].copy()


def _selected_row_indices(selection_state):
    if selection_state is None:
        return []
    if isinstance(selection_state, dict):
        return list(selection_state.get("selection", {}).get("rows", []))
    selection = getattr(selection_state, "selection", None)
    if selection is None:
        return []
    return list(getattr(selection, "rows", []))


def set_selected_pair(source: str, target: str, origin: str | None = None):
    st.session_state["selected_pair_source"] = source
    st.session_state["selected_pair_target"] = target
    st.session_state["pair_trend_requested"] = True
    if origin:
        st.session_state["selected_pair_origin"] = origin


def update_selected_pair_from_table(selection_state, table_df: pd.DataFrame, source_col: str, target_col: str, origin: str):
    rows = _selected_row_indices(selection_state)
    if not rows:
        return None
    row = table_df.iloc[rows[0]]
    source = str(row[source_col])
    target = str(row[target_col])
    set_selected_pair(source, target, origin)
    return source, target


def build_pair_interval_summary(pair_rows: pd.DataFrame, start_year: int, end_year: int, interval_years: int = 5) -> pd.DataFrame:
    rows = []
    for interval_start in range(start_year, end_year + 1, interval_years):
        interval_end = min(interval_start + interval_years - 1, end_year)
        interval_df = pair_rows[
            (pair_rows["year"] >= interval_start)
            & (pair_rows["year"] <= interval_end)
        ].copy()
        if interval_df.empty:
            continue
        values = (interval_df["nvs_year"].to_numpy(dtype=float) * 12)
        years = interval_df["year"].to_numpy(dtype=int)
        mean_v = float(np.mean(values))
        std_v = float(np.std(values))
        cv_v = float(std_v / (mean_v + 1e-6))
        slope_v = linear_trend_slope(years, values)
        stability = float(max(0.0, 1.0 - cv_v))
        relationship_class = classify_relationship(mean_v, std_v, cv_v, slope_v, stability)
        status_counts = interval_df["status"].value_counts()
        rows.append({
            "Interval": f"{interval_start}–{interval_end}",
            "Years": int(interval_df["year"].nunique()),
            "Mean NVS": mean_v,
            "Std Dev": std_v,
            "Trend Slope": slope_v,
            "Stability": stability,
            "Relationship Class": relationship_class,
            "Years gave points": int(status_counts.get("voted", 0)),
            "Years gave 0": int(status_counts.get("abstained", 0)),
        })
    return pd.DataFrame(rows)


def build_final_standings(actual: pd.DataFrame, participants_by_year: dict) -> pd.DataFrame:
    if actual.empty:
        return pd.DataFrame(columns=["year", "country_id", "country_label", "final_points", "final_place", "is_winner"])
    target_points = (
        actual.groupby(["year", "target"], as_index=False)["points"]
        .sum()
        .rename(columns={"target": "country_id", "points": "final_points"})
    )
    rows = []
    for year, participants in participants_by_year.items():
        country_ids = sorted({str(c) for c in participants})
        if not country_ids:
            continue
        year_targets = target_points[target_points["year"] == year].set_index("country_id")
        year_rows = pd.DataFrame({"country_id": country_ids})
        year_rows["year"] = int(year)
        year_rows["final_points"] = year_rows["country_id"].map(year_targets["final_points"]).fillna(0)
        year_rows["final_place"] = year_rows["final_points"].rank(method="min", ascending=False).astype(int)
        year_rows["is_winner"] = year_rows["final_place"] == 1
        year_rows["country_label"] = year_rows["country_id"].map(id2label).fillna(year_rows["country_id"])
        rows.append(year_rows)
    if not rows:
        return pd.DataFrame(columns=["year", "country_id", "country_label", "final_points", "final_place", "is_winner"])
    standings = pd.concat(rows, ignore_index=True)
    return standings[["year", "country_id", "country_label", "final_points", "final_place", "is_winner"]]


def build_standing_lookup(final_standings: pd.DataFrame) -> dict:
    lookup = {}
    if final_standings.empty:
        return lookup
    for _, row in final_standings.iterrows():
        lookup[(int(row["year"]), str(row["country_label"]))] = {
            "final_points": float(row["final_points"]),
            "final_place": int(row["final_place"]),
            "is_winner": bool(row["is_winner"]),
        }
    return lookup


def build_pair_interval_figure(pair_rows, source_country, target_country, start_year, end_year, standing_lookup=None, interval_years=5):
    fig_pair = go.Figure()
    palette = ["#0b3c6f","#2f7fbe","#5ea7d8","#93c3e5","#165a9c","#2166ac","#4c78a8","#8da0cb","#7b3294","#c51b7d"]
    segment_count = 0
    for interval_start in range(start_year, end_year + 1, interval_years):
        interval_end = min(interval_start + interval_years - 1, end_year)
        interval_df = pair_rows[
            (pair_rows["year"] >= interval_start) & (pair_rows["year"] <= interval_end)
        ].sort_values("year")
        if interval_df.empty:
            continue
        color = palette[segment_count % len(palette)]
        segment_count += 1
        hover_lines = []
        for _, row in interval_df.iterrows():
            target_meta = standing_lookup.get((int(row["year"]), target_country), {}) if standing_lookup else {}
            source_meta = standing_lookup.get((int(row["year"]), source_country), {}) if standing_lookup else {}
            target_place = target_meta.get("final_place")
            source_place = source_meta.get("final_place")
            winner_country = source_country if source_meta.get("is_winner") else target_country if target_meta.get("is_winner") else None
            hover_lines.append(
                f"Year {int(row['year'])}<br>NVS {float(row['nvs_year']) * 12:.2f}<br>"
                f"Status: {'gave points' if row['status'] == 'voted' else '0 points'}<br>"
                f"Source final place: #{int(source_place) if pd.notna(source_place) else 'N/A'}<br>"
                f"Target final place: #{int(target_place) if pd.notna(target_place) else 'N/A'}"
                + (f"<br><b>Winner of the year: {winner_country}</b>" if winner_country else "")
                + "<extra></extra>"
            )
        fig_pair.add_trace(go.Scatter(
            x=interval_df["year"], y=interval_df["nvs_year"] * 12,
            mode="lines+markers", name=f"{interval_start}–{interval_end}",
            line=dict(color=color, width=3), marker=dict(color=color, size=8),
            hovertext=hover_lines, hovertemplate="%{hovertext}",
        ))
    if not pair_rows.empty:
        values = (pair_rows["nvs_year"].to_numpy(dtype=float) * 12)
        years = pair_rows["year"].to_numpy(dtype=int)
        mean_v = float(np.mean(values))
        cp_idx, cp_score = compute_simple_change_point(values)
        cp_year = int(years[cp_idx]) if cp_idx is not None else None
        fig_pair.add_hline(y=mean_v, line_dash="dot", line_color="#5ea7d8",
                           annotation_text=f"mean {mean_v:.2f}", annotation_position="top right")
        if cp_year is not None:
            fig_pair.add_vline(x=cp_year, line_width=2, line_dash="dash", line_color="red",
                               annotation_text=f"change ~{cp_year}", annotation_position="top left")
    fig_pair.update_layout(
        title=f"Pair trend: {source_country} → {target_country}",
        xaxis_title="Year", yaxis_title="NVS (0–12)", yaxis=dict(range=[-0.5, 13]),
        legend=dict(orientation="h", y=-0.2), height=520, paper_bgcolor="white", plot_bgcolor="white"
    )
    return fig_pair


def build_full_pair_figure(pair_rows, source_country, target_country, standing_lookup=None, highlight_abstained_years=None):
    years_all = pair_rows["year"].to_numpy()
    nvs_all = pair_rows["nvs_year"].to_numpy() * 12
    status_all = pair_rows["status"].to_numpy()
    target_places = np.array([
        standing_lookup.get((int(year), target_country), {}).get("final_place")
        for year in years_all
    ], dtype=object) if standing_lookup else np.array([None] * len(years_all), dtype=object)
    source_places = np.array([
        standing_lookup.get((int(year), source_country), {}).get("final_place")
        for year in years_all
    ], dtype=object) if standing_lookup else np.array([None] * len(years_all), dtype=object)
    source_winner_mask = np.array([
        bool(standing_lookup.get((int(year), source_country), {}).get("is_winner"))
        for year in years_all
    ]) if standing_lookup else np.zeros(len(years_all), dtype=bool)
    target_winner_mask = np.array([
        bool(standing_lookup.get((int(year), target_country), {}).get("is_winner"))
        for year in years_all
    ]) if standing_lookup else np.zeros(len(years_all), dtype=bool)
    winner_mask = source_winner_mask | target_winner_mask
    winner_country = np.array([
        source_country if s else target_country if t else None
        for s, t in zip(source_winner_mask, target_winner_mask)
    ], dtype=object)
    voted_mask = status_all == "voted"
    abstained_mask = status_all == "abstained"
    absent_mask = status_all == "absent"
    voted_nonwinner_mask = voted_mask & ~winner_mask
    voted_winner_mask = voted_mask & winner_mask
    abstained_nonwinner_mask = abstained_mask & ~winner_mask
    abstained_winner_mask = abstained_mask & winner_mask

    def _hover_rows(mask, status_label):
        rows = []
        for year, value, src_place, tgt_place, wname in zip(
            years_all[mask], nvs_all[mask], source_places[mask],
            target_places[mask], winner_country[mask],
        ):
            rows.append(
                f"Year {int(year)}<br>NVS {float(value):.2f}<br>Status: {status_label}<br>"
                f"Source final place: #{int(src_place) if pd.notna(src_place) else 'N/A'}<br>"
                f"Target final place: #{int(tgt_place) if pd.notna(tgt_place) else 'N/A'}"
                + (f"<br><b>Winner of the year: {wname}</b>" if wname else "")
                + "<extra></extra>"
            )
        return rows

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_all[voted_nonwinner_mask], y=nvs_all[voted_nonwinner_mask],
        mode="markers", name="Gave points",
        marker=dict(size=8, color="#2f7fbe", symbol="circle"),
        hovertext=_hover_rows(voted_nonwinner_mask, "gave points"), hovertemplate="%{hovertext}"))
    fig.add_trace(go.Scatter(x=years_all[voted_winner_mask], y=nvs_all[voted_winner_mask],
        mode="markers", name="Winner year",
        marker=dict(size=12, color="#f59e0b", symbol="star", line=dict(width=1.5, color="#ffffff")),
        hovertext=_hover_rows(voted_winner_mask, "gave points"), hovertemplate="%{hovertext}"))
    fig.add_trace(go.Scatter(x=years_all[abstained_nonwinner_mask], y=nvs_all[abstained_nonwinner_mask],
        mode="markers", name="0 points, both present",
        marker=dict(size=8, color="#d6604d", symbol="circle-open", line=dict(width=2)),
        hovertext=_hover_rows(abstained_nonwinner_mask, "0 points"), hovertemplate="%{hovertext}"))
    fig.add_trace(go.Scatter(x=years_all[abstained_winner_mask], y=nvs_all[abstained_winner_mask],
        mode="markers", name="Winner year, 0 points",
        marker=dict(size=12, color="#f59e0b", symbol="star-open", line=dict(width=1.5, color="#ffffff")),
        hovertext=_hover_rows(abstained_winner_mask, "0 points"), hovertemplate="%{hovertext}"))
    fig.add_trace(go.Scatter(x=years_all[absent_mask], y=[None] * absent_mask.sum(),
        mode="markers", name="Absent", marker=dict(size=9, color="#888780", symbol="x"),
        hovertemplate="Year %{x}<br>N/A<extra></extra>"))
    if highlight_abstained_years:
        ha = np.isin(years_all, np.array(highlight_abstained_years)) & abstained_mask
        if ha.any():
            fig.add_trace(go.Scatter(x=years_all[ha], y=nvs_all[ha], mode="markers",
                name="Qualified but received 0",
                marker=dict(size=11, color="#ff3333", symbol="diamond"),
                hovertemplate="Year %{x}<br>NVS 0<br>Recipient qualified but received 0 points<extra></extra>"))
    present_mask = voted_mask | abstained_mask
    years_present = years_all[present_mask]
    nvs_present = nvs_all[present_mask]
    if nvs_present.size > 0:
        roll = pd.Series(nvs_present, index=years_present).rolling(window=5, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines",
            name="5-year rolling mean", line=dict(color="#0b3c6f", width=2.5),
            hovertemplate="Year %{x}<br>Rolling NVS %{y:.2f}<extra></extra>"))
        mean_v = float(np.mean(nvs_present))
        fig.add_hline(y=mean_v, line_dash="dot", line_color="#5ea7d8",
                      annotation_text=f"mean {mean_v:.2f}", annotation_position="top right")
        cp_idx, cp_score = compute_simple_change_point(nvs_present)
        cp_year = int(years_present[cp_idx]) if cp_idx is not None else None
        if cp_year is not None:
            fig.add_vline(x=cp_year, line_width=2, line_dash="dash", line_color="red",
                          annotation_text=f"change ~{cp_year}", annotation_position="top left")
    fig.update_layout(
        title=f"Full-year Pair Trend: {source_country} → {target_country}",
        xaxis_title="Year", yaxis_title="NVS (0–12)", yaxis=dict(range=[-0.5, 13]),
        legend=dict(orientation="h", y=-0.2), height=420,
        paper_bgcolor="white", plot_bgcolor="white"
    )
    return fig


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data(nodes_file: str, edges_file: str):
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)
    nodes.columns = [c.strip().lower() for c in nodes.columns]
    edges.columns = [c.strip().lower() for c in edges.columns]
    if not {"id", "label"}.issubset(nodes.columns):
        raise ValueError("nodes_with_coordinates.csv must contain at least: id, label")
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
        raise ValueError("No rows remain after applying filters.")
    src_col = find_col(edges.columns, {"source","from","from_country"}, ["source","from","voter"])
    tgt_col = find_col(edges.columns, {"target","to","to_country"}, ["target","to","recip"])
    numeric_cols = [c for c in edges.columns if pd.api.types.is_numeric_dtype(edges[c])]
    pts_col = None
    for c in numeric_cols:
        if any(k in c.lower() for k in ["point","score","pts","value","weight"]):
            pts_col = c
            break
    if pts_col is None:
        if "weight" in edges.columns and pd.api.types.is_numeric_dtype(edges["weight"]):
            pts_col = "weight"
        else:
            raise ValueError(f"No valid points column found. Available numeric: {numeric_cols}")
    edges = edges.rename(columns={pts_col: "points"})
    pts_col = "points"
    return nodes, edges, id2label, src_col, tgt_col, pts_col


nodes, edges, id2label, src_col, tgt_col, pts_col = load_data(NODES_FILE, EDGES_FILE)

@st.cache_data
def load_raw_edges(edges_file: str) -> pd.DataFrame:
    """
    Load the FULL edges CSV without filtering score_type to 'total'.
    Needed for Draft 15 (Jury vs Public Divergence) which requires
    separate jury and televote rows.  Filters to round==final only.
    """
    raw = pd.read_csv(edges_file)
    raw.columns = [c.strip().lower() for c in raw.columns]
    for col in ["score_type", "round", "category", "source", "target"]:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip().str.lower()
    if "round" in raw.columns and "final" in raw["round"].unique():
        raw = raw[raw["round"] == "final"]
    # Rename points column
    numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
    pts_col_raw = next((c for c in numeric_cols
                        if any(k in c.lower() for k in ["point","score","pts","value","weight"])),
                       None)
    if pts_col_raw and pts_col_raw != "points":
        raw = raw.rename(columns={pts_col_raw: "points"})
    return raw

raw_edges = load_raw_edges(EDGES_FILE)

all_years_available = sorted(edges["year"].unique())
ROOT_START = max(ROOT_START, min(all_years_available))
ROOT_END = min(ROOT_END, max(all_years_available))

# =============================================================================
# BLOC PRECOMPUTATION — runs ONCE per server process, cached forever
# =============================================================================

@st.cache_resource(show_spinner="Precomputing voting blocs… (once per session)")
def _warm_bloc_cache():
    if not DRAFTVIZ_OK:
        return {"skipped": True}
    return draftviz.precompute_blocs(edges.copy(), id2label, min_years=10)

_bloc_precompute_summary = _warm_bloc_cache()

st.info(
    f"ℹ️ **Data scope for this app:** every chart, table, and statistic here "
    f"is computed using Eurovision voting data from **{ROOT_START} to "
    f"{ROOT_END}** only ({ROOT_END - ROOT_START + 1} contest years). "
    "Data from years outside this range is never loaded or used."
)


# =============================================================================
# DRAFT VISUALISATION GALLERY
# =============================================================================

DRAFT_REGISTRY = {
    "— None (show main app) —":                        None,
    "1. Unrequited Love — Voting Asymmetry":            "unrequited_love",
    "2. The Neighbour Effect":                          "neighbour_effect",
    "3. Alliance Lifespan Arcs":                        "lifespan_arcs",
    "4. Rise and Fall — Era Dominance":                 "rise_and_fall",
    "5. Voting Hall of Fame":                           "hall_of_fame",
    "6. Bloc Migration Sankey":                         "bloc_migration_sankey",
    "7. Hierarchical Bloc Structure (Storyboard)":      "hierarchical_bloc_poster",
    "8. Geographic Bloc Migration Poster":              "geo_bloc_migration_poster",
    "9. Circular HEB — Hierarchical Edge Bundling":     "circular_heb_poster",
    "10. Geographic HEB — Nodes on World Map":          "geographic_heb_poster",
    "11. Split-Triangle Matrix (light)":                "split_triangle_matrix",
    "11b. Split-Triangle Matrix (dark)":                "split_triangle_matrix_dark",
    "12. Radial Tidy Tree + HEB":                       "radial_tidy_tree",
    "13. Geographic Story Map — Five Acts":             "story_map",
    "14. Bloc Territory Map — Who Dominated?":          "bloc_territory_map",
    "15. Jury vs Public — Divergence Network":          "jury_public_divergence",
}

# ---------------------------------------------------------------------------
# @st.cache_data wraps every draft builder so that the ENTIRE Plotly figure
# is cached after the first render.  Subsequent loads (page refresh, sidebar
# change, switching back to the same draft) return the cached figure instantly
# without rebuilding.  The cache key includes `_data_fp` — a lightweight
# fingerprint of the source data — so the cache invalidates if the CSV changes.
# ---------------------------------------------------------------------------

def _data_fingerprint() -> str:
    """Cheap fingerprint of the edges DataFrame for cache key use."""
    return f"{len(edges)}_{int(edges['year'].max())}_{int(edges['points'].sum())}"


@st.cache_data(show_spinner=False, ttl=None)
def _build_draft(draft_key: str, _data_fp: str):
    """
    Build and cache a draft figure.  First call computes the figure;
    every subsequent call within the same Streamlit server process returns
    the cached result instantly.

    `_data_fp` is a lightweight fingerprint of the source data so the cache
    invalidates correctly if the CSV is reloaded.  Prefixing with `_` makes
    Streamlit include it in the cache key but skip trying to hash it.
    """
    df = edges.copy()

    if draft_key == "unrequited_love":
        return draftviz.build_unrequited_love(df, id2label, min_years=15, top_n_arcs=40)

    elif draft_key == "neighbour_effect":
        return draftviz.build_neighbour_effect(df, id2label, nodes, min_years=15, n_labels=8)

    elif draft_key == "lifespan_arcs":
        return draftviz.build_lifespan_arcs(df, id2label, min_years=15, threshold=0.35, top_n=35)

    elif draft_key == "rise_and_fall":
        return draftviz.build_rise_and_fall(df, id2label, min_years=10, decade_size=10, top_supporters=5)

    elif draft_key == "hall_of_fame":
        return draftviz.build_hall_of_fame(df, id2label, min_years=15)

    elif draft_key == "bloc_migration_sankey":
        return draftviz.build_bloc_migration_sankey(df, id2label, min_years=25, affinity_q=0.65)

    elif draft_key == "hierarchical_bloc_poster":
        return draftviz.build_hierarchical_bloc_poster(
            df, id2label, nodes, min_years=10, top_k_out=3, min_nvs_strength=2.0)

    elif draft_key == "geo_bloc_migration_poster":
        return draftviz.build_geo_bloc_migration_poster(
            df, id2label, nodes,
            min_years=15, min_years_per_half=5,
            disparity_alpha=0.05, min_nvs_strength=1.5)

    elif draft_key == "circular_heb_poster":
        return draftviz.build_circular_heb_poster(
            df, id2label, nodes,
            min_years=10, top_k_out=3, min_nvs_strength=2.0, beta=0.80)

    elif draft_key == "geographic_heb_poster":
        return draftviz.build_geographic_heb_poster(
            df, id2label, nodes,
            min_years=10, top_k_out=3, min_nvs_strength=2.0, beta=0.80)

    elif draft_key == "split_triangle_matrix":
        return draftviz.build_split_triangle_matrix(
            df, id2label, nodes, min_years_overall=10, dark_theme=False)

    elif draft_key == "split_triangle_matrix_dark":
        return draftviz.build_split_triangle_matrix(
            df, id2label, nodes, min_years_overall=10, dark_theme=True)

    elif draft_key == "radial_tidy_tree":
        return draftviz.build_radial_tidy_tree(
            df, id2label, nodes, min_years=10)
    elif draft_key == "story_map":
        return draftviz.build_story_map(
            df, id2label, nodes, min_years=15, top_n=5)
    elif draft_key == "bloc_territory_map":
        return draftviz.build_bloc_territory_map(
            df, id2label, nodes, min_years=10)
    elif draft_key == "jury_public_divergence":
        # raw_edges contains jury + televote rows (not filtered to total)
        return draftviz.build_jury_public_divergence(
            raw_edges, id2label, nodes, min_years=3, top_n=8)

    return None, "Unknown draft", "Unknown draft key."


if "active_draft" not in st.session_state:
    st.session_state["active_draft"] = None

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎨 Draft visualisations (GD Contest 2026)")
    st.caption(
        "Twelve standalone poster-draft diagrams built on 1975–2025 NVS data. "
        "Figures are cached after first load — switching back to a previously "
        "viewed draft is instant."
    )
    draft_choice_label = st.selectbox(
        "View a draft visualisation",
        list(DRAFT_REGISTRY.keys()),
        index=0,
        key="draft_selectbox",
    )
    draft_choice = DRAFT_REGISTRY[draft_choice_label]
    if draft_choice is not None:
        st.session_state["active_draft"] = draft_choice
    elif st.session_state.get("active_draft") is not None and draft_choice is None:
        st.session_state["active_draft"] = None


def _render_draft_header(title: str, explanation_md: str):
    st.title(f"📊 Draft: {title}")
    st.caption(
        f"Data scope: Eurovision {ROOT_START}–{ROOT_END} only. "
        "Standalone draft for the Graph Drawing Contest 2026 poster."
    )
    st.markdown(explanation_md)
    if st.button("⬅ Back to main app", key=f"back_btn_{title}"):
        st.session_state["active_draft"] = None
        st.rerun()
    st.markdown("---")


if st.session_state.get("active_draft") is not None:
    _draft_key = st.session_state["active_draft"]

    if not DRAFTVIZ_OK:
        st.error(
            "Could not load draft_visualizations.py. "
            "Make sure it is in the same folder as app.py."
        )
        if st.button("⬅ Back to main app"):
            st.session_state["active_draft"] = None
            st.rerun()
        st.stop()

    _fp = _data_fingerprint()
    # Show a spinner only on the first build (cache miss); subsequent loads are instant.
    _already_cached = _build_draft.cache_info().currsize > 0 if hasattr(_build_draft, 'cache_info') else False

    with st.spinner("Building draft… (cached after first load)"):
        fig, title, explanation = _build_draft(_draft_key, _fp)

    _render_draft_header(title, explanation)

    if fig is None:
        st.warning("This draft could not be generated with the current data.")
    else:
        st.plotly_chart(fig, use_container_width=True)

    st.stop()


# =============================================================================
# PARTICIPATION COUNTS
# =============================================================================

@st.cache_data
def compute_participation_counts(start_year: int | None = None, end_year: int | None = None):
    df = edges
    if start_year is not None and end_year is not None:
        df = edges[(edges["year"] >= start_year) & (edges["year"] <= end_year)]
    src_years = df.groupby(src_col)["year"].nunique().rename("years")
    tgt_years = df.groupby(tgt_col)["year"].nunique().rename("years")
    combined = pd.concat([src_years, tgt_years]).groupby(level=0).max()
    return combined.to_dict()


def participation_years_for_label(label: str, counts: dict) -> int:
    for cid, lbl in id2label.items():
        if lbl == label:
            return counts.get(cid, 0)
    return counts.get(label, 0)


# =============================================================================
# CORE PERIOD COMPUTATION
# =============================================================================

@st.cache_data
def compute_period_data(start_year: int, end_year: int):
    df = edges[(edges["year"] >= start_year) & (edges["year"] <= end_year)].copy()
    if df.empty:
        return None
    if "points" not in df.columns:
        raise ValueError("CRITICAL: 'points' column missing after load.")
    years_with_data = sorted(df["year"].unique())
    actual = (
        df.groupby(["year", src_col, tgt_col], as_index=False)["points"]
          .sum()
          .rename(columns={src_col: "source", tgt_col: "target"})
    )
    actual["points"] = pd.to_numeric(actual["points"], errors="coerce").fillna(0)
    actual = actual[actual["points"] > 0].copy()
    participants_by_year = {}
    for yr in years_with_data:
        yr_df = df[df["year"] == yr]
        participants_by_year[yr] = (
            set(yr_df[src_col].astype(str)) | set(yr_df[tgt_col].astype(str))
        )
    pair_rows = [
        (yr, s, t)
        for yr in years_with_data
        for s in participants_by_year[yr]
        for t in participants_by_year[yr]
        if s != t
    ]
    eligible = pd.DataFrame(pair_rows, columns=["year", "source", "target"])
    yr_agg = eligible.merge(actual, on=["year", "source", "target"], how="left")
    yr_agg["points"] = yr_agg["points"].fillna(0)
    yr_agg["status"] = np.where(yr_agg["points"] > 0, "voted", "abstained")
    yr_agg["era_max"] = yr_agg["year"].map(ERA_MAX).fillna(12)
    yr_agg["nvs_year"] = (yr_agg["points"] / yr_agg["era_max"]).clip(0, 1)
    raw_total_df = (
        actual.groupby(["source", "target"], as_index=False)["points"]
        .sum()
        .rename(columns={"points": "total_votes"})
    )
    agg = (
        yr_agg.groupby(["source", "target"], as_index=False)
        .agg(nvs_sum=("nvs_year", "sum"), years_eligible=("year", "nunique"))
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
        "participants_by_year": participants_by_year,
        "final_standings": build_final_standings(actual, participants_by_year),
    }


def build_matrix(agg, value_col, order):
    m = agg.pivot(index="src_label", columns="tgt_label", values=value_col).fillna(0)
    return m.reindex(index=order, columns=order, fill_value=0)


def get_country_order(period_data, all_labels, order_mode="strength"):
    agg = period_data["agg"]
    if order_mode == "strength":
        out_s = agg.groupby("src_label")["nvs_score"].sum()
        in_s = agg.groupby("tgt_label")["nvs_score"].sum()
        strength = out_s.add(in_s, fill_value=0).to_dict()
        for c in all_labels:
            strength.setdefault(c, 0.0)
        return sorted(all_labels, key=lambda x: strength.get(x, 0.0), reverse=True)
    if order_mode == "cluster" and SCIPY_OK:
        m = build_matrix(agg, "nvs_score", all_labels)
        X = m.values.astype(float)
        if len(all_labels) > 2:
            dist = pdist(X, metric="euclidean")
            Z = linkage(dist, method="average")
            Z = optimal_leaf_ordering(Z, dist)
            return [all_labels[i] for i in leaves_list(Z)]
    return sorted(all_labels)


# =============================================================================
# PAIR BEHAVIOUR COMPUTATION
# =============================================================================

@st.cache_data
def compute_pair_behaviour(start_year: int, end_year: int):
    pdata = compute_period_data(start_year, end_year)
    if pdata is None:
        return pd.DataFrame()
    yr = pdata["yr_agg"].copy()
    yr["src_label"] = yr["source"].map(id2label).fillna(yr["source"])
    yr["tgt_label"] = yr["target"].map(id2label).fillna(yr["target"])
    yr["nvs_score_year"] = yr["nvs_year"] * 12
    rows = []
    for (src, tgt), g in yr.groupby(["src_label", "tgt_label"]):
        g = g.sort_values("year")
        vals = g["nvs_score_year"].to_numpy()
        years = g["year"].to_numpy()
        if len(vals) < 3:
            continue
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        cv_v = float(std_v / (mean_v + 1e-6))
        slope_v = linear_trend_slope(years, vals)
        stability = float(max(0.0, 1.0 - cv_v))
        label = classify_relationship(mean_v, std_v, cv_v, slope_v, stability)
        rows.append({
            "Source": src, "Target": tgt, "Mean NVS": mean_v,
            "Std Dev": std_v, "CV": cv_v, "Trend Slope": slope_v,
            "Stability": stability, "Total Votes": float(g["points"].sum()),
            "Eligible Years": int(g["year"].nunique()), "Class": label,
        })
    return pd.DataFrame(rows)


# =============================================================================
# COMMUNITY DETECTION
# =============================================================================

def detect_communities_from_nvs(matrix_df, min_edge_weight=1.0):
    if not NETWORKX_OK:
        return pd.DataFrame(), None
    countries = list(matrix_df.index)
    G = nx.Graph()
    for c in countries:
        G.add_node(c)
    for i, src in enumerate(countries):
        for j, tgt in enumerate(countries):
            if i >= j:
                continue
            w1 = matrix_df.loc[src, tgt]
            w2 = matrix_df.loc[tgt, src]
            weight = float((w1 + w2) / 2)
            if weight >= min_edge_weight:
                G.add_edge(src, tgt, weight=weight)
    if G.number_of_edges() == 0:
        return pd.DataFrame(), G
    try:
        communities = nx.community.louvain_communities(G, weight="weight", seed=42)
        method = "Louvain"
    except Exception:
        communities = nx.community.greedy_modularity_communities(G, weight="weight")
        method = "Greedy modularity"
    rows = []
    for idx, community in enumerate(communities, start=1):
        members = sorted(list(community))
        sub = G.subgraph(members)
        rows.append({
            "Community": f"C{idx}", "Method": method, "Size": len(members),
            "Members": ", ".join(members), "Internal Edges": sub.number_of_edges(),
            "Average Internal Weight": (
                np.mean([d["weight"] for _, _, d in sub.edges(data=True)])
                if sub.number_of_edges() > 0 else 0
            )
        })
    return pd.DataFrame(rows), G


def build_community_color_map(communities_df):
    if communities_df is None or communities_df.empty or "Community" not in communities_df.columns:
        return {}
    palette = [
        "#1f4e79","#d1495b","#2a9d8f","#f4a261","#6a4c93",
        "#7f5539","#577590","#4d908e","#b56576","#3a86ff",
        "#8338ec","#fb5607","#2b9348","#9b5de5","#e63946","#8ac926",
    ]
    community_names = communities_df["Community"].dropna().astype(str).tolist()
    return {
        community: palette[idx % len(palette)]
        for idx, community in enumerate(community_names)
    }


def _find_geo_columns(nodes_df):
    if nodes_df is None or nodes_df.empty:
        return None, None
    columns = {str(c).lower(): c for c in nodes_df.columns}
    lat_col = columns.get("lat") or columns.get("latitude") or columns.get("y")
    lon_col = columns.get("lon") or columns.get("long") or columns.get("longitude") or columns.get("x")
    return lat_col, lon_col


def make_community_network_figure(G, communities_df):
    if G is None or G.number_of_nodes() == 0:
        return None
    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.8)
    community_lookup = {}
    for _, row in communities_df.iterrows():
        members = [m.strip() for m in row["Members"].split(",")]
        for m in members:
            community_lookup[m] = row["Community"]
    community_colors = build_community_color_map(communities_df)
    edge_x, edge_y = [], []
    for u, v, _ in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.7, color="#9ca3af"), hoverinfo="none", name="Affinity edges")
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]; comm = community_lookup.get(node, "NA")
        node_x.append(x); node_y.append(y)
        node_text.append(f"{node}<br>Community: {comm}")
        node_color.append(community_colors.get(comm, "#64748b"))
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text",
        text=list(G.nodes()), textposition="top center",
        hovertext=node_text, hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(size=13, color=node_color, line=dict(width=1, color="#ffffff")),
        name="Countries")
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Detected Voting Blocs / Communities", height=700,
        showlegend=False, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False)
    )
    return fig


def make_community_world_map_figure(nodes_df, communities_df):
    if nodes_df is None or nodes_df.empty or communities_df is None or communities_df.empty:
        return None
    lat_col, lon_col = _find_geo_columns(nodes_df)
    if not lat_col or not lon_col or "label" not in nodes_df.columns:
        return None
    community_lookup = {}
    for _, row in communities_df.iterrows():
        members = [m.strip() for m in str(row["Members"]).split(",") if m.strip()]
        for member in members:
            community_lookup[member] = str(row["Community"])
    plot_df = nodes_df.copy()
    plot_df["community"] = plot_df["label"].astype(str).map(community_lookup)
    plot_df = plot_df.dropna(subset=["community"])
    if plot_df.empty:
        return None
    communities = communities_df["Community"].dropna().astype(str).tolist()
    community_colors = build_community_color_map(communities_df)
    fig = go.Figure()
    for community in communities:
        sub = plot_df[plot_df["community"] == community].sort_values("label")
        if sub.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lon=sub[lon_col], lat=sub[lat_col], text=sub["label"].astype(str),
            mode="markers", name=community,
            marker=dict(size=12, color=community_colors.get(community, "#64748b"),
                        opacity=0.92, line=dict(width=1.2, color="rgba(255,255,255,0.95)")),
            hovertemplate="<b>%{text}</b><br>Community: " + community + "<extra></extra>",
        ))
    fig.update_geos(projection_type="natural earth", showland=True, landcolor="#f4f6f9",
        showocean=True, oceancolor="#fbfcfe", showcountries=True, countrycolor="#c5cfdb",
        showcoastlines=True, coastlinecolor="#b3bcc8", showframe=False)
    fig.update_layout(title="Detected voting blocs / communities on the world map",
        height=650, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=70, b=10), legend=dict(orientation="h", y=-0.05))
    return fig


def build_top_voters_for_country(period_data, target_country: str, top_n: int = 3):
    if period_data is None or not target_country:
        return pd.DataFrame()
    agg = period_data.get("agg", pd.DataFrame()).copy()
    if agg.empty or "tgt_label" not in agg.columns:
        return pd.DataFrame()
    target_rows = agg[agg["tgt_label"] == target_country].copy()
    if target_rows.empty:
        return pd.DataFrame()
    top_voters = (
        target_rows.sort_values(["nvs_score", "total_votes"], ascending=[False, False])
        .head(top_n)
        [["src_label","tgt_label","nvs_score","total_votes","years_eligible","raw_avg_per_year"]]
        .rename(columns={"src_label":"Voter","tgt_label":"Recipient","nvs_score":"NVS (0–12)",
                         "total_votes":"Total votes","years_eligible":"Eligible years","raw_avg_per_year":"Avg/year"})
        .reset_index(drop=True)
    )
    return top_voters


def make_directed_country_world_map_figure(nodes_df, target_country: str, top_voters_df):
    if nodes_df is None or nodes_df.empty or top_voters_df is None or top_voters_df.empty:
        return None
    lat_col, lon_col = _find_geo_columns(nodes_df)
    if not lat_col or not lon_col or "label" not in nodes_df.columns:
        return None
    coords_df = nodes_df.copy()
    coords_df["label"] = coords_df["label"].astype(str)
    coords_df = coords_df.dropna(subset=[lat_col, lon_col, "label"])
    if coords_df.empty:
        return None
    coord_lookup = {row["label"]: (float(row[lat_col]), float(row[lon_col])) for _, row in coords_df.iterrows()}
    if target_country not in coord_lookup:
        return None
    target_lat, target_lon = coord_lookup[target_country]
    target_rows = top_voters_df[top_voters_df["Recipient"] == target_country].copy()
    if target_rows.empty:
        return None
    voter_rows = []
    for _, row in target_rows.iterrows():
        voter = str(row["Voter"])
        if voter not in coord_lookup:
            continue
        voter_lat, voter_lon = coord_lookup[voter]
        voter_rows.append({
            "Voter": voter, "Recipient": target_country,
            "NVS (0–12)": float(row["NVS (0–12)"]), "Total votes": float(row["Total votes"]),
            "Eligible years": int(row["Eligible years"]), "Avg/year": float(row["Avg/year"]),
            "line_lon": [voter_lon, target_lon], "line_lat": [voter_lat, target_lat],
            "voter_lat": voter_lat, "voter_lon": voter_lon,
        })
    if not voter_rows:
        return None
    plot_rows = pd.DataFrame(voter_rows)
    max_weight = float(plot_rows["NVS (0–12)"].max()) if not plot_rows.empty else 1.0
    if max_weight <= 0:
        max_weight = 1.0
    fig = go.Figure()
    for _, row in plot_rows.iterrows():
        norm_weight = float(row["NVS (0–12)"]) / max_weight
        line_width = 1.8 + (3.2 * norm_weight)
        line_opacity = 0.35 + (0.45 * norm_weight)
        fig.add_trace(go.Scattergeo(
            lon=row["line_lon"], lat=row["line_lat"], mode="lines",
            line=dict(color="#2f7fbe", width=line_width), opacity=line_opacity,
            hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattergeo(
        lon=plot_rows["voter_lon"], lat=plot_rows["voter_lat"],
        text=plot_rows["Voter"],
        customdata=plot_rows[["NVS (0–12)","Total votes","Eligible years","Avg/year"]],
        mode="markers", name="Top voters",
        marker=dict(size=11, color="#d1495b", symbol="circle", line=dict(width=1.2, color="white")),
        hovertemplate=(
            "<b>%{text}</b><br>Votes to " + target_country +
            "<br>NVS: %{customdata[0]:.2f}<br>Total votes: %{customdata[1]:.0f}"
            "<br>Eligible years: %{customdata[2]}<br>Avg/year: %{customdata[3]:.2f}<extra></extra>"
        ), showlegend=False))
    fig.add_trace(go.Scattergeo(
        lon=[target_lon], lat=[target_lat], text=[target_country],
        mode="markers+text", textposition="top center", name=target_country,
        marker=dict(size=15, color="#1f4e79", symbol="star", line=dict(width=1.5, color="white")),
        hovertemplate="<b>%{text}</b><br>Selected country<extra></extra>", showlegend=False))
    fig.update_geos(projection_type="natural earth", showland=True, landcolor="#f4f6f9",
        showocean=True, oceancolor="#fbfcfe", showcountries=True, countrycolor="#c5cfdb",
        showcoastlines=True, coastlinecolor="#b3bcc8", showframe=False)
    fig.update_layout(title=f"Top voters flowing to {target_country}", height=620,
        paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=10,r=10,t=70,b=10), showlegend=False)
    return fig


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

with st.sidebar:
    st.header("Controls")
    year_options = list(range(ROOT_START, ROOT_END + 1))
    start_year = st.selectbox("Start year", year_options, index=0)
    end_year = st.selectbox("End year", year_options, index=len(year_options) - 1)
    if start_year > end_year:
        st.error("Start year must be less than or equal to end year.")
        st.stop()
    st.markdown("---")
    st.markdown("**Country filtering**")
    all_country_ids = sorted(
        set(edges[src_col].dropna().astype(str).unique())
        | set(edges[tgt_col].dropna().astype(str).unique())
    )
    all_labels_unfiltered = [id2label.get(c, c) for c in all_country_ids]
    participation_counts = compute_participation_counts(start_year, end_year)
    window_span = end_year - start_year + 1
    max_possible_participation = min(window_span, int(max(participation_counts.values(), default=1)))
    min_participation = st.slider(
        "Minimum participation years", min_value=1,
        max_value=max(1, max_possible_participation),
        value=min(5, max(1, max_possible_participation)), step=1,
        help=(f"Countries with fewer participation years THAN THIS, WITHIN {start_year}–{end_year} "
              f"({window_span} years total), are excluded from all charts and tables.")
    )
    filtered_labels = [
        lbl for lbl in all_labels_unfiltered
        if participation_years_for_label(lbl, participation_counts) >= min_participation
    ]
    st.caption(f"{len(filtered_labels)} / {len(all_labels_unfiltered)} countries pass this filter")
    top_n_choice = st.selectbox("Top N countries by NVS strength",
        ["All", 10, 15, 20, 25, 30, 40, 50], index=0)
    TOP_N = None if top_n_choice == "All" else int(top_n_choice)
    order_mode = st.selectbox("Country order", ["strength","cluster","alphabetical"], index=0)
    view_mode = st.selectbox("Main matrix view", ["both","nvs","raw","correlation"], index=0)
    threshold = st.slider("Hide weak NVS cells below", min_value=0.0, max_value=6.0, value=0.0, step=0.1)
    st.markdown("---")
    st.markdown("**Community detection**")
    community_min_weight = st.slider("Minimum mutual NVS edge for bloc detection",
        min_value=0.0, max_value=8.0, value=2.0, step=0.25)
    community_top_rank = st.selectbox("Top 3 communities ranked by",
        ["Size","Average Internal Weight"], index=0)
    st.markdown("---")
    st.markdown("**Country drill-down**")
    drilldown_start_year = st.selectbox("Drill-down start year", year_options,
        index=year_options.index(start_year))
    drilldown_end_year = st.selectbox("Drill-down end year", year_options,
        index=year_options.index(end_year))
    drilldown_country = st.selectbox("Country", sorted(filtered_labels), index=0)
    st.markdown("---")
    st.markdown("**Dynamic comparison**")
    enable_comparison = st.checkbox("Enable period comparison", value=True)
    if enable_comparison:
        comp_a_start = st.selectbox("Period A start", year_options, index=0)
        comp_a_end = st.selectbox("Period A end", year_options, index=min(24, len(year_options)-1))
        comp_b_start = st.selectbox("Period B start", year_options, index=min(25, len(year_options)-1))
        comp_b_end = st.selectbox("Period B end", year_options, index=len(year_options)-1)


# =============================================================================
# MAIN PERIOD COMPUTATION
# =============================================================================

pdata = compute_period_data(start_year, end_year)
if pdata is None:
    st.error("No data for the selected period.")
    st.stop()

year_label = str(start_year) if start_year == end_year else f"{start_year}–{end_year}"

if order_mode == "alphabetical":
    order = sorted(filtered_labels)
else:
    order = get_country_order(pdata, filtered_labels, order_mode=order_mode)

if TOP_N is not None:
    order = order[:TOP_N]

agg = filter_agg_by_order(pdata["agg"], order)
m_nvs   = build_matrix(agg, "nvs_score", order)
m_total = build_matrix(agg, "total_votes", order)
m_years = build_matrix(agg, "years_eligible", order)
m_avg   = build_matrix(agg, "raw_avg_per_year", order)

z_nvs   = m_nvs.values.astype(float)
z_total = m_total.values.astype(float)
z_years = m_years.values.astype(float)
z_avg   = m_avg.values.astype(float)

z_nvs_plot   = np.where(z_nvs >= threshold, z_nvs, np.nan) if threshold > 0 else z_nvs.copy()
z_total_plot = np.where(z_nvs >= threshold, z_total, np.nan) if threshold > 0 else z_total.copy()

m_corr = (
    m_nvs.T.corr(method="pearson")
    .fillna(0)
    .reindex(index=order, columns=order, fill_value=0)
)
z_corr = m_corr.values.astype(float)


# =============================================================================
# HOVER TEXT
# =============================================================================

hover_nvs = []
for r in range(len(order)):
    row = []
    for c in range(len(order)):
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
    hover_nvs.append(row)

row_strength = m_nvs.sum(axis=1).to_dict()
hover_corr = [
    [
        (
            f"<b>{order[r]}</b> ↔ <b>{order[c]}</b><br>"
            f"Correlation: <b>{z_corr[r,c]:.3f}</b><br>"
            f"{order[r]} profile strength: <b>{row_strength.get(order[r], 0):.2f}</b><br>"
            f"{order[c]} profile strength: <b>{row_strength.get(order[c], 0):.2f}</b>"
        )
        for c in range(len(order))
    ]
    for r in range(len(order))
]


# =============================================================================
# MAIN MATRIX FIGURE
# =============================================================================

fig = go.Figure()

if view_mode in ("both", "nvs"):
    fig.add_trace(go.Heatmap(
        z=z_nvs_plot, x=order, y=order,
        text=make_text(z_nvs_plot, "{:.1f}"), texttemplate="%{text}",
        textfont=dict(size=7, color="#f0faf4"),
        hovertext=hover_nvs, hovertemplate="%{hovertext}<extra></extra>",
        colorscale=NVS_SCALE, zmin=0, zmax=12, xgap=2, ygap=2,
        visible=True, name="NVS Matrix",
        colorbar=dict(title=dict(text="NVS Score<br>(0–12)", font_size=11), thickness=14, len=0.5)
    ))

if view_mode in ("both", "raw"):
    fig.add_trace(go.Heatmap(
        z=z_total_plot, x=order, y=order,
        text=make_text(z_total_plot, "{:.0f}"), texttemplate="%{text}",
        textfont=dict(size=7, color="#f0faf4"),
        hovertext=hover_nvs, hovertemplate="%{hovertext}<extra></extra>",
        colorscale=NVS_SCALE, zmin=0, xgap=2, ygap=2,
        visible=(view_mode == "raw"), name="Raw Total",
        colorbar=dict(title=dict(text="Total Votes", font_size=11), thickness=14, len=0.5)
    ))

if view_mode in ("both", "correlation"):
    fig.add_trace(go.Heatmap(
        z=z_corr, x=order, y=order,
        text=make_text(z_corr, "{:.2f}"), texttemplate="%{text}",
        textfont=dict(size=7, color="#111111"),
        hovertext=hover_corr, hovertemplate="%{hovertext}<extra></extra>",
        colorscale=CORR_SCALE, zmid=0, zmin=-1, zmax=1, xgap=1, ygap=1,
        visible=(view_mode == "correlation"), name="Correlation",
        colorbar=dict(title=dict(text="Correlation", font_size=11), thickness=14, len=0.5)
    ))

if view_mode == "both":
    labels = [tr.name for tr in fig.data]
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="right", x=0.5, xanchor="center", y=1.08, yanchor="top",
        buttons=[dict(label=l, method="update", args=[{"visible": [name == l for name in labels]}])
                 for l in labels]
    )])

fig.update_layout(
    title=dict(
        text=(
            f"Eurovision Voting Explorer · {year_label}"
            f"<br><span style='font-size:12px; color:#4b5563;'>"
            f"Order: {order_mode} | Minimum participation: {min_participation}+ years | "
            f"Countries shown: {len(order)}</span>"
        ),
        x=0.5, xanchor="center"
    ),
    xaxis=dict(title="Receives points →", type="category", categoryorder="array",
               categoryarray=order, tickvals=order, ticktext=order,
               tickangle=-45, tickfont=dict(size=8), side="bottom", showgrid=False),
    yaxis=dict(title="Gives points ←", type="category", categoryorder="array",
               categoryarray=order, tickvals=order, ticktext=order,
               autorange="reversed", tickfont=dict(size=8), showgrid=False),
    height=max(900, len(order) * 22 + 250),
    paper_bgcolor="white", plot_bgcolor="white",
    margin=dict(l=170, r=120, t=130, b=170)
)

st.plotly_chart(fig, use_container_width=True)

st.download_button(
    "Download current matrix as HTML",
    data=fig_to_html(fig, f"eurovision_{view_mode}_{year_label}").encode("utf-8"),
    file_name=f"eurovision_{view_mode}_{year_label.replace('–','-')}.html",
    mime="text/html"
)


# =============================================================================
# PERIOD COMPARISON
# =============================================================================

if enable_comparison:
    st.subheader("Dynamic period comparison")
    if comp_a_start > comp_a_end or comp_b_start > comp_b_end:
        st.error("Invalid comparison periods.")
    else:
        pdata_a = compute_period_data(comp_a_start, comp_a_end)
        pdata_b = compute_period_data(comp_b_start, comp_b_end)
        if pdata_a is None or pdata_b is None:
            st.warning("No data for one of the comparison periods.")
        else:
            label_a = f"{comp_a_start}" if comp_a_start == comp_a_end else f"{comp_a_start}–{comp_a_end}"
            label_b = f"{comp_b_start}" if comp_b_start == comp_b_end else f"{comp_b_start}–{comp_b_end}"
            agg_a = filter_agg_by_order(pdata_a["agg"], order)
            agg_b = filter_agg_by_order(pdata_b["agg"], order)
            m_a = build_matrix(agg_a, "nvs_score", order)
            m_b = build_matrix(agg_b, "nvs_score", order)
            z_a = m_a.values.astype(float)
            z_b = m_b.values.astype(float)
            z_diff = z_b - z_a
            weak_both = (
                (z_a < threshold) & (z_b < threshold) if threshold > 0
                else np.zeros_like(z_diff, dtype=bool)
            )
            z_diff_plot = np.where(weak_both, np.nan, z_diff)
            diff_hover = [
                [
                    (
                        f"<b>{order[r]}</b> → <b>{order[c]}</b><br>"
                        f"{label_a} NVS: <b>{z_a[r,c]:.2f}</b><br>"
                        f"{label_b} NVS: <b>{z_b[r,c]:.2f}</b><br>"
                        f"Change: <b>{z_diff[r,c]:+.2f}</b>"
                    ) if pd.notna(z_diff_plot[r, c]) else ""
                    for c in range(len(order))
                ]
                for r in range(len(order))
            ]
            fig_diff = go.Figure(go.Heatmap(
                z=z_diff_plot, x=order, y=order,
                hovertext=diff_hover, hovertemplate="%{hovertext}<extra></extra>",
                colorscale=DIFF_SCALE, zmin=-12, zmax=12, zmid=0, xgap=1.5, ygap=1.5,
                colorbar=dict(title=dict(text="Δ NVS<br>B − A", font_size=12),
                              thickness=16, len=0.6, tickvals=[-12,-6,0,6,12],
                              ticktext=["-12","-6","0","+6","+12"])
            ))
            fig_diff.update_layout(
                title=dict(
                    text=(f"Change in Voting Affinity: {label_b} minus {label_a}"
                          f"<br><span style='font-size:12px; color:#4b5563;'>"
                          f"Red = stronger in later period | Blue = weaker in later period</span>"),
                    x=0.5, xanchor="center"),
                xaxis=dict(title="Receives points →", type="category", categoryorder="array",
                           categoryarray=order, tickvals=order, ticktext=order,
                           tickangle=-45, tickfont=dict(size=8), side="bottom"),
                yaxis=dict(title="Gives points ←", type="category", categoryorder="array",
                           categoryarray=order, tickvals=order, ticktext=order,
                           autorange="reversed", tickfont=dict(size=8)),
                height=max(850, len(order) * 22 + 230),
                paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(l=170, r=120, t=130, b=170)
            )
            st.plotly_chart(fig_diff, use_container_width=True)
            diff_rows = []
            for r, src in enumerate(order):
                for c, tgt in enumerate(order):
                    if src == tgt or pd.isna(z_diff_plot[r, c]):
                        continue
                    diff_rows.append({
                        "Source": src, "Target": tgt,
                        f"NVS {label_a}": z_a[r, c], f"NVS {label_b}": z_b[r, c],
                        "Change": z_diff[r, c],
                        "Direction": ("Strengthened" if z_diff[r, c] > 0
                                      else "Weakened" if z_diff[r, c] < 0 else "No major change")
                    })
            diff_df = pd.DataFrame(diff_rows)
            if not diff_df.empty:
                strengthened = diff_df[diff_df["Change"] > 0].sort_values("Change", ascending=False).head(15)
                weakened = diff_df[diff_df["Change"] < 0].sort_values("Change", ascending=True).head(15)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Most strengthened relationships**")
                    st.dataframe(strengthened.round(2), use_container_width=True)
                with col2:
                    st.markdown("**Most weakened relationships**")
                    st.dataframe(weakened.round(2), use_container_width=True)
            else:
                st.info("No comparable relationships available after the selected filters.")


# =============================================================================
# COMMUNITY DETECTION
# =============================================================================

st.subheader("Detected voting blocs / communities")
st.markdown(
    """
    <div style='background:#f8fafc; border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px; margin-bottom:12px;'>
        <b>How communities are calculated</b><br>
        Countries are grouped together when they often vote for each other.
        The app checks how strong the two-way voting connection is, removes very weak links,
        and then puts the countries into clusters using the Louvain algorithm.
    </div>
    """,
    unsafe_allow_html=True,
)

if not NETWORKX_OK:
    st.warning("NetworkX is not installed, so community detection is unavailable.")
else:
    communities_df, community_graph = detect_communities_from_nvs(m_nvs, min_edge_weight=community_min_weight)
    if communities_df.empty:
        st.warning("No communities detected. Try lowering the minimum mutual NVS edge threshold.")
    else:
        community_colors = build_community_color_map(communities_df)
        st.dataframe(communities_df.round({"Average Internal Weight": 2}), use_container_width=True)
        fig_community = make_community_network_figure(community_graph, communities_df)
        if fig_community is not None:
            st.plotly_chart(fig_community, use_container_width=True)
        fig_community_map = make_community_world_map_figure(nodes, communities_df)
        if fig_community_map is not None:
            st.markdown("### Community world map")
            st.caption("The same communities projected onto the countries' geographic locations.")
            st.plotly_chart(fig_community_map, use_container_width=True)
        else:
            st.info("World map view is unavailable because geographic coordinates were not found.")

        st.markdown("---")
        st.markdown("### Country voter drill-down")
        st.caption("Pick a country and year range from the sidebar to see the top 3 incoming voters as directed flows on the world map.")
        st.info(
            f"**Data scope:** all figures below are computed strictly from "
            f"**{ROOT_START}–{ROOT_END}** Eurovision data."
        )

        if drilldown_start_year > drilldown_end_year:
            st.error("Drill-down start year must be less than or equal to the end year.")
        else:
            drill_period = compute_period_data(drilldown_start_year, drilldown_end_year)
            if drill_period is None:
                st.warning("No data available for the selected drill-down years.")
            else:
                top_voters_df = build_top_voters_for_country(drill_period, drilldown_country, top_n=3)
                if top_voters_df.empty:
                    st.info(f"No incoming voting flows found for {drilldown_country} in the selected years.")
                else:
                    st.markdown("#### Top 3 voters")
                    st.caption(f"Showing votes flowing into {drilldown_country} from {drilldown_start_year} to {drilldown_end_year}.")
                    st.dataframe(top_voters_df.round({"NVS (0–12)": 2, "Avg/year": 2}), use_container_width=True)
                    fig_country_map = make_directed_country_world_map_figure(nodes, drilldown_country, top_voters_df)
                    if fig_country_map is not None:
                        st.plotly_chart(fig_country_map, use_container_width=True)
                    else:
                        st.info("Country flow map is unavailable because geographic coordinates were not found.")


# =============================================================================
# GLOBAL BEHAVIOUR TABLE
# =============================================================================

st.subheader("Dynamic relationship classification")
behaviour_df = compute_pair_behaviour(start_year, end_year)
behaviour_df = behaviour_df[
    behaviour_df["Source"].isin(order) & behaviour_df["Target"].isin(order)
].copy()

if behaviour_df.empty:
    st.warning("No dynamic relationship data available for the selected period and filters.")
else:
    st.caption("Click a row in any table below to open the pair trend analysis for that alliance.")
    class_filter = st.selectbox("Filter relationship class",
        ["All"] + sorted(behaviour_df["Class"].unique().tolist()))
    shown_behaviour = behaviour_df.copy()
    if class_filter != "All":
        shown_behaviour = shown_behaviour[shown_behaviour["Class"] == class_filter]
    shown_behaviour = shown_behaviour.sort_values(["Mean NVS","Stability"], ascending=[False,False])
    shown_behaviour_view = shown_behaviour.head(50)
    behaviour_selection = st.dataframe(
        shown_behaviour_view.round({"Mean NVS":2,"Std Dev":2,"CV":2,"Trend Slope":3,"Stability":2,"Total Votes":0}),
        on_select="rerun", selection_mode="single-row", key="behaviour_table", use_container_width=True
    )
    update_selected_pair_from_table(behaviour_selection, shown_behaviour_view,
                                    "Source", "Target", "dynamic relationship classification")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Top stable alliances**")
        stable = behaviour_df[behaviour_df["Class"] == "Strong stable alliance"].sort_values(
            ["Mean NVS","Stability"], ascending=[False,False]).head(10)
        stable_view = stable[["Source","Target","Mean NVS","Stability","Eligible Years"]]
        stable_selection = st.dataframe(stable_view.round(2), on_select="rerun",
            selection_mode="single-row", key="stable_alliances_table", use_container_width=True)
        update_selected_pair_from_table(stable_selection, stable_view, "Source", "Target", "top stable alliances")
    with col2:
        st.markdown("**Top emerging relationships**")
        emerging = behaviour_df[behaviour_df["Class"] == "Emerging relationship"].sort_values(
            "Trend Slope", ascending=False).head(10)
        emerging_view = emerging[["Source","Target","Mean NVS","Trend Slope","Eligible Years"]]
        emerging_selection = st.dataframe(emerging_view.round(2), on_select="rerun",
            selection_mode="single-row", key="emerging_alliances_table", use_container_width=True)
        update_selected_pair_from_table(emerging_selection, emerging_view, "Source", "Target", "top emerging relationships")
    with col3:
        st.markdown("**Top declining relationships**")
        declining = behaviour_df[behaviour_df["Class"] == "Declining relationship"].sort_values(
            "Trend Slope", ascending=True).head(10)
        declining_view = declining[["Source","Target","Mean NVS","Trend Slope","Eligible Years"]]
        declining_selection = st.dataframe(declining_view.round(2), on_select="rerun",
            selection_mode="single-row", key="declining_alliances_table", use_container_width=True)
        update_selected_pair_from_table(declining_selection, declining_view, "Source", "Target", "top declining relationships")


# =============================================================================
# TOP DIRECTED PAIRS
# =============================================================================

st.subheader("Top directed pairs in selected period")
top_pairs = (
    agg.sort_values("nvs_score", ascending=False)
    [["src_label","tgt_label","nvs_score","total_votes","years_eligible","raw_avg_per_year"]]
    .head(25)
    .rename(columns={"src_label":"Voter","tgt_label":"Recipient","nvs_score":"NVS (0–12)",
                     "total_votes":"Total votes","years_eligible":"Eligible years","raw_avg_per_year":"Avg/year"})
)
top_pairs_selection = st.dataframe(
    top_pairs.round({"NVS (0–12)":2,"Avg/year":2}),
    on_select="rerun", selection_mode="single-row", key="top_pairs_table", use_container_width=True
)
update_selected_pair_from_table(top_pairs_selection, top_pairs, "Voter", "Recipient", "top directed pairs")


# =============================================================================
# PAIR TREND ANALYSIS
# =============================================================================

st.subheader("Pair trend analysis")
st.info(
    f"**Data scope:** this analysis only uses Eurovision data from "
    f"**{ROOT_START} to {ROOT_END}** ({ROOT_END - ROOT_START + 1} contest years)."
)

requested_open = bool(st.session_state.pop("pair_trend_requested", False))
if "show_pair_trend" not in st.session_state:
    st.session_state["show_pair_trend"] = requested_open
elif requested_open:
    st.session_state["show_pair_trend"] = True

show_pair_trend = st.toggle("Show pair trend analysis", key="show_pair_trend",
    help="Click a row in one of the alliance tables above to switch this on automatically.")

if show_pair_trend:
    selected_source = st.session_state.get("selected_pair_source")
    selected_target = st.session_state.get("selected_pair_target")
    if selected_source not in order:
        selected_source = order[0]
    if selected_target not in order:
        selected_target = order[min(1, len(order) - 1)]

    pair_col1, pair_col2 = st.columns(2)
    with pair_col1:
        source_country = st.selectbox("Source country", order, index=order.index(selected_source))
    with pair_col2:
        target_country = st.selectbox("Target country", order, index=order.index(selected_target))

    pair_df = pdata["yr_agg"].copy()
    pair_df["src_label"] = pair_df["source"].map(id2label).fillna(pair_df["source"])
    pair_df["tgt_label"] = pair_df["target"].map(id2label).fillna(pair_df["target"])
    final_standing_lookup = build_standing_lookup(pdata.get("final_standings", pd.DataFrame()))

    pair_rows = pair_df[
        (pair_df["src_label"] == source_country) & (pair_df["tgt_label"] == target_country)
    ].copy()

    if pair_rows.empty:
        st.warning("No eligible pair-years found for this pair.")
    else:
        pair_rows = pair_rows.sort_values("year")
        status_counts = pair_rows["status"].value_counts()
        participants_by_year = pdata.get("participants_by_year", {})
        label2id = {v: k for k, v in id2label.items()}
        target_id = label2id.get(target_country, target_country)
        source_id = label2id.get(source_country, source_country)
        years_range = [yr for yr in pdata.get("years", []) if yr >= start_year and yr <= end_year]
        recipient_nonfinal_years = [yr for yr in years_range if str(target_id) not in participants_by_year.get(yr, set())]
        sender_absent_years = [yr for yr in years_range if str(source_id) not in participants_by_year.get(yr, set())]
        highlight_abstained_years = pair_rows.loc[pair_rows["status"] == "abstained", "year"].tolist()

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Years gave points", status_counts.get("voted", 0))
        sc2.metric("Years gave 0", status_counts.get("abstained", 0))
        sc3.metric("Years sender absent", len(sender_absent_years))

        years_all = pair_rows["year"].to_numpy()
        nvs_all = pair_rows["nvs_year"].to_numpy() * 12
        status_all = pair_rows["status"].to_numpy()
        voted_mask = status_all == "voted"
        abstained_mask = status_all == "abstained"
        present_mask = voted_mask | abstained_mask
        years_present = years_all[present_mask]
        nvs_present = nvs_all[present_mask]

        if nvs_present.size > 0:
            mean_v = float(np.mean(nvs_present))
            std_v = float(np.std(nvs_present))
            cv_v = float(std_v / (mean_v + 1e-6))
            slope_v = linear_trend_slope(years_present, nvs_present)
            stability = float(max(0.0, 1.0 - cv_v))
            cp_idx, cp_score = compute_simple_change_point(nvs_present)
            cp_year = int(years_present[cp_idx]) if cp_idx is not None else None
        else:
            mean_v = std_v = cv_v = slope_v = stability = 0.0
            cp_year = None; cp_score = 0.0

        relationship_class = classify_relationship(mean_v, std_v, cv_v, slope_v, stability)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Mean NVS", f"{mean_v:.2f}")
        m2.metric("Std dev", f"{std_v:.2f}")
        m3.metric("CV", f"{cv_v:.2f}")
        m4.metric("Trend slope", f"{slope_v:.3f}")
        m5.metric("Stability", f"{stability:.2f}")
        st.success(f"Relationship classification: **{relationship_class}**")
        if cp_year:
            st.info(f"Possible relationship change around **{cp_year}** (heuristic score {cp_score:.2f}).")

        fig_full = build_full_pair_figure(pair_rows, source_country, target_country,
            standing_lookup=final_standing_lookup, highlight_abstained_years=highlight_abstained_years)
        st.caption("Full series across all years")
        st.plotly_chart(fig_full, use_container_width=True)

        if recipient_nonfinal_years:
            st.info(f"Recipient did not qualify for the final in: {', '.join(str(y) for y in recipient_nonfinal_years)}")
        if sender_absent_years:
            st.info(f"Sender was not present in: {', '.join(str(y) for y in sender_absent_years)}")

        fig_pair = build_pair_interval_figure(pair_rows, source_country, target_country,
            start_year, end_year, standing_lookup=final_standing_lookup, interval_years=5)
        st.caption("Each colored segment is a 5-year interval.")
        st.plotly_chart(fig_pair, use_container_width=True)

        interval_summary = build_pair_interval_summary(pair_rows, start_year, end_year, interval_years=5)
        if interval_summary.empty:
            st.info("No 5-year interval summary available for this pair.")
        else:
            st.dataframe(
                interval_summary.round({"Mean NVS":2,"Std Dev":2,"Trend Slope":3,"Stability":2}),
                use_container_width=True, hide_index=True,
            )