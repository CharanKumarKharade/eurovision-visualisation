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
    """
    Exploratory classification of a directed relationship A -> B.

    These thresholds are heuristic and should be described as exploratory
    thresholds, not as formal statistical hypothesis tests.
    """
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
    """
    Keep only relationships where both source and target are currently visible.

    This applies:
    - minimum participation filtering
    - TOP N filtering
    - selected country ordering
    """
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
    """
    Build yearly final-round standings from the vote totals already present in the dataset.

    The standings are based on points received in the final for each year and country.
    """
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


def build_pair_interval_figure(
    pair_rows: pd.DataFrame,
    source_country: str,
    target_country: str,
    start_year: int,
    end_year: int,
    standing_lookup: dict | None = None,
    interval_years: int = 5,
):
    fig_pair = go.Figure()

    palette = [
        "#0b3c6f",
        "#2f7fbe",
        "#5ea7d8",
        "#93c3e5",
        "#165a9c",
        "#2166ac",
        "#4c78a8",
        "#8da0cb",
        "#7b3294",
        "#c51b7d",
    ]

    segment_count = 0

    for interval_start in range(start_year, end_year + 1, interval_years):
        interval_end = min(interval_start + interval_years - 1, end_year)
        interval_df = pair_rows[
            (pair_rows["year"] >= interval_start)
            & (pair_rows["year"] <= interval_end)
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
                f"Year {int(row['year'])}<br>"
                f"NVS {float(row['nvs_year']) * 12:.2f}<br>"
                f"Status: {'gave points' if row['status'] == 'voted' else '0 points'}<br>"
                f"Source final place: #{int(source_place) if pd.notna(source_place) else 'N/A'}<br>"
                f"Target final place: #{int(target_place) if pd.notna(target_place) else 'N/A'}"
                + (f"<br><b>Winner of the year: {winner_country}</b>" if winner_country else "")
                + "<extra></extra>"
            )

        fig_pair.add_trace(go.Scatter(
            x=interval_df["year"],
            y=interval_df["nvs_year"] * 12,
            mode="lines+markers",
            name=f"{interval_start}–{interval_end}",
            line=dict(color=color, width=3),
            marker=dict(color=color, size=8),
            hovertext=hover_lines,
            hovertemplate="%{hovertext}",
        ))

    if not pair_rows.empty:
        values = (pair_rows["nvs_year"].to_numpy(dtype=float) * 12)
        years = pair_rows["year"].to_numpy(dtype=int)
        mean_v = float(np.mean(values))
        cp_idx, cp_score = compute_simple_change_point(values)
        cp_year = int(years[cp_idx]) if cp_idx is not None else None

        fig_pair.add_hline(
            y=mean_v,
            line_dash="dot",
            line_color="#5ea7d8",
            annotation_text=f"mean {mean_v:.2f}",
            annotation_position="top right"
        )

        if cp_year is not None:
            fig_pair.add_vline(
                x=cp_year,
                line_width=2,
                line_dash="dash",
                line_color="red",
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
        plot_bgcolor="white"
    )

    return fig_pair


def build_full_pair_figure(
    pair_rows: pd.DataFrame,
    source_country: str,
    target_country: str,
    standing_lookup: dict | None = None,
    highlight_abstained_years: list | None = None,
):
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
        source_country if source_is_winner else target_country if target_is_winner else None
        for source_is_winner, target_is_winner in zip(source_winner_mask, target_winner_mask)
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

        for year, value, src_place, tgt_place, winner_name in zip(
            years_all[mask],
            nvs_all[mask],
            source_places[mask],
            target_places[mask],
            winner_country[mask],
        ):
            rows.append(
                f"Year {int(year)}<br>"
                f"NVS {float(value):.2f}<br>"
                f"Status: {status_label}<br>"
                f"Source final place: #{int(src_place) if pd.notna(src_place) else 'N/A'}<br>"
                f"Target final place: #{int(tgt_place) if pd.notna(tgt_place) else 'N/A'}"
                + (f"<br><b>Winner of the year: {winner_name}</b>" if winner_name else "")
                + "<extra></extra>"
            )

        return rows

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=years_all[voted_nonwinner_mask],
        y=nvs_all[voted_nonwinner_mask],
        mode="markers",
        name="Gave points",
        marker=dict(size=8, color="#2f7fbe", symbol="circle"),
        hovertext=_hover_rows(voted_nonwinner_mask, "gave points"),
        hovertemplate="%{hovertext}",
    ))

    fig.add_trace(go.Scatter(
        x=years_all[voted_winner_mask],
        y=nvs_all[voted_winner_mask],
        mode="markers",
        name="Winner year",
        marker=dict(size=12, color="#f59e0b", symbol="star", line=dict(width=1.5, color="#ffffff")),
        hovertext=_hover_rows(voted_winner_mask, "gave points"),
        hovertemplate="%{hovertext}",
    ))

    fig.add_trace(go.Scatter(
        x=years_all[abstained_nonwinner_mask],
        y=nvs_all[abstained_nonwinner_mask],
        mode="markers",
        name="0 points, both present",
        marker=dict(size=8, color="#d6604d", symbol="circle-open", line=dict(width=2)),
        hovertext=_hover_rows(abstained_nonwinner_mask, "0 points"),
        hovertemplate="%{hovertext}",
    ))

    fig.add_trace(go.Scatter(
        x=years_all[abstained_winner_mask],
        y=nvs_all[abstained_winner_mask],
        mode="markers",
        name="Winner year, 0 points",
        marker=dict(size=12, color="#f59e0b", symbol="star-open", line=dict(width=1.5, color="#ffffff")),
        hovertext=_hover_rows(abstained_winner_mask, "0 points"),
        hovertemplate="%{hovertext}",
    ))

    fig.add_trace(go.Scatter(
        x=years_all[absent_mask],
        y=[None] * absent_mask.sum(),
        mode="markers",
        name="Absent",
        marker=dict(size=9, color="#888780", symbol="x"),
        hovertemplate="Year %{x}<br>N/A<extra></extra>"
    ))

    # Highlight specific abstained years (e.g., recipient qualified but sender gave 0)
    if highlight_abstained_years:
        ha = np.isin(years_all, np.array(highlight_abstained_years)) & abstained_mask
        if ha.any():
            fig.add_trace(go.Scatter(
                x=years_all[ha],
                y=nvs_all[ha],
                mode="markers",
                name="Qualified but received 0",
                marker=dict(size=11, color="#ff3333", symbol="diamond"),
                hovertemplate="Year %{x}<br>NVS 0<br>Recipient qualified but received 0 points<extra></extra>",
            ))

    # 5-year rolling mean for the full series
    present_mask = voted_mask | abstained_mask
    years_present = years_all[present_mask]
    nvs_present = nvs_all[present_mask]

    if nvs_present.size > 0:
        roll = pd.Series(nvs_present, index=years_present).rolling(window=5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=roll.index,
            y=roll.values,
            mode="lines",
            name="5-year rolling mean",
            line=dict(color="#0b3c6f", width=2.5),
            hovertemplate="Year %{x}<br>Rolling NVS %{y:.2f}<extra></extra>"
        ))

        mean_v = float(np.mean(nvs_present))
        fig.add_hline(
            y=mean_v,
            line_dash="dot",
            line_color="#5ea7d8",
            annotation_text=f"mean {mean_v:.2f}",
            annotation_position="top right"
        )

        cp_idx, cp_score = compute_simple_change_point(nvs_present)
        cp_year = int(years_present[cp_idx]) if cp_idx is not None else None
        if cp_year is not None:
            fig.add_vline(
                x=cp_year,
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text=f"change ~{cp_year}",
                annotation_position="top left"
            )

    fig.update_layout(
        title=f"Full-year Pair Trend: {source_country} → {target_country}",
        xaxis_title="Year",
        yaxis_title="NVS (0–12)",
        yaxis=dict(range=[-0.5, 13]),
        legend=dict(orientation="h", y=-0.2),
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white"
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

    src_col = find_col(
        edges.columns,
        {"source", "from", "from_country"},
        ["source", "from", "voter"]
    )

    tgt_col = find_col(
        edges.columns,
        {"target", "to", "to_country"},
        ["target", "to", "recip"]
    )

    # 🔥 STRONGER POINT COLUMN DETECTION
    numeric_cols = [
        c for c in edges.columns
        if pd.api.types.is_numeric_dtype(edges[c])
    ]

    pts_col = None
    for c in numeric_cols:
        if any(k in c.lower() for k in ["point", "score", "pts", "value", "weight"]):
            pts_col = c
            break

    if pts_col is None:
        if "weight" in edges.columns and pd.api.types.is_numeric_dtype(edges["weight"]):
            pts_col = "weight"
        else:
            raise ValueError(f"No valid points column found. Available numeric: {numeric_cols}")

    # 🔥 FORCE STANDARD NAME
    edges = edges.rename(columns={pts_col: "points"})
    pts_col = "points"

    return nodes, edges, id2label, src_col, tgt_col, pts_col


nodes, edges, id2label, src_col, tgt_col, pts_col = load_data(NODES_FILE, EDGES_FILE)

all_years_available = sorted(edges["year"].unique())
ROOT_START = max(ROOT_START, min(all_years_available))
ROOT_END = min(ROOT_END, max(all_years_available))

# =============================================================================
# GLOBAL DATA-SCOPE NOTICE
#
# This app intentionally restricts every computation to the
# [ROOT_START, ROOT_END] window (default: 1975-2025). Even if the source
# CSV (eurovision_senior.csv) contains rows for years outside this range
# (for example, earlier contest history before 1975), those rows are
# excluded everywhere in this app: the main NVS matrix, community/bloc
# detection, period comparisons, "top voters" drill-down, and pair trend
# analysis. This keeps the NVS normalisation consistent, since the
# era-based scoring system (ERA_MAX) defined above is only calibrated for
# 1975 onward.
# =============================================================================

st.info(
    f"ℹ️ **Data scope for this app:** every chart, table, and statistic here "
    f"is computed using Eurovision voting data from **{ROOT_START} to "
    f"{ROOT_END}** only ({ROOT_END - ROOT_START + 1} contest years). This "
    f"includes the main voting matrix, community/bloc detection, period "
    f"comparisons, the country voter drill-down ('top voters'), and the "
    f"pair trend analysis further down the page. Data from years outside "
    f"this range — if present in the source file — is never loaded or "
    f"used anywhere in this app."
)


# =============================================================================
# DRAFT VISUALISATION GALLERY  (Graph Drawing Contest 2026 poster drafts)
#
# This section lets a reader jump from the main exploratory app into any of
# five standalone "draft" visualisations (built in draft_visualizations.py),
# each exploring a different, non-overlapping angle on the same 1975-2025
# NVS dataset. Selecting a draft below REPLACES the rest of the app for
# this run — a "Back to main app" button returns to the normal dashboard.
#
# Navigation is done via st.session_state["active_draft"]:
#   None            -> show the normal app (everything below this block)
#   "<draft name>"  -> render only that draft's figure + explanation,
#                      then st.stop() so nothing else on the page renders.
# =============================================================================

DRAFT_REGISTRY = {
    "— None (show main app) —": None,
    "1. Unrequited Love — Voting Asymmetry":     "unrequited_love",
    "2. The Neighbour Effect":                   "neighbour_effect",
    "3. Alliance Lifespan Arcs":                 "lifespan_arcs",
    "4. Rise and Fall — Era Dominance":          "rise_and_fall",
    "5. Voting Hall of Fame":                    "hall_of_fame",
}

if "active_draft" not in st.session_state:
    st.session_state["active_draft"] = None

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎨 Draft visualisations (GD Contest 2026)")
    st.caption(
        "Five standalone poster-draft diagrams, each using a different "
        "analytical lens on the same 1975–2025 NVS data. Pick one to view "
        "it full-page; use 'Back to main app' there to return here."
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
        # User explicitly reset the dropdown back to "None"
        st.session_state["active_draft"] = None


def _render_draft_header(title: str, explanation_md: str):
    """Shared header block for every draft page: title, explanation, back button."""
    st.title(f"📊 Draft: {title}")
    st.caption(
        f"Data scope: Eurovision {ROOT_START}–{ROOT_END} only. "
        f"This is a standalone draft for the Graph Drawing Contest 2026 poster — "
        f"it is computed independently from the sidebar filters on the main app."
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
            f"Could not load draft_visualizations.py: {_draft_key}. "
            "Make sure draft_visualizations.py is in the same folder as this app. "
            f"Import error: {_draftviz_import_error if '_draftviz_import_error' in dir() else 'unknown'}"
        )
        if st.button("⬅ Back to main app"):
            st.session_state["active_draft"] = None
            st.rerun()
        st.stop()

    # Full 1975-2025 dataset, independent of the main app's sidebar filters —
    # drafts are meant to stand alone as poster candidates.
    draft_df_full = edges.copy()

    with st.spinner("Building draft visualisation — this can take a few seconds for community detection..."):

        if _draft_key == "unrequited_love":
            fig, title, explanation = draftviz.build_unrequited_love(
                draft_df_full, id2label, min_years=15, top_n_arcs=40
            )

        elif _draft_key == "neighbour_effect":
            fig, title, explanation = draftviz.build_neighbour_effect(
                draft_df_full, id2label, nodes, min_years=15, n_labels=8
            )

        elif _draft_key == "lifespan_arcs":
            fig, title, explanation = draftviz.build_lifespan_arcs(
                draft_df_full, id2label, min_years=15, threshold=0.35, top_n=35
            )

        elif _draft_key == "rise_and_fall":
            fig, title, explanation = draftviz.build_rise_and_fall(
                draft_df_full, id2label, min_years=10, decade_size=10, top_supporters=5
            )

        elif _draft_key == "hall_of_fame":
            fig, title, explanation = draftviz.build_hall_of_fame(
                draft_df_full, id2label, min_years=15
            )

        else:
            fig, title, explanation = None, "Unknown draft", "Unknown draft key."

    _render_draft_header(title, explanation)

    if fig is None:
        st.warning("This draft could not be generated with the current data/filters.")
    else:
        st.plotly_chart(fig, use_container_width=True)

    st.stop()  # Do not render the rest of the main app below this point.


# =============================================================================
# PARTICIPATION COUNTS
# =============================================================================

@st.cache_data
def compute_participation_counts(start_year: int | None = None, end_year: int | None = None):
    """
    Count distinct years each country appears in (as source or target).

    If start_year/end_year are provided, only years within that window are
    counted — so 'participated 25+ years' means 25+ years WITHIN the
    selected range, not 25+ years across the entire 1975-2025 dataset.
    """
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

    # 🔥 ENSURE points EXISTS
    if "points" not in df.columns:
        raise ValueError("CRITICAL: 'points' column missing after load.")

    years_with_data = sorted(df["year"].unique())

    actual = (
        df.groupby(["year", src_col, tgt_col], as_index=False)["points"]
          .sum()
          .rename(columns={
              src_col: "source",
              tgt_col: "target"
          })
    )

    actual["points"] = pd.to_numeric(actual["points"], errors="coerce").fillna(0)
    actual = actual[actual["points"] > 0].copy()

    participants_by_year = {}

    for yr in years_with_data:
        yr_df = df[df["year"] == yr]
        participants_by_year[yr] = (
            set(yr_df[src_col].astype(str))
            | set(yr_df[tgt_col].astype(str))
        )

    pair_rows = [
        (yr, s, t)
        for yr in years_with_data
        for s in participants_by_year[yr]
        for t in participants_by_year[yr]
        if s != t
    ]

    eligible = pd.DataFrame(pair_rows, columns=["year", "source", "target"])

    yr_agg = eligible.merge(
        actual,
        on=["year", "source", "target"],
        how="left"
    )

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
        "participants_by_year": participants_by_year,
        "final_standings": build_final_standings(actual, participants_by_year),
    }

def build_matrix(agg, value_col, order):
    m = agg.pivot(
        index="src_label",
        columns="tgt_label",
        values=value_col
    ).fillna(0)

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
    """
    Compute dynamic relationship metrics only for the selected year range.

    DATA SCOPE: `start_year` and `end_year` are always values from the
    sidebar selectboxes, which are themselves constrained to
    [ROOT_START, ROOT_END] = [1975, 2025]. This function (and therefore the
    "Pair trend analysis" and relationship classification tables it feeds)
    never sees Eurovision data outside that fixed window.
    """
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
            "Source": src,
            "Target": tgt,
            "Mean NVS": mean_v,
            "Std Dev": std_v,
            "CV": cv_v,
            "Trend Slope": slope_v,
            "Stability": stability,
            "Total Votes": float(g["points"].sum()),
            "Eligible Years": int(g["year"].nunique()),
            "Class": label,
        })

    return pd.DataFrame(rows)


# =============================================================================
# COMMUNITY DETECTION
# =============================================================================

def detect_communities_from_nvs(matrix_df, min_edge_weight=1.0):
    """
    Detect voting blocs from the NVS matrix.

    The directed matrix is converted into an undirected mutual-affinity graph:
    weight(A,B) = mean(NVS(A->B), NVS(B->A))
    """
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
        communities = nx.community.louvain_communities(
            G,
            weight="weight",
            seed=42
        )
        method = "Louvain"
    except Exception:
        communities = nx.community.greedy_modularity_communities(
            G,
            weight="weight"
        )
        method = "Greedy modularity"

    rows = []

    for idx, community in enumerate(communities, start=1):
        members = sorted(list(community))
        sub = G.subgraph(members)

        rows.append({
            "Community": f"C{idx}",
            "Method": method,
            "Size": len(members),
            "Members": ", ".join(members),
            "Internal Edges": sub.number_of_edges(),
            "Average Internal Weight": (
                np.mean([d["weight"] for _, _, d in sub.edges(data=True)])
                if sub.number_of_edges() > 0 else 0
            )
        })

    return pd.DataFrame(rows), G


def build_community_color_map(communities_df):
    """
    Build a stable, high-contrast categorical palette for detected communities.
    The same mapping is reused by the network graph and the world map.
    """
    if communities_df is None or communities_df.empty or "Community" not in communities_df.columns:
        return {}

    palette = [
        "#1f4e79",  # deep blue
        "#d1495b",  # red
        "#2a9d8f",  # teal
        "#f4a261",  # orange
        "#6a4c93",  # purple
        "#7f5539",  # brown
        "#577590",  # slate blue
        "#4d908e",  # muted cyan
        "#b56576",  # rose
        "#3a86ff",  # bright blue
        "#8338ec",  # violet
        "#fb5607",  # orange-red
        "#2b9348",  # green
        "#9b5de5",  # light purple
        "#e63946",  # strong red
        "#8ac926",  # lime
    ]

    community_names = communities_df["Community"].dropna().astype(str).tolist()
    return {
        community: palette[idx % len(palette)]
        for idx, community in enumerate(community_names)
    }


def _find_geo_columns(nodes_df):
    """
    Resolve coordinate column names from common variants.
    """
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

    edge_x = []
    edge_y = []

    for u, v, _ in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.7, color="#9ca3af"),
        hoverinfo="none",
        name="Affinity edges"
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        comm = community_lookup.get(node, "NA")

        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Community: {comm}")
        node_color.append(community_colors.get(comm, "#64748b"))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        hovertext=node_text,
        hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(
            size=13,
            color=node_color,
            line=dict(width=1, color="#ffffff")
        ),
        name="Countries"
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title="Detected Voting Blocs / Communities",
        height=700,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
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
            lon=sub[lon_col],
            lat=sub[lat_col],
            text=sub["label"].astype(str),
            mode="markers",
            name=community,
            marker=dict(
                size=12,
                color=community_colors.get(community, "#64748b"),
                opacity=0.92,
                line=dict(width=1.2, color="rgba(255,255,255,0.95)")
            ),
            hovertemplate="<b>%{text}</b><br>Community: " + community + "<extra></extra>",
        ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#f4f6f9",
        showocean=True,
        oceancolor="#fbfcfe",
        showcountries=True,
        countrycolor="#c5cfdb",
        showcoastlines=True,
        coastlinecolor="#b3bcc8",
        showframe=False,
    )

    fig.update_layout(
        title="Detected voting blocs / communities on the world map",
        height=650,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", y=-0.05),
    )

    return fig


def build_top_voters_for_country(period_data, target_country: str, top_n: int = 3):
    """
    Find the top-N countries (by NVS score) that voted for `target_country`.

    IMPORTANT — SCOPE OF DATA:
    `period_data` must come from `compute_period_data(start_year, end_year)`,
    where start_year and end_year are constrained to the 1975-2025 window
    (ROOT_START / ROOT_END). This function does NOT independently re-filter
    by year — it only sees whatever year range was already baked into
    `period_data` by the caller. If the underlying eurovision_senior.csv
    contains years before 1975 (e.g. early contest history), those rows are
    excluded upstream by ROOT_START and never reach this function.

    In other words: "top voters" here always means "top voters within the
    1975-2025 era using NVS normalisation", never the full unfiltered
    historical dataset.
    """
    if period_data is None or not target_country:
        return pd.DataFrame()

    agg = period_data.get("agg", pd.DataFrame()).copy()
    if agg.empty or "tgt_label" not in agg.columns:
        return pd.DataFrame()

    target_rows = agg[agg["tgt_label"] == target_country].copy()
    if target_rows.empty:
        return pd.DataFrame()

    top_voters = (
        target_rows
        .sort_values(["nvs_score", "total_votes"], ascending=[False, False])
        .head(top_n)
        [["src_label", "tgt_label", "nvs_score", "total_votes", "years_eligible", "raw_avg_per_year"]]
        .rename(columns={
            "src_label": "Voter",
            "tgt_label": "Recipient",
            "nvs_score": "NVS (0–12)",
            "total_votes": "Total votes",
            "years_eligible": "Eligible years",
            "raw_avg_per_year": "Avg/year",
        })
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

    coord_lookup = {
        row["label"]: (float(row[lat_col]), float(row[lon_col]))
        for _, row in coords_df.iterrows()
    }

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
            "Voter": voter,
            "Recipient": target_country,
            "NVS (0–12)": float(row["NVS (0–12)"]),
            "Total votes": float(row["Total votes"]),
            "Eligible years": int(row["Eligible years"]),
            "Avg/year": float(row["Avg/year"]),
            "line_lon": [voter_lon, target_lon],
            "line_lat": [voter_lat, target_lat],
            "voter_lat": voter_lat,
            "voter_lon": voter_lon,
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
            lon=row["line_lon"],
            lat=row["line_lat"],
            mode="lines",
            line=dict(color="#2f7fbe", width=line_width),
            opacity=line_opacity,
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.add_trace(go.Scattergeo(
        lon=plot_rows["voter_lon"],
        lat=plot_rows["voter_lat"],
        text=plot_rows["Voter"],
        customdata=plot_rows[["NVS (0–12)", "Total votes", "Eligible years", "Avg/year"]],
        mode="markers",
        name="Top voters",
        marker=dict(
            size=11,
            color="#d1495b",
            symbol="circle",
            line=dict(width=1.2, color="white"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>Votes to " + target_country +
            "<br>NVS: %{customdata[0]:.2f}" +
            "<br>Total votes: %{customdata[1]:.0f}" +
            "<br>Eligible years: %{customdata[2]}" +
            "<br>Avg/year: %{customdata[3]:.2f}<extra></extra>"
        ),
        showlegend=False,
    ))

    fig.add_trace(go.Scattergeo(
        lon=[target_lon],
        lat=[target_lat],
        text=[target_country],
        mode="markers+text",
        textposition="top center",
        name=target_country,
        marker=dict(
            size=15,
            color="#1f4e79",
            symbol="star",
            line=dict(width=1.5, color="white"),
        ),
        hovertemplate="<b>%{text}</b><br>Selected country<extra></extra>",
        showlegend=False,
    ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#f4f6f9",
        showocean=True,
        oceancolor="#fbfcfe",
        showcountries=True,
        countrycolor="#c5cfdb",
        showcoastlines=True,
        coastlinecolor="#b3bcc8",
        showframe=False,
    )

    fig.update_layout(
        title=f"Top voters flowing to {target_country}",
        height=620,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=70, b=10),
        showlegend=False,
    )

    return fig


def make_directed_community_flow_map_figure(
    nodes_df,
    community_graph,
    community_name,
    community_color,
    max_flows: int = 20,
):
    """
    Plot a directed flow map for a single community.

    For readability, this shows the strongest outgoing edge per source country
    within the selected community, so the direction of flow is visible without
    overloading the map.
    """
    if (
        nodes_df is None or nodes_df.empty
        or community_graph is None
        or community_graph.number_of_nodes() == 0
    ):
        return None

    lat_col, lon_col = _find_geo_columns(nodes_df)
    if not lat_col or not lon_col or "label" not in nodes_df.columns:
        return None

    coords_df = nodes_df.copy()
    coords_df["label"] = coords_df["label"].astype(str)
    coords_df = coords_df.dropna(subset=[lat_col, lon_col, "label"])

    if coords_df.empty:
        return None

    coord_lookup = {
        row["label"]: (float(row[lat_col]), float(row[lon_col]))
        for _, row in coords_df.iterrows()
    }

    base_nodes = []
    for node in community_graph.nodes():
        if node in coord_lookup:
            lat, lon = coord_lookup[node]
            base_nodes.append((node, lat, lon))

    if not base_nodes:
        return None

    flow_edges = []

    for source in community_graph.nodes():
        source_edges = list(community_graph.out_edges(source, data=True))
        if not source_edges:
            continue

        best_edge = max(
            source_edges,
            key=lambda e: float(e[2].get("display_weight", e[2].get("weight", 0.0)))
        )

        _, target, data = best_edge
        if source not in coord_lookup or target not in coord_lookup:
            continue

        s_lat, s_lon = coord_lookup[source]
        t_lat, t_lon = coord_lookup[target]
        weight = float(data.get("display_weight", data.get("weight", 0.0)))

        flow_edges.append({
            "source": source,
            "target": target,
            "weight": weight,
            "line_lon": [s_lon, t_lon],
            "line_lat": [s_lat, t_lat],
            "mid_lon": (s_lon + t_lon) / 2,
            "mid_lat": (s_lat + t_lat) / 2,
        })

    if not flow_edges:
        return None

    flow_edges = sorted(flow_edges, key=lambda d: d["weight"], reverse=True)[:max_flows]
    max_weight = max((edge["weight"] for edge in flow_edges), default=1.0)
    if max_weight <= 0:
        max_weight = 1.0

    fig = go.Figure()

    # Base community nodes.
    fig.add_trace(go.Scattergeo(
        lon=[lon for _, _, lon in base_nodes],
        lat=[lat for _, lat, _ in base_nodes],
        text=[node for node, _, _ in base_nodes],
        mode="markers+text",
        textposition="top center",
        name=f"{community_name} countries",
        marker=dict(
            size=11,
            color=community_color,
            opacity=0.92,
            line=dict(width=1.2, color="rgba(255,255,255,0.95)")
        ),
        hovertemplate="<b>%{text}</b><br>Community: " + community_name + "<extra></extra>",
        showlegend=False,
    ))

    # Directed flows: one line per strongest source->target connection.
    source_lons = []
    source_lats = []
    source_text = []
    target_lons = []
    target_lats = []
    target_text = []

    for edge in flow_edges:
        norm_weight = edge["weight"] / max_weight
        line_width = 0.6 + (5.0 * norm_weight)
        line_opacity = 0.22 + (0.58 * norm_weight)

        fig.add_trace(go.Scattergeo(
            lon=edge["line_lon"],
            lat=edge["line_lat"],
            mode="lines",
            line=dict(color=community_color, width=line_width),
            opacity=line_opacity,
            hoverinfo="skip",
            showlegend=False,
        ))

        source_lons.append(edge["line_lon"][0])
        source_lats.append(edge["line_lat"][0])
        source_text.append(f"<b>{edge['source']}</b> (source)<br>→ <b>{edge['target']}</b>")

        target_lons.append(edge["line_lon"][1])
        target_lats.append(edge["line_lat"][1])
        target_text.append(f"<b>{edge['target']}</b> (target)<br>from <b>{edge['source']}</b>")

    if source_lons:
        fig.add_trace(go.Scattergeo(
            lon=source_lons,
            lat=source_lats,
            mode="markers",
            marker=dict(
                size=7,
                color=community_color,
                symbol="circle",
                line=dict(width=1, color="white"),
            ),
            hovertext=source_text,
            hovertemplate="%{hovertext}<br>Direction: source<extra></extra>",
            showlegend=False,
        ))

    if target_lons:
        fig.add_trace(go.Scattergeo(
            lon=target_lons,
            lat=target_lats,
            mode="markers",
            marker=dict(
                size=10,
                color=community_color,
                symbol="triangle-right",
                line=dict(width=1.2, color="white"),
            ),
            hovertext=target_text,
            hovertemplate="%{hovertext}<br>Direction: target<extra></extra>",
            showlegend=False,
        ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#f4f6f9",
        showocean=True,
        oceancolor="#fbfcfe",
        showcountries=True,
        countrycolor="#c5cfdb",
        showcoastlines=True,
        coastlinecolor="#b3bcc8",
        showframe=False,
    )

    fig.update_layout(
        title=f"{community_name} directed flow map",
        height=620,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=70, b=10),
        showlegend=False,
    )

    return fig


def make_directed_community_vote_map_figure(
    nodes_df,
    directed_edge_table: pd.DataFrame,
    community_name: str,
    community_color: str,
    max_edges: int = 25,
):
    if nodes_df is None or nodes_df.empty or directed_edge_table is None or directed_edge_table.empty:
        return None

    lat_col, lon_col = _find_geo_columns(nodes_df)
    if not lat_col or not lon_col or "label" not in nodes_df.columns:
        return None

    coords_df = nodes_df.copy()
    coords_df["label"] = coords_df["label"].astype(str)
    coords_df = coords_df.dropna(subset=[lat_col, lon_col, "label"])

    if coords_df.empty:
        return None

    coord_lookup = {
        row["label"]: (float(row[lat_col]), float(row[lon_col]))
        for _, row in coords_df.iterrows()
    }

    plot_edges = directed_edge_table.copy().head(max_edges)
    plot_edges = plot_edges[
        plot_edges["Source"].isin(coord_lookup)
        & plot_edges["Target"].isin(coord_lookup)
    ].copy()

    if plot_edges.empty:
        return None

    max_votes = float(plot_edges["Total votes"].max()) if not plot_edges.empty else 1.0
    if max_votes <= 0:
        max_votes = 1.0

    node_stats = {}
    for _, row in plot_edges.iterrows():
        source = str(row["Source"])
        target = str(row["Target"])
        votes = float(row["Total votes"])

        source_stats = node_stats.setdefault(source, {"incoming": 0.0, "outgoing": 0.0})
        target_stats = node_stats.setdefault(target, {"incoming": 0.0, "outgoing": 0.0})
        source_stats["outgoing"] += votes
        target_stats["incoming"] += votes

    max_incoming = max((stats["incoming"] for stats in node_stats.values()), default=1.0)
    if max_incoming <= 0:
        max_incoming = 1.0

    fig = go.Figure()

    for _, row in plot_edges.iterrows():
        source = str(row["Source"])
        target = str(row["Target"])
        votes = float(row["Total votes"])
        nvs_score = float(row["NVS (0–12)"])

        s_lat, s_lon = coord_lookup[source]
        t_lat, t_lon = coord_lookup[target]
        norm_votes = votes / max_votes
        line_width = 0.4 + (6.2 * norm_votes)
        line_opacity = 0.2 + (0.62 * norm_votes)
        edge_color = sample_colorscale(sequential.Turbo, [norm_votes])[0]

        fig.add_trace(go.Scattergeo(
            lon=[s_lon, t_lon],
            lat=[s_lat, t_lat],
            mode="lines",
            line=dict(color=edge_color, width=line_width),
            opacity=line_opacity,
            hoverinfo="skip",
            showlegend=False,
        ))

        fig.add_trace(go.Scattergeo(
            lon=[s_lon],
            lat=[s_lat],
            mode="markers",
            marker=dict(
                size=5 + (3 * norm_votes),
                color=edge_color,
                symbol="circle-open",
                line=dict(width=1, color="white"),
                opacity=0.9,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

        fig.add_trace(go.Scattergeo(
            lon=[t_lon],
            lat=[t_lat],
            mode="markers",
            marker=dict(
                size=8 + (4 * norm_votes),
                color=edge_color,
                symbol="triangle-right",
                line=dict(width=1.1, color="white"),
                opacity=0.95,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

    node_lons = []
    node_lats = []
    node_text = []
    node_sizes = []
    node_colors = []
    node_customdata = []

    for country, stats in node_stats.items():
        if country not in coord_lookup:
            continue

        lat, lon = coord_lookup[country]
        incoming_votes = float(stats["incoming"])
        outgoing_votes = float(stats["outgoing"])
        norm_incoming = incoming_votes / max_incoming
        node_size = 7.0 + (18.0 * np.sqrt(norm_incoming))
        node_color = sample_colorscale(sequential.Turbo, [norm_incoming])[0]

        node_lons.append(lon)
        node_lats.append(lat)
        node_text.append(country)
        node_sizes.append(node_size)
        node_colors.append(node_color)
        node_customdata.append([incoming_votes, outgoing_votes])

    fig.add_trace(go.Scattergeo(
        lon=node_lons,
        lat=node_lats,
        text=node_text,
        customdata=node_customdata,
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=10, color="#111827"),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1.3, color="white"),
            opacity=0.95,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>Votes received: %{customdata[0]:.0f}<br>Votes sent: %{customdata[1]:.0f}<extra></extra>"
        ),
        showlegend=False,
    ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#f7fafc",
        showocean=True,
        oceancolor="#eaf4ff",
        showcountries=True,
        countrycolor="#aebed2",
        showcoastlines=True,
        coastlinecolor="#8aa0ba",
        showframe=False,
        center=dict(lon=35, lat=38),
        projection_scale=1.65,
        lonaxis=dict(range=[-25, 180]),
        lataxis=dict(range=[-45, 75]),
    )

    fig.update_layout(
        title=f"{community_name} vote-weighted community map",
        height=660,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=70, b=10),
        showlegend=False,
        geo=dict(
            showland=True,
            landcolor="#f7fafc",
            showocean=True,
            oceancolor="#eaf4ff",
            showcountries=True,
            countrycolor="#aebed2",
            showcoastlines=True,
            coastlinecolor="#8aa0ba",
            showframe=False,
            center=dict(lon=35, lat=38),
            projection=dict(type="natural earth", scale=1.65),
            lonaxis=dict(range=[-25, 180]),
            lataxis=dict(range=[-45, 75]),
        ),
        annotations=[
            dict(
                x=0.5,
                y=1.03,
                xref="paper",
                yref="paper",
                text="Lines show vote direction from source to target. Circles mark the sender and triangles mark the receiver.",
                showarrow=False,
                font=dict(size=12, color="#4b5563"),
                xanchor="center",
            )
        ],
    )

    return fig


def build_vote_color_toolkit(max_votes: float) -> str:
    if max_votes <= 0:
        max_votes = 1.0

    steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = ["Very low", "Low", "Medium", "High", "Very high"]
    colors = sample_colorscale(sequential.Turbo, steps)

    rows = []
    for label, frac, color in zip(labels, steps, colors):
        vote_value = max_votes * frac
        rows.append(
            f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:8px;'>"
            f"<span style='width:18px; height:18px; border-radius:4px; background:{color}; border:1px solid #d1d5db; display:inline-block;'></span>"
            f"<span><b>{label}</b><br><span style='color:#6b7280; font-size:12px;'>{vote_value:.0f} votes</span></span>"
            f"</div>"
        )

    return (
        "<div style='background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:14px; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>"
        "<div style='font-weight:700; font-size:15px; margin-bottom:8px; color:#111827;'>Color toolkit</div>"
        "<div style='font-size:13px; color:#4b5563; margin-bottom:12px;'>"
        "Brighter / deeper colors mean stronger incoming vote totals. Bigger nodes also mean more incoming votes, thicker lines mean stronger vote links, circles show the source of a vote, and triangles show the receiving country."
        "</div>"
        + "".join(rows)
        + "</div>"
    )


def build_directed_graph_from_edge_rows(edge_rows: pd.DataFrame, member_labels: list[str] | None = None):
    if not NETWORKX_OK:
        return None

    if edge_rows is None or edge_rows.empty:
        return None

    graph = nx.DiGraph()

    if member_labels:
        for label in member_labels:
            graph.add_node(label)

    for _, row in edge_rows.iterrows():
        source = str(row["Source"])
        target = str(row["Target"])
        votes = float(row["Total votes"])
        nvs_score = float(row["NVS (0–12)"])

        graph.add_edge(
            source,
            target,
            weight=votes,
            display_weight=votes,
            nvs_score=nvs_score,
        )

    return graph


def build_community_edge_table(agg: pd.DataFrame, members: list[str], top_n: int = 20):
    if agg is None or agg.empty or not members:
        return pd.DataFrame()

    edge_df = agg[
        agg["src_label"].isin(members)
        & agg["tgt_label"].isin(members)
        & (agg["total_votes"] > 0)
    ].copy()

    if edge_df.empty:
        return pd.DataFrame()

    edge_df = edge_df.sort_values(["total_votes", "nvs_score"], ascending=[False, False]).head(top_n)

    return edge_df[
        ["src_label", "tgt_label", "total_votes", "nvs_score", "years_eligible"]
    ].rename(columns={
        "src_label": "Source",
        "tgt_label": "Target",
        "total_votes": "Total votes",
        "nvs_score": "NVS (0–12)",
        "years_eligible": "Eligible years",
    }).reset_index(drop=True)


def build_directed_graph_from_nvs(matrix_df, min_edge_weight=0.0, weight_scale=1.0):
    """
    Build a directed graph from the NVS matrix.
    Edges are directed and weighted by NVS intensity.
    
    Args:
        matrix_df: NVS score matrix (index and columns are country labels)
        min_edge_weight: drop edges below this threshold
        weight_scale: multiplier for edge width scaling
    
    Returns:
        directed networkx graph
    """
    if not NETWORKX_OK:
        return None
    
    G = nx.DiGraph()
    countries = list(matrix_df.index)
    
    for c in countries:
        G.add_node(c)
    
    # Add directed edges with weights from the NVS matrix
    for src in countries:
        for tgt in countries:
            if src != tgt:
                weight = float(matrix_df.loc[src, tgt])
                if weight >= min_edge_weight:
                    # Scale the weight for visualization
                    G.add_edge(src, tgt, weight=weight, display_weight=weight * weight_scale)
    
    return G


def make_directed_network_figure(G, communities_df, show_labels=True, min_node_size=20, title=None):
    """
    Render a directed network with visible arrowheads, edge width by intensity,
    and node colors by Louvain community.
    
    Args:
        G: directed networkx graph
        communities_df: dataframe with community assignments
        show_labels: whether to show country labels
        min_node_size: minimum node size
    
    Returns:
        plotly figure
    """
    if G is None or G.number_of_nodes() == 0:
        return None
    
    # Build undirected version for layout (spring layout works better on undirected)
    G_undirected = G.to_undirected()
    pos = nx.spring_layout(G_undirected, weight="weight", seed=42, k=1.5, iterations=50)
    
    # Build community lookup
    community_lookup = {}
    for _, row in communities_df.iterrows():
        members = [m.strip() for m in row["Members"].split(",")]
        for m in members:
            community_lookup[m] = row["Community"]
    
    community_names = sorted(communities_df["Community"].unique().tolist())
    community_to_num = {c: i for i, c in enumerate(community_names)}
    
    # Get max weight for normalization
    max_weight = max([data.get("display_weight", 1.0) for _, _, data in G.edges(data=True)], default=1.0)
    if max_weight == 0:
        max_weight = 1.0

    node_in_strength = {node: float(G.in_degree(node, weight="weight")) for node in G.nodes()}
    max_in_strength = max(node_in_strength.values(), default=1.0)
    if max_in_strength <= 0:
        max_in_strength = 1.0
    
    # Prepare edges with directional arrows and color gradient
    edge_traces = []
    arrow_annotations = []
    
    # Color palette: cool to hot based on intensity
    cmap = get_matplotlib_cmap('YlOrRd')  # Yellow → Orange → Red gradient
    
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        weight = data.get("display_weight", 1.0)
        norm_weight = np.clip(weight / max_weight, 0, 1)
        r, g, b, _ = cmap(norm_weight)
        edge_color = f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.8)"

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=0.8 + (2.8 * norm_weight), color=edge_color),
                hoverinfo="none",
                showlegend=False
            )
        )

        arrow_annotations.append(
            dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.2 + (1.8 * norm_weight),
                arrowcolor=edge_color,
                opacity=0.9
            )
        )
    
    # Prepare nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    max_degree = max([d for n, d in G.degree(weight="weight")], default=1.0)
    if max_degree == 0:
        max_degree = 1.0
    
    for node in G.nodes():
        x, y = pos[node]
        comm = community_lookup.get(node, "NA")
        
        # Node size based on incoming vote strength
        out_degree = G.out_degree(node, weight="weight")
        in_degree = G.in_degree(node, weight="weight")
        total_degree = out_degree + in_degree
        size = min_node_size + (np.sqrt(in_degree / max_in_strength) * 34)
        size = max(min_node_size * 0.8, size)
        
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"<b>{node}</b><br>Community: {comm}<br>Incoming: {in_degree:.1f}<br>Outgoing: {out_degree:.1f}")
        node_color.append(community_to_num.get(comm, 0))
        node_size.append(size)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if show_labels else "markers",
        text=list(G.nodes()) if show_labels else [],
        textposition="top center",
        hovertext=node_text,
        hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Turbo",
            line=dict(width=2, color="#ffffff"),
            showscale=False
        ),
        name="Countries"
    )
    
    # Combine all traces
    fig_traces = edge_traces + [node_trace]
    
    fig = go.Figure(data=fig_traces)
    
    # Add arrows as annotations
    fig.update_layout(
        annotations=arrow_annotations,
        title=title or "Directed Voting Network (Normalized Votes) with Communities",
        height=750,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        hovermode="closest"
    )
    
    return fig


# =============================================================================
# DRAFT VISUALISATION GALLERY — BUILDER FUNCTIONS
#
# These five functions build the static/semi-static draft visualisations
# explored for the GD Contest 2026 poster. Each one is intentionally a
# DIFFERENT mathematical lens on the same NVS-normalised voting data, so
# none of them duplicate the Sankey / Sunburst / GeoMap drafts built
# separately. Every builder returns (figure, title, description_markdown)
# so the gallery dispatcher can render a consistent header above each plot.
# =============================================================================

def draft_unrequited_love(agg: pd.DataFrame, order: list[str], top_n: int = 40):
    """
    DRAFT 1 — Unrequited Love: Voting Asymmetry Arc Diagram

    For every country pair (A, B), computes:
        asymmetry(A,B) = NVS(A->B) - NVS(B->A)

    Only the `top_n` pairs with the LARGEST absolute asymmetry are drawn as
    arcs around a circle. Balanced relationships (where both countries give
    roughly equal NVS to each other) are excluded entirely — this diagram
    only shows imbalance, not overall strength.

    Arc colour:  red  = the country at the ARROW END receives MORE than it gives
                 blue = the country at the ARROW END gives MORE than it receives
    Arc width:   magnitude of the asymmetry (|NVS(A->B) - NVS(B->A)|)
    """
    countries = list(order)
    n = len(countries)
    angle = {c: 2 * np.pi * i / n for i, c in enumerate(countries)}
    pos = {c: (np.cos(angle[c]), np.sin(angle[c])) for c in countries}

    # Build symmetric pair table with directed NVS in both directions
    nvs_lookup = {(r["src_label"], r["tgt_label"]): r["nvs_score"] for _, r in agg.iterrows()}

    pairs = []
    seen = set()
    for a in countries:
        for b in countries:
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            ab = nvs_lookup.get((a, b), 0.0)
            ba = nvs_lookup.get((b, a), 0.0)
            asym = ab - ba
            if abs(asym) < 1e-9:
                continue
            pairs.append({"a": a, "b": b, "ab": ab, "ba": ba, "asym": asym})

    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        return None, "Unrequited Love", "No asymmetric pairs found with the current filters."

    pairs_df["abs_asym"] = pairs_df["asym"].abs()
    pairs_df = pairs_df.sort_values("abs_asym", ascending=False).head(top_n)

    max_asym = float(pairs_df["abs_asym"].max()) or 1.0

    fig = go.Figure()

    # Node circle
    node_x = [pos[c][0] for c in countries]
    node_y = [pos[c][1] for c in countries]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=countries, textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=9, color="#374151", line=dict(width=1, color="white")),
        hoverinfo="text", showlegend=False,
    ))

    for _, row in pairs_df.iterrows():
        # The arrow points toward whoever RECEIVES more (the "beloved")
        if row["asym"] > 0:
            giver, receiver = row["a"], row["b"]   # a gives more to b than b gives to a
            color = "rgba(214,96,77,0.75)"          # red: receiver is loved more than they give back
        else:
            giver, receiver = row["b"], row["a"]
            color = "rgba(43,108,176,0.75)"          # blue

        x0, y0 = pos[giver]
        x1, y1 = pos[receiver]
        # Bezier curve pulled toward the centre for a chord-like look
        cx, cy = (x0 + x1) / 2 * 0.25, (y0 + y1) / 2 * 0.25
        t = np.linspace(0, 1, 24)
        bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
        by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1

        width = 0.6 + 5.0 * (row["abs_asym"] / max_asym)

        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="text",
            text=(
                f"{giver} → {receiver} unrequited<br>"
                f"{giver}→{receiver} NVS: {row['ab'] if giver==row['a'] else row['ba']:.2f}<br>"
                f"{receiver}→{giver} NVS: {row['ba'] if giver==row['a'] else row['ab']:.2f}<br>"
                f"Asymmetry: {row['abs_asym']:.2f}"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-1.3, 1.3]),
        yaxis=dict(visible=False, range=[-1.3, 1.3], scaleanchor="x", scaleratio=1),
        height=820, width=820,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )

    title = "Unrequited Love — Voting Asymmetry Arcs"
    desc = (
        "**Metric:** `asymmetry(A,B) = NVS(A→B) − NVS(B→A)`, computed from the mean "
        "Normalised Voting Share in each direction within the selected period.\n\n"
        f"Only the **top {top_n} most imbalanced pairs** are drawn — perfectly "
        "reciprocal relationships are invisible here on purpose. "
        "**Red arcs** end at the country that receives more affection than it gives "
        "back; **blue arcs** end at the country that gives more than it receives. "
        "Arc thickness scales with the size of the imbalance, not the absolute "
        "voting strength."
    )
    return fig, title, desc


def draft_compass_rose(nodes_df: pd.DataFrame, agg: pd.DataFrame, order: list[str]):
    """
    DRAFT 2 — Compass Voting Rose

    Bins every qualifying country by COMPASS BEARING and DISTANCE from the
    geographic centroid of all included countries, then colours each
    angle/distance cell by the mean NVS that countries in that cell receive
    from all other countries. Tests whether voting strength is geographically
    directional (e.g. concentrated to the East/North) or distance-decaying.
    """
    lat_col, lon_col = _find_geo_columns(nodes_df)
    if not lat_col or not lon_col or "label" not in nodes_df.columns:
        return None, "Compass Voting Rose", "Geographic coordinates not found in nodes file."

    coords = nodes_df.dropna(subset=[lat_col, lon_col, "label"]).copy()
    coords = coords[coords["label"].isin(order)]
    if coords.empty:
        return None, "Compass Voting Rose", "No coordinate data for the currently selected countries."

    centroid_lat = coords[lat_col].astype(float).mean()
    centroid_lon = coords[lon_col].astype(float).mean()

    def bearing(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        return (np.degrees(np.arctan2(x, y)) + 360) % 360

    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 6371 * 2 * np.arcsin(np.sqrt(a))

    coords["bearing"]  = coords.apply(
        lambda r: bearing(centroid_lat, centroid_lon, float(r[lat_col]), float(r[lon_col])), axis=1)
    coords["distance"] = coords.apply(
        lambda r: haversine(centroid_lat, centroid_lon, float(r[lat_col]), float(r[lon_col])), axis=1)

    received = agg.groupby("tgt_label")["nvs_score"].mean().rename("mean_nvs_received")
    coords = coords.merge(received, left_on="label", right_index=True, how="left")
    coords["mean_nvs_received"] = coords["mean_nvs_received"].fillna(0)

    n_sectors = 16
    sector_width = 360 / n_sectors
    coords["sector"] = (coords["bearing"] // sector_width).astype(int) % n_sectors

    dist_bins = [0, 500, 1000, 1800, 4000, np.inf]
    dist_labels = ["0-500km", "500-1000km", "1000-1800km", "1800-4000km", "4000km+"]
    coords["dist_band"] = pd.cut(coords["distance"], bins=dist_bins, labels=dist_labels)

    pivot = (
        coords.groupby(["sector", "dist_band"])["mean_nvs_received"]
        .mean().reset_index()
    )

    sector_names = [
        "N","NNE","NE","ENE","E","ESE","SE","SSE",
        "S","SSW","SW","WSW","W","WNW","NW","NNW",
    ]

    fig = go.Figure()
    for band in dist_labels:
        band_rows = pivot[pivot["dist_band"] == band]
        theta = [sector_names[int(s)] for s in band_rows["sector"]]
        r = [float(band_rows.loc[band_rows["sector"] == s, "mean_nvs_received"].values[0])
             if s in band_rows["sector"].values else 0
             for s in band_rows["sector"]]
        fig.add_trace(go.Barpolar(
            r=band_rows["mean_nvs_received"],
            theta=theta,
            name=band,
            hovertemplate="Direction: %{theta}<br>Mean NVS received: %{r:.2f}<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(title="Mean NVS received", showticklabels=True),
            angularaxis=dict(direction="clockwise", rotation=90),
        ),
        height=750, width=850,
        paper_bgcolor="white",
        legend_title_text="Distance band from<br>Europe centroid",
    )

    title = "Compass Voting Rose"
    desc = (
        "**Metric:** each country is binned by **compass bearing** (16 sectors) "
        "and **great-circle distance** (5 bands) from the geographic centroid of "
        "all currently-included countries. Each wedge is coloured/sized by the "
        "**mean NVS received** by countries in that direction/distance cell.\n\n"
        "This tests whether Eurovision voting strength is geographically directional "
        "(e.g. concentrated toward specific compass directions) or whether it decays "
        "with distance — a question not addressed by bloc/community detection alone."
    )
    return fig, title, desc


def draft_lifespan_arcs(yr_agg: pd.DataFrame, threshold: float = 4.0, min_run: int = 3, top_n: int = 25):
    """
    DRAFT 3 — Alliance Lifespan Arcs

    For every directed pair, finds contiguous runs of years where the yearly
    NVS score stayed >= `threshold`. Each qualifying run (length >= min_run)
    becomes a horizontal segment on a year-axis timeline. Pairs are sorted by
    TOTAL qualifying duration, and only the top_n longest-lived alliances are
    shown — this makes DURATION itself the primary visual variable, distinct
    from the Sankey (bloc snapshot) and Sunburst (top-3 supporters) drafts.
    """
    df = yr_agg.copy()
    df["src_label"] = df.get("src_label", df.get("source"))
    df["tgt_label"] = df.get("tgt_label", df.get("target"))
    df["nvs_score_year"] = df["nvs_year"] * 12

    runs = []
    for (src, tgt), g in df.groupby(["src_label", "tgt_label"]):
        g = g.sort_values("year")
        years = g["year"].to_numpy()
        vals  = g["nvs_score_year"].to_numpy()
        above = vals >= threshold

        run_start = None
        for i in range(len(years)):
            if above[i] and run_start is None:
                run_start = years[i]
            if (not above[i] or i == len(years) - 1) and run_start is not None:
                run_end = years[i] if above[i] else years[i - 1]
                if run_end - run_start + 1 >= min_run:
                    seg_vals = vals[(years >= run_start) & (years <= run_end)]
                    runs.append({
                        "pair": f"{src} → {tgt}",
                        "src": src, "tgt": tgt,
                        "start": int(run_start), "end": int(run_end),
                        "duration": int(run_end - run_start + 1),
                        "mean_nvs": float(np.mean(seg_vals)),
                        "stability": float(max(0.0, 1.0 - np.std(seg_vals) / (np.mean(seg_vals) + 1e-6))),
                    })
                run_start = None

    runs_df = pd.DataFrame(runs)
    if runs_df.empty:
        return None, "Alliance Lifespan Arcs", "No alliances met the duration/threshold criteria."

    total_duration = runs_df.groupby("pair")["duration"].sum().sort_values(ascending=False)
    top_pairs = total_duration.head(top_n).index.tolist()
    runs_df = runs_df[runs_df["pair"].isin(top_pairs)]

    pair_order = total_duration.loc[top_pairs].index.tolist()
    y_index = {p: i for i, p in enumerate(pair_order)}

    fig = go.Figure()
    for _, row in runs_df.iterrows():
        y = y_index[row["pair"]]
        color = sample_colorscale(
            sequential.Tealgrn if hasattr(sequential, "Tealgrn") else sequential.Teal,
            [min(max(row["stability"], 0.0), 1.0)]
        )[0]
        fig.add_trace(go.Scatter(
            x=[row["start"], row["end"]],
            y=[y, y],
            mode="lines",
            line=dict(width=10, color=color),
            hovertemplate=(
                f"<b>{row['pair']}</b><br>"
                f"{row['start']}–{row['end']} ({row['duration']} yrs)<br>"
                f"Mean NVS in run: {row['mean_nvs']:.2f}<br>"
                f"Stability: {row['stability']:.2f}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(pair_order))),
            ticktext=pair_order,
            title="Country pair (sorted by total years above threshold)",
            autorange="reversed",
        ),
        xaxis=dict(title="Year", range=[1973, 2027]),
        height=max(500, len(pair_order) * 26 + 150),
        width=1000,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=220, r=40, t=20, b=60),
    )

    title = "Alliance Lifespan Arcs"
    desc = (
        f"**Metric:** for each directed pair, finds contiguous year-runs where "
        f"yearly NVS stayed ≥ **{threshold}/12**, keeping only runs of "
        f"**{min_run}+ consecutive years**. Pairs are ranked by **total qualifying "
        f"duration** and only the top {top_n} longest-lived alliances are shown.\n\n"
        "**Segment colour** encodes stability (darker = more consistent NVS within "
        "that run). Gaps in a row mean the alliance temporarily fell below threshold "
        "— this is the only draft that makes *duration* itself the primary visual "
        "variable."
    )
    return fig, title, desc


def draft_cohesion_streamgraph(yr_agg: pd.DataFrame, communities_df: pd.DataFrame, window: int = 5):
    """
    DRAFT 4 — Bloc Cohesion Streamgraph

    Bloc MEMBERSHIP is fixed (taken from the already-detected communities for
    the current period/filters). For each rolling `window`-year period, this
    recomputes the MEAN MUTUAL AFFINITY *within* each fixed bloc — i.e. how
    tightly-knit the bloc's existing members are voting for each other in
    that window, even though who belongs to the bloc never changes.

    This answers a question snapshots can't: blocs can keep the same members
    but drift from tightly-knit to barely-cohesive (or vice versa) over time.
    """
    if communities_df is None or communities_df.empty:
        return None, "Bloc Cohesion Streamgraph", "No communities detected — adjust the bloc threshold first."

    df = yr_agg.copy()
    df["src_label"] = df.get("src_label", df.get("source"))
    df["tgt_label"] = df.get("tgt_label", df.get("target"))
    df["nvs_score_year"] = df["nvs_year"] * 12

    bloc_members = {
        row["Community"]: [m.strip() for m in str(row["Members"]).split(",") if m.strip()]
        for _, row in communities_df.iterrows()
    }
    # keep only the 5 largest blocs for readability
    bloc_members = dict(sorted(bloc_members.items(), key=lambda kv: -len(kv[1]))[:5])

    years = sorted(df["year"].unique())
    if len(years) < window:
        return None, "Bloc Cohesion Streamgraph", "Not enough years in the selected range for a rolling window."

    rows = []
    for end_idx in range(window - 1, len(years)):
        win_years = years[end_idx - window + 1: end_idx + 1]
        win_df = df[df["year"].isin(win_years)]
        for bloc, members in bloc_members.items():
            sub = win_df[
                win_df["src_label"].isin(members) & win_df["tgt_label"].isin(members)
                & (win_df["src_label"] != win_df["tgt_label"])
            ]
            cohesion = float(sub["nvs_score_year"].mean()) if not sub.empty else 0.0
            rows.append({"year": win_years[-1], "bloc": bloc, "cohesion": cohesion})

    stream_df = pd.DataFrame(rows)
    pivot = stream_df.pivot(index="year", columns="bloc", values="cohesion").fillna(0)

    # Symmetric streamgraph baseline (wiggle = centred stack)
    n_series = pivot.shape[1]
    cumulative = pivot.cumsum(axis=1)
    total = cumulative.iloc[:, -1]
    baseline = -total / 2

    fig = go.Figure()
    palette = px_colors = [
        "#E63946", "#2A9D8F", "#F4A261", "#457B9D", "#8338EC",
    ]
    running_base = baseline.copy()
    for i, bloc in enumerate(pivot.columns):
        top = running_base + pivot[bloc]
        fig.add_trace(go.Scatter(
            x=pivot.index, y=top, mode="lines",
            line=dict(width=0.5, color=palette[i % len(palette)]),
            fill="tonexty" if i > 0 else "tozeroy",
            fillcolor=palette[i % len(palette)],
            opacity=0.85,
            name=bloc,
            hovertemplate=f"{bloc}<br>Year %{{x}}<br>Cohesion (mean NVS): %{{customdata:.2f}}<extra></extra>",
            customdata=pivot[bloc],
        ))
        running_base = top

    fig.update_layout(
        xaxis=dict(title=f"Year (rolling {window}-year window, right-aligned)"),
        yaxis=dict(title="Internal bloc cohesion (stacked, centred)", showticklabels=False),
        height=650, width=1050,
        paper_bgcolor="white", plot_bgcolor="white",
        legend_title_text="Bloc (fixed membership)",
        margin=dict(l=40, r=40, t=20, b=60),
    )

    title = "Bloc Cohesion Streamgraph"
    desc = (
        f"**Metric:** bloc **membership is fixed** (from current community detection). "
        f"For each rolling **{window}-year window**, recomputes the **mean mutual NVS "
        f"within each bloc's existing members only**. Band thickness = cohesion in "
        f"that window.\n\n"
        "Unlike the Sankey (which shows membership migration) or Sunburst (which shows "
        "top supporters), this isolates **how tightly a bloc votes together internally "
        "over time**, even when its membership never changes — a bloc can stay intact "
        "but drift from cohesive to fragmented."
    )
    return fig, title, desc


def draft_genome_barcode(pdata: dict, id2label: dict):
    """
    DRAFT 5 — Eurovision Genome Barcode

    Each country is a horizontal track; each year is one coloured tick:
      white = country absent from the contest that year
      teal  = country participated (active)
      gold  = country WON that year
    Countries are stacked top-to-bottom by total years active. Designed to
    be readable as a poster from across a room (colour bands), while
    rewarding close inspection (exact year-by-year patterns).
    """
    participants_by_year = pdata.get("participants_by_year", {})
    final_standings = pdata.get("final_standings", pd.DataFrame())
    years = pdata.get("years", [])

    if not participants_by_year or not years:
        return None, "Eurovision Genome Barcode", "No participation data available for the current filters."

    winners_by_year = {}
    if not final_standings.empty:
        win_rows = final_standings[final_standings["is_winner"]]
        winners_by_year = dict(zip(win_rows["year"], win_rows["country_id"].astype(str)))

    all_ids = sorted(
        set().union(*participants_by_year.values()) if participants_by_year else set()
    )
    active_count = {
        cid: sum(1 for y in years if cid in participants_by_year.get(y, set()))
        for cid in all_ids
    }
    ordered_ids = sorted(all_ids, key=lambda c: -active_count.get(c, 0))
    ordered_labels = [id2label.get(c, c) for c in ordered_ids]

    z = np.zeros((len(ordered_ids), len(years)))
    for r, cid in enumerate(ordered_ids):
        for c, yr in enumerate(years):
            if cid in participants_by_year.get(yr, set()):
                z[r, c] = 2.0 if winners_by_year.get(yr) == cid else 1.0
            else:
                z[r, c] = 0.0

    colorscale = [
        [0.00, "#ffffff"], [0.33, "#ffffff"],
        [0.34, "#2A9D8F"], [0.66, "#2A9D8F"],
        [0.67, "#F4D35E"], [1.00, "#F4D35E"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z, x=years, y=ordered_labels,
        colorscale=colorscale, zmin=0, zmax=2,
        showscale=False,
        xgap=0.5, ygap=0.5,
        hovertemplate="%{y} — %{x}<extra></extra>",
    ))

    cell_h = max(8, min(16, 700 // max(len(ordered_labels), 1)))
    fig.update_layout(
        height=max(500, len(ordered_labels) * cell_h + 150),
        width=1200,
        xaxis=dict(title="Year", dtick=5),
        yaxis=dict(title="Country (sorted by total years active)", tickfont=dict(size=8)),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=160, r=20, t=20, b=60),
    )

    title = "Eurovision Genome Barcode"
    desc = (
        "**Encoding:** white = absent that year · teal = participated · "
        "gold = **won** that year. Countries are stacked by **total years "
        "active**, so the densest tracks (most consistent participants) sit "
        "at the top.\n\n"
        "This is the most information-dense static image of the set — readable "
        "at a glance as coloured bands from across a room, while individual gold "
        "ticks reward close inspection for exact win years."
    )
    return fig, title, desc


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

with st.sidebar:
    st.header("Controls")

    year_options = list(range(ROOT_START, ROOT_END + 1))

    start_year = st.selectbox(
        "Start year",
        year_options,
        index=0
    )

    end_year = st.selectbox(
        "End year",
        year_options,
        index=len(year_options) - 1
    )

    if start_year > end_year:
        st.error("Start year must be less than or equal to end year.")
        st.stop()

    st.markdown("---")
    st.markdown("**Country filtering**")

    all_country_ids = sorted(
        set(edges[src_col].dropna().astype(str).unique())
        | set(edges[tgt_col].dropna().astype(str).unique())
    )

    all_labels_unfiltered = [
        id2label.get(c, c)
        for c in all_country_ids
    ]

    # Participation is now computed ONLY within the selected start_year/end_year
    # window, so "25+ years" means 25+ years inside this range — not 25+ years
    # across the full 1975-2025 dataset.
    participation_counts = compute_participation_counts(start_year, end_year)

    window_span = end_year - start_year + 1
    max_possible_participation = min(
        window_span,
        int(max(participation_counts.values(), default=1))
    )

    min_participation = st.slider(
        "Minimum participation years",
        min_value=1,
        max_value=max(1, max_possible_participation),
        value=min(5, max(1, max_possible_participation)),
        step=1,
        help=(
            f"Countries with fewer participation years THAN THIS, "
            f"WITHIN {start_year}–{end_year} ({window_span} years total), "
            f"are excluded from all charts and tables."
        )
    )

    filtered_labels = [
        lbl for lbl in all_labels_unfiltered
        if participation_years_for_label(lbl, participation_counts) >= min_participation
    ]

    st.caption(f"{len(filtered_labels)} / {len(all_labels_unfiltered)} countries pass this filter")

    top_n_choice = st.selectbox(
        "Top N countries by NVS strength",
        ["All", 10, 15, 20, 25, 30, 40, 50],
        index=0
    )

    TOP_N = None if top_n_choice == "All" else int(top_n_choice)

    order_mode = st.selectbox(
        "Country order",
        ["strength", "cluster", "alphabetical"],
        index=0
    )

    view_mode = st.selectbox(
        "Main matrix view",
        ["both", "nvs", "raw", "correlation"],
        index=0
    )

    threshold = st.slider(
        "Hide weak NVS cells below",
        min_value=0.0,
        max_value=6.0,
        value=0.0,
        step=0.1
    )

    st.markdown("---")
    st.markdown("**Community detection**")

    community_min_weight = st.slider(
        "Minimum mutual NVS edge for bloc detection",
        min_value=0.0,
        max_value=8.0,
        value=2.0,
        step=0.25
    )

    community_top_rank = st.selectbox(
        "Top 3 communities ranked by",
        ["Size", "Average Internal Weight"],
        index=0,
        help="Controls how the three communities shown on the world map are chosen."
    )

    st.markdown("---")
    st.markdown("**Country drill-down**")

    drilldown_start_year = st.selectbox(
        "Drill-down start year",
        year_options,
        index=year_options.index(start_year)
    )

    drilldown_end_year = st.selectbox(
        "Drill-down end year",
        year_options,
        index=year_options.index(end_year)
    )

    drilldown_country = st.selectbox(
        "Country",
        sorted(filtered_labels),
        index=0
    )

    st.markdown("---")
    st.markdown("**Dynamic comparison**")

    enable_comparison = st.checkbox(
        "Enable period comparison",
        value=True
    )

    if enable_comparison:
        comp_a_start = st.selectbox(
            "Period A start",
            year_options,
            index=0
        )

        comp_a_end = st.selectbox(
            "Period A end",
            year_options,
            index=min(24, len(year_options) - 1)
        )

        comp_b_start = st.selectbox(
            "Period B start",
            year_options,
            index=min(25, len(year_options) - 1)
        )

        comp_b_end = st.selectbox(
            "Period B end",
            year_options,
            index=len(year_options) - 1
        )


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
    order = get_country_order(
        pdata,
        filtered_labels,
        order_mode=order_mode
    )

if TOP_N is not None:
    order = order[:TOP_N]

agg = filter_agg_by_order(pdata["agg"], order)

m_nvs = build_matrix(agg, "nvs_score", order)
m_total = build_matrix(agg, "total_votes", order)
m_years = build_matrix(agg, "years_eligible", order)
m_avg = build_matrix(agg, "raw_avg_per_year", order)

z_nvs = m_nvs.values.astype(float)
z_total = m_total.values.astype(float)
z_years = m_years.values.astype(float)
z_avg = m_avg.values.astype(float)

z_nvs_plot = (
    np.where(z_nvs >= threshold, z_nvs, np.nan)
    if threshold > 0 else z_nvs.copy()
)

z_total_plot = (
    np.where(z_nvs >= threshold, z_total, np.nan)
    if threshold > 0 else z_total.copy()
)

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
        z=z_nvs_plot,
        x=order,
        y=order,
        text=make_text(z_nvs_plot, "{:.1f}"),
        texttemplate="%{text}",
        textfont=dict(size=7, color="#f0faf4"),
        hovertext=hover_nvs,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale=NVS_SCALE,
        zmin=0,
        zmax=12,
        xgap=2,
        ygap=2,
        visible=True,
        name="NVS Matrix",
        colorbar=dict(
            title=dict(text="NVS Score<br>(0–12)", font_size=11),
            thickness=14,
            len=0.5
        )
    ))

if view_mode in ("both", "raw"):
    fig.add_trace(go.Heatmap(
        z=z_total_plot,
        x=order,
        y=order,
        text=make_text(z_total_plot, "{:.0f}"),
        texttemplate="%{text}",
        textfont=dict(size=7, color="#f0faf4"),
        hovertext=hover_nvs,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale=NVS_SCALE,
        zmin=0,
        xgap=2,
        ygap=2,
        visible=(view_mode == "raw"),
        name="Raw Total",
        colorbar=dict(
            title=dict(text="Total Votes", font_size=11),
            thickness=14,
            len=0.5
        )
    ))

if view_mode in ("both", "correlation"):
    fig.add_trace(go.Heatmap(
        z=z_corr,
        x=order,
        y=order,
        text=make_text(z_corr, "{:.2f}"),
        texttemplate="%{text}",
        textfont=dict(size=7, color="#111111"),
        hovertext=hover_corr,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale=CORR_SCALE,
        zmid=0,
        zmin=-1,
        zmax=1,
        xgap=1,
        ygap=1,
        visible=(view_mode == "correlation"),
        name="Correlation",
        colorbar=dict(
            title=dict(text="Correlation", font_size=11),
            thickness=14,
            len=0.5
        )
    ))

if view_mode == "both":
    labels = [tr.name for tr in fig.data]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                xanchor="center",
                y=1.08,
                yanchor="top",
                buttons=[
                    dict(
                        label=l,
                        method="update",
                        args=[{"visible": [name == l for name in labels]}]
                    )
                    for l in labels
                ]
            )
        ]
    )

fig.update_layout(
    title=dict(
        text=(
            f"Eurovision Voting Explorer · {year_label}"
            f"<br><span style='font-size:12px; color:#4b5563;'>"
            f"Order: {order_mode} | Minimum participation: {min_participation}+ years | "
            f"Countries shown: {len(order)}"
            f"</span>"
        ),
        x=0.5,
        xanchor="center"
    ),
    xaxis=dict(
        title="Receives points →",
        type="category",
        categoryorder="array",
        categoryarray=order,
        tickvals=order,
        ticktext=order,
        tickangle=-45,
        tickfont=dict(size=8),
        side="bottom",
        showgrid=False
    ),
    yaxis=dict(
        title="Gives points ←",
        type="category",
        categoryorder="array",
        categoryarray=order,
        tickvals=order,
        ticktext=order,
        autorange="reversed",
        tickfont=dict(size=8),
        showgrid=False
    ),
    height=max(900, len(order) * 22 + 250),
    paper_bgcolor="white",
    plot_bgcolor="white",
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
                (z_a < threshold) & (z_b < threshold)
                if threshold > 0 else np.zeros_like(z_diff, dtype=bool)
            )

            z_diff_plot = np.where(weak_both, np.nan, z_diff)

            diff_hover = [
                [
                    (
                        f"<b>{order[r]}</b> → <b>{order[c]}</b><br>"
                        f"{label_a} NVS: <b>{z_a[r,c]:.2f}</b><br>"
                        f"{label_b} NVS: <b>{z_b[r,c]:.2f}</b><br>"
                        f"Change: <b>{z_diff[r,c]:+.2f}</b>"
                    )
                    if pd.notna(z_diff_plot[r, c]) else ""
                    for c in range(len(order))
                ]
                for r in range(len(order))
            ]

            fig_diff = go.Figure(go.Heatmap(
                z=z_diff_plot,
                x=order,
                y=order,
                hovertext=diff_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                colorscale=DIFF_SCALE,
                zmin=-12,
                zmax=12,
                zmid=0,
                xgap=1.5,
                ygap=1.5,
                colorbar=dict(
                    title=dict(text="Δ NVS<br>B − A", font_size=12),
                    thickness=16,
                    len=0.6,
                    tickvals=[-12, -6, 0, 6, 12],
                    ticktext=["-12", "-6", "0", "+6", "+12"]
                )
            ))

            fig_diff.update_layout(
                title=dict(
                    text=(
                        f"Change in Voting Affinity: {label_b} minus {label_a}"
                        f"<br><span style='font-size:12px; color:#4b5563;'>"
                        f"Red = stronger in later period | Blue = weaker in later period"
                        f"</span>"
                    ),
                    x=0.5,
                    xanchor="center"
                ),
                xaxis=dict(
                    title="Receives points →",
                    type="category",
                    categoryorder="array",
                    categoryarray=order,
                    tickvals=order,
                    ticktext=order,
                    tickangle=-45,
                    tickfont=dict(size=8),
                    side="bottom"
                ),
                yaxis=dict(
                    title="Gives points ←",
                    type="category",
                    categoryorder="array",
                    categoryarray=order,
                    tickvals=order,
                    ticktext=order,
                    autorange="reversed",
                    tickfont=dict(size=8)
                ),
                height=max(850, len(order) * 22 + 230),
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=170, r=120, t=130, b=170)
            )

            st.plotly_chart(fig_diff, use_container_width=True)

            diff_rows = []

            for r, src in enumerate(order):
                for c, tgt in enumerate(order):
                    if src == tgt:
                        continue

                    if pd.isna(z_diff_plot[r, c]):
                        continue

                    diff_rows.append({
                        "Source": src,
                        "Target": tgt,
                        f"NVS {label_a}": z_a[r, c],
                        f"NVS {label_b}": z_b[r, c],
                        "Change": z_diff[r, c],
                        "Direction": (
                            "Strengthened" if z_diff[r, c] > 0
                            else "Weakened" if z_diff[r, c] < 0
                            else "No major change"
                        )
                    })

            diff_df = pd.DataFrame(diff_rows)

            if not diff_df.empty:
                strengthened = (
                    diff_df[diff_df["Change"] > 0]
                    .sort_values("Change", ascending=False)
                    .head(15)
                )

                weakened = (
                    diff_df[diff_df["Change"] < 0]
                    .sort_values("Change", ascending=True)
                    .head(15)
                )

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
        In simple terms: countries are grouped together when they often vote for each other.
        The app checks how strong the two-way voting connection is, removes very weak links,
        and then puts the countries into clusters. If the main method is not available,
        it uses a simpler fallback method.
    </div>
    """,
    unsafe_allow_html=True,
)

if not NETWORKX_OK:
    st.warning("NetworkX is not installed, so community detection is unavailable.")
else:
    communities_df, community_graph = detect_communities_from_nvs(
        m_nvs,
        min_edge_weight=community_min_weight
    )

    if communities_df.empty:
        st.warning("No communities detected. Try lowering the minimum mutual NVS edge threshold.")
    else:
        community_colors = build_community_color_map(communities_df)

        st.dataframe(
            communities_df.round({"Average Internal Weight": 2}),
            use_container_width=True
        )

        fig_community = make_community_network_figure(
            community_graph,
            communities_df
        )

        if fig_community is not None:
            st.plotly_chart(fig_community, use_container_width=True)

        fig_community_map = make_community_world_map_figure(
            nodes,
            communities_df,
        )

        if fig_community_map is not None:
            st.markdown("### Community world map")
            st.caption("The same communities projected onto the countries' geographic locations.")
            st.plotly_chart(fig_community_map, use_container_width=True)
        else:
            st.info("World map view is unavailable because geographic coordinates were not found for the detected communities.")

        st.markdown("---")
        st.markdown("### Country voter drill-down")
        st.caption("Pick a country and year range from the sidebar to see the top 3 incoming voters as directed flows on the world map.")
        st.info(
            f"**Data scope:** all figures below are computed strictly from "
            f"**{ROOT_START}–{ROOT_END}** Eurovision data. Years outside this "
            f"window are not loaded into the app at all, so 'top voters' and "
            f"'pair trend' results never reflect any Eurovision history before "
            f"{ROOT_START} or after {ROOT_END}, even if such records exist in "
            f"the source dataset."
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
                    st.caption(
                        f"Showing votes flowing into {drilldown_country} from "
                        f"{drilldown_start_year} to {drilldown_end_year} "
                        f"(within the app's fixed {ROOT_START}–{ROOT_END} scope)."
                    )
                    st.dataframe(top_voters_df.round({"NVS (0–12)": 2, "Avg/year": 2}), use_container_width=True)

                    fig_country_map = make_directed_country_world_map_figure(
                        nodes,
                        drilldown_country,
                        top_voters_df,
                    )

                    if fig_country_map is not None:
                        st.plotly_chart(fig_country_map, use_container_width=True)
                    else:
                        st.info("Country flow map is unavailable because geographic coordinates were not found for one or more voters.")

                drill_order = get_country_order(
                    drill_period,
                    filtered_labels,
                    order_mode=order_mode
                )

                drill_agg = filter_agg_by_order(drill_period["agg"], drill_order)
                drill_matrix = build_matrix(drill_agg, "nvs_score", drill_order)
                drill_communities_df, drill_community_graph = detect_communities_from_nvs(
                    drill_matrix,
                    min_edge_weight=community_min_weight
                )

                if drill_communities_df.empty:
                    st.info("No communities detected for the selected drill-down years with the current threshold.")
                else:
                    if community_top_rank == "Average Internal Weight":
                        sort_keys = ["Average Internal Weight", "Size"]
                    else:
                        sort_keys = ["Size", "Average Internal Weight"]

                    top_communities_df = (
                        drill_communities_df
                        .sort_values(sort_keys, ascending=[False, False])
                        .head(3)
                        .reset_index(drop=True)
                    )

                    st.markdown("#### Top 3 communities")
                    st.caption(f"Showing the top 3 communities detected from {drilldown_start_year} to {drilldown_end_year}.")
                    st.dataframe(top_communities_df.round({"Average Internal Weight": 2}), use_container_width=True)

                    fig_top_community_map = make_community_world_map_figure(
                        nodes,
                        top_communities_df,
                    )

                    if fig_top_community_map is not None:
                        st.plotly_chart(fig_top_community_map, use_container_width=True)
                    else:
                        st.info("Top-community world map is unavailable because geographic coordinates were not found for the detected communities.")

        st.markdown("---")
        st.markdown("### Community detail view")
        st.caption("Select one community to inspect its directed voting network, strongest vote edges, and world-map flow.")

        sorted_communities_df = communities_df.sort_values(
            ["Size", "Average Internal Weight"],
            ascending=[False, False]
        ).reset_index(drop=True)

        selected_community_name = st.selectbox(
            "Inspect community",
            sorted_communities_df["Community"].tolist(),
            index=0
        )

        selected_community_row = sorted_communities_df[
            sorted_communities_df["Community"] == selected_community_name
        ].iloc[0]

        selected_members = [m.strip() for m in str(selected_community_row["Members"]).split(",") if m.strip()]

        selected_edge_rows = agg[
            agg["src_label"].isin(selected_members)
            & agg["tgt_label"].isin(selected_members)
            & (agg["total_votes"] > 0)
        ].copy()

        if selected_edge_rows.empty:
            st.info(f"No directed vote edges found inside {selected_community_name} for the current filters.")
        else:
            selected_edge_rows = selected_edge_rows.sort_values(
                ["total_votes", "nvs_score"],
                ascending=[False, False]
            )

            directed_edge_table = selected_edge_rows[
                ["src_label", "tgt_label", "total_votes", "nvs_score", "years_eligible"]
            ].rename(columns={
                "src_label": "Source",
                "tgt_label": "Target",
                "total_votes": "Total votes",
                "nvs_score": "NVS (0–12)",
                "years_eligible": "Eligible years",
            }).reset_index(drop=True)

            community_directed_graph = build_directed_graph_from_edge_rows(
                directed_edge_table,
                selected_members,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Size", int(selected_community_row["Size"]))
            c2.metric("Internal edges", int(selected_community_row["Internal Edges"]))
            c3.metric("Avg internal weight", f"{float(selected_community_row['Average Internal Weight']):.2f}")

            st.markdown("#### Strongest directed edges")
            st.caption("Edge thickness in the graph follows raw vote totals for the selected community and period.")
            st.dataframe(
                directed_edge_table.head(15).round({"Total votes": 0, "NVS (0–12)": 2}),
                use_container_width=True,
            )

            fig_directed = make_directed_network_figure(
                community_directed_graph,
                communities_df,
                show_labels=True,
                min_node_size=25,
                title=f"{selected_community_name} directed voting network"
            )

            if fig_directed is not None:
                st.plotly_chart(fig_directed, use_container_width=True)
            else:
                st.info(f"No directed network could be built for {selected_community_name}.")

            fig_flow_map = make_directed_community_vote_map_figure(
                nodes,
                directed_edge_table,
                selected_community_name,
                community_colors.get(selected_community_name, "#64748b"),
                max_edges=20,
            )

            if fig_flow_map is not None:
                st.markdown("#### Community vote map")
                st.caption("The map below plots directed vote flows. Bigger nodes mean more incoming votes, circles mark the sender, and triangles mark the receiver.")
                map_col, toolkit_col = st.columns([4, 1])

                with map_col:
                    st.plotly_chart(fig_flow_map, use_container_width=True)

                with toolkit_col:
                    st.markdown(build_vote_color_toolkit(float(directed_edge_table["Total votes"].max())), unsafe_allow_html=True)
            else:
                st.info(f"No vote-map view available for {selected_community_name} with the current filters.")


# =============================================================================
# GLOBAL BEHAVIOUR TABLE
# =============================================================================

st.subheader("Dynamic relationship classification")

behaviour_df = compute_pair_behaviour(start_year, end_year)

behaviour_df = behaviour_df[
    behaviour_df["Source"].isin(order)
    & behaviour_df["Target"].isin(order)
].copy()

if behaviour_df.empty:
    st.warning("No dynamic relationship data available for the selected period and filters.")
else:
    st.caption("Click a row in any table below to open the pair trend analysis for that alliance.")

    class_filter = st.selectbox(
        "Filter relationship class",
        ["All"] + sorted(behaviour_df["Class"].unique().tolist())
    )

    shown_behaviour = behaviour_df.copy()

    if class_filter != "All":
        shown_behaviour = shown_behaviour[
            shown_behaviour["Class"] == class_filter
        ]

    shown_behaviour = shown_behaviour.sort_values(
        ["Mean NVS", "Stability"],
        ascending=[False, False]
    )

    shown_behaviour_view = shown_behaviour.head(50)

    behaviour_selection = st.dataframe(
        shown_behaviour_view.round({
            "Mean NVS": 2,
            "Std Dev": 2,
            "CV": 2,
            "Trend Slope": 3,
            "Stability": 2,
            "Total Votes": 0
        }),
        on_select="rerun",
        selection_mode="single-row",
        key="behaviour_table",
        use_container_width=True
    )

    update_selected_pair_from_table(
        behaviour_selection,
        shown_behaviour_view,
        "Source",
        "Target",
        "dynamic relationship classification"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Top stable alliances**")
        stable = behaviour_df[
            behaviour_df["Class"] == "Strong stable alliance"
        ].sort_values(
            ["Mean NVS", "Stability"],
            ascending=[False, False]
        ).head(10)

        stable_view = stable[["Source", "Target", "Mean NVS", "Stability", "Eligible Years"]]

        stable_selection = st.dataframe(
            stable_view.round(2),
            on_select="rerun",
            selection_mode="single-row",
            key="stable_alliances_table",
            use_container_width=True
        )

        update_selected_pair_from_table(
            stable_selection,
            stable_view,
            "Source",
            "Target",
            "top stable alliances"
        )

    with col2:
        st.markdown("**Top emerging relationships**")
        emerging = behaviour_df[
            behaviour_df["Class"] == "Emerging relationship"
        ].sort_values("Trend Slope", ascending=False).head(10)

        emerging_view = emerging[["Source", "Target", "Mean NVS", "Trend Slope", "Eligible Years"]]

        emerging_selection = st.dataframe(
            emerging_view.round(2),
            on_select="rerun",
            selection_mode="single-row",
            key="emerging_alliances_table",
            use_container_width=True
        )

        update_selected_pair_from_table(
            emerging_selection,
            emerging_view,
            "Source",
            "Target",
            "top emerging relationships"
        )

    with col3:
        st.markdown("**Top declining relationships**")
        declining = behaviour_df[
            behaviour_df["Class"] == "Declining relationship"
        ].sort_values("Trend Slope", ascending=True).head(10)

        declining_view = declining[["Source", "Target", "Mean NVS", "Trend Slope", "Eligible Years"]]

        declining_selection = st.dataframe(
            declining_view.round(2),
            on_select="rerun",
            selection_mode="single-row",
            key="declining_alliances_table",
            use_container_width=True
        )
        update_selected_pair_from_table(
            declining_selection,
            declining_view,
            "Source",
            "Target",
            "top declining relationships"
        )


# =============================================================================
# TOP DIRECTED PAIRS
# =============================================================================

st.subheader("Top directed pairs in selected period")

top_pairs = (
    agg
    .sort_values("nvs_score", ascending=False)
    [[
        "src_label",
        "tgt_label",
        "nvs_score",
        "total_votes",
        "years_eligible",
        "raw_avg_per_year"
    ]]
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

top_pairs_selection = st.dataframe(
    top_pairs.round({
        "NVS (0–12)": 2,
        "Avg/year": 2
    }),
    on_select="rerun",
    selection_mode="single-row",
    key="top_pairs_table",
    use_container_width=True
)

update_selected_pair_from_table(
    top_pairs_selection,
    top_pairs,
    "Voter",
    "Recipient",
    "top directed pairs"
)


# =============================================================================
# PAIR TREND ANALYSIS
#
# DATA SCOPE NOTE:
# All series, statistics (mean/std/CV/slope/stability), and change-point
# detection below are computed exclusively from `pdata`, which is built by
# `compute_period_data(start_year, end_year)`. Both `start_year` and
# `end_year` are restricted to the [ROOT_START, ROOT_END] = [1975, 2025]
# window earlier in the sidebar. No year outside this fixed app-wide range
# is ever loaded, aggregated, or plotted here — even if the source CSV
# contains older or newer Eurovision contest data.
# =============================================================================

st.subheader("Pair trend analysis")

st.info(
    f"**Data scope:** this analysis only uses Eurovision data from "
    f"**{ROOT_START} to {ROOT_END}** ({ROOT_END - ROOT_START + 1} contest years). "
    f"All statistics below (mean NVS, stability, trend slope, relationship "
    f"classification) describe behaviour strictly within that window — they "
    f"do not reflect any voting history outside {ROOT_START}–{ROOT_END}, "
    f"even if it exists elsewhere in the raw dataset."
)

requested_open = bool(st.session_state.pop("pair_trend_requested", False))

if "show_pair_trend" not in st.session_state:
    st.session_state["show_pair_trend"] = requested_open
elif requested_open:
    st.session_state["show_pair_trend"] = True

show_pair_trend = st.toggle(
    "Show pair trend analysis",
    key="show_pair_trend",
    help="Click a row in one of the alliance tables above to switch this on automatically."
)

if show_pair_trend:
    selected_source = st.session_state.get("selected_pair_source")
    selected_target = st.session_state.get("selected_pair_target")

    if selected_source not in order:
        selected_source = order[0]

    if selected_target not in order:
        selected_target = order[min(1, len(order) - 1)]

    pair_col1, pair_col2 = st.columns(2)

    with pair_col1:
        source_country = st.selectbox(
            "Source country",
            order,
            index=order.index(selected_source)
        )

    with pair_col2:
        target_country = st.selectbox(
            "Target country",
            order,
            index=order.index(selected_target)
        )

    pair_df = pdata["yr_agg"].copy()
    pair_df["src_label"] = pair_df["source"].map(id2label).fillna(pair_df["source"])
    pair_df["tgt_label"] = pair_df["target"].map(id2label).fillna(pair_df["target"])
    final_standing_lookup = build_standing_lookup(pdata.get("final_standings", pd.DataFrame()))

    pair_rows = pair_df[
        (pair_df["src_label"] == source_country)
        & (pair_df["tgt_label"] == target_country)
    ].copy()

    if pair_rows.empty:
        st.warning("No eligible pair-years found for this pair.")
    else:
        pair_rows = pair_rows.sort_values("year")

        status_counts = pair_rows["status"].value_counts()

        # compute years where recipient or sender were not present in the final
        participants_by_year = pdata.get("participants_by_year", {})
        label2id = {v: k for k, v in id2label.items()}
        target_id = label2id.get(target_country, target_country)
        source_id = label2id.get(source_country, source_country)

        years_range = [yr for yr in pdata.get("years", []) if yr >= start_year and yr <= end_year]

        recipient_nonfinal_years = [yr for yr in years_range if str(target_id) not in participants_by_year.get(yr, set())]
        sender_absent_years = [yr for yr in years_range if str(source_id) not in participants_by_year.get(yr, set())]

        # years where recipient qualified but sender gave 0
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
        absent_mask = status_all == "absent"

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
            cp_year = None
            cp_score = 0.0

        relationship_class = classify_relationship(
            mean_v,
            std_v,
            cv_v,
            slope_v,
            stability
        )

        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric("Mean NVS", f"{mean_v:.2f}")
        m2.metric("Std dev", f"{std_v:.2f}")
        m3.metric("CV", f"{cv_v:.2f}")
        m4.metric("Trend slope", f"{slope_v:.3f}")
        m5.metric("Stability", f"{stability:.2f}")

        st.success(f"Relationship classification: **{relationship_class}**")

        if cp_year:
            st.info(
                f"Possible relationship change around **{cp_year}** "
                f"(heuristic score {cp_score:.2f})."
            )


        # full-year chart (all years)
        fig_full = build_full_pair_figure(
            pair_rows,
            source_country,
            target_country,
            standing_lookup=final_standing_lookup,
            highlight_abstained_years=highlight_abstained_years,
        )

        st.caption("Full series across all years")
        st.plotly_chart(fig_full, use_container_width=True)

        # Notes about recipient and sender participation
        if recipient_nonfinal_years:
            st.info(
                f"Recipient did not qualify for the final in: {', '.join(str(y) for y in recipient_nonfinal_years)}"
            )

        if sender_absent_years:
            st.info(
                f"Sender was not present in: {', '.join(str(y) for y in sender_absent_years)}"
            )

        # 5-year segmented chart below
        fig_pair = build_pair_interval_figure(
            pair_rows,
            source_country,
            target_country,
            start_year,
            end_year,
            standing_lookup=final_standing_lookup,
            interval_years=5,
        )

        st.caption("Each colored segment is a 5-year interval. The table below classifies each interval separately.")
        st.plotly_chart(fig_pair, use_container_width=True)

        interval_summary = build_pair_interval_summary(
            pair_rows,
            start_year,
            end_year,
            interval_years=5,
        )

        if interval_summary.empty:
            st.info("No 5-year interval summary available for this pair.")
        else:
            st.dataframe(
                interval_summary.round({
                    "Mean NVS": 2,
                    "Std Dev": 2,
                    "Trend Slope": 3,
                    "Stability": 2,
                }),
                use_container_width=True,
                hide_index=True,
            )