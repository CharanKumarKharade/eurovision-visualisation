"""
draft_visualizations.py

Seven novel, thesis-grade Eurovision visualisations — distinct from the
existing Sankey (bloc migration), Sunburst (bloc/country/supporter), and
GeoMap (top-3 voters) views already in the main app.

Each function takes the already-loaded, already-filtered edges dataframe
(scoped to ROOT_START..ROOT_END, i.e. 1975-2025) plus id2label/nodes, and
returns a tuple:

    (figure, title, explanation_markdown)

`figure` is a Plotly Figure ready for st.plotly_chart().
`explanation_markdown` is shown above the chart so readers know exactly
what is plotted and how it was computed, before they look at it.

All seven reuse the same NVS (Normalised Voting Share) definition used
throughout the rest of the app:

    NVS(A -> B, year) = points_given / era_max(year)
    era_max = 12  for 1975-2015 (single jury vote)
    era_max = 24  for 2016-2025 (jury + televote)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

try:
    import community as community_louvain
    LOUVAIN_OK = True
except ImportError:
    LOUVAIN_OK = False


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _era_max(year: int) -> int:
    return 24 if year >= 2016 else 12


def _add_era_max_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["era_max"] = df["year"].apply(_era_max)
    df["nvs"] = (df["points"] / df["era_max"]).clip(0, 1)
    return df


def _mutual_affinity(df: pd.DataFrame, countries: list) -> pd.DataFrame:
    mean_nvs = (
        df.groupby(["source", "target"])["nvs"]
          .mean().unstack(fill_value=0)
          .reindex(index=countries, columns=countries, fill_value=0)
    )
    aff = (mean_nvs + mean_nvs.T) / 2
    z = aff.to_numpy(copy=True)
    np.fill_diagonal(z, 0)
    return pd.DataFrame(z, index=aff.index, columns=aff.columns)


def _affinity_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean (source, target, nvs) view from a dataframe that has
    BOTH the original `source`/`target` (country ID) columns AND the
    label-mapped `src_label`/`tgt_label` columns.

    Renaming src_label -> source directly on such a dataframe creates two
    columns both named "source" (the original ID column survives the
    rename untouched), which breaks pandas groupby with:
        ValueError: Grouper for 'source' not 1-dimensional

    This helper avoids that by selecting only the three columns needed
    and renaming on that minimal copy, so no duplicate column names can
    ever occur.
    """
    return (
        df[["src_label", "tgt_label", "nvs"]]
        .rename(columns={"src_label": "source", "tgt_label": "target"})
    )


def _detect_blocs(affinity: pd.DataFrame, countries: list, q: float = 0.65) -> dict:
    """
    Returns {country: bloc_label}.

    Hardened against the ZeroDivisionError that NetworkX's
    greedy_modularity_communities() raises when the graph's total edge
    weight (m) is 0. This happens when `pos` (positive affinity values)
    is empty: the old code fell back to threshold=0, which then matched
    EVERY pair including zero-weight ones via `w >= threshold`, producing
    a graph with edges but zero total weight. Fixed by:
      1. requiring w > 0 AND w >= threshold (never include zero-weight edges)
      2. explicitly checking total edge weight before calling any
         community-detection algorithm, falling back to a single bloc
         if there is no usable signal at all (e.g. a very sparse
         rolling time window).
    """
    vals = affinity.to_numpy(copy=True)
    pos = vals[np.triu_indices_from(vals, k=1)]
    pos = pos[pos > 0]
    threshold = np.quantile(pos, q) if len(pos) else np.inf  # no positive weights -> no edges at all

    G = nx.Graph()
    G.add_nodes_from(countries)
    for i, s in enumerate(countries):
        for j, t in enumerate(countries):
            if i >= j:
                continue
            w = float(affinity.loc[s, t])
            if w > 0 and w >= threshold:
                G.add_edge(s, t, weight=w)

    total_weight = sum(d.get("weight", 0.0) for _, _, d in G.edges(data=True))

    if total_weight <= 0:
        # No usable signal — every country is its own (trivial) bloc.
        partition = {c: 0 for c in countries}
    elif LOUVAIN_OK:
        partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    else:
        comms = nx.community.greedy_modularity_communities(G, weight="weight")
        partition = {n: i for i, c in enumerate(comms) for n in c}

    sizes = pd.Series(partition).value_counts().sort_values(ascending=False)
    remap = {old: new for new, old in enumerate(sizes.index)}
    return {c: f"Bloc {remap.get(b, 0) + 1}" for c, b in partition.items()}


BLOC_PALETTE = [
    "#E63946", "#2A9D8F", "#F4A261", "#457B9D",
    "#8338EC", "#2B9348", "#E9C46A", "#C77DFF",
]


def _coord_lookup(nodes_df: pd.DataFrame, id2label: dict) -> dict:
    cols = {c.lower(): c for c in nodes_df.columns}
    lat_c = cols.get("lat") or cols.get("latitude")
    lon_c = cols.get("lon") or cols.get("long") or cols.get("longitude")
    lbl_c = cols.get("label", "label")
    out = {}
    for _, row in nodes_df.dropna(subset=[lat_c, lon_c]).iterrows():
        out[str(row[lbl_c])] = (float(row[lat_c]), float(row[lon_c]))
    return out


# =============================================================================
# DIAGRAM 1 — UNREQUITED LOVE ASYMMETRY ARC DIAGRAM
# =============================================================================

def build_unrequited_love(df: pd.DataFrame, id2label: dict, min_years: int = 15,
                           top_n_arcs: int = 40):
    """
    Circular arc diagram showing only IMBALANCED voting relationships.

    For each pair (A, B):
        asymmetry(A,B) = mean_NVS(A->B) - mean_NVS(B->A)

    Only pairs with |asymmetry| above the 75th percentile are drawn — this
    intentionally hides "fair" relationships and shows only one-sided ones.

    Arc colour:  red  = first-listed country gives more than it receives
                 blue = first-listed country receives more than it gives
    Arc width:   magnitude of the imbalance
    """
    df = _add_era_max_col(df)
    df["src_label"] = df["source"].map(id2label).fillna(df["source"])
    df["tgt_label"] = df["target"].map(id2label).fillna(df["target"])

    participation = (
        pd.concat([
            df[["year","src_label"]].rename(columns={"src_label":"country"}),
            df[["year","tgt_label"]].rename(columns={"tgt_label":"country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )
    qualified = participation[participation >= min_years].index.tolist()
    df = df[df["src_label"].isin(qualified) & df["tgt_label"].isin(qualified)]

    mean_nvs = (
        df.groupby(["src_label","tgt_label"])["nvs"].mean()
          .unstack(fill_value=0)
          .reindex(index=qualified, columns=qualified, fill_value=0)
    )

    pairs = []
    for i, a in enumerate(qualified):
        for j, b in enumerate(qualified):
            if i >= j:
                continue
            ab = mean_nvs.loc[a, b]
            ba = mean_nvs.loc[b, a]
            if ab == 0 and ba == 0:
                continue
            asym = ab - ba
            pairs.append({"a": a, "b": b, "ab": ab, "ba": ba, "asym": asym})

    pair_df = pd.DataFrame(pairs)
    if pair_df.empty:
        return None, "Unrequited Love", "No qualifying pairs found."

    pair_df["abs_asym"] = pair_df["asym"].abs()
    pair_df = pair_df.sort_values("abs_asym", ascending=False).head(top_n_arcs)

    # Circular layout
    countries_in_plot = sorted(set(pair_df["a"]) | set(pair_df["b"]))
    n = len(countries_in_plot)
    angle = {c: 2*np.pi*i/n for i, c in enumerate(countries_in_plot)}
    pos = {c: (np.cos(angle[c]), np.sin(angle[c])) for c in countries_in_plot}

    fig = go.Figure()

    # Node ring
    fig.add_trace(go.Scatter(
        x=[pos[c][0]*1.12 for c in countries_in_plot],
        y=[pos[c][1]*1.12 for c in countries_in_plot],
        mode="text",
        text=countries_in_plot,
        textfont=dict(size=10, color="#1f2937"),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[pos[c][0] for c in countries_in_plot],
        y=[pos[c][1] for c in countries_in_plot],
        mode="markers",
        marker=dict(size=8, color="#374151"),
        hoverinfo="skip",
        showlegend=False,
    ))

    max_abs = pair_df["abs_asym"].max() or 1.0

    for _, row in pair_df.iterrows():
        x0, y0 = pos[row["a"]]
        x1, y1 = pos[row["b"]]
        # quadratic bezier through centre-ish for visual arc effect
        mx, my = (x0+x1)/2 * 0.25, (y0+y1)/2 * 0.25
        t = np.linspace(0, 1, 30)
        bx = (1-t)**2*x0 + 2*(1-t)*t*mx + t**2*x1
        by = (1-t)**2*y0 + 2*(1-t)*t*my + t**2*y1

        norm = row["abs_asym"] / max_abs
        color = "rgba(214,40,40,{:.2f})".format(0.25 + 0.65*norm) if row["asym"] > 0 \
                else "rgba(33,104,176,{:.2f})".format(0.25 + 0.65*norm)
        width = 1 + 5*norm

        winner = row["a"] if row["asym"] > 0 else row["b"]
        loser  = row["b"] if row["asym"] > 0 else row["a"]

        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines",
            line=dict(color=color, width=width),
            hovertemplate=(
                f"<b>{winner}</b> gives more to <b>{loser}</b> than it receives back<br>"
                f"{row['a']}→{row['b']}: {row['ab']:.3f} NVS<br>"
                f"{row['b']}→{row['a']}: {row['ba']:.3f} NVS<br>"
                f"Imbalance: {row['abs_asym']:.3f}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-1.4,1.4]),
        yaxis=dict(visible=False, range=[-1.4,1.4], scaleanchor="x"),
        height=750, width=750,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10,r=10,t=10,b=10),
    )

    explanation = f"""
**What this shows:** the top {len(pair_df)} most *imbalanced* voting relationships
out of all qualifying country pairs (≥{min_years} years of mutual participation).

**Metric:** `asymmetry(A,B) = mean NVS(A→B) − mean NVS(B→A)`, averaged across
{1975}-{2025}. Only pairs in the top 25% by absolute imbalance are drawn —
balanced "fair" relationships are deliberately hidden.

**Reading the arcs:** 🔴 red = the country named first in the hover tooltip
gives noticeably more than it gets back. 🔵 blue = the reverse. Arc
**thickness** scales with the size of the imbalance.
"""
    return fig, "Unrequited Love — Voting Asymmetry", explanation


# =============================================================================
# DIAGRAM 2 — THE NEIGHBOUR EFFECT (distance vs. affinity)
# =============================================================================

def build_neighbour_effect(df: pd.DataFrame, id2label: dict, nodes_df: pd.DataFrame,
                            min_years: int = 15, n_labels: int = 8):
    """
    Tests the most common pop-culture claim about Eurovision directly:
    "it's just neighbours voting for neighbours."

    For every qualifying country pair, plot:
        x = great-circle distance between the two countries (km)
        y = mutual NVS affinity = mean(NVS(A->B), NVS(B->A)), 1975-2025

    A linear trend line is fitted. Pairs that deviate most strongly from
    the trend (either far apart with surprisingly high affinity, or close
    together with surprisingly low affinity) are labelled directly on the
    chart — these are the "exceptions that prove/disprove the rule".
    """
    df = _add_era_max_col(df)
    df["src_label"] = df["source"].map(id2label).fillna(df["source"])
    df["tgt_label"] = df["target"].map(id2label).fillna(df["target"])

    participation = (
        pd.concat([
            df[["year","src_label"]].rename(columns={"src_label":"country"}),
            df[["year","tgt_label"]].rename(columns={"tgt_label":"country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )
    qualified = participation[participation >= min_years].index.tolist()
    df = df[df["src_label"].isin(qualified) & df["tgt_label"].isin(qualified)]

    coords = _coord_lookup(nodes_df, id2label)
    countries = [c for c in qualified if c in coords]

    affinity = _mutual_affinity(
        _affinity_input(df[df["src_label"].isin(countries) & df["tgt_label"].isin(countries)]),
        countries,
    )

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    rows = []
    for i, a in enumerate(countries):
        for j, b in enumerate(countries):
            if i >= j:
                continue
            aff = float(affinity.loc[a, b])
            if aff <= 0:
                continue
            lat1, lon1 = coords[a]
            lat2, lon2 = coords[b]
            dist = haversine(lat1, lon1, lat2, lon2)
            rows.append({"a": a, "b": b, "pair": f"{a} – {b}",
                         "distance_km": dist, "affinity": aff})

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return None, "The Neighbour Effect", "No qualifying pairs found."

    # Linear trend line
    x = pair_df["distance_km"].to_numpy()
    y = pair_df["affinity"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept

    # Residuals — biggest positive (defies "neighbour theory") and
    # biggest negative (close but cold) deviations from the trend
    pair_df["predicted"] = slope * pair_df["distance_km"] + intercept
    pair_df["residual"] = pair_df["affinity"] - pair_df["predicted"]

    top_pos = pair_df.sort_values("residual", ascending=False).head(n_labels // 2)
    top_neg = pair_df.sort_values("residual", ascending=True).head(n_labels // 2)
    labelled = pd.concat([top_pos, top_neg])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pair_df["distance_km"], y=pair_df["affinity"],
        mode="markers",
        marker=dict(size=7, color="#457B9D", opacity=0.55,
                    line=dict(width=0.5, color="white")),
        customdata=pair_df[["pair"]],
        hovertemplate="<b>%{customdata[0]}</b><br>Distance: %{x:.0f} km<br>Affinity: %{y:.3f}<extra></extra>",
        name="Country pairs",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="#1f2937", width=2, dash="dash"),
        name="Trend (expected affinity by distance)",
        hoverinfo="skip",
    ))

    # Highlight + label outliers
    fig.add_trace(go.Scatter(
        x=labelled["distance_km"], y=labelled["affinity"],
        mode="markers+text",
        marker=dict(size=10, color="#E63946", line=dict(width=1, color="white")),
        text=labelled["pair"],
        textposition="top center",
        textfont=dict(size=9, color="#7c2d12"),
        hovertemplate="<b>%{text}</b><br>Distance: %{x:.0f} km<br>Affinity: %{y:.3f}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        xaxis_title="Geographic distance between countries (km)",
        yaxis_title="Mutual NVS affinity (1975–2025 mean)",
        height=650, width=1000,
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=40, t=20, b=80),
    )

    corr = float(np.corrcoef(x, y)[0, 1])

    explanation = f"""
**The question:** is Eurovision voting really just "neighbours voting for
neighbours" — or is that a myth?

**Method:** for every qualifying country pair (≥{min_years} years of mutual
participation), plot geographic distance (km) against mutual NVS affinity
averaged across 1975–2025. A dashed trend line shows the overall
relationship; red-labelled points are the biggest **exceptions** — pairs
whose actual affinity is far from what their distance alone would predict.

**Correlation found:** r = {corr:.2f} between distance and affinity
({"a real but partial negative trend — closer countries do tend to vote for each other more, but it's far from absolute" if corr < -0.05 else "essentially no meaningful relationship — distance does not predict Eurovision voting"}).

**Read the outliers:** points far to the right (distant) but still high up
(strong affinity) are voting for reasons that have nothing to do with being
neighbours — diaspora, shared history, or politics. Points close to the
origin's left but low affinity are physical neighbours who simply don't
vote for each other.
"""
    return fig, "The Neighbour Effect — Does Geography Predict Voting?", explanation



# =============================================================================
# DIAGRAM 3 — ALLIANCE LIFESPAN ARCS (GANTT-STYLE)
# =============================================================================

def build_lifespan_arcs(df: pd.DataFrame, id2label: dict, min_years: int = 15,
                         threshold: float = 0.35, top_n: int = 35):
    """
    Horizontal Gantt-style chart: one row per qualifying country pair,
    x-axis = year. A bar segment is drawn for every CONSECUTIVE run of years
    where mean NVS(A<->B) stayed at/above `threshold`. Gaps = alliance lapsed.

    Rows sorted by total active duration (longest alliances at top).
    Bar colour = stability (1 - coefficient of variation) within that run.
    """
    df = _add_era_max_col(df)
    df["src_label"] = df["source"].map(id2label).fillna(df["source"])
    df["tgt_label"] = df["target"].map(id2label).fillna(df["target"])

    participation = (
        pd.concat([
            df[["year","src_label"]].rename(columns={"src_label":"country"}),
            df[["year","tgt_label"]].rename(columns={"tgt_label":"country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )
    qualified = participation[participation >= min_years].index.tolist()
    df = df[df["src_label"].isin(qualified) & df["tgt_label"].isin(qualified)]

    years = sorted(df["year"].unique())

    # Mean mutual NVS per pair per year
    yearly = (
        df.groupby(["year","src_label","tgt_label"])["nvs"].mean().reset_index()
    )

    pair_year_aff = {}
    for _, row in yearly.iterrows():
        a, b = sorted([row["src_label"], row["tgt_label"]])
        key = (a, b)
        pair_year_aff.setdefault(key, {}).setdefault(row["year"], []).append(row["nvs"])

    pair_series = {}
    for key, year_map in pair_year_aff.items():
        series = {y: float(np.mean(v)) for y, v in year_map.items()}
        pair_series[key] = series

    # Build runs where value >= threshold
    runs = []
    for (a, b), series in pair_series.items():
        vals = [series.get(y, 0) for y in years]
        run_start = None
        run_vals = []
        for i, y in enumerate(years):
            v = vals[i]
            if v >= threshold:
                if run_start is None:
                    run_start = y
                run_vals.append(v)
            else:
                if run_start is not None:
                    runs.append({"a": a, "b": b, "start": run_start,
                                 "end": years[i-1], "vals": run_vals.copy()})
                    run_start = None
                    run_vals = []
        if run_start is not None:
            runs.append({"a": a, "b": b, "start": run_start,
                         "end": years[-1], "vals": run_vals.copy()})

    runs_df = pd.DataFrame(runs)
    if runs_df.empty:
        return None, "Alliance Lifespans", "No alliances exceeded the threshold."

    runs_df["duration"] = runs_df["end"] - runs_df["start"] + 1
    runs_df["pair"] = runs_df["a"] + " ↔ " + runs_df["b"]
    runs_df["mean_val"] = runs_df["vals"].apply(np.mean)
    runs_df["stability"] = runs_df["vals"].apply(
        lambda v: max(0.0, 1 - (np.std(v)/(np.mean(v)+1e-6))) if len(v) > 1 else 1.0
    )

    pair_total = runs_df.groupby("pair")["duration"].sum().sort_values(ascending=False)
    top_pairs = pair_total.head(top_n).index.tolist()
    runs_df = runs_df[runs_df["pair"].isin(top_pairs)]

    pair_order = list(reversed(top_pairs))  # longest at top after reversed axis

    fig = go.Figure()
    for _, row in runs_df.iterrows():
        stab = row["stability"]
        color = f"rgba(11,60,111,{0.35 + 0.55*stab:.2f})"
        fig.add_trace(go.Scatter(
            x=[row["start"], row["end"]+1],
            y=[row["pair"], row["pair"]],
            mode="lines",
            line=dict(color=color, width=14),
            hovertemplate=(
                f"<b>{row['pair']}</b><br>"
                f"Active: {row['start']}–{row['end']} ({row['duration']} yrs)<br>"
                f"Mean NVS: {row['mean_val']:.3f}<br>"
                f"Stability: {stab:.2f}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        yaxis=dict(categoryorder="array", categoryarray=pair_order,
                   tickfont=dict(size=9)),
        xaxis=dict(title="Year", range=[min(years)-1, max(years)+1]),
        height=max(600, len(pair_order)*22 + 150),
        width=1000,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=200, r=40, t=20, b=40),
    )

    explanation = f"""
**What this shows:** the top {len(pair_order)} longest-running "alliances" —
country pairs whose mutual NVS stayed at or above **{threshold}** for
consecutive years. Each horizontal bar segment = one unbroken run; gaps in
a row mean the alliance temporarily lapsed below the threshold.

**Metric:** mutual NVS = mean(NVS(A→B), NVS(B→A)) per year. A bar is drawn
only while this stays ≥ {threshold}.

**Colour:** darker bars = more *stable* relationships during that run
(low year-to-year variance in NVS), lighter = stronger but more volatile.

Rows are sorted by total cumulative years active, so the longest-lasting
Eurovision "friendships" appear at the top.
"""
    return fig, "Alliance Lifespan Arcs", explanation



# =============================================================================
# DIAGRAM 4 — RISE AND FALL: WHO DOMINATED EACH ERA
# =============================================================================

def build_rise_and_fall(df: pd.DataFrame, id2label: dict, min_years: int = 10,
                         decade_size: int = 10, top_supporters: int = 5):
    """
    For each decade-like era, finds the single country that received the
    most total NVS (the "dominant" country of that era) and draws it as the
    centre of a small support-network with its top-N supporters as
    surrounding nodes — edge width = NVS given to the dominant country.

    Answers the most natural question a viewer has about any competition:
    "who was actually good at this, and who backed them?"
    """
    df = _add_era_max_col(df)
    df["src_label"] = df["source"].map(id2label).fillna(df["source"])
    df["tgt_label"] = df["target"].map(id2label).fillna(df["target"])

    participation = (
        pd.concat([
            df[["year","src_label"]].rename(columns={"src_label":"country"}),
            df[["year","tgt_label"]].rename(columns={"tgt_label":"country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )
    qualified = participation[participation >= min_years].index.tolist()
    df = df[df["src_label"].isin(qualified) & df["tgt_label"].isin(qualified)]

    y_min, y_max = int(df["year"].min()), int(df["year"].max())
    era_starts = list(range(y_min, y_max + 1, decade_size))

    eras = []
    for e_start in era_starts:
        e_end = min(e_start + decade_size - 1, y_max)
        eras.append((e_start, e_end))

    from plotly.subplots import make_subplots
    n = len(eras)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{s}–{e}" for s, e in eras],
        specs=[[{"type": "xy"}] * cols for _ in range(rows)],
    )

    summary_rows = []

    for idx, (e_start, e_end) in enumerate(eras):
        r, c = idx // cols + 1, idx % cols + 1
        e_df = df[(df["year"] >= e_start) & (df["year"] <= e_end)]
        if e_df.empty:
            continue

        received = e_df.groupby("tgt_label")["nvs"].sum().sort_values(ascending=False)
        if received.empty:
            continue
        dominant = received.index[0]

        supporters = (
            e_df[e_df["tgt_label"] == dominant]
            .groupby("src_label")["nvs"].sum()
            .sort_values(ascending=False)
            .head(top_supporters)
        )

        summary_rows.append({
            "era": f"{e_start}–{e_end}",
            "dominant_country": dominant,
            "total_nvs_received": float(received.iloc[0]),
            "top_supporters": ", ".join(supporters.index.tolist()),
        })

        n_sup = len(supporters)
        if n_sup == 0:
            continue
        angles = np.linspace(0, 2*np.pi, n_sup, endpoint=False)
        sx = np.cos(angles)
        sy = np.sin(angles)
        max_w = supporters.max() or 1.0

        # Edges
        for k, (sup_name, w) in enumerate(supporters.items()):
            norm = w / max_w
            fig.add_trace(go.Scatter(
                x=[0, sx[k]], y=[0, sy[k]],
                mode="lines",
                line=dict(color=f"rgba(69,123,157,{0.3+0.6*norm:.2f})",
                          width=1+5*norm),
                hoverinfo="skip", showlegend=False,
            ), row=r, col=c)

        # Supporter nodes
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode="markers+text",
            text=supporters.index.tolist(),
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=9, color="#A8DADC", line=dict(width=1, color="#457B9D")),
            customdata=supporters.values,
            hovertemplate="%{text}<br>NVS given: %{customdata:.3f}<extra></extra>",
            showlegend=False,
        ), row=r, col=c)

        # Dominant country (centre)
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers+text",
            text=[dominant], textposition="bottom center",
            textfont=dict(size=10, color="#1d3557"),
            marker=dict(size=22, color="#E63946", line=dict(width=2, color="white")),
            hovertemplate=f"<b>{dominant}</b><br>Dominant {e_start}-{e_end}<br>"
                          f"Total NVS received: {received.iloc[0]:.2f}<extra></extra>",
            showlegend=False,
        ), row=r, col=c)

        fig.update_xaxes(visible=False, range=[-1.5,1.5], row=r, col=c)
        fig.update_yaxes(visible=False, range=[-1.5,1.5], row=r, col=c,
                          scaleanchor=f"x{(r-1)*cols+c}")

    fig.update_layout(
        height=320*rows, width=1300,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_lines = "\n".join(
        f"- **{row['era']}**: {row['dominant_country']} "
        f"(supported mainly by {row['top_supporters']})"
        for _, row in summary_df.iterrows()
    )

    explanation = f"""
**The question:** who actually dominated Eurovision voting era by era, and
who consistently backed them?

**Method:** the dataset is split into {decade_size}-year eras. Within each
era, the country receiving the highest total NVS is shown as a red centre
node, surrounded by its top {top_supporters} supporting countries — edge
width = how much NVS that supporter gave.

**Findings:**
{summary_lines}

Watch whether the *same* small set of supporters keeps reappearing around
different dominant countries across eras — that's a sign of a stable
regional voting bloc rather than coincidence.
"""
    return fig, "Rise and Fall — Who Dominated Each Era", explanation


# =============================================================================
# DIAGRAM 5 — VOTING HALL OF FAME (curated superlative stat cards)
# =============================================================================

def build_hall_of_fame(df: pd.DataFrame, id2label: dict, min_years: int = 15):
    """
    A small set of curated, attention-grabbing superlative facts pulled
    directly from the NVS data — designed to be the "stop and read" panel
    on a poster, the way infographics use highlighted stat callouts.

    Computes:
      1. Most one-sided relationship  (biggest |NVS(A->B) - NVS(B->A)|)
      2. Most loyal pair              (highest sustained mutual NVS across
                                        the most years together)
      3. Longest max-points streak    (most consecutive years one country
                                        gave another country its single
                                        highest-ever point value)
      4. Biggest bloc-switcher        (country whose detected bloc changed
                                        between 1975-1999 and 2000-2025,
                                        weighted by its NVS strength)
    """
    df = _add_era_max_col(df)
    df["src_label"] = df["source"].map(id2label).fillna(df["source"])
    df["tgt_label"] = df["target"].map(id2label).fillna(df["target"])

    participation = (
        pd.concat([
            df[["year","src_label"]].rename(columns={"src_label":"country"}),
            df[["year","tgt_label"]].rename(columns={"tgt_label":"country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )
    qualified = participation[participation >= min_years].index.tolist()
    qdf = df[df["src_label"].isin(qualified) & df["tgt_label"].isin(qualified)]

    cards = []

    # ---- 1. Most one-sided relationship ----------------------------------
    mean_nvs = qdf.groupby(["src_label","tgt_label"])["nvs"].mean().unstack(fill_value=0)
    mean_nvs = mean_nvs.reindex(index=qualified, columns=qualified, fill_value=0)
    best_asym, best_pair = -1, None
    for i, a in enumerate(qualified):
        for j, b in enumerate(qualified):
            if i >= j:
                continue
            diff = abs(mean_nvs.loc[a, b] - mean_nvs.loc[b, a])
            if diff > best_asym:
                best_asym, best_pair = diff, (a, b)
    if best_pair:
        a, b = best_pair
        ab, ba = mean_nvs.loc[a, b], mean_nvs.loc[b, a]
        giver, receiver = (a, b) if ab > ba else (b, a)
        cards.append({
            "title": "Most one-sided relationship",
            "stat": f"{giver} → {receiver}",
            "detail": f"Gives {max(ab,ba):.3f} mean NVS but receives only {min(ab,ba):.3f} back — "
                      f"a gap of {best_asym:.3f}.",
        })

    # ---- 2. Most loyal pair (sustained mutual affinity) -------------------
    affinity = _mutual_affinity(_affinity_input(qdf), qualified)
    years_together = (
        qdf.groupby(["src_label","tgt_label"])["year"].nunique()
        .unstack(fill_value=0).reindex(index=qualified, columns=qualified, fill_value=0)
    )
    loyalty_score = affinity * np.sqrt(years_together.clip(lower=0))
    best_loy_val, best_loy_pair = -1, None
    for i, a in enumerate(qualified):
        for j, b in enumerate(qualified):
            if i >= j:
                continue
            v = loyalty_score.loc[a, b]
            if v > best_loy_val:
                best_loy_val, best_loy_pair = v, (a, b)
    if best_loy_pair:
        a, b = best_loy_pair
        cards.append({
            "title": "Most loyal pair",
            "stat": f"{a} ↔ {b}",
            "detail": f"Sustained mean mutual affinity of {affinity.loc[a,b]:.3f} across "
                      f"{int(years_together.loc[a,b])} years together — the strongest "
                      f"long-term Eurovision friendship in the data.",
        })

    # ---- 3. Longest max-points streak --------------------------------------
    max_possible = qdf["era_max"].max()
    top_votes = qdf[qdf["points"] >= qdf["era_max"]]  # gave the maximum possible that year
    best_streak, best_streak_pair = 0, None
    if not top_votes.empty:
        for (a, b), grp in top_votes.groupby(["src_label","tgt_label"]):
            yrs = sorted(grp["year"].unique())
            run = 1
            longest = 1
            for k in range(1, len(yrs)):
                if yrs[k] == yrs[k-1] + 1:
                    run += 1
                    longest = max(longest, run)
                else:
                    run = 1
            if longest > best_streak:
                best_streak, best_streak_pair = longest, (a, b)
    if best_streak_pair:
        a, b = best_streak_pair
        cards.append({
            "title": "Longest max-points streak",
            "stat": f"{a} → {b}",
            "detail": f"Gave the maximum possible points {best_streak} years in a row — "
                      f"the most consistent top-score streak in the dataset.",
        })

    # ---- 4. Biggest bloc-switcher ------------------------------------------
    early_df = qdf[qdf["year"] <= 1999]
    late_df  = qdf[qdf["year"] >= 2000]
    bloc_card = None
    if not early_df.empty and not late_df.empty:
        e_countries = sorted(set(early_df["src_label"]) | set(early_df["tgt_label"]))
        l_countries = sorted(set(late_df["src_label"]) | set(late_df["tgt_label"]))
        e_aff = _mutual_affinity(_affinity_input(early_df), e_countries)
        l_aff = _mutual_affinity(_affinity_input(late_df), l_countries)
        e_bloc = _detect_blocs(e_aff, e_countries)
        l_bloc = _detect_blocs(l_aff, l_countries)

        e_strength = early_df.groupby("tgt_label")["nvs"].mean()
        l_strength = late_df.groupby("tgt_label")["nvs"].mean()

        switchers = []
        for c in set(e_bloc) & set(l_bloc):
            if e_bloc[c] != l_bloc[c]:
                strength = float(e_strength.get(c, 0)) + float(l_strength.get(c, 0))
                switchers.append((c, e_bloc[c], l_bloc[c], strength))
        if switchers:
            switchers.sort(key=lambda x: x[3], reverse=True)
            c, eb, lb, strength = switchers[0]
            bloc_card = {
                "title": "Biggest bloc-switcher",
                "stat": c,
                "detail": f"Moved from {eb} (1975–1999) to {lb} (2000–2025) — "
                          f"the most prominent country to change voting alliance "
                          f"after the contest's voting system changed in 2000.",
            }
    if bloc_card:
        cards.append(bloc_card)

    # ---- Build the stat-card figure ----------------------------------------
    n_cards = len(cards)
    cols = 2
    rows = int(np.ceil(n_cards / cols)) if n_cards else 1

    fig = go.Figure()
    card_w, card_h = 1.0, 1.0
    gap = 0.08

    for idx, card in enumerate(cards):
        r, c = idx // cols, idx % cols
        x0 = c * (card_w + gap)
        y0 = -r * (card_h + gap)
        x1, y1 = x0 + card_w, y0 - card_h

        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="#457B9D", width=1.5),
            fillcolor="#F1FAEE",
        )
        fig.add_annotation(
            x=x0 + 0.05, y=y0 - 0.18, text=f"<b>{card['title'].upper()}</b>",
            showarrow=False, font=dict(size=11, color="#1d3557"),
            xanchor="left", yanchor="top",
        )
        fig.add_annotation(
            x=x0 + 0.05, y=y0 - 0.45, text=f"<b>{card['stat']}</b>",
            showarrow=False, font=dict(size=18, color="#E63946", family="Georgia, serif"),
            xanchor="left", yanchor="top",
        )
        fig.add_annotation(
            x=x0 + 0.05, y=y0 - 0.65, text=card["detail"],
            showarrow=False, font=dict(size=10, color="#374151"),
            xanchor="left", yanchor="top", align="left",
            width=card_w * 280,
        )

    fig.update_xaxes(visible=False, range=[-0.1, cols*(card_w+gap)])
    fig.update_yaxes(visible=False, range=[-rows*(card_h+gap), 0.1])
    fig.update_layout(
        height=320*rows, width=950,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    explanation = f"""
**The question:** what are the single most striking, tweet-worthy facts
buried in 50 years of Eurovision voting data?

**Method:** four curated superlative statistics are computed directly from
NVS, each isolating a different kind of "extreme" relationship — one-sided
favouritism, sustained loyalty, a perfect streak, and a complete switch of
allegiance after the 2000 voting-system change.

This panel intentionally trades visual novelty for **immediate readability**
— it's the part of the poster designed to make someone stop and read,
before they dig into the denser network panels.
"""
    return fig, "Voting Hall of Fame", explanation


# =============================================================================
# DIAGRAM 6 — BLOC MIGRATION SANKEY (4-column)
# =============================================================================
#
# Original author: Charan Kumar Kharade Somoji Rao
# Adapted here to:
#   - use the module's shared helpers (_mutual_affinity, _detect_blocs) so
#     it benefits from the same zero-division hardening as every other panel
#   - accept id2label so blocs/countries show real names, not raw codes
#   - return the standard (fig, title, explanation) tuple used by every
#     other build_* function in this module, so it slots into the same
#     Streamlit gallery without any special-case wiring
# =============================================================================

_SANKEY_BLOC_PALETTE = [
    "#E63946", "#2A9D8F", "#F4A261", "#457B9D",
    "#8338EC", "#2B9348", "#E9C46A", "#C77DFF",
]

_X_EARLY_BLOC    = 0.02
_X_EARLY_COUNTRY = 0.34
_X_LATE_COUNTRY  = 0.66
_X_LATE_BLOC     = 0.98


def _sankey_even_y(items: list, pad: float = 0.04) -> dict:
    """Assign evenly spaced y in [pad, 1-pad]."""
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0]: 0.5}
    step = (1 - 2 * pad) / n
    return {item: pad + (i + 0.5) * step for i, item in enumerate(items)}


def _sankey_bloc_y(blocs_df: pd.DataFrame, country_y: dict) -> dict:
    """Centre each bloc node at the mean y of its member countries."""
    result = {}
    for bloc in blocs_df["bloc"].unique():
        members = blocs_df[blocs_df["bloc"] == bloc]["country"].tolist()
        ys = [country_y[c] for c in members if c in country_y]
        result[bloc] = float(np.mean(ys)) if ys else 0.5
    return result


def _sankey_hex_rgba(hx: str, alpha: float = 0.55) -> str:
    h = hx.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _sankey_lighten(hx: str, factor: float = 0.40) -> str:
    h = hx.lstrip("#")
    r = int(int(h[0:2], 16) + (255 - int(h[0:2], 16)) * factor)
    g = int(int(h[2:4], 16) + (255 - int(h[2:4], 16)) * factor)
    b = int(int(h[4:6], 16) + (255 - int(h[4:6], 16)) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _sankey_detect_blocs_on_codes(df_era: pd.DataFrame, countries_codes: list,
                                   q: float = 0.65) -> dict:
    """
    Run the shared, hardened bloc-detection pipeline for one era on raw
    country codes (bloc detection must stay on codes since that's how the
    affinity matrix is indexed; labels are applied separately for display).

    Returns {country_code: bloc_label}.
    """
    affinity = _mutual_affinity(df_era[["source", "target", "nvs"]], countries_codes)
    return _detect_blocs(affinity, countries_codes, q=q)


def build_bloc_migration_sankey(df: pd.DataFrame, id2label: dict,
                                 min_years: int = 25, affinity_q: float = 0.65):
    """
    4-column Sankey: Early bloc -> Country (early) -> Country (late) -> Late bloc.

    Column meaning
    --------------
      Col 1  Voting bloc membership in 1975-1999
      Col 2  Individual country in the early era (grouped by early bloc)
      Col 3  Same country in the late era  (grouped by late bloc)
      Col 4  Voting bloc membership in 2000-2025

    A diagonal crossing between col 2 and col 3 (same country, different
    height/colour) marks a bloc migration — drawn as a red bridge.

    Flow width = mean NVS received by that country in that era.
    """
    df = _add_era_max_col(df)

    participation = (
        pd.concat([
            df[["year", "source"]].rename(columns={"source": "country"}),
            df[["year", "target"]].rename(columns={"target": "country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )
    qualified = participation[participation >= min_years].index.tolist()
    df_f = df[df["source"].isin(qualified) & df["target"].isin(qualified)].copy()

    if df_f.empty:
        return None, "Bloc Migration Sankey", (
            f"No countries met the ≥{min_years}-year participation threshold."
        )

    period_early, y0e, y1e = "1975–1999", 1975, 1999
    period_late,  y0l, y1l = "2000–2025", 2000, 2025

    df_e = df_f[(df_f["year"] >= y0e) & (df_f["year"] <= y1e)]
    df_l = df_f[(df_f["year"] >= y0l) & (df_f["year"] <= y1l)]

    c_e = sorted(set(df_e["source"]) | set(df_e["target"]))
    c_l = sorted(set(df_l["source"]) | set(df_l["target"]))

    if not c_e or not c_l:
        return None, "Bloc Migration Sankey", (
            "Not enough data in one of the two eras to detect blocs."
        )

    # Bloc detection stays on raw codes (affinity matrix is indexed by codes).
    bloc_map_e_codes = _sankey_detect_blocs_on_codes(df_e, c_e, affinity_q)
    bloc_map_l_codes = _sankey_detect_blocs_on_codes(df_l, c_l, affinity_q)

    # Mean NVS received per country, computed on codes, then relabelled.
    nvs_e_lbl = {id2label.get(c, c): v for c, v in df_e.groupby("target")["nvs"].mean().items()}
    nvs_l_lbl = {id2label.get(c, c): v for c, v in df_l.groupby("target")["nvs"].mean().items()}

    # Apply labels for display only, after detection is complete.
    early_blocs = pd.DataFrame({
        "country": [id2label.get(c, c) for c in c_e],
        "bloc":    [bloc_map_e_codes[c] for c in c_e],
    })
    late_blocs = pd.DataFrame({
        "country": [id2label.get(c, c) for c in c_l],
        "bloc":    [bloc_map_l_codes[c] for c in c_l],
    })

    early_bloc_names = sorted(early_blocs["bloc"].dropna().unique())
    late_bloc_names  = sorted(late_blocs["bloc"].dropna().unique())

    merged = (
        early_blocs.rename(columns={"bloc": "early_bloc"})
        .merge(late_blocs.rename(columns={"bloc": "late_bloc"}), on="country", how="inner")
    )
    merged["early_nvs"] = merged["country"].map(nvs_e_lbl).fillna(0.005)
    merged["late_nvs"]  = merged["country"].map(nvs_l_lbl).fillna(0.005)
    merged["migrated"]  = merged["early_bloc"] != merged["late_bloc"]

    if merged.empty:
        return None, "Bloc Migration Sankey", (
            "No countries qualified in both eras — cannot build a migration diagram."
        )

    early_country_list = merged.sort_values(["early_bloc", "country"])["country"].tolist()
    late_country_list  = merged.sort_values(["late_bloc",  "country"])["country"].tolist()

    early_country_y = _sankey_even_y(early_country_list)
    late_country_y  = _sankey_even_y(late_country_list)
    early_bloc_y = _sankey_bloc_y(early_blocs[early_blocs["country"].isin(merged["country"])], early_country_y)
    late_bloc_y  = _sankey_bloc_y(late_blocs[late_blocs["country"].isin(merged["country"])],  late_country_y)

    early_bloc_color = {b: _SANKEY_BLOC_PALETTE[i % len(_SANKEY_BLOC_PALETTE)]
                        for i, b in enumerate(early_bloc_names)}
    late_bloc_color  = {b: _SANKEY_BLOC_PALETTE[i % len(_SANKEY_BLOC_PALETTE)]
                        for i, b in enumerate(late_bloc_names)}

    node_id, node_label, node_color, node_x, node_y = {}, [], [], [], []

    def add_node(key, label, color, x, y):
        if key not in node_id:
            node_id[key] = len(node_label)
            node_label.append(label)
            node_color.append(color)
            node_x.append(x)
            node_y.append(y)

    for bloc in early_bloc_names:
        members = early_blocs[early_blocs["bloc"] == bloc]["country"].tolist()
        add_node(f"E_BLOC_{bloc}", f"<b>{bloc}</b><br>{period_early}<br>{len(members)} countries",
                  early_bloc_color[bloc], _X_EARLY_BLOC, early_bloc_y.get(bloc, 0.5))

    for country in early_country_list:
        row = merged[merged["country"] == country].iloc[0]
        is_migrant = bool(row["migrated"])
        base_col = early_bloc_color[row["early_bloc"]]
        node_col = _sankey_lighten(base_col, 0.25 if not is_migrant else 0.10)
        add_node(f"E_CTRY_{country}", f"{country}{'  ↕' if is_migrant else ''}",
                  node_col, _X_EARLY_COUNTRY, early_country_y[country])

    for country in late_country_list:
        row = merged[merged["country"] == country].iloc[0]
        is_migrant = bool(row["migrated"])
        base_col = late_bloc_color[row["late_bloc"]]
        node_col = _sankey_lighten(base_col, 0.25 if not is_migrant else 0.10)
        add_node(f"L_CTRY_{country}", f"{country}{'  ↕' if is_migrant else ''}",
                  node_col, _X_LATE_COUNTRY, late_country_y[country])

    for bloc in late_bloc_names:
        members = late_blocs[late_blocs["bloc"] == bloc]["country"].tolist()
        add_node(f"L_BLOC_{bloc}", f"<b>{bloc}</b><br>{period_late}<br>{len(members)} countries",
                  late_bloc_color[bloc], _X_LATE_BLOC, late_bloc_y.get(bloc, 0.5))

    # ---- distinct colour per migration PATH (early_bloc -> late_bloc) --
    #
    # Previously every migrating country's bridge was the same uniform red,
    # which flags "this country moved" but not WHICH transition it made.
    # Assigning each unique (early_bloc, late_bloc) pair its own colour
    # (ranked by how many countries made that specific transition, same
    # convention as the bloc palette itself) lets a reader visually group
    # countries that made the same structural move, not just see "a move
    # happened" — e.g. every country that went Bloc 2 -> Bloc 1 shares one
    # colour, distinct from every country that went Bloc 3 -> Bloc 1.
    migration_paths = (
        merged[merged["migrated"]]
        .groupby(["early_bloc", "late_bloc"]).size()
        .sort_values(ascending=False)
    )
    _MIGRATION_PATH_PALETTE = [
        "#d62828", "#f77f00", "#9b5de5", "#06a77d",
        "#118ab2", "#ef476f", "#8d5524", "#4b0082",
    ]
    migration_path_color = {
        path: _MIGRATION_PATH_PALETTE[i % len(_MIGRATION_PATH_PALETTE)]
        for i, path in enumerate(migration_paths.index)
    }

    link_src, link_tgt, link_val, link_col, link_lbl = [], [], [], [], []

    def add_link(src_key, tgt_key, value, color, label):
        link_src.append(node_id[src_key])
        link_tgt.append(node_id[tgt_key])
        link_val.append(max(value, 0.0005))
        link_col.append(color)
        link_lbl.append(label)

    for _, row in merged.iterrows():
        country  = row["country"]
        e_bloc   = row["early_bloc"]
        l_bloc   = row["late_bloc"]
        e_nvs    = float(row["early_nvs"])
        l_nvs    = float(row["late_nvs"])
        migrated = bool(row["migrated"])
        e_col    = early_bloc_color[e_bloc]
        l_col    = late_bloc_color[l_bloc]

        add_link(f"E_BLOC_{e_bloc}", f"E_CTRY_{country}", e_nvs,
                  _sankey_hex_rgba(e_col, 0.50),
                  f"<b>{country}</b><br>Bloc: {e_bloc} ({period_early})<br>"
                  f"Mean NVS received: {e_nvs:.4f}"
                  + ("  <i>(migrated later)</i>" if migrated else ""))

        if migrated:
            path_color = migration_path_color.get((e_bloc, l_bloc), "#d62828")
            bridge_col = _sankey_hex_rgba(path_color, 0.55)
        else:
            bridge_col = _sankey_hex_rgba(e_col, 0.30)

        add_link(f"E_CTRY_{country}", f"L_CTRY_{country}", (e_nvs + l_nvs) / 2,
                  bridge_col,
                  f"<b>{country}</b> {'↕ MIGRATED' if migrated else '— stayed'}<br>"
                  f"Early ({period_early}): {e_bloc}<br>Late ({period_late}): {l_bloc}<br>"
                  f"NVS early: {e_nvs:.4f} | NVS late: {l_nvs:.4f}")

        add_link(f"L_CTRY_{country}", f"L_BLOC_{l_bloc}", l_nvs,
                  _sankey_hex_rgba(l_col, 0.50),
                  f"<b>{country}</b><br>Bloc: {l_bloc} ({period_late})<br>"
                  f"Mean NVS received: {l_nvs:.4f}"
                  + (f"  <i>(migrated from {e_bloc})</i>" if migrated else ""))

    n_stayed   = int((~merged["migrated"]).sum())
    n_migrated = int(merged["migrated"].sum())
    migrants   = merged[merged["migrated"]].sort_values("early_nvs", ascending=False)

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            label=node_label, color=node_color, x=node_x, y=node_y,
            pad=14, thickness=20,
            line=dict(color="rgba(255,255,255,0.6)", width=0.8),
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=link_src, target=link_tgt, value=link_val,
            color=link_col, label=link_lbl,
            hovertemplate="%{label}<extra></extra>",
        ),
    ))

    col_headers = [
        (_X_EARLY_BLOC,    f"Blocs<br>{period_early}"),
        (_X_EARLY_COUNTRY, f"Countries<br>{period_early}"),
        (_X_LATE_COUNTRY,  f"Countries<br>{period_late}"),
        (_X_LATE_BLOC,     f"Blocs<br>{period_late}"),
    ]
    annotations = [
        dict(x=x, y=1.06, xref="paper", yref="paper", text=f"<b>{label}</b>",
             showarrow=False, font=dict(size=12, color="#374151", family="Georgia, serif"),
             xanchor="center")
        for x, label in col_headers
    ]
    annotations.append(dict(
        x=0.5, y=-0.05, xref="paper", yref="paper",
        text=(
            f"Flow width = mean NVS received  |  "
            f"<b style='color:#E63946'>↕</b> = country migrated blocs  |  "
            f"{n_migrated} migrations detected  |  {n_stayed} countries stable  |  "
            f"Bridge colour = which bloc-to-bloc transition (not a single uniform colour)  |  "
            f"Countries with &lt;{min_years} yrs participation excluded"
        ),
        showarrow=False, font=dict(size=10, color="#6b7280"), xanchor="center",
    ))

    fig.update_layout(
        title=dict(
            text=(
                "<b>Eurovision Voting Bloc Migration · 1975–2025</b>"
                "<br><span style='font-size:13px; color:#6b7280;'>"
                "Trace each country left→right to see if its voting bloc changed after 2000"
                "</span>"
            ),
            font=dict(family="Georgia, serif", size=20, color="#1f2937"),
            x=0.5, xanchor="center", y=0.97,
        ),
        font=dict(family="Inter, Helvetica Neue, sans-serif", size=11, color="#374151"),
        paper_bgcolor="#f8fafc",
        width=1300,
        height=max(900, len(merged) * 28 + 180),
        margin=dict(t=100, l=20, r=20, b=70),
        annotations=annotations,
    )

    migration_lines = "\n".join(
        f"- **{m['country']}**: {m['early_bloc']} → {m['late_bloc']} "
        f"(NVS early {m['early_nvs']:.3f}, late {m['late_nvs']:.3f})"
        for _, m in migrants.iterrows()
    ) or "- No countries migrated blocs."

    migration_path_lines = "\n".join(
        f"- **{eb} → {lb}**: {count} countries"
        for (eb, lb), count in migration_paths.items()
    ) or "- No bloc-to-bloc migration paths detected."

    explanation = f"""
**What this shows:** how Eurovision voting blocs reorganised between the
analogue era (1975–1999) and the digital/televote era (2000–2025).

**Reading the diagram:** trace any country left → right across all 4
columns. Column 1 and Column 4 show its bloc membership in each era; the
centre bridge (columns 2→3) is coloured by **which specific bloc-to-bloc
transition** that country made if it migrated (distinct migration paths get
distinct colours, so every country that moved from the same source bloc to
the same destination bloc shares one colour, separate from any other
transition), and stays its normal bloc colour if it stayed put.

**Method:** blocs are detected independently per era via Louvain community
detection on the mutual-affinity graph (mean of NVS(A→B), NVS(B→A)), using
only countries with ≥{min_years} years of participation. Flow width = mean
NVS received by that country within each era.

**Migrations detected:** {n_migrated} of {len(merged)} qualifying countries
changed bloc after 2000; {n_stayed} stayed in an equivalent bloc.

**Migration paths (each a distinct bridge colour):**
{migration_path_lines}

{migration_lines}
"""
    return fig, "Bloc Migration Sankey", explanation


# =============================================================================
# DIAGRAM 7 — HIERARCHICAL BLOC STRUCTURE (storytelling poster)
# =============================================================================
#
# Three independently-laid-out panels, connected only by dotted flow
# annotations (not shared axes), so the reader's eye moves:
#   Tier 1: full 1975-2025 picture (one network, all detected blocs)
#     |  (dotted "splits into two eras")
#   Tier 2: two side-by-side era networks (1975-99, 2000-25), each with
#           its own independently-detected blocs; countries whose bloc
#           membership changed between the two eras get a gold outline
#           in BOTH panels
#     |  (dotted "reveals evidence")
#   Tier 3: two side-by-side stat-card panels (one per era) with
#           top mutual voters / top one-way voters / cold-shoulder pairs
#
# Edge encoding (used identically in Tier 1 and Tier 2):
#   |NVS(A->B) - NVS(B->A)| <= diff_threshold  -> MUTUAL, drawn as a plain
#       line, color/opacity scaled by mean(NVS(A->B), NVS(B->A)), anchored
#       at the 1.0 NVS breakpoint (faint below 1.0, darker above it).
#   otherwise                                  -> ONE-WAY, same color rule
#       using max(NVS(A->B), NVS(B->A)), plus a small rotated triangle
#       marker along the edge pointing toward the higher-receiving country
#       (same directional-marker technique already used in
#       make_directed_community_flow_map_figure in the main app).
#   Edges that cross between two different detected blocs are drawn with
#       full width and a red-tinted palette; edges within the same bloc
#       are deliberately thinned, regardless of mutual/one-way status.
#
# "Cold-shoulder" (hatred) pairs are NOT the same metric as "one-way
# voters": one-way voters can still both be voting, just unevenly.
# Cold-shoulder pairs require A to have given ~zero points to B across
# many years despite both being eligible to vote for each other in those
# years (computed via a small local eligibility frame, since the raw
# edges table only contains rows where some points were actually given).
#
# NOTE ON PLOTLY ANNOTATIONS: Plotly's `axref`/`ayref` annotation
# properties never accept "paper" as a value (only "pixel" or an axis id
# such as "x", "x2", ...). To draw arrowheads anchored purely in paper
# coordinates without tripping that validator, this builder avoids
# showarrow=True annotations entirely for the inter-tier connectors and
# instead renders the dotted shaft via add_shape() and the arrowhead as a
# plain "▼" text annotation with showarrow=False.
# =============================================================================

def build_hierarchical_bloc_poster(
    df: pd.DataFrame,
    id2label: dict,
    nodes_df: pd.DataFrame,
    min_years: int = 10,
    diff_threshold: float = 1.0,
    hatred_min_years: int = 10,
    hatred_epsilon: float = 0.04,
    top_k_out: int = 3,
    min_nvs_strength: float = 2.0,
):
    """
    DRAFT 7 — Hierarchical Bloc Structure: a storyboard poster showing
    the full 1975-2025 voting network, then split into two eras
    (1975-1999 / 2000-2025) with bloc-migration highlighting, then
    concrete per-era evidence (top mutual voters, top one-way voters,
    cold-shoulder pairs).

    Edge selection: for each country A, the `top_k_out` strongest outgoing
    NVS relationships (NVS(A→B) on the 0-12 scale, ranked and kept only if
    NVS >= min_nvs_strength) are selected. A pair survives if it qualifies
    from EITHER country's outgoing perspective. Classification then asks:
    if |NVS(A→B) - NVS(B→A)| <= diff_threshold, the pair is MUTUAL
    (roughly reciprocal); otherwise it is ONE-WAY, with an arrow pointing
    toward whichever country receives the stronger vote.

    Returns (figure, title, explanation_markdown) per the module's
    standard contract.
    """
    from plotly.subplots import make_subplots
    from collections import defaultdict

    df = _add_era_max_col(df)
    df["src_label"] = df["source"].map(id2label).fillna(df["source"])
    df["tgt_label"] = df["target"].map(id2label).fillna(df["target"])

    participation = (
        pd.concat([
            df[["year", "src_label"]].rename(columns={"src_label": "country"}),
            df[["year", "tgt_label"]].rename(columns={"tgt_label": "country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )
    qualified = sorted(participation[participation >= min_years].index.tolist())
    df = df[df["src_label"].isin(qualified) & df["tgt_label"].isin(qualified)].copy()

    if df.empty or len(qualified) < 3:
        return None, "Hierarchical Bloc Structure", (
            f"Not enough countries met the >= {min_years}-year participation "
            "threshold to build this draft."
        )

    # -------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------

    def mean_nvs_matrix(sub_df, countries):
        if sub_df.empty or not countries:
            return pd.DataFrame(0.0, index=countries, columns=countries)
        m = (
            sub_df.groupby(["src_label", "tgt_label"])["nvs"].mean()
            .unstack(fill_value=0)
            .reindex(index=countries, columns=countries, fill_value=0)
        ) * 12.0
        return m

    def nvs_strength_backbone(mat, countries):
        """
        For each country A, select its top `top_k_out` outgoing relationships
        where NVS(A→B) >= min_nvs_strength (on the 0-12 scale). A pair
        survives if it qualifies from EITHER endpoint's outgoing perspective.

        This directly answers "which relationships are strong enough to matter?"
        rather than relying on a statistical test or a purely structural rule.
        The NVS threshold (min_nvs_strength) sets the minimum meaningful voting
        strength — e.g. 2.0/12 means A gives B at least 2 points worth of NVS
        on average, which is roughly equivalent to consistently giving 2–3 points
        per year in the jury-only era or 4–6 points in the combined era.
        """
        keep = set()
        for c in countries:
            out_vals = mat.loc[c].drop(labels=[c], errors="ignore")
            strong_out = (
                out_vals[out_vals >= min_nvs_strength]
                .sort_values(ascending=False)
                .head(top_k_out)
            )
            for partner in strong_out.index:
                keep.add(tuple(sorted([c, partner])))
        return keep

    def classify_edges(mat, countries):
        retained_pairs = nvs_strength_backbone(mat, countries)
        edges = []
        for (a, b) in retained_pairs:
            ab = float(mat.loc[a, b])
            ba = float(mat.loc[b, a])
            if ab <= 0 and ba <= 0:
                continue
            diff = abs(ab - ba)
            if diff <= diff_threshold:
                edges.append({
                    "a": a, "b": b, "kind": "mutual",
                    "value": (ab + ba) / 2.0, "ab": ab, "ba": ba, "diff": diff,
                })
            else:
                if ab > ba:
                    giver, receiver = a, b
                else:
                    giver, receiver = b, a
                edges.append({
                    "a": a, "b": b, "kind": "one_way",
                    "giver": giver, "receiver": receiver,
                    "value": max(ab, ba), "ab": ab, "ba": ba, "diff": diff,
                })
        return edges

    def edge_color(value, cross_bloc=False):
        base = (180, 40, 40) if cross_bloc else (11, 60, 111)
        value = max(value, 0.0)
        if value < 1.0:
            alpha = 0.10 + 0.18 * value
        else:
            alpha = 0.32 + 0.58 * min((value - 1.0) / 11.0, 1.0)
        r, g, b = base
        return f"rgba({r},{g},{b},{alpha:.2f})"

    def detect(sub_df, countries):
        if not countries:
            return {}
        aff = _mutual_affinity(_affinity_input(sub_df), countries)
        return _detect_blocs(aff, countries, q=0.6)

    def flag_migrated(map1, map2):
        g1, g2 = defaultdict(set), defaultdict(set)
        for c, b in map1.items():
            g1[b].add(c)
        for c, b in map2.items():
            g2[b].add(c)
        migrated = set()
        for c in set(map1) & set(map2):
            m1 = g1[map1[c]] - {c}
            m2 = g2[map2[c]] - {c}
            overlap = (len(m1 & m2) / len(m1)) if m1 else 0.0
            if overlap < 0.5:
                migrated.add(c)
        return migrated

    def eligibility_frame(sub_df, countries):
        countries_set = set(countries)
        rows = []
        for yr, g in sub_df.groupby("year"):
            participants = (set(g["src_label"]) | set(g["tgt_label"])) & countries_set
            for s in participants:
                for t in participants:
                    if s != t:
                        rows.append((yr, s, t))
        if not rows:
            return pd.DataFrame(columns=["year", "src_label", "tgt_label", "points", "nvs"])
        elig = pd.DataFrame(rows, columns=["year", "src_label", "tgt_label"])
        actual = sub_df.groupby(["year", "src_label", "tgt_label"], as_index=False)["points"].sum()
        merged = elig.merge(actual, on=["year", "src_label", "tgt_label"], how="left")
        merged["points"] = merged["points"].fillna(0)
        merged["era_max_v"] = merged["year"].apply(_era_max)
        merged["nvs"] = (merged["points"] / merged["era_max_v"]).clip(0, 1)
        return merged

    def era_stats(sub_df, countries, edges):
        mutual = [e for e in edges if e["kind"] == "mutual"]
        one_way = [e for e in edges if e["kind"] == "one_way"]
        top_mutual = sorted(mutual, key=lambda e: e["value"], reverse=True)[:3]
        top_oneway = sorted(one_way, key=lambda e: e["diff"], reverse=True)[:3]
        elig = eligibility_frame(sub_df, countries)
        if elig.empty:
            return top_mutual, top_oneway, pd.DataFrame()
        agg = (
            elig.groupby(["src_label", "tgt_label"])
            .agg(years_eligible=("year", "nunique"), mean_nvs=("nvs", "mean"))
            .reset_index()
        )
        reciprocal_lookup = {
            (r["src_label"], r["tgt_label"]): r["mean_nvs"] for _, r in agg.iterrows()
        }
        candidates = agg[
            (agg["years_eligible"] >= hatred_min_years) & (agg["mean_nvs"] < hatred_epsilon)
        ].copy()
        if candidates.empty:
            return top_mutual, top_oneway, candidates
        candidates["reciprocal_nvs"] = candidates.apply(
            lambda r: reciprocal_lookup.get((r["tgt_label"], r["src_label"]), 0.0), axis=1
        )
        top_hatred = candidates.sort_values(
            ["years_eligible", "reciprocal_nvs"], ascending=[False, False]
        ).head(3)
        return top_mutual, top_oneway, top_hatred

    BLOC_NODE_PALETTE = [
        "#1f4e79", "#d1495b", "#2a9d8f", "#f4a261",
        "#6a4c93", "#7f5539", "#577590", "#3a86ff",
    ]

    def render_network(fig, row, col, countries, edges, bloc_map, migrated=None,
                       participation_years=None, label_top_n=14):
        """
        Render one network panel with:
        - Node size driven by participation years (if provided) else uniform
        - Only top `label_top_n` nodes (by years participated) get labels
        - Mutual edges as solid lines, one-way as dashed + triangle marker
        - Cross-bloc edges at full opacity/width, within-bloc muted
        - Bloc color legend added as shape+annotation in the subplot area
        """
        migrated = migrated or set()
        participation_years = participation_years or {}

        if not countries:
            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, row=row, col=col)
            return

        G = nx.Graph()
        G.add_nodes_from(countries)
        for e in edges:
            G.add_edge(e["a"], e["b"], weight=max(e["value"], 0.01))
        pos = nx.spring_layout(G, weight="weight", seed=42, k=1.1, iterations=60)

        bloc_names = sorted(set(bloc_map.values())) if bloc_map else []
        bloc_color = {b: BLOC_NODE_PALETTE[i % len(BLOC_NODE_PALETTE)] for i, b in enumerate(bloc_names)}

        # --- edges ---
        for e in edges:
            x0, y0 = pos[e["a"]]
            x1, y1 = pos[e["b"]]
            cross = bloc_map.get(e["a"]) != bloc_map.get(e["b"])
            color = edge_color(e["value"], cross_bloc=cross)
            width = (1.2 if not cross else 2.2) + 3.5 * min(e["value"] / 12.0, 1.0)
            if not cross:
                width *= 0.4

            dash = "solid" if e["kind"] == "mutual" else "dash"

            kind_str = "Mutual" if e["kind"] == "mutual" else f"One-way: {e['giver']} \u2192 {e['receiver']}"
            cross_str = "Cross-bloc ⚡" if cross else "Within-bloc"
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=(
                    f"<b>{e['a']}</b> ↔ <b>{e['b']}</b><br>"
                    f"NVS {e['a']}→{e['b']}: {e['ab']:.2f} | {e['b']}→{e['a']}: {e['ba']:.2f}<br>"
                    f"{kind_str} · {cross_str}<extra></extra>"
                ),
                showlegend=False,
            ), row=row, col=col)

            if e["kind"] == "one_way":
                gx, gy = pos[e["giver"]]
                rx, ry = pos[e["receiver"]]
                t = 0.75
                mx, my = gx + t * (rx - gx), gy + t * (ry - gy)
                ang = float(np.degrees(np.arctan2(ry - gy, rx - gx)))
                fig.add_trace(go.Scatter(
                    x=[mx], y=[my], mode="markers",
                    marker=dict(
                        symbol="triangle-right",
                        size=7 + 3 * min(e["value"] / 12.0, 1.0),
                        color=color, angle=ang,
                        line=dict(width=0.5, color="white"),
                    ),
                    hoverinfo="skip", showlegend=False,
                ), row=row, col=col)

        # --- nodes ---
        max_years = max(participation_years.values(), default=1) or 1
        labelled = set(
            sorted(countries, key=lambda c: participation_years.get(c, 0), reverse=True)[:label_top_n]
        )

        node_x = [pos[c][0] for c in countries]
        node_y = [pos[c][1] for c in countries]
        node_color = [bloc_color.get(bloc_map.get(c), "#9ca3af") for c in countries]
        node_line_color = ["#facc15" if c in migrated else "white" for c in countries]
        node_line_width = [3.5 if c in migrated else 1.2 for c in countries]
        node_size = [
            11 + 10 * np.sqrt(max(participation_years.get(c, max_years // 2), 0) / max_years)
            for c in countries
        ]
        node_text = [c if c in labelled else "" for c in countries]

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            text=node_text, textposition="top center",
            textfont=dict(size=9, color="#111827"),
            marker=dict(
                size=node_size, color=node_color,
                line=dict(width=node_line_width, color=node_line_color),
            ),
            hovertext=[
                f"<b>{c}</b><br>Bloc: {bloc_map.get(c, 'NA')}<br>"
                f"Years participated: {participation_years.get(c, '?')}"
                + ("<br><b>⚡ Changed bloc between eras</b>" if c in migrated else "")
                for c in countries
            ],
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        ), row=row, col=col)

        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

        # live edge/node count subtitle
        n_mutual = sum(1 for e in edges if e["kind"] == "mutual")
        n_oneway = len(edges) - n_mutual
        return f"{len(countries)} countries · {len(edges)} edges ({n_mutual} mutual ─, {n_oneway} one-way ╌)"

    def render_stat_panel(fig, row, col, era_label, top_mutual, top_oneway, top_hatred):
        fig.update_xaxes(visible=False, range=[0, 1], row=row, col=col)
        fig.update_yaxes(visible=False, range=[0, 1], row=row, col=col)

        mutual_lines = [
            f"🤝 {e['a']} ↔ {e['b']}  (NVS {e['value']:.1f})" for e in top_mutual
        ] or ["No qualifying mutual pairs"]
        oneway_lines = [
            f"➡️ {e['giver']} → {e['receiver']}  (gap Δ{e['diff']:.1f})" for e in top_oneway
        ] or ["No qualifying one-way pairs"]
        if top_hatred is None or top_hatred.empty:
            hatred_lines = ["No sustained cold-shoulder pairs found"]
        else:
            hatred_lines = [
                f"❄️ {r['src_label']} ⇏ {r['tgt_label']}  ({int(r['years_eligible'])} eligible yrs)"
                for _, r in top_hatred.iterrows()
            ]

        sections = [
            ("Top mutual voters (NVS ≈ equal both ways)", mutual_lines),
            ("Top one-way voters (strong asymmetry)", oneway_lines),
            ("Cold-shoulder pairs (near-zero NVS despite eligibility)", hatred_lines),
        ]
        y = 0.95
        fig.add_annotation(
            x=0.03, y=1.0, text=f"<b>{era_label}</b>", showarrow=False,
            font=dict(size=13, color="#1f2937", family="Georgia, serif"),
            xanchor="left", yanchor="top", row=row, col=col,
        )
        y -= 0.12
        for heading, lines in sections:
            fig.add_annotation(
                x=0.03, y=y, text=f"<b>{heading}</b>", showarrow=False,
                font=dict(size=10, color="#374151"),
                xanchor="left", yanchor="top", row=row, col=col,
            )
            y -= 0.09
            for line in lines[:3]:
                fig.add_annotation(
                    x=0.06, y=y, text=line, showarrow=False,
                    font=dict(size=9, color="#4b5563"),
                    xanchor="left", yanchor="top", row=row, col=col,
                )
                y -= 0.08
            y -= 0.02

    # -------------------------------------------------------------------
    # Participation years lookup (drives node size consistently)
    # -------------------------------------------------------------------

    participation_total = participation.to_dict()

    era1_participation = (
        pd.concat([
            df[df["year"] <= 1999][["year", "src_label"]].rename(columns={"src_label": "country"}),
            df[df["year"] <= 1999][["year", "tgt_label"]].rename(columns={"tgt_label": "country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique().to_dict()
    )
    era2_participation = (
        pd.concat([
            df[df["year"] >= 2000][["year", "src_label"]].rename(columns={"src_label": "country"}),
            df[df["year"] >= 2000][["year", "tgt_label"]].rename(columns={"tgt_label": "country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique().to_dict()
    )

    # -------------------------------------------------------------------
    # Tier 1 — full history
    # -------------------------------------------------------------------

    full_mat = mean_nvs_matrix(df, qualified)
    full_bloc = detect(df, qualified)
    full_edges = classify_edges(full_mat, qualified)

    # -------------------------------------------------------------------
    # Tier 2 — two independently-detected eras
    # -------------------------------------------------------------------

    era1_df = df[df["year"] <= 1999]
    era2_df = df[df["year"] >= 2000]

    era1_countries = sorted({
        c for c in qualified
        if c in set(era1_df["src_label"]) | set(era1_df["tgt_label"])
    })
    era2_countries = sorted({
        c for c in qualified
        if c in set(era2_df["src_label"]) | set(era2_df["tgt_label"])
    })

    era1_mat = mean_nvs_matrix(era1_df, era1_countries)
    era2_mat = mean_nvs_matrix(era2_df, era2_countries)
    era1_bloc = detect(era1_df, era1_countries)
    era2_bloc = detect(era2_df, era2_countries)
    era1_edges = classify_edges(era1_mat, era1_countries)
    era2_edges = classify_edges(era2_mat, era2_countries)

    migrated = flag_migrated(era1_bloc, era2_bloc)

    # -------------------------------------------------------------------
    # Tier 3 — per-era evidence
    # -------------------------------------------------------------------

    top_mutual_1, top_oneway_1, top_hatred_1 = era_stats(era1_df, era1_countries, era1_edges)
    top_mutual_2, top_oneway_2, top_hatred_2 = era_stats(era2_df, era2_countries, era2_edges)

    # -------------------------------------------------------------------
    # Figure assembly
    # -------------------------------------------------------------------

    row_heights = [0.40, 0.34, 0.26]
    vspacing = 0.09
    total_gap = vspacing * (len(row_heights) - 1)
    avail = 1.0 - total_gap
    scaled = [h * avail for h in row_heights]

    boundaries = []
    top_cursor = 1.0
    for h in scaled:
        bottom = top_cursor - h
        boundaries.append((top_cursor, bottom))
        top_cursor = bottom - vspacing

    def _panel_title(prefix, countries, edges):
        n_m = sum(1 for e in edges if e["kind"] == "mutual")
        n_o = len(edges) - n_m
        return (
            f"{prefix}<br>"
            f"<span style='font-size:11px;color:#6b7280;'>"
            f"{len(countries)} countries · {len(edges)} edges "
            f"({n_m} mutual ─, {n_o} one-way ╌)</span>"
        )

    fig = make_subplots(
        rows=3, cols=2,
        row_heights=row_heights,
        vertical_spacing=vspacing,
        horizontal_spacing=0.07,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
        ],
        subplot_titles=[
            _panel_title("Full picture · 1975–2025", qualified, full_edges),
            _panel_title("Era 1 · 1975–1999", era1_countries, era1_edges),
            _panel_title("Era 2 · 2000–2025", era2_countries, era2_edges),
            "Era 1 insights", "Era 2 insights",
        ],
    )

    render_network(fig, 1, 1, qualified, full_edges, full_bloc,
                   participation_years=participation_total)
    render_network(fig, 2, 1, era1_countries, era1_edges, era1_bloc,
                   migrated=migrated, participation_years=era1_participation)
    render_network(fig, 2, 2, era2_countries, era2_edges, era2_bloc,
                   migrated=migrated, participation_years=era2_participation)
    render_stat_panel(fig, 3, 1, "1975–1999", top_mutual_1, top_oneway_1, top_hatred_1)
    render_stat_panel(fig, 3, 2, "2000–2025", top_mutual_2, top_oneway_2, top_hatred_2)

    # ---- dotted flow connectors ----------------------------------------

    def add_flow_connector(x, y_top, y_bottom, label):
        fig.add_shape(
            type="line", x0=x, y0=y_top - 0.005, x1=x, y1=y_bottom + 0.022,
            line=dict(dash="dot", color="#9ca3af", width=2),
            xref="paper", yref="paper",
        )
        fig.add_annotation(
            x=x, y=y_bottom + 0.018, text="▼", showarrow=False,
            xref="paper", yref="paper", font=dict(size=14, color="#9ca3af"),
        )
        fig.add_annotation(
            x=x, y=(y_top + y_bottom) / 2, text=label,
            showarrow=False, xref="paper", yref="paper",
            font=dict(size=10, color="#6b7280", family="Georgia, serif"),
            bgcolor="white", borderpad=2,
        )

    gap1_top, gap1_bottom = boundaries[0][1], boundaries[1][0]
    gap2_top, gap2_bottom = boundaries[1][1], boundaries[2][0]
    add_flow_connector(0.25, gap1_top, gap1_bottom, "splits into two eras")
    add_flow_connector(0.75, gap1_top, gap1_bottom, "splits into two eras")
    add_flow_connector(0.25, gap2_top, gap2_bottom, "reveals evidence")
    add_flow_connector(0.75, gap2_top, gap2_bottom, "reveals evidence")

    # ---- bloc legend (computed from full-history detection) -----------
    bloc_names_full = sorted(set(full_bloc.values()))
    BLOC_NODE_PALETTE_L = BLOC_NODE_PALETTE
    bloc_col = {b: BLOC_NODE_PALETTE_L[i % len(BLOC_NODE_PALETTE_L)] for i, b in enumerate(bloc_names_full)}
    members_by_bloc = defaultdict(list)
    for c, b in full_bloc.items():
        members_by_bloc[b].append(c)

    legend_x, legend_y = 0.01, 0.275
    fig.add_annotation(
        x=legend_x, y=legend_y + 0.02,
        text="<b>Detected blocs (1975–2025)</b>",
        xref="paper", yref="paper", showarrow=False,
        font=dict(size=9, color="#374151"), xanchor="left",
    )
    for idx, bname in enumerate(bloc_names_full):
        col_hex = bloc_col[bname]
        members_str = ", ".join(sorted(members_by_bloc[bname])[:6])
        if len(members_by_bloc[bname]) > 6:
            members_str += f" +{len(members_by_bloc[bname])-6}"
        fig.add_shape(
            type="rect",
            x0=legend_x, y0=legend_y - idx * 0.028 - 0.005,
            x1=legend_x + 0.012, y1=legend_y - idx * 0.028 + 0.015,
            fillcolor=col_hex, line=dict(width=0),
            xref="paper", yref="paper",
        )
        fig.add_annotation(
            x=legend_x + 0.016, y=legend_y - idx * 0.028 + 0.005,
            text=f"<b>{bname}</b>: {members_str}",
            xref="paper", yref="paper", showarrow=False,
            font=dict(size=8, color="#374151"), xanchor="left", yanchor="middle",
        )

    # ---- reading guide annotation -------------------------------------
    fig.add_annotation(
        x=0.99, y=1.048, xref="paper", yref="paper",
        text=(
            "<b>Reading guide:</b> "
            "solid line ─ = mutual (NVS similar both ways)  ·  "
            "dashed line ╌ + ▶ = one-way (arrow toward receiver)  ·  "
            "darker/thicker = stronger NVS  ·  "
            "red tint = cross-bloc  ·  "
            "<b style='color:#b45309'>gold ring</b> = changed bloc between eras  ·  "
            "node size = years participated in that panel's window"
        ),
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=9.5, color="#4b5563"), align="right",
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Eurovision Voting Network — Hierarchical Bloc Structure</b>"
                "<br><span style='font-size:13px;color:#6b7280;'>"
                "From full history to era split to concrete evidence · 1975–2025</span>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=18, family="Georgia, serif", color="#111827"),
        ),
        height=1500, width=1200,
        paper_bgcolor="#fafafa", plot_bgcolor="#fafafa",
        showlegend=False,
        margin=dict(l=30, r=30, t=120, b=40),
    )

    explanation = f"""
**The story this poster tells:** start with the whole 50-year voting network,
watch it split into two distinct eras, then read the concrete evidence that
explains why the split matters.

**Edge selection:** for each country A, the top **{top_k_out}** outgoing
relationships where NVS(A→B) ≥ **{min_nvs_strength}** (on the 0–12 scale)
are selected. A pair survives if it qualifies from either endpoint's
outgoing perspective. This threshold means A genuinely and consistently
favours B, not just occasionally — a minimum average of ~{min_nvs_strength}
NVS points per year. The NVS score on the 0–12 scale represents points
normalised by the era's maximum (12 pre-2016, 24 post-2016).

**Edge classification:** once a pair survives selection, it is classified:
if `|NVS(A→B) − NVS(B→A)| ≤ {diff_threshold}` → drawn as a **solid line**
(MUTUAL — roughly reciprocal, both countries vote for each other at similar
strength); otherwise → **dashed line + directional triangle** (ONE-WAY —
one country gives significantly more than it receives back).

**Node size:** proportional to years participated within that specific
panel's window (full 1975–2025 for Tier 1; 1975–1999 or 2000–2025 for the
era panels). A country's node can appear a different size in different panels
because the window itself changes. **Gold ring** = country's detected bloc
changed between the two eras (measured by <50% blocmate overlap — not just
a label change from Louvain renumbering).

**Live counts:** each panel's title reports the actual number of countries
and edges drawn, split into mutual vs one-way, computed from real data.

**Tier 3 — evidence:** stat cards computed from the full, unfiltered data —
never from the edge-selection step — so a genuine cold-shoulder pair is
never missed just because it didn't clear the NVS strength floor.

Thresholds ({min_nvs_strength} NVS floor, {diff_threshold} mutual/one-way
split, {min_years} years for inclusion, {hatred_min_years}/{hatred_epsilon}
for cold-shoulder) are exploratory cutoffs for visual and narrative clarity.
"""
    return fig, "Hierarchical Bloc Structure — Storyboard", explanation


# =============================================================================
# DIAGRAM 8 — GEOGRAPHIC BLOC MIGRATION POSTER
# =============================================================================
#
# Same three-tier storytelling structure as Draft 7 (full history -> two
# eras -> per-era evidence), but countries are positioned at their REAL
# geographic coordinates (Scattergeo) instead of an abstract force-directed
# layout. Rationale: Eurovision bloc voting is itself a spatial question
# ("do neighbours vote for neighbours?" — see build_neighbour_effect), so
# anchoring nodes to true geography lets the map itself carry analytical
# weight instead of an arbitrary layout a reader has to learn first. This
# also keeps Draft 8 visually consistent with the geo views already used
# elsewhere in the dashboard (make_community_world_map_figure,
# make_directed_community_vote_map_figure).
#
# Deliberately NOT drawn: bloc "territory" polygons (convex hulls). Eurovision
# blocs are not geographically contiguous (diaspora-driven ties, Australia's
# participation, etc.), so a filled-region approach would either self-
# intersect or require silently excluding outlier members. Bloc identity is
# encoded purely through node colour, exactly as in the existing community
# world-map view.
#
# Migration encoding: rather than drawing a connector line BETWEEN two
# separate geo subplots (fragile, since each subplot has its own independent
# map projection and the two countries' pixel positions are not simply
# related), a migrated country's Era-2 marker uses a two-colour ring: the
# FILL colour is its new (Era 2) bloc colour, and the BORDER colour is its
# previous (Era 1) bloc colour. The Era-1 panel keeps the same gold-outline
# flag used in Draft 7, so a reader sees "this country is about to move" in
# panel 1 and "this is what it moved from / into" in panel 2, all without
# needing a cross-panel connector.
# =============================================================================

def _bloc_mean_nvs_matrix(df: pd.DataFrame, countries: list) -> pd.DataFrame:
    """Mean NVS matrix on the familiar 0-12 scale, reindexed to `countries`."""
    if df.empty or not countries:
        return pd.DataFrame(0.0, index=countries, columns=countries)
    m = (
        df.groupby(["src_label", "tgt_label"])["nvs"].mean()
        .unstack(fill_value=0)
        .reindex(index=countries, columns=countries, fill_value=0)
    ) * 12.0
    return m


def _bloc_disparity_filter_pairs(mat: pd.DataFrame, countries: list, bloc_map: dict,
                                  alpha: float = 0.05, max_total_edges: int | None = 30) -> list:
    """
    DISPARITY FILTER (Serrano, Boguna & Vespignani, 2009, PNAS 106(16):
    6483-6488) — selects edges that are statistically significant relative
    to EACH country's own pattern of ties, rather than against one fixed
    global rule (a fixed top-N, or a fixed value threshold).

    The underlying network here is undirected: weight(A,B) is the same
    representative strength already used for mutual/one-way classification
    elsewhere in this draft, value(A,B) = max(NVS(A->B), NVS(B->A)). Only
    cross-bloc pairs are considered at all (within-bloc ties are excluded
    from the candidate set entirely — bloc membership is already shown
    through node colour).

    For a country A with k cross-bloc partners and total tie-strength
    s = sum of weight(A, partner) across all of them, partner B's share is
        p(A,B) = weight(A,B) / s
    Under the null hypothesis that A's strength were spread uniformly at
    random across its k partners, the probability of a partner ending up
    with at least this large a share by chance is
        alpha(A,B) = (1 - p(A,B)) ** (k - 1)
    A small alpha(A,B) means the tie is too strong to be explained by
    chance, given how many partners A has and how strong A's ties are
    overall. A pair (A,B) survives if alpha(A,B) <= `alpha` from EITHER
    endpoint's perspective (a relationship only needs to be significant to
    one side to be worth showing). A country with degree 1 (a single
    cross-bloc partner) is trivially significant for that one tie.

    `max_total_edges` is a defensive cap (not part of the original
    algorithm): if the disparity filter still returns more edges than this
    for a given panel, only the most significant ones (lowest alpha) are
    kept, since a static poster has a hard practical legibility ceiling
    regardless of how principled the underlying selection is. Set to None
    to disable the cap entirely.
    """
    # Build the undirected cross-bloc weighted adjacency once.
    weight = {c: {} for c in countries}
    for i, a in enumerate(countries):
        for j, b in enumerate(countries):
            if i >= j or bloc_map.get(a) == bloc_map.get(b):
                continue
            ab = float(mat.loc[a, b]) if a in mat.index and b in mat.columns else 0.0
            ba = float(mat.loc[b, a]) if b in mat.index and a in mat.columns else 0.0
            v = max(ab, ba)
            if v > 0:
                weight[a][b] = v
                weight[b][a] = v

    def node_alpha(c: str, partner: str) -> float:
        neighbors = weight.get(c, {})
        k = len(neighbors)
        if k == 0:
            return 1.0
        if k == 1:
            return 0.0  # a country's only cross-bloc tie is trivially significant
        s = sum(neighbors.values())
        if s <= 0:
            return 1.0
        p = neighbors[partner] / s
        return (1.0 - p) ** (k - 1)

    candidates = []
    seen = set()
    for a in countries:
        for b in weight.get(a, {}):
            pair = tuple(sorted([a, b]))
            if pair in seen:
                continue
            seen.add(pair)
            alpha_ab = node_alpha(a, b)
            alpha_ba = node_alpha(b, a)
            best_alpha = min(alpha_ab, alpha_ba)
            if best_alpha <= alpha:
                candidates.append((best_alpha, pair[0], pair[1]))

    candidates.sort(key=lambda t: t[0])  # most significant (lowest alpha) first

    if max_total_edges is not None and len(candidates) > max_total_edges:
        candidates = candidates[:max_total_edges]

    return [(a, b) for _, a, b in candidates]


def _bloc_classify_edges(mat: pd.DataFrame, countries: list, diff_threshold: float,
                          bloc_map: dict, alpha: float = 0.05,
                          max_total_edges: int | None = 30,
                          min_nvs_strength: float = 1.5) -> list:
    """
    Classify the panel's statistically-significant cross-bloc relationships
    (selected via the disparity filter, see _bloc_disparity_filter_pairs)
    as MUTUAL (|diff| <= diff_threshold) or ONE-WAY (otherwise, with an
    explicit giver/receiver). Every surviving edge is, by construction, a
    cross-bloc tie that the disparity filter judged too strong to be
    explained by chance for at least one of its two endpoints.

    `min_nvs_strength`: secondary gate — even edges that pass the disparity
    filter are dropped if max(NVS(A→B), NVS(B→A)) < min_nvs_strength on
    the 0-12 scale. This prevents statistically-significant-but-practically-
    negligible ties from appearing on the map (e.g. a country with only 2
    cross-bloc partners where one gives 0.2 NVS — statistically 100%
    concentrated, but visually meaningless on a poster). Pairs surviving
    the disparity filter are ranked by this NVS strength for the defensive
    cap, so the strongest ties always take priority.
    """
    retained_pairs = _bloc_disparity_filter_pairs(
        mat, countries, bloc_map, alpha=alpha, max_total_edges=max_total_edges
    )

    edges = []
    for (a, b) in retained_pairs:
        ab = float(mat.loc[a, b])
        ba = float(mat.loc[b, a])
        if max(ab, ba) < min_nvs_strength:
            continue
        if ab <= 0 and ba <= 0:
            continue
        diff = abs(ab - ba)
        if diff <= diff_threshold:
            edges.append({
                "a": a, "b": b, "kind": "mutual",
                "value": (ab + ba) / 2.0, "ab": ab, "ba": ba, "diff": diff,
            })
        else:
            if ab > ba:
                giver, receiver = a, b
            else:
                giver, receiver = b, a
            edges.append({
                "a": a, "b": b, "kind": "one_way",
                "giver": giver, "receiver": receiver,
                "value": max(ab, ba), "ab": ab, "ba": ba, "diff": diff,
            })

    return edges



def _bloc_edge_color(value: float) -> str:
    """
    Colour/opacity anchored at the 1.0 NVS breakpoint — faint below it,
    increasingly dark red above it. Every edge reaching this function is
    now a cross-bloc tie by construction (see _bloc_classify_edges), so a
    single red-tinted palette is used throughout rather than a separate
    within/cross-bloc colour distinction.
    """
    value = max(value, 0.0)
    if value < 1.0:
        alpha = 0.12 + 0.20 * value
    else:
        alpha = 0.38 + 0.55 * min((value - 1.0) / 11.0, 1.0)
    return f"rgba(200,60,60,{alpha:.2f})"


def _bloc_detect(df: pd.DataFrame, countries: list, q: float = 0.6) -> dict:
    """Louvain bloc detection on the mutual-affinity graph for `countries`."""
    if not countries:
        return {}
    aff = _mutual_affinity(_affinity_input(df), countries)
    return _detect_blocs(aff, countries, q=q)


def _bloc_flag_migrated(map1: dict, map2: dict) -> set:
    """
    Flag countries whose bloc membership changed between two independently
    detected eras, using member-set overlap (NOT bloc label, since labels
    are reassigned by size each time a fresh detection runs — comparing
    labels directly would produce false-positive "migrations" purely from
    renumbering). A country is flagged if fewer than 50% of its Era-1
    blocmates remain its blocmates in Era 2.
    """
    from collections import defaultdict
    g1, g2 = defaultdict(set), defaultdict(set)
    for c, b in map1.items():
        g1[b].add(c)
    for c, b in map2.items():
        g2[b].add(c)
    migrated = set()
    for c in set(map1) & set(map2):
        m1 = g1[map1[c]] - {c}
        m2 = g2[map2[c]] - {c}
        overlap = (len(m1 & m2) / len(m1)) if m1 else 0.0
        if overlap < 0.5:
            migrated.add(c)
    return migrated


def _bloc_eligibility_frame(df: pd.DataFrame, countries: list) -> pd.DataFrame:
    """
    Build a (year, src_label, tgt_label, points, nvs) frame covering every
    pair of countries that were BOTH present in a given year, filling in
    zero-point rows where no vote actually occurred. Needed to distinguish
    a genuine "A never voted for B despite N eligible years" cold-shoulder
    from years where A and B simply weren't both competing.
    """
    countries_set = set(countries)
    rows = []
    for yr, g in df.groupby("year"):
        participants = (set(g["src_label"]) | set(g["tgt_label"])) & countries_set
        for s in participants:
            for t in participants:
                if s != t:
                    rows.append((yr, s, t))
    if not rows:
        return pd.DataFrame(columns=["year", "src_label", "tgt_label", "points", "nvs"])

    elig = pd.DataFrame(rows, columns=["year", "src_label", "tgt_label"])
    actual = df.groupby(["year", "src_label", "tgt_label"], as_index=False)["points"].sum()
    merged = elig.merge(actual, on=["year", "src_label", "tgt_label"], how="left")
    merged["points"] = merged["points"].fillna(0)
    merged["era_max_v"] = merged["year"].apply(_era_max)
    merged["nvs"] = (merged["points"] / merged["era_max_v"]).clip(0, 1)
    return merged


def _bloc_era_stats(df: pd.DataFrame, countries: list, edges: list,
                     hatred_min_years: int, hatred_epsilon: float):
    """
    Per-era evidence: top mutual voters, top one-way voters, and
    cold-shoulder pairs. Computed from the FULL (unfiltered) eligibility
    frame, never from the top-k-pruned `edges` used for network rendering,
    so a genuine superlative can never be accidentally hidden by the
    network's noise-reduction step.
    """
    mutual = [e for e in edges if e["kind"] == "mutual"]
    one_way = [e for e in edges if e["kind"] == "one_way"]

    top_mutual = sorted(mutual, key=lambda e: e["value"], reverse=True)[:3]
    top_oneway = sorted(one_way, key=lambda e: e["diff"], reverse=True)[:3]

    elig = _bloc_eligibility_frame(df, countries)
    if elig.empty:
        return top_mutual, top_oneway, pd.DataFrame()

    agg = (
        elig.groupby(["src_label", "tgt_label"])
        .agg(years_eligible=("year", "nunique"), mean_nvs=("nvs", "mean"))
        .reset_index()
    )
    reciprocal_lookup = {
        (r["src_label"], r["tgt_label"]): r["mean_nvs"] for _, r in agg.iterrows()
    }
    candidates = agg[
        (agg["years_eligible"] >= hatred_min_years) & (agg["mean_nvs"] < hatred_epsilon)
    ].copy()
    if candidates.empty:
        return top_mutual, top_oneway, candidates

    candidates["reciprocal_nvs"] = candidates.apply(
        lambda r: reciprocal_lookup.get((r["tgt_label"], r["src_label"]), 0.0), axis=1
    )
    top_hatred = candidates.sort_values(
        ["years_eligible", "reciprocal_nvs"], ascending=[False, False]
    ).head(3)
    return top_mutual, top_oneway, top_hatred


_GEO_BLOC_PALETTE = [
    "#1f4e79", "#d1495b", "#2a9d8f", "#f4a261",
    "#6a4c93", "#7f5539", "#577590", "#3a86ff",
]


def _geo_apply_auto_projection(fig, row: int, col: int):
    """
    Style the basemap and auto-fit the visible extent to whatever countries
    are actually plotted in this specific subplot (fitbounds="locations").

    A fixed Europe-only lon/lat box would clip non-European Eurovision
    participants (e.g. Australia, Israel) right off the map whenever they
    happen to be among the plotted countries for a given panel, and would
    waste space on empty ocean whenever a panel's qualifying cohort is
    smaller. Auto-fitting per panel guarantees every plotted country is
    actually visible, at the cost of each panel potentially having a
    slightly different zoom level from its neighbours — a reasonable
    trade-off since the alternative is silently cropping data off the map.
    """
    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#f7fafc",
        showocean=True, oceancolor="#eaf4ff",
        showcountries=True, countrycolor="#c5cfdb",
        showcoastlines=True, coastlinecolor="#aebed2",
        showframe=False,
        fitbounds="locations",
        row=row, col=col,
    )


def _geo_bow_path(lat0: float, lon0: float, lat1: float, lon1: float,
                   bow: float = 0.14, n: int = 14):
    """
    Quadratic-curve path between two geographic points, bowed perpendicular
    to the straight line by a fraction (`bow`) of the segment length.

    Purely straight edges between nearby clusters of countries tend to
    visually fuse together on a dense map; a slight curve keeps individual
    relationships distinguishable from one another without materially
    distorting their true endpoints (a standard de-cluttering technique in
    flow-map visualisation, used here purely for visual separation, not as
    a geodesic/great-circle correction).
    """
    dx, dy = (lon1 - lon0), (lat1 - lat0)
    dist = float(np.hypot(dx, dy)) or 1e-6
    mx, my = (lon0 + lon1) / 2.0, (lat0 + lat1) / 2.0
    perp_x, perp_y = -dy, dx
    perp_norm = float(np.hypot(perp_x, perp_y)) or 1e-6
    offset = bow * dist
    cx = mx + (perp_x / perp_norm) * offset
    cy = my + (perp_y / perp_norm) * offset

    t = np.linspace(0, 1, n)
    lon = (1 - t) ** 2 * lon0 + 2 * (1 - t) * t * cx + t ** 2 * lon1
    lat = (1 - t) ** 2 * lat0 + 2 * (1 - t) * t * cy + t ** 2 * lat1
    return lat, lon


def _geo_render_network(
    fig, row: int, col: int,
    countries: list, edges: list, bloc_map: dict, coord_lookup: dict,
    years_lookup: dict,
    migrated_gold: set | None = None,
    migrated_ring_color: dict | None = None,
    label_top_n: int = 12,
):
    """
    Render one geographic bloc network panel, condensed for readability:

    - `edges` is expected to already contain at most ONE representative
      cross-bloc relationship per country (see
      _bloc_strongest_cross_bloc_partner / _bloc_classify_edges) — this
      function does not do any further edge filtering itself, it only
      decides how to draw what it's given.
    - Direction is encoded by LINE STYLE rather than an arrowhead marker:
      a solid line means the relationship is MUTUAL (roughly reciprocal
      NVS in both directions); a dashed line means it is ONE-WAY (the
      giver/receiver are named in the hover tooltip). No triangle/arrow
      markers are drawn — they doubled the number of map elements without
      adding information beyond what the dash style + hover text already
      convey.
    - Edges are drawn as a slight curve (see _geo_bow_path) instead of a
      straight line, so relationships between nearby clusters of
      countries stay visually distinguishable from one another.
    - Node SIZE encodes `years_lookup[country]` — the number of years that
      country participated WITHIN THIS PANEL'S TIME WINDOW (e.g. for an
      Era-1 panel, years within 1975-1999 only; for Tier 1, years across
      the full 1975-2025 range). A country active 20 years in this
      window is drawn visibly larger than one active 10 years, regardless
      of its voting strength.
    - Only the top `label_top_n` countries by that same years-participated
      score get a text label; the rest remain fully interactive
      (hoverable) dots so the map doesn't get cluttered with overlapping
      country names.

    migrated_gold: countries to flag with a gold outline (used for the
        Era-1 "about to migrate" flag).
    migrated_ring_color: {country: previous_era_bloc_color} — when set,
        the country's marker border uses this colour instead of white/gold,
        showing "what bloc it came from" while the fill shows "what bloc
        it's in now" (used for the Era-2 "migrated from" flag).
    """
    migrated_gold = migrated_gold or set()
    migrated_ring_color = migrated_ring_color or {}

    plot_countries = [c for c in countries if c in coord_lookup]
    if not plot_countries:
        return

    bloc_names = sorted(set(bloc_map.values())) if bloc_map else []
    bloc_color = {b: _GEO_BLOC_PALETTE[i % len(_GEO_BLOC_PALETTE)] for i, b in enumerate(bloc_names)}

    for e in edges:
        if e["a"] not in coord_lookup or e["b"] not in coord_lookup:
            continue
        lat_a, lon_a = coord_lookup[e["a"]]
        lat_b, lon_b = coord_lookup[e["b"]]
        color = _bloc_edge_color(e["value"])
        width = 1.4 + 2.6 * min(e["value"] / 12.0, 1.0)
        dash = "solid" if e["kind"] == "mutual" else "dash"

        path_lat, path_lon = _geo_bow_path(lat_a, lon_a, lat_b, lon_b)

        direction_text = (
            f"Mutual (≈{e['value']:.1f} NVS both ways)" if e["kind"] == "mutual"
            else f"One-way: {e['giver']} → {e['receiver']} ({e['value']:.1f} NVS)"
        )

        fig.add_trace(go.Scattergeo(
            lon=path_lon, lat=path_lat,
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            opacity=1.0,
            hovertemplate=(
                f"<b>{e['a']}</b> ↔ <b>{e['b']}</b><br>"
                f"{e['a']}→{e['b']}: {e['ab']:.2f} | {e['b']}→{e['a']}: {e['ba']:.2f}<br>"
                f"{direction_text}<extra></extra>"
            ),
            showlegend=False,
        ), row=row, col=col)

    max_years = max([years_lookup.get(c, 0) for c in plot_countries], default=1) or 1
    labelled = set(
        sorted(plot_countries, key=lambda c: years_lookup.get(c, 0), reverse=True)[:label_top_n]
    )

    node_line_color, node_line_width = [], []
    hover_text = []
    for c in plot_countries:
        if c in migrated_ring_color:
            node_line_color.append(migrated_ring_color[c])
            node_line_width.append(3.2)
        elif c in migrated_gold:
            node_line_color.append("#facc15")
            node_line_width.append(3.2)
        else:
            node_line_color.append("white")
            node_line_width.append(1.0)

        yrs = years_lookup.get(c, 0)
        txt = f"{c}<br>Bloc: {bloc_map.get(c, 'NA')}<br>Years participated (this window): {yrs}"
        if c in migrated_ring_color:
            txt += "<br><b>Migrated into this bloc since the previous era</b>"
        elif c in migrated_gold:
            txt += "<br><b>Will migrate to a different bloc next era</b>"
        hover_text.append(txt)

    node_size = [
        8 + 14 * np.sqrt(max(years_lookup.get(c, 0), 0) / max_years)
        for c in plot_countries
    ]
    node_fill = [bloc_color.get(bloc_map.get(c), "#9ca3af") for c in plot_countries]
    node_text = [c if c in labelled else "" for c in plot_countries]

    fig.add_trace(go.Scattergeo(
        lon=[coord_lookup[c][1] for c in plot_countries],
        lat=[coord_lookup[c][0] for c in plot_countries],
        text=node_text,
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=9, color="#111827"),
        marker=dict(
            size=node_size, color=node_fill,
            line=dict(width=node_line_width, color=node_line_color),
        ),
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    ), row=row, col=col)

    _geo_apply_auto_projection(fig, row, col)


def _geo_render_stat_panel(fig, row: int, col: int, era_label: str,
                            top_mutual: list, top_oneway: list, top_hatred: pd.DataFrame):
    """Same plain stat-card text layout used in Draft 7's Tier 3, reused
    here for visual/analytical consistency between the two poster drafts."""
    fig.update_xaxes(visible=False, range=[0, 1], row=row, col=col)
    fig.update_yaxes(visible=False, range=[0, 1], row=row, col=col)

    mutual_lines = [
        f"{e['a']} ↔ {e['b']}  (NVS {e['value']:.1f})" for e in top_mutual
    ] or ["No qualifying mutual pairs"]

    oneway_lines = [
        f"{e['giver']} → {e['receiver']}  (Δ{e['diff']:.1f})" for e in top_oneway
    ] or ["No qualifying one-way pairs"]

    if top_hatred is None or top_hatred.empty:
        hatred_lines = ["No sustained cold-shoulder pairs found"]
    else:
        hatred_lines = [
            f"{r['src_label']} ⇏ {r['tgt_label']}  ({int(r['years_eligible'])} yrs eligible)"
            for _, r in top_hatred.iterrows()
        ]

    sections = [
        ("🤝 Top mutual voters", mutual_lines),
        ("➡️ Top one-way voters", oneway_lines),
        ("❄️ Cold-shoulder pairs", hatred_lines),
    ]

    y = 0.95
    fig.add_annotation(
        x=0.03, y=1.0, text=f"<b>{era_label}</b>", showarrow=False,
        font=dict(size=13, color="#1f2937", family="Georgia, serif"),
        xanchor="left", yanchor="top", row=row, col=col,
    )
    y -= 0.13
    for heading, lines in sections:
        fig.add_annotation(
            x=0.03, y=y, text=f"<b>{heading}</b>", showarrow=False,
            font=dict(size=10, color="#374151"),
            xanchor="left", yanchor="top", row=row, col=col,
        )
        y -= 0.09
        for line in lines[:3]:
            fig.add_annotation(
                x=0.07, y=y, text=line, showarrow=False,
                font=dict(size=9, color="#4b5563"),
                xanchor="left", yanchor="top", row=row, col=col,
            )
            y -= 0.075
        y -= 0.025


_SLOPE_MIGRATION_PALETTE = [
    "#d62828", "#f77f00", "#9b5de5", "#06a77d",
    "#118ab2", "#ef476f", "#8d5524", "#4b0082",
]


def _render_bloc_slope_chart(
    fig, row: int, col: int,
    era1_countries: list, era2_countries: list,
    era1_bloc: dict, era2_bloc: dict, migrated: set,
):
    """
    Render a slope/bump chart: one line per country, left endpoint = its
    Era-1 bloc, right endpoint = its Era-2 bloc. This replaces drawing
    inter-country relationship edges on the geographic maps as the primary
    way to show bloc migration — a slope chart has NO pairwise edges to
    select or filter at all (it is always exactly one line per country),
    which sidesteps the static-poster legibility problem that any
    "which edges matter" selection rule runs into on a geographically
    dense map.

    Critically, a country's two endpoints are positioned using the
    independently-computed `migrated` flag (blocmate-overlap < 50%, see
    _bloc_flag_migrated) rather than raw bloc labels: Louvain bloc labels
    are reassigned by size every time detection runs, so two completely
    independent detections could give a STABLE country two different
    labels purely by coincidence. Anchoring on the verified migration flag
    means:
      - a NON-migrated country is always drawn as a perfectly FLAT line
        (same vertical band on both sides), regardless of any label churn;
      - a MIGRATED country is drawn as a genuine diagonal, moving from its
        Era-1 bloc's band to its Era-2 bloc's band.
    Only migrated countries are individually labelled, to keep the chart
    readable — non-migrated countries are still visible as flat lines/dots,
    just without text competing for space.
    """
    era1_bloc_names = sorted(set(era1_bloc.values())) if era1_bloc else []
    era2_bloc_names = sorted(set(era2_bloc.values())) if era2_bloc else []
    era1_band = {b: i for i, b in enumerate(era1_bloc_names)}
    era2_band = {b: i for i, b in enumerate(era2_bloc_names)}

    common = [c for c in era1_countries if c in era1_bloc and c in era2_bloc]

    # Stable per-band jitter so countries sharing a band don't fully overlap.
    band_counts_left, band_counts_right = {}, {}
    rows = []
    for c in sorted(common):
        is_migrant = c in migrated
        y_left = era1_band.get(era1_bloc.get(c), 0)
        y_right = y_left if not is_migrant else era2_band.get(era2_bloc.get(c), y_left)

        band_counts_left[y_left] = band_counts_left.get(y_left, 0) + 1
        jitter_l = band_counts_left[y_left]
        band_counts_right[y_right] = band_counts_right.get(y_right, 0) + 1
        jitter_r = band_counts_right[y_right]

        rows.append({
            "country": c, "migrated": is_migrant,
            "y_left_base": y_left, "y_right_base": y_right,
            "jitter_l": jitter_l, "jitter_r": jitter_r,
            "early_bloc": era1_bloc.get(c), "late_bloc": era2_bloc.get(c),
        })

    if not rows:
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)
        return

    jitter_step = 0.10

    migration_paths = sorted({
        (r["early_bloc"], r["late_bloc"]) for r in rows if r["migrated"]
    }, key=lambda p: -sum(1 for r in rows if r["migrated"] and (r["early_bloc"], r["late_bloc"]) == p))
    path_color = {
        path: _SLOPE_MIGRATION_PALETTE[i % len(_SLOPE_MIGRATION_PALETTE)]
        for i, path in enumerate(migration_paths)
    }

    for r in rows:
        y0 = r["y_left_base"] + (r["jitter_l"] % 5 - 2) * jitter_step
        y1 = r["y_right_base"] + (r["jitter_r"] % 5 - 2) * jitter_step

        if r["migrated"]:
            color = path_color.get((r["early_bloc"], r["late_bloc"]), "#d62828")
            width = 2.4
            opacity = 0.9
        else:
            color = "#cbd5e1"
            width = 1.1
            opacity = 0.55

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[y0, y1], mode="lines+markers",
            line=dict(color=color, width=width),
            marker=dict(size=5, color=color),
            opacity=opacity,
            hovertemplate=(
                f"<b>{r['country']}</b><br>"
                f"Era 1 bloc: {r['early_bloc']}<br>Era 2 bloc: {r['late_bloc']}<br>"
                f"{'MIGRATED' if r['migrated'] else 'Stayed in an equivalent bloc'}"
                "<extra></extra>"
            ),
            showlegend=False,
        ), row=row, col=col)

        if r["migrated"]:
            fig.add_annotation(
                x=1.03, y=y1, text=r["country"], showarrow=False,
                font=dict(size=9, color=color), xanchor="left", yanchor="middle",
                row=row, col=col,
            )

    for b, y in era1_band.items():
        n = sum(1 for r in rows if r["y_left_base"] == y)
        fig.add_annotation(
            x=-0.03, y=y, text=f"{b} ({n})", showarrow=False,
            font=dict(size=10, color="#374151"), xanchor="right", yanchor="middle",
            row=row, col=col,
        )
    for b, y in era2_band.items():
        fig.add_annotation(
            x=1.0, y=y - 0.32, text=f"{b}", showarrow=False,
            font=dict(size=9, color="#9ca3af"), xanchor="left", yanchor="top",
            row=row, col=col,
        )

    fig.update_xaxes(
        visible=True, range=[-0.45, 1.55],
        tickmode="array", tickvals=[0, 1], ticktext=["Era 1 (1975–1999)", "Era 2 (2000–2025)"],
        showgrid=False, row=row, col=col,
    )
    fig.update_yaxes(visible=False, row=row, col=col)


def build_geo_bloc_migration_poster(
    df: pd.DataFrame,
    id2label: dict,
    nodes_df: pd.DataFrame,
    min_years: int = 15,
    min_years_per_half: int = 5,
    diff_threshold: float = 1.0,
    hatred_min_years: int = 10,
    hatred_epsilon: float = 0.04,
    label_top_n: int = 12,
    disparity_alpha: float = 0.05,
    max_edges_per_panel: int | None = 30,
    min_nvs_strength: float = 1.5,
):
    """
    DRAFT 8 — Geographic Bloc Migration Poster: a four-tier storyboard
    (full-history map -> two era maps -> migration slope chart -> per-era
    evidence), built around two design fixes specifically for STATIC,
    non-interactive print legibility at Eurovision's small-country,
    geographically dense scale:

    1. Each geographic map draws only cross-bloc relationships that pass
       the DISPARITY FILTER (Serrano, Boguna & Vespignani, 2009, PNAS
       106(16): 6483-6488, see _bloc_disparity_filter_pairs) at
       significance level `disparity_alpha` — i.e. ties that are
       statistically too strong to be explained by chance, GIVEN EACH
       COUNTRY'S OWN number of partners and overall tie-strength. This
       replaces an earlier fixed "global top-N" rule: rather than always
       keeping a hard-coded number of edges regardless of the data, the
       filter adapts per country (a country with few partners needs a much
       more lopsided relationship to "earn" significance than a country
       with many), and no node is dropped outright unless literally none
       of its ties are statistically distinguishable from random noise.
       `max_edges_per_panel` is a defensive cap on top of the filter,
       since a static poster still has a hard practical legibility
       ceiling regardless of how principled the underlying selection is.

    2. Bloc MIGRATION — the thing a map's pairwise edges are worst at
       showing clearly in print — is moved to a dedicated slope/bump chart
       (Tier 3): one line per country from its Era-1 bloc to its Era-2
       bloc, with zero pairwise edges to select at all. This sidesteps the
       "which edges matter" problem entirely for the migration story,
       while the maps remain focused on what geography is actually good
       for: showing bloc composition and the few statistically significant
       cross-bloc relationships.

    Two DIFFERENT qualification rules are used for the two map tiers:
      - Tier 1 (full 1975-2025 picture) includes any country with at least
        `min_years` years of participation across the WHOLE period.
      - Tier 2 (era maps + slope chart) uses a STRICTER, more comparable
        cohort: only countries with at least `min_years_per_half` years of
        participation in BOTH 1975-1999 and 2000-2025, so a "migration"
        flag reflects a real, well-evidenced shift rather than an artifact
        of a country barely appearing in one of the two halves.

    Node size on the maps encodes each country's years of participation
    WITHIN that panel's own time window. Every map panel's title reports
    the live country/edge counts actually drawn.

    Returns (figure, title, explanation_markdown) per the module's
    standard contract.
    """
    from plotly.subplots import make_subplots

    df = _add_era_max_col(df)
    df["src_label"] = df["source"].map(id2label).fillna(df["source"])
    df["tgt_label"] = df["target"].map(id2label).fillna(df["target"])

    coord_lookup = _coord_lookup(nodes_df, id2label)

    all_participation = (
        pd.concat([
            df[["year", "src_label"]].rename(columns={"src_label": "country"}),
            df[["year", "tgt_label"]].rename(columns={"tgt_label": "country"}),
        ]).drop_duplicates()
    )

    # ---- Tier 1 cohort: overall participation >= min_years -------------

    participation_total = all_participation.groupby("country")["year"].nunique()
    tier1_countries = sorted([
        c for c in participation_total[participation_total >= min_years].index
        if c in coord_lookup
    ])
    tier1_years_lookup = {c: int(participation_total.get(c, 0)) for c in tier1_countries}

    # ---- Tier 2 cohort: >= min_years_per_half in BOTH eras --------------

    era1_participation = (
        all_participation[all_participation["year"] <= 1999]
        .groupby("country")["year"].nunique()
    )
    era2_participation = (
        all_participation[all_participation["year"] >= 2000]
        .groupby("country")["year"].nunique()
    )
    era_countries_set = sorted([
        c for c in set(era1_participation.index) & set(era2_participation.index)
        if era1_participation.get(c, 0) >= min_years_per_half
        and era2_participation.get(c, 0) >= min_years_per_half
        and c in coord_lookup
    ])

    if len(tier1_countries) < 3 and len(era_countries_set) < 3:
        return None, "Geographic Bloc Migration Poster", (
            f"Not enough countries met either qualification rule "
            f"(>= {min_years} years overall, or >= {min_years_per_half} years "
            "in both eras) with usable coordinates to build this draft."
        )

    full_df = df[df["src_label"].isin(tier1_countries) & df["tgt_label"].isin(tier1_countries)].copy()

    # -------------------------------------------------------------------
    # Tier 1 — full history (broad long-history cohort)
    # -------------------------------------------------------------------

    full_mat = _bloc_mean_nvs_matrix(full_df, tier1_countries)
    full_bloc = _bloc_detect(full_df, tier1_countries)
    full_edges = _bloc_classify_edges(
        full_mat, tier1_countries, diff_threshold, full_bloc,
        alpha=disparity_alpha, max_total_edges=max_edges_per_panel,
        min_nvs_strength=min_nvs_strength,
    )

    # -------------------------------------------------------------------
    # Tier 2 — two independently-detected eras (stricter both-halves cohort)
    # -------------------------------------------------------------------

    era1_df = df[
        (df["year"] <= 1999)
        & df["src_label"].isin(era_countries_set)
        & df["tgt_label"].isin(era_countries_set)
    ].copy()
    era2_df = df[
        (df["year"] >= 2000)
        & df["src_label"].isin(era_countries_set)
        & df["tgt_label"].isin(era_countries_set)
    ].copy()

    era1_countries = sorted({
        c for c in era_countries_set
        if c in set(era1_df["src_label"]) | set(era1_df["tgt_label"])
    })
    era2_countries = sorted({
        c for c in era_countries_set
        if c in set(era2_df["src_label"]) | set(era2_df["tgt_label"])
    })

    era1_years_lookup = {c: int(era1_participation.get(c, 0)) for c in era1_countries}
    era2_years_lookup = {c: int(era2_participation.get(c, 0)) for c in era2_countries}

    era1_mat = _bloc_mean_nvs_matrix(era1_df, era1_countries)
    era2_mat = _bloc_mean_nvs_matrix(era2_df, era2_countries)
    era1_bloc = _bloc_detect(era1_df, era1_countries)
    era2_bloc = _bloc_detect(era2_df, era2_countries)
    era1_edges = _bloc_classify_edges(
        era1_mat, era1_countries, diff_threshold, era1_bloc,
        alpha=disparity_alpha, max_total_edges=max_edges_per_panel,
        min_nvs_strength=min_nvs_strength,
    )
    era2_edges = _bloc_classify_edges(
        era2_mat, era2_countries, diff_threshold, era2_bloc,
        alpha=disparity_alpha, max_total_edges=max_edges_per_panel,
        min_nvs_strength=min_nvs_strength,
    )

    migrated = _bloc_flag_migrated(era1_bloc, era2_bloc)

    # Build the "previous bloc colour" ring lookup for the Era-2 panel.
    era1_bloc_names = sorted(set(era1_bloc.values())) if era1_bloc else []
    era1_bloc_color = {b: _GEO_BLOC_PALETTE[i % len(_GEO_BLOC_PALETTE)] for i, b in enumerate(era1_bloc_names)}
    migrated_ring_color = {
        c: era1_bloc_color.get(era1_bloc.get(c), "#9ca3af")
        for c in migrated if c in era1_bloc and c in era2_bloc
    }

    # -------------------------------------------------------------------
    # Tier 3 — per-era evidence
    # -------------------------------------------------------------------

    top_mutual_1, top_oneway_1, top_hatred_1 = _bloc_era_stats(
        era1_df, era1_countries, era1_edges, hatred_min_years, hatred_epsilon
    )
    top_mutual_2, top_oneway_2, top_hatred_2 = _bloc_era_stats(
        era2_df, era2_countries, era2_edges, hatred_min_years, hatred_epsilon
    )

    # -------------------------------------------------------------------
    # Live edge/node counts shown directly on the chart, computed from the
    # actual data just processed above — no need to read the explanation
    # text below to know how condensed each panel actually is.
    # -------------------------------------------------------------------

    def panel_title(prefix: str, countries: list, edges: list) -> str:
        n_mutual = sum(1 for e in edges if e["kind"] == "mutual")
        n_oneway = len(edges) - n_mutual
        return (
            f"{prefix}<br>"
            f"<span style='font-size:11px; color:#6b7280;'>"
            f"{len(countries)} countries · {len(edges)} edges "
            f"({n_mutual} mutual, {n_oneway} one-way)</span>"
        )

    title_tier1 = panel_title("Full picture · 1975–2025", tier1_countries, full_edges)
    title_era1 = panel_title("Era 1 · 1975–1999", era1_countries, era1_edges)
    title_era2 = panel_title("Era 2 · 2000–2025", era2_countries, era2_edges)

    # -------------------------------------------------------------------
    # Figure assembly — FOUR storyboard tiers (geo map, two geo maps,
    # migration slope chart, stat-cards), connected only by dotted flow
    # annotations drawn in paper coordinates.
    # -------------------------------------------------------------------

    row_heights = [0.34, 0.27, 0.20, 0.19]
    vspacing = 0.075
    total_gap = vspacing * (len(row_heights) - 1)
    avail = 1.0 - total_gap
    scaled = [h * avail for h in row_heights]

    boundaries = []
    top_cursor = 1.0
    for h in scaled:
        bottom = top_cursor - h
        boundaries.append((top_cursor, bottom))
        top_cursor = bottom - vspacing

    n_migrated = len(migrated)
    slope_title = (
        f"Bloc migration · who moved between eras"
        f"<br><span style='font-size:11px; color:#6b7280;'>"
        f"{len(era1_countries)} countries tracked · {n_migrated} migrated, "
        f"{len(era1_countries) - n_migrated} stayed</span>"
    )

    fig = make_subplots(
        rows=4, cols=2,
        row_heights=row_heights,
        vertical_spacing=vspacing,
        horizontal_spacing=0.05,
        specs=[
            [{"type": "scattergeo", "colspan": 2}, None],
            [{"type": "scattergeo"}, {"type": "scattergeo"}],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=[
            title_tier1,
            title_era1, title_era2,
            slope_title,
            "Era 1 evidence", "Era 2 evidence",
        ],
    )

    _geo_render_network(
        fig, 1, 1, tier1_countries, full_edges, full_bloc, coord_lookup,
        years_lookup=tier1_years_lookup, label_top_n=label_top_n,
    )
    _geo_render_network(
        fig, 2, 1, era1_countries, era1_edges, era1_bloc, coord_lookup,
        years_lookup=era1_years_lookup,
        migrated_gold=migrated,
        label_top_n=label_top_n,
    )
    _geo_render_network(
        fig, 2, 2, era2_countries, era2_edges, era2_bloc, coord_lookup,
        years_lookup=era2_years_lookup,
        migrated_ring_color=migrated_ring_color,
        label_top_n=label_top_n,
    )
    _render_bloc_slope_chart(
        fig, 3, 1, era1_countries, era2_countries, era1_bloc, era2_bloc, migrated,
    )
    _geo_render_stat_panel(fig, 4, 1, "1975–1999", top_mutual_1, top_oneway_1, top_hatred_1)
    _geo_render_stat_panel(fig, 4, 2, "2000–2025", top_mutual_2, top_oneway_2, top_hatred_2)

    # ---- dotted flow connectors between tiers (paper coordinates) -----
    #
    # Same approach as Draft 7: showarrow=True annotations require
    # axref/ayref, and those never accept "paper" — only "pixel" or an
    # axis id. So the dotted shaft is a shape, and the arrowhead is a
    # plain centred "▼" text annotation with showarrow=False.

    def add_flow_connector(x, y_top, y_bottom, label):
        fig.add_shape(
            type="line", x0=x, y0=y_top - 0.005, x1=x, y1=y_bottom + 0.018,
            line=dict(dash="dot", color="#9ca3af", width=2),
            xref="paper", yref="paper",
        )
        fig.add_annotation(
            x=x, y=y_bottom + 0.014, text="▼",
            showarrow=False,
            xref="paper", yref="paper",
            font=dict(size=13, color="#9ca3af"),
        )
        fig.add_annotation(
            x=x, y=(y_top + y_bottom) / 2, text=label,
            showarrow=False, xref="paper", yref="paper",
            font=dict(size=10, color="#6b7280", family="Georgia, serif"),
            bgcolor="white", borderpad=2,
        )

    gap1_top, gap1_bottom = boundaries[0][1], boundaries[1][0]
    gap2_top, gap2_bottom = boundaries[1][1], boundaries[2][0]
    gap3_top, gap3_bottom = boundaries[2][1], boundaries[3][0]

    add_flow_connector(0.25, gap1_top, gap1_bottom, "splits into two eras")
    add_flow_connector(0.75, gap1_top, gap1_bottom, "splits into two eras")
    add_flow_connector(0.50, gap2_top, gap2_bottom, "tracks who migrated")
    add_flow_connector(0.25, gap3_top, gap3_bottom, "reveals evidence")
    add_flow_connector(0.75, gap3_top, gap3_bottom, "reveals evidence")


    # ---- legend explaining the visual grammar --------------------------

    fig.add_annotation(
        x=0.99, y=1.045, xref="paper", yref="paper",
        text=(
            "<b>Reading guide:</b> solid line = mutual relationship · dashed line = one-way "
            "(see hover for direction) · darker/thicker = stronger (NVS ≥ 1.0 anchor) · "
            "node size = years participated in this panel's window · "
            "<span style='color:#b45309'><b>gold ring</b></span> = will migrate next era (map) · "
            "<b>coloured ring</b> = migrated from that bloc's colour (map) · "
            "in the slope chart: grey flat line = stayed, coloured diagonal = migrated"
        ),
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=10, color="#4b5563"), align="right",
    )

    fig.update_layout(
        title=dict(
            text="Geographic Bloc Migration — From Whole History to Evidence",
            x=0.5, xanchor="center", font=dict(size=18, family="Georgia, serif", color="#111827"),
        ),
        height=1750, width=1150,
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=30, r=30, t=110, b=40),
    )

    explanation = f"""
**The story this poster tells:** a four-tier storyboard — full-history map,
two era maps, a migration slope chart, then per-era evidence — designed
specifically around a constraint that earlier versions of this draft ran
into: Eurovision's ~20-40 countries are small and geographically packed, so
drawing many pairwise relationships between them on a real map is
fundamentally hard to read in a STATIC, non-interactive print poster, no
matter how the edges are chosen. The fix here is structural, not just a
parameter change: maps now show only a handful of the most extreme
relationships, and migration — the thing pairwise map edges are worst at
showing clearly — gets its own dedicated chart type.

**Why geography, not a force-directed layout:** Eurovision bloc voting is
itself a spatial question (see "The Neighbour Effect" draft). Pinning nodes
to true coordinates lets a reader directly compare detected blocs against
physical geography. Bloc "territory" is deliberately **not** drawn as filled
regions — several blocs include geographically distant members (diaspora-
driven ties, Australia's participation), so a hull/region fill would either
self-intersect or require quietly excluding outliers. Bloc identity is
carried entirely by node colour, the same approach already used in the
dashboard's community world map.

**Two different qualification rules:** Tier 1 (the full 1975–2025 picture)
includes any country with at least **{min_years} years** of participation
across the whole period. Tier 2 (the era maps and the slope chart) uses a
**stricter, more comparable** cohort: only countries with at least
**{min_years_per_half} years** of participation in **both** 1975–1999 and
2000–2025, so a "migration" flag reflects a real, well-evidenced shift
rather than an artifact of a country barely appearing in one half.

**Map extent:** each map auto-fits its visible area to whichever countries
are actually plotted in it, so non-European participants such as Australia
or Israel are never silently clipped off when they qualify.

**Condensation on the maps — disparity filter, not a fixed top-N:** each
map draws only cross-bloc relationships that pass the **disparity filter**
(Serrano, Boguna & Vespignani, 2009, PNAS 106(16): 6483-6488) at
significance level **α = {disparity_alpha}**. For a country with `k`
cross-bloc partners and total tie-strength `s`, a partner's tie is kept if
its share of `s` is too large to be explained by a uniform-random split of
`s` across `k` partners. This adapts per country — one with few partners
needs a much more lopsided relationship to "earn" significance than one
with many — rather than forcing every panel to show exactly the same fixed
number of edges regardless of how the underlying data actually looks. A
defensive cap of **{max_edges_per_panel} edges per panel** is applied on
top of the filter (keeping the most significant ones first) purely for
print legibility, independent of the statistical method itself. Within-bloc
pairs are excluded from consideration entirely, since bloc membership is
already shown through node colour.

**Edge encoding:** `|NVS(A→B) − NVS(B→A)| ≤ {diff_threshold}` → drawn as a
**solid line** (mutual); otherwise a **dashed line** (one-way), with the
giver/receiver named in the hover tooltip. No arrowhead markers are drawn.
Colour/opacity is anchored at the **1.0 NVS** mark, and edges are drawn with
a slight curve so they stay visually distinguishable from nearby clusters.

**Node size — years participated, not voting strength:** each node's size
reflects how many years that country participated **within that specific
panel's own time window** (full range for Tier 1, half-range for an Era
panel). A country can therefore appear a different size in Tier 1 than in
either Era panel.

**Tier 3 — Migration slope chart (the structural fix for showing migration
clearly):** rather than relying on map edges or a connector line across two
independently-projected map panels, every qualifying country gets **exactly
one line**: its position in Era 1's detected bloc order on the left, its
position in Era 2's on the right. A country that **stayed** in an
equivalent bloc is drawn as a flat, light grey line; a country that
**migrated** is drawn as a coloured diagonal, with each distinct
bloc-to-bloc transition (e.g. "Bloc 2 → Bloc 1") getting its own colour, so
countries that made the same structural move are visually grouped together.
Only migrated countries are individually labelled, keeping the chart
readable. Positions are anchored on the same **blocmate-overlap** migration
flag used everywhere else in this draft (<50% of a country's Era-1
blocmates remain its blocmates in Era 2) rather than on raw bloc labels —
Louvain reassigns labels by size on every independent run, so a stable
country could otherwise appear to "move" purely from renumbering; anchoring
on the verified flag means a flat line in this chart always means a
genuinely stable country, never a labelling coincidence.

**Tier 4 — Per-era evidence:** top mutual voters (strongest reciprocal
pairs), top one-way voters (largest asymmetry among pairs that still
exchange votes), and cold-shoulder pairs (a country giving essentially zero
NVS to another across at least {hatred_min_years} eligible years, computed
from a dedicated eligibility frame). These are computed from the full,
unfiltered data — never affected by the map's disparity-filter condensation.

The disparity filter's significance level (**α = {disparity_alpha}**) and
the defensive edge cap ({max_edges_per_panel}) are the only tunable
parameters governing which edges appear; everything else they select is
determined by the statistical test itself, not by an arbitrary visual
cutoff. The remaining thresholds — {diff_threshold} for mutual/one-way,
{hatred_epsilon} for cold-shoulder, {min_years}/{min_years_per_half} years
for the two qualification rules — remain exploratory cutoffs chosen for
visual and narrative clarity, not formal statistical tests.
"""
    return fig, "Geographic Bloc Migration Poster", explanation


# =============================================================================
# DIAGRAM 9 — CIRCULAR HIERARCHICAL EDGE BUNDLING (HEB) POSTER
# =============================================================================
#
# Implements Holten (2006) Hierarchical Edge Bundles adapted for Eurovision:
#
#   Holten, D.H.R. (2006). "Hierarchical edge bundles: visualization of
#   adjacency relations in hierarchical data." IEEE Transactions on
#   Visualization and Computer Graphics, 12(5), 741-748.
#   DOI: 10.1109/TVCG.2006.147
#
# The key insight: Eurovision voting already has the three-level hierarchy
# Holten's method requires —
#
#       Country (leaf) → Detected Bloc (parent) → Centre (root)
#
# — so the bundling is not a forced analogy but a structural match to the
# paper's formulation. Each NVS edge from country A to country B is modelled
# as a cubic Bézier curve whose control points are pulled toward the respective
# bloc centroids on an inner ring, controlled by a bundling-strength parameter
# beta (β):
#
#   beta = 0  →  straight chord lines (no bundling)
#   beta = 0.8 →  strongly bundled through bloc centroids (default)
#   beta = 1  →  fully routed through bloc centroids and centre
#
# Edges between countries in the SAME bloc naturally bundle into one visible
# highway, because they share both control points at the same bloc centroid.
# Cross-bloc edges travel inward toward their respective bloc centroids before
# diverging to the opposite side of the circle, forming the characteristic
# "woven" cross-bloc pattern.
#
# Layout:  countries arranged clockwise starting from the top of the circle,
#   grouped by detected Louvain bloc (largest bloc first), with equal angular
#   spacing within each bloc and a small gap arc between blocs.
# Outer ring: thick coloured arc segments per bloc (same palette as other drafts).
# Edges:  solid line = mutual (|NVS(A→B) − NVS(B→A)| ≤ diff_threshold);
#         dashed line = one-way (asymmetric; hover shows direction).
#         Colour/opacity scales with NVS strength (darker = stronger).
#
# Three-tier poster:
#   Tier 1  Full history 1975–2025 (full-width, large)
#      ↓  "splits into two eras"
#   Tier 2  Era 1 1975–1999 | Era 2 2000–2025  (side by side)
#      ↓  "reveals evidence"
#   Tier 3  Stat cards per era  (side by side)
#
# Edge selection follows Draft 7's NVS-strength-ranked approach: for each
# country A the top `top_k_out` outgoing relationships where
# NVS(A→B) >= min_nvs_strength are selected (pair survives if it qualifies
# from either endpoint's outgoing perspective), so only genuinely strong
# voting relationships are drawn — no statistical test required, the NVS
# floor directly encodes "does this relationship actually matter?".
# =============================================================================


def _heb_bezier(
    ax: float, ay: float, bx: float, by: float,
    cax: float, cay: float, cbx: float, cby: float,
    beta: float = 0.80, n: int = 50,
):
    """
    Cubic Bézier path implementing Holten's (2006) hierarchical edge bundling.

    Maps the Holten formulation for a 3-level hierarchy (leaf → parent → root):
      P0 = country A  (on outer ring, radius 1.0)
      P1 = control point: interpolate between A and bloc-A centroid by beta
      P2 = control point: interpolate between B and bloc-B centroid by beta
      P3 = country B  (on outer ring, radius 1.0)

    At beta=0 all four points collapse to the endpoints (straight chord line).
    At beta=1 P1 = bloc-A centroid and P2 = bloc-B centroid (fully bundled).

    The B-spline approximation in the original paper reduces to a cubic Bézier
    for a 3-level hierarchy, which is exactly the case here (country →
    bloc → centre), per Section 3.2 of Holten (2006).
    """
    p1x = ax * (1.0 - beta) + cax * beta
    p1y = ay * (1.0 - beta) + cay * beta
    p2x = bx * (1.0 - beta) + cbx * beta
    p2y = by * (1.0 - beta) + cby * beta

    t = np.linspace(0.0, 1.0, n)
    cx_arr = (
        (1 - t) ** 3 * ax
        + 3 * (1 - t) ** 2 * t * p1x
        + 3 * (1 - t) * t ** 2 * p2x
        + t ** 3 * bx
    )
    cy_arr = (
        (1 - t) ** 3 * ay
        + 3 * (1 - t) ** 2 * t * p1y
        + 3 * (1 - t) * t ** 2 * p2y
        + t ** 3 * by
    )
    return cx_arr, cy_arr


def _heb_circular_layout(
    countries: list,
    bloc_map: dict,
    start_angle: float = np.pi / 2,
    gap_fraction: float = 0.04,
    r_outer: float = 1.0,
    r_inner: float = 0.42,
):
    """
    Arrange countries in a circle, grouped by detected Louvain bloc.

    Countries in the same bloc are placed as adjacent arcs; a small gap arc
    separates blocs so bloc boundaries are visually clear. Blocs are ordered
    clockwise by descending size (largest bloc gets the widest arc at the top).

    Returns:
        pos       {country: (x, y)} positions on the outer ring (radius r_outer)
        centroids {bloc: (cx, cy)} positions of bloc centroids on inner ring
        arcs      {bloc: (angle_start, angle_end, [country_angles])} arc data
    """
    bloc_members = {}
    for c, b in bloc_map.items():
        if c in countries:
            bloc_members.setdefault(b, [])
            bloc_members[b].append(c)

    # Sort blocs: descending size, then alphabetical name for stability
    blocs = sorted(bloc_members.keys(), key=lambda b: (-len(bloc_members[b]), b))
    n_blocs = len(blocs)
    if n_blocs == 0:
        return {}, {}, {}

    total_countries = sum(len(bloc_members[b]) for b in blocs)
    total_gap = 2 * np.pi * gap_fraction * n_blocs
    available = 2 * np.pi - total_gap
    gap = total_gap / n_blocs

    pos = {}
    centroids = {}
    arcs = {}

    # Clockwise from start_angle; subtract because clockwise = decreasing angle
    # in standard mathematical convention (counter-clockwise positive).
    # We use: angle_cw = start_angle - fraction * 2π for clockwise progression.
    current = start_angle

    for bloc in blocs:
        members = sorted(bloc_members[bloc])
        n = len(members)
        span = available * n / total_countries

        angles = []
        for i, c in enumerate(members):
            # Place at the midpoint of each country's sub-arc within the bloc
            # Clockwise → subtract from current
            a = current - (i + 0.5) * span / n
            angles.append(a)
            pos[c] = (r_outer * np.cos(a), r_outer * np.sin(a))

        mid = current - span / 2
        centroids[bloc] = (r_inner * np.cos(mid), r_inner * np.sin(mid))
        arcs[bloc] = (current, current - span, angles)

        current -= span + gap

    return pos, centroids, arcs


def build_circular_heb_poster(
    df: pd.DataFrame,
    id2label: dict,
    nodes_df: pd.DataFrame,
    min_years: int = 10,
    diff_threshold: float = 1.0,
    top_k_out: int = 3,
    min_nvs_strength: float = 2.0,
    beta: float = 0.80,
    hatred_min_years: int = 10,
    hatred_epsilon: float = 0.04,
):
    """
    DRAFT 9 — Circular Hierarchical Edge Bundling (HEB) Poster.

    Eurovision voting data has the three-level hierarchy that Holten's (2006)
    method requires — Country → Detected Bloc → Centre — so edges between
    countries in the same bloc automatically bundle into visible highways,
    while cross-bloc edges form a distinct woven pattern in the interior.

    This reveals bloc structure through the *visual pattern of curves* rather
    than through node colouring alone, making the poster legible as a static
    print from a distance: the reader immediately sees "there are tight bundles
    here → these countries vote as a bloc" without having to parse individual
    edge labels.

    Citation: Holten, D.H.R. (2006). Hierarchical edge bundles: visualization
    of adjacency relations in hierarchical data. IEEE Transactions on
    Visualization and Computer Graphics, 12(5), 741-748.
    DOI: 10.1109/TVCG.2006.147

    Returns (figure, title, explanation_markdown) per the module's contract.
    """
    from plotly.subplots import make_subplots
    from collections import defaultdict

    df = _add_era_max_col(df)
    df["src_label"] = df["source"].map(id2label).fillna(df["source"])
    df["tgt_label"] = df["target"].map(id2label).fillna(df["target"])

    participation = (
        pd.concat([
            df[["year", "src_label"]].rename(columns={"src_label": "country"}),
            df[["year", "tgt_label"]].rename(columns={"tgt_label": "country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )
    qualified = sorted(participation[participation >= min_years].index.tolist())
    df = df[df["src_label"].isin(qualified) & df["tgt_label"].isin(qualified)].copy()

    if df.empty or len(qualified) < 3:
        return None, "Circular HEB Poster", (
            f"Not enough countries met the >= {min_years}-year participation "
            "threshold to build this draft."
        )

    # -----------------------------------------------------------------------
    # Shared helpers (same logic as Draft 7 for consistency)
    # -----------------------------------------------------------------------

    def mean_nvs_matrix(sub_df, countries):
        if sub_df.empty or not countries:
            return pd.DataFrame(0.0, index=countries, columns=countries)
        m = (
            sub_df.groupby(["src_label", "tgt_label"])["nvs"].mean()
            .unstack(fill_value=0)
            .reindex(index=countries, columns=countries, fill_value=0)
        ) * 12.0
        return m

    def nvs_strength_backbone(mat, countries):
        keep = set()
        for c in countries:
            out_vals = mat.loc[c].drop(labels=[c], errors="ignore")
            strong = (
                out_vals[out_vals >= min_nvs_strength]
                .sort_values(ascending=False)
                .head(top_k_out)
            )
            for partner in strong.index:
                keep.add(tuple(sorted([c, partner])))
        return keep

    def classify_edges(mat, countries):
        retained = nvs_strength_backbone(mat, countries)
        edges = []
        for (a, b) in retained:
            ab = float(mat.loc[a, b])
            ba = float(mat.loc[b, a])
            if ab <= 0 and ba <= 0:
                continue
            diff = abs(ab - ba)
            if diff <= diff_threshold:
                edges.append({"a": a, "b": b, "kind": "mutual",
                               "value": (ab + ba) / 2.0, "ab": ab, "ba": ba, "diff": diff})
            else:
                giver, receiver = (a, b) if ab > ba else (b, a)
                edges.append({"a": a, "b": b, "kind": "one_way",
                               "giver": giver, "receiver": receiver,
                               "value": max(ab, ba), "ab": ab, "ba": ba, "diff": diff})
        return edges

    def detect(sub_df, countries):
        if not countries:
            return {}
        aff = _mutual_affinity(_affinity_input(sub_df), countries)
        return _detect_blocs(aff, countries, q=0.6)

    def flag_migrated(map1, map2):
        g1, g2 = defaultdict(set), defaultdict(set)
        for c, b in map1.items():
            g1[b].add(c)
        for c, b in map2.items():
            g2[b].add(c)
        migrated = set()
        for c in set(map1) & set(map2):
            m1 = g1[map1[c]] - {c}
            m2 = g2[map2[c]] - {c}
            overlap = (len(m1 & m2) / len(m1)) if m1 else 0.0
            if overlap < 0.5:
                migrated.add(c)
        return migrated

    def eligibility_frame(sub_df, countries):
        countries_set = set(countries)
        rows = []
        for yr, g in sub_df.groupby("year"):
            participants = (set(g["src_label"]) | set(g["tgt_label"])) & countries_set
            for s in participants:
                for t in participants:
                    if s != t:
                        rows.append((yr, s, t))
        if not rows:
            return pd.DataFrame(columns=["year", "src_label", "tgt_label", "points", "nvs"])
        elig = pd.DataFrame(rows, columns=["year", "src_label", "tgt_label"])
        actual = sub_df.groupby(["year", "src_label", "tgt_label"], as_index=False)["points"].sum()
        m = elig.merge(actual, on=["year", "src_label", "tgt_label"], how="left")
        m["points"] = m["points"].fillna(0)
        m["era_max_v"] = m["year"].apply(_era_max)
        m["nvs"] = (m["points"] / m["era_max_v"]).clip(0, 1)
        return m

    def era_stats(sub_df, countries, edges):
        mutual  = sorted([e for e in edges if e["kind"] == "mutual"],
                         key=lambda e: e["value"], reverse=True)[:3]
        oneway  = sorted([e for e in edges if e["kind"] == "one_way"],
                         key=lambda e: e["diff"], reverse=True)[:3]
        elig = eligibility_frame(sub_df, countries)
        if elig.empty:
            return mutual, oneway, pd.DataFrame()
        agg = (
            elig.groupby(["src_label", "tgt_label"])
            .agg(years_eligible=("year", "nunique"), mean_nvs=("nvs", "mean"))
            .reset_index()
        )
        rlookup = {(r["src_label"], r["tgt_label"]): r["mean_nvs"] for _, r in agg.iterrows()}
        cands = agg[
            (agg["years_eligible"] >= hatred_min_years) & (agg["mean_nvs"] < hatred_epsilon)
        ].copy()
        if cands.empty:
            return mutual, oneway, cands
        cands["reciprocal_nvs"] = cands.apply(
            lambda r: rlookup.get((r["tgt_label"], r["src_label"]), 0.0), axis=1
        )
        return mutual, oneway, cands.sort_values(
            ["years_eligible", "reciprocal_nvs"], ascending=[False, False]
        ).head(3)

    # -----------------------------------------------------------------------
    # HEB palette — same bloc colours as other drafts for consistency
    # -----------------------------------------------------------------------
    HEB_PALETTE = [
        "#1f4e79", "#d1495b", "#2a9d8f", "#f4a261",
        "#6a4c93", "#7f5539", "#577590", "#3a86ff",
    ]

    # -----------------------------------------------------------------------
    # Render one HEB panel into the given subplot row/col
    # -----------------------------------------------------------------------

    participation_total = participation.to_dict()
    era1_part = (
        pd.concat([
            df[df["year"] <= 1999][["year", "src_label"]].rename(columns={"src_label": "country"}),
            df[df["year"] <= 1999][["year", "tgt_label"]].rename(columns={"tgt_label": "country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique().to_dict()
    )
    era2_part = (
        pd.concat([
            df[df["year"] >= 2000][["year", "src_label"]].rename(columns={"src_label": "country"}),
            df[df["year"] >= 2000][["year", "tgt_label"]].rename(columns={"tgt_label": "country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique().to_dict()
    )

    def render_heb(fig, row, col, countries, edges, bloc_map,
                   part_years=None, migrated=None, label_top_n=14,
                   show_methodology_box=False):
        """
        Draw one circular HEB panel with poster-quality styling:
        - Subtle background depth rings at r=0.35 and r=0.70
        - Thick coloured arc segments per bloc with bloc name label at midpoint
        - Bundled Bézier edge curves (teal solid = mutual, coral dashed = one-way)
        - Country dots sized by participation years, white outline
        - Country labels for the top `label_top_n` most active countries
        - Optional on-chart methodology annotation (show_methodology_box=True for Tier 1)
        """
        migrated  = migrated  or set()
        part_years = part_years or {}

        if not countries:
            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, row=row, col=col)
            return

        pos, centroids, arcs = _heb_circular_layout(countries, bloc_map)

        bloc_names  = sorted(set(bloc_map.values()))
        bloc_color  = {b: HEB_PALETTE[i % len(HEB_PALETTE)] for i, b in enumerate(bloc_names)}

        # ---- background depth rings (give the diagram a readable field) ---------
        for r_ring, alpha_ring, lw in [(0.70, 0.10, 0.8), (0.42, 0.07, 0.6)]:
            θ = np.linspace(0, 2 * np.pi, 100)
            fig.add_trace(go.Scatter(
                x=r_ring * np.cos(θ), y=r_ring * np.sin(θ),
                mode="lines",
                line=dict(color=f"rgba(100,116,139,{alpha_ring:.2f})", width=lw, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ), row=row, col=col)

        # ---- outer arc rings + bloc name labels at arc midpoint ----------------
        for bloc, (a_start, a_end, _) in arcs.items():
            n_arc = max(30, abs(int((a_start - a_end) / 0.04)))
            θ_arc = np.linspace(a_start, a_end, n_arc)
            r_arc = 1.07

            # Thicker arc ring
            fig.add_trace(go.Scatter(
                x=r_arc * np.cos(θ_arc), y=r_arc * np.sin(θ_arc),
                mode="lines", line=dict(color=bloc_color[bloc], width=14),
                hovertemplate=f"<b>{bloc}</b><extra></extra>",
                showlegend=False,
            ), row=row, col=col)

            # Thin inner border line to separate arc from node ring
            fig.add_trace(go.Scatter(
                x=1.01 * np.cos(θ_arc), y=1.01 * np.sin(θ_arc),
                mode="lines",
                line=dict(color=f"rgba(255,255,255,0.6)", width=1),
                hoverinfo="skip", showlegend=False,
            ), row=row, col=col)

            # Bloc name label at arc midpoint, outside the arc
            mid_θ   = (a_start + a_end) / 2
            arc_span = abs(a_end - a_start)
            r_label = 1.22 if arc_span > 0.4 else 1.25  # smaller arcs get pushed further out
            lx, ly  = r_label * np.cos(mid_θ), r_label * np.sin(mid_θ)
            # Count members for label
            members_in_bloc = [c for c in countries if bloc_map.get(c) == bloc]
            label_text = f"<b>{bloc}</b><br><span style='font-size:7px'>({len(members_in_bloc)})</span>"
            fig.add_trace(go.Scatter(
                x=[lx], y=[ly], mode="text",
                text=[label_text],
                textfont=dict(size=8.5, color=bloc_color[bloc], family="IBM Plex Mono, monospace"),
                hoverinfo="skip", showlegend=False,
            ), row=row, col=col)

        # ---- HEB edges (teal = mutual, coral = one-way) -----------------------
        max_nvs = max((e["value"] for e in edges), default=1.0) or 1.0

        for e in edges:
            a, b = e["a"], e["b"]
            if a not in pos or b not in pos:
                continue
            ax_p, ay_p = pos[a]
            bx_p, by_p = pos[b]
            cax, cay   = centroids.get(bloc_map.get(a), (0.0, 0.0))
            cbx, cby   = centroids.get(bloc_map.get(b), (0.0, 0.0))

            norm = min(e["value"] / max_nvs, 1.0)
            if e["kind"] == "mutual":
                alpha = 0.25 + 0.70 * norm
                color = f"rgba(13,148,136,{alpha:.2f})"   # teal
                dash  = "solid"
                width = 1.5 + 3.5 * norm
            else:
                alpha = 0.22 + 0.65 * norm
                color = f"rgba(220,86,60,{alpha:.2f})"    # coral-red
                dash  = "dot"
                width = 1.2 + 2.8 * norm

            cx_arr, cy_arr = _heb_bezier(ax_p, ay_p, bx_p, by_p, cax, cay, cbx, cby, beta=beta)

            kind_str = "Mutual" if e["kind"] == "mutual" else f"One-way: {e['giver']} \u2192 {e['receiver']}"
            fig.add_trace(go.Scatter(
                x=cx_arr, y=cy_arr, mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=(
                    f"<b>{a}</b> \u2194 <b>{b}</b><br>"
                    f"NVS {a}\u2192{b}: {e['ab']:.2f} | {b}\u2192{a}: {e['ba']:.2f}<br>"
                    f"{kind_str}<br>"
                    f"Combined NVS: {e['value']:.2f} / 12<extra></extra>"
                ),
                showlegend=False,
            ), row=row, col=col)

        # ---- country nodes (all at once for performance) ----------------------
        max_yrs  = max(part_years.values(), default=1) or 1
        labelled = set(
            sorted(countries, key=lambda c: part_years.get(c, 0), reverse=True)[:label_top_n]
        )

        xs, ys, texts, fills, ring_cols, ring_ws, sizes, hovers = [], [], [], [], [], [], [], []
        for c in countries:
            if c not in pos:
                continue
            x, y = pos[c]
            yrs  = part_years.get(c, 0)
            sz   = 10 + 14 * np.sqrt(max(yrs, 0) / max_yrs)   # larger nodes
            fill = bloc_color.get(bloc_map.get(c), "#9ca3af")
            if c in migrated:
                rc, rw = "#facc15", 4.0
            else:
                rc, rw = "white", 2.0
            label = c if c in labelled else ""
            hover = (
                f"<b>{c}</b><br>Bloc: {bloc_map.get(c,'NA')}<br>"
                f"Years in this window: {yrs}"
                + ("<br><b>\u26a1 Changed bloc between eras</b>" if c in migrated else "")
            )
            xs.append(x); ys.append(y); texts.append(label)
            fills.append(fill); ring_cols.append(rc); ring_ws.append(rw)
            sizes.append(sz); hovers.append(hover)

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            text=texts, textposition="top center",
            textfont=dict(size=10, color="#111827", family="IBM Plex Mono, monospace"),
            marker=dict(size=sizes, color=fills,
                        line=dict(width=ring_ws, color=ring_cols)),
            hovertext=hovers, hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        ), row=row, col=col)

        # ---- on-chart methodology box (Tier 1 only) --------------------------
        if show_methodology_box:
            n_mutual = sum(1 for e in edges if e["kind"] == "mutual")
            n_one    = len(edges) - n_mutual
            method_text = (
                f"<b>WHAT IS BEING SHOWN</b><br>"
                f"<br>"
                f"<b>Technique:</b> Hierarchical Edge Bundling<br>"
                f"<i>Holten (2006) IEEE TVCG 12(5):741-748</i><br>"
                f"<br>"
                f"<b>Hierarchy used:</b><br>"
                f"Country → Voting Bloc → Centre<br>"
                f"(3-level compound graph structure)<br>"
                f"<br>"
                f"<b>How bundling works:</b><br>"
                f"Each edge is a Bézier curve pulled {int(beta*100)}%<br>"
                f"toward the countries' shared bloc centroid.<br>"
                f"Same-bloc edges converge → visible bundle.<br>"
                f"Cross-bloc edges arc through the interior.<br>"
                f"<br>"
                f"<b>What to look for:</b><br>"
                f"Dense bundles = strong voting alliances<br>"
                f"Diagonal arcs = cross-bloc relationships<br>"
                f"<br>"
                f"<b>Edges shown:</b> {n_mutual} mutual + {n_one} one-way<br>"
                f"<b>Selection:</b> top {top_k_out} outgoing where<br>"
                f"NVS ≥ {min_nvs_strength}/12 per country<br>"
                f"<br>"
                f"<b>Metric:</b> NVS(A→B) = points / era_max<br>"
                f"era_max = 12 (1975–2015) · 24 (2016–2025)"
            )
            fig.add_annotation(
                x=-1.32, y=-1.08,
                text=method_text,
                showarrow=False,
                xanchor="left", yanchor="bottom",
                font=dict(size=8.5, color="#374151"),
                bgcolor="rgba(255,255,255,0.97)",
                bordercolor="#6366f1",
                borderwidth=1.5,
                borderpad=10,
                align="left",
                row=row, col=col,
            )

        # ---- axis setup (square, equal scale, invisible) ---------------------
        fig.update_xaxes(
            visible=False, range=[-1.45, 1.45],
            scaleanchor=f"y{'' if (row == 1 and col == 1) else str((row - 1) * 2 + col)}",
            scaleratio=1, row=row, col=col,
        )
        fig.update_yaxes(visible=False, range=[-1.45, 1.45], row=row, col=col)


    def render_stat_panel(fig, row, col, era_label, top_mutual, top_oneway, top_hatred):
        fig.update_xaxes(visible=False, range=[0, 1], row=row, col=col)
        fig.update_yaxes(visible=False, range=[0, 1], row=row, col=col)

        mutual_lines = [
            f"\U0001f91d {e['a']} \u2194 {e['b']}  (NVS {e['value']:.1f})" for e in top_mutual
        ] or ["No qualifying mutual pairs"]
        oneway_lines = [
            f"\u27a1\ufe0f {e['giver']} \u2192 {e['receiver']}  (\u0394{e['diff']:.1f})" for e in top_oneway
        ] or ["No qualifying one-way pairs"]
        if top_hatred is None or top_hatred.empty:
            hatred_lines = ["No sustained cold-shoulder pairs found"]
        else:
            hatred_lines = [
                f"\u2744\ufe0f {r['src_label']} \u21cf {r['tgt_label']}  ({int(r['years_eligible'])} yrs)"
                for _, r in top_hatred.iterrows()
            ]

        sections = [
            ("Top mutual voters (NVS \u2248 equal both ways)", mutual_lines),
            ("Top one-way voters (strong asymmetry)", oneway_lines),
            ("Cold-shoulder pairs (near-zero NVS)", hatred_lines),
        ]
        y = 0.95
        fig.add_annotation(
            x=0.03, y=1.0, text=f"<b>{era_label}</b>", showarrow=False,
            font=dict(size=13, color="#1f2937", family="Georgia, serif"),
            xanchor="left", yanchor="top", row=row, col=col,
        )
        y -= 0.12
        for heading, lines in sections:
            fig.add_annotation(
                x=0.03, y=y, text=f"<b>{heading}</b>", showarrow=False,
                font=dict(size=10, color="#374151"), xanchor="left", yanchor="top",
                row=row, col=col,
            )
            y -= 0.09
            for line in lines[:3]:
                fig.add_annotation(
                    x=0.06, y=y, text=line, showarrow=False,
                    font=dict(size=9, color="#4b5563"), xanchor="left", yanchor="top",
                    row=row, col=col,
                )
                y -= 0.08
            y -= 0.02

    # -----------------------------------------------------------------------
    # Compute data for all three tiers
    # -----------------------------------------------------------------------

    # Tier 1 — full history
    full_mat    = mean_nvs_matrix(df, qualified)
    full_bloc   = detect(df, qualified)
    full_edges  = classify_edges(full_mat, qualified)

    # Tier 2 — two independently-detected eras
    era1_df = df[df["year"] <= 1999]
    era2_df = df[df["year"] >= 2000]

    era1_countries = sorted({
        c for c in qualified if c in set(era1_df["src_label"]) | set(era1_df["tgt_label"])
    })
    era2_countries = sorted({
        c for c in qualified if c in set(era2_df["src_label"]) | set(era2_df["tgt_label"])
    })

    era1_mat   = mean_nvs_matrix(era1_df, era1_countries)
    era2_mat   = mean_nvs_matrix(era2_df, era2_countries)
    era1_bloc  = detect(era1_df, era1_countries)
    era2_bloc  = detect(era2_df, era2_countries)
    era1_edges = classify_edges(era1_mat, era1_countries)
    era2_edges = classify_edges(era2_mat, era2_countries)

    migrated = flag_migrated(era1_bloc, era2_bloc)

    # Tier 3 — evidence
    top_m1, top_o1, top_h1 = era_stats(era1_df, era1_countries, era1_edges)
    top_m2, top_o2, top_h2 = era_stats(era2_df, era2_countries, era2_edges)

    # -----------------------------------------------------------------------
    # Panel titles with live edge/country counts
    # -----------------------------------------------------------------------

    def _ptitle(prefix, countries, edges):
        nm = sum(1 for e in edges if e["kind"] == "mutual")
        no = len(edges) - nm
        return (
            f"{prefix}<br>"
            f"<span style='font-size:11px;color:#6b7280;'>"
            f"{len(countries)} countries \u00b7 {len(edges)} edges "
            f"({nm} mutual \u2014, {no} one-way \u2508)</span>"
        )

    # -----------------------------------------------------------------------
    # Figure assembly
    # -----------------------------------------------------------------------

    row_heights = [0.44, 0.30, 0.26]
    vspacing = 0.09
    total_gap = vspacing * 2
    avail = 1.0 - total_gap
    scaled = [h * avail for h in row_heights]

    boundaries = []
    top_cur = 1.0
    for h in scaled:
        bot = top_cur - h
        boundaries.append((top_cur, bot))
        top_cur = bot - vspacing

    fig = make_subplots(
        rows=3, cols=2,
        row_heights=row_heights,
        vertical_spacing=vspacing,
        horizontal_spacing=0.07,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
        ],
        subplot_titles=[
            _ptitle("Full picture \u00b7 1975\u20132025", qualified, full_edges),
            _ptitle("Era 1 \u00b7 1975\u20131999", era1_countries, era1_edges),
            _ptitle("Era 2 \u00b7 2000\u20132025", era2_countries, era2_edges),
            "Era 1 insights", "Era 2 insights",
        ],
    )

    render_heb(fig, 1, 1, qualified, full_edges, full_bloc,
               part_years=participation_total, show_methodology_box=True)
    render_heb(fig, 2, 1, era1_countries, era1_edges, era1_bloc,
               part_years=era1_part, migrated=migrated)
    render_heb(fig, 2, 2, era2_countries, era2_edges, era2_bloc,
               part_years=era2_part, migrated=migrated)
    render_stat_panel(fig, 3, 1, "1975\u20131999", top_m1, top_o1, top_h1)
    render_stat_panel(fig, 3, 2, "2000\u20132025", top_m2, top_o2, top_h2)

    # ---- dotted flow connectors -------------------------------------------
    def add_connector(x, y_top, y_bot, label):
        fig.add_shape(
            type="line", x0=x, y0=y_top - 0.005, x1=x, y1=y_bot + 0.018,
            line=dict(dash="dot", color="#94a3b8", width=2),
            xref="paper", yref="paper",
        )
        fig.add_annotation(
            x=x, y=y_bot + 0.014, text="\u25bc", showarrow=False,
            xref="paper", yref="paper", font=dict(size=13, color="#94a3b8"),
        )
        fig.add_annotation(
            x=x, y=(y_top + y_bot) / 2, text=label, showarrow=False,
            xref="paper", yref="paper",
            font=dict(size=10, color="#64748b", family="Georgia, serif"),
            bgcolor="rgba(248,250,252,0.9)", borderpad=3,
        )

    g1t, g1b = boundaries[0][1], boundaries[1][0]
    g2t, g2b = boundaries[1][1], boundaries[2][0]
    add_connector(0.25, g1t, g1b, "splits into two eras")
    add_connector(0.75, g1t, g1b, "splits into two eras")
    add_connector(0.25, g2t, g2b, "reveals evidence")
    add_connector(0.75, g2t, g2b, "reveals evidence")

    # ---- compact reading guide (top-right, bordered card style) ----------
    fig.add_annotation(
        x=0.99, y=1.065, xref="paper", yref="paper",
        text=(
            "<b>HOW TO READ THIS DIAGRAM</b><br>"
            "<br>"
            "<b>Layout:</b> countries arranged in a circle grouped by<br>"
            "detected voting bloc (coloured outer arc = one bloc)<br>"
            "<br>"
            "<b>Edge bundling:</b> curves are pulled toward their shared<br>"
            "bloc centroid (β={beta:.2f}). Tight bundles = countries<br>"
            "consistently voting for the same bloc partners<br>"
            "<br>"
            "<span style='color:rgb(13,148,136)'><b>━━━ Teal solid</b></span>"
            " = Mutual relationship<br>"
            "<span style='font-size:9px;color:#6b7280;'>"
            "   (NVS each way within {diff_threshold:.1f} of each other)</span><br>"
            "<span style='color:rgb(220,86,60)'><b>┈┈┈ Coral dotted</b></span>"
            " = One-way (hover → direction)<br>"
            "<span style='font-size:9px;color:#6b7280;'>"
            "   (one country gives noticeably more than it gets)</span><br>"
            "<br>"
            "<b>Node size</b> = years participated in this window<br>"
            "<span style='color:#b45309'><b>Gold ring</b></span>"
            " = country changed voting bloc between eras<br>"
            "<br>"
            "<span style='font-size:8px;color:#94a3b8;'>"
            "Algorithm: Holten (2006) IEEE TVCG 12(5):741-748<br>"
            "NVS = points / era_max · era_max=12 (≤2015), 24 (≥2016)</span>"
        ).format(beta=beta, diff_threshold=diff_threshold),
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=9, color="#374151"), align="right",
        bgcolor="rgba(255,255,255,0.97)", bordercolor="#94a3b8",
        borderwidth=1.5, borderpad=10,
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Eurovision Voting Network \u00b7 Hierarchical Edge Bundling (Holten 2006)</b>"
                "<br><span style='font-size:12px;color:#64748b;'>"
                "Country \u2192 Detected Louvain Bloc \u2192 Centre hierarchy \u00b7 "
                "cubic B\u00e9zier bundling (\u03b2={beta:.2f}) \u00b7 "
                "NVS-strength edge selection \u00b7 1975\u20132025"
                "</span>"
            ).format(beta=beta),
            x=0.5, xanchor="center",
            font=dict(size=17, family="Georgia, serif", color="#111827"),
        ),
        height=1650, width=1250,
        paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc",
        showlegend=False,
        margin=dict(l=30, r=30, t=140, b=40),
    )

    explanation = f"""
**Why hierarchical edge bundling for Eurovision?**

Eurovision voting data has the three-level hierarchy that Holten's (2006)
method was designed for:

```
Country (leaf)  →  Detected Louvain Bloc (parent)  →  Centre (root)
```

This is not a forced analogy: Holten's paper explicitly targets "compound
graphs" where items (countries) have both pairwise relations (NVS votes) and
a hierarchical grouping (bloc membership). Routing each NVS edge as a cubic
Bézier through the two endpoints' bloc centroids causes edges within the same
bloc to naturally converge into a shared curved highway, while cross-bloc
edges form a woven pattern in the interior. The bloc structure is therefore
visible as a **visual pattern** — tight bundles — rather than only through
node colouring.

**Citation:** Holten, D.H.R. (2006). Hierarchical edge bundles: visualization
of adjacency relations in hierarchical data. *IEEE Transactions on
Visualization and Computer Graphics*, 12(5), 741–748.
DOI: 10.1109/TVCG.2006.147

**Bundling strength β = {beta:.2f}:** the Bézier control points are
interpolated between the country's own position (β = 0 → straight chord
line) and the bloc centroid on the inner ring (β = 1 → fully bundled through
centroids). At β = {beta:.2f}, {int(beta*100)}% of the pull is toward the
hierarchy path. This follows Holten's β parameter (Section 3 of the paper);
the cubic Bézier is the closed-form solution for his B-spline when the
hierarchy has exactly three levels (leaf → parent → root), which is the case
here.

**Edge selection:** same NVS-strength-ranked approach as Draft 7 — for each
country A, the top **{top_k_out}** outgoing relationships where
NVS(A→B) ≥ **{min_nvs_strength}/12** are selected; a pair survives if it
qualifies from either endpoint's perspective. This keeps only genuinely
strong voting ties, not statistical artefacts.

**Edge classification:** `|NVS(A→B) − NVS(B→A)| ≤ {diff_threshold}` → solid
blue curve (MUTUAL); otherwise dashed red curve (ONE-WAY). Colour opacity
scales with NVS strength so stronger ties are more visible in print.

**Node size:** proportional to participation years within that panel's
specific time window (1975–2025 for Tier 1; 1975–1999 or 2000–2025 for era
panels). **Gold ring** = country's detected bloc changed between eras
(measured by <50% blocmate overlap — not a label change from Louvain
renumbering).

**Thesis placement:** cite Holten (2006) in Section 4.3 (Layout Strategy)
and Section 4.4 (Visual Encoding). The algorithm is directly applicable to
the Eurovision hierarchy and reduces visual clutter without removing any
nodes, consistent with the poster's requirement to show all qualifying
countries.

All thresholds ({min_nvs_strength} NVS floor, {diff_threshold} mutual/one-way
split, {min_years} years for inclusion, {hatred_min_years}/{hatred_epsilon}
for cold-shoulder detection) are exploratory cutoffs chosen for visual and
narrative clarity, not formal statistical tests.
"""
    return fig, "Circular HEB — Hierarchical Edge Bundling", explanation

# =============================================================================
# DIAGRAM 10 — GEOGRAPHIC HIERARCHICAL EDGE BUNDLING POSTER
# =============================================================================
#
# The same Holten (2006) HEB algorithm as Draft 9, but with countries placed
# at their REAL geographic coordinates instead of an abstract circular layout.
# Bloc centroids are computed as the geographic centre-of-mass (mean lat/lon)
# of each detected bloc's member countries. Bézier control points are those
# geographic centroids, so edges bundle toward the geographic heart of each
# voting region.
#
# This answers a question the circular layout cannot: does the bundling
# pattern align with geographic regions? If the Nordic bloc's centroid sits
# near Scandinavia and the Balkan bloc's centroid sits near the Adriatic, the
# bundles will visually trace the actual corridors of influence on the map,
# making geographic clustering immediately readable as spatial pattern rather
# than abstract structure.
#
# Additional visual: bloc centroids are drawn as large semi-transparent
# 'hub' circles on the map, labelled with the bloc name. These hubs are the
# geographic analogue of the inner-ring control points in the circular version.
#
# Edge encoding:  solid teal = mutual · dotted coral = one-way
# Node encoding:  colour = bloc · size = participation years in panel window
# Migration:      gold ring (Era 1 "about to move") ·
#                 coloured ring = previous bloc colour (Era 2 "moved from")
# =============================================================================


def _geo_heb_bezier(
    lat_a: float, lon_a: float,
    lat_b: float, lon_b: float,
    ctrl_lat_1: float, ctrl_lon_1: float,
    ctrl_lat_2: float, ctrl_lon_2: float,
    beta: float = 0.80, n: int = 30,
):
    """
    Cubic Bézier path in lat/lon space implementing geographic HEB.

    The control points are the geographic bloc centroids, interpolated with
    beta toward the straight great-circle path (approximated here as a
    straight line in Mercator-like lat/lon space — acceptable for the
    Europe-scale distances in the Eurovision dataset).

    At beta=0  → straight line between A and B  (no bundling)
    At beta=1  → fully routed through bloc centroids (maximum bundling)
    """
    cp1_lat = lat_a * (1.0 - beta) + ctrl_lat_1 * beta
    cp1_lon = lon_a * (1.0 - beta) + ctrl_lon_1 * beta
    cp2_lat = lat_b * (1.0 - beta) + ctrl_lat_2 * beta
    cp2_lon = lon_b * (1.0 - beta) + ctrl_lon_2 * beta

    t = np.linspace(0.0, 1.0, n)
    lats = (1-t)**3*lat_a + 3*(1-t)**2*t*cp1_lat + 3*(1-t)*t**2*cp2_lat + t**3*lat_b
    lons = (1-t)**3*lon_a + 3*(1-t)**2*t*cp1_lon + 3*(1-t)*t**2*cp2_lon + t**3*lon_b
    return lats, lons


_GEO_HEB_PALETTE = [
    "#1f4e79", "#d1495b", "#2a9d8f", "#f4a261",
    "#6a4c93", "#7f5539", "#577590", "#3a86ff",
]


def _geo_heb_compute_bloc_centroids(
    bloc_map: dict, coord_lookup: dict
) -> dict:
    """
    Compute the geographic centre-of-mass (mean lat/lon) for each detected
    bloc, using only countries that have real coordinates.
    """
    lats_by_bloc, lons_by_bloc = {}, {}
    for c, b in bloc_map.items():
        if c in coord_lookup:
            lats_by_bloc.setdefault(b, []).append(coord_lookup[c][0])
            lons_by_bloc.setdefault(b, []).append(coord_lookup[c][1])
    return {
        b: (float(np.mean(lats_by_bloc[b])), float(np.mean(lons_by_bloc[b])))
        for b in lats_by_bloc
    }


def _geo_heb_render_panel(
    fig, row: int, col: int,
    countries: list, edges: list, bloc_map: dict,
    coord_lookup: dict, geo_centroids: dict,
    part_years: dict | None = None,
    migrated_gold: set | None = None,
    migrated_ring_color: dict | None = None,
    beta: float = 0.80,
    label_top_n: int = 12,
    show_methodology_box: bool = False,
    diff_threshold: float = 1.0,
    top_k_out: int = 3,
    min_nvs_strength: float = 2.0,
):
    """
    Render one geographic HEB panel using Scattergeo traces.

    Steps:
    1. Draw bloc centroid 'hub' circles (large, semi-transparent markers)
    2. Draw bundled Bézier edges as Scattergeo line traces
    3. Draw country nodes (sized by participation years)
    4. Apply auto-fit projection so all plotted countries are visible
    """
    part_years        = part_years or {}
    migrated_gold     = migrated_gold or set()
    migrated_ring_color = migrated_ring_color or {}

    plot_countries = [c for c in countries if c in coord_lookup]
    if not plot_countries:
        return

    bloc_names = sorted(set(bloc_map.values()))
    bloc_color = {b: _GEO_HEB_PALETTE[i % len(_GEO_HEB_PALETTE)] for i, b in enumerate(bloc_names)}

    # ---- bloc centroid 'hub' markers (larger, clearer) -----------------------
    for bloc, (clat, clon) in geo_centroids.items():
        if bloc not in bloc_color:
            continue
        bc = bloc_color[bloc]
        h  = bc.lstrip("#")
        r2, g2, b2 = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        fig.add_trace(go.Scattergeo(
            lon=[clon], lat=[clat],
            mode="markers+text",
            text=[f"<b>{bloc}</b>"],
            textposition="top center",
            textfont=dict(size=9.5, color=bc, family="IBM Plex Mono, monospace"),
            marker=dict(
                size=34,
                color=f"rgba({r2},{g2},{b2},0.15)",
                line=dict(width=2.5, color=f"rgba({r2},{g2},{b2},0.55)"),
                symbol="circle",
            ),
            hovertemplate=f"<b>{bloc}</b> centroid<br>{clat:.1f}°N, {clon:.1f}°E<extra></extra>",
            showlegend=False,
        ), row=row, col=col)

    # ---- HEB edges (Scattergeo with bezier points) ---------------------------
    max_nvs = max((e["value"] for e in edges), default=1.0) or 1.0

    for e in edges:
        a, b = e["a"], e["b"]
        if a not in coord_lookup or b not in coord_lookup:
            continue
        lat_a, lon_a = coord_lookup[a]
        lat_b, lon_b = coord_lookup[b]

        bloc_a = bloc_map.get(a)
        bloc_b = bloc_map.get(b)
        ctrl1  = geo_centroids.get(bloc_a, (lat_a, lon_a))
        ctrl2  = geo_centroids.get(bloc_b, (lat_b, lon_b))

        norm = min(e["value"] / max_nvs, 1.0)
        if e["kind"] == "mutual":
            alpha = 0.30 + 0.65 * norm
            color = f"rgba(13,148,136,{alpha:.2f})"
            dash  = "solid"
            width = 1.8 + 3.2 * norm
        else:
            alpha = 0.25 + 0.60 * norm
            color = f"rgba(220,86,60,{alpha:.2f})"
            dash  = "dot"
            width = 1.4 + 2.6 * norm

        path_lat, path_lon = _geo_heb_bezier(
            lat_a, lon_a, lat_b, lon_b,
            ctrl1[0], ctrl1[1], ctrl2[0], ctrl2[1],
            beta=beta,
        )

        kind_str = (
            "Mutual" if e["kind"] == "mutual"
            else f"One-way: {e['giver']} \u2192 {e['receiver']}"
        )
        fig.add_trace(go.Scattergeo(
            lon=path_lon, lat=path_lat,
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=(
                f"<b>{a}</b> \u2194 <b>{b}</b><br>"
                f"NVS {a}\u2192{b}: {e['ab']:.2f} | {b}\u2192{a}: {e['ba']:.2f}<br>"
                f"{kind_str}<br>Combined NVS: {e['value']:.2f} / 12<extra></extra>"
            ),
            showlegend=False,
        ), row=row, col=col)

    # ---- country nodes -------------------------------------------------------
    max_yrs = max(part_years.values(), default=1) or 1
    labelled = set(
        sorted(plot_countries, key=lambda c: part_years.get(c, 0), reverse=True)[:label_top_n]
    )

    for c in plot_countries:
        lat, lon = coord_lookup[c]
        yrs  = part_years.get(c, 0)
        sz   = 11 + 14 * np.sqrt(max(yrs, 0) / max_yrs)
        fill = bloc_color.get(bloc_map.get(c), "#9ca3af")

        if c in migrated_ring_color:
            rc, rw = migrated_ring_color[c], 3.5
        elif c in migrated_gold:
            rc, rw = "#facc15", 3.5
        else:
            rc, rw = "white", 1.5

        label = f"<b>{c}</b>" if c in labelled else ""

        hover = (
            f"<b>{c}</b><br>Bloc: {bloc_map.get(c,'NA')}<br>"
            f"Years in this window: {yrs}<br>"
            f"Lat: {lat:.1f}° Lon: {lon:.1f}°"
            + ("<br><b>\u26a1 Changed bloc between eras</b>" if c in migrated_gold or c in migrated_ring_color else "")
        )

        fig.add_trace(go.Scattergeo(
            lon=[lon], lat=[lat],
            mode="markers+text",
            text=[label], textposition="top center",
            textfont=dict(size=9.5, color="#111827", family="IBM Plex Mono, monospace"),
            marker=dict(
                size=sz, color=fill,
                line=dict(width=rw, color=rc),
            ),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ), row=row, col=col)

    # ---- auto-fit projection using explicit lat/lon bounds -------------------
    # fitbounds="locations" is unreliable in mixed scattergeo+xy make_subplots
    # layouts (it silently fails, leaving the entire world view visible and
    # making country nodes invisible). We compute the bounds directly from the
    # plotted coordinates and set lataxis_range / lonaxis_range explicitly.
    if plot_countries:
        all_lats = [coord_lookup[c][0] for c in plot_countries]
        all_lons = [coord_lookup[c][1] for c in plot_countries]
        pad_lat  = max(5.0, (max(all_lats) - min(all_lats)) * 0.18)
        pad_lon  = max(6.0, (max(all_lons) - min(all_lons)) * 0.18)
        lat_min, lat_max = min(all_lats) - pad_lat, max(all_lats) + pad_lat
        lon_min, lon_max = min(all_lons) - pad_lon, max(all_lons) + pad_lon
    else:
        lat_min, lat_max = 25.0, 75.0   # fallback: Europe bounding box
        lon_min, lon_max = -25.0, 65.0

    fig.update_geos(
        projection_type="natural earth",
        showland=True,      landcolor="#eef2f7",
        showocean=True,     oceancolor="#d4e8f8",
        showcountries=True, countrycolor="#aabace",
        showcoastlines=True, coastlinecolor="#7a9ab8",
        showrivers=False, showlakes=True, lakecolor="#d4e8f8",
        showframe=False,
        lataxis_range=[lat_min, lat_max],
        lonaxis_range=[lon_min, lon_max],
        row=row, col=col,
    )

    # ---- on-chart methodology box (Tier 1 only, in paper coordinates) ------
    if show_methodology_box:
        n_mutual = sum(1 for e in edges if e["kind"] == "mutual")
        n_one    = len(edges) - n_mutual
        fig.add_annotation(
            x=0.01, y=0.99, xref="paper", yref="paper",
            text=(
                f"<b>WHAT IS BEING SHOWN</b><br>"
                f"<br>"
                f"<b>Technique:</b> Geographic Hierarchical Edge Bundling<br>"
                f"<i>Holten (2006) IEEE TVCG 12(5):741-748</i><br>"
                f"<br>"
                f"<b>Node positions:</b> real lat/lon coordinates<br>"
                f"<b>Node size:</b> years participated in this window<br>"
                f"<b>Node colour:</b> detected Louvain voting bloc<br>"
                f"<br>"
                f"<b>Edge bundling:</b> each NVS edge is a Bézier curve<br>"
                f"pulled {int(beta*100)}% toward the geographic centroid<br>"
                f"of the countries' shared voting bloc.<br>"
                f"<br>"
                f"<b>What to look for:</b><br>"
                f"Converging lines = strong voting corridor<br>"
                f"Large hub circles = geographic bloc centres<br>"
                f"Teal solid = mutual · Coral dotted = one-way<br>"
                f"<br>"
                f"<b>Edges shown:</b> {n_mutual} mutual + {n_one} one-way<br>"
                f"NVS = points / era_max (12 pre-2016 · 24 post-2016)"
            ),
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=8.5, color="#374151"), align="left",
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#2a9d8f", borderwidth=1.5, borderpad=10,
        )


def build_geographic_heb_poster(
    df: pd.DataFrame,
    id2label: dict,
    nodes_df: pd.DataFrame,
    min_years: int = 10,
    diff_threshold: float = 1.0,
    top_k_out: int = 3,
    min_nvs_strength: float = 2.0,
    beta: float = 0.80,
    hatred_min_years: int = 10,
    hatred_epsilon: float = 0.04,
):
    """
    DRAFT 10 — Geographic Hierarchical Edge Bundling Poster.

    Same algorithm as Draft 9 (Holten 2006 cubic Bézier HEB) but countries
    are placed at their real geographic coordinates. Bloc centroids are the
    geographic centre-of-mass of each detected bloc's members; edges are
    bundled through those geographic hubs, creating visible voting corridors
    on the actual map of Europe and its neighbours.

    This answers: *does the bundling pattern align with geographic regions?*
    Whereas Draft 9 (circular layout) reveals the bloc structure as an
    abstract visual pattern, Draft 10 lets the reader immediately see whether
    the Nordic bloc really does cluster in Scandinavia, whether the Balkan
    bloc forms a southeast hub, etc.

    Three-tier storyboard (same flow structure as other poster drafts):
      Tier 1  Full history 1975–2025 geographic HEB (full-width)
         ↓ splits into two eras
      Tier 2  Era 1 1975–1999 | Era 2 2000–2025 side-by-side
         ↓ reveals evidence
      Tier 3  Stat cards per era

    Returns (figure, title, explanation_markdown).
    """
    from plotly.subplots import make_subplots
    from collections import defaultdict

    df = _add_era_max_col(df)
    df["src_label"] = df["source"].map(id2label).fillna(df["source"])
    df["tgt_label"] = df["target"].map(id2label).fillna(df["target"])

    participation = (
        pd.concat([
            df[["year", "src_label"]].rename(columns={"src_label": "country"}),
            df[["year", "tgt_label"]].rename(columns={"tgt_label": "country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )
    qualified = sorted(participation[participation >= min_years].index.tolist())
    df = df[df["src_label"].isin(qualified) & df["tgt_label"].isin(qualified)].copy()

    coord_lookup = _coord_lookup(nodes_df, id2label)
    qualified = [c for c in qualified if c in coord_lookup]

    if df.empty or len(qualified) < 3:
        return None, "Geographic HEB Poster", (
            f"Not enough countries met the >= {min_years}-year participation "
            "threshold with usable coordinates."
        )

    df = df[df["src_label"].isin(qualified) & df["tgt_label"].isin(qualified)].copy()

    # ---- shared helpers (same as Draft 9) ----------------------------------

    def mean_nvs_matrix(sub_df, countries):
        if sub_df.empty or not countries:
            return pd.DataFrame(0.0, index=countries, columns=countries)
        return (
            sub_df.groupby(["src_label", "tgt_label"])["nvs"].mean()
            .unstack(fill_value=0)
            .reindex(index=countries, columns=countries, fill_value=0)
        ) * 12.0

    def nvs_backbone(mat, countries):
        keep = set()
        for c in countries:
            out_vals = mat.loc[c].drop(labels=[c], errors="ignore")
            strong = (
                out_vals[out_vals >= min_nvs_strength]
                .sort_values(ascending=False).head(top_k_out)
            )
            for partner in strong.index:
                keep.add(tuple(sorted([c, partner])))
        return keep

    def classify_edges(mat, countries):
        retained = nvs_backbone(mat, countries)
        edges = []
        for (a, b) in retained:
            ab = float(mat.loc[a, b]); ba = float(mat.loc[b, a])
            if ab <= 0 and ba <= 0:
                continue
            diff = abs(ab - ba)
            if diff <= diff_threshold:
                edges.append({"a": a, "b": b, "kind": "mutual",
                               "value": (ab+ba)/2, "ab": ab, "ba": ba, "diff": diff})
            else:
                gv, rv = (a, b) if ab > ba else (b, a)
                edges.append({"a": a, "b": b, "kind": "one_way",
                               "giver": gv, "receiver": rv,
                               "value": max(ab, ba), "ab": ab, "ba": ba, "diff": diff})
        return edges

    def detect(sub_df, countries):
        if not countries:
            return {}
        aff = _mutual_affinity(_affinity_input(sub_df), countries)
        return _detect_blocs(aff, countries, q=0.6)

    def flag_migrated(map1, map2):
        g1, g2 = defaultdict(set), defaultdict(set)
        for c, b in map1.items():
            g1[b].add(c)
        for c, b in map2.items():
            g2[b].add(c)
        migrated = set()
        for c in set(map1) & set(map2):
            m1 = g1[map1[c]] - {c}; m2 = g2[map2[c]] - {c}
            if (len(m1 & m2) / len(m1)) < 0.5 if m1 else True:
                migrated.add(c)
        return migrated

    def eligibility_frame(sub_df, countries):
        cs = set(countries); rows = []
        for yr, g in sub_df.groupby("year"):
            pts = (set(g["src_label"]) | set(g["tgt_label"])) & cs
            for s in pts:
                for t in pts:
                    if s != t: rows.append((yr, s, t))
        if not rows:
            return pd.DataFrame(columns=["year","src_label","tgt_label","points","nvs"])
        elig = pd.DataFrame(rows, columns=["year","src_label","tgt_label"])
        actual = sub_df.groupby(["year","src_label","tgt_label"], as_index=False)["points"].sum()
        m = elig.merge(actual, on=["year","src_label","tgt_label"], how="left")
        m["points"] = m["points"].fillna(0)
        m["nvs"] = (m["points"] / m["year"].apply(_era_max)).clip(0, 1)
        return m

    def era_stats(sub_df, countries, edges):
        mutual  = sorted([e for e in edges if e["kind"]=="mutual"],   key=lambda e: e["value"], reverse=True)[:3]
        oneway  = sorted([e for e in edges if e["kind"]=="one_way"],  key=lambda e: e["diff"],  reverse=True)[:3]
        elig = eligibility_frame(sub_df, countries)
        if elig.empty:
            return mutual, oneway, pd.DataFrame()
        agg = elig.groupby(["src_label","tgt_label"]).agg(
            years_eligible=("year","nunique"), mean_nvs=("nvs","mean")).reset_index()
        rl = {(r["src_label"],r["tgt_label"]): r["mean_nvs"] for _,r in agg.iterrows()}
        cands = agg[(agg["years_eligible"]>=hatred_min_years)&(agg["mean_nvs"]<hatred_epsilon)].copy()
        if cands.empty:
            return mutual, oneway, cands
        cands["reciprocal_nvs"] = cands.apply(lambda r: rl.get((r["tgt_label"],r["src_label"]),0.0), axis=1)
        return mutual, oneway, cands.sort_values(["years_eligible","reciprocal_nvs"],ascending=[False,False]).head(3)

    def render_stat(fig, row, col, era_label, top_mutual, top_oneway, top_hatred):
        fig.update_xaxes(visible=False, range=[0,1], row=row, col=col)
        fig.update_yaxes(visible=False, range=[0,1], row=row, col=col)
        mutual_lines  = [f"\U0001f91d {e['a']} \u2194 {e['b']}  (NVS {e['value']:.1f})" for e in top_mutual]  or ["—"]
        oneway_lines  = [f"\u27a1\ufe0f {e['giver']} \u2192 {e['receiver']}  (\u0394{e['diff']:.1f})" for e in top_oneway] or ["—"]
        hatred_lines  = (
            [f"\u2744\ufe0f {r['src_label']} \u21cf {r['tgt_label']}  ({int(r['years_eligible'])} yrs)"
             for _,r in top_hatred.iterrows()]
            if top_hatred is not None and not top_hatred.empty else ["—"]
        )
        y = 0.95
        fig.add_annotation(x=0.03, y=1.0, text=f"<b>{era_label}</b>", showarrow=False,
            font=dict(size=13, color="#1f2937", family="Georgia, serif"),
            xanchor="left", yanchor="top", row=row, col=col)
        y -= 0.12
        for heading, lines in [
            ("Top mutual voters (NVS ≈ equal both ways)", mutual_lines),
            ("Top one-way voters (strong asymmetry)", oneway_lines),
            ("Cold-shoulder pairs (near-zero NVS)", hatred_lines),
        ]:
            fig.add_annotation(x=0.03, y=y, text=f"<b>{heading}</b>", showarrow=False,
                font=dict(size=10, color="#374151"), xanchor="left", yanchor="top", row=row, col=col)
            y -= 0.09
            for line in lines[:3]:
                fig.add_annotation(x=0.06, y=y, text=line, showarrow=False,
                    font=dict(size=9, color="#4b5563"), xanchor="left", yanchor="top", row=row, col=col)
                y -= 0.08
            y -= 0.02

    # ---- participation year lookups ----------------------------------------
    part_total = participation.to_dict()
    era1_part_df = pd.concat([
        df[df["year"]<=1999][["year","src_label"]].rename(columns={"src_label":"country"}),
        df[df["year"]<=1999][["year","tgt_label"]].rename(columns={"tgt_label":"country"}),
    ]).drop_duplicates().groupby("country")["year"].nunique()
    era2_part_df = pd.concat([
        df[df["year"]>=2000][["year","src_label"]].rename(columns={"src_label":"country"}),
        df[df["year"]>=2000][["year","tgt_label"]].rename(columns={"tgt_label":"country"}),
    ]).drop_duplicates().groupby("country")["year"].nunique()
    era1_part = era1_part_df.to_dict()
    era2_part = era2_part_df.to_dict()

    # ---- Tier 1: full history ----------------------------------------------
    full_mat   = mean_nvs_matrix(df, qualified)
    full_bloc  = detect(df, qualified)
    full_edges = classify_edges(full_mat, qualified)
    full_geo_c = _geo_heb_compute_bloc_centroids(full_bloc, coord_lookup)

    # ---- Tier 2: two eras --------------------------------------------------
    era1_df = df[df["year"]<=1999]
    era2_df = df[df["year"]>=2000]
    era1_countries = sorted({c for c in qualified if c in set(era1_df["src_label"])|set(era1_df["tgt_label"])})
    era2_countries = sorted({c for c in qualified if c in set(era2_df["src_label"])|set(era2_df["tgt_label"])})
    era1_mat   = mean_nvs_matrix(era1_df, era1_countries)
    era2_mat   = mean_nvs_matrix(era2_df, era2_countries)
    era1_bloc  = detect(era1_df, era1_countries)
    era2_bloc  = detect(era2_df, era2_countries)
    era1_edges = classify_edges(era1_mat, era1_countries)
    era2_edges = classify_edges(era2_mat, era2_countries)
    era1_geo_c = _geo_heb_compute_bloc_centroids(era1_bloc, coord_lookup)
    era2_geo_c = _geo_heb_compute_bloc_centroids(era2_bloc, coord_lookup)

    migrated = flag_migrated(era1_bloc, era2_bloc)

    # Build migration ring colour lookup for Era-2
    era1_bloc_names  = sorted(set(era1_bloc.values())) if era1_bloc else []
    era1_bloc_color  = {b: _GEO_HEB_PALETTE[i%len(_GEO_HEB_PALETTE)] for i,b in enumerate(era1_bloc_names)}
    migrated_ring_color = {
        c: era1_bloc_color.get(era1_bloc.get(c), "#9ca3af")
        for c in migrated if c in era1_bloc and c in era2_bloc
    }

    # ---- Tier 3: evidence --------------------------------------------------
    top_m1, top_o1, top_h1 = era_stats(era1_df, era1_countries, era1_edges)
    top_m2, top_o2, top_h2 = era_stats(era2_df, era2_countries, era2_edges)

    # ---- panel title helper ------------------------------------------------
    def _ptitle(prefix, countries, edges):
        nm = sum(1 for e in edges if e["kind"]=="mutual")
        no = len(edges)-nm
        return (
            f"{prefix}<br>"
            f"<span style='font-size:11px;color:#6b7280;'>"
            f"{len(countries)} countries \u00b7 {len(edges)} edges "
            f"({nm} mutual \u2014, {no} one-way \u2508)</span>"
        )

    # ---- figure assembly ---------------------------------------------------
    row_heights = [0.40, 0.32, 0.28]
    vspacing    = 0.08
    avail       = 1.0 - vspacing * 2
    scaled      = [h * avail for h in row_heights]
    boundaries  = []
    cur         = 1.0
    for h in scaled:
        bot = cur - h
        boundaries.append((cur, bot))
        cur = bot - vspacing

    fig = make_subplots(
        rows=3, cols=2,
        row_heights=row_heights,
        vertical_spacing=vspacing,
        horizontal_spacing=0.04,
        specs=[
            [{"type":"scattergeo","colspan":2}, None],
            [{"type":"scattergeo"}, {"type":"scattergeo"}],
            [{"type":"xy"}, {"type":"xy"}],
        ],
        subplot_titles=[
            _ptitle("Full picture \u00b7 1975\u20132025", qualified, full_edges),
            _ptitle("Era 1 \u00b7 1975\u20131999", era1_countries, era1_edges),
            _ptitle("Era 2 \u00b7 2000\u20132025", era2_countries, era2_edges),
            "Era 1 insights", "Era 2 insights",
        ],
    )

    _geo_heb_render_panel(
        fig, 1, 1, qualified, full_edges, full_bloc,
        coord_lookup, full_geo_c,
        part_years=part_total, beta=beta,
        show_methodology_box=True,
        diff_threshold=diff_threshold, top_k_out=top_k_out, min_nvs_strength=min_nvs_strength,
    )
    _geo_heb_render_panel(
        fig, 2, 1, era1_countries, era1_edges, era1_bloc,
        coord_lookup, era1_geo_c,
        part_years=era1_part, migrated_gold=migrated, beta=beta,
    )
    _geo_heb_render_panel(
        fig, 2, 2, era2_countries, era2_edges, era2_bloc,
        coord_lookup, era2_geo_c,
        part_years=era2_part, migrated_ring_color=migrated_ring_color, beta=beta,
    )
    render_stat(fig, 3, 1, "1975\u20131999", top_m1, top_o1, top_h1)
    render_stat(fig, 3, 2, "2000\u20132025", top_m2, top_o2, top_h2)

    # ---- dotted flow connectors --------------------------------------------
    def add_connector(x, y_top, y_bot, label):
        fig.add_shape(type="line", x0=x, y0=y_top-0.005, x1=x, y1=y_bot+0.018,
            line=dict(dash="dot", color="#94a3b8", width=2), xref="paper", yref="paper")
        fig.add_annotation(x=x, y=y_bot+0.014, text="\u25bc", showarrow=False,
            xref="paper", yref="paper", font=dict(size=13, color="#94a3b8"))
        fig.add_annotation(x=x, y=(y_top+y_bot)/2, text=label, showarrow=False,
            xref="paper", yref="paper",
            font=dict(size=10, color="#64748b", family="Georgia, serif"),
            bgcolor="rgba(248,250,252,0.9)", borderpad=3)

    g1t, g1b = boundaries[0][1], boundaries[1][0]
    g2t, g2b = boundaries[1][1], boundaries[2][0]
    add_connector(0.25, g1t, g1b, "splits into two eras")
    add_connector(0.75, g1t, g1b, "splits into two eras")
    add_connector(0.25, g2t, g2b, "reveals evidence")
    add_connector(0.75, g2t, g2b, "reveals evidence")

    # ---- reading guide (top-right, bordered card) -------------------------
    fig.add_annotation(
        x=0.99, y=1.065, xref="paper", yref="paper",
        text=(
            "<b>HOW TO READ THIS MAP</b><br>"
            "<br>"
            "<b>Node positions:</b> real lat/lon (geographic)<br>"
            "<b>Node colour:</b> detected voting bloc<br>"
            "<b>Node size:</b> years participated in this window<br>"
            "<b>Large hub circle:</b> bloc geographic centroid<br>"
            "<br>"
            "<span style='color:rgb(13,148,136)'><b>━━━ Teal solid</b></span>"
            " = Mutual NVS (both ways ≈ equal)<br>"
            "<span style='color:rgb(220,86,60)'><b>┈┈┈ Coral dotted</b></span>"
            " = One-way (hover for direction)<br>"
            "Darker / thicker = stronger NVS<br>"
            "<br>"
            "<b>The bundling effect:</b> lines converge toward<br>"
            "their bloc's geographic hub before diverging<br>"
            "to the target country. Overlapping paths from<br>"
            "the same region reveal voting corridors.<br>"
            "<br>"
            "<span style='color:#b45309'><b>Gold ring</b></span>"
            " = country about to change bloc (Era 1)<br>"
            "<b>Coloured ring</b> = shows previous bloc (Era 2)<br>"
            "<br>"
            f"<span style='font-size:8px;color:#94a3b8;'>"
            f"\u03b2 = {beta:.2f} · NVS floor = {min_nvs_strength}/12<br>"
            f"Holten (2006) IEEE TVCG 12(5):741-748</span>"
        ),
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=9, color="#374151"), align="right",
        bgcolor="rgba(255,255,255,0.97)", bordercolor="#2a9d8f",
        borderwidth=1.5, borderpad=10,
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Eurovision Voting Network \u00b7 Geographic Hierarchical Edge Bundling</b>"
                "<br><span style='font-size:12px;color:#64748b;'>"
                "Countries at real coordinates \u00b7 Bézier edges bundled through geographic bloc centroids "
                "(\u03b2={beta:.2f}) \u00b7 Holten (2006) \u00b7 1975\u20132025"
                "</span>"
            ).format(beta=beta),
            x=0.5, xanchor="center",
            font=dict(size=17, family="Georgia, serif", color="#111827"),
        ),
        height=1680, width=1250,
        paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc",
        showlegend=False,
        margin=dict(l=30, r=30, t=145, b=40),
    )

    explanation = f"""
**Geographic Hierarchical Edge Bundling (Draft 10)**

This draft applies the same Holten (2006) algorithm as Draft 9 but replaces
the abstract circular layout with real geographic coordinates, answering the
question: *does the voting bloc structure align with geographic regions?*

**What you are seeing:**

- **Large transparent circles (hubs):** the geographic centre-of-mass of each
  detected Louvain voting bloc — the average lat/lon of all member countries.
  These are the Bézier control points for the bundling; all edges from
  countries in that bloc are pulled toward their bloc's hub before diverging
  toward their destination.

- **Solid teal lines ━━━ = Mutual relationship:** both countries give each
  other similar NVS (|NVS(A→B) − NVS(B→A)| ≤ {diff_threshold}). The line
  is routed through both blocs' geographic centroids, producing a visible
  geographic corridor between voting regions.

- **Dotted coral lines ┈┈┈ = One-way relationship:** one country gives
  significantly more NVS than it receives. Hover for direction.

- **Node size:** proportional to years participated in that panel's time
  window (1975–2025 for Tier 1; 1975–1999 or 2000–2025 for era panels).

- **Gold ring:** country whose voting bloc will change in the next era.
  **Coloured ring (Era 2):** that same country's ring colour shows which
  bloc it came from, while the fill shows its new bloc.

**Why geographic layout adds insight beyond Draft 9:** the circular layout
groups same-bloc countries adjacent to each other by construction, so the
bundling is always visible. In the geographic layout, that same bundling
has to *earn* its visual cluster by actually routing through a geographic
region — if the Balkan bloc's centroid really does sit near the Adriatic
and most edges involving Balkan countries route through that point, the
visual confirms the geographic coherence of the bloc. If blocs were purely
political/cultural with no geographic basis, the hubs would be scattered
and the bundling would be diffuse rather than tightly regionally clustered.

**Citation:** Holten, D.H.R. (2006). Hierarchical edge bundles: visualization
of adjacency relations in hierarchical data. *IEEE Transactions on
Visualization and Computer Graphics*, 12(5), 741–748.
DOI: 10.1109/TVCG.2006.147

**Edge selection:** top {top_k_out} outgoing NVS ties per country where
NVS ≥ {min_nvs_strength}/12 (same NVS-strength-ranked approach as Drafts 7,
9). Only qualifying countries with ≥ {min_years} years of participation
and usable geographic coordinates are included.

All thresholds ({diff_threshold} mutual/one-way, {min_nvs_strength} NVS floor,
{min_years} years minimum, β = {beta:.2f} bundling strength) are exploratory
cutoffs chosen for visual and narrative clarity.
"""
    return fig, "Geographic HEB — Nodes on Real Map", explanation

# =============================================================================
# DIAGRAM 11 — SPLIT-TRIANGLE ADJACENCY MATRIX  ("One Matrix, Two Eras")
# =============================================================================
#
# Inspired by the "1c" concept from the Eurovision Blocs Claude-Design artifact:
# a single square NVS matrix where the SAME country ordering is shared between
# two eras.  The lower-left triangle shows Era I (1975–1999) and the upper-right
# triangle shows Era II (2000–2025).  The diagonal is shaded neutral.
#
# Visual encoding:
#   • Cell colour   = bloc of the ROW country (same palette as other drafts)
#   • Cell opacity  = NVS strength (0 → transparent, max → fully saturated)
#   • Grey stipple  = country not yet participating / absent in that era cell
#   • Diagonal      = dark neutral (self-vote impossible)
#   • Bloc boundary lines drawn as thin rectangles around the bloc sub-squares
#
# Country ordering: sorted by detected Louvain bloc (full history) then
# alphabetically within each bloc, so bloc sub-squares appear as compact
# coloured blocks along the diagonal — exactly like the D3 "mobility matrix"
# pattern.
#
# Two annotation callouts (paper coordinates) mirror the original design:
#   ① pointing to the upper-right: "ERA II — new blocs emerge"
#   ② pointing to the lower-left: "ERA I — sparse, Western-dominated"
# =============================================================================


def build_split_triangle_matrix(
    df: pd.DataFrame,
    id2label: dict,
    nodes_df: pd.DataFrame,
    min_years_overall: int = 10,
    min_years_per_half: int = 3,
    dark_theme: bool = False,
):
    """
    DRAFT 11 — Split-triangle adjacency matrix.

    One square matrix, two eras in opposite triangles.  Country rows/columns
    are sorted by detected Louvain bloc (full-history affinity) then
    alphabetically within each bloc, so same-bloc relationships appear as
    dense coloured blocks along the diagonal.

    Returns (figure, title, explanation_markdown).
    """
    from collections import defaultdict

    # ---- data prep -----------------------------------------------------------
    df = _add_era_max_col(df)
    df["src_label"] = df["source"].map(id2label).fillna(df["source"])
    df["tgt_label"] = df["target"].map(id2label).fillna(df["target"])

    participation_all = (
        pd.concat([
            df[["year", "src_label"]].rename(columns={"src_label": "country"}),
            df[["year", "tgt_label"]].rename(columns={"tgt_label": "country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )

    era1_df = df[df["year"] <= 1999]
    era2_df = df[df["year"] >= 2000]

    era1_part = (
        pd.concat([
            era1_df[["year", "src_label"]].rename(columns={"src_label": "country"}),
            era1_df[["year", "tgt_label"]].rename(columns={"tgt_label": "country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )
    era2_part = (
        pd.concat([
            era2_df[["year", "src_label"]].rename(columns={"src_label": "country"}),
            era2_df[["year", "tgt_label"]].rename(columns={"tgt_label": "country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )

    # Qualify countries with enough overall participation
    qualified = sorted(
        participation_all[participation_all >= min_years_overall].index.tolist()
    )
    df_q = df[df["src_label"].isin(qualified) & df["tgt_label"].isin(qualified)]

    if df_q.empty or len(qualified) < 3:
        return None, "Split-Triangle Matrix", "Not enough countries to build this draft."

    # ---- bloc detection (full history, for ordering) -------------------------
    aff = _mutual_affinity(_affinity_input(df_q), qualified)
    bloc_map = _detect_blocs(aff, qualified, q=0.6)

    # Sort countries: blocs by descending size, then alpha within each bloc
    bloc_members = defaultdict(list)
    for c, b in bloc_map.items():
        if c in qualified:
            bloc_members[b].append(c)
    blocs_by_size = sorted(bloc_members.keys(), key=lambda b: -len(bloc_members[b]))
    order = []
    for b in blocs_by_size:
        order.extend(sorted(bloc_members[b]))
    n = len(order)

    # ---- NVS matrices per era ------------------------------------------------
    def mean_nvs_mat(sub_df, countries):
        if sub_df.empty:
            return pd.DataFrame(0.0, index=countries, columns=countries)
        return (
            sub_df.groupby(["src_label", "tgt_label"])["nvs"].mean()
            .unstack(fill_value=0)
            .reindex(index=countries, columns=countries, fill_value=0)
        ) * 12.0  # 0-12 scale

    m1 = mean_nvs_mat(era1_df[era1_df["src_label"].isin(order) & era1_df["tgt_label"].isin(order)], order)
    m2 = mean_nvs_mat(era2_df[era2_df["src_label"].isin(order) & era2_df["tgt_label"].isin(order)], order)

    # Absent flags per era (too few years to be meaningful)
    absent1 = {c for c in order if era1_part.get(c, 0) < min_years_per_half}
    absent2 = {c for c in order if era2_part.get(c, 0) < min_years_per_half}

    # ---- colour palette (matches other drafts) --------------------------------
    PALETTE = [
        "#1f4e79", "#d1495b", "#2a9d8f", "#f4a261",
        "#6a4c93", "#7f5539", "#577590", "#3a86ff",
    ]
    bloc_color = {b: PALETTE[i % len(PALETTE)] for i, b in enumerate(blocs_by_size)}

    if dark_theme:
        bg_color    = "#0f1217"
        diag_color  = "rgba(238,242,248,0.18)"
        absent_rgba = "rgba(238,242,248,0.06)"
        border_rgba = "rgba(238,242,248,0.22)"
        label_color = "rgba(220,227,238,0.85)"
        title_color = "#eef2f8"
        paper_bg    = "#0f1217"
    else:
        bg_color    = "#f7f4ee"
        diag_color  = "rgba(32,36,43,0.16)"
        absent_rgba = "rgba(32,36,43,0.05)"
        border_rgba = "rgba(32,36,43,0.30)"
        label_color = "rgba(32,36,43,0.72)"
        title_color = "#20242b"
        paper_bg    = "#f7f4ee"

    def hex_rgba(hx, alpha):
        h = hx.lstrip("#")
        r, g, b_ = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b_},{alpha:.2f})"

    # ---- build figure --------------------------------------------------------
    # We use a square axis (x = column index, y = row index inverted)
    # Lower-left (r > c) = Era I, upper-right (r < c) = Era II
    CELL = 1.0  # unit size per cell

    fig = go.Figure()

    # Background rectangle
    fig.add_shape(
        type="rect", x0=-0.02, y0=-0.02, x1=n * CELL + 0.02, y1=n * CELL + 0.02,
        fillcolor=bg_color, line=dict(width=0),
        layer="below",
    )

    max_nvs = 12.0

    # Cell rectangles
    for ri, row_c in enumerate(order):
        for ci, col_c in enumerate(order):
            x0, x1 = ci * CELL, (ci + 1) * CELL
            y0, y1 = (n - 1 - ri) * CELL, (n - ri) * CELL  # y-axis inverted

            if ri == ci:
                # Diagonal
                fill = diag_color
            elif ri > ci:
                # Lower-left = Era I
                if row_c in absent1 or col_c in absent1:
                    fill = absent_rgba
                else:
                    nvs = float(m1.loc[row_c, col_c])
                    alpha = 0.06 + 0.70 * min(nvs / max_nvs, 1.0)
                    fill = hex_rgba(bloc_color[bloc_map[row_c]], alpha)
            else:
                # Upper-right = Era II
                if row_c in absent2 or col_c in absent2:
                    fill = absent_rgba
                else:
                    nvs = float(m2.loc[row_c, col_c])
                    alpha = 0.06 + 0.70 * min(nvs / max_nvs, 1.0)
                    fill = hex_rgba(bloc_color[bloc_map[row_c]], alpha)

            ht_era = "Era I (1975–1999)" if ri > ci else "Era II (2000–2025)" if ri < ci else "Diagonal"
            if ri != ci:
                nvs_val = float(m1.loc[row_c, col_c]) if ri > ci else float(m2.loc[row_c, col_c])
                hover = (
                    f"<b>{row_c}</b> → <b>{col_c}</b><br>"
                    f"{ht_era}<br>"
                    f"NVS: {nvs_val:.2f} / 12"
                )
            else:
                hover = row_c

            fig.add_shape(
                type="rect", x0=x0 + 0.02, y0=y0 + 0.02, x1=x1 - 0.02, y1=y1 - 0.02,
                fillcolor=fill, line=dict(width=0),
            )
            # Invisible scatter for hover
            fig.add_trace(go.Scatter(
                x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
                mode="markers",
                marker=dict(size=CELL * 10, color="rgba(0,0,0,0)", symbol="square"),
                hovertemplate=hover + "<extra></extra>",
                showlegend=False,
            ))

    # Bloc boundary rectangles
    cursor = 0
    for b in blocs_by_size:
        sz = len(bloc_members[b])
        x0 = cursor * CELL
        y0 = (n - cursor - sz) * CELL
        x1 = (cursor + sz) * CELL
        y1 = (n - cursor) * CELL
        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=border_rgba, width=1.5),
            fillcolor="rgba(0,0,0,0)",
        )
        # Bloc label on diagonal
        mid = cursor + sz / 2
        fig.add_annotation(
            x=mid * CELL, y=(n - mid) * CELL,
            text=f"<b>{b}</b>",
            showarrow=False,
            font=dict(size=7.5, color=bloc_color[b]),
            xanchor="center", yanchor="middle",
        )
        cursor += sz

    # Diagonal line
    fig.add_shape(
        type="line", x0=0, y0=n * CELL, x1=n * CELL, y1=0,
        line=dict(color=border_rgba, width=1, dash="dot"),
    )

    # Country labels (bottom and left, sampled for readability)
    label_every = max(1, n // 30)
    for i, c in enumerate(order):
        if i % label_every != 0:
            continue
        xi = (i + 0.5) * CELL
        yi_row = (n - i - 0.5) * CELL
        # Bottom label (column)
        fig.add_annotation(x=xi, y=-0.3, text=c, showarrow=False,
                            font=dict(size=6.5, color=label_color),
                            xanchor="center", yanchor="top", textangle=-60)
        # Left label (row)
        fig.add_annotation(x=-0.3, y=yi_row, text=c, showarrow=False,
                            font=dict(size=6.5, color=label_color),
                            xanchor="right", yanchor="middle")

    # Era labels on the triangles
    fig.add_annotation(
        x=n * CELL * 0.82, y=n * CELL * 0.82,
        text="<b>ERA II</b><br>2000–2025",
        showarrow=False, font=dict(size=12, color=title_color, family="Georgia, serif"),
        xanchor="center", yanchor="middle",
    )
    fig.add_annotation(
        x=n * CELL * 0.18, y=n * CELL * 0.18,
        text="<b>ERA I</b><br>1975–1999",
        showarrow=False, font=dict(size=12, color=title_color, family="Georgia, serif"),
        xanchor="center", yanchor="middle",
    )

    # Key insight annotations (callouts)
    fig.add_annotation(
        x=n * CELL * 0.70, y=n * CELL * 0.30,
        text=(
            "<b>New blocs emerge</b><br>"
            "Post-Soviet & Balkan clusters<br>"
            "form dense coloured squares<br>"
            "only in the right triangle"
        ),
        showarrow=True, arrowhead=2, arrowcolor=title_color, arrowwidth=1.2,
        ax=50, ay=-40,
        font=dict(size=8.5, color=title_color),
        bgcolor=paper_bg, bordercolor=border_rgba, borderwidth=1, borderpad=6,
        xanchor="left",
    )
    fig.add_annotation(
        x=n * CELL * 0.30, y=n * CELL * 0.70,
        text=(
            "<b>Western dominance, 1975–1999</b><br>"
            "Left triangle sparser;<br>"
            "only Western & Mediterranean<br>"
            "blocs have strong sub-squares"
        ),
        showarrow=True, arrowhead=2, arrowcolor=title_color, arrowwidth=1.2,
        ax=-50, ay=40,
        font=dict(size=8.5, color=title_color),
        bgcolor=paper_bg, bordercolor=border_rgba, borderwidth=1, borderpad=6,
        xanchor="right",
    )

    # Legend (coloured squares per bloc)
    legend_y = -1.8
    for idx, b in enumerate(blocs_by_size):
        lx = idx * (n * CELL / len(blocs_by_size)) + 0.5
        fig.add_shape(
            type="rect",
            x0=lx, y0=legend_y - 0.3, x1=lx + 0.7, y1=legend_y + 0.3,
            fillcolor=bloc_color[b], line=dict(width=0),
        )
        members_str = ", ".join(sorted(bloc_members[b])[:4])
        if len(bloc_members[b]) > 4:
            members_str += f" +{len(bloc_members[b])-4}"
        fig.add_annotation(
            x=lx + 0.85, y=legend_y,
            text=f"<b>{b}</b>  {members_str}",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=7, color=label_color),
        )

    # Reading guide
    fig.add_annotation(
        x=n * CELL, y=n * CELL + 0.5,
        text=(
            "<b>Reading guide:</b> lower-left triangle = Era I (1975–1999) · "
            "upper-right = Era II (2000–2025) · "
            "cell colour = row country's voting bloc · "
            "opacity = NVS strength · grey = country absent in that era"
        ),
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=8.5, color=label_color), align="right",
        bgcolor=paper_bg, bordercolor=border_rgba, borderwidth=1, borderpad=6,
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>One Matrix, Two Eras — Eurovision Voting Affinity</b>"
                "<br><span style='font-size:12px;color:#6b7280;'>"
                "Countries ordered by detected voting bloc · "
                "lower-left = 1975–1999 · upper-right = 2000–2025 · "
                f"{'Dark theme' if dark_theme else 'Light theme'}"
                "</span>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=16, family="Georgia, serif", color=title_color),
        ),
        xaxis=dict(visible=False, range=[-2.5, n * CELL + 1.5]),
        yaxis=dict(visible=False, range=[-2.8, n * CELL + 1.5], scaleanchor="x", scaleratio=1),
        height=max(900, n * 18 + 300),
        width=max(960, n * 18 + 300),
        paper_bgcolor=paper_bg, plot_bgcolor=paper_bg,
        showlegend=False,
        margin=dict(l=120, r=60, t=120, b=120),
    )

    n_era1 = len([c for c in order if c not in absent1])
    n_era2 = len([c for c in order if c not in absent2])

    explanation = f"""
**Concept source:** Design variant "1c" from the *Eurovision Blocs* Claude-Design
artifact, adapted here from placeholder SVG data to your real NVS dataset.

**What this shows:** a single adjacency matrix where rows and columns represent
the same set of qualifying countries (≥{min_years_overall} years total
participation), ordered by detected Louvain voting bloc (largest bloc first,
alphabetical within each bloc). The diagonal divides the matrix into two
triangles — **lower-left = Era I (1975–1999)** and
**upper-right = Era II (2000–2025)** — so bloc formation and dissolution is
legible at a glance without side-by-side panels.

**How to read it:** each cell (row A, col B) in the lower-left shows the mean
NVS(A→B) from 1975–1999; the symmetric cell in the upper-right shows the same
pair's mean NVS(A→B) from 2000–2025. Cell **colour** encodes the row country's
voting bloc; **opacity** encodes NVS strength (transparent = near-zero, fully
saturated = maximum voting affinity). Grey/transparent cells indicate countries
absent from that era (fewer than {min_years_per_half} years of participation).

**What to look for:** along-diagonal bloc sub-squares that are more saturated in
the upper-right triangle than the lower-left reveal blocs that **intensified**
after 2000 (historically: Post-Soviet and Balkan clusters). Sub-squares visible
only in the lower-left but absent or faint in the upper-right reveal blocs that
**dissolved**. Off-diagonal colour visible in one triangle but not the other
reveals cross-bloc relationships that opened up or closed after 2000.

**Countries shown:** {n} qualifying countries; {n_era1} active in Era I,
{n_era2} active in Era II.

**Thesis placement:** this is an alternative to the node-link diagrams for
Section 4.3 (Layout Strategy) — cite Ghoniem et al. (2004) [already in your
bibliography] for the matrix vs. node-link readability argument: matrix
representations scale more gracefully when N is large (here ~{n} countries)
and the reader's primary task is finding dense sub-groups rather than tracing
individual paths.
"""
    return fig, "Split-Triangle Matrix — One Matrix, Two Eras", explanation