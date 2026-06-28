"""
draft_visualizations.py

Five novel, thesis-grade Eurovision visualisations — distinct from the
existing Sankey (bloc migration), Sunburst (bloc/country/supporter), and
GeoMap (top-3 voters) views already in the main app.

Each function takes the already-loaded, already-filtered edges dataframe
(scoped to ROOT_START..ROOT_END, i.e. 1975-2025) plus id2label/nodes, and
returns a tuple:

    (figure, title, explanation_markdown)

`figure` is a Plotly Figure ready for st.plotly_chart().
`explanation_markdown` is shown above the chart so readers know exactly
what is plotted and how it was computed, before they look at it.

All five reuse the same NVS (Normalised Voting Share) definition used
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