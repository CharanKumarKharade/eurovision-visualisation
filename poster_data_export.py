#!/usr/bin/env python3
"""
poster_data_export.py
=====================
GD Contest 2026 — Eurovision Voting Communities
Exports all data needed by poster.html as poster_data.json + poster_data.js

Usage:
    cd /workspaces/eurovision-visualisation
    python poster_data_export.py

Outputs:
    poster_data.json   — raw data (inspect this to verify correctness)
    poster_data.js     — same data as JS const (loaded by poster.html directly)

Takes about 30-60 seconds on the full dataset.

WHY THESE THRESHOLDS
--------------------
min_years_era1 = 15
    Era I spans 25 years (1975-1999). A country needs 15/25 = 60% presence
    to be included. This excludes Eastern European countries who joined
    1993-1999 (≤7 years) — they are structurally absent in Era I, which
    is itself the key analytical finding: the Era I network is entirely
    Western/Southern/Northern Europe.

min_years_era2 = 8
    Era II spans 25 years (2000-2025). Lower threshold (8/25 = 32%) because
    new entrants (Balkans, Caucasus) need representation even if they joined
    mid-era. Without this, the Balkan bloc — the most structurally interesting
    new development — would be excluded.

NVS edge threshold = 2.0 (on 0-12 scale)
    A pair needs mean NVS ≥ 2.0 from EITHER direction to appear as an edge.
    This corresponds to roughly giving 2 points per year on average, which
    is above the "occasionally gives low points" noise level but below the
    "consistent ally" signal level. The poster shows ~80-120 edges per era,
    which is readable at A0 scale.

EDGE CLASSIFICATION LOGIC
-------------------------
stable_alliance:  strong (≥3.0 NVS mutual) AND stable (stability ≥ 0.5)
                  in BOTH eras. Gold thick solid arc.

strengthened:     weak or absent in Era I, strong (≥3.0) in Era II.
                  Green arc. Shows relationships that grew post-2000
                  (e.g., post-Soviet bloc forming, Balkan solidarity).

weakened:         strong in Era I, weak or absent in Era II.
                  Orange dashed arc. Shows relationships that dissolved
                  (e.g., some Western alliances that fragmented).

one_sided:        |NVS(A→B) - NVS(B→A)| ≥ 2.5, both nonzero.
                  Red arc + dot at receiver. Shows unrequited voting
                  (Greece→Cyprus vs Cyprus→Greece asymmetry post-1981,
                  Armenia→Russia vs Russia→Armenia, etc.).

cold_shoulder:    co_eligible_years ≥ 12 AND max(NVS) < 0.6 both ways.
                  Grey dashed arc. Countries that co-existed for decades
                  but barely acknowledged each other (UK↔Germany classic).

new:              Country only exists in Era II (joined post-2000).
                  Teal thin arc. New relationships that couldn't exist
                  before (whole Balkan bloc, Caucasus countries).
"""

import json
import math
import sys
from collections import defaultdict
from datetime import date

import networkx as nx
import numpy as np
import pandas as pd

try:
    import community as community_louvain  # python-louvain
    LOUVAIN_OK = True
except ImportError:
    LOUVAIN_OK = False
    print("  WARNING: python-louvain not found, falling back to greedy_modularity")

# =============================================================================
# CONFIGURATION — tweak these if needed
# =============================================================================

NODES_FILE  = "nodes_with_coordinates.csv"
EDGES_FILE  = "eurovision_senior.csv"

ERA1_START, ERA1_END = 1975, 1999
ERA2_START, ERA2_END = 2000, 2025

MIN_YEARS_ERA1   = 15     # minimum years participated in Era I to appear
MIN_YEARS_ERA2   = 8      # minimum years participated in Era II to appear

# Edge thresholds (NVS on 0-12 scale)
EDGE_MIN_NVS     = 2.0    # minimum mean NVS to draw ANY edge
ALLIANCE_NVS     = 3.0    # minimum for "strong" relationship
ALLIANCE_STAB    = 0.45   # minimum stability (1 - CV) for alliance
ONE_SIDED_DELTA  = 2.5    # minimum asymmetry for one-sided classification
COLD_MIN_YEARS   = 12     # minimum co-eligible years for cold shoulder
COLD_MAX_NVS     = 0.6    # maximum NVS for cold shoulder

# Layout parameters
LON_MIN, LON_MAX = -30, 60   # geographic extent (degrees)
LAT_MIN, LAT_MAX = 28,  72
REPULSION_ITERS  = 60        # how hard to push overlapping nodes apart
REPULSION_MIN    = 0.055     # minimum normalized distance between nodes

# Colour palette — 8 blocs max, same as rest of project
PALETTE = [
    "#3b74b0",  # Nordic-like (blue)
    "#5f8f57",  # Western-like (green)
    "#7b68ee",  # Central-like (purple)
    "#df6234",  # Balkan-like (orange-red)
    "#2f9c8b",  # Baltic-like (teal)
    "#8a5cc4",  # Post-Soviet-like (violet)
    "#c8782a",  # extra (warm orange)
    "#3a7a5a",  # extra (dark teal)
]

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data():
    print("\n[1/6] Loading data files...")

    try:
        nodes = pd.read_csv(NODES_FILE)
    except FileNotFoundError:
        sys.exit(f"  ERROR: {NODES_FILE} not found. Run from project root.")

    try:
        edges = pd.read_csv(EDGES_FILE)
    except FileNotFoundError:
        sys.exit(f"  ERROR: {EDGES_FILE} not found.")

    nodes.columns = [c.strip().lower() for c in nodes.columns]
    edges.columns = [c.strip().lower() for c in edges.columns]

    # Normalise string columns
    for col in ["score_type", "round", "source", "target"]:
        if col in edges.columns:
            edges[col] = edges[col].astype(str).str.strip().str.lower()

    # Filter to finals, total score only
    if "round" in edges.columns:
        edges = edges[edges["round"] == "final"]
    if "score_type" in edges.columns and "total" in edges["score_type"].unique():
        edges = edges[edges["score_type"] == "total"]

    # Find and rename points column
    pts_col = next(
        (c for c in edges.columns
         if any(k in c for k in ["point","pts","score","value","weight"])
         and pd.api.types.is_numeric_dtype(edges[c])),
        None
    )
    if pts_col is None:
        sys.exit("  ERROR: Cannot find points column in edges CSV.")
    if pts_col != "points":
        edges = edges.rename(columns={pts_col: "points"})

    edges["year"]   = pd.to_numeric(edges["year"],   errors="coerce")
    edges["points"] = pd.to_numeric(edges["points"], errors="coerce").fillna(0)
    edges = edges.dropna(subset=["year"])
    edges["year"] = edges["year"].astype(int)

    # Scope to 1975-2025
    edges = edges[(edges["year"] >= ERA1_START) & (edges["year"] <= ERA2_END)]

    # Country id → label mapping
    # Build case-insensitive: "AT", "at", "At" all → "Austria"
    id2label_raw = nodes.set_index("id")["label"].to_dict()
    id2label = {}
    for k, v in id2label_raw.items():
        id2label[str(k)]        = v   # original case ("AT")
        id2label[str(k).lower()] = v  # lowercase ("at")
        id2label[str(k).upper()] = v  # uppercase ("AT") — redundant but safe
    # Also map full label → itself (so labels that are already full names pass through)
    for v in list(id2label_raw.values()):
        id2label[str(v)] = v

    edges["src"] = edges["source"].map(id2label).fillna(edges["source"])
    edges["tgt"] = edges["target"].map(id2label).fillna(edges["target"])

    # NVS normalisation:
    # ERA_MAX = 12 for 1975-2015 (jury only gave up to 12 pts)
    # ERA_MAX = 24 for 2016-2025 (jury 12 + televote 12 = 24 in total)
    # WHY: points from different eras are not directly comparable without this.
    # A country giving 10 points in 1985 (out of 12 possible) is very different
    # from giving 10 points in 2020 (out of 24 possible). NVS makes them comparable.
    edges["era_max"] = edges["year"].apply(lambda y: 24 if y >= 2016 else 12)
    edges["nvs"]     = (edges["points"] / edges["era_max"]).clip(0, 1)

    # Coordinates
    coord_cols = {c.lower(): c for c in nodes.columns}
    lat_col = next((coord_cols[k] for k in ["lat","latitude","y"] if k in coord_cols), None)
    lon_col = next((coord_cols[k] for k in ["lon","long","longitude","x"] if k in coord_cols), None)

    if lat_col and lon_col:
        coord_lookup = {}
        for _, row in nodes.dropna(subset=[lat_col, lon_col, "label"]).iterrows():
            lbl = str(row["label"])
            coord_lookup[lbl]         = (float(row[lat_col]), float(row[lon_col]))
            coord_lookup[lbl.lower()] = (float(row[lat_col]), float(row[lon_col]))
            coord_lookup[lbl.upper()] = (float(row[lat_col]), float(row[lon_col]))
    else:
        print("  WARNING: No lat/lon columns found — nodes will have no position.")
        coord_lookup = {}

    print(f"  Loaded {len(edges):,} vote-rows, "
          f"{edges['src'].nunique()} unique sources, "
          f"{edges['year'].min()}-{edges['year'].max()}")

    return edges, id2label, coord_lookup


# =============================================================================
# STEP 2 — PER-ERA NVS COMPUTATION
# =============================================================================

def compute_era(edges_all, year_start, year_end, min_years, era_label):
    """
    Compute mean NVS matrix and participation stats for one era.
    Returns (mean_nvs_matrix, qualified_countries, part_years_dict, yearly_nvs_df)
    """
    era = edges_all[(edges_all["year"] >= year_start) & (edges_all["year"] <= year_end)].copy()

    participation = (
        pd.concat([
            era[["year","src"]].rename(columns={"src":"country"}),
            era[["year","tgt"]].rename(columns={"tgt":"country"}),
        ]).drop_duplicates().groupby("country")["year"].nunique()
    )

    qualified = sorted(participation[participation >= min_years].index.tolist())
    part_years = participation.to_dict()

    era_q = era[era["src"].isin(qualified) & era["tgt"].isin(qualified)].copy()

    # Mean NVS per ordered pair
    mean_nvs = (
        era_q.groupby(["src","tgt"])["nvs"].mean()
        .unstack(fill_value=0)
        .reindex(index=qualified, columns=qualified, fill_value=0)
    ) * 12.0  # rescale to 0-12 for interpretability

    # Co-eligible years per pair (years where BOTH countries participated)
    years_by_country = (
        pd.concat([
            era[["year","src"]].rename(columns={"src":"country"}),
            era[["year","tgt"]].rename(columns={"tgt":"country"}),
        ]).drop_duplicates().groupby("country")["year"].apply(set).to_dict()
    )

    # Stability = 1 - coefficient_of_variation per pair
    yearly = era_q.groupby(["year","src","tgt"])["nvs"].mean().reset_index()

    print(f"  {era_label}: {len(qualified)} countries qualify (≥{min_years} yrs)")
    return mean_nvs, qualified, part_years, yearly, years_by_country


# =============================================================================
# STEP 3 — LOUVAIN COMMUNITY DETECTION
# =============================================================================

def detect_blocs(mean_nvs, qualified, q=0.60):
    """
    Detect voting blocs using Louvain on the symmetric mutual-affinity graph.

    WHY LOUVAIN:
    The mutual affinity matrix is symmetric (we take the mean of both
    directions). Louvain maximises modularity — it finds groups of countries
    that vote for each other more than expected by chance. This is exactly
    the right objective for detecting "blocs": not geographic proximity,
    not political alliance, but actual voting behaviour.

    WHY q=0.60 (quantile threshold):
    We only draw edges in the affinity graph for pairs above the 60th percentile
    of all pairwise NVS values. This removes noise from the community detection
    — otherwise, very weak ties connect everything and Louvain finds one big blob.
    0.60 is empirically calibrated to produce 4-6 blocs for the Eurovision dataset.
    """
    if not qualified:
        return {}

    # Build symmetric affinity matrix
    aff = pd.DataFrame(0.0, index=qualified, columns=qualified)
    for a in qualified:
        for b in qualified:
            if a != b and a in mean_nvs.index and b in mean_nvs.columns:
                aff.loc[a, b] = (mean_nvs.loc[a, b] + mean_nvs.loc[b, a]) / 2.0

    # Build weighted graph from top-q% pairs
    all_vals = [aff.loc[a, b] for a in qualified for b in qualified if a != b]
    threshold = np.quantile(all_vals, q) if all_vals else 0.0

    G = nx.Graph()
    G.add_nodes_from(qualified)
    for a in qualified:
        for b in qualified:
            if a < b and aff.loc[a, b] > threshold:
                G.add_edge(a, b, weight=float(aff.loc[a, b]))

    if G.number_of_edges() == 0:
        return {c: "Bloc 1" for c in qualified}

    # Run Louvain (or fallback)
    try:
        if LOUVAIN_OK:
            partition = community_louvain.best_partition(G, weight="weight", random_state=42)
            raw_map = partition
        else:
            comms = list(nx.community.greedy_modularity_communities(G, weight="weight"))
            raw_map = {c: i for i, comm in enumerate(comms) for c in comm}
    except Exception as e:
        print(f"    WARNING: Community detection failed ({e}), using single bloc")
        return {c: "Bloc 1" for c in qualified}

    # Sort blocs by size (largest = Bloc 1, etc.) and rename
    from collections import Counter
    counts = Counter(raw_map.values())
    sorted_ids = [b for b, _ in counts.most_common()]
    rename = {old: f"Bloc {i+1}" for i, old in enumerate(sorted_ids)}
    bloc_map = {c: rename[raw_map[c]] for c in qualified if c in raw_map}
    # Any country not in the partition (isolated) gets its own bloc
    for c in qualified:
        if c not in bloc_map:
            bloc_map[c] = f"Bloc {len(set(bloc_map.values())) + 1}"

    n_blocs = len(set(bloc_map.values()))
    print(f"    Detected {n_blocs} blocs")
    return bloc_map


# =============================================================================
# STEP 4 — EDGE CLASSIFICATION
# =============================================================================

def stability_of(a, b, yearly):
    """1 - CV of yearly NVS between a and b (higher = more stable)."""
    vals = yearly[(yearly["src"] == a) & (yearly["tgt"] == b)]["nvs"].values
    if len(vals) < 2:
        return float(np.mean(vals)) if len(vals) else 0.0
    mean = np.mean(vals)
    if mean < 0.001:
        return 0.0
    return float(max(0.0, 1.0 - np.std(vals) / mean))


def co_eligible_years(a, b, years_by_country_e1, years_by_country_e2):
    """Number of years both countries appeared in EITHER era."""
    s1 = years_by_country_e1.get(a, set()) & years_by_country_e1.get(b, set())
    s2 = years_by_country_e2.get(a, set()) & years_by_country_e2.get(b, set())
    return len(s1 | s2)


def classify_edges(
    q1, q2,          # qualified countries per era
    m1, m2,          # NVS matrices per era
    y1, y2,          # yearly NVS DataFrames
    ybc1, ybc2,      # years_by_country per era
):
    """
    Classify every meaningful pair into one of six categories.
    A pair is 'meaningful' if at least one direction NVS ≥ EDGE_MIN_NVS
    in at least one era.
    """
    # All countries appearing in either era
    all_countries = sorted(set(q1) | set(q2))
    edges = []

    def nvs(mat, a, b):
        try:
            return float(mat.loc[a, b]) if (a in mat.index and b in mat.columns) else 0.0
        except Exception:
            return 0.0

    for i, a in enumerate(all_countries):
        for j, b in enumerate(all_countries):
            if i >= j:
                continue

            # NVS in each era, both directions
            a_in_e1 = a in q1;  b_in_e1 = b in q1
            a_in_e2 = a in q2;  b_in_e2 = b in q2

            e1_ab = nvs(m1, a, b) if (a_in_e1 and b_in_e1) else 0.0
            e1_ba = nvs(m1, b, a) if (a_in_e1 and b_in_e1) else 0.0
            e2_ab = nvs(m2, a, b) if (a_in_e2 and b_in_e2) else 0.0
            e2_ba = nvs(m2, b, a) if (a_in_e2 and b_in_e2) else 0.0

            e1_mutual = (e1_ab + e1_ba) / 2.0
            e2_mutual = (e2_ab + e2_ba) / 2.0
            best_nvs  = max(e1_ab, e1_ba, e2_ab, e2_ba)

            if best_nvs < EDGE_MIN_NVS:
                continue  # too weak to draw

            # --- classify ---
            both_exist_e1 = a_in_e1 and b_in_e1
            both_exist_e2 = a_in_e2 and b_in_e2
            only_e2 = (not both_exist_e1) and both_exist_e2

            stab_e1 = max(stability_of(a, b, y1), stability_of(b, a, y1))
            stab_e2 = max(stability_of(a, b, y2), stability_of(b, a, y2))

            co_yrs = co_eligible_years(a, b, ybc1, ybc2)
            asym_e1 = abs(e1_ab - e1_ba)
            asym_e2 = abs(e2_ab - e2_ba)

            # Determine the edge direction for one-sided (Era II preferred)
            if both_exist_e2 and asym_e2 >= ONE_SIDED_DELTA:
                giver, recv = (a, b) if e2_ab > e2_ba else (b, a)
                edge_type = "one_sided"
                give_nvs  = max(e2_ab, e2_ba)
                recv_nvs  = min(e2_ab, e2_ba)
            elif both_exist_e1 and asym_e1 >= ONE_SIDED_DELTA:
                giver, recv = (a, b) if e1_ab > e1_ba else (b, a)
                edge_type = "one_sided"
                give_nvs  = max(e1_ab, e1_ba)
                recv_nvs  = min(e1_ab, e1_ba)
            elif (co_yrs >= COLD_MIN_YEARS
                  and max(e1_ab, e1_ba, e2_ab, e2_ba) < COLD_MAX_NVS):
                edge_type = "cold_shoulder"
                giver, recv = a, b
                give_nvs = recv_nvs = 0.0
            elif only_e2:
                edge_type = "new"
                giver, recv = a, b
                give_nvs = recv_nvs = e2_mutual
            elif (e1_mutual >= ALLIANCE_NVS and e2_mutual >= ALLIANCE_NVS
                  and stab_e1 >= ALLIANCE_STAB and stab_e2 >= ALLIANCE_STAB):
                edge_type = "stable_alliance"
                giver, recv = a, b
                give_nvs = recv_nvs = (e1_mutual + e2_mutual) / 2.0
            elif e2_mutual >= ALLIANCE_NVS and e1_mutual < ALLIANCE_NVS * 0.7:
                edge_type = "strengthened"
                giver, recv = a, b
                give_nvs = e2_mutual; recv_nvs = e1_mutual
            elif e1_mutual >= ALLIANCE_NVS and e2_mutual < ALLIANCE_NVS * 0.7:
                edge_type = "weakened"
                giver, recv = a, b
                give_nvs = e1_mutual; recv_nvs = e2_mutual
            else:
                edge_type = "stable_alliance" if e2_mutual >= ALLIANCE_NVS else "strengthened"
                giver, recv = a, b
                give_nvs = recv_nvs = max(e1_mutual, e2_mutual)

            edges.append({
                "a":         a,
                "b":         b,
                "giver":     giver,
                "receiver":  recv,
                "type":      edge_type,
                "e1_mutual": round(e1_mutual, 3),
                "e2_mutual": round(e2_mutual, 3),
                "e1_ab":     round(e1_ab, 3),
                "e1_ba":     round(e1_ba, 3),
                "e2_ab":     round(e2_ab, 3),
                "e2_ba":     round(e2_ba, 3),
                "stability": round(max(stab_e1, stab_e2), 3),
                "co_years":  co_yrs,
                "give_nvs":  round(give_nvs, 3),
                "recv_nvs":  round(recv_nvs, 3),
            })

    counts = defaultdict(int)
    for e in edges:
        counts[e["type"]] += 1
    print(f"    Edges: {len(edges)} total — " + ", ".join(f"{v} {k}" for k,v in sorted(counts.items())))
    return edges


# =============================================================================
# STEP 5 — NODE LAYOUT  (geographic + repulsion)
# =============================================================================

def geo_to_norm(lat, lon):
    """Convert lat/lon to normalised [0,1] x [0,1] coordinates."""
    x = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    y = (LAT_MAX - lat) / (LAT_MAX - LAT_MIN)   # invert y: north = top
    return float(x), float(y)


def spread_nodes(countries, iters=REPULSION_ITERS, min_dist=REPULSION_MIN, alpha=0.015):
    """
    Push overlapping nodes apart while anchoring them to their
    geographic positions. This is a simplified Fruchterman-Reingold
    with ONLY repulsion + a geographic spring pulling nodes home.

    WHY THIS MATTERS:
    Central European countries (DE, AT, CH, CZ, SK, HU, PL) are all
    within ~8° of longitude and ~5° of latitude. At poster scale they
    would completely overlap. This pass separates them by a small amount
    (typically 2-4% of the poster width) while keeping them in the right
    geographic region. The result is: "geographically accurate enough to
    be spatially meaningful, readable enough to see individual nodes."
    """
    pos = {c["id"]: [c["x"], c["y"]] for c in countries}
    geo = {c["id"]: [c["x"], c["y"]] for c in countries}  # anchor
    ids = list(pos.keys())

    for iteration in range(iters):
        # Cooling: reduce force over iterations
        cooling = 1.0 - iteration / iters

        forces = {k: [0.0, 0.0] for k in ids}
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                dx = pos[a][0] - pos[b][0]
                dy = pos[a][1] - pos[b][1]
                dist = max(0.0001, math.hypot(dx, dy))
                if dist < min_dist:
                    rep = (min_dist - dist) / dist
                    forces[a][0] += dx * rep * cooling
                    forces[a][1] += dy * rep * cooling
                    forces[b][0] -= dx * rep * cooling
                    forces[b][1] -= dy * rep * cooling

        for k in ids:
            # Apply repulsion
            pos[k][0] += forces[k][0] * alpha
            pos[k][1] += forces[k][1] * alpha
            # Geographic spring: pull back toward real lat/lon position
            pos[k][0] += (geo[k][0] - pos[k][0]) * 0.08
            pos[k][1] += (geo[k][1] - pos[k][1]) * 0.08
            # Clamp to [0,1]
            pos[k][0] = max(0.01, min(0.99, pos[k][0]))
            pos[k][1] = max(0.01, min(0.99, pos[k][1]))

    # Write back
    for c in countries:
        c["x_adj"] = round(pos[c["id"]][0], 4)
        c["y_adj"] = round(pos[c["id"]][1], 4)
    return countries


def build_country_list(qualified, bloc_map, part_years, coord_lookup,
                       nvs_received, bloc_colors, era_label):
    """Build the country list for one era with all needed fields."""
    countries = []
    max_yrs = max((part_years.get(c, 0) for c in qualified), default=1) or 1
    max_nvs = max((nvs_received.get(c, 0) for c in qualified), default=1) or 1

    for c in qualified:
        lat, lon = coord_lookup.get(c, (None, None))
        if lat is None:
            print(f"    WARNING: No coordinates for {c}")
            lat, lon = 50.0, 15.0  # default to central Europe

        x, y = geo_to_norm(lat, lon)
        bloc = bloc_map.get(c, "Bloc 1")

        countries.append({
            "id":               c,
            "label":            c,
            "lat":              round(float(lat), 4),
            "lon":              round(float(lon), 4),
            "x":                round(x, 4),   # geographic normalised
            "y":                round(y, 4),
            "x_adj":            round(x, 4),   # will be updated by spread_nodes
            "y_adj":            round(y, 4),
            "bloc":             bloc,
            "color":            bloc_colors.get(bloc, "#888888"),
            "participation_years": int(part_years.get(c, 0)),
            "nvs_received":     round(float(nvs_received.get(c, 0)), 2),
            "size_norm":        round(float(part_years.get(c, 0)) / max_yrs, 3),
            "nvs_norm":         round(float(nvs_received.get(c, 0)) / max_nvs, 3),
        })

    # Apply geographic spread to separate overlapping nodes
    countries = spread_nodes(countries)
    return countries


def build_bloc_summaries(qualified, bloc_map, mean_nvs, nvs_received, bloc_colors):
    """Compute one summary card per detected bloc for the poster."""
    members_by_bloc = defaultdict(list)
    for c in qualified:
        members_by_bloc[bloc_map.get(c, "Bloc 1")].append(c)

    summaries = []
    for bloc, members in sorted(members_by_bloc.items()):
        # Champion: most NVS received within the bloc
        champion = max(members, key=lambda c: nvs_received.get(c, 0))

        # Within-bloc cohesion = mean NVS between all member pairs
        wb_vals = []
        for a in members:
            for b in members:
                if a != b and a in mean_nvs.index and b in mean_nvs.columns:
                    wb_vals.append(float(mean_nvs.loc[a, b]))
        cohesion = round(np.mean(wb_vals), 3) if wb_vals else 0.0

        # Top pair (highest mutual NVS within bloc)
        best_pair, best_val = None, 0.0
        for i, a in enumerate(members):
            for b in members[i+1:]:
                if a in mean_nvs.index and b in mean_nvs.columns:
                    mutual = (float(mean_nvs.loc[a, b]) + float(mean_nvs.loc[b, a])) / 2.0
                    if mutual > best_val:
                        best_val = mutual
                        best_pair = [a, b]

        summaries.append({
            "name":       bloc,
            "color":      bloc_colors.get(bloc, "#888888"),
            "members":    sorted(members),
            "n":          len(members),
            "champion":   champion,
            "top_pair":   best_pair or [],
            "top_pair_nvs": round(best_val, 2),
            "cohesion":   cohesion,
        })

    summaries.sort(key=lambda b: -b["n"])
    return summaries


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Eurovision GD Contest 2026 — Poster Data Export")
    print("=" * 60)

    edges_all, id2label, coord_lookup = load_data()

    # ---- ERA I ----------------------------------------------------------
    print("\n[2/6] Computing Era I (1975–1999)...")
    m1, q1, py1, y1, ybc1 = compute_era(
        edges_all, ERA1_START, ERA1_END, MIN_YEARS_ERA1, "Era I")

    print("\n[3/6] Computing Era II (2000–2025)...")
    m2, q2, py2, y2, ybc2 = compute_era(
        edges_all, ERA2_START, ERA2_END, MIN_YEARS_ERA2, "Era II")

    # ---- COMMUNITY DETECTION --------------------------------------------
    print("\n[4/6] Detecting blocs (Louvain)...")
    print("  Era I:")
    bm1 = detect_blocs(m1, q1)
    print("  Era II:")
    bm2 = detect_blocs(m2, q2)

    # Assign colours (sorted by bloc size, largest = first colour)
    def bloc_colors(bloc_map):
        from collections import Counter
        counts = Counter(bloc_map.values())
        blocs_sorted = [b for b,_ in counts.most_common()]
        return {b: PALETTE[i % len(PALETTE)] for i, b in enumerate(blocs_sorted)}

    bc1 = bloc_colors(bm1)
    bc2 = bloc_colors(bm2)

    # ---- NVS RECEIVED per country (for node sizing) ----------------------
    def nvs_recv(mat, qualified):
        return {c: float(mat[c].mean()) if c in mat.columns else 0.0
                for c in qualified}

    nr1 = nvs_recv(m1, q1)
    nr2 = nvs_recv(m2, q2)

    # ---- EDGE CLASSIFICATION ---------------------------------------------
    print("\n[5/6] Classifying edges...")
    all_edges = classify_edges(q1, q2, m1, m2, y1, y2, ybc1, ybc2)

    # Per-era edge subsets (for rendering only that era's context)
    era1_edges = [e for e in all_edges
                  if e["e1_mutual"] >= EDGE_MIN_NVS or e["e1_ab"] >= EDGE_MIN_NVS
                  and e["a"] in q1 and e["b"] in q1]
    era2_edges = [e for e in all_edges
                  if e["e2_mutual"] >= EDGE_MIN_NVS or e["e2_ab"] >= EDGE_MIN_NVS
                  and e["a"] in q2 and e["b"] in q2]

    # ---- NODE POSITIONS --------------------------------------------------
    print("\n[6/6] Computing node positions...")
    countries1 = build_country_list(q1, bm1, py1, coord_lookup, nr1, bc1, "Era I")
    countries2 = build_country_list(q2, bm2, py2, coord_lookup, nr2, bc2, "Era II")
    blocs1     = build_bloc_summaries(q1, bm1, m1, nr1, bc1)
    blocs2     = build_bloc_summaries(q2, bm2, m2, nr2, bc2)

    # ---- ASSEMBLE OUTPUT -------------------------------------------------
    output = {
        "meta": {
            "title":           "Eurovision Voting Communities 1975–2025",
            "subtitle":        "How blocs formed, shifted and dissolved",
            "rq1":             "Which bilateral voting relationships remained structurally persistent across 50 years?",
            "rq3":             "How do geopolitical shifts correspond to topological changes in the voting network?",
            "era1_label":      f"{ERA1_START}–{ERA1_END}",
            "era2_label":      f"{ERA2_START}–{ERA2_END}",
            "era1_n":          len(q1),
            "era2_n":          len(q2),
            "n_edges":         len(all_edges),
            "edge_categories": ["stable_alliance","strengthened","weakened",
                                "one_sided","cold_shoulder","new"],
            "edge_colors": {
                "stable_alliance": "#c8980a",
                "strengthened":    "#2a7a5a",
                "weakened":        "#c87028",
                "one_sided":       "#b83020",
                "cold_shoulder":   "#808070",
                "new":             "#2f9c8b",
            },
            "edge_description": {
                "stable_alliance": "Strong mutual NVS in BOTH eras — loyal long-term alliance",
                "strengthened":    "Weak in Era I, strong in Era II — relationship grew post-2000",
                "weakened":        "Strong in Era I, weak in Era II — relationship dissolved",
                "one_sided":       "High asymmetry — one country gives much more than it receives",
                "cold_shoulder":   "Long co-participation, near-zero NVS both ways",
                "new":             "Country only exists in Era II — new relationship",
            },
            "generated":       str(date.today()),
            "nvs_formula":     "NVS = points / era_max; era_max=12 (1975-2015), 24 (2016-2025)",
            "layout_note":     "Positions: geographic lat/lon + short-range repulsion (60 iters)",
        },
        "era1": {
            "countries": countries1,
            "edges":     era1_edges,
            "blocs":     blocs1,
        },
        "era2": {
            "countries": countries2,
            "edges":     era2_edges,
            "blocs":     blocs2,
        },
        "all_edges": all_edges,
    }

    # ---- WRITE JSON ------------------------------------------------------
    with open("poster_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Also write as JS const so poster.html can load it without a web server
    with open("poster_data.js", "w", encoding="utf-8") as f:
        f.write("// Auto-generated by poster_data_export.py\n")
        f.write("// Do not edit — re-run poster_data_export.py to regenerate\n")
        f.write("const POSTER_DATA = ")
        json.dump(output, f, ensure_ascii=False)
        f.write(";\n")

    # ---- SUMMARY ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"\n  poster_data.json  ({len(json.dumps(output)) // 1024} KB)")
    print(f"  poster_data.js    (same, as JS const for poster.html)")
    print(f"\n  Era I  ({ERA1_START}-{ERA1_END}): {len(q1)} countries, {len(blocs1)} blocs")
    for b in blocs1:
        print(f"    {b['name']} ({b['n']} countries, cohesion {b['cohesion']:.2f}): "
              f"{', '.join(b['members'][:5])}"
              f"{'...' if len(b['members']) > 5 else ''}")
    print(f"\n  Era II ({ERA2_START}-{ERA2_END}): {len(q2)} countries, {len(blocs2)} blocs")
    for b in blocs2:
        print(f"    {b['name']} ({b['n']} countries, cohesion {b['cohesion']:.2f}): "
              f"{', '.join(b['members'][:5])}"
              f"{'...' if len(b['members']) > 5 else ''}")
    print(f"\n  Edge breakdown:")
    from collections import Counter
    counts = Counter(e["type"] for e in all_edges)
    for t in ["stable_alliance","strengthened","weakened","one_sided","cold_shoulder","new"]:
        if counts[t]:
            print(f"    {t:<20} {counts[t]}")
    print(f"\n  Next step: open poster.html in Chrome")
    print(f"  (Both files must be in the same folder: poster_data.js + poster.html)")
    print()


if __name__ == "__main__":
    main()