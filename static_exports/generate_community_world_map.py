"""Generate a static Eurovision community world map.

This script reproduces the app's 1975-2025 filtering and community detection,
then writes a static PNG map to the local static_exports directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import networkx as nx
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError("networkx is required to build the community map") from exc


ROOT = Path(__file__).resolve().parents[1]
NODES_FILE = ROOT / "nodes_with_coordinates.csv"
EDGES_FILE = ROOT / "eurovision_senior.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

ROOT_START = 1975
ROOT_END = 2025
DEFAULT_MIN_PARTICIPATION = 21
DEFAULT_MIN_MUTUAL_EDGE = 2.0

ERA_MAX = {year: 12 for year in range(1975, 2016)}
ERA_MAX.update({year: 24 for year in range(2016, 2026)})


def find_col(columns, exact, fuzzy):
    for column in columns:
        if column.lower() in exact:
            return column

    for column in columns:
        if any(key in column.lower() for key in fuzzy):
            return column

    raise ValueError(f"Column not found. exact={exact}, fuzzy={fuzzy}")


def load_data(nodes_file: Path, edges_file: Path):
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)

    nodes.columns = [column.strip().lower() for column in nodes.columns]
    edges.columns = [column.strip().lower() for column in edges.columns]

    if not {"id", "label"}.issubset(nodes.columns):
        raise ValueError("nodes_with_coordinates.csv must contain at least: id, label")

    if "score_type" in edges.columns:
        edges["score_type"] = edges["score_type"].astype(str).str.strip().str.lower()
        if "total" in set(edges["score_type"].dropna().unique()):
            edges = edges[edges["score_type"] == "total"].copy()

    if "round" in edges.columns:
        edges["round"] = edges["round"].astype(str).str.strip().str.lower()
        if "final" in set(edges["round"].dropna().unique()):
            edges = edges[edges["round"] == "final"].copy()

    src_col = find_col(edges.columns, {"source", "from", "from_country"}, ["source", "from", "voter"])
    tgt_col = find_col(edges.columns, {"target", "to", "to_country"}, ["target", "to", "recip"])
    year_col = find_col(edges.columns, {"year"}, ["year"])

    numeric_cols = [column for column in edges.columns if pd.api.types.is_numeric_dtype(edges[column])]
    pts_col = None
    for column in numeric_cols:
        if any(key in column.lower() for key in ["point", "score", "pts", "value", "weight"]):
            pts_col = column
            break

    if pts_col is None:
        if "weight" in edges.columns and pd.api.types.is_numeric_dtype(edges["weight"]):
            pts_col = "weight"
        else:
            raise ValueError(f"No valid points column found. Available numeric: {numeric_cols}")

    edges = edges.rename(columns={year_col: "year", src_col: "source", tgt_col: "target", pts_col: "points"})
    edges["year"] = pd.to_numeric(edges["year"], errors="coerce")
    edges["points"] = pd.to_numeric(edges["points"], errors="coerce").fillna(0)
    edges = edges.dropna(subset=["year", "source", "target"])
    edges["year"] = edges["year"].astype(int)

    id2label = dict(zip(nodes["id"].astype(str), nodes["label"].astype(str)))
    return nodes, edges, id2label


def compute_participation_counts(edges: pd.DataFrame):
    src_years = edges.groupby("source")["year"].nunique().rename("years")
    tgt_years = edges.groupby("target")["year"].nunique().rename("years")
    combined = pd.concat([src_years, tgt_years]).groupby(level=0).max()
    return combined.to_dict()


def participation_years_for_label(label: str, id2label: dict[str, str], participation_counts: dict[str, int]) -> int:
    for cid, mapped_label in id2label.items():
        if mapped_label == label:
            return int(participation_counts.get(cid, 0))

    return int(participation_counts.get(label, 0))


def compute_period_data(edges: pd.DataFrame, start_year: int, end_year: int, id2label: dict[str, str]):
    df = edges[(edges["year"] >= start_year) & (edges["year"] <= end_year)].copy()
    if df.empty:
        return None

    years_with_data = sorted(df["year"].dropna().astype(int).unique())

    actual = (
        df.groupby(["year", "source", "target"], as_index=False)["points"]
        .sum()
    )
    actual = actual[actual["points"] > 0].copy()

    participants_by_year = {}
    for year in years_with_data:
        yr_df = df[df["year"] == year]
        participants_by_year[year] = set(yr_df["source"].astype(str)) | set(yr_df["target"].astype(str))

    pair_rows = [
        (year, source, target)
        for year in years_with_data
        for source in participants_by_year[year]
        for target in participants_by_year[year]
        if source != target
    ]

    eligible = pd.DataFrame(pair_rows, columns=["year", "source", "target"])

    yr_agg = eligible.merge(actual, on=["year", "source", "target"], how="left")
    yr_agg["points"] = yr_agg["points"].fillna(0)
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
            years_eligible=("year", "nunique"),
        )
    )

    agg = agg.merge(raw_total_df, on=["source", "target"], how="left")
    agg["total_votes"] = agg["total_votes"].fillna(0)
    agg["nvs_mean"] = agg["nvs_sum"] / agg["years_eligible"]
    agg["nvs_score"] = agg["nvs_mean"] * 12
    agg["src_label"] = agg["source"].map(id2label).fillna(agg["source"])
    agg["tgt_label"] = agg["target"].map(id2label).fillna(agg["target"])

    return {
        "df": df,
        "agg": agg,
        "years": years_with_data,
    }


def build_matrix(agg: pd.DataFrame, value_col: str, order: list[str]) -> pd.DataFrame:
    matrix = agg.pivot(index="src_label", columns="tgt_label", values=value_col).fillna(0)
    return matrix.reindex(index=order, columns=order, fill_value=0)


def detect_communities_from_nvs(matrix_df: pd.DataFrame, min_edge_weight: float = DEFAULT_MIN_MUTUAL_EDGE):
    countries = list(matrix_df.index)
    graph = nx.Graph()

    for country in countries:
        graph.add_node(country)

    for i, src in enumerate(countries):
        for j, tgt in enumerate(countries):
            if i >= j:
                continue

            weight = float((matrix_df.loc[src, tgt] + matrix_df.loc[tgt, src]) / 2)
            if weight >= min_edge_weight:
                graph.add_edge(src, tgt, weight=weight)

    if graph.number_of_edges() == 0:
        return pd.DataFrame(), graph

    try:
        communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
        method = "Louvain"
    except Exception:
        communities = nx.community.greedy_modularity_communities(graph, weight="weight")
        method = "Greedy modularity"

    rows = []
    for idx, community in enumerate(communities, start=1):
        members = sorted(list(community))
        subgraph = graph.subgraph(members)
        rows.append(
            {
                "Community": f"C{idx}",
                "Method": method,
                "Size": len(members),
                "Members": ", ".join(members),
                "Internal Edges": subgraph.number_of_edges(),
                "Average Internal Weight": (
                    float(np.mean([edge_data["weight"] for _, _, edge_data in subgraph.edges(data=True)]))
                    if subgraph.number_of_edges() > 0
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows), graph


def build_community_color_map(communities_df: pd.DataFrame):
    if communities_df is None or communities_df.empty or "Community" not in communities_df.columns:
        return {}

    palette = [
        "#1f4e79",
        "#d1495b",
        "#2a9d8f",
        "#f4a261",
        "#6a4c93",
        "#7f5539",
        "#577590",
        "#4d908e",
        "#b56576",
        "#3a86ff",
        "#8338ec",
        "#fb5607",
        "#2b9348",
        "#9b5de5",
        "#e63946",
        "#8ac926",
    ]

    community_names = communities_df["Community"].dropna().astype(str).tolist()
    return {community: palette[idx % len(palette)] for idx, community in enumerate(community_names)}


def find_geo_columns(nodes_df: pd.DataFrame):
    columns = {str(column).lower(): column for column in nodes_df.columns}
    lat_col = columns.get("lat") or columns.get("latitude") or columns.get("y")
    lon_col = columns.get("lon") or columns.get("long") or columns.get("longitude") or columns.get("x")
    return lat_col, lon_col


def make_community_world_map_figure(nodes_df: pd.DataFrame, communities_df: pd.DataFrame):
    if nodes_df is None or nodes_df.empty or communities_df is None or communities_df.empty:
        return None

    lat_col, lon_col = find_geo_columns(nodes_df)
    if not lat_col or not lon_col or "label" not in nodes_df.columns:
        return None

    community_lookup = {}
    for _, row in communities_df.iterrows():
        members = [member.strip() for member in str(row["Members"]).split(",") if member.strip()]
        for member in members:
            community_lookup[member] = str(row["Community"])

    plot_df = nodes_df.copy()
    plot_df["label"] = plot_df["label"].astype(str)
    plot_df["community"] = plot_df["label"].map(community_lookup)
    plot_df = plot_df.dropna(subset=["community", lat_col, lon_col])

    if plot_df.empty:
        return None

    community_colors = build_community_color_map(communities_df)
    fig = go.Figure()

    for community in communities_df["Community"].dropna().astype(str).tolist():
        sub = plot_df[plot_df["community"] == community].sort_values("label")
        if sub.empty:
            continue

        fig.add_trace(
            go.Scattergeo(
                lon=sub[lon_col],
                lat=sub[lat_col],
                text=sub["label"].astype(str),
                mode="markers+text",
                textposition="top center",
                name=community,
                marker=dict(
                    size=11,
                    color=community_colors.get(community, "#64748b"),
                    opacity=0.95,
                    line=dict(width=1.1, color="rgba(255,255,255,0.95)"),
                ),
                hovertemplate="<b>%{text}</b><br>Community: " + community + "<extra></extra>",
            )
        )

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
        height=760,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", y=-0.05),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate a static Eurovision community world map.")
    parser.add_argument("--start-year", type=int, default=ROOT_START)
    parser.add_argument("--end-year", type=int, default=ROOT_END)
    parser.add_argument("--min-participation", type=int, default=DEFAULT_MIN_PARTICIPATION)
    parser.add_argument("--min-mutual-edge", type=float, default=DEFAULT_MIN_MUTUAL_EDGE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "community_world_map_1975_2025_min21.png")
    parser.add_argument("--also-html", action="store_true", help="Write a matching HTML copy next to the PNG")
    parser.add_argument("--html-only", action="store_true", help="Skip PNG export and write only HTML output")
    args = parser.parse_args()

    nodes_df, edges_df, id2label = load_data(NODES_FILE, EDGES_FILE)
    participation_counts = compute_participation_counts(edges_df)

    all_country_ids = sorted(set(edges_df["source"].dropna().astype(str).unique()) | set(edges_df["target"].dropna().astype(str).unique()))
    all_labels = [id2label.get(country_id, country_id) for country_id in all_country_ids]
    filtered_labels = [
        label
        for label in all_labels
        if participation_years_for_label(label, id2label, participation_counts) >= args.min_participation
    ]

    period_data = compute_period_data(edges_df, args.start_year, args.end_year, id2label)
    if period_data is None:
        raise RuntimeError("No data available for the selected year range")

    agg = period_data["agg"]
    order = sorted(filtered_labels)
    matrix = build_matrix(agg, "nvs_score", order)

    communities_df, _ = detect_communities_from_nvs(matrix, min_edge_weight=args.min_mutual_edge)
    if communities_df.empty:
        raise RuntimeError("No communities were detected for the selected filters")

    fig = make_community_world_map_figure(nodes_df, communities_df)
    if fig is None:
        raise RuntimeError("Unable to build the world map figure because geographic coordinates were not found")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    html_output = args.output.with_suffix(".html")

    if args.html_only:
        fig.write_html(str(html_output), include_plotlyjs="cdn", full_html=True)
        print(f"Wrote HTML map to {html_output}")
    else:
        try:
            fig.write_image(str(args.output), scale=2)
            print(f"Wrote static map to {args.output}")
        except Exception as exc:
            fig.write_html(str(html_output), include_plotlyjs="cdn", full_html=True)
            print(f"PNG export failed because a Chrome-compatible browser was not available: {exc}")
            print(f"Wrote HTML fallback to {html_output}")

    summary_output = args.output.with_suffix(".csv")
    communities_df.to_csv(summary_output, index=False)

    print(f"Wrote community table to {summary_output}")
    if args.also_html:
        if not args.html_only and html_output.exists():
            print(f"Wrote HTML fallback to {html_output}")


if __name__ == "__main__":
    main()