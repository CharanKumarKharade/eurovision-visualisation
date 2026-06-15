Eurovision Voting Explorer
===========================

Interactive analytical dashboard for exploring dynamic Eurovision voting behaviour using directed network analysis, normalized voting scores, temporal comparison, and community detection.

--------------------------------------------------------------------
PROJECT OVERVIEW
--------------------------------------------------------------------

This application provides an interactive environment for analysing Eurovision Song Contest voting patterns across different years and periods.

The dashboard focuses on:

- Directed voting relationships
- Normalized voting behaviour
- Temporal evolution of alliances
- Voting-profile similarity
- Emerging and declining relationships
- Community / bloc detection
- Pairwise trend analysis

The system was designed as part of a Master's thesis focused on visualizing dynamic multivariate voting networks.

--------------------------------------------------------------------
MAIN FEATURES
--------------------------------------------------------------------

1. Directed NVS Matrix
----------------------
Displays a directed country-to-country voting matrix based on the Normalized Voting Score (NVS).

Rows:
    Countries giving votes

Columns:
    Countries receiving votes

The matrix allows comparison of voting affinity independent of Eurovision rule changes.

--------------------------------------------------------------------

2. Raw Total Vote Matrix
------------------------
Shows cumulative raw points exchanged between countries.

Useful for:
- Historical magnitude analysis
- Long-term total vote accumulation

--------------------------------------------------------------------

3. Voting Profile Correlation Matrix
------------------------------------
Computes Pearson correlation between country voting profiles.

This identifies:
- Countries that vote similarly
- Shared voting behaviour
- Cultural or regional similarity patterns

IMPORTANT:
This is NOT direct voting affinity.
It measures similarity in voting behaviour.

--------------------------------------------------------------------

4. Dynamic Period Comparison
----------------------------
Compares two selected time periods and computes changes in voting affinity.

Example:
    1975–1999 vs 2000–2025

Outputs:
- Strengthened relationships
- Weakened relationships
- Difference heatmap

--------------------------------------------------------------------

5. Pair Trend Analysis
----------------------
Detailed longitudinal analysis of a selected directed relationship.

Includes:
- Full yearly timeline
- Rolling averages
- Change-point estimation
- Stability metrics
- Trend slopes
- Relationship classification

--------------------------------------------------------------------

6. Relationship Classification
------------------------------
Each directed pair is classified heuristically into categories such as:

- Strong stable alliance
- Emerging relationship
- Declining relationship
- Volatile relationship
- Weak stable relationship

Classification is based on:
- Mean NVS
- Variability
- Stability
- Trend slope

NOTE:
These are exploratory heuristic categories and not formal statistical tests.

--------------------------------------------------------------------

7. Community / Bloc Detection
-----------------------------
The application detects voting blocs using network community detection.

Methodology:
- Directed voting relationships are converted into an undirected mutual-affinity graph
- Reciprocal voting intensity is averaged
- Weak edges are removed using a threshold
- Louvain modularity optimization is applied

Detected communities represent:
- Mutual voting alliances
- Dense reciprocal voting structures
- Voting blocs

--------------------------------------------------------------------

8. Participation Filtering
--------------------------
Countries can be filtered based on minimum participation years.

This ensures:
- Robust comparisons
- Reduced noise from short-lived participants

--------------------------------------------------------------------

9. Top-N Country Filtering
--------------------------
Users may restrict the visualization to the strongest countries by overall NVS strength.

Useful for:
- Reducing visual clutter
- Focusing on dominant relationships

--------------------------------------------------------------------

10. Interactive Exploration
---------------------------
The dashboard supports:
- Hover exploration
- Interactive filtering
- Dynamic thresholding
- Pair selection
- Downloadable HTML visualizations

--------------------------------------------------------------------

11. Static Community World Map Export
------------------------------------
If you need a non-Streamlit output, run the standalone exporter in
`static_exports/` to generate a static PNG of the detected communities on a
world map.

Default settings:
- Year range: 1975 to 2025
- Minimum participation: 21 years

Run:
    python static_exports/generate_community_world_map.py

--------------------------------------------------------------------
NORMALIZED VOTING SCORE (NVS)
--------------------------------------------------------------------

The core metric used throughout the application is the Normalized Voting Score (NVS).

Definition:
    NVS = points_received / maximum_possible_points

Historical normalization:
- 1975–2015:
      maximum = 12
- 2016 onward:
      maximum = 24

This normalization allows fair comparison across Eurovision rule changes.

The final displayed score is scaled to:
    0–12

--------------------------------------------------------------------
TECHNOLOGIES USED
--------------------------------------------------------------------

Frontend:
- Streamlit

Visualization:
- Plotly

Data Processing:
- Pandas
- NumPy

Network Analysis:
- NetworkX

Clustering:
- SciPy

--------------------------------------------------------------------
REQUIRED FILES
--------------------------------------------------------------------

1. nodes_with_coordinates.csv
--------------------------------
Contains:
- country id
- country label
- optional coordinates

Required columns:
- id
- label

--------------------------------------------------------------------

2. eurovision_senior.csv
-------------------------
Contains Eurovision voting data.

Expected fields include:
- source country
- target country
- year
- points
- score_type
- round

The application automatically detects relevant column names.

--------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------

1. Clone the repository

2. Install dependencies

Example:
    pip install -r requirements.txt

3. Run the application

Example:
    streamlit run app.py

--------------------------------------------------------------------
SUGGESTED REQUIREMENTS.TXT
--------------------------------------------------------------------

streamlit
pandas
numpy
plotly
networkx
scipy

--------------------------------------------------------------------
COMMUNITY DETECTION DETAILS
--------------------------------------------------------------------

The community detection process follows these steps:

1. Construct directed voting matrix
2. Convert to mutual-affinity graph:
       weight(A,B) =
       average(NVS(A→B), NVS(B→A))
3. Remove weak edges below threshold
4. Apply Louvain modularity optimization

Interpretation:
Communities represent reciprocal voting blocs rather than one-sided influence structures.

--------------------------------------------------------------------
ACADEMIC CONTEXT
--------------------------------------------------------------------

This application was developed for research purposes within a Master's thesis focused on:

- Network visualization
- Dynamic graph analysis
- Temporal relationship analysis
- Eurovision voting behaviour
- Community detection in weighted networks

--------------------------------------------------------------------
AUTHOR
--------------------------------------------------------------------

Charan Kumar Kharade Somoji Rao

Master's in Software Technology
HFT Stuttgart

--------------------------------------------------------------------
LICENSE
--------------------------------------------------------------------

This project is intended for academic and research purposes.
