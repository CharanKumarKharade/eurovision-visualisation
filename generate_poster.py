#!/usr/bin/env python3
"""
generate_poster.py
==================
Reads poster_data.json and embeds it directly into poster.html,
producing poster_standalone.html — a single file you can open
in any browser without a web server.

Usage:
    python generate_poster.py

Requires:
    poster_data.json   (produced by poster_data_export.py)
    poster.html        (the D3 renderer template)
"""
import json, sys, re, os

# Check inputs exist
for f in ["poster_data.json", "poster.html"]:
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} not found. Run poster_data_export.py first.")

# Load data
with open("poster_data.json", encoding="utf-8") as f:
    data = json.load(f)

data_js = "const POSTER_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n"

# Load HTML template
with open("poster.html", encoding="utf-8") as f:
    html = f.read()

# Replace the external script tag with inline data
html = html.replace(
    '<script src="poster_data.js"></script>',
    f'<script>\n{data_js}</script>'
)

# Write standalone file
with open("poster_standalone.html", "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize("poster_standalone.html") // 1024
print(f"  Written: poster_standalone.html ({size_kb} KB)")
print(f"  Open this file directly in Chrome — no server needed.")