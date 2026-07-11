#!/usr/bin/env python3
"""
build_poster.py
===============
ONE-FILE solution — no other files needed except poster_data.json.

Usage:
    cd /workspaces/eurovision-visualisation
    python build_poster.py

Output: poster_standalone.html  (open this in Chrome)
"""
import json, sys, os

if not os.path.exists("poster_data.json"):
    sys.exit(
        "ERROR: poster_data.json not found.\n"
        "Run first:  python poster_data_export.py"
    )

with open("poster_data.json", encoding="utf-8") as f:
    data = json.load(f)

data_js = "const POSTER_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n"

# HTML template is embedded below — no poster.html needed
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Eurovision Voting Communities — GD Contest 2026</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #2a2a2a;
    display: flex; flex-direction: column;
    align-items: center; padding: 24px 16px;
    font-family: -apple-system, sans-serif;
    min-height: 100vh;
  }

  /* ── Controls (NOT part of poster) ── */
  #controls {
    display: flex; gap: 12px; margin-bottom: 18px; align-items: center;
  }
  #controls button {
    background: #4a9eff; color: white; border: none;
    padding: 9px 20px; border-radius: 6px; cursor: pointer;
    font-size: 14px; font-weight: 500;
  }
  #controls button:hover { background: #2a7edf; }
  #controls button.sec { background: #555; }
  #controls button.sec:hover { background: #444; }
  #controls span { color: #aaa; font-size: 13px; }

  /* ── Poster ── */
  #poster-wrap {
    background: #f5f0e4;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
    border-radius: 3px;
    overflow: visible;
  }
  #poster-svg { display: block; }

  /* tooltip */
  #tooltip {
    position: fixed; pointer-events: none;
    background: rgba(25,20,10,0.92); color: #f0ead8;
    padding: 7px 10px; border-radius: 5px;
    font-size: 12px; line-height: 1.5;
    max-width: 220px; display: none;
    z-index: 999;
  }
</style>
</head>
<body>

<div id="controls">
  <button onclick="downloadSVG()">⬇ Download SVG</button>
  <button class="sec" onclick="downloadPNG()">⬇ Download PNG (1×)</button>
  <span>Open in Figma: File → Place Image → select the SVG</span>
</div>

<div id="poster-wrap">
  <svg id="poster-svg"></svg>
</div>
<div id="tooltip"></div>

<!-- Load data first, then D3 -->
<script src="poster_data.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
// ═══════════════════════════════════════════════════════════════
//  POSTER DIMENSIONS  (A0 portrait ratio ≈ 1 : 1.414)
// ═══════════════════════════════════════════════════════════════
const W = 1200, H = 1680;
const PAD = 18;

// Panel geometry
const TITLE_H   = 108;
const PANEL_GAP = 18;
const LEGEND_H  = 68;
const CARDS_H   = 170;
const FOOTER_H  = 40;

const PANEL_Y = TITLE_H;
const PANEL_H = H - TITLE_H - LEGEND_H - CARDS_H - FOOTER_H - PAD;
const PANEL_W = (W - PAD*2 - PANEL_GAP) / 2;
const P1_X    = PAD;
const P2_X    = PAD + PANEL_W + PANEL_GAP;

const MAP_INNER_PAD = 26;   // inside each panel before geographic content

// ═══════════════════════════════════════════════════════════════
//  COLOURS & STYLES
// ═══════════════════════════════════════════════════════════════
const EDGE_STYLE = {
  stable_alliance: { color:"#c8980a", width:2.8, dash:"none",   label:"Loyal alliance (both eras)" },
  strengthened:    { color:"#2a7a5a", width:2.0, dash:"none",   label:"Strengthened post-2000" },
  weakened:        { color:"#c87028", width:1.8, dash:"5,3",    label:"Weakened post-2000" },
  one_sided:       { color:"#b83020", width:1.8, dash:"none",   label:"Unrequited (one-sided)" },
  cold_shoulder:   { color:"#808070", width:1.4, dash:"4,3",    label:"Cold shoulder (decades, ~0 NVS)" },
  new:             { color:"#2f9c8b", width:1.4, dash:"none",   label:"New in Era II" },
};
const PAPER = "#f5f0e4";
const INK   = "#1a1408";

// ═══════════════════════════════════════════════════════════════
//  EUROPE LAND OUTLINE  (approximate, as normalised coords)
//  Pre-projected using lon→x = (lon+30)/90, lat→y = (72-lat)/44
// ═══════════════════════════════════════════════════════════════
const EUROPE_OUTLINE = [
  // Iceland
  [0.133,0.159],[0.155,0.136],[0.200,0.136],[0.222,0.159],[0.200,0.182],[0.133,0.159],
  // gap — move to Scandinavia
  null,
  [0.400,0.023],[0.488,0.000],[0.600,0.045],[0.633,0.068],[0.611,0.091],
  [0.566,0.068],[0.511,0.136],[0.477,0.159],[0.444,0.204],[0.422,0.250],
  [0.433,0.295],[0.411,0.318],[0.400,0.295],[0.366,0.295],[0.333,0.318],
  [0.311,0.295],[0.288,0.318],[0.266,0.340],[0.244,0.363],[0.222,0.409],
  [0.200,0.432],[0.177,0.409],[0.155,0.409],[0.133,0.432],[0.122,0.454],
  // Iberia
  [0.133,0.500],[0.111,0.545],[0.088,0.590],[0.077,0.613],[0.088,0.659],
  [0.122,0.681],[0.166,0.681],[0.188,0.636],[0.200,0.590],[0.222,0.545],
  // France coast + Italy + SE
  [0.288,0.636],[0.333,0.636],[0.355,0.681],[0.377,0.727],[0.422,0.772],
  [0.444,0.772],[0.455,0.750],[0.433,0.727],[0.411,0.704],[0.422,0.659],
  // Balkans + Caucasus
  [0.466,0.659],[0.511,0.636],[0.533,0.636],[0.555,0.659],[0.577,0.681],
  [0.622,0.704],[0.666,0.727],[0.711,0.750],[0.755,0.772],[0.800,0.772],
  [0.833,0.750],[0.866,0.750],[0.888,0.727],[0.900,0.704],[0.888,0.681],
  [0.866,0.659],[0.855,0.636],[0.833,0.636],[0.811,0.613],[0.800,0.590],
  [0.800,0.568],[0.822,0.545],[0.833,0.522],[0.822,0.500],[0.800,0.500],
  // close back to Norway
  [0.666,0.454],[0.644,0.409],[0.622,0.386],[0.600,0.363],[0.622,0.318],
  [0.644,0.295],[0.666,0.272],[0.644,0.227],[0.622,0.204],[0.600,0.182],
  [0.566,0.136],[0.533,0.113],[0.488,0.068],[0.444,0.045],[0.400,0.023],
];

// ═══════════════════════════════════════════════════════════════
//  HELPER — map normalised coords to panel SVG coords
// ═══════════════════════════════════════════════════════════════
function toPanel(xn, yn, panelX) {
  const mx = MAP_INNER_PAD, mw = PANEL_W - MAP_INNER_PAD*2;
  const my = MAP_INNER_PAD + 22, mh = PANEL_H - MAP_INNER_PAD*2 - 22;
  return [panelX + mx + xn * mw, PANEL_Y + my + yn * mh];
}

function bowed(x1,y1,x2,y2, bowFrac=0.14) {
  const dx=x2-x1, dy=y2-y1, len=Math.hypot(dx,dy)||1;
  const cx=(x1+x2)/2 - (dy/len)*bowFrac*len;
  const cy=(y1+y2)/2 + (dx/len)*bowFrac*len;
  return { cx, cy, path: `M${x1},${y1} Q${cx},${cy} ${x2},${y2}` };
}

function pointOnQuad(x1,y1,cx,cy,x2,y2,t) {
  return [
    (1-t)*(1-t)*x1 + 2*(1-t)*t*cx + t*t*x2,
    (1-t)*(1-t)*y1 + 2*(1-t)*t*cy + t*t*y2,
  ];
}

// ═══════════════════════════════════════════════════════════════
//  DRAW ONE PANEL
// ═══════════════════════════════════════════════════════════════
function drawPanel(g, era, panelX, panelLabel, labelColor) {
  const countries = era.countries;
  const edges     = era.edges;
  const blocs     = era.blocs;

  const posMap = {};
  for (const c of countries) {
    posMap[c.id] = toPanel(c.x_adj, c.y_adj, panelX);
  }

  // Panel background
  g.append("rect").attr("x", panelX).attr("y", PANEL_Y)
   .attr("width", PANEL_W).attr("height", PANEL_H)
   .attr("fill", "#dde8f3").attr("rx", 2);

  // Ocean fill (full panel)
  g.append("rect").attr("x", panelX+1).attr("y", PANEL_Y+1)
   .attr("width", PANEL_W-2).attr("height", PANEL_H-2)
   .attr("fill", "#d4e4f0").attr("rx", 2);

  // Land outline
  const mxOff = panelX + MAP_INNER_PAD, mw = PANEL_W - MAP_INNER_PAD*2;
  const myOff = PANEL_Y + MAP_INNER_PAD + 22, mh = PANEL_H - MAP_INNER_PAD*2 - 22;

  let landPath = "";
  let started = false;
  for (const pt of EUROPE_OUTLINE) {
    if (pt === null) { started = false; continue; }
    const sx = mxOff + pt[0] * mw, sy = myOff + pt[1] * mh;
    landPath += started ? `L${sx},${sy}` : `M${sx},${sy}`;
    started = true;
  }
  g.append("path").attr("d", landPath + "Z")
   .attr("fill","#e8eed4").attr("stroke","#c4d0b4").attr("stroke-width",0.7);

  // Bloc convex hulls
  const blocMembers = {};
  for (const c of countries) {
    if (!blocMembers[c.bloc]) blocMembers[c.bloc] = [];
    if (posMap[c.id]) blocMembers[c.bloc].push(posMap[c.id]);
  }
  const blocColorMap = {};
  for (const b of blocs) blocColorMap[b.name] = b.color;

  for (const [bloc, pts] of Object.entries(blocMembers)) {
    if (pts.length < 2) continue;
    const color = blocColorMap[bloc] || "#888";
    const hex = color.replace("#","");
    const [rr,gg,bb] = [parseInt(hex.slice(0,2),16),parseInt(hex.slice(2,4),16),parseInt(hex.slice(4,6),16)];
    let hullPath;
    if (pts.length >= 3) {
      const hull = d3.polygonHull(pts);
      if (hull) hullPath = "M" + hull.map(p=>p.join(",")).join("L") + "Z";
    }
    if (!hullPath && pts.length === 2) {
      const [[x1,y1],[x2,y2]] = pts;
      const r = Math.hypot(x2-x1,y2-y1)/2 + 20;
      const mx=(x1+x2)/2, my=(y1+y2)/2;
      hullPath = `M${mx-r},${my} A${r},${r} 0 1 0 ${mx+r},${my} A${r},${r} 0 1 0 ${mx-r},${my}`;
    }
    if (hullPath) {
      g.append("path").attr("d", hullPath)
       .attr("fill",`rgba(${rr},${gg},${bb},0.13)`)
       .attr("stroke",`rgba(${rr},${gg},${bb},0.45)`)
       .attr("stroke-width",1.0).attr("stroke-dasharray","4,2");
    }
  }

  // Edges
  const edgeG = g.append("g").attr("class","edges");
  const tt = document.getElementById("tooltip");

  for (const e of edges) {
    const pa = posMap[e.a], pb = posMap[e.b];
    if (!pa || !pb) continue;
    const st = EDGE_STYLE[e.type] || EDGE_STYLE.stable_alliance;
    const { cx, cy, path } = bowed(pa[0],pa[1],pb[0],pb[1]);

    edgeG.append("path").attr("d", path)
      .attr("fill","none")
      .attr("stroke", st.color)
      .attr("stroke-width", st.width)
      .attr("stroke-dasharray", st.dash)
      .attr("stroke-linecap","round")
      .attr("opacity", 0.82)
      .on("mouseover", (ev) => {
        const nv = e.e2_mutual || e.e1_mutual;
        tt.style.display = "block";
        tt.innerHTML = `<b>${e.a} ↔ ${e.b}</b><br>${e.type.replace(/_/g," ")}<br>NVS ${nv.toFixed(2)} / 12<br>${e.co_years || "?"} co-eligible years`;
      })
      .on("mousemove", (ev) => {
        tt.style.left = (ev.clientX+14)+"px"; tt.style.top = (ev.clientY-6)+"px";
      })
      .on("mouseout", () => { tt.style.display="none"; });

    // Arrowhead circle for one_sided
    if (e.type === "one_sided") {
      const recv = posMap[e.receiver];
      if (recv) {
        const [px,py] = pointOnQuad(pa[0],pa[1],cx,cy,pb[0],pb[1], e.receiver===e.b ? 0.82 : 0.18);
        edgeG.append("circle").attr("cx",px).attr("cy",py).attr("r",3.5)
          .attr("fill", st.color).attr("opacity",0.90);
      }
    }
  }

  // Country nodes
  const nodeG = g.append("g").attr("class","nodes");
  const maxYrs = Math.max(...countries.map(c=>c.participation_years),1);
  const maxNVS = Math.max(...countries.map(c=>c.nvs_received),1);

  // Decide which countries get labels
  const labelled = new Set(
    countries
      .slice().sort((a,b)=>(b.nvs_received||0)-(a.nvs_received||0))
      .slice(0, 18).map(c=>c.id)
  );

  for (const c of countries) {
    const [cx2,cy2] = posMap[c.id] || [0,0];
    const r = 4 + c.size_norm * 7;
    const grp = nodeG.append("g").attr("class","country-node")
      .style("cursor","pointer");

    grp.append("circle").attr("cx",cx2).attr("cy",cy2).attr("r",r)
      .attr("fill", c.color || "#888")
      .attr("stroke","rgba(245,240,228,0.9)").attr("stroke-width",1)
      .attr("opacity",0.88);

    if (labelled.has(c.id)) {
      const lbl = c.label.length > 12 ? c.label.slice(0,10)+"." : c.label;
      nodeG.append("text").attr("x",cx2).attr("y",cy2-r-2)
        .attr("text-anchor","middle").attr("font-size","8")
        .attr("font-family","IBM Plex Mono, 'Courier New', monospace")
        .attr("fill",INK).attr("opacity",0.78)
        .text(lbl.includes(" ") ? lbl.split(" ").map((w,i)=>i===0?w[0]+".":w).join(" ") : lbl);
    }

    grp.on("mouseover",(ev)=>{
      tt.style.display="block";
      tt.innerHTML=`<b>${c.label}</b><br>Bloc: ${c.bloc}<br>Years: ${c.participation_years}<br>NVS received: ${(c.nvs_received||0).toFixed(1)}`;
    })
    .on("mousemove",(ev)=>{tt.style.left=(ev.clientX+14)+"px";tt.style.top=(ev.clientY-6)+"px";})
    .on("mouseout",()=>{tt.style.display="none";});
  }

  // Panel header
  g.append("rect").attr("x",panelX).attr("y",PANEL_Y)
   .attr("width",PANEL_W).attr("height",20)
   .attr("fill",labelColor).attr("opacity",0.9).attr("rx",2);
  g.append("text").attr("x",panelX+PANEL_W/2).attr("y",PANEL_Y+14)
   .attr("text-anchor","middle")
   .attr("font-size","11").attr("font-family","'Georgia', serif")
   .attr("font-weight","600").attr("fill","white").attr("letter-spacing","0.5")
   .text(panelLabel);

  // Bloc legend inside panel (bottom-left)
  const bly = PANEL_Y + PANEL_H - 12 - blocs.length * 16;
  blocs.slice(0,8).forEach((b,i) => {
    const ly = bly + i * 16;
    g.append("rect").attr("x",panelX+MAP_INNER_PAD).attr("y",ly-6)
     .attr("width",9).attr("height",9).attr("rx",1).attr("fill",b.color);
    g.append("text").attr("x",panelX+MAP_INNER_PAD+13).attr("y",ly+1)
     .attr("font-size","8.5").attr("font-family","IBM Plex Mono, monospace")
     .attr("fill",INK).attr("opacity",0.75)
     .text(`${b.name} (${b.n}) — ★ ${b.champion}`);
  });
}

// ═══════════════════════════════════════════════════════════════
//  MAIN RENDER
// ═══════════════════════════════════════════════════════════════
function render() {
  const svg = d3.select("#poster-svg")
    .attr("width", W).attr("height", H)
    .attr("viewBox", `0 0 ${W} ${H}`)
    .attr("xmlns","http://www.w3.org/2000/svg")
    .attr("font-family", "'Georgia', Georgia, serif");

  document.getElementById("poster-wrap").style.maxWidth = W + "px";

  // ── Paper background
  svg.append("rect").attr("width",W).attr("height",H).attr("fill",PAPER);

  // ── Title strip
  svg.append("rect").attr("width",W).attr("height",TITLE_H).attr("fill","#1a1408");
  svg.append("text").attr("x",W/2).attr("y",40)
    .attr("text-anchor","middle").attr("font-size","22")
    .attr("font-family","'Georgia', serif").attr("font-weight","500")
    .attr("fill","#f5f0e4").attr("letter-spacing","-0.3")
    .text("Eurovision Voting Communities · How Blocs Formed, Shifted and Dissolved");
  svg.append("text").attr("x",W/2).attr("y",63)
    .attr("text-anchor","middle").attr("font-size","11")
    .attr("font-family","IBM Plex Mono, monospace").attr("fill","#9a8e78")
    .text("Geographically-constrained layout · Louvain community detection on mutual NVS affinity · 1975–2025");
  // RQ
  svg.append("line").attr("x1",PAD).attr("y1",76).attr("x2",W-PAD).attr("y2",76)
    .attr("stroke","#3a3020").attr("stroke-width",0.5);
  svg.append("text").attr("x",PAD+6).attr("y",92)
    .attr("font-size","10").attr("font-style","italic")
    .attr("font-family","'Georgia', serif").attr("fill","#6a8ab0")
    .text("RQ1: Which bilateral voting relationships remain structurally persistent across 50 years?  ·  RQ3: How do geopolitical shifts reshape the voting network?");

  const D = POSTER_DATA;
  const g = svg.append("g");

  // ── Two panels
  drawPanel(g, D.era1, P1_X, `ERA I · ${D.meta.era1_label} · N=${D.meta.era1_n} countries`, "#1a4e88");
  drawPanel(g, D.era2, P2_X, `ERA II · ${D.meta.era2_label} · N=${D.meta.era2_n} countries`, "#7a2810");

  // ── Divider between panels
  svg.append("rect").attr("x",PAD+PANEL_W+2).attr("y",PANEL_Y)
   .attr("width",PANEL_GAP-4).attr("height",PANEL_H).attr("fill",PAPER);
  svg.append("text").attr("x",PAD+PANEL_W+PANEL_GAP/2).attr("y",PANEL_Y+PANEL_H/2-18)
    .attr("text-anchor","middle").attr("writing-mode","tb")
    .attr("font-size","9").attr("font-family","IBM Plex Mono, monospace")
    .attr("fill","#9a8e78").attr("opacity",0.6)
    .text("Yugoslavia fractures · Eastern Europe enters · Baltic states join · Caucasus arrives ↓");

  // ── Legend strip
  const LY = PANEL_Y + PANEL_H + 10;
  svg.append("line").attr("x1",PAD).attr("y1",LY-2).attr("x2",W-PAD).attr("y2",LY-2)
    .attr("stroke","#d0c8b0").attr("stroke-width",0.5);
  svg.append("text").attr("x",PAD).attr("y",LY+14)
    .attr("font-size","9").attr("font-weight","500")
    .attr("font-family","IBM Plex Mono, monospace").attr("fill",INK).attr("opacity",0.6)
    .text("EDGE ENCODING");

  const edgeTypes = Object.entries(EDGE_STYLE);
  const colW = (W - PAD*2) / edgeTypes.length;
  edgeTypes.forEach(([type, st], i) => {
    const ex = PAD + i * colW;
    const ey = LY + 28;
    // Sample line
    svg.append("line").attr("x1",ex).attr("y1",ey).attr("x2",ex+40).attr("y2",ey)
      .attr("stroke",st.color).attr("stroke-width",st.width)
      .attr("stroke-dasharray",st.dash).attr("stroke-linecap","round");
    if (type==="one_sided") {
      svg.append("circle").attr("cx",ex+32).attr("cy",ey).attr("r",3.5).attr("fill",st.color);
    }
    svg.append("text").attr("x",ex+44).attr("y",ey+4)
      .attr("font-size","9").attr("font-family","IBM Plex Mono, monospace")
      .attr("fill",INK).attr("opacity",0.72)
      .text(st.label);
  });

  // Node size legend
  const nly = LY + 50;
  [4,7,11].forEach((r,i) => {
    const nx = PAD + 20 + i*40;
    svg.append("circle").attr("cx",nx).attr("cy",nly).attr("r",r)
      .attr("fill","none").attr("stroke","#888").attr("stroke-width",0.8);
  });
  svg.append("text").attr("x",PAD+150).attr("y",nly+4)
    .attr("font-size","9").attr("font-family","IBM Plex Mono, monospace")
    .attr("fill",INK).attr("opacity",0.65)
    .text("Node size = years participated  ·  Node colour = detected Louvain voting bloc");

  // ── Community cards
  const CY = LY + LEGEND_H;
  svg.append("line").attr("x1",PAD).attr("y1",CY-2).attr("x2",W-PAD).attr("y2",CY-2)
    .attr("stroke","#d0c8b0").attr("stroke-width",0.5);
  svg.append("text").attr("x",PAD).attr("y",CY+13)
    .attr("font-size","9").attr("font-weight","500")
    .attr("font-family","IBM Plex Mono, monospace").attr("fill",INK).attr("opacity",0.6)
    .text(`COMMUNITY BLOCS · ERA II (${D.era2.blocs.length} detected)`);

  const CARD_TOP = CY + 22;
  const cardW = (W - PAD*2 - (D.era2.blocs.length-1)*8) / Math.min(D.era2.blocs.length, 8);

  D.era2.blocs.slice(0,8).forEach((b, i) => {
    const cx = PAD + i * (cardW + 8);
    const cy2 = CARD_TOP;

    // Card background
    svg.append("rect").attr("x",cx).attr("y",cy2).attr("width",cardW).attr("height",CARDS_H-28)
      .attr("rx",3).attr("fill","#ede8d8").attr("stroke","#d0c8b0").attr("stroke-width",0.5);

    // Colour stripe top
    svg.append("rect").attr("x",cx).attr("y",cy2).attr("width",cardW).attr("height",6)
      .attr("rx",3).attr("fill",b.color);

    // Bloc name
    svg.append("text").attr("x",cx+8).attr("y",cy2+22)
      .attr("font-size","11").attr("font-weight","600")
      .attr("font-family","'Georgia', serif").attr("fill",b.color)
      .text(b.name);

    // Country count + cohesion
    svg.append("text").attr("x",cx+8).attr("y",cy2+36)
      .attr("font-size","8.5").attr("font-family","IBM Plex Mono, monospace")
      .attr("fill",INK).attr("opacity",0.65)
      .text(`${b.n} countries · cohesion ${b.cohesion.toFixed(2)}`);

    // Champion
    svg.append("text").attr("x",cx+8).attr("y",cy2+52)
      .attr("font-size","9").attr("font-family","IBM Plex Mono, monospace")
      .attr("fill",INK).attr("opacity",0.85)
      .text(`★ ${b.champion}`);

    // Top pair
    if (b.top_pair && b.top_pair.length === 2) {
      svg.append("text").attr("x",cx+8).attr("y",cy2+66)
        .attr("font-size","8.5").attr("font-family","IBM Plex Mono, monospace")
        .attr("fill",INK).attr("opacity",0.65)
        .text(`⟷ ${b.top_pair[0]} · ${b.top_pair[1]}`);
    }

    // Members (wrap if needed)
    const members = b.members.slice(0,8).join(", ") + (b.members.length>8?"…":"");
    svg.append("text").attr("x",cx+8).attr("y",cy2+82)
      .attr("font-size","7.5").attr("font-family","IBM Plex Mono, monospace")
      .attr("fill",INK).attr("opacity",0.55)
      .text(members);
  });

  // ── Footer
  const FY = CY + CARDS_H;
  svg.append("line").attr("x1",PAD).attr("y1",FY).attr("x2",W-PAD).attr("y2",FY)
    .attr("stroke","#d0c8b0").attr("stroke-width",0.5);
  const meta = D.meta;
  svg.append("text").attr("x",PAD).attr("y",FY+22)
    .attr("font-size","8.5").attr("font-family","IBM Plex Mono, monospace")
    .attr("fill",INK).attr("opacity",0.50)
    .text(`Method: ${meta.nvs_formula} · ${meta.layout_note} · Louvain community detection · Edge classification: ${meta.edge_categories.join(", ")}`);
  svg.append("text").attr("x",W-PAD).attr("y",FY+22)
    .attr("text-anchor","end")
    .attr("font-size","8.5").attr("font-family","IBM Plex Mono, monospace")
    .attr("fill",INK).attr("opacity",0.45)
    .text(`Graph Drawing Contest 2026 · Eurovision Dataset · Generated ${meta.generated}`);
}

// ═══════════════════════════════════════════════════════════════
//  EXPORT
// ═══════════════════════════════════════════════════════════════
function downloadSVG() {
  const svg = document.getElementById("poster-svg");
  const ser = new XMLSerializer().serializeToString(svg);
  const blob = new Blob(
    ['<?xml version="1.0" encoding="UTF-8"?>\\n', ser],
    { type:"image/svg+xml;charset=utf-8" }
  );
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "eurovision_poster_gd2026.svg";
  a.click();
}

function downloadPNG() {
  const svg = document.getElementById("poster-svg");
  const ser = new XMLSerializer().serializeToString(svg);
  const img = new Image();
  const canvas = document.createElement("canvas");
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d");
  img.onload = () => {
    ctx.drawImage(img, 0, 0);
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = "eurovision_poster_gd2026.png";
    a.click();
  };
  img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(ser)));
}

// ── Boot
if (typeof POSTER_DATA === "undefined") {
  document.body.innerHTML =
    '<div style="color:#f55;padding:40px;font-family:monospace;font-size:16px">' +
    'ERROR: poster_data.js not found.<br><br>' +
    'Make sure poster_data.js is in the same folder as poster.html,<br>' +
    'then open poster.html via a local server or directly in Chrome.<br><br>' +
    'Run first: <code>python poster_data_export.py</code>' +
    '</div>';
} else {
  render();
}
</script>
</body>
</html>
"""

html_out = HTML_TEMPLATE.replace(
    '<script src="poster_data.js"></script>',
    f'<script>\n{data_js}</script>'
)

with open("poster_standalone.html", "w", encoding="utf-8") as f:
    f.write(html_out)

kb = os.path.getsize("poster_standalone.html") // 1024
print(f"\n  Done! poster_standalone.html ({kb} KB)")
print(f"  Open it in Chrome — drag the file onto a Chrome window.")
print(f"  Or via server: python3 -m http.server 8080")
print(f"  then go to:    http://localhost:8080/poster_standalone.html")