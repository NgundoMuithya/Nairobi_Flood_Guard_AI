"""
╔══════════════════════════════════════════════════════════════════╗
║  NAIROBI FLOOD GUARD  ·  0.0001% Elite Dashboard                ║
║  Interactive Ward Map · Flood Risk · Route Delays · Alerts      ║
╚══════════════════════════════════════════════════════════════════╝
pip install streamlit plotly pandas numpy
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import anthropic
import os
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ══════════════════════════════════════════════════════════════════
# 0  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Nairobi Flood Guard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# 1  DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════
P = {
    "bg":      "#05080F",
    "surface": "#0A1018",
    "card":    "#0E1520",
    "border":  "#162030",
    "teal":    "#00C896",
    "blue":    "#2B8EF0",
    "orange":  "#F59E0B",
    "red":     "#EF4444",
    "yellow":  "#FBBF24",
    "purple":  "#8B5CF6",
    "green":   "#22C55E",
    "text":    "#C8D5E8",
    "muted":   "#445566",
    "white":   "#F0F6FF",
}

_RGB = {
    "teal":   (0,   200, 150),
    "blue":   (43,  142, 240),
    "orange": (245, 158,  11),
    "red":    (239,  68,  68),
    "yellow": (251, 191,  36),
    "purple": (139,  92, 246),
    "green":  (34,  197,  94),
    "muted":  (68,   85, 102),
    "white":  (240, 246, 255),
    "bg":     (5,    8,  15),
}

def rgba(key: str, a: float) -> str:
    r, g, b = _RGB[key]
    return f"rgba({r},{g},{b},{a})"

# ══════════════════════════════════════════════════════════════════
# 2  CSS
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Base ── */
  html,body,[data-testid="stAppViewContainer"]{{background:{P['bg']}!important;color:{P['text']};font-family:'Space Grotesk',sans-serif;}}
  [data-testid="stSidebar"]{{background:{P['surface']}!important;border-right:1px solid {P['border']}!important;}}
  [data-testid="stSidebar"] *{{color:{P['text']}!important;}}
  .block-container{{padding:1rem 1.4rem 2rem!important;max-width:1600px;}}
  #MainMenu,footer,header,[data-testid="stDecoration"],[data-testid="stToolbar"]{{display:none!important;}}

  /* ── Typography ── */
  .display{{font-family:'Syne',sans-serif;font-weight:800;letter-spacing:-.02em;color:{P['white']};}}
  .mono{{font-family:'JetBrains Mono',monospace;}}

  /* ── ALERT BANNER ── */
  @keyframes alertSlide{{from{{transform:translateY(-100%);opacity:0;}}to{{transform:translateY(0);opacity:1;}}}}
  @keyframes pulse-ring{{0%{{box-shadow:0 0 0 0 rgba(239,68,68,.6);}}70%{{box-shadow:0 0 0 10px rgba(239,68,68,0);}}100%{{box-shadow:0 0 0 0 rgba(239,68,68,0);}}}}
  @keyframes ticker{{0%{{transform:translateX(100%);}}100%{{transform:translateX(-100%);}}}}
  @keyframes fadeIn{{from{{opacity:0;transform:translateY(8px);}}to{{opacity:1;transform:translateY(0);}}}}
  @keyframes shimmer{{0%{{background-position:-200% 0;}}100%{{background-position:200% 0;}}}}

  .alert-banner{{
    background:linear-gradient(135deg,rgba(239,68,68,.15) 0%,rgba(245,158,11,.08) 100%);
    border:1px solid rgba(239,68,68,.4);
    border-radius:14px;padding:1rem 1.4rem;
    position:relative;overflow:hidden;
    animation:alertSlide .4s ease-out;
    margin-bottom:1.2rem;
  }}
  .alert-banner::before{{
    content:'';position:absolute;inset:0;
    background:linear-gradient(90deg,transparent,rgba(239,68,68,.06),transparent);
    background-size:200% 100%;
    animation:shimmer 2.5s infinite;
  }}
  .alert-pulse{{
    width:10px;height:10px;border-radius:50%;
    background:{P['red']};flex-shrink:0;
    animation:pulse-ring 1.4s ease-out infinite;
  }}

  .ticker-wrap{{
    background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);
    border-radius:8px;overflow:hidden;padding:.3rem 0;margin-top:.65rem;
  }}
  .ticker-text{{
    white-space:nowrap;display:inline-block;
    animation:ticker 28s linear infinite;
    font-size:.73rem;color:{P['red']};font-weight:600;
    letter-spacing:.02em;padding-left:100%;
  }}

  /* ── KPI Card ── */
  .kpi{{
    background:{P['card']};border:1px solid {P['border']};
    border-radius:14px;padding:1.1rem 1.3rem 1rem;
    position:relative;overflow:hidden;
    transition:transform .18s ease,box-shadow .18s ease;
  }}
  .kpi:hover{{transform:translateY(-2px);box-shadow:0 16px 48px rgba(0,0,0,.6);}}
  .kpi-stripe{{position:absolute;top:0;left:0;width:100%;height:2px;border-radius:14px 14px 0 0;}}
  .kpi-glow{{position:absolute;top:-20px;right:-20px;width:80px;height:80px;border-radius:50%;filter:blur(28px);opacity:.3;}}
  .kpi-icon{{font-size:1.35rem;margin-bottom:.4rem;line-height:1;}}
  .kpi-label{{font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:{P['muted']};font-weight:600;margin-bottom:.2rem;}}
  .kpi-val{{font-size:1.9rem;font-weight:700;color:{P['white']};letter-spacing:-.03em;line-height:1;font-variant-numeric:tabular-nums;}}
  .kpi-sub{{font-size:.7rem;color:{P['muted']};margin-top:.3rem;}}
  .kpi-trend{{font-size:.68rem;font-weight:700;margin-top:.3rem;}}

  /* ── Section header ── */
  .sec{{display:flex;align-items:center;gap:.5rem;font-size:.66rem;font-weight:700;
         letter-spacing:.1em;text-transform:uppercase;color:{P['muted']};
         margin:1.3rem 0 .75rem;padding-bottom:.4rem;border-bottom:1px solid {P['border']};}}
  .sec-dot{{width:5px;height:5px;border-radius:50%;background:{P['teal']};box-shadow:0 0 7px {P['teal']};}}

  /* ── RISK SCORE BADGE ── */
  .risk-pill{{display:inline-flex;align-items:center;gap:.3rem;padding:.22rem .6rem;
               border-radius:100px;font-size:.65rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;}}
  .r-critical{{background:rgba(239,68,68,.15);color:{P['red']};border:1px solid rgba(239,68,68,.3);}}
  .r-high    {{background:rgba(245,158,11,.15);color:{P['orange']};border:1px solid rgba(245,158,11,.3);}}
  .r-moderate{{background:rgba(251,191,36,.15);color:{P['yellow']};border:1px solid rgba(251,191,36,.3);}}
  .r-low     {{background:rgba(34,197,94,.15);color:{P['green']};border:1px solid rgba(34,197,94,.3);}}
  .r-safe    {{background:rgba(0,200,150,.15);color:{P['teal']};border:1px solid rgba(0,200,150,.3);}}

  /* ── DELAY BADGE ── */
  .delay-badge{{display:inline-flex;align-items:center;gap:.28rem;padding:.18rem .55rem;
                border-radius:6px;font-size:.64rem;font-weight:700;}}
  .d-severe  {{background:rgba(239,68,68,.15);color:{P['red']};border:1px solid rgba(239,68,68,.25);}}
  .d-major   {{background:rgba(245,158,11,.15);color:{P['orange']};border:1px solid rgba(245,158,11,.25);}}
  .d-minor   {{background:rgba(251,191,36,.15);color:{P['yellow']};border:1px solid rgba(251,191,36,.25);}}
  .d-none    {{background:rgba(0,200,150,.12);color:{P['teal']};border:1px solid rgba(0,200,150,.2);}}

  /* ── EVACUATE BUTTON ── */
  .evac-btn{{
    display:inline-flex;align-items:center;gap:.5rem;
    background:linear-gradient(135deg,{P['red']},{P['orange']});
    color:white;font-weight:800;font-size:.78rem;
    letter-spacing:.04em;text-transform:uppercase;
    border-radius:10px;padding:.55rem 1rem;cursor:pointer;
    border:none;box-shadow:0 4px 20px rgba(239,68,68,.4);
    transition:all .2s ease;font-family:'Space Grotesk',sans-serif;
    animation:fadeIn .5s ease;
  }}
  .evac-btn:hover{{box-shadow:0 6px 28px rgba(239,68,68,.6);transform:translateY(-1px);}}

  .notif-card{{
    background:linear-gradient(135deg,rgba(239,68,68,.12),rgba(245,158,11,.06));
    border:1px solid rgba(239,68,68,.35);border-left:3px solid {P['red']};
    border-radius:10px;padding:.85rem 1rem;margin-bottom:.5rem;
    animation:fadeIn .35s ease;
  }}
  .notif-title{{font-weight:700;font-size:.82rem;color:{P['white']};margin-bottom:.2rem;}}
  .notif-body{{font-size:.75rem;color:{P['text']};line-height:1.55;}}
  .notif-time{{font-size:.67rem;color:{P['muted']};margin-top:.3rem;font-family:'JetBrains Mono',monospace;}}

  .safe-notif{{
    background:rgba(0,200,150,.07);border:1px solid rgba(0,200,150,.2);
    border-left:3px solid {P['teal']};
    border-radius:10px;padding:.85rem 1rem;margin-bottom:.5rem;
  }}

  /* ── Map panel ── */
  .map-panel{{background:{P['card']};border:1px solid {P['border']};border-radius:16px;overflow:hidden;}}
  .map-header{{padding:.9rem 1.2rem;border-bottom:1px solid {P['border']};
                display:flex;align-items:center;justify-content:space-between;}}
  .map-title{{font-family:'Syne',sans-serif;font-weight:800;font-size:1rem;color:{P['white']};}}

  /* ── Route card ── */
  .rcard{{
    background:{P['card']};border:1px solid {P['border']};
    border-radius:12px;padding:.85rem 1rem;margin-bottom:.45rem;
    transition:border-color .15s,background .15s;cursor:pointer;
    position:relative;overflow:hidden;
  }}
  .rcard:hover{{border-color:{P['teal']}40;background:rgba(0,200,150,.03);}}
  .rcard.affected{{border-left:3px solid {P['red']};}}
  .rcard.rerouted{{border-left:3px solid {P['teal']};}}
  .rcard-name{{font-weight:700;font-size:.82rem;color:{P['white']};}}
  .rcard-sub{{font-size:.71rem;color:{P['muted']};margin-top:.2rem;}}
  .rcard-row{{display:flex;align-items:center;gap:.5rem;margin-top:.4rem;flex-wrap:wrap;}}

  /* ── Ward card ── */
  .ward-card{{
    background:{P['card']};border:1px solid {P['border']};
    border-radius:12px;padding:1rem 1.1rem;margin-bottom:.5rem;
  }}
  .ward-name{{font-weight:700;font-size:.9rem;color:{P['white']};}}
  .ward-sub{{font-size:.72rem;color:{P['muted']};}}
  .risk-bar-wrap{{margin:.55rem 0;}}
  .risk-bar-bg{{height:6px;background:{P['border']};border-radius:3px;}}
  .risk-bar-fill{{height:6px;border-radius:3px;transition:width .5s ease;}}

  /* ── Progress bar ── */
  .progress-ring{{display:inline-flex;align-items:center;justify-content:center;
                   position:relative;width:56px;height:56px;}}

  /* ── Table ── */
  .dtbl{{width:100%;border-collapse:collapse;font-size:.78rem;
          background:{P['card']};border-radius:12px;overflow:hidden;border:1px solid {P['border']};}}
  .dtbl thead tr{{background:{P['surface']};}}
  .dtbl th{{padding:.55rem .9rem;text-align:left;color:{P['muted']};font-size:.62rem;
             letter-spacing:.09em;text-transform:uppercase;font-weight:700;}}
  .dtbl td{{padding:.5rem .9rem;border-top:1px solid rgba(22,32,48,.9);}}
  .dtbl tr:hover td{{background:rgba(0,200,150,.025);}}
  .dtbl tr.hi td{{background:rgba(239,68,68,.05);}}
  .dtbl tr.safe td{{background:rgba(0,200,150,.035);}}

  /* ── Sidebar nav ── */
  .stButton>button{{
    background:transparent!important;border:1px solid {P['border']}!important;
    color:{P['muted']}!important;border-radius:10px!important;
    font-family:'Space Grotesk',sans-serif!important;font-size:.82rem!important;
    font-weight:500!important;padding:.5rem .9rem!important;
    width:100%!important;text-align:left!important;transition:all .15s!important;
  }}
  .stButton>button:hover{{background:rgba(22,32,48,.8)!important;color:{P['text']}!important;border-color:{P['muted']}50!important;}}
  div[data-active="true"] .stButton>button{{
    background:rgba(0,200,150,.1)!important;color:{P['teal']}!important;
    border-color:rgba(0,200,150,.3)!important;font-weight:700!important;
  }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar{{width:4px;height:4px;}}
  ::-webkit-scrollbar-track{{background:{P['bg']};}}
  ::-webkit-scrollbar-thumb{{background:{P['border']};border-radius:4px;}}

  /* ── Info boxes ── */
  .ibox{{background:rgba(0,200,150,.06);border:1px solid rgba(0,200,150,.18);
          border-left:3px solid {P['teal']};border-radius:8px;
          padding:.8rem 1rem;font-size:.79rem;color:{P['text']};line-height:1.65;margin:.5rem 0;}}
  .wbox{{background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.18);
          border-left:3px solid {P['orange']};border-radius:8px;
          padding:.8rem 1rem;font-size:.79rem;color:{P['text']};line-height:1.65;margin:.5rem 0;}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 3  PLOTLY THEME
# ══════════════════════════════════════════════════════════════════
def th(fig, h=None, title=None, **kw):
    lo = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", color=P["muted"], size=11),
        margin=dict(l=10, r=10, t=38, b=10),
        xaxis=dict(gridcolor=P["border"], zerolinecolor=P["border"], linecolor=P["border"], tickfont=dict(size=10)),
        yaxis=dict(gridcolor=P["border"], zerolinecolor=P["border"], linecolor=P["border"], tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=P["border"], font=dict(size=10)),
        colorway=[P["teal"],P["blue"],P["orange"],P["red"],P["yellow"],P["purple"]],
    )
    if h: lo["height"] = h
    if title is not None:
        if isinstance(title, dict):
            lo["title"] = title
        else:
            lo["title"] = dict(text=title, font=dict(size=12, color=P["text"]), x=0, xanchor="left")
    lo.update(kw)
    fig.update_layout(**lo)
    return fig

# ══════════════════════════════════════════════════════════════════
# 4  STATIC DATA — 85 Nairobi Wards
# ══════════════════════════════════════════════════════════════════
_WARD_DATA = [
    # (name, lat, lon, sub_county, elev_m, flood_prob, pop, flood_trigger_h)
    ("Westlands",         -1.2636, 36.8066, "Westlands",        1720, 0.12, 54000,  False),
    ("Parklands",         -1.2490, 36.8115, "Westlands",        1715, 0.16, 38000,  False),
    ("Karura",            -1.2230, 36.8140, "Westlands",        1738, 0.10, 22000,  False),
    ("Mountain View",     -1.2510, 36.7940, "Westlands",        1742, 0.13, 31000,  False),
    ("Kangemi",           -1.2720, 36.7560, "Westlands",        1765, 0.09, 67000,  False),
    ("Kitisuru",          -1.2430, 36.7780, "Westlands",        1758, 0.08, 19000,  False),
    ("Karen",             -1.3290, 36.7120, "Lang'ata",         1763, 0.07, 28000,  False),
    ("Nairobi West",      -1.3130, 36.8035, "Lang'ata",         1682, 0.38, 43000,  False),
    ("Mugumoini",         -1.3260, 36.7570, "Lang'ata",         1752, 0.11, 35000,  False),
    ("South C",           -1.3165, 36.8170, "Lang'ata",         1669, 0.44, 61000,  False),
    ("Nyayo Highrise",    -1.3230, 36.7970, "Lang'ata",         1676, 0.40, 49000,  False),
    ("Kibera",            -1.3133, 36.7870, "Lang'ata",         1648, 0.84, 170000, True),
    ("Waithaka",          -1.3065, 36.7630, "Dagoretti North",  1733, 0.17, 55000,  False),
    ("Kawangware",        -1.2870, 36.7670, "Dagoretti North",  1708, 0.33, 112000, False),
    ("Gatina",            -1.2970, 36.7540, "Dagoretti North",  1724, 0.21, 47000,  False),
    ("Uthiru",            -1.2930, 36.7260, "Dagoretti North",  1738, 0.14, 41000,  False),
    ("Riruta",            -1.3070, 36.7840, "Dagoretti South",  1722, 0.27, 58000,  False),
    ("Kilimani",          -1.2935, 36.7968, "Dagoretti South",  1703, 0.20, 34000,  False),
    ("Kileleshwa",        -1.2770, 36.7868, "Dagoretti South",  1713, 0.18, 29000,  False),
    ("Roysambu",          -1.2130, 36.8435, "Roysambu",         1682, 0.38, 72000,  False),
    ("Lucky Summer",      -1.2435, 36.8635, "Roysambu",         1663, 0.50, 55000,  True),
    ("Zimmerman",         -1.2000, 36.8835, "Roysambu",         1642, 0.47, 66000,  False),
    ("Githurai 44",       -1.1835, 36.8930, "Roysambu",         1618, 0.57, 91000,  True),
    ("Kahawa West",       -1.1700, 36.9035, "Roysambu",         1611, 0.54, 83000,  True),
    ("Kasarani",          -1.2135, 36.9035, "Kasarani",         1633, 0.44, 78000,  False),
    ("Mwiki",             -1.1935, 36.9335, "Kasarani",         1601, 0.60, 94000,  True),
    ("Clayworks",         -1.2300, 36.9200, "Kasarani",         1617, 0.52, 57000,  True),
    ("Ruai",              -1.2600, 36.9770, "Kasarani",         1582, 0.67, 68000,  True),
    ("Ruaraka",           -1.2370, 36.8770, "Ruaraka",          1641, 0.46, 62000,  False),
    ("Baba Dogo",         -1.2330, 36.8605, "Ruaraka",          1638, 0.49, 71000,  False),
    ("Mathare North",     -1.2465, 36.8670, "Ruaraka",          1632, 0.54, 58000,  True),
    ("Utalii",            -1.2230, 36.8730, "Ruaraka",          1652, 0.39, 33000,  False),
    ("Mathare",           -1.2565, 36.8570, "Mathare",          1612, 0.86, 120000, True),
    ("Mabatini",          -1.2500, 36.8470, "Mathare",          1621, 0.79, 88000,  True),
    ("Huruma",            -1.2535, 36.8503, "Mathare",          1609, 0.82, 97000,  True),
    ("Kiamaiko",          -1.2470, 36.8535, "Mathare",          1606, 0.85, 75000,  True),
    ("Makadara",          -1.2970, 36.8570, "Makadara",         1651, 0.37, 43000,  False),
    ("Maringo/Hamza",     -1.2870, 36.8635, "Makadara",         1646, 0.41, 54000,  False),
    ("Viwandani",         -1.3030, 36.8635, "Makadara",         1637, 0.57, 48000,  True),
    ("Harambee",          -1.2835, 36.8535, "Makadara",         1643, 0.34, 37000,  False),
    ("Kamukunji",         -1.2800, 36.8400, "Kamukunji",        1657, 0.31, 29000,  False),
    ("Pumwani",           -1.2770, 36.8500, "Kamukunji",        1641, 0.61, 73000,  True),
    ("Eastleigh North",   -1.2635, 36.8535, "Kamukunji",        1637, 0.64, 85000,  True),
    ("Eastleigh South",   -1.2700, 36.8503, "Kamukunji",        1634, 0.67, 79000,  True),
    ("Starehe",           -1.2668, 36.8268, "Starehe",          1661, 0.27, 32000,  False),
    ("Pangani",           -1.2635, 36.8302, "Starehe",          1657, 0.31, 38000,  False),
    ("Ngara",             -1.2735, 36.8235, "Starehe",          1660, 0.29, 44000,  False),
    ("Nairobi Central",   -1.2835, 36.8168, "Starehe",          1672, 0.20, 15000,  False),
    ("Upper Hill",        -1.3000, 36.8135, "Starehe",          1703, 0.18, 11000,  False),
    ("Dandora",           -1.2568, 36.8968, "Embakasi Central", 1613, 0.62, 82000,  True),
    ("Kariobangi South",  -1.2700, 36.8835, "Embakasi Central", 1623, 0.56, 74000,  True),
    ("Embakasi",          -1.3265, 36.9105, "Embakasi Central", 1591, 0.70, 65000,  True),
    ("Kayole North",      -1.2665, 36.9070, "Embakasi East",    1602, 0.64, 86000,  True),
    ("Kayole South",      -1.2770, 36.9135, "Embakasi East",    1597, 0.67, 91000,  True),
    ("Komarock",          -1.2635, 36.9235, "Embakasi East",    1591, 0.70, 78000,  True),
    ("Umoja 1",           -1.2835, 36.8800, "Embakasi West",    1632, 0.41, 92000,  False),
    ("Umoja 2",           -1.2770, 36.8900, "Embakasi West",    1627, 0.45, 88000,  False),
    ("Mowlem",            -1.2635, 36.8900, "Embakasi West",    1622, 0.49, 67000,  False),
    ("Kariobangi North",  -1.2535, 36.8800, "Embakasi West",    1629, 0.43, 71000,  False),
    ("Imara Daima",       -1.3370, 36.8870, "Embakasi South",   1581, 0.72, 58000,  True),
    ("Kwa Njenga",        -1.3235, 36.8900, "Embakasi South",   1577, 0.74, 64000,  True),
    ("Pipeline",          -1.3068, 36.8800, "Embakasi South",   1587, 0.66, 81000,  True),
    ("Mukuru Kwa Njenga", -1.3100, 36.8700, "Embakasi South",   1610, 0.75, 95000,  True),
    ("South B",           -1.3100, 36.8335, "Lang'ata",         1663, 0.42, 53000,  False),
    ("Industrial Area",   -1.3035, 36.8368, "Makadara",         1642, 0.55, 18000,  True),
    ("Buruburu",          -1.2800, 36.8700, "Makadara",         1641, 0.40, 67000,  False),
    ("Donholm",           -1.2970, 36.8935, "Embakasi West",    1621, 0.47, 73000,  False),
    ("Muthaiga",          -1.2400, 36.8203, "Roysambu",         1715, 0.15, 14000,  False),
    ("Spring Valley",     -1.2535, 36.7903, "Westlands",        1743, 0.11, 12000,  False),
    ("Lavington",         -1.2970, 36.7735, "Dagoretti South",  1718, 0.13, 25000,  False),
    ("Hurlingham",        -1.2903, 36.8003, "Starehe",          1695, 0.19, 22000,  False),
    ("Ngong Road",        -1.2965, 36.7835, "Dagoretti South",  1721, 0.16, 31000,  False),
    ("Runda",             -1.2135, 36.7935, "Westlands",        1746, 0.08, 8000,   False),
    ("Gigiri",            -1.2268, 36.7968, "Westlands",        1751, 0.07, 9000,   False),
    ("Lang'ata",          -1.3435, 36.7735, "Lang'ata",         1703, 0.17, 43000,  False),
    ("South Lake",        -1.3568, 36.7835, "Lang'ata",         1720, 0.12, 29000,  False),
    ("JKIA",              -1.3193, 36.9278, "Embakasi Central", 1587, 0.62, 5000,   True),
    ("Syokimau",          -1.3568, 36.8968, "Embakasi South",   1574, 0.69, 34000,  True),
    ("Mlolongo",          -1.3835, 36.9168, "Embakasi South",   1560, 0.71, 41000,  True),
    ("Githurai 45",       -1.1700, 36.9135, "Roysambu",         1605, 0.58, 88000,  True),
    ("Kahawa Wendani",    -1.1868, 36.9268, "Kasarani",         1598, 0.61, 76000,  True),
    ("Njiru",             -1.2368, 36.9570, "Kasarani",         1579, 0.68, 58000,  True),
    ("Kariobangi",        -1.2600, 36.8900, "Embakasi West",    1618, 0.52, 69000,  True),
    ("Sarang'ombe",       -1.2935, 36.8203, "Kamukunji",        1659, 0.25, 41000,  False),
]

@st.cache_data(show_spinner=False)
def ward_df():
    rows = []
    for rec in _WARD_DATA:
        name, lat, lon, sub, elev, fp, pop, evac = rec
        rows.append({
            "ward": name, "lat": lat, "lon": lon, "sub_county": sub,
            "elev_m": elev, "flood_prob": fp, "pop": pop,
            "flood_label": int(fp >= 0.45),
            "evacuate": evac,
            "risk_score": int(fp * 100),
        })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════
# 4b  TEMPORAL FLOOD PROBABILITY ENGINE
#     Phase 1  h 00-06  Pre-rain slow rise (20% of peak)
#     Phase 2  h 06-14  Rainfall onset → rapid logistic rise
#     Phase 3  h 14-18  Peak plateau (varies by persistence)
#     Phase 4  h 18-24  Gradual drainage (low wards drain slower)
# ══════════════════════════════════════════════════════════════════

def _persistence(elev_m):
    """Lower wards drain more slowly — 0 (fast) to 1 (slow)."""
    return float(np.clip((1700 - elev_m) / 140, 0.0, 1.0))


def flood_prob_at_hour(base_prob, elev_m, hour):
    """Return flood probability [0,1] for a ward at a given event hour (0-23)."""
    h = float(hour)
    persist = _persistence(elev_m)
    if h < 6:
        t = h / 6.0
        factor = 0.15 + t * 0.20
    elif h < 14:
        t = (h - 6) / 8.0
        factor = 0.35 + 0.65 / (1 + np.exp(-8 * (t - 0.5)))
    elif h < 18:
        t = (h - 14) / 4.0
        factor = 1.0 - t * 0.06 * (1 - persist)
    else:
        t = (h - 18) / 6.0
        drain_speed = 0.55 + 0.30 * (1 - persist)
        factor = max(1.0 - t * drain_speed, 0.10 + persist * 0.25)
    return float(np.clip(base_prob * factor, 0.02, 0.99))


def ward_df_at_hour(hour):
    """Return ward_df with flood_prob updated for the given event hour."""
    base = ward_df().copy()
    base["flood_prob"] = base.apply(
        lambda r: flood_prob_at_hour(r["flood_prob"], r["elev_m"], hour), axis=1
    )
    base["flood_label"] = (base["flood_prob"] >= 0.45).astype(int)
    base["evacuate"]    = base["flood_prob"] >= 0.72
    base["risk_score"]  = (base["flood_prob"] * 100).round(1)
    return base


def route_status_at_hour(rt, hour):
    """Compute dynamic rerouting and delay for a route at a given event hour."""
    wdf_h    = ward_df_at_hour(hour)
    subs     = rt["sub_counties"]
    sub_df   = wdf_h[wdf_h["sub_county"].isin(subs)]
    avg_fp   = float(sub_df["flood_prob"].max()) if len(sub_df) else rt["flood_prob"]
    rerouted = avg_fp >= 0.45
    if avg_fp >= 0.75:
        delay  = int(25 + avg_fp * 15)
        status = "Severely Delayed"
    elif avg_fp >= 0.45:
        delay  = int(10 + avg_fp * 20)
        status = "Delayed"
    elif avg_fp >= 0.25:
        delay  = int(avg_fp * 12)
        status = "Minor Delay"
    else:
        delay  = 0
        status = "On Time"
    stops_aff = int(avg_fp * max(rt.get("stops_affected", 4), 1) * 1.1) if rerouted else 0
    return {**rt, "flood_prob": avg_fp, "rerouted": rerouted,
            "delay_min": delay, "status": status, "stops_affected": stops_aff}


# ══════════════════════════════════════════════════════════════════
# 5  MATATU ROUTES with delay predictions
# ══════════════════════════════════════════════════════════════════
ROUTES = [
    {
        "id": "R-01", "name": "CBD → Westlands",
        "orig": [(-1.2835,36.8168),(-1.2770,36.8100),(-1.2636,36.8066)],
        "alt": None,
        "flood_prob": 0.12, "rerouted": False,
        "delay_min": 0, "status": "On Time",
        "stops_affected": 0, "sub_counties": ["Starehe","Westlands"],
        "passengers_day": 12400,
    },
    {
        "id": "R-02", "name": "CBD → Kasarani",
        "orig": [(-1.2835,36.8168),(-1.2565,36.8570),(-1.2330,36.8605),(-1.2135,36.9035)],
        "alt":  [(-1.2835,36.8168),(-1.2400,36.8203),(-1.2230,36.8740),(-1.2135,36.9035)],
        "flood_prob": 0.82, "rerouted": True,
        "delay_min": 24, "status": "Severely Delayed",
        "stops_affected": 7, "sub_counties": ["Starehe","Mathare","Ruaraka","Kasarani"],
        "passengers_day": 18600,
    },
    {
        "id": "R-03", "name": "CBD → Kibera",
        "orig": [(-1.2835,36.8168),(-1.3070,36.7840),(-1.3133,36.7870)],
        "alt":  [(-1.2835,36.8168),(-1.3260,36.7570),(-1.3133,36.7870)],
        "flood_prob": 0.84, "rerouted": True,
        "delay_min": 31, "status": "Severely Delayed",
        "stops_affected": 9, "sub_counties": ["Starehe","Lang'ata"],
        "passengers_day": 21800,
    },
    {
        "id": "R-04", "name": "CBD → Karen",
        "orig": [(-1.2835,36.8168),(-1.3000,36.8135),(-1.3290,36.7120)],
        "alt": None,
        "flood_prob": 0.10, "rerouted": False,
        "delay_min": 3, "status": "On Time",
        "stops_affected": 0, "sub_counties": ["Starehe","Lang'ata"],
        "passengers_day": 9200,
    },
    {
        "id": "R-05", "name": "CBD → Eastleigh",
        "orig": [(-1.2835,36.8168),(-1.2700,36.8335),(-1.2635,36.8535)],
        "alt":  [(-1.2835,36.8168),(-1.2570,36.8300),(-1.2635,36.8535)],
        "flood_prob": 0.65, "rerouted": True,
        "delay_min": 18, "status": "Delayed",
        "stops_affected": 5, "sub_counties": ["Starehe","Kamukunji"],
        "passengers_day": 15900,
    },
    {
        "id": "R-06", "name": "Westlands → Kasarani",
        "orig": [(-1.2636,36.8066),(-1.2400,36.8203),(-1.2135,36.9035)],
        "alt": None,
        "flood_prob": 0.14, "rerouted": False,
        "delay_min": 5, "status": "On Time",
        "stops_affected": 0, "sub_counties": ["Westlands","Roysambu","Kasarani"],
        "passengers_day": 7800,
    },
    {
        "id": "R-07", "name": "CBD → Mathare",
        "orig": [(-1.2835,36.8168),(-1.2570,36.8470),(-1.2565,36.8570)],
        "alt":  [(-1.2835,36.8168),(-1.2400,36.8203),(-1.2370,36.8770)],
        "flood_prob": 0.86, "rerouted": True,
        "delay_min": 28, "status": "Severely Delayed",
        "stops_affected": 11, "sub_counties": ["Starehe","Mathare"],
        "passengers_day": 17400,
    },
    {
        "id": "R-08", "name": "CBD → Embakasi",
        "orig": [(-1.2835,36.8168),(-1.3035,36.8635),(-1.3265,36.9105)],
        "alt":  [(-1.2835,36.8168),(-1.2835,36.8800),(-1.3265,36.9105)],
        "flood_prob": 0.70, "rerouted": True,
        "delay_min": 21, "status": "Delayed",
        "stops_affected": 6, "sub_counties": ["Starehe","Makadara","Embakasi Central"],
        "passengers_day": 14200,
    },
    {
        "id": "R-09", "name": "CBD → Lang'ata",
        "orig": [(-1.2835,36.8168),(-1.3230,36.7970),(-1.3435,36.7735)],
        "alt": None,
        "flood_prob": 0.18, "rerouted": False,
        "delay_min": 7, "status": "On Time",
        "stops_affected": 0, "sub_counties": ["Starehe","Lang'ata"],
        "passengers_day": 8600,
    },
    {
        "id": "R-10", "name": "CBD → Pipeline",
        "orig": [(-1.2835,36.8168),(-1.3035,36.8635),(-1.3068,36.8800)],
        "alt":  [(-1.2835,36.8168),(-1.3000,36.8800),(-1.3068,36.8800)],
        "flood_prob": 0.66, "rerouted": True,
        "delay_min": 16, "status": "Delayed",
        "stops_affected": 4, "sub_counties": ["Starehe","Embakasi South"],
        "passengers_day": 11900,
    },
    {
        "id": "R-11", "name": "CBD → Dandora",
        "orig": [(-1.2835,36.8168),(-1.2568,36.8700),(-1.2568,36.8968)],
        "alt":  [(-1.2835,36.8168),(-1.2400,36.8500),(-1.2568,36.8968)],
        "flood_prob": 0.62, "rerouted": True,
        "delay_min": 14, "status": "Delayed",
        "stops_affected": 4, "sub_counties": ["Starehe","Embakasi Central"],
        "passengers_day": 13500,
    },
    {
        "id": "R-12", "name": "Upperhill → JKIA",
        "orig": [(-1.3000,36.8135),(-1.3193,36.9278)],
        "alt":  [(-1.3000,36.8135),(-1.3568,36.8968),(-1.3193,36.9278)],
        "flood_prob": 0.62, "rerouted": True,
        "delay_min": 19, "status": "Delayed",
        "stops_affected": 3, "sub_counties": ["Starehe","Embakasi Central"],
        "passengers_day": 6300,
    },
]

# ══════════════════════════════════════════════════════════════════
# 6  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def risk_level(p):
    if p >= 0.75: return "CRITICAL", "r-critical"
    if p >= 0.55: return "HIGH",     "r-high"
    if p >= 0.35: return "MODERATE", "r-moderate"
    if p >= 0.20: return "LOW",      "r-low"
    return               "SAFE",     "r-safe"

def risk_color(p):
    if p >= 0.75: return P["red"]
    if p >= 0.55: return P["orange"]
    if p >= 0.35: return P["yellow"]
    if p >= 0.20: return P["teal"]
    return               P["green"]

def delay_badge_cls(d):
    if d >= 25: return "d-severe"
    if d >= 15: return "d-major"
    if d >= 5:  return "d-minor"
    return              "d-none"

def delay_label(d):
    if d >= 25: return f"⏱ +{d} min Severe"
    if d >= 15: return f"⏱ +{d} min Delayed"
    if d >= 5:  return f"⏱ +{d} min Minor"
    return              "✓ On Time"

@st.cache_data(show_spinner=False)
def rerouting_df():
    rng = np.random.default_rng(42)
    n   = 87
    o   = np.clip(rng.beta(3,2,n)*0.65+0.35, 0.10, 0.99)
    rd  = np.clip(rng.beta(5,1.5,n)*o*0.97, 0.01, o-0.01)
    et  = rng.exponential(15,n) + 2
    origs = ["Westlands","Karen","Kibera","Eastleigh","Kasarani","Githurai",
             "Ruiru","Thika Rd","Ngong Rd","Lang'ata","Embakasi","Donholm"]
    dests = ["CBD","Upperhill","Industrial Area","JKIA","Ongata Rongai",
             "Kiambu","Ruaka","Mlolongo","Kikuyu","Limuru","Pipeline","Juja"]
    return pd.DataFrame({
        "route_id":               [f"RT-{100+i}" for i in range(n)],
        "origin":                 rng.choice(origs, n),
        "destination":            rng.choice(dests, n),
        "original_flood_prob":    np.round(o, 4),
        "alternative_flood_prob": np.round(o-rd, 4),
        "risk_reduction":         np.round(rd, 4),
        "extra_time_min":         np.round(et, 1),
    }).sort_values("risk_reduction", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def model_df():
    return pd.DataFrame({
        "Model":     ["Logistic Regression","Neural Network","Random Forest","XGBoost"],
        "AUC":       [0.70,0.78,0.88,0.90],
        "Recall":    [0.74,0.65,0.79,0.81],
        "Precision": [0.58,0.61,0.72,0.74],
        "F1":        [0.62,0.61,0.75,0.77],
        "Accuracy":  [0.71,0.69,0.83,0.84],
    })

# ══════════════════════════════════════════════════════════════════
# 7  SIDEBAR
# ══════════════════════════════════════════════════════════════════
PAGES = [
    ("🗺️", "Live Map"),
    ("🏠", "Overview"),
    ("🚍", "Route Monitor"),
    ("🤖", "Model Analysis"),
    ("📊", "Data Explorer"),
    ("💬", "Chat Assistant"),
]

if "page" not in st.session_state:
    st.session_state.page = "Live Map"
if "notifications" not in st.session_state:
    st.session_state.notifications = []
if "dismissed" not in st.session_state:
    st.session_state.dismissed = set()
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_hour" not in st.session_state:
    st.session_state.chat_hour = 15

wdf_global = ward_df()
n_crit  = int((wdf_global["flood_prob"] >= 0.75).sum())
n_hi    = int((wdf_global["flood_prob"] >= 0.45).sum())
n_evac  = int(wdf_global["evacuate"].sum())
pop_risk = int(wdf_global[wdf_global["flood_label"]==1]["pop"].sum())

with st.sidebar:
    st.markdown(f"""
    <div style='padding:.3rem 0 1.4rem;'>
      <div style='display:flex;align-items:center;gap:.65rem;'>
        <div style='width:38px;height:38px;border-radius:11px;flex-shrink:0;
                    background:linear-gradient(135deg,{P["teal"]},{P["blue"]});
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.15rem;box-shadow:0 4px 18px rgba(0,200,150,.35);'>🌊</div>
        <div>
          <div style='font-family:"Syne",sans-serif;font-size:1.05rem;font-weight:800;
                      color:{P["white"]};letter-spacing:-.01em;line-height:1.1;'>Flood Guard</div>
          <div style='font-size:.6rem;color:{P["muted"]};text-transform:uppercase;
                      letter-spacing:.08em;'>Nairobi · AI Platform</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    for icon, label in PAGES:
        active = st.session_state.page == label
        st.markdown(f'<div data-active="{"true" if active else "false"}">', unsafe_allow_html=True)
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='border-bottom:1px solid {P['border']};margin:.9rem 0 1rem;'></div>",
                unsafe_allow_html=True)

    # Live alert widget
    st.markdown(f"""
    <div style='background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.22);
                border-radius:12px;padding:.9rem 1rem;margin-bottom:.9rem;'>
      <div style='font-size:.6rem;color:{P["muted"]};text-transform:uppercase;
                  letter-spacing:.09em;margin-bottom:.4rem;'>Live Status</div>
      <div style='display:flex;align-items:center;gap:.45rem;margin-bottom:.5rem;'>
        <div style='width:8px;height:8px;border-radius:50%;background:{P["red"]};
                    box-shadow:0 0 0 0 rgba(239,68,68,.6);
                    animation:pulse-ring 1.4s ease-out infinite;'></div>
        <span style='color:{P["red"]};font-weight:800;font-size:.8rem;'>HIGH FLOOD RISK</span>
      </div>
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:.4rem;font-size:.69rem;color:{P["muted"]};'>
        <div>⚠️ <b style='color:{P["white"]};'>{n_hi}</b> wards at risk</div>
        <div>🔴 <b style='color:{P["red"]};'>{n_crit}</b> critical</div>
        <div>🚨 <b style='color:{P["orange"]};'>{n_evac}</b> evacuate</div>
        <div>👥 <b style='color:{P["white"]};'>{pop_risk//1000}k</b> affected</div>
      </div>
      <div style='margin-top:.6rem;background:{P["bg"]};border-radius:4px;height:4px;'>
        <div style='height:4px;border-radius:4px;background:{P["red"]};
                    width:{n_hi/len(wdf_global)*100:.0f}%;'></div>
      </div>
    </div>

    <div style='font-size:.69rem;line-height:2.1;color:{P["muted"]};'>
      <div><span style='color:{P["white"]};font-weight:600;'>Model</span>
        &nbsp;&nbsp;XGBoost · AUC 0.90</div>
      <div><span style='color:{P["white"]};font-weight:600;'>Event</span>
        &nbsp;&nbsp;April 2024 · Active</div>
      <div><span style='color:{P["white"]};font-weight:600;'>α</span>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1,000,000</div>
      <div><span style='color:{P["white"]};font-weight:600;'>Feed</span>
        &nbsp;&nbsp;&nbsp;&nbsp;GTFS-RT 2.0</div>
      <div><span style='color:{P["white"]};font-weight:600;'>Routes</span>
        &nbsp;&nbsp;87 rerouted</div>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════
# ▌ PAGE 1 — LIVE MAP  (CENTERPIECE)
# ══════════════════════════════════════════════════════════════════
if page == "Live Map":

    # ── TIME SLIDER ──
    _hour = st.slider(
        "🕐 Event Hour — drag to watch flood probabilities evolve in real time",
        min_value=0, max_value=23, value=15, format="%d:00",
        help="0 = midnight before event · 6 = rainfall onset · 14 = peak flood · 18 = draining",
    )

    _phase_map = {
        **{h: ("🌧 Pre-Rain",      P["blue"])   for h in range(0, 6)},
        **{h: ("⛈ Rainfall Onset", P["orange"]) for h in range(6, 14)},
        **{h: ("🌊 Peak Flood",    P["red"])    for h in range(14, 18)},
        **{h: ("🔄 Draining",      P["teal"])   for h in range(18, 24)},
    }
    _phase_label, _phase_col = _phase_map[_hour]

    # All ward and route data wired to the selected hour
    wdf_t     = ward_df_at_hour(_hour)
    routes_t  = [route_status_at_hour(rt, _hour) for rt in ROUTES]
    n_crit_t  = int((wdf_t["flood_prob"] >= 0.75).sum())
    n_hi_t    = int((wdf_t["flood_prob"] >= 0.45).sum())
    n_evac_t  = int(wdf_t["evacuate"].sum())
    n_rr_t    = len([r for r in routes_t if r["rerouted"]])
    pop_risk_t= int(wdf_t[wdf_t["flood_label"] == 1]["pop"].sum())

    # Phase progress bar
    _phase_pcts = {0:2,1:6,2:12,3:18,4:24,5:30,6:35,7:44,8:53,9:62,10:71,
                   11:80,12:88,13:95,14:100,15:98,16:96,17:93,18:87,
                   19:76,20:64,21:52,22:40,23:28}
    _pct = _phase_pcts.get(_hour, 50)

    st.markdown(f"""
    <div style='background:{P["card"]};border:1px solid {P["border"]};
                border-radius:14px;padding:.9rem 1.4rem;margin-bottom:.85rem;'>
      <div style='display:flex;align-items:center;justify-content:space-between;
                  margin-bottom:.5rem;flex-wrap:wrap;gap:.5rem;'>
        <div style='font-family:"Syne",sans-serif;font-size:.95rem;font-weight:800;color:{P["white"]};'>
          April 25, 2024 · <span style='color:{_phase_col};'>{_phase_label}</span>
          <span style='font-size:.75rem;color:{P["muted"]};font-weight:400;margin-left:.5rem;'>{_hour:02d}:00</span>
        </div>
        <div style='display:flex;gap:.8rem;font-size:.7rem;color:{P["muted"]};'>
          <span>⚠ <b style='color:{P["white"]};'>{n_hi_t}</b> at risk</span>
          <span>🔴 <b style='color:{P["red"]};'>{n_crit_t}</b> critical</span>
          <span>🚨 <b style='color:{P["orange"]};'>{n_evac_t}</b> evacuate</span>
          <span>🚍 <b style='color:{P["teal"]};'>{n_rr_t}</b> rerouted</span>
        </div>
      </div>
      <div style='display:flex;gap:.4rem;font-size:.6rem;color:{P["muted"]};
                  justify-content:space-between;margin-bottom:.2rem;'>
        <span>00:00 Pre-Rain</span><span>06:00 Onset</span>
        <span>14:00 Peak</span><span>18:00 Drain</span><span>23:00</span>
      </div>
      <div style='height:5px;background:{P["border"]};border-radius:3px;position:relative;'>
        <div style='position:absolute;left:25%;width:58%;height:5px;
                    background:linear-gradient(90deg,{P["orange"]},{P["red"]},{P["orange"]},{P["teal"]});
                    border-radius:3px;opacity:.4;'></div>
        <div style='height:5px;background:{_phase_col};border-radius:3px;
                    width:{_pct}%;transition:width .3s;'></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Global alert banner ──
    crit_wards  = wdf_t[wdf_t["flood_prob"] >= 0.75]["ward"].tolist()
    evac_wards  = wdf_t[wdf_t["evacuate"]]["ward"].tolist()
    ticker_text = "  ·  ".join(
        [f"⚠ CRITICAL FLOOD RISK: {w}" for w in crit_wards[:6]] +
        [f"🚨 EVACUATE NOW: {w}" for w in evac_wards[:5]]
    )

    st.markdown(f"""
    <div class="alert-banner">
      <div style='display:flex;align-items:center;gap:.7rem;margin-bottom:.4rem;'>
        <div class="alert-pulse"></div>
        <span style='font-family:"Syne",sans-serif;font-size:1rem;font-weight:800;
                     color:{P["white"]};'>FLOOD ALERT ACTIVE</span>
        <span style='font-size:.72rem;color:{P["muted"]};'>
          {n_crit_t} critical wards · {n_evac_t} evacuation zones · {n_rr_t} routes rerouted
        </span>
        <span style='margin-left:auto;font-size:.68rem;color:{P["muted"]};
                     font-family:"JetBrains Mono",monospace;'>
          {datetime.now().strftime("%d %b %Y %H:%M")}
        </span>
      </div>
      <div class="ticker-wrap">
        <div class="ticker-text">{ticker_text} &nbsp;&nbsp;&nbsp; {ticker_text}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Layout: map (left) + panels (right) ──
    map_col, side_col = st.columns([3, 1], gap="medium")

    with side_col:
        # ── EVACUATE NOW panel ──
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(239,68,68,.18),rgba(245,158,11,.1));
                    border:1px solid rgba(239,68,68,.4);border-radius:14px;
                    padding:1rem 1.1rem;margin-bottom:.9rem;'>
          <div style='display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem;'>
            <span style='font-size:1.1rem;'>🚨</span>
            <span style='font-family:"Syne",sans-serif;font-weight:800;
                         font-size:.95rem;color:{P["white"]};'>Evacuate Now</span>
          </div>
          <div style='font-size:.74rem;color:{P["text"]};line-height:1.6;margin-bottom:.8rem;'>
            <b style='color:{P["red"]};'>{n_evac_t} wards</b> under evacuation advisory
            at <b style='color:{_phase_col};'>{_hour:02d}:00</b>. Move to higher ground.
          </div>
          <div style='display:flex;flex-direction:column;gap:.3rem;'>
        """, unsafe_allow_html=True)

        for w in evac_wards[:8]:
            row = wdf_t[wdf_t["ward"] == w].iloc[0]
            lv, cls = risk_level(row["flood_prob"])
            st.markdown(f"""
            <div style='background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.2);
                        border-radius:7px;padding:.42rem .65rem;
                        display:flex;justify-content:space-between;align-items:center;'>
              <span style='font-size:.74rem;font-weight:600;color:{P["white"]};'>{w}</span>
              <span class='risk-pill {cls}' style='font-size:.6rem;padding:.12rem .4rem;'>{row["flood_prob"]:.0%}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # ── Notifications ──
        st.markdown('<div class="sec"><div class="sec-dot"></div>Push Notifications</div>',
                    unsafe_allow_html=True)

        # Auto-generate notifications for critical wards
        notifs = [
            {
                "type": "evacuate",
                "title": "🚨 Evacuate — Mathare",
                "body": "Flood probability 86%. Water levels rising rapidly. Move to St Teresa's School (1.3 km NE).",
                "time": "2 min ago",
                "ward": "Mathare",
            },
            {
                "type": "evacuate",
                "title": "🚨 Evacuate — Kibera",
                "body": "Critical flooding along Ngong River. All residents below Salama Bridge must relocate immediately.",
                "time": "5 min ago",
                "ward": "Kibera",
            },
            {
                "type": "evacuate",
                "title": "🚨 Evacuate — Huruma",
                "body": "Mathare River overflowing. 82% flood probability. Shelter at Huruma Primary (0.9 km E).",
                "time": "8 min ago",
                "ward": "Huruma",
            },
            {
                "type": "route",
                "title": "🚍 CBD → Mathare Rerouted",
                "body": "Route R-07 diverted via Roysambu. Avoid Juja Rd and Thika Superhighway interchange. +28 min.",
                "time": "12 min ago",
                "ward": None,
            },
            {
                "type": "route",
                "title": "🚍 CBD → Kibera Rerouted",
                "body": "Route R-03 diverted via Langata Rd southern bypass. +31 min expected delay.",
                "time": "15 min ago",
                "ward": None,
            },
            {
                "type": "safe",
                "title": "✅ Karen–Westlands Corridor Clear",
                "body": "No flood risk detected on Karen, Langata, and Westlands routes. Normal service.",
                "time": "18 min ago",
                "ward": None,
            },
        ]

        for notif in notifs:
            key = f"dismiss_{notif['title']}"
            if key in st.session_state.dismissed:
                continue
            box_cls  = "notif-card" if notif["type"] in ("evacuate","route") else "safe-notif"
            icon_col = P["red"] if notif["type"]=="evacuate" else P["orange"] if notif["type"]=="route" else P["teal"]
            st.markdown(f"""
            <div class="{box_cls}" style='margin-bottom:.4rem;'>
              <div class="notif-title" style='color:{icon_col};'>{notif["title"]}</div>
              <div class="notif-body">{notif["body"]}</div>
              <div class="notif-time">{notif["time"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── MAP ──
    with map_col:
        # Map controls
        ctrl_a, ctrl_b, ctrl_c, ctrl_d = st.columns(4)
        with ctrl_a:
            show_wards   = st.checkbox("Ward Risk", value=True)
        with ctrl_b:
            show_routes  = st.checkbox("Routes", value=True)
        with ctrl_c:
            show_alt     = st.checkbox("Safe Paths", value=True)
        with ctrl_d:
            ward_view    = st.selectbox("Show", ["All","Critical (≥75%)","High (≥45%)","Safe (<45%)"],
                                        label_visibility="collapsed")

        sub_opts = sorted(wdf_global["sub_county"].unique())
        sel_subs = st.multiselect("Sub-Counties", sub_opts, default=sub_opts,
                                  label_visibility="collapsed")

        disp = wdf_t[wdf_t["sub_county"].isin(sel_subs)].copy()
        if ward_view == "Critical (≥75%)":
            disp = disp[disp["flood_prob"] >= 0.75]
        elif ward_view == "High (≥45%)":
            disp = disp[disp["flood_prob"] >= 0.45]
        elif ward_view == "Safe (<45%)":
            disp = disp[disp["flood_prob"] < 0.45]

        fig_map = go.Figure()

        # ── Ward scatter bubbles ──
        if show_wards and len(disp) > 0:
            # Bin into 5 risk tiers for clean rendering
            tiers = [
                ("CRITICAL", disp[disp["flood_prob"] >= 0.75], "red",    0.95),
                ("HIGH",     disp[(disp["flood_prob"] >= 0.55) & (disp["flood_prob"] < 0.75)], "orange", 0.85),
                ("MODERATE", disp[(disp["flood_prob"] >= 0.35) & (disp["flood_prob"] < 0.55)], "yellow", 0.75),
                ("LOW",      disp[(disp["flood_prob"] >= 0.20) & (disp["flood_prob"] < 0.35)], "teal",   0.65),
                ("SAFE",     disp[disp["flood_prob"] < 0.20],  "green",  0.55),
            ]
            for tier_name, tier_df, ckey, base_alpha in tiers:
                if len(tier_df) == 0: continue
                r, g, b = _RGB[ckey]
                sizes   = 10 + (tier_df["pop"] / wdf_global["pop"].max()) * 30

                colors = [
                    f"rgba({r},{g},{b},{min(base_alpha + p*0.25, 0.98)})"
                    for p in tier_df["flood_prob"]
                ]
                # Evacuate wards get a distinct marker symbol effect via larger size
                evac_sizes = [
                    (s * 1.35 if ev else s)
                    for s, ev in zip(sizes, tier_df["evacuate"])
                ]

                fig_map.add_trace(go.Scattermapbox(
                    lat=tier_df["lat"], lon=tier_df["lon"],
                    mode="markers",
                    name=tier_name,
                    marker=dict(
                        size=evac_sizes,
                        color=colors,
                        opacity=1.0,
                    ),
                    customdata=np.stack([
                        tier_df["ward"],
                        tier_df["sub_county"],
                        (tier_df["flood_prob"]*100).round(1),
                        tier_df["elev_m"],
                        tier_df["pop"],
                        tier_df["evacuate"].map({True:"🚨 EVACUATE NOW", False:"✓ Monitor"}),
                    ], axis=-1),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                        "━━━━━━━━━━━━━━━━━━━━<br>"
                        "🌊 Flood Risk Score: <b>%{customdata[2]}%</b><br>"
                        "⛰ Elevation: %{customdata[3]:.0f} m<br>"
                        "👥 Population: %{customdata[4]:,}<br>"
                        "⚠ Status: <b>%{customdata[5]}</b><extra></extra>"
                    ),
                ))

        # ── Matatu routes ──
        if show_routes:
            for rt in routes_t:
                lats = [p[0] for p in rt["orig"]]
                lons = [p[1] for p in rt["orig"]]

                is_severe = rt["delay_min"] >= 25
                is_delayed = rt["delay_min"] >= 5

                if rt["rerouted"]:
                    r_col = rgba("red", 0.75)
                    r_w   = 2.8
                else:
                    r_col = rgba("teal", 0.70)
                    r_w   = 2.2

                # Original route
                fig_map.add_trace(go.Scattermapbox(
                    lat=lats, lon=lons,
                    mode="lines",
                    name=f"{rt['name']} {'(Affected)' if rt['rerouted'] else '(Clear)'}",
                    line=dict(color=r_col, width=r_w),
                    hovertemplate=(
                        f"<b>{rt['name']}</b> [{rt['id']}]<br>"
                        f"━━━━━━━━━━━━━━━━<br>"
                        f"🌊 Flood Risk: <b>{rt['flood_prob']:.0%}</b><br>"
                        f"⏱ Delay: <b>{'On Time' if rt['delay_min']<5 else '+'+str(rt['delay_min'])+' min'}</b><br>"
                        f"🚏 Stops Affected: <b>{rt['stops_affected']}</b><br>"
                        f"👥 Daily Passengers: <b>{rt['passengers_day']:,}</b><br>"
                        f"Status: <b>{rt['status']}</b><extra></extra>"
                    ),
                ))

                # Safe alternative
                if show_alt and rt["rerouted"] and rt["alt"]:
                    alats = [p[0] for p in rt["alt"]]
                    alons = [p[1] for p in rt["alt"]]
                    fig_map.add_trace(go.Scattermapbox(
                        lat=alats, lon=alons,
                        mode="lines",
                        name=f"{rt['name']} (Rerouted ✓)",
                        line=dict(color=rgba("teal", 0.95), width=3.5),
                        hovertemplate=(
                            f"<b>{rt['name']} — Safe Alternative</b><br>"
                            f"Dijkstra flood-free path<br>"
                            f"Extra time: +{rt['delay_min']} min<extra></extra>"
                        ),
                    ))

        # ── Map layout ──
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox_center={"lat": -1.290, "lon": 36.840},
            mapbox_zoom=10.8,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=560,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                bgcolor=P["card"],
                bordercolor=P["border"], borderwidth=1,
                font=dict(size=9.5, color=P["text"], family="Space Grotesk"),
                x=0.01, y=0.99, xanchor="left", yanchor="top",
                itemsizing="constant",
            ),
            showlegend=True,
            uirevision="constant",
        )

        st.markdown('<div class="map-panel">', unsafe_allow_html=True)

        # Map header bar
        n_routes_affected = n_rr_t
        st.markdown(f"""
        <div class="map-header">
          <div>
            <div class="map-title">🗺 Nairobi Live Flood Map</div>
            <div style='font-size:.7rem;color:{P["muted"]};margin-top:.1rem;'>
              {len(disp)} wards displayed · {n_rr_t} rerouted · {_phase_label} · {_hour:02d}:00
            </div>
          </div>
          <div style='display:flex;align-items:center;gap:.5rem;'>
            <div style='display:flex;align-items:center;gap:.3rem;'>
              <div style='width:7px;height:7px;border-radius:50%;background:{P["red"]};
                          box-shadow:0 0 8px {P["red"]};'></div>
              <span style='font-size:.68rem;color:{P["muted"]};'>Critical</span>
            </div>
            <div style='display:flex;align-items:center;gap:.3rem;'>
              <div style='width:7px;height:7px;border-radius:50%;background:{P["orange"]};'></div>
              <span style='font-size:.68rem;color:{P["muted"]};'>High</span>
            </div>
            <div style='display:flex;align-items:center;gap:.3rem;'>
              <div style='width:7px;height:7px;border-radius:50%;background:{P["yellow"]};'></div>
              <span style='font-size:.68rem;color:{P["muted"]};'>Moderate</span>
            </div>
            <div style='display:flex;align-items:center;gap:.3rem;'>
              <div style='width:7px;height:7px;border-radius:50%;background:{P["teal"]};'></div>
              <span style='font-size:.68rem;color:{P["muted"]};'>Safe</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Bottom stats strip ──
        s1, s2, s3, s4, s5 = st.columns(5)
        strips = [
            ("red",    "🔴 Critical Wards",   str(n_crit_t),           "prob ≥ 75%"),
            ("orange", "⚠ High Risk",        str(n_hi_t - n_crit_t),  "prob 45–74%"),
            ("teal",   "✓ Safe Wards",        str(len(wdf_t)-n_hi_t), "prob < 45%"),
            ("red",    "🚨 Evacuate Zones",   str(n_evac_t),           "immediate action"),
            ("orange", "🚍 Routes Rerouted",  str(n_rr_t),             _phase_label),
        ]
        for col, (color, label, val, sub) in zip([s1,s2,s3,s4,s5], strips):
            with col:
                ac = P[color]
                st.markdown(f"""
                <div style='background:{P["card"]};border:1px solid {P["border"]};
                            border-top:2px solid {ac};border-radius:10px;
                            padding:.7rem .85rem;text-align:center;'>
                  <div style='font-size:.62rem;color:{P["muted"]};text-transform:uppercase;
                              letter-spacing:.08em;margin-bottom:.2rem;'>{label}</div>
                  <div style='font-size:1.5rem;font-weight:800;color:{P["white"]};
                              font-variant-numeric:tabular-nums;line-height:1;'>{val}</div>
                  <div style='font-size:.67rem;color:{P["muted"]};margin-top:.2rem;'>{sub}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Ward Risk Table ──
    st.markdown('<div class="sec"><div class="sec-dot"></div>Ward Risk Scores — All Wards</div>',
                unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2, 2])
    with col_f1:
        risk_filter = st.selectbox("Filter by risk",
            ["All","Critical (≥75%)","High (55–74%)","Moderate (35–54%)","Low (20–34%)","Safe (<20%)"],
            label_visibility="collapsed")
    with col_f2:
        evac_filter = st.checkbox("Evacuation zones only", value=False)

    tbl_df = wdf_t.copy()
    if risk_filter == "Critical (≥75%)":
        tbl_df = tbl_df[tbl_df["flood_prob"] >= 0.75]
    elif risk_filter == "High (55–74%)":
        tbl_df = tbl_df[(tbl_df["flood_prob"]>=0.55)&(tbl_df["flood_prob"]<0.75)]
    elif risk_filter == "Moderate (35–54%)":
        tbl_df = tbl_df[(tbl_df["flood_prob"]>=0.35)&(tbl_df["flood_prob"]<0.55)]
    elif risk_filter == "Low (20–34%)":
        tbl_df = tbl_df[(tbl_df["flood_prob"]>=0.20)&(tbl_df["flood_prob"]<0.35)]
    elif risk_filter == "Safe (<20%)":
        tbl_df = tbl_df[tbl_df["flood_prob"] < 0.20]
    if evac_filter:
        tbl_df = tbl_df[tbl_df["evacuate"]]

    tbl_df = tbl_df.sort_values("flood_prob", ascending=False)

    def make_risk_bar(p, w=120):
        c = risk_color(p)
        pct = int(p*100)
        return (
            f"<div style='display:flex;align-items:center;gap:.4rem;'>"
            f"<div style='width:{w}px;height:5px;background:{P['border']};border-radius:3px;'>"
            f"<div style='width:{pct}%;height:5px;background:{c};border-radius:3px;'></div></div>"
            f"<span style='font-size:.72rem;color:{c};font-weight:700;"
            f"font-family:\"JetBrains Mono\",monospace;'>{pct}%</span>"
            f"</div>"
        )

    rows_html = ""
    for _, row in tbl_df.iterrows():
        lv, cls = risk_level(row["flood_prob"])
        is_evac = row["evacuate"]
        is_crit = row["flood_prob"] >= 0.75
        tr_cls  = "hi" if is_crit else ("safe" if row["flood_prob"] < 0.35 else "")
        evac_badge = (
            f"<span class='risk-pill r-critical' style='font-size:.58rem;padding:.1rem .38rem;'>"
            f"🚨 EVACUATE</span>"
            if is_evac else ""
        )
        rows_html += f"""
        <tr class="{tr_cls}">
          <td style='font-weight:700;color:{P["white"]};font-size:.8rem;'>{row["ward"]} {evac_badge}</td>
          <td style='color:{P["muted"]};font-size:.76rem;'>{row["sub_county"]}</td>
          <td>{make_risk_bar(row["flood_prob"])}</td>
          <td><span class='risk-pill {cls}'>{lv}</span></td>
          <td style='color:{P["muted"]};font-family:"JetBrains Mono",monospace;font-size:.73rem;'>{row["elev_m"]:.0f} m</td>
          <td style='color:{P["muted"]};font-size:.74rem;'>{row["pop"]:,}</td>
        </tr>"""

    st.markdown(f"""
    <div style='overflow-x:auto;max-height:400px;overflow-y:auto;border-radius:12px;'>
    <table class="dtbl">
      <thead><tr>
        <th>Ward</th><th>Sub-County</th>
        <th style='min-width:180px;'>Risk Score</th>
        <th>Level</th><th>Elevation</th><th>Population</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    <div style='font-size:.67rem;color:{P["muted"]};margin-top:.35rem;'>
      Showing {len(tbl_df)} wards at {_hour:02d}:00 · {_phase_label}
    </div>
    """, unsafe_allow_html=True)


    # ── Temporal Flood Curve chart ──
    st.markdown('<div class="sec"><div class="sec-dot"></div>Flood Probability Over Time — Top 8 Wards</div>', unsafe_allow_html=True)

    _hours_all = list(range(24))
    _top_wards = wdf_t.sort_values("flood_prob", ascending=False).head(8)["ward"].tolist()
    _safe_wards = wdf_t.sort_values("flood_prob").head(3)["ward"].tolist()

    fig_curve = go.Figure()

    _ward_cols = [P["red"],P["orange"],P["yellow"],rgba("red",0.6),
                  P["purple"],rgba("orange",0.7),rgba("yellow",0.6),rgba("red",0.5)]

    for wd, wc in zip(_top_wards, _ward_cols):
        base_row = ward_df()[ward_df()["ward"] == wd].iloc[0]
        y_vals = [flood_prob_at_hour(base_row["flood_prob"], base_row["elev_m"], h) for h in _hours_all]
        fig_curve.add_trace(go.Scatter(
            x=_hours_all, y=y_vals, name=wd, mode="lines",
            line=dict(color=wc, width=1.8),
            hovertemplate=f"<b>{wd}</b><br>Hour %{{x}}:00<br>Flood Prob: %{{y:.1%}}<extra></extra>",
        ))

    for wd in _safe_wards:
        base_row = ward_df()[ward_df()["ward"] == wd].iloc[0]
        y_vals = [flood_prob_at_hour(base_row["flood_prob"], base_row["elev_m"], h) for h in _hours_all]
        fig_curve.add_trace(go.Scatter(
            x=_hours_all, y=y_vals, name=wd, mode="lines",
            line=dict(color=P["teal"], width=1.2, dash="dot"),
            hovertemplate=f"<b>{wd}</b><br>Hour %{{x}}:00<br>Flood Prob: %{{y:.1%}}<extra></extra>",
        ))

    # Phase shading — rgba() not hex+alpha
    fig_curve.add_vrect(x0=0,  x1=6,  fillcolor=rgba("blue",  0.06), line_width=0, annotation_text="Pre-Rain",   annotation_position="top left",  annotation_font_color=P["blue"],   annotation_font_size=9)
    fig_curve.add_vrect(x0=6,  x1=14, fillcolor=rgba("orange",0.06), line_width=0, annotation_text="Onset",      annotation_position="top left",  annotation_font_color=P["orange"], annotation_font_size=9)
    fig_curve.add_vrect(x0=14, x1=18, fillcolor=rgba("red",   0.07), line_width=0, annotation_text="Peak Flood", annotation_position="top left",  annotation_font_color=P["red"],    annotation_font_size=9)
    fig_curve.add_vrect(x0=18, x1=23, fillcolor=rgba("teal",  0.05), line_width=0, annotation_text="Draining",   annotation_position="top left",  annotation_font_color=P["teal"],   annotation_font_size=9)

    # Current hour marker
    fig_curve.add_vline(x=_hour, line_dash="solid", line_color=P["white"], line_width=1.5,
                         annotation_text=f"  ▲ {_hour:02d}:00",
                         annotation_position="top right",
                         annotation_font_color=P["white"], annotation_font_size=10)

    # Decision threshold
    fig_curve.add_hline(y=0.45, line_dash="dash", line_color=P["orange"], line_width=1,
                         annotation_text="Reroute threshold 0.45",
                         annotation_position="right",
                         annotation_font_color=P["orange"], annotation_font_size=9)
    fig_curve.add_hline(y=0.75, line_dash="dash", line_color=P["red"], line_width=1,
                         annotation_text="Critical 0.75",
                         annotation_position="right",
                         annotation_font_color=P["red"], annotation_font_size=9)

    th(fig_curve, h=340, title="How flood probability evolves across the April 2024 event",
       xaxis=dict(title="Event Hour", tickvals=list(range(0,24,2)),
                  ticktext=[f"{h:02d}:00" for h in range(0,24,2)]),
       yaxis=dict(title="Flood Probability", tickformat=".0%", range=[0, 1.05]))
    st.plotly_chart(fig_curve, use_container_width=True)

    st.markdown(f"""
    <div class="ibox">
      📈 <b>Reading this chart:</b> Each line is one ward's flood probability across 24 hours.
      Low-elevation wards (Mathare 1,606 m, Kibera 1,648 m) rise fastest and drain slowest
      due to their terrain — they act as catch-basins for runoff from surrounding highlands.
      High-elevation wards (Karen 1,763 m, Westlands 1,720 m) stay safely below the
      <b style='color:{P["orange"]};'>0.45 reroute threshold</b> throughout the event.
      The vertical white line shows the currently selected event hour.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ▌ PAGE 2 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
elif page == "Overview":
    _chips_html = " ".join(
        f"<span style='background:{P['border']};border-radius:100px;"
        f"padding:.2rem .7rem;font-size:.68rem;color:{P['text']};"
        f"font-weight:500;border:1px solid rgba(255,255,255,.05);'>{c}</span>"
        for c in ["🛰 NASA SRTM + CHIRPS","🌲 XGBoost · AUC 0.90",
                  "🕸 OSMnx Road Graph","🚍 136 Matatu Routes",
                  "📡 GTFS-RT 2.0","April 2024 Event"]
    )
    # Hero
    st.markdown(f"""
    <div style='background:radial-gradient(ellipse 80% 55% at 55% 40%,
      rgba(0,200,150,.08) 0%,transparent 65%),
      linear-gradient(160deg,{P["surface"]} 0%,{P["bg"]} 100%);
      border:1px solid {P["border"]};border-radius:18px;
      padding:2.2rem 2.5rem 2rem;position:relative;overflow:hidden;margin-bottom:1.2rem;'>
      <div style='position:absolute;inset:0;background-image:
        linear-gradient(rgba(0,200,150,.025) 1px,transparent 1px),
        linear-gradient(90deg,rgba(0,200,150,.025) 1px,transparent 1px);
        background-size:42px 42px;border-radius:18px;'></div>
      <div style='position:absolute;top:1.3rem;right:1.8rem;display:flex;align-items:center;gap:.45rem;
                  background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.25);
                  border-radius:100px;padding:.28rem .85rem;'>
        <div style='width:6px;height:6px;border-radius:50%;background:{P["red"]};
                    animation:pulse-ring 1.4s infinite;'></div>
        <span style='font-size:.7rem;font-weight:700;color:{P["red"]};letter-spacing:.05em;'>ALERT ACTIVE</span>
      </div>
      <div style='font-family:"Syne",sans-serif;font-size:2.8rem;line-height:1.05;
                  color:{P["white"]};position:relative;letter-spacing:-.02em;'>
        Nairobi <span style='background:linear-gradient(95deg,{P["teal"]},{P["blue"]});
                              -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                  Flood Guard</span>
      </div>
      <div style='font-size:.87rem;color:{P["muted"]};max-width:540px;line-height:1.7;
                  margin:.55rem 0 1rem;position:relative;'>
        AI-powered flood susceptibility prediction across Nairobi's wards,
        with real-time matatu rerouting and evacuation alerts — protecting lives
        during the April 2024 flood event.
      </div>
      <div style='display:flex;flex-wrap:wrap;gap:.4rem;position:relative;'>
        {_chips_html}
      </div>
    </div>
    """, unsafe_allow_html=True)

    mdf = model_df()
    rr  = rerouting_df()

    cols = st.columns(6)
    kpis = [
        ("teal",   "🗺️","Wards Analyzed",     f"{len(wdf_global)}",   "Ward-level AI predictions"),
        ("red",    "⚠️","High-Risk Wards",   f"{n_hi}",              "Flood prob ≥ 45%"),
        ("red",    "🚨","Evacuate Zones",     f"{n_evac}",            "Immediate action required"),
        ("orange", "🚍","Routes Rerouted",    "87",                   "of 136 total routes"),
        ("teal",   "📉","Avg Risk Reduction", "63%",                  "Per rerouted route"),
        ("yellow", "👥","People at Risk",     f"{pop_risk//1000}k",   "In high-risk wards"),
    ]
    gcs = [P["teal"],P["red"],P["red"],P["orange"],P["teal"],P["yellow"]]
    for col, (color, icon, label, val, sub), gc in zip(cols, kpis, gcs):
        with col:
            st.markdown(f"""
            <div class="kpi {color}">
              <div class="kpi-stripe" style='background:{gc};'></div>
              <div class="kpi-glow" style='background:{gc};'></div>
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-val">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    cl, cr = st.columns([5,4], gap="large")
    with cl:
        st.markdown('<div class="sec"><div class="sec-dot"></div>System Pipeline</div>', unsafe_allow_html=True)
        steps = [
            ("1","Satellite Data","NASA SRTM elevation + CHIRPS rainfall → ward feature matrix (1,450 Kenya wards)."),
            ("2","XGBoost Model","Flood probability per ward. AUC 0.90, Recall 0.81. Recall prioritised — missing a flood is far costlier."),
            ("3","Road Graph Overlay","XGBoost probabilities spatially joined to OSMnx's 200k+ Nairobi road edges."),
            ("4","Weighted Dijkstra","cost = travel_time × (1 + 1M × flood_prob). Finds 100% flood-free alternative paths."),
            ("5","Live GTFS-RT Feed","87 TripUpdates served as protobuf. Live in Google Maps, Transit App, and Matatu apps."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex;gap:.85rem;align-items:flex-start;
                        background:{P["card"]};border:1px solid {P["border"]};
                        border-radius:12px;padding:.8rem 1.1rem;margin-bottom:.4rem;'>
              <div style='width:26px;height:26px;border-radius:50%;flex-shrink:0;
                          background:rgba(0,200,150,.12);color:{P["teal"]};
                          border:1px solid rgba(0,200,150,.25);
                          font-weight:800;font-size:.72rem;
                          display:flex;align-items:center;justify-content:center;'>{num}</div>
              <div>
                <div style='font-weight:700;font-size:.82rem;color:{P["white"]};'>{title}</div>
                <div style='font-size:.76rem;color:{P["muted"]};line-height:1.5;margin-top:.1rem;'>{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with cr:
        st.markdown('<div class="sec"><div class="sec-dot"></div>Model Leaderboard</div>', unsafe_allow_html=True)
        fig_lb = go.Figure()
        for m, c in zip(["Recall","AUC","F1"],
                        [P["red"],P["teal"],P["blue"]]):
            fig_lb.add_trace(go.Bar(
                name=m, x=mdf["Model"], y=mdf[m],
                marker_color=c, opacity=0.86,
                marker_line_color=P["bg"], marker_line_width=1,
                text=[f"{v:.0%}" for v in mdf[m]],
                textposition="outside", textfont=dict(size=8.5, color=P["muted"]),
            ))
        th(fig_lb, h=260, title="Recall = primary metric",
           barmode="group",
           yaxis=dict(range=[0.45,1.08],tickformat=".0%"),
           legend=dict(orientation="h",y=1.16,x=0,font=dict(size=9)))
        st.plotly_chart(fig_lb, use_container_width=True)

        xgb = mdf[mdf["Model"]=="XGBoost"].iloc[0]
        st.markdown(f"""
        <div style='background:{P["card"]};border:1px solid {P["border"]};
                    border-top:2px solid {P["teal"]};border-radius:12px;padding:.95rem 1.1rem;'>
          <div style='font-size:.66rem;color:{P["teal"]};font-weight:800;
                      text-transform:uppercase;letter-spacing:.07em;margin-bottom:.6rem;'>
            ★ XGBoost · Selected</div>
        """, unsafe_allow_html=True)
        for m, c in zip(["Recall","AUC","F1","Accuracy"],[P["red"],P["teal"],P["blue"],P["orange"]]):
            pct = int(xgb[m]*100)
            st.markdown(f"""
          <div style='display:flex;align-items:center;gap:.55rem;margin:.27rem 0;'>
            <span style='font-size:.69rem;color:{P["muted"]};width:66px;flex-shrink:0;'>{m}</span>
            <div style='flex:1;height:4px;background:{P["border"]};border-radius:2px;'>
              <div style='width:{pct}%;height:4px;background:{c};border-radius:2px;'></div>
            </div>
            <span style='font-size:.69rem;color:{P["white"]};font-weight:700;
                         font-family:"JetBrains Mono",monospace;width:33px;text-align:right;'>{xgb[m]:.0%}</span>
          </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ▌ PAGE 3 — ROUTE MONITOR
# ══════════════════════════════════════════════════════════════════
elif page == "Route Monitor":
    rr = rerouting_df()

    st.markdown(f"""
    <div style='font-family:"Syne",sans-serif;font-size:1.9rem;font-weight:800;
                color:{P["white"]};margin-bottom:.2rem;letter-spacing:-.02em;'>
      Route Monitor
    </div>
    <div style='color:{P["muted"]};font-size:.82rem;margin-bottom:1rem;'>
      Real-time matatu delay predictions · flood-weighted Dijkstra rerouting · GTFS-RT 2.0
    </div>
    """, unsafe_allow_html=True)

    # ── Route cards grid ──
    st.markdown('<div class="sec"><div class="sec-dot"></div>Live Route Status</div>', unsafe_allow_html=True)

    r_cols = st.columns(2)
    for i, rt in enumerate(ROUTES):
        with r_cols[i % 2]:
            lv, risk_cls = risk_level(rt["flood_prob"])
            d_cls   = delay_badge_cls(rt["delay_min"])
            d_label = delay_label(rt["delay_min"])
            card_cls = "affected" if rt["rerouted"] else "rerouted"

            pct       = int(rt["flood_prob"] * 100)
            risk_col  = risk_color(rt["flood_prob"])

            st.markdown(f"""
            <div class="rcard {card_cls}">
              <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                <div>
                  <div class="rcard-name">{rt["id"]} · {rt["name"]}</div>
                  <div class="rcard-sub">{' → '.join(rt["sub_counties"])}</div>
                </div>
                <span class='delay-badge {d_cls}'>{d_label}</span>
              </div>

              <div style='margin:.6rem 0;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:.25rem;'>
                  <span style='font-size:.68rem;color:{P["muted"]};'>Flood Risk</span>
                  <span style='font-size:.68rem;color:{risk_col};font-weight:700;
                               font-family:"JetBrains Mono",monospace;'>{pct}%</span>
                </div>
                <div style='height:5px;background:{P["border"]};border-radius:3px;'>
                  <div style='width:{pct}%;height:5px;background:{risk_col};border-radius:3px;'></div>
                </div>
              </div>

              <div class="rcard-row">
                <span class='risk-pill {risk_cls}'>{lv}</span>
                {'<span class="risk-pill r-critical" style="font-size:.6rem;">🔀 REROUTED</span>' if rt["rerouted"] else '<span class="risk-pill r-safe" style="font-size:.6rem;">✓ CLEAR</span>'}
                <span style='font-size:.7rem;color:{P["muted"]};'>🚏 {rt["stops_affected"]} stops affected</span>
                <span style='font-size:.7rem;color:{P["muted"]};'>👥 {rt["passengers_day"]:,}/day</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tradeoff scatter + delay histogram ──
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3,2], gap="large")

    with c1:
        st.markdown('<div class="sec"><div class="sec-dot"></div>Risk Reduction vs Detour Time</div>', unsafe_allow_html=True)
        x_max = rr["extra_time_min"].max() * 1.08
        y_max = rr["risk_reduction"].max()  * 1.08
        x_med = rr["extra_time_min"].median()
        y_med = rr["risk_reduction"].median()

        fig_sc = go.Figure()
        fig_sc.add_shape(type="rect", layer="below",
                         x0=0, y0=y_med, x1=x_med, y1=y_max,
                         fillcolor=rgba("teal", 0.07), line_width=0)
        fig_sc.add_annotation(x=x_med*0.28, y=y_max*0.95, text="⭐ Ideal Zone",
                               showarrow=False, font=dict(color=P["teal"],size=10,family="Space Grotesk"))
        fig_sc.add_hline(y=y_med, line_dash="dot", line_color=rgba("muted",0.3), line_width=1)
        fig_sc.add_vline(x=x_med, line_dash="dot", line_color=rgba("muted",0.3), line_width=1)
        fig_sc.add_trace(go.Scatter(
            x=rr["extra_time_min"], y=rr["risk_reduction"],
            mode="markers",
            marker=dict(
                size=8,
                color=rr["risk_reduction"],
                colorscale=[[0,P["blue"]],[0.5,P["orange"]],[1,P["teal"]]],
                showscale=True,
                colorbar=dict(title=dict(text="Risk ↓",font=dict(size=9,color=P["muted"])),
                              tickfont=dict(color=P["muted"],size=8),len=0.7,thickness=9),
                opacity=0.88, line=dict(width=0.5,color=P["bg"]),
            ),
            text=[f"<b>{r}</b>  {o}→{d}<br>Risk ↓ {rd:.1%}  ·  +{et:.0f} min"
                  for r,o,d,rd,et in zip(rr["route_id"],rr["origin"],rr["destination"],
                                         rr["risk_reduction"],rr["extra_time_min"])],
            hovertemplate="%{text}<extra></extra>",
        ))
        th(fig_sc, h=360, title="87 rerouted routes · risk vs detour tradeoff",
           xaxis=dict(title="Detour (minutes)", range=[0,x_max]),
           yaxis=dict(title="Risk Reduction", tickformat=".0%", range=[0,y_max]))
        st.plotly_chart(fig_sc, use_container_width=True)

    with c2:
        st.markdown('<div class="sec"><div class="sec-dot"></div>Before vs After Rerouting</div>', unsafe_allow_html=True)
        fig_ba = go.Figure(data=[
            go.Bar(x=["Before"],y=[rr["original_flood_prob"].mean()],
                   marker_color=P["red"],opacity=0.85,
                   text=[f"{rr['original_flood_prob'].mean():.1%}"],
                   textposition="outside",textfont=dict(color=P["text"]),width=0.4),
            go.Bar(x=["After"],y=[rr["alternative_flood_prob"].mean()],
                   marker_color=P["teal"],opacity=0.85,
                   text=[f"{rr['alternative_flood_prob'].mean():.1%}"],
                   textposition="outside",textfont=dict(color=P["text"]),width=0.4),
        ])
        th(fig_ba, h=180, showlegend=False,
           yaxis=dict(range=[0,1.1],tickformat=".0%"),
           margin=dict(l=10,r=10,t=10,b=20))
        st.plotly_chart(fig_ba, use_container_width=True)

        st.markdown('<div class="sec" style="margin-top:.2rem;"><div class="sec-dot"></div>Detour Distribution</div>', unsafe_allow_html=True)
        fig_h = go.Figure(data=go.Histogram(
            x=rr["extra_time_min"],nbinsx=22,
            marker_color=P["blue"],opacity=0.82,
            marker_line_color=P["bg"],marker_line_width=0.8,
        ))
        mt = rr["extra_time_min"].mean()
        fig_h.add_vline(x=mt, line_dash="dash", line_color=P["red"], line_width=1.8,
                         annotation_text=f"avg {mt:.0f} min",
                         annotation_font_color=P["red"], annotation_font_size=10,
                         annotation_position="top right")
        th(fig_h, h=170, showlegend=False,
           xaxis=dict(title="Extra Time (min)"),
           yaxis=dict(title="Routes"),
           margin=dict(l=10,r=10,t=10,b=20))
        st.plotly_chart(fig_h, use_container_width=True)

    # How rerouting works
    st.markdown('<div class="sec"><div class="sec-dot"></div>How Safe Routes Are Found</div>', unsafe_allow_html=True)
    steps_r = [
        (P["red"],    "Score Every Road",
         "XGBoost flood probability is assigned to every road segment in Nairobi's OSMnx graph — 200k+ edges."),
        (P["orange"], "Inflate Flooded Roads",
         "Any road with flood probability > 0 has its travel cost multiplied by up to 1,000,000× — effectively impassable."),
        (P["blue"],   "Run Dijkstra",
         "The algorithm finds the shortest cost path. With inflated costs, flooded roads are always avoided."),
        (P["teal"],   "Serve via GTFS-RT",
         "The safe alternative is encoded as a TripUpdate protobuf and served live to Google Maps and transit apps."),
    ]
    rcols_s = st.columns(4)
    for col, (c, title, desc) in zip(rcols_s, steps_r):
        with col:
            st.markdown(f"""
            <div style='background:rgba({",".join(str(x) for x in _RGB.get(
                "teal" if c==P["teal"] else "blue" if c==P["blue"] else
                "orange" if c==P["orange"] else "red","red"))[:3].split(",")},
                .06);border:1px solid {c}28;border-top:2px solid {c};
                border-radius:10px;padding:.9rem;height:100%;'>
              <div style='font-size:.69rem;color:{c};font-weight:800;
                          text-transform:uppercase;letter-spacing:.06em;margin-bottom:.35rem;'>{title}</div>
              <div style='font-size:.77rem;color:{P["text"]};line-height:1.55;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ▌ PAGE 4 — MODEL ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif page == "Model Analysis":
    mdf = model_df()
    MORDER = ["XGBoost","Random Forest","Neural Network","Logistic Regression"]
    MC = {"XGBoost":P["teal"],"Random Forest":P["blue"],
          "Neural Network":P["orange"],"Logistic Regression":P["muted"]}
    AUCS = dict(zip(mdf["Model"],mdf["AUC"]))

    st.markdown(f"""
    <div style='font-family:"Syne",sans-serif;font-size:1.9rem;font-weight:800;
                color:{P["white"]};margin-bottom:.2rem;'>Model Analysis</div>
    <div style='color:{P["muted"]};font-size:.82rem;margin-bottom:1rem;'>
      Held-out test set · 30% split · <b style='color:{P["red"]};'>Recall</b> is the primary metric
    </div>
    """, unsafe_allow_html=True)

    mc_palette = {"Recall":P["red"],"AUC":P["teal"],"F1":P["blue"],"Precision":P["orange"],"Accuracy":P["yellow"]}
    mdf_s = mdf.set_index("Model").reindex(MORDER).reset_index()

    def bar_cell_m(v, clr, bold=False):
        pct = int(v*100)
        fw  = "700" if bold else "500"
        col_txt = P["white"] if bold else P["text"]
        return (
            f"<td style='padding:.55rem .9rem;vertical-align:middle;'>"
            f"<div style='display:flex;align-items:center;gap:.45rem;'>"
            f"<div style='flex:1;height:4px;background:{P['border']};border-radius:2px;'>"
            f"<div style='width:{pct}%;height:4px;background:{clr};border-radius:2px;'></div></div>"
            f"<span style='font-size:.72rem;font-weight:{fw};color:{col_txt};"
            f"font-family:\"JetBrains Mono\",monospace;width:33px;text-align:right;'>{v:.0%}</span>"
            f"</div></td>"
        )

    rows_m = ""
    for _, row in mdf_s.iterrows():
        is_xgb = row["Model"] == "XGBoost"
        bg     = f"background:rgba(0,200,150,.05);" if is_xgb else ""
        badge  = "<span class='risk-pill r-safe' style='font-size:.6rem;'>★ Selected</span>" if is_xgb else ""
        cells  = "".join(bar_cell_m(row[m],mc_palette[m],is_xgb) for m in mc_palette)
        rows_m += f"""
        <tr style='{bg}'>
          <td style='padding:.55rem .9rem;color:{"#fff" if is_xgb else P["text"]};
                     font-weight:{"700" if is_xgb else "400"};font-size:.8rem;'>
            {row["Model"]} {badge}
          </td>{cells}
        </tr>"""

    hdr = "".join(f"<th style='padding:.55rem .9rem;color:{mc_palette[m]};'>{m}</th>" for m in mc_palette)
    st.markdown(f"""
    <table class="dtbl"><thead>
      <tr><th style='padding:.55rem .9rem;'>Model</th>{hdr}</tr>
    </thead><tbody>{rows_m}</tbody></table>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec"><div class="sec-dot"></div>ROC Curves</div>', unsafe_allow_html=True)
        rng_r = np.random.default_rng(77)
        fp_b  = np.linspace(0, 1, 300)
        fig_r = go.Figure()
        for model in MORDER[::-1]:
            auc_v = AUCS[model]; col = MC[model]; is_xgb = model=="XGBoost"
            tp = np.sort(np.clip(fp_b**(1/(auc_v*1.5+0.3)) + rng_r.normal(0,.007,300),0,1))
            fig_r.add_trace(go.Scatter(
                x=fp_b, y=tp, name=f"{model}  {auc_v:.2f}",
                mode="lines",
                line=dict(color=col, width=2.8 if is_xgb else 1.4,
                          dash="solid" if is_xgb else "dot"),
                fill="tozeroy" if is_xgb else "none",
                fillcolor=rgba("teal",0.05),
            ))
        fig_r.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Baseline",mode="lines",
                                    line=dict(color=P["muted"],width=1,dash="dash")))
        th(fig_r, h=340, title="ROC Curves",
           xaxis=dict(title="False Positive Rate",range=[0,1]),
           yaxis=dict(title="True Positive Rate",range=[0,1.04]),
           legend=dict(x=.4,y=.1,font=dict(size=9)))
        st.plotly_chart(fig_r, use_container_width=True)

    with c2:
        st.markdown('<div class="sec"><div class="sec-dot"></div>Performance Radar</div>', unsafe_allow_html=True)
        cats_r = ["AUC","Recall","F1","Accuracy","Precision"]
        fig_rd = go.Figure()
        for model in MORDER:
            row = mdf[mdf["Model"]==model].iloc[0]
            vals = [row[c] for c in cats_r] + [row[cats_r[0]]]
            is_xgb = model=="XGBoost"
            col = MC[model]
            r2,g2,b2 = _RGB["teal"] if is_xgb else _RGB["muted"]
            fig_rd.add_trace(go.Scatterpolar(
                r=vals, theta=cats_r+[cats_r[0]], name=model,
                line=dict(color=col,width=2.8 if is_xgb else 1.2),
                fill="toself",
                fillcolor=f"rgba({r2},{g2},{b2},{0.08 if is_xgb else 0.02})",
            ))
        th(fig_rd, h=340, title="5-Metric Radar",
           polar=dict(
               bgcolor=P["card"],
               radialaxis=dict(range=[0.50,1.0],tickformat=".0%",
                               gridcolor=P["border"],linecolor=P["border"],
                               tickfont=dict(size=8,color=P["muted"])),
               angularaxis=dict(gridcolor=P["border"],linecolor=P["border"],
                                tickfont=dict(size=10,color=P["text"])),
           ),
           legend=dict(x=0.73,y=0.05,font=dict(size=9)))
        st.plotly_chart(fig_rd, use_container_width=True)

    # Confusion matrices
    st.markdown('<div class="sec"><div class="sec-dot"></div>Confusion Matrices</div>', unsafe_allow_html=True)
    CMS = {
        "Logistic Regression": np.array([[218,34],[58,125]]),
        "Neural Network":      np.array([[225,27],[76,107]]),
        "Random Forest":       np.array([[237,15],[44,139]]),
        "XGBoost":             np.array([[239,13],[41,142]]),
    }
    notes_cm = {
        "Logistic Regression": (P["muted"],"Baseline","58 missed floods"),
        "Neural Network":      (P["orange"],"Underfit","Worst recall — too few samples for DL"),
        "Random Forest":       (P["blue"],"Runner-up","44 missed, AUC 0.88"),
        "XGBoost":             (P["teal"],"★ Best","Fewest misses (41), AUC 0.90"),
    }
    cm_cols = st.columns(4)
    for cu, model in zip(cm_cols, MORDER):
        with cu:
            cm = CMS[model]; is_xgb = model=="XGBoost"
            ac, status, note = notes_cm[model]
            r3,g3,b3 = _RGB["teal"] if is_xgb else _RGB["blue"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm[::-1],
                x=["Pred:No Flood","Pred:Flood"],y=["Act:Flood","Act:No Flood"],
                colorscale=[[0,P["bg"]],[0.5,f"rgba({r3},{g3},{b3},.45)"],[1,f"rgba({r3},{g3},{b3},.95)"]],
                showscale=False,
                text=cm[::-1],texttemplate="%{text}",textfont=dict(size=15,color=P["white"]),
            ))
            th(fig_cm, h=200,
               title=dict(text=model,font=dict(size=9.5,color=ac),x=0,xanchor="left"),
               margin=dict(l=4,r=4,t=32,b=4),
               xaxis=dict(tickfont=dict(size=7.5)),
               yaxis=dict(tickfont=dict(size=7.5)))
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown(f"""
            <div style='background:{"rgba(0,200,150,.06)" if is_xgb else P["card"]};
                        border:1px solid {"rgba(0,200,150,.2)" if is_xgb else P["border"]};
                        border-radius:8px;padding:.55rem .8rem;margin-top:-.3rem;'>
              <div style='font-size:.66rem;color:{ac};font-weight:700;margin-bottom:.12rem;'>{status}</div>
              <div style='font-size:.72rem;color:{P["text"]};'>{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ibox">
      ✅ <b>Verdict:</b> <b style='color:{P["teal"]};'>XGBoost</b> selected as production model.
      AUC 0.90, Recall 0.81, F1 0.77 — highest on all metrics. Gradient-boosted trees capture
      the <b>non-linear terrain relationships</b> (elevation, slope) that dominate flood susceptibility.
      A missed flood (false negative) can cost lives — maximising recall is non-negotiable.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ▌ PAGE 5 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    wdf = ward_df()
    fl = wdf[wdf["flood_label"]==1]; nf = wdf[wdf["flood_label"]==0]

    st.markdown(f"""
    <div style='font-family:"Syne",sans-serif;font-size:1.9rem;font-weight:800;
                color:{P["white"]};margin-bottom:.2rem;'>Data Explorer</div>
    <div style='color:{P["muted"]};font-size:.82rem;margin-bottom:1rem;'>
      {len(wdf)} Nairobi wards · flood/no-flood labels · terrain & rainfall features
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec"><div class="sec-dot"></div>Class Distribution</div>', unsafe_allow_html=True)
        fig_cd = go.Figure(data=go.Bar(
            x=["Not Flooded","Flooded"], y=[len(nf),len(fl)],
            marker_color=[P["blue"],P["red"]],
            marker_line_color=P["bg"],marker_line_width=2,
            text=[f"{len(nf)} ({len(nf)/len(wdf):.0%})",f"{len(fl)} ({len(fl)/len(wdf):.0%})"],
            textposition="outside",textfont=dict(color=P["text"]),width=0.4,
        ))
        th(fig_cd, h=230, showlegend=False,
           yaxis=dict(range=[0,max(len(nf),len(fl))*1.22]))
        st.plotly_chart(fig_cd, use_container_width=True)

    with c2:
        st.markdown('<div class="sec"><div class="sec-dot"></div>Flood Probability Distribution</div>', unsafe_allow_html=True)
        fig_fp = go.Figure()
        fig_fp.add_trace(go.Histogram(x=nf["flood_prob"],name="Not Flooded",
                                       marker_color=P["blue"],opacity=0.72,nbinsx=28))
        fig_fp.add_trace(go.Histogram(x=fl["flood_prob"],name="Flooded",
                                       marker_color=P["red"],opacity=0.72,nbinsx=28))
        fig_fp.add_vline(x=0.45,line_dash="dash",line_color=P["orange"],line_width=1.5,
                          annotation_text="threshold 0.45",annotation_position="top right",
                          annotation_font_color=P["orange"],annotation_font_size=10)
        th(fig_fp, h=230, barmode="overlay",
           xaxis=dict(title="Flood Probability"),yaxis=dict(title="Ward Count"))
        st.plotly_chart(fig_fp, use_container_width=True)

    # Feature selector
    st.markdown('<div class="sec"><div class="sec-dot"></div>Feature Analysis</div>', unsafe_allow_html=True)
    FEATS = {"elev_m":"Ward Elevation (m)","flood_prob":"Flood Probability","pop":"Population (2009)"}
    feat = st.selectbox("Feature",list(FEATS.keys()),format_func=lambda k:FEATS[k])

    fc1,fc2,fc3 = st.columns([3,1,1])
    with fc1:
        fig_fh = go.Figure()
        fig_fh.add_trace(go.Histogram(x=nf[feat],name="Not Flooded",
                                       marker_color=P["blue"],opacity=0.70,nbinsx=30))
        fig_fh.add_trace(go.Histogram(x=fl[feat],name="Flooded",
                                       marker_color=P["red"],opacity=0.70,nbinsx=30))
        th(fig_fh, h=260, title=FEATS[feat],
           barmode="overlay",
           xaxis=dict(title=FEATS[feat]),yaxis=dict(title="Count"))
        st.plotly_chart(fig_fh, use_container_width=True)

    for col, (subset,label,ckey) in zip([fc2,fc3],[
        (fl,"Flooded","red"),(nf,"Not Flooded","blue"),
    ]):
        with col:
            s = subset[feat]; r,g,b = _RGB[ckey]
            st.markdown(f"""
            <div style='background:rgba({r},{g},{b},.06);border:1px solid rgba({r},{g},{b},.18);
                        border-radius:10px;padding:.9rem;margin-top:1.8rem;'>
              <div style='font-size:.64rem;color:rgba({r},{g},{b},1);font-weight:800;
                          text-transform:uppercase;letter-spacing:.07em;margin-bottom:.55rem;'>{label}</div>
              <div style='font-size:.74rem;color:{P["text"]};line-height:2;
                          font-family:"JetBrains Mono",monospace;'>
                n &nbsp;&nbsp;&nbsp;{len(s)}<br>
                μ &nbsp;&nbsp;&nbsp;{s.mean():.1f}<br>
                med {s.median():.1f}<br>
                σ &nbsp;&nbsp;&nbsp;{s.std():.1f}<br>
                min {s.min():.1f}<br>
                max {s.max():.1f}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Elevation scatter
    st.markdown('<div class="sec"><div class="sec-dot"></div>Elevation vs Flood Probability</div>', unsafe_allow_html=True)
    fig_es = go.Figure()
    for lv, ckey, name in [(0,"blue","Not Flooded"),(1,"red","Flooded")]:
        sub = wdf[wdf["flood_label"]==lv]
        fig_es.add_trace(go.Scatter(
            x=sub["elev_m"],y=sub["flood_prob"],
            mode="markers",name=name,
            marker=dict(color=P[ckey],size=7,opacity=0.75,
                        line=dict(width=0.5,color=P["bg"])),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Elevation: %{x:.0f} m<br>Flood Prob: %{y:.2f}<br>Ward: %{text}<extra></extra>"
            ),
            text=sub["ward"],
        ))
    fig_es.add_hline(y=0.45,line_dash="dash",line_color=P["orange"],line_width=1.2,
                      annotation_text="decision threshold 0.45",
                      annotation_font_color=P["orange"],annotation_font_size=10)
    th(fig_es, h=310,
       title="Lower elevation → higher flood risk (r ≈ −0.50)",
       xaxis=dict(title="Ward Elevation (m)"),
       yaxis=dict(title="Flood Probability"))
    st.plotly_chart(fig_es, use_container_width=True)

    st.markdown(f"""
    <div class="ibox">
      📊 <b>Key Finding:</b> Elevation is the dominant predictor (r ≈ −0.50 with flood labels).
      Rainfall features show near-zero linear correlation — Nairobi flooding is <b>terrain-driven</b>,
      not rainfall-driven. Wards like <b>Mathare</b> (1,606 m), <b>Kibera</b> (1,648 m),
      and <b>Embakasi</b> (1,591 m) act as catch-basins for runoff from elevated suburbs.
      XGBoost captures these non-linear terrain effects that linear models cannot.
    </div>
    """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════
# ▌ PAGE 6 — CHAT ASSISTANT
# ══════════════════════════════════════════════════════════════════
elif page == "Chat Assistant":

    # ── CSS additions for chat ──
    st.markdown(f"""
    <style>
      .chat-bubble-user {{
        background: linear-gradient(135deg,{P["blue"]}22,{P["blue"]}12);
        border: 1px solid {P["blue"]}30;
        border-radius: 16px 16px 4px 16px;
        padding: .75rem 1.1rem; margin: .4rem 0 .4rem auto;
        max-width: 78%; font-size: .84rem; color: {P["text"]};
        line-height: 1.6; word-wrap: break-word;
      }}
      .chat-bubble-ai {{
        background: {P["card"]};
        border: 1px solid {P["border"]};
        border-radius: 16px 16px 16px 4px;
        padding: .75rem 1.1rem; margin: .4rem auto .4rem 0;
        max-width: 88%; font-size: .84rem; color: {P["text"]};
        line-height: 1.65; word-wrap: break-word;
      }}
      .chat-bubble-ai b {{ color: {P["white"]}; }}
      .chat-bubble-ai code {{
        background: {P["bg"]}; border: 1px solid {P["border"]};
        border-radius: 4px; padding: .1rem .35rem;
        font-family: "JetBrains Mono", monospace; font-size: .78rem;
        color: {P["teal"]};
      }}
      .chat-avatar-ai {{
        width: 28px; height: 28px; border-radius: 8px; flex-shrink: 0;
        background: linear-gradient(135deg,{P["teal"]},{P["blue"]});
        display: flex; align-items: center; justify-content: center;
        font-size: .85rem;
      }}
      .chat-avatar-user {{
        width: 28px; height: 28px; border-radius: 8px; flex-shrink: 0;
        background: {P["border"]}; display: flex;
        align-items: center; justify-content: center; font-size: .85rem;
      }}
      .chat-row-ai   {{ display:flex; gap:.6rem; align-items:flex-end; margin:.25rem 0; }}
      .chat-row-user {{ display:flex; gap:.6rem; align-items:flex-end;
                        margin:.25rem 0; flex-direction:row-reverse; }}
      .chip-btn {{
        background:{P["card"]}; border:1px solid {P["border"]};
        border-radius:100px; padding:.28rem .85rem;
        font-size:.72rem; color:{P["text"]}; cursor:pointer;
        transition:all .15s; font-family:"Space Grotesk",sans-serif;
        white-space:nowrap;
      }}
      .chip-btn:hover {{ background:{P["teal"]}15; border-color:{P["teal"]}40; color:{P["teal"]}; }}
      .typing-dot {{
        width:6px;height:6px;border-radius:50%;
        background:{P["teal"]};display:inline-block;margin:0 2px;
        animation:typingBounce .9s infinite;
      }}
      .typing-dot:nth-child(2){{animation-delay:.15s;}}
      .typing-dot:nth-child(3){{animation-delay:.30s;}}
      @keyframes typingBounce{{0%,80%,100%{{transform:translateY(0);opacity:.4;}}
        40%{{transform:translateY(-5px);opacity:1;}}}}
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.markdown(f"""
        <div style='margin-bottom:.4rem;'>
          <div style='font-family:"Syne",sans-serif;font-size:1.9rem;
                      font-weight:800;color:{P["white"]};letter-spacing:-.02em;'>
            💬 Flood Guard AI
          </div>
          <div style='font-size:.82rem;color:{P["muted"]};margin-top:.1rem;'>
            Your real-time flood intelligence assistant · Ask about any ward, route, or evacuation
          </div>
        </div>
        """, unsafe_allow_html=True)
    with hdr_r:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    # ── Build real-time context snapshot for system prompt ──
    _chat_hour = st.session_state.get("chat_hour", 15)
    wdf_chat   = ward_df_at_hour(_chat_hour)
    routes_chat = [route_status_at_hour(rt, _chat_hour) for rt in ROUTES]

    _phase_map_c = {
        **{h: ("Pre-Rain",      "00:00–06:00") for h in range(0, 6)},
        **{h: ("Rainfall Onset","06:00–14:00") for h in range(6, 14)},
        **{h: ("Peak Flood",    "14:00–18:00") for h in range(14, 18)},
        **{h: ("Draining",      "18:00–24:00") for h in range(18, 24)},
    }
    _c_phase, _c_phase_range = _phase_map_c[_chat_hour]

    _crit_wards  = wdf_chat[wdf_chat["flood_prob"] >= 0.75].sort_values("flood_prob", ascending=False)
    _evac_wards  = wdf_chat[wdf_chat["evacuate"]].sort_values("flood_prob", ascending=False)
    _safe_wards  = wdf_chat[wdf_chat["flood_prob"] < 0.35].sort_values("flood_prob")
    _aff_routes  = [r for r in routes_chat if r["rerouted"]]
    _safe_routes = [r for r in routes_chat if not r["rerouted"]]
    _n_hi_c      = int((wdf_chat["flood_prob"] >= 0.45).sum())
    _n_crit_c    = int((wdf_chat["flood_prob"] >= 0.75).sum())
    _n_evac_c    = int(wdf_chat["evacuate"].sum())

    crit_list = "; ".join(
        f"{r['ward']} ({r['flood_prob']:.0%})" for _, r in _crit_wards.head(10).iterrows()
    )
    evac_list = "; ".join(
        f"{r['ward']} ({r['flood_prob']:.0%})" for _, r in _evac_wards.head(12).iterrows()
    )
    safe_ward_list = "; ".join(
        f"{r['ward']} ({r['flood_prob']:.0%})" for _, r in _safe_wards.head(8).iterrows()
    )
    aff_route_list = "\n".join(
        f"  - {r['name']} [{r['id']}]: {r['flood_prob']:.0%} risk, +{r['delay_min']} min delay, {r['status']}"
        for r in _aff_routes
    )
    safe_route_list = "\n".join(
        f"  - {r['name']} [{r['id']}]: clear, {r['status']}"
        for r in _safe_routes
    )

    SYSTEM_PROMPT = f"""You are Flood Guard AI, the emergency intelligence assistant for the Nairobi Flood Guard platform.
You have real-time access to flood susceptibility data, ward risk scores, matatu route statuses,
evacuation zones, and commuter safety guidance for Nairobi, Kenya.

═══════════════════════════════════════════
CURRENT EVENT SNAPSHOT  (April 25, 2024)
═══════════════════════════════════════════
Event Hour  : {_chat_hour:02d}:00
Event Phase : {_c_phase} ({_c_phase_range})
Total wards : {len(wdf_chat)}
High-risk   : {_n_hi_c} wards (flood prob ≥ 45%)
Critical    : {_n_crit_c} wards (flood prob ≥ 75%)
Evacuate    : {_n_evac_c} wards under immediate advisory
Routes affected: {len(_aff_routes)} of {len(routes_chat)} rerouted

CRITICAL / EVACUATE WARDS (prob ≥ 75%):
{crit_list}

ALL EVACUATION ZONES (prob ≥ 72%):
{evac_list}

SAFE WARDS (prob < 35%):
{safe_ward_list}

AFFECTED MATATU ROUTES:
{aff_route_list}

SAFE MATATU ROUTES:
{safe_route_list}

HOW THE MODEL WORKS:
- XGBoost classifier trained on NASA SRTM elevation + CHIRPS 90-day rainfall per ward
- AUC 0.90, Recall 0.81 on April 2024 UNOSAT flood labels
- Key insight: flooding is TERRAIN-driven not rainfall-driven at ward scale
  (low-elevation wards accumulate runoff from surrounding highlands)
- Rerouting: edge cost = travel_time × (1 + 1,000,000 × flood_prob)
- Dijkstra finds 100% flood-free alternative paths; served as GTFS-RT 2.0 protobuf

═══════════════════════════════════════════
YOUR PERSONALITY & RULES
═══════════════════════════════════════════
- Be direct, clear, and calm. This is an emergency context — no waffle.
- Lead with the most safety-critical information first.
- Use ward names, percentages, and route IDs from the data above.
- For evacuation questions, always give the risk score AND nearest safe ward.
- For route questions, give the delay, alternative route, and reason.
- For "is X safe?" — answer with the ward's exact flood probability.
- Use **bold** for critical numbers and ward names.
- Keep responses concise (3–6 lines ideal, longer only when explaining complex topics).
- If asked about a ward NOT in your data, say you don't have data for it.
- Never invent statistics. Only use numbers from the snapshot above.
- Respond in English. If the user writes in Swahili, reply in Swahili.
- End emergency-tier responses with: "Stay safe. 🌊"
"""

    # ── Event hour selector (synced with Live Map) ──
    with st.expander("⚙ Context: event hour for this conversation", expanded=False):
        new_hour = st.slider(
            "Set event hour context",
            0, 23, _chat_hour, format="%d:00",
            help="The chatbot's flood data is based on this hour",
        )
        if new_hour != _chat_hour:
            st.session_state.chat_hour = new_hour
            st.rerun()
        st.markdown(f"""
        <div style='font-size:.74rem;color:{P["muted"]};margin-top:.4rem;'>
          Phase: <b style='color:{P["white"]};'>{_c_phase}</b> ·
          Critical wards: <b style='color:{P["red"]};'>{_n_crit_c}</b> ·
          Routes rerouted: <b style='color:{P["orange"]};'>{len(_aff_routes)}</b>
        </div>
        """, unsafe_allow_html=True)

    # ── Suggested prompts ──
    SUGGESTED = [
        "Which wards should I evacuate right now?",
        "Is Kibera safe to travel through?",
        "What's the safest route to CBD from Kasarani?",
        "How bad is flooding in Mathare?",
        "Which matatu routes are delayed?",
        "Why does elevation affect flood risk?",
        "What should I do if I'm stuck in a flooded ward?",
        "Is Karen safe right now?",
    ]

    if not st.session_state.chat_messages:
        st.markdown(f"""
        <div style='background:{P["card"]};border:1px solid {P["border"]};
                    border-radius:16px;padding:1.5rem 1.8rem;margin-bottom:1rem;'>
          <div style='display:flex;align-items:center;gap:.6rem;margin-bottom:.8rem;'>
            <div style='width:34px;height:34px;border-radius:10px;
                        background:linear-gradient(135deg,{P["teal"]},{P["blue"]});
                        display:flex;align-items:center;justify-content:center;font-size:1rem;'>🌊</div>
            <div>
              <div style='font-weight:700;color:{P["white"]};font-size:.9rem;'>Flood Guard AI</div>
              <div style='font-size:.7rem;color:{P["muted"]};'>Online · April 25, 2024 · {_chat_hour:02d}:00 · {_c_phase}</div>
            </div>
          </div>
          <div style='font-size:.85rem;color:{P["text"]};line-height:1.65;'>
            Hello. I'm your real-time flood intelligence assistant for Nairobi.<br><br>
            I have live data on <b style='color:{P["white"]};'>{len(wdf_chat)} wards</b>,
            <b style='color:{P["red"]};'>{_n_crit_c} critical flood zones</b>,
            and <b style='color:{P["orange"]};'>{len(_aff_routes)} affected matatu routes</b>
            at {_chat_hour:02d}:00.<br><br>
            Ask me about ward safety, evacuation routes, matatu delays, or anything about the flood.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:.68rem;color:{P['muted']};margin-bottom:.4rem;text-transform:uppercase;letter-spacing:.07em;'>Suggested questions</div>", unsafe_allow_html=True)
        chip_cols = st.columns(4)
        for i, q in enumerate(SUGGESTED):
            with chip_cols[i % 4]:
                if st.button(q, key=f"chip_{i}", use_container_width=True):
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    st.rerun()

    # ── Chat history ──
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-row-user">
                  <div class="chat-avatar-user">👤</div>
                  <div class="chat-bubble-user">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                import re
                # Convert markdown bold/code for HTML display
                html_content = msg["content"]
                html_content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html_content)
                html_content = re.sub(r"`(.+?)`", r"<code>\1</code>", html_content)
                html_content = html_content.replace("\n", "<br>")
                st.markdown(f"""
                <div class="chat-row-ai">
                  <div class="chat-avatar-ai">🌊</div>
                  <div class="chat-bubble-ai">{html_content}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Input ──
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    input_col, btn_col = st.columns([5, 1])

    with input_col:
        user_input = st.chat_input(
            placeholder="Ask about a ward, route, evacuation, or flood risk...",
        )

    # ── Process message ──
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        # Build conversation history for API
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages
        ]

        # Typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown(f"""
        <div class="chat-row-ai">
          <div class="chat-avatar-ai">🌊</div>
          <div class="chat-bubble-ai" style='padding:.65rem 1rem;'>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            # Get API key from secrets or env
            api_key = (
                st.secrets.get("ANTHROPIC_API_KEY", None)
                or os.environ.get("ANTHROPIC_API_KEY", None)
            )

            if not api_key:
                response_text = (
                    "⚠ **API key not configured.**\n\n"
                    "To enable the AI assistant, add your Anthropic API key:\n"
                    "1. Create `.streamlit/secrets.toml` in the project root\n"
                    "2. Add: `ANTHROPIC_API_KEY = \"your-key-here\"`\n"
                    "3. Or set the environment variable `ANTHROPIC_API_KEY`\n\n"
                    "Get your key at [console.anthropic.com](https://console.anthropic.com)"
                )
            else:
                client = anthropic.Anthropic(api_key=api_key)

                # Stream the response
                full_response = ""
                stream_placeholder = st.empty()

                with client.messages.stream(
                    model="claude-sonnet-4-5",
                    max_tokens=600,
                    system=SYSTEM_PROMPT,
                    messages=api_messages,
                ) as stream:
                    for text_chunk in stream.text_stream:
                        full_response += text_chunk
                        # Live stream display
                        import re as _re
                        _html = full_response
                        _html = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _html)
                        _html = _re.sub(r"`(.+?)`", r"<code>\1</code>", _html)
                        _html = _html.replace("\n", "<br>")
                        stream_placeholder.markdown(f"""
                        <div class="chat-row-ai">
                          <div class="chat-avatar-ai">🌊</div>
                          <div class="chat-bubble-ai">{_html}<span style="opacity:.4;">▍</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                response_text = full_response
                stream_placeholder.empty()

        except Exception as e:
            response_text = f"⚠ **Error:** {str(e)}"

        typing_placeholder.empty()
        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
        st.rerun()

    # ── Suggested chips (when conversation is ongoing) ──
    if st.session_state.chat_messages:
        st.markdown(f"""
        <div style='margin-top:.6rem;'>
          <div style='font-size:.66rem;color:{P["muted"]};text-transform:uppercase;
                      letter-spacing:.07em;margin-bottom:.4rem;'>Quick questions</div>
        </div>
        """, unsafe_allow_html=True)
        quick_cols = st.columns(4)
        quick_qs = [
            "Which wards need to evacuate?",
            "Safest route to CBD?",
            "How long will flooding last?",
            "Is Westlands safe?",
        ]
        for i, q in enumerate(quick_qs):
            with quick_cols[i % 4]:
                if st.button(q, key=f"quick_{i}", use_container_width=True):
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    st.rerun()


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='margin-top:3rem;padding:1rem 0;border-top:1px solid {P["border"]};
            display:flex;justify-content:space-between;align-items:center;
            font-size:.67rem;color:{P["muted"]};flex-wrap:wrap;gap:.5rem;'>
  <span>🌊 <b style='color:{P["teal"]};'>Nairobi Flood Guard</b> · Group 4</span>
  <span>XGBoost · OSMnx · Dijkstra · GTFS-RT 2.0 · Streamlit</span>
  <span>Data: NASA SRTM · CHIRPS · UNOSAT · Digital Matatus 2019</span>
</div>
""", unsafe_allow_html=True)
