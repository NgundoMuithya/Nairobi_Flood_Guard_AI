"""
Nairobi Flood Guard - Streamlit UI
Run with: streamlit run app.py
"""

import warnings
import json
import pickle
import base64
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import africastalking

    AT_AVAILABLE = True
except ImportError:
    AT_AVAILABLE = False

import folium
import streamlit as st
from streamlit_folium import st_folium
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go

from Utils.rainfall_fetcher import RAIN_COLS, apply_live_rainfall, rainfall_summary
from Utils.live_routing import run_live_rerouting

try:
    import osmnx as ox

    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

warnings.filterwarnings("ignore")


# -- Paths --------------------------------------------------------------------
BASE = Path(__file__).parent
DATA = BASE / "Data"
MODELS = BASE / "Models"
GTFS_DIR = DATA / "GTFS_FEED_2019"
REPORTS = BASE / "Route_Optimization" / "Reports"
GROQ_MODEL = "llama-3.3-70b-versatile"

FLOODS_GPKG = DATA / "floods.gpkg"
XGB_MODEL = MODELS / "best_xgboost_model.pkl"
REROUTING_CSV = REPORTS / "rerouting_summary.csv"
TRADEOFF_PNG = REPORTS / "rerouting_tradeoff.png"
ROUTE_GEOMETRIES = REPORTS / "route_geometries.json"
ROAD_GRAPH = DATA / "nairobi_road_network.graphml"

NAIROBI_LAT, NAIROBI_LON = -1.286389, 36.817223

FEATURE_COLS = [
    "pop2009",
    "rain_cumulative_mm",
    "rain_max_daily_mm",
    "rain_preflood_7d_mm",
    "elevation_mean_m",
    "elevation_min_m",
    "elevation_max_m",
    "elevation_range_m",
    "slope_mean_deg",
]

# -- Page config --------------------------------------------------------------
st.set_page_config(
    page_title="Nairobi Flood Guard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS -----------------------------------------------------------
# Design tokens - Highland Terrain system
#   Ground     #07110D   panel      #0E2318   panel-raised #12301F
#   Line       #1F4A32   line-soft  #17321F   text         #E8DFC8
#   Text-dim   #8FA894   accent     #D4A24C  (ochre - the working accent)
#   Safe       #3FA66B   moderate   #D4A24C   high  #C4622D   critical #8B2E2E
# Display: Fraunces (surveyed/geological serif) - restrained, headers only
# Data:    Space Mono (coordinates, figures, labels)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --ground: #07110D; --panel: #0E2318; --panel-raised: #12301F;
    --line: #1F4A32; --line-soft: #17321F;
    --text: #E8DFC8; --text-dim: #8FA894; --text-faint: #4E6357;
    --accent: #D4A24C;
    --safe: #3FA66B; --moderate: #D4A24C; --high: #C4622D; --critical: #8B2E2E;
}

html, body, [class*="css"] { font-family: 'Space Mono', monospace; color: var(--text); }
.stApp { background: var(--ground); }
h1, h2, h3, h4 { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; font-weight: 600; }

/* Contour-line texture, reused wherever the terrain motif appears */
.contour-field {
    background-image:
        repeating-radial-gradient(ellipse 140% 100% at 15% 120%,
            transparent 0, transparent 22px, rgba(232,223,200,0.035) 23px, transparent 24px);
}

.header-banner {
    background: linear-gradient(180deg, #0B1F14 0%, #07110D 100%);
    border-bottom: 1px solid var(--line);
    padding: 2.75rem 2.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    position: relative; overflow: hidden;
}
.header-banner::before {
    content: ''; position: absolute; inset: 0;
    background-image:
        repeating-radial-gradient(ellipse 160% 120% at 88% 140%,
            transparent 0px, transparent 26px, rgba(212,162,76,0.05) 27px, transparent 28px),
        repeating-radial-gradient(ellipse 160% 120% at 88% 140%,
            transparent 0px, transparent 52px, rgba(232,223,200,0.04) 53px, transparent 54px);
    pointer-events: none;
}
.header-eyebrow {
    font-family: 'Space Mono', monospace; font-size: 0.68rem; color: var(--accent);
    letter-spacing: 0.22em; text-transform: uppercase; margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.header-eyebrow::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 8px var(--accent);
}
.header-title {
    font-family: 'Fraunces', serif !important;
    font-size: 3rem; font-weight: 600; font-style: normal;
    color: var(--text); letter-spacing: -0.02em; margin: 0; line-height: 1.02;
    position: relative;
}
.header-subtitle {
    font-family: 'Space Mono', monospace; font-size: 0.76rem; color: var(--text-dim);
    margin-top: 0.65rem; letter-spacing: 0.08em; max-width: 640px; line-height: 1.6;
}
.badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.3rem 0.8rem; border-radius: 100px;
    font-family: 'Space Mono', monospace; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.badge::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.badge-low      { background: rgba(63,166,107,0.12);  color: var(--safe);     border: 1px solid rgba(63,166,107,0.4); }
.badge-moderate { background: rgba(212,162,76,0.12);  color: var(--moderate); border: 1px solid rgba(212,162,76,0.4); }
.badge-high     { background: rgba(196,98,45,0.14);   color: var(--high);     border: 1px solid rgba(196,98,45,0.45); }
.badge-critical { background: rgba(139,46,46,0.18);   color: #E8A0A0;         border: 1px solid rgba(139,46,46,0.6); }

.metric-card {
    background: var(--panel); border: 1px solid var(--line-soft);
    border-left: 2px solid var(--accent); border-radius: 3px;
    padding: 1rem 1.2rem; margin-bottom: 0.75rem;
    transition: border-color 0.15s ease, background 0.15s ease;
}
.metric-card:hover { background: var(--panel-raised); border-left-color: var(--text); }
.metric-label {
    font-size: 0.65rem; color: var(--text-dim);
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Fraunces', serif; font-size: 1.7rem;
    font-weight: 600; color: var(--text); line-height: 1;
}
.metric-unit { font-family: 'Space Mono', monospace; font-size: 0.68rem; color: var(--text-faint); margin-left: 0.35rem; }

.section-header {
    font-family: 'Fraunces', serif; font-size: 1.35rem; font-weight: 600;
    color: var(--text); border-bottom: 1px solid var(--line);
    padding-bottom: 0.6rem; margin-bottom: 1.1rem; letter-spacing: -0.01em;
    display: flex; align-items: baseline; justify-content: space-between;
}

section[data-testid="stSidebar"] {
    background: #06100B; border-right: 1px solid var(--line-soft);
}
section[data-testid="stSidebar"] h3 {
    font-family: 'Space Mono', monospace !important; font-size: 0.7rem !important;
    font-weight: 700 !important; color: var(--accent) !important;
    letter-spacing: 0.16em !important; text-transform: uppercase !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stSidebarCollapseButton"] { display: none !important; }
.material-symbols-rounded, [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}

.ward-panel {
    background: linear-gradient(160deg, var(--panel) 0%, var(--ground) 130%);
    border: 1px solid var(--line); border-radius: 4px;
    padding: 1.5rem 1.6rem; margin-top: 1rem; position: relative; overflow: hidden;
}
.ward-panel::after {
    content: ''; position: absolute; inset: 0;
    background-image: repeating-radial-gradient(ellipse 150% 130% at 105% 130%,
        transparent 0, transparent 20px, rgba(212,162,76,0.045) 21px, transparent 22px);
    pointer-events: none;
}
.ward-name {
    font-family: 'Fraunces', serif; font-size: 1.6rem;
    font-weight: 600; color: var(--text); margin-bottom: 0.25rem; position: relative;
}
.ward-meta {
    font-family: 'Space Mono', monospace; font-size: 0.7rem; color: var(--text-dim);
    letter-spacing: 0.06em; position: relative;
}

.stSelectbox label, .stSlider label, .stTextInput label, .stTextArea label, .stRadio label {
    font-family: 'Space Mono', monospace !important; font-size: 0.72rem !important;
    color: var(--text-dim) !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetric"] {
    background: var(--panel); border: 1px solid var(--line-soft);
    border-radius: 3px; padding: 0.75rem 1rem;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif; color: var(--text);
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-dim);
}

.route-stat-card {
    background: var(--panel); border: 1px solid var(--line-soft);
    border-radius: 3px; padding: 1.1rem 1.2rem; text-align: center;
    border-top: 2px solid var(--line);
}
.route-stat-label {
    font-size: 0.63rem; color: var(--text-dim);
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.4rem;
}
.route-stat-value {
    font-family: 'Fraunces', serif; font-size: 1.5rem;
    font-weight: 600; color: var(--text);
}

/* Buttons */
.stButton > button, .stFormSubmitButton > button {
    background: var(--panel-raised) !important; color: var(--text) !important;
    border: 1px solid var(--line) !important; border-radius: 3px !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    transition: border-color 0.15s ease, background 0.15s ease !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    border-color: var(--accent) !important; background: #16351F !important; color: var(--accent) !important;
}

/* Inputs, sliders, selects */
.stSelectbox [data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea {
    background: var(--panel) !important; border-color: var(--line-soft) !important;
    color: var(--text) !important; font-family: 'Space Mono', monospace !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--accent) !important; }
div[data-baseweb="slider"] > div > div { background: var(--line) !important; }
div[data-baseweb="slider"] > div > div > div { background: var(--accent) !important; }

/* Expanders */
.streamlit-expanderHeader, div[data-testid="stExpander"] summary {
    background: var(--panel) !important; border: 1px solid var(--line-soft) !important;
    font-family: 'Space Mono', monospace !important; color: var(--text) !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] { border: 1px solid var(--line-soft); border-radius: 3px; }

/* Map frame - the basemap tiles run cooler than the rest of the UI;
   a deliberate border makes that a frame rather than a seam. */
iframe[title="streamlit_folium.st_folium"] {
    border: 1px solid var(--line) !important;
    border-radius: 4px !important;
}

hr { border-top: 1px solid var(--line-soft) !important; }
</style>
""",
    unsafe_allow_html=True,
)


# -- Helpers ------------------------------------------------------------------
def flood_prob_fingerprint(wards_gdf, alpha, threshold, horizon_hours=0) -> str:
    """Hash of ward flood_prob values + alpha/threshold/horizon - changes
    whenever the underlying risk data (or routing params) actually change,
    so the live-routing cache invalidates correctly without hashing the
    full gdf."""
    import hashlib

    vals = tuple(
        np.round(wards_gdf.sort_values("ward")["flood_prob"].values, 4).tolist()
    )
    return hashlib.md5(
        f"{vals}-{alpha}-{threshold}-{horizon_hours}".encode()
    ).hexdigest()


def risk_label(prob: float) -> tuple[str, str]:
    if prob >= 0.70:
        return "Critical", "badge-critical"
    if prob >= 0.45:
        return "High", "badge-high"
    if prob >= 0.20:
        return "Moderate", "badge-moderate"
    return "Low", "badge-low"


def risk_color(prob: float) -> str:
    if prob >= 0.70:
        return "#8B2E2E"
    if prob >= 0.45:
        return "#C4622D"
    if prob >= 0.20:
        return "#D4A24C"
    return "#3FA66B"


def delta_color(delta: float) -> str:
    """Diverging color for live-vs-historical flood_prob shift. Buckets rather
    than a continuous scale, since a handful of pp of real movement should
    read clearly rather than blur into a gradient."""
    if delta >= 0.03:
        return "#C4622D"  # risk rose meaningfully
    if delta >= 0.01:
        return "#D4A24C"  # risk rose slightly
    if delta <= -0.03:
        return "#2E7D9E"  # risk fell meaningfully
    if delta <= -0.01:
        return "#5FA8C4"  # risk fell slightly
    return "#1F4A32"  # negligible change


def normalize(col: str, df: pd.DataFrame) -> pd.Series:
    """Min-max normalise a DataFrame column to [0, 1]."""
    mn, mx = df[col].min(), df[col].max()
    return (df[col] - mn) / (mx - mn + 1e-9)


def highlight_best(s: pd.Series) -> list[str]:
    """Highlight the highest value in each column of the metrics table."""
    is_best = s == s.max()
    return [
        "background-color: #12301F; color: #3FA66B; font-weight:600" if v else ""
        for v in is_best
    ]


def render_message(role: str, content: str) -> None:
    """
    Render a chat message bubble with a clipboard icon that appears on hover.
    Content is base64-encoded to avoid all quote/backtick escaping issues.
    """
    b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    icon_color = "#8FA894"
    icon_hover = "#D4A24C"
    align = "flex-end" if role == "user" else "flex-start"
    bubble_bg = "#12301F" if role == "user" else "#0E2318"
    border_col = "#D4A24C" if role == "user" else "#1F4A32"
    avatar = "👤" if role == "user" else "🌦️"
    btn_side = "left" if role == "user" else "right"

    html = f"""
<div style="display:flex; flex-direction:column; align-items:{align};
            margin-bottom:0.75rem; width:100%;">
  <div style="display:flex; align-items:flex-start; gap:0.5rem;
              flex-direction:{'row-reverse' if role == 'user' else 'row'}">
    <span style="font-size:1.2rem; margin-top:0.2rem;">{avatar}</span>
    <div class="msg-wrapper" style="position:relative; max-width:85%;">
      <div style="background:{bubble_bg}; border:1px solid {border_col};
                  border-radius:6px; padding:0.75rem 1rem;
                  font-family:'Space Mono',monospace; font-size:0.85rem;
                  color:#E8DFC8; line-height:1.6; white-space:pre-wrap;
                  word-break:break-word;">{content}</div>
      <button
        data-content="{b64}"
        onclick="
          var text = atob(this.getAttribute('data-content'));
          navigator.clipboard.writeText(text).then(() => {{
            var icon = this.querySelector('.icon-default');
            var check = this.querySelector('.icon-check');
            icon.style.display = 'none';
            check.style.display = 'inline';
            setTimeout(() => {{
              icon.style.display = 'inline';
              check.style.display = 'none';
            }}, 1500);
          }});
        "
        onmouseenter="this.style.color='{icon_hover}';this.style.borderColor='{icon_hover}';"
        onmouseleave="this.style.color='{icon_color}';this.style.borderColor='transparent';"
        style="position:absolute; top:0.4rem; {btn_side}:-2.2rem;
               background:transparent; border:1px solid transparent;
               border-radius:4px; padding:0.2rem 0.3rem;
               color:{icon_color}; cursor:pointer;
               font-size:0.9rem; line-height:1;
               opacity:0; transition:opacity 0.15s ease;"
        class="copy-icon-btn"
        title="Copy to clipboard">
        <span class="icon-default">📋</span>
        <span class="icon-check" style="display:none;">✓</span>
      </button>
    </div>
  </div>
</div>
<style>
  .msg-wrapper:hover .copy-icon-btn {{ opacity: 1 !important; }}
</style>
"""
    n_lines = content.count("\n") + len(content) // 80 + 1
    height = max(100, n_lines * 24 + 60)
    st.iframe(html, height=height)


# -- Data loading -------------------------------------------------------------
@st.cache_data
def load_data():
    df = gpd.read_file(FLOODS_GPKG)
    df["elevation_range_m"] = df["elevation_max_m"] - df["elevation_min_m"]
    df["ward"] = df["ward"].str.title()
    df["county"] = df["county"].str.title()
    df["subcounty"] = df["subcounty"].str.title()
    return df


@st.cache_resource
def load_model():
    with open(XGB_MODEL, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_road_graph():
    """~87k nodes / ~213k edges - load once per process, never per rerun."""
    return ox.load_graphml(ROAD_GRAPH)


@st.cache_data(ttl=3600, show_spinner=False)
def get_live_routes(
    _G, _wards_gdf, _stops_df, _stop_times, _trips, alpha, threshold, fingerprint
):
    """
    Recompute flood-weighted rerouting against current ward flood_prob.
    Leading-underscore args are excluded from Streamlit's hash (the graph and
    GTFS tables are large/static); `fingerprint` - a hash of the actual
    flood_prob values plus alpha/threshold - is what invalidates the cache
    when risk data genuinely changes.
    """
    return run_live_rerouting(
        _G, _wards_gdf, _stops_df, _stop_times, _trips, alpha=alpha, threshold=threshold
    )


@st.cache_data
def load_rerouting():
    return pd.read_csv(REROUTING_CSV)


@st.cache_data
def load_model_comparison(path):
    return pd.read_csv(path)


@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


@st.cache_data
def load_gtfs():
    routes = pd.read_csv(GTFS_DIR / "routes.txt")
    trips = pd.read_csv(GTFS_DIR / "trips.txt")
    shapes = pd.read_csv(GTFS_DIR / "shapes.txt")
    stops = pd.read_csv(GTFS_DIR / "stops.txt")
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt")
    return routes, trips, shapes, stops, stop_times


@st.cache_data
def load_route_geometries():
    if not ROUTE_GEOMETRIES.exists():
        return {}
    with open(ROUTE_GEOMETRIES, "r") as f:
        return json.load(f)


def generate_predictions(model, df):
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    return model.predict_proba(X)[:, 1]


def add_risk_columns(model, df):
    scored = df.copy()
    scored["flood_prob"] = generate_predictions(model, scored)
    scored["risk_label"], _ = zip(*scored["flood_prob"].map(risk_label))
    return scored


# Build choropleth - not cached since Folium maps with lambdas can't be pickled
def build_choropleth(map_df, centre_lat, centre_lon, zoom):
    fmap = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
    )
    folium.GeoJson(
        map_df[["ward", "subcounty", "county", "flood_prob", "risk_label", "geometry"]],
        style_function=lambda feature: {
            "fillColor": risk_color(float(feature["properties"]["flood_prob"])),
            "fillOpacity": 0.55,
            "color": risk_color(float(feature["properties"]["flood_prob"])),
            "weight": 0.8,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["ward", "subcounty", "county", "flood_prob", "risk_label"],
            aliases=["Ward", "Sub-County", "County", "Flood Probability", "Risk Level"],
            localize=True,
            sticky=False,
        ),
    ).add_to(fmap)
    return fmap


def build_delta_choropleth(map_df, centre_lat, centre_lon, zoom):
    fmap = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
    )
    folium.GeoJson(
        map_df[
            ["ward", "subcounty", "historical_prob", "live_prob", "delta", "geometry"]
        ],
        style_function=lambda feature: {
            "fillColor": delta_color(float(feature["properties"]["delta"])),
            "fillOpacity": 0.65,
            "color": delta_color(float(feature["properties"]["delta"])),
            "weight": 0.8,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["ward", "subcounty", "historical_prob", "live_prob", "delta"],
            aliases=[
                "Ward",
                "Sub-County",
                "Historical",
                "Live",
                "Δ (Live − Historical)",
            ],
            localize=True,
            sticky=False,
        ),
    ).add_to(fmap)
    return fmap


def get_route_shape(route_id, trips, shapes):
    """Return list of [lat, lon] for the first trip of a route."""
    trip_rows = trips[trips["route_id"] == route_id]
    if trip_rows.empty:
        return []
    shape_id = trip_rows.iloc[0]["shape_id"]
    pts = shapes[shapes["shape_id"] == shape_id].sort_values("shape_pt_sequence")
    return [[row["shape_pt_lat"], row["shape_pt_lon"]] for _, row in pts.iterrows()]


def get_route_stops(route_id, trips, stop_times, stops):
    """Return DataFrame of stops for the first trip of a route."""
    trip_rows = trips[trips["route_id"] == route_id]
    if trip_rows.empty:
        return pd.DataFrame()
    trip_id = trip_rows.iloc[0]["trip_id"]
    return (
        stop_times[stop_times["trip_id"] == trip_id]
        .sort_values("stop_sequence")
        .merge(stops, on="stop_id")
    )


def get_affected_stop_ids(nairobi_df, stops_df, flood_threshold):
    """Return set of stop_ids falling inside high-risk Nairobi wards."""
    high_risk = nairobi_df[nairobi_df["flood_prob"] >= flood_threshold][
        ["geometry"]
    ].copy()
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(stops_gdf, high_risk, how="inner", predicate="within")
    return set(joined["stop_id"].tolist())


def apply_horizon_rainfall(gdf, horizon_hours: int, use_cache: bool):
    vc_key = st.secrets.get("VISUALCROSSING_API_KEY")
    try:
        return apply_live_rainfall(
            gdf,
            use_cache=use_cache,
            horizon_hours=horizon_hours,
            visualcrossing_api_key=vc_key,
        )
    except TypeError as exc:
        # Only treat this as "deployed rainfall_fetcher.py predates horizon
        # support" if it's unmistakably a call-signature mismatch on
        # apply_live_rainfall itself - not any TypeError that happens to
        # mention these identifiers internally (which was swallowing real
        # bugs on the horizon!=0 code path and mislabeling them).
        msg = str(exc)
        is_signature_mismatch = "apply_live_rainfall()" in msg and (
            "unexpected keyword argument" in msg
            or ("missing" in msg and "argument" in msg)
        )
        if not is_signature_mismatch:
            raise  # surface the real error instead of masking it
        if horizon_hours == 0:
            return apply_live_rainfall(gdf, use_cache=use_cache)
        raise RuntimeError(
            "The deployed rainfall_fetcher.py does not support forecast "
            "horizons yet. Redeploy Utils/rainfall_fetcher.py with the "
            "time-series forecast changes."
        ) from exc


@st.cache_data(ttl=21600, show_spinner=False)
def get_open_meteo_ward_dataframe(
    cache_bust: int, skip_file_cache: bool, horizon_hours: int
):
    """
    Fetch Open-Meteo rainfall for Nairobi wards only (~91 wards) and merge it
    into the full nationwide dataframe; other counties keep their historical
    CHIRPS values. Horizon 0 is the live/as-of-now dataset, while 24 and 48 use
    forecast precipitation rolled into the model rainfall windows.
    """
    base = load_data()
    nairobi_mask = base["county"].str.lower() == "nairobi"

    forecast_nairobi, meta = apply_horizon_rainfall(
        base[nairobi_mask],
        horizon_hours=horizon_hours,
        use_cache=not skip_file_cache,
    )

    combined = base.copy()
    combined.loc[nairobi_mask, RAIN_COLS] = forecast_nairobi[RAIN_COLS].values
    meta["scope"] = (
        f"Nairobi ({int(nairobi_mask.sum())} wards) · other counties remain historical"
    )
    return combined, meta


# -- Load base data & model ---------------------------------------------------
base_df = load_data()
model = load_model()

# -- Header -------------------------------------------------------------------
st.markdown(
    """
<div class="header-banner">
    <div class="header-eyebrow">Kenya &middot; 1,450 wards monitored</div>
    <div class="header-title">Nairobi Flood Guard</div>
    <div class="header-subtitle">Water follows elevation, not rainfall alone. This model reads
    the terrain each ward sits in &mdash; and where the surrounding highland will send the
    water next &mdash; to flag risk before it arrives, and to route matatus around it.</div>
</div>
""",
    unsafe_allow_html=True,
)

# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "",
        [
            "Flood Risk Dashboard",
            "Ward Lookup",
            "Route Optimization",
            "AI Assistant",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### Rainfall Data")
    data_mode = st.radio(
        "Source",
        [
            "Live (Open-Meteo)",
            "Historical (Apr 2024)",
            "24hr Prediction",
            "48hr Prediction",
        ],
        help=(
            "Live mode uses the latest Open-Meteo rainfall. The 24hr and 48hr "
            "prediction modes roll forecast precipitation into the model's "
            "90-day, max-daily, and 7-day rainfall features before prediction."
        ),
    )
    forecast_horizon_hours = {
        "Live (Open-Meteo)": 0,
        "24hr Prediction": 24,
        "48hr Prediction": 48,
    }.get(data_mode)
    use_open_meteo = forecast_horizon_hours is not None
    use_live = forecast_horizon_hours == 0
    if use_open_meteo:
        if st.button("Refresh rainfall now", use_container_width=True):
            st.session_state["force_rainfall_refresh"] = True
            bust_key = f"rainfall_cache_bust_{forecast_horizon_hours}"
            st.session_state[bust_key] = st.session_state.get(bust_key, 0) + 1
        st.caption(
            "Open-Meteo rainfall applies to Nairobi's wards only (where route "
            "optimization and alerts operate). Other counties keep the "
            "historical Apr 2024 data. First fetch typically takes a few "
            "seconds; results are cached for 6 hours."
        )
    st.markdown("---")
    st.markdown("### Risk Threshold")
    threshold = st.slider(
        "Flag wards above this probability as high-risk",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        format="%.2f",
    )

# -- Apply rainfall source & generate predictions -------------------------------
force_refresh = st.session_state.pop("force_rainfall_refresh", False)
cache_bust = (
    st.session_state.get(f"rainfall_cache_bust_{forecast_horizon_hours}", 0)
    if use_open_meteo
    else 0
)
rainfall_meta: dict = {"source": "historical", "label": "CHIRPS Feb-Apr 2024"}

if use_open_meteo:
    try:
        spinner_label = (
            "Loading live rainfall..."
            if forecast_horizon_hours == 0
            else f"Loading {forecast_horizon_hours}hr forecast rainfall..."
        )
        with st.spinner(f"{spinner_label} (first fetch may take ~2 min)"):
            df, rainfall_meta = get_open_meteo_ward_dataframe(
                cache_bust, force_refresh, int(forecast_horizon_hours)
            )
    except Exception as exc:
        st.warning(
            f"Could not fetch Open-Meteo rainfall ({exc}). "
            "Falling back to historical Apr 2024 data."
        )
        df = base_df.copy()
        rainfall_meta = {"source": "historical", "label": "CHIRPS Feb-Apr 2024"}
else:
    df = base_df.copy()

df = add_risk_columns(model, df)
nairobi = df[df["county"].str.lower() == "nairobi"].copy()

n_critical = (df["flood_prob"] >= 0.70).sum()
n_high = ((df["flood_prob"] >= threshold) & (df["flood_prob"] < 0.70)).sum()
n_total = (df["flood_prob"] >= threshold).sum()

with st.sidebar:
    st.markdown("---")
    st.markdown("### Kenya-Wide Summary")
    st.metric("Total Wards", len(df))
    st.metric("Critical Risk", int(n_critical))
    st.metric("High Risk", int(n_high))
    st.metric("Above Threshold", int(n_total))
    if use_open_meteo:
        st.caption(rainfall_summary(rainfall_meta))
    st.markdown("---")

    # -- SMS Early Warning System ---------------------------------------------
    st.markdown("### 🔔 SMS Early Warning")

    with st.expander("Send flood alert via SMS", expanded=False):

        if not AT_AVAILABLE:
            st.error(
                "africastalking not installed.\n\n" "Run: `pip install africastalking`"
            )
        else:
            if st.button("💰 Check AT account balance"):
                try:
                    africastalking.initialize(
                        username=st.secrets.get("AT_USERNAME", "sandbox"),
                        api_key=st.secrets["AT_API_KEY"],
                    )
                    app_data = africastalking.Application.fetch_application_data()
                    st.info(f"Balance: {app_data['UserData']['balance']}")
                except Exception as e:
                    st.error(f"Balance check failed: {e}")

            # ── Recipient input ──────────────────────────────────────────────
            sms_input_mode = st.radio(
                "Recipients",
                ["Enter numbers manually", "Auto (all critical wards)"],
                label_visibility="visible",
                horizontal=True,
            )

            if sms_input_mode == "Enter numbers manually":
                raw_numbers = st.text_area(
                    "Phone numbers (one per line, include country code)",
                    placeholder="+254712345678\n+254722345678",
                    height=100,
                )
                recipient_list = [
                    n.strip()
                    for n in raw_numbers.splitlines()
                    if n.strip().startswith("+")
                ]
            else:
                # Pull county from Ward Lookup selection if available,
                # otherwise default to Nairobi
                target_county = st.selectbox(
                    "County to alert",
                    sorted(df["county"].unique()),
                    index=(
                        list(sorted(df["county"].unique())).index("Nairobi")
                        if "Nairobi" in df["county"].unique()
                        else 0
                    ),
                )
                # In production: replace this with a real contacts DB keyed by county.
                # Here we show the count of wards that would be notified so the
                # operator understands the scope before sending.
                critical_wards = df[
                    (df["county"] == target_county) & (df["flood_prob"] >= 0.70)
                ][["ward", "flood_prob"]]
                st.caption(
                    f"{len(critical_wards)} critical ward(s) in {target_county}. "
                    "Connect a contacts database to auto-populate numbers."
                )
                raw_numbers = st.text_area(
                    "Phone numbers for this county (one per line)",
                    placeholder="+254712345678",
                    height=80,
                )
                recipient_list = [
                    n.strip()
                    for n in raw_numbers.splitlines()
                    if n.strip().startswith("+")
                ]

            # ── Message composer ─────────────────────────────────────────────
            st.markdown("**Alert message**")

            # Build a smart default message from live model data
            top_wards = (
                nairobi[df["flood_prob"] >= threshold]
                .nlargest(3, "flood_prob")[["ward", "flood_prob"]]
                .apply(lambda r: f"{r['ward']} ({r['flood_prob']:.0%})", axis=1)
                .tolist()
            )
            default_msg = (
                f"FLOOD ALERT: Nairobi Flood Guard\n"
                f"High-risk wards: {', '.join(top_wards) if top_wards else 'None at current threshold'}.\n"
                f"Threshold: {threshold:.0%}. Avoid low-lying areas & flooded routes.\n"
                f"Details: https://nairobi-flood-guard.streamlit.app"
            )
            sms_body = st.text_area(
                "Edit message before sending",
                value=default_msg,
                height=140,
            )
            char_count = len(sms_body)
            sms_pages = max(1, -(-char_count // 160))  # ceiling division
            st.caption(
                f"{char_count} characters · {sms_pages} SMS page(s) · "
                f"{len(recipient_list)} recipient(s)"
            )

            # ── Send button ──────────────────────────────────────────────────
            send_disabled = len(recipient_list) == 0 or len(sms_body.strip()) == 0
            if st.button(
                "📤 Send Alert",
                disabled=send_disabled,
                use_container_width=True,
            ):
                try:
                    at_username = st.secrets.get("AT_USERNAME", "NgundoMuithya")
                    at_api_key = st.secrets["AT_API_KEY"]

                    africastalking.initialize(
                        username=at_username,
                        api_key=at_api_key,
                    )
                    sms = africastalking.SMS

                    # Africa's Talking accepts a list of E.164 numbers
                    response = sms.send(
                        message=sms_body,
                        recipients=recipient_list,
                        # sender_id="FloodGuard",  # uncomment once AT approves
                    )

                    # Parse the response to show per-number status
                    results = response.get("SMSMessageData", {}).get("Recipients", [])
                    success = [r for r in results if r.get("status") == "Success"]
                    failed = [r for r in results if r.get("status") != "Success"]

                    if success:
                        st.success(f"✅ Sent to {len(success)} recipient(s).")
                    if failed:
                        st.warning(
                            "⚠️ Failed for:\n"
                            + "\n".join(
                                f"- {r.get('number')}: **{r.get('status')}**"
                                for r in failed
                            )
                        )
                        with st.expander("Raw AT response (debug)"):
                            st.json(response)

                    # Log to session state so operator can review sends
                    if "sms_log" not in st.session_state:
                        st.session_state.sms_log = []
                    st.session_state.sms_log.append(
                        {
                            "recipients": recipient_list,
                            "sent": len(success),
                            "failed": len(failed),
                            "message": sms_body[:80] + "...",
                        }
                    )

                except KeyError:
                    st.error(
                        "AT_API_KEY not found in Streamlit secrets.\n\n"
                        "Add it to `.streamlit/secrets.toml`:\n"
                        '```\nAT_USERNAME = "your_username"\nAT_API_KEY  = "your_key"\n```'
                    )
                except Exception as e:
                    st.error(f"SMS send failed: {e}")

            # ── Send log ─────────────────────────────────────────────────────
            if st.session_state.get("sms_log"):
                st.markdown("**Send log (this session)**")
                for entry in reversed(st.session_state.sms_log[-5:]):
                    st.caption(
                        f"✉ {entry['sent']} sent · {entry['failed']} failed: "
                        f"\"{entry['message']}\""
                    )

    st.markdown("---")
    rainfall_label = (
        rainfall_summary(rainfall_meta)
        if use_open_meteo
        else "Rainfall: CHIRPS Feb-Apr 2024"
    )
    st.markdown(
        f"<span style='font-size:0.65rem;color:#4E6357;'>Model: XGBoost · "
        f"Labels: UNOSAT Apr 2024 · Terrain: SRTM 90m · "
        f"{rainfall_label}</span>",
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 1 - FLOOD RISK DASHBOARD
# =============================================================================
if page == "Flood Risk Dashboard":

    if use_open_meteo:
        mode_label = (
            "Live mode" if use_live else f"{forecast_horizon_hours}hr prediction mode"
        )
        st.info(
            f"**{mode_label}**: predictions use rainfall features from "
            f"{rainfall_summary(rainfall_meta)}. "
            "Terrain features remain static (SRTM). "
            "Switch rainfall source in the sidebar to compare live, historical, "
            "24hr, and 48hr flood-risk maps."
        )

    st.markdown(
        '<div class="section-header">County Flood Risk Map</div>',
        unsafe_allow_html=True,
    )

    counties = sorted(df["county"].unique())
    default_idx = counties.index("Nairobi") if "Nairobi" in counties else 0
    selected_county = st.selectbox("Filter by county", counties, index=default_idx)

    map_df = df[df["county"] == selected_county]

    st.caption(f"{len(map_df)} wards · hover a ward for details")

    centre_lat = float(map_df.geometry.centroid.y.mean())
    centre_lon = float(map_df.geometry.centroid.x.mean())
    zoom = 7 if selected_county == "All Kenya" else 10

    with st.spinner("Rendering map..."):
        fmap = build_choropleth(map_df, centre_lat, centre_lon, zoom)
    st_folium(fmap, width="stretch", height=520)

    st.markdown(
        '<div class="section-header" style="margin-top:2rem">Flood Probability Distribution</div>',
        unsafe_allow_html=True,
    )
    fig = px.histogram(
        map_df,
        x="flood_prob",
        nbins=40,
        color_discrete_sequence=["#3FA66B"],
        labels={"flood_prob": "Flood Probability", "count": "Number of Wards"},
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#C4622D",
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_font_color="#C4622D",
        annotation_position="top right",
    )
    fig.update_layout(
        paper_bgcolor="#07110D",
        plot_bgcolor="#0E2318",
        font_color="#E8DFC8",
        font_family="Space Mono",
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(gridcolor="#1F4A32", tickformat=".0%"),
        yaxis=dict(gridcolor="#1F4A32"),
        bargap=0.05,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        '<div class="section-header">Highest Risk Wards</div>',
        unsafe_allow_html=True,
    )
    top10 = (
        map_df[["ward", "subcounty", "county", "flood_prob", "risk_label"]]
        .sort_values("flood_prob", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top10["flood_prob"] = top10["flood_prob"].map("{:.1%}".format)
    top10.index += 1
    st.dataframe(
        top10.rename(
            columns={
                "ward": "Ward",
                "subcounty": "Sub-County",
                "county": "County",
                "flood_prob": "Flood Probability",
                "risk_label": "Risk Level",
            }
        ),
        width="stretch",
    )


# =============================================================================
# PAGE 2 - WARD LOOKUP
# =============================================================================
elif page == "Ward Lookup":

    st.markdown(
        '<div class="section-header">Ward Flood Risk Lookup</div>',
        unsafe_allow_html=True,
    )

    ward_names = sorted(df["ward"].unique())
    selected_ward = st.selectbox("Search for a ward", ward_names)

    ward_row = df[df["ward"] == selected_ward].iloc[0]
    prob = float(ward_row["flood_prob"])
    label, badge_class = risk_label(prob)

    st.markdown(
        f"""
    <div class="ward-panel">
        <div class="ward-name">{selected_ward}</div>
        <div class="ward-meta">{ward_row['subcounty']} &nbsp;·&nbsp; {ward_row['county']}</div>
        <div style="margin-top:1rem; position:relative;">
            <span class="badge {badge_class}">{label} Risk</span>
            &nbsp;
            <span style="font-family:'Fraunces',serif;font-size:1.5rem;font-weight:600;
                         color:{risk_color(prob)}">{prob:.1%}</span>
            <span style="font-size:0.7rem;color:#8FA894;margin-left:0.3rem">flood probability</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Feature Breakdown")
    feature_labels = {
        "pop2009": ("Population (2009)", "people"),
        "elevation_mean_m": ("Mean Elevation", "m"),
        "elevation_min_m": ("Min Elevation", "m"),
        "elevation_max_m": ("Max Elevation", "m"),
        "elevation_range_m": ("Elevation Range", "m"),
        "slope_mean_deg": ("Mean Slope", "°"),
        "rain_cumulative_mm": ("Cumulative Rainfall (90d)", "mm"),
        "rain_max_daily_mm": ("Max Daily Rainfall", "mm"),
        "rain_preflood_7d_mm": ("Recent 7-Day Rainfall", "mm"),
    }
    cols = st.columns(4)
    for i, (col_name, (label_text, unit)) in enumerate(feature_labels.items()):
        val = ward_row[col_name]
        with cols[i % 4]:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">{label_text}</div>
                <div class="metric-value">{val:,.0f}<span class="metric-unit">{unit}</span></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("#### Ward vs. Kenya Average")
    features_for_radar = [
        "elevation_mean_m",
        "slope_mean_deg",
        "rain_cumulative_mm",
        "rain_max_daily_mm",
        "rain_preflood_7d_mm",
        "pop2009",
    ]
    radar_labels = [
        "Elevation",
        "Slope",
        "Cumul. Rain",
        "Max Daily Rain",
        "Pre-Flood Rain",
        "Population",
    ]

    ward_vals = [
        float(normalize(c, df)[df["ward"] == selected_ward].values[0])
        for c in features_for_radar
    ]
    avg_vals = [float(normalize(c, df).mean()) for c in features_for_radar]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=ward_vals + [ward_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=selected_ward,
            line_color="#D4A24C",
            fillcolor="rgba(212,162,76,0.2)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=avg_vals + [avg_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="Kenya Average",
            line_color="#3FA66B",
            fillcolor="rgba(63,166,107,0.1)",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="#0E2318",
            radialaxis=dict(visible=True, gridcolor="#1F4A32", color="#4E6357"),
            angularaxis=dict(gridcolor="#1F4A32", color="#8FA894"),
        ),
        paper_bgcolor="#07110D",
        font_color="#E8DFC8",
        font_family="Space Mono",
        legend=dict(bgcolor="#07110D", bordercolor="#1F4A32", borderwidth=1),
        margin=dict(t=30, b=30, l=30, r=30),
        height=380,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Ward Location")
    ward_lat = float(ward_row.geometry.centroid.y)
    ward_lon = float(ward_row.geometry.centroid.x)

    mini_map = folium.Map(
        location=[ward_lat, ward_lon], zoom_start=11, tiles="CartoDB dark_matter"
    )
    folium.GeoJson(
        ward_row.geometry.__geo_interface__,
        style_function=lambda _: {
            "fillColor": risk_color(prob),
            "fillOpacity": 0.55,
            "color": risk_color(prob),
            "weight": 2,
        },
        tooltip=f"{selected_ward} - {prob:.1%} flood probability",
    ).add_to(mini_map)
    st_folium(mini_map, width="stretch", height=320)

    # -- SHAP Feature Explanation ---------------------------------------------
    st.markdown("#### Why this flood risk score?")
    st.caption(
        "Each bar shows how much a feature pushed the flood probability "
        "up (positive) or down (negative) from the baseline."
    )

    ward_idx = df[df["ward"] == selected_ward].index[0]
    X_all = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model.named_steps["classifier"])
            shap_vals = explainer.shap_values(X_all)
            ward_shap = shap_vals[df.index.get_loc(ward_idx)]
            ward_data = X_all.loc[ward_idx]

            shap_df = pd.DataFrame(
                {
                    "Feature": FEATURE_COLS,
                    "SHAP Value": ward_shap,
                    "Feature Value": ward_data.values,
                }
            ).sort_values("SHAP Value")
            shap_df["Label"] = shap_df.apply(
                lambda r: f"{r['Feature'].replace('_', ' ').title()}  ({r['Feature Value']:,.1f})",
                axis=1,
            )
            shap_df["Color"] = shap_df["SHAP Value"].apply(
                lambda v: "#C4622D" if v > 0 else "#3FA66B"
            )

            fig_shap = go.Figure(
                go.Bar(
                    x=shap_df["SHAP Value"],
                    y=shap_df["Label"],
                    orientation="h",
                    marker_color=shap_df["Color"],
                    hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
                )
            )
            fig_shap.add_vline(x=0, line_color="#4E6357", line_width=1)
            fig_shap.update_layout(
                paper_bgcolor="#07110D",
                plot_bgcolor="#0E2318",
                font_color="#E8DFC8",
                font_family="Space Mono",
                xaxis=dict(
                    title="Impact on flood probability",
                    gridcolor="#1F4A32",
                    zerolinecolor="#4E6357",
                ),
                yaxis=dict(gridcolor="#1F4A32"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=360,
                showlegend=False,
            )
            st.plotly_chart(fig_shap, width="stretch")

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

    else:
        # Fallback: weighted feature importance bar chart (no shap needed)
        try:
            raw_imp = model.feature_importances_
        except AttributeError:
            raw_imp = np.ones(len(FEATURE_COLS))

        ward_data = X_all.loc[ward_idx]
        kenya_mean = X_all.mean()
        deviation = (ward_data - kenya_mean) / (X_all.std() + 1e-9)
        contribution = raw_imp * deviation.values

        imp_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLS,
                "Contribution": contribution,
                "Value": ward_data.values,
            }
        ).sort_values("Contribution")
        imp_df["Label"] = imp_df.apply(
            lambda r: f"{r['Feature'].replace('_', ' ').title()}  ({r['Value']:,.1f})",
            axis=1,
        )
        imp_df["Color"] = imp_df["Contribution"].apply(
            lambda v: "#C4622D" if v > 0 else "#3FA66B"
        )

        fig_imp = go.Figure(
            go.Bar(
                x=imp_df["Contribution"],
                y=imp_df["Label"],
                orientation="h",
                marker_color=imp_df["Color"],
                hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
            )
        )
        fig_imp.add_vline(x=0, line_color="#4E6357", line_width=1)
        fig_imp.update_layout(
            paper_bgcolor="#07110D",
            plot_bgcolor="#0E2318",
            font_color="#E8DFC8",
            font_family="Space Mono",
            xaxis=dict(
                title="Feature contribution (importance × deviation from Kenya avg)",
                gridcolor="#1F4A32",
            ),
            yaxis=dict(gridcolor="#1F4A32"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=360,
            showlegend=False,
        )
        st.plotly_chart(fig_imp, width="stretch")
        st.caption("Install `shap` for exact SHAP values: `pip install shap`")


# =============================================================================
# PAGE 3 - ROUTE OPTIMIZATION
# =============================================================================
elif page == "Route Optimization":

    st.markdown(
        '<div class="section-header">Matatu Route Optimization - April 2024 Flood Event</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "The route optimization system uses XGBoost flood predictions to identify which "
        "Nairobi matatu routes pass through high-risk wards. Flooded road segments are "
        "blocked outright and Dijkstra's algorithm finds the safest available alternative "
        "path for each affected route. The GTFS-RT feed generated is immediately "
        "consumable by transit apps."
    )

    routing_source = "historical"
    routing_meta: dict = {}
    routes, trips, shapes, stops, stop_times = load_gtfs()

    mode_label = (
        "live"
        if use_live
        else f"{forecast_horizon_hours}hr forecast" if use_open_meteo else "historical"
    )

    if use_open_meteo:
        if not OSMNX_AVAILABLE:
            st.warning(
                "osmnx isn't installed, so recomputed rerouting isn't "
                "available here. Falling back to the historical (Apr 2024) "
                "rerouting results below. Install with `pip install osmnx`."
            )
        else:
            st.info(
                f"**{mode_label.capitalize()} routing**: automatically "
                "recomputed using the current Nairobi flood risk from the "
                "sidebar. Results are cached by the underlying risk data, so "
                "this only takes a few seconds the first time risk actually "
                "changes; repeat visits with the same data are instant."
            )
            alpha_col, refresh_col = st.columns([3, 1])
            with alpha_col:
                alpha = st.slider(
                    "alpha (flood-cost multiplier)",
                    min_value=1.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0,
                    help="Higher alpha makes the algorithm avoid flooded "
                    "roads more aggressively, at the cost of longer detours.",
                )
            with refresh_col:
                st.write("")
                if st.button("Force refresh", use_container_width=True):
                    get_live_routes.clear()

            with st.spinner(
                "Loading road network & running flood-weighted Dijkstra..."
            ):
                try:
                    G = load_road_graph()
                    fingerprint = flood_prob_fingerprint(
                        nairobi, alpha, threshold, forecast_horizon_hours
                    )
                    rerouting_df, route_geoms, routing_meta = get_live_routes(
                        G,
                        nairobi,
                        stops,
                        stop_times,
                        trips,
                        alpha,
                        threshold,
                        fingerprint,
                    )
                    routing_source = "live"
                    routing_meta["mode_label"] = mode_label
                except Exception as exc:
                    st.warning(
                        f"Recomputing {mode_label} routing failed ({exc}). "
                        "Falling back to historical results."
                    )

    if routing_source == "historical":
        if not REROUTING_CSV.exists():
            st.info(
                "No historical rerouting data on disk yet. "
                + (
                    "Live/forecast routing was attempted automatically above; "
                    "if it also failed, check the warning shown."
                    if use_open_meteo and OSMNX_AVAILABLE
                    else "Run Route_Optimization/route_optimization.ipynb first."
                )
            )
            st.stop()
        rerouting_df = load_rerouting()
        route_geoms = load_route_geometries()

    if routing_source == "live":
        st.success(
            f"Showing **{routing_meta.get('mode_label', mode_label)}** rerouting "
            f"(alpha={routing_meta['alpha']:.0f}, threshold={routing_meta['threshold']:.2f}) · "
            f"{routing_meta['rerouted_routes']} of "
            f"{routing_meta['total_affected_routes']} affected routes rerouted."
        )
    else:
        st.caption(
            "Showing **historical** rerouting results from the April 2024 "
            "flood event (Route_Optimization/route_optimization.ipynb)."
        )

    if rerouting_df.empty:
        st.success(
            "No routes currently need rerouting. Flood risk is below the "
            f"{threshold:.2f} threshold across all monitored wards."
        )
        st.stop()

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Affected Routes", len(rerouting_df))
    with c2:
        st.metric("Avg Risk Reduction", f"{rerouting_df['risk_reduction'].mean():.3f}")
    with c3:
        st.metric("Avg Extra Time", f"{rerouting_df['extra_time_min'].mean():.1f} min")
    with c4:
        st.metric("Routes Improved", int((rerouting_df["risk_reduction"] > 0).sum()))

    # Rerouting summary table
    st.markdown(
        '<div class="section-header" style="margin-top:1.5rem">Rerouting Summary</div>',
        unsafe_allow_html=True,
    )
    sort_col = st.selectbox(
        "Sort by",
        [
            "risk_reduction",
            "extra_time_min",
            "original_flood_prob",
            "alternative_flood_prob",
        ],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    display_df = (
        rerouting_df[
            [
                "route_id",
                "origin",
                "destination",
                "original_flood_prob",
                "alternative_flood_prob",
                "risk_reduction",
                "extra_time_min",
            ]
        ]
        .sort_values(sort_col, ascending=False)
        .reset_index(drop=True)
    )
    display_df.index += 1
    display_df.columns = [
        "Route ID",
        "Origin",
        "Destination",
        "Original Risk",
        "Alternative Risk",
        "Risk Reduction",
        "Extra Time (min)",
    ]
    st.dataframe(display_df, width="stretch")
    st.download_button(
        label="⬇ Download Rerouting CSV",
        data=rerouting_df.to_csv(index=False),
        file_name="rerouting_summary.csv",
        mime="text/csv",
    )

    # Tradeoff chart
    st.markdown(
        '<div class="section-header" style="margin-top:1.5rem">Risk-Time Tradeoff</div>',
        unsafe_allow_html=True,
    )
    if routing_source == "live":
        # Build fresh so it reflects the live results just computed, rather
        # than the static image saved from the April 2024 notebook run.
        fig_tradeoff = px.scatter(
            rerouting_df,
            x="extra_time_min",
            y="risk_reduction",
            hover_data=["route_id", "origin", "destination"],
            labels={
                "extra_time_min": "Extra Travel Time (minutes)",
                "risk_reduction": "Flood Risk Reduction",
            },
            color_discrete_sequence=["#D4A24C"],
        )
        fig_tradeoff.update_traces(marker=dict(size=9, opacity=0.8))
        fig_tradeoff.add_hline(y=0, line_dash="dash", line_color="#4E6357")
        fig_tradeoff.add_vline(x=0, line_dash="dash", line_color="#4E6357")
        fig_tradeoff.update_layout(
            paper_bgcolor="#07110D",
            plot_bgcolor="#0E2318",
            font_color="#E8DFC8",
            font_family="Space Mono",
            xaxis=dict(gridcolor="#1F4A32"),
            yaxis=dict(gridcolor="#1F4A32"),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_tradeoff, width="stretch")
    elif TRADEOFF_PNG.exists():
        st.image(str(TRADEOFF_PNG), width="stretch")
    else:
        st.info(
            "Tradeoff chart not found. Run route_optimization.ipynb to generate it."
        )

    # Interactive map section
    st.markdown(
        '<div class="section-header" style="margin-top:1.5rem">Interactive Map</div>',
        unsafe_allow_html=True,
    )
    map_view = st.radio(
        "View",
        ["Flood Risk Map", "Route Explorer"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if map_view == "Flood Risk Map":
        st.caption("Nairobi ward flood risk · hover a ward for details")
        with st.spinner("Rendering flood risk map..."):
            risk_map = build_choropleth(nairobi, NAIROBI_LAT, NAIROBI_LON, zoom=11)
        st_folium(risk_map, width="stretch", height=520)

    else:
        affected_stop_ids = get_affected_stop_ids(nairobi, stops, threshold)
        affected_route_ids = rerouting_df["route_id"].tolist()
        n_routes = len(affected_route_ids)

        if "route_idx" not in st.session_state:
            st.session_state.route_idx = 0

        nav_left, nav_centre, nav_right = st.columns([1, 4, 1])

        with nav_left:
            if st.button("← Previous", width="stretch"):
                st.session_state.route_idx = (st.session_state.route_idx - 1) % n_routes

        with nav_right:
            if st.button("Next →", width="stretch"):
                st.session_state.route_idx = (st.session_state.route_idx + 1) % n_routes

        idx = st.session_state.route_idx
        route_row = rerouting_df.iloc[idx]
        route_id = route_row["route_id"]

        with nav_centre:
            st.markdown(
                f"<div style='text-align:center;padding:0.4rem 0;'>"
                f"<span style='font-family:Fraunces,serif;font-size:1.05rem;"
                f"font-weight:600;color:#E8DFC8;'>Route {route_id}</span>"
                f"<span style='font-size:0.72rem;color:#8FA894;margin-left:0.6rem;'>"
                f"{route_row['origin']} → {route_row['destination']}</span>"
                f"<span style='font-size:0.65rem;color:#4E6357;margin-left:0.6rem;'>"
                f"({idx + 1} / {n_routes})</span></div>",
                unsafe_allow_html=True,
            )

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f"""<div class="route-stat-card">
                <div class="route-stat-label">Original Flood Risk</div>
                <div class="route-stat-value" style="color:#C4622D;">
                    {route_row['original_flood_prob']:.1%}
                </div></div>""",
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f"""<div class="route-stat-card">
                <div class="route-stat-label">Alternative Flood Risk</div>
                <div class="route-stat-value" style="color:#3FA66B;">
                    {route_row['alternative_flood_prob']:.1%}
                </div></div>""",
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f"""<div class="route-stat-card">
                <div class="route-stat-label">Risk Reduction</div>
                <div class="route-stat-value" style="color:#D4A24C;">
                    {route_row['risk_reduction']:.3f}
                </div></div>""",
                unsafe_allow_html=True,
            )
        with s4:
            st.markdown(
                f"""<div class="route-stat-card">
                <div class="route-stat-label">Extra Travel Time</div>
                <div class="route-stat-value" style="color:#5FA8C4;">
                    +{route_row['extra_time_min']:.1f} min
                </div></div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
        route_view = st.radio(
            "Route view",
            ["Original Route", "Alternative Route"],
            horizontal=True,
            label_visibility="collapsed",
        )

        route_map = folium.Map(
            location=[NAIROBI_LAT, NAIROBI_LON],
            zoom_start=12,
            tiles="CartoDB dark_matter",
        )
        route_coords = get_route_shape(route_id, trips, shapes)
        route_stops = get_route_stops(route_id, trips, stop_times, stops)

        if route_view == "Original Route":
            st.caption(
                "🔵 Original route path · 🔴 Affected stops (in flood-risk wards) · "
                "⚪ Safe stops"
            )
            if route_coords:
                folium.PolyLine(
                    route_coords,
                    color="#378ADD",
                    weight=4,
                    opacity=0.9,
                    tooltip=f"Route {route_id} - Original",
                ).add_to(route_map)
            if not route_stops.empty:
                for _, stop_row in route_stops.iterrows():
                    is_affected = stop_row["stop_id"] in affected_stop_ids
                    folium.CircleMarker(
                        location=[stop_row["stop_lat"], stop_row["stop_lon"]],
                        radius=5 if is_affected else 3,
                        color="#C4622D" if is_affected else "#4E6357",
                        fill=True,
                        fill_color="#C4622D" if is_affected else "#4E6357",
                        fill_opacity=0.9,
                        tooltip=(
                            f"⚠ Affected: {stop_row.get('stop_name', stop_row['stop_id'])}"
                            if is_affected
                            else str(stop_row.get("stop_name", stop_row["stop_id"]))
                        ),
                    ).add_to(route_map)
            if route_coords:
                lats = [c[0] for c in route_coords]
                lons = [c[1] for c in route_coords]
                route_map.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        else:
            alt_coords = route_geoms.get(str(route_id), {}).get("alternative", [])
            st.caption(
                "🟡 Alternative route (Dijkstra, flood-weighted) · "
                "🟢 Original route (faded reference) · "
                "🔴 Affected stops skipped · "
                "risk reduced by {:.3f} · +{:.1f} min".format(
                    route_row["risk_reduction"], route_row["extra_time_min"]
                )
            )
            if route_coords:
                folium.PolyLine(
                    route_coords,
                    color="#2E5C42",
                    weight=3,
                    opacity=0.5,
                    tooltip=f"Route {route_id} - Original (reference)",
                    dash_array="6",
                ).add_to(route_map)
            if alt_coords:
                folium.PolyLine(
                    alt_coords,
                    color="#D4A24C",
                    weight=4,
                    opacity=0.95,
                    dash_array="8",
                    tooltip=f"Route {route_id} - Alternative",
                ).add_to(route_map)
            else:
                st.warning(
                    "Alternative path geometry not found. "
                    "Re-run route_optimization.ipynb."
                )
            if not route_stops.empty:
                for _, stop_row in route_stops.iterrows():
                    is_affected = stop_row["stop_id"] in affected_stop_ids
                    folium.CircleMarker(
                        location=[stop_row["stop_lat"], stop_row["stop_lon"]],
                        radius=5 if is_affected else 3,
                        color="#C4622D" if is_affected else "#2E4038",
                        fill=True,
                        fill_color="#C4622D" if is_affected else "#2E4038",
                        fill_opacity=0.85,
                        tooltip=(
                            f"🚫 Skipped: {stop_row.get('stop_name', stop_row['stop_id'])}"
                            if is_affected
                            else str(stop_row.get("stop_name", stop_row["stop_id"]))
                        ),
                    ).add_to(route_map)
            coords_for_bounds = alt_coords if alt_coords else route_coords
            if coords_for_bounds:
                lats = [c[0] for c in coords_for_bounds]
                lons = [c[1] for c in coords_for_bounds]
                route_map.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
            if alt_coords:
                mid = alt_coords[len(alt_coords) // 2]
                folium.Marker(
                    location=mid,
                    icon=folium.DivIcon(
                        html=(
                            f"<div style='background:#0E2318;border:1px solid #D4A24C;"
                            f"border-radius:4px;padding:4px 8px;font-family:monospace;"
                            f"font-size:11px;color:#E8DFC8;white-space:nowrap;'>"
                            f"Alternative · Risk ↓{route_row['risk_reduction']:.3f}"
                            f" · +{route_row['extra_time_min']:.1f} min</div>"
                        ),
                        icon_size=(270, 30),
                        icon_anchor=(135, 15),
                    ),
                ).add_to(route_map)

        st_folium(route_map, width="stretch", height=500)


# =============================================================================
# PAGE 4 - AI ASSISTANT (Mlinzi)
# =============================================================================
elif page == "AI Assistant":

    st.markdown(
        '<div class="section-header">Flood Guard AI Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Hi. My name is Mlinzi, an AI Chatbot specifically designed to help you "
        "with any questions you might have regarding flooding in Kenya. "
        "Ask anything about flood risk, affected wards, route recommendations, "
        "or how to interpret the model results."
    )

    # Input form at the top
    with st.form(key="chat_form", clear_on_submit=True):
        input_col, btn_col = st.columns([11, 1])
        with input_col:
            user_input = st.text_input(
                "",
                placeholder="Ask about flood risk, routes, or the model...",
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button("➤")

    st.markdown(
        "<hr style='border:none;border-top:1px solid #1F4A32;margin:0.5rem 0 1rem 0;'>",
        unsafe_allow_html=True,
    )

    # Process new input
    if submitted and user_input.strip():
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append(
            {"role": "user", "content": user_input.strip()}
        )
        with st.spinner("Mlinzi is thinking..."):
            # Context is built here, only when a message is actually being
            # sent to the LLM, rather than on every rerun of this page
            # (typing, clicking elsewhere, clearing the conversation, etc.
            # all trigger a rerun and previously rebuilt this unconditionally).
            wards_context = (
                df[df["county"].str.lower().str.strip() == "nairobi"][
                    [
                        "ward",
                        "subcounty",
                        "county",
                        "flood_prob",
                        "risk_label",
                        "elevation_mean_m",
                        "elevation_min_m",
                        "elevation_max_m",
                        "slope_mean_deg",
                        "rain_cumulative_mm",
                        "rain_max_daily_mm",
                        "rain_preflood_7d_mm",
                        "pop2009",
                    ]
                ]
                .sort_values("flood_prob", ascending=False)
                .head(100)
                .assign(flood_prob=lambda x: x["flood_prob"].map("{:.1%}".format))
                .to_string(index=False)
            )

            model_perf_context = ""
            model_csv = DATA / "model_comparison.csv"
            if model_csv.exists():
                model_perf_context = (
                    "\nModel Performance Comparison:\n"
                    + load_model_comparison(model_csv).to_string(index=False)
                )

            rerouting_context = ""
            if REROUTING_CSV.exists():
                r = load_rerouting()
                rerouting_context = (
                    f"\nFull Rerouting Summary ({len(r)} affected routes):\n"
                    + r.to_string(index=False)
                    + "\n\nAggregate stats:"
                    + f"\n  Average risk reduction : {r['risk_reduction'].mean():.3f}"
                    + f"\n  Average extra time (min): {r['extra_time_min'].mean():.1f}"
                    + f"\n  Routes with risk > 0   : {(r['risk_reduction'] > 0).sum()}"
                )

            system_prompt = (
                "You are Mlinzi, an AI assistant for Nairobi Flood Guard - a data science project\n"
                "that predicts flood susceptibility across Kenya's 1,450 administrative wards and\n"
                "recommends alternative matatu routes during flood events. Your name means\n"
                "'guardian' or 'protector' in Swahili, which reflects your purpose.\n\n"
                "--- PROJECT OVERVIEW ---\n"
                "The prediction model is XGBoost trained on the following features:\n"
                "  Terrain  : elevation (mean, min, max, range in metres), slope (degrees)\n"
                "  Rainfall : cumulative 90-day (mm), max single-day (mm), recent 7-day (mm)\n"
                "  Population: 2009 Kenya census ward population\n\n"
                f"--- ACTIVE RAINFALL SOURCE ---\n"
                f"{rainfall_summary(rainfall_meta) if use_open_meteo else 'Historical CHIRPS data from the April 2024 flood event.'}\n\n"
                "Key insight: flooding in Kenya is primarily terrain-driven at ward scale.\n"
                "Low-lying wards flood not because they receive more rain but because water drains\n"
                "into them from surrounding higher ground. Elevation features dominate predictions;\n"
                "rainfall adds marginal predictive value at this spatial resolution.\n\n"
                "--- CURRENT RISK SUMMARY ---\n"
                f"Total wards         : {len(df)}\n"
                f"High-risk (>= {threshold:.0%}) : {int(n_total)}\n"
                f"Critical risk (>= 70%): {int(n_critical)}\n\n"
                "Risk thresholds:\n"
                "  Low      : flood probability < 20%\n"
                "  Moderate : 20% - 45%\n"
                "  High     : 45% - 70%\n"
                "  Critical : >= 70%\n\n"
                "--- ALL NAIROBI DATA (top 100 by flood probability) ---\n"
                f"{wards_context}\n\n"
                "--- MODEL PERFORMANCE ---\n"
                f"{model_perf_context if model_perf_context else 'Model comparison data not available.'}\n\n"
                "--- ROUTE OPTIMIZATION ---\n"
                "The route optimization system:\n"
                "  - Uses XGBoost flood probabilities to identify high-risk road segments\n"
                "  - Assigns flood cost = travel_time x (1 + alpha x flood_probability), "
                "alpha = 10\n"
                "    (effectively blocking all flood-affected roads outright)\n"
                "  - Runs weighted Dijkstra to find the safest alternative path\n"
                "  - Outputs a GTFS-RT feed consumable by transit apps\n\n"
                f"{rerouting_context if rerouting_context else 'Rerouting data not available.'}\n\n"
                "--- INSTRUCTIONS ---\n"
                "- Be concise, factual, and actionable.\n"
                "- When asked about a specific ward, look it up in the ward data above and quote\n"
                "  its exact flood probability, risk level, and key terrain/rainfall figures.\n"
                "- When asked about routes, reference the rerouting summary above by route ID.\n"
                "- If asked which wards are most at risk, list the top entries from the ward data.\n"
                "- If asked about the model, explain the XGBoost pipeline and feature importance.\n"
                "- Do not make up data - all ward and route figures are provided above.\n"
                "- Respond in English unless the user writes in another language.\n"
            )

            client = get_groq_client()
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *st.session_state.messages,
                ],
                max_tokens=1024,
                temperature=0.4,
            )
            reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})

    # Render message history below the input
    for msg in st.session_state.get("messages", []):
        render_message(msg["role"], msg["content"])

    if st.session_state.get("messages"):
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()
