"""
Live flood-aware matatu rerouting for Nairobi Flood Guard.

Ports the pipeline from Route_Optimization/route_optimization.ipynb into
callable functions, so app.py can rerun weighted Dijkstra against *current*
ward flood_prob values instead of only displaying a rerouting_summary.csv
frozen at the April 2024 event.

Pipeline (unchanged from the notebook):
    ward flood_prob
        -> spatial join onto OSMnx road edges
        -> flood_cost = travel_time * (1 + alpha * flood_prob)
        -> weighted Dijkstra per affected route (terminal-to-terminal)
        -> summary dataframe + path geometries

Notes on cost / runtime:
    - The graph is large (Nairobi: ~87k nodes / ~213k edges). Loading it is
      the expensive part - always cache with st.cache_resource, never reload
      per rerun.
    - Rerunning Dijkstra for every affected route (~dozens) typically takes a
      few seconds, not milliseconds. Callers should gate this behind an
      explicit "recompute" action (button), not run it on every Streamlit
      rerun / widget tweak.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

WGS84 = "EPSG:4326"


def load_road_graph(graphml_path) -> nx.MultiDiGraph:
    """Load the OSMnx road network. Callers should wrap this in
    st.cache_resource - it's ~87k nodes / ~213k edges and shouldn't be
    reloaded on every script rerun."""
    return ox.load_graphml(graphml_path)


def compute_edge_flood_map(
    G: nx.MultiDiGraph, wards_gdf: gpd.GeoDataFrame
) -> dict[tuple[Any, Any, int], float]:
    """Assign each road edge the flood_prob of the ward its midpoint falls in.
    Edges outside any ward default to 0.0."""
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)[["geometry"]].reset_index()
    edges_gdf["midpoint"] = edges_gdf["geometry"].interpolate(0.5, normalized=True)
    edges_mid = gpd.GeoDataFrame(
        edges_gdf[["u", "v", "key", "midpoint"]], geometry="midpoint", crs=WGS84
    )

    edges_joined = gpd.sjoin(
        edges_mid,
        wards_gdf[["flood_prob", "ward", "geometry"]],
        how="left",
        predicate="within",
    )
    edges_joined = (
        edges_joined.groupby(["u", "v", "key"])["flood_prob"].max().reset_index()
    )
    edges_joined["flood_prob"] = edges_joined["flood_prob"].fillna(0.0)

    return {
        (row.u, row.v, row.key): row.flood_prob for row in edges_joined.itertuples()
    }


def build_flood_weighted_graph(
    G: nx.MultiDiGraph,
    flood_prob_map: dict[tuple[Any, Any, int], float],
    alpha: float,
) -> nx.MultiDiGraph:
    """Return a COPY of G with a 'flood_cost' attribute on every edge.
    Copies rather than mutates so a cached base graph can be reused across
    different alpha/threshold values without cross-contamination."""
    G = G.copy()
    flood_costs = {
        (u, v, key): data.get("travel_time", 60)
        * (1 + alpha * flood_prob_map.get((u, v, key), 0.0))
        for u, v, key, data in G.edges(keys=True, data=True)
    }
    nx.set_edge_attributes(G, flood_costs, "flood_cost")
    return G


def _get_route_terminals(
    route_id: str, trips: pd.DataFrame, stop_times: pd.DataFrame, stops: pd.DataFrame
) -> tuple:
    trip_id = trips[trips["route_id"] == route_id]["trip_id"].iloc[0]
    route_stops = (
        stop_times[stop_times["trip_id"] == trip_id]
        .sort_values("stop_sequence")
        .merge(stops, on="stop_id")
    )
    origin = route_stops.iloc[0]
    destination = route_stops.iloc[-1]
    return (
        (origin["stop_lat"], origin["stop_lon"], origin["stop_name"]),
        (destination["stop_lat"], destination["stop_lon"], destination["stop_name"]),
    )


def find_alternative_route(
    route_id: str,
    G: nx.MultiDiGraph,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    stops: pd.DataFrame,
    flood_prob_map: dict[tuple[Any, Any, int], float],
    orig_node: Any = None,
    dest_node: Any = None,
    terminals: tuple | None = None,
) -> dict | None:
    """Find the flood-cost-minimizing path between a route's terminal stops.
    Returns None if terminals can't be resolved or no path exists.

    `orig_node`/`dest_node`/`terminals` let a caller (run_live_rerouting)
    pass in values it already resolved in a batch, avoiding rebuilding
    OSMnx's spatial index once per route. Left as None, behavior is
    identical to the original single-route lookup."""
    try:
        if terminals is not None:
            origin, destination = terminals
        else:
            origin, destination = _get_route_terminals(
                route_id, trips, stop_times, stops
            )

        if orig_node is None:
            orig_node = ox.nearest_nodes(G, X=origin[1], Y=origin[0])
        if dest_node is None:
            dest_node = ox.nearest_nodes(G, X=destination[1], Y=destination[0])

        original_path = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
        alternative_path = nx.shortest_path(
            G, orig_node, dest_node, weight="flood_cost"
        )

        def _best_parallel_edge(u, v, weight_key):
            """G is a MultiDiGraph: (u, v) can have multiple parallel edges
            (keys). nx.shortest_path picks whichever parallel edge is
            cheapest at each hop but doesn't report which key it used, so
            reconstruct it here rather than assuming key 0 - which silently
            misreports metrics (or KeyErrors and drops the whole route) on
            any node pair with more than one edge between them."""
            edges = G[u][v]
            best_key = min(edges, key=lambda k: edges[k].get(weight_key, 0))
            return best_key, edges[best_key]

        def path_metrics(path, weight_key):
            total_time = 0.0
            probs = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key, edge_data = _best_parallel_edge(u, v, weight_key)
                total_time += edge_data.get("travel_time", 0)
                probs.append(flood_prob_map.get((u, v, key), 0.0))
            return total_time, float(np.mean(probs)) if probs else 0.0

        orig_time, orig_flood = path_metrics(original_path, "travel_time")
        alt_time, alt_flood = path_metrics(alternative_path, "flood_cost")

        return {
            "route_id": route_id,
            "origin": origin[2],
            "destination": destination[2],
            "original_path": original_path,
            "alternative_path": alternative_path,
            "original_time_s": orig_time,
            "alternative_time_s": alt_time,
            "extra_time_min": round((alt_time - orig_time) / 60, 1),
            "original_flood_prob": round(float(orig_flood), 3),
            "alternative_flood_prob": round(float(alt_flood), 3),
            "risk_reduction": round(float(orig_flood - alt_flood), 3),
        }
    except Exception:
        return None


def compute_affected_routes(
    wards_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    threshold: float,
) -> tuple[list[str], set[str]]:
    """Which routes serve a stop inside a high-risk ward, given current
    flood_prob and threshold."""
    high_risk_wards = wards_gdf[wards_gdf["flood_prob"] >= threshold]
    if high_risk_wards.empty:
        return [], set()
    stops_joined = gpd.sjoin(
        stops_gdf,
        high_risk_wards[["ward", "flood_prob", "geometry"]],
        how="left",
        predicate="within",
    )
    affected_stops = set(
        stops_joined[stops_joined["flood_prob"].notna()]["stop_id"].tolist()
    )

    affected_trip_ids = stop_times[stop_times["stop_id"].isin(affected_stops)][
        "trip_id"
    ].unique()
    affected_route_ids = (
        trips[trips["trip_id"].isin(affected_trip_ids)]["route_id"].unique().tolist()
    )
    return affected_route_ids, affected_stops


def _path_to_coords(path: list, G: nx.MultiDiGraph) -> list:
    return [[G.nodes[node]["y"], G.nodes[node]["x"]] for node in path]


def run_live_rerouting(
    G: nx.MultiDiGraph,
    wards_gdf: gpd.GeoDataFrame,
    stops_df: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    alpha: float = 10,
    threshold: float = 0.45,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Full pipeline: current ward flood_prob -> flood-weighted graph ->
    affected routes -> weighted Dijkstra per route.

    Returns (results_df, route_geometries, meta) matching the shape of the
    notebook's rerouting_summary.csv / route_geometries.json, so the same
    Streamlit UI code can render either.
    """
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
        crs=WGS84,
    )

    flood_prob_map = compute_edge_flood_map(G, wards_gdf)
    G_weighted = build_flood_weighted_graph(G, flood_prob_map, alpha)

    affected_route_ids, affected_stops = compute_affected_routes(
        wards_gdf, stops_gdf, stop_times, trips, threshold
    )

    # Resolve every affected route's terminals up front, then look up
    # nearest road-network nodes for all of them in a single batched call.
    # ox.nearest_nodes rebuilds a spatial index over the graph's ~87k nodes
    # on every call, so doing this once instead of twice per route is the
    # difference between one index build and ~2N of them.
    terminals_by_route: dict[str, tuple] = {}
    for route_id in affected_route_ids:
        try:
            terminals_by_route[route_id] = _get_route_terminals(
                route_id, trips, stop_times, stops_df
            )
        except Exception:
            continue  # this route is skipped below, same as pre-batching behavior

    node_lookup: dict[str, tuple] = {}
    if terminals_by_route:
        route_order = list(terminals_by_route)
        lats, lons = [], []
        for route_id in route_order:
            origin, destination = terminals_by_route[route_id]
            lats += [origin[0], destination[0]]
            lons += [origin[1], destination[1]]
        try:
            nearest = ox.nearest_nodes(G_weighted, X=lons, Y=lats)
            for i, route_id in enumerate(route_order):
                node_lookup[route_id] = (nearest[2 * i], nearest[2 * i + 1])
        except Exception:
            # A single malformed coordinate can fail a batched lookup outright.
            # Fall back to resolving one route at a time so that one bad
            # coordinate only drops that route, not every affected route.
            for route_id in route_order:
                origin, destination = terminals_by_route[route_id]
                try:
                    node_lookup[route_id] = (
                        ox.nearest_nodes(G_weighted, X=origin[1], Y=origin[0]),
                        ox.nearest_nodes(
                            G_weighted, X=destination[1], Y=destination[0]
                        ),
                    )
                except Exception:
                    pass  # this route is skipped below, same as pre-batching behavior

    results = []
    for route_id in affected_route_ids:
        if route_id not in node_lookup:
            continue
        orig_node, dest_node = node_lookup[route_id]
        result = find_alternative_route(
            route_id,
            G_weighted,
            trips,
            stop_times,
            stops_df,
            flood_prob_map,
            orig_node=orig_node,
            dest_node=dest_node,
            terminals=terminals_by_route[route_id],
        )
        if result:
            results.append(result)

    RESULT_COLS = [
        "route_id",
        "origin",
        "destination",
        "original_path",
        "alternative_path",
        "original_time_s",
        "alternative_time_s",
        "extra_time_min",
        "original_flood_prob",
        "alternative_flood_prob",
        "risk_reduction",
    ]
    results_df = pd.DataFrame(results, columns=RESULT_COLS)

    route_geometries = {
        str(r["route_id"]): {
            "original": _path_to_coords(r["original_path"], G_weighted),
            "alternative": _path_to_coords(r["alternative_path"], G_weighted),
        }
        for r in results
    }

    meta = {
        "alpha": alpha,
        "threshold": threshold,
        "total_affected_routes": len(affected_route_ids),
        "rerouted_routes": len(results_df),
        "affected_stops": len(affected_stops),
    }

    return results_df, route_geometries, meta
