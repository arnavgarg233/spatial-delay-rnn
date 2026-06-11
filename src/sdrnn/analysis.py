"""Self-organization metrics: does learnable movement produce functional regions?

Tests whether functionally similar units (same recurrent-weight community) end
up spatially close, i.e. region-like organization. NumPy in / float (or label
array) out, mirroring ``graph_metrics``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from sdrnn.graph_metrics import symmetrize, threshold_proportional

try:
    import community as community_louvain  # type: ignore
    import networkx as nx  # type: ignore
    _HAVE_GRAPH = True
except Exception:  # pragma: no cover
    _HAVE_GRAPH = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


def functional_communities(weight: np.ndarray, density: float = 0.1) -> np.ndarray:
    """Louvain community label per neuron from the binarized weight graph.

    These are the network's *functional* modules - groups of units densely
    interconnected in the recurrent matrix, independent of physical position.
    """
    if not _HAVE_GRAPH:
        raise ImportError("functional_communities needs networkx + python-louvain")
    a = symmetrize(weight)
    a = threshold_proportional(a, density)
    a_bin = (a > 0).astype(float)
    g = nx.from_numpy_array(a_bin)
    part = community_louvain.best_partition(g, weight="weight", random_state=0)
    return np.array([part[i] for i in range(a.shape[0])])


def coord_displacement(final_coords: np.ndarray, init_coords: np.ndarray) -> Dict[str, float]:
    """How far neurons moved from their initial positions (per-neuron L2)."""
    d = np.linalg.norm(np.asarray(final_coords) - np.asarray(init_coords), axis=1)
    return {"mean": float(d.mean()), "max": float(d.max()), "std": float(d.std())}


def coord_cluster_quality(coords: np.ndarray, k_range=range(2, 9)) -> Dict[str, float]:
    """Best-k silhouette of the coordinate cloud - did neurons form clusters?

    Returns the best silhouette and its k. High silhouette = neurons collapsed
    into discrete spatial groups (region-like); near 0 = still a diffuse cloud.
    """
    if not _HAVE_SK:
        raise ImportError("coord_cluster_quality needs scikit-learn")
    coords = np.asarray(coords)
    best = {"silhouette": -1.0, "k": 0}
    for k in k_range:
        if k >= len(coords):
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(coords)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(coords, labels)
        if s > best["silhouette"]:
            best = {"silhouette": float(s), "k": int(k)}
    return best


def regionalization(weight: np.ndarray, coords: np.ndarray, density: float = 0.1) -> float:
    """AMI between spatial clusters and functional communities.

    Cluster neurons by position (k-means at the functional community count) and
    compare to the functional Louvain partition via adjusted mutual information.
    ~0 = position says nothing about function; ->1 = functionally-similar units
    are co-located = region-like organization.
    """
    if not (_HAVE_GRAPH and _HAVE_SK):
        raise ImportError("regionalization needs networkx + python-louvain + sklearn")
    comm = functional_communities(weight, density=density)
    # Exclude singleton communities (isolated nodes) when choosing k - they
    # inflate the community count and bias the k-means partition.
    _, counts = np.unique(comm, return_counts=True)
    k = int((counts >= 2).sum())
    if k < 2:
        return float("nan")
    spatial = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(np.asarray(coords))
    return float(adjusted_mutual_info_score(comm, spatial))


def community_spatial_compactness(
    weight: np.ndarray, coords: np.ndarray, density: float = 0.1
) -> float:
    """Mean within-community / between-community neuron distance.

    <1 means functional communities are spatially compact (their members sit
    close together). A purely positional restatement of regionalization that
    does not depend on k-means, as a cross-check.
    """
    if not _HAVE_GRAPH:
        raise ImportError("community_spatial_compactness needs networkx + python-louvain")
    comm = functional_communities(weight, density=density)
    coords = np.asarray(coords)
    n = len(coords)
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            (within if comm[i] == comm[j] else between).append(d)
    if not within or not between:
        return float("nan")
    return float(np.mean(within) / np.mean(between))


def analyze_geometry(
    weight: np.ndarray,
    final_coords: np.ndarray,
    init_coords: Optional[np.ndarray] = None,
    density: float = 0.1,
) -> Dict[str, float]:
    """One-call self-organization summary for a (trained, movable) network."""
    out: Dict[str, float] = {}
    out["regionalization_ami"] = regionalization(weight, final_coords, density=density)
    out["community_compactness"] = community_spatial_compactness(weight, final_coords, density=density)
    cq = coord_cluster_quality(final_coords)
    out["coord_silhouette"] = cq["silhouette"]
    out["coord_best_k"] = float(cq["k"])
    if init_coords is not None:
        disp = coord_displacement(final_coords, init_coords)
        out["displacement_mean"] = disp["mean"]
        out["displacement_max"] = disp["max"]
    return out
