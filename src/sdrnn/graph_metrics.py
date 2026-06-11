"""Graph metrics for brain-like structure: modularity, small-worldness, path
length, and spatial homophily. Self-contained, NumPy in / dict out.

Conventions:
* Input is a weighted adjacency W (N x N, non-negative; pass |W_rec|).
* The recurrent matrix is directed and signed; structural metrics are defined on
  undirected non-negative graphs, so we analyze the symmetrized magnitude
  A = (|W| + |W|^T) / 2. Every variant in a sweep gets the same transform.
* Self-loops removed (diagonal zeroed) before any metric.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:  # community detection (python-louvain); optional at import time
    import community as community_louvain  # type: ignore
    _HAVE_LOUVAIN = True
except Exception:  # pragma: no cover
    _HAVE_LOUVAIN = False

try:
    import networkx as nx  # type: ignore
    _HAVE_NX = True
except Exception:  # pragma: no cover
    _HAVE_NX = False


# -- preprocessing ---------------------------------------------------------
def symmetrize(weight: np.ndarray) -> np.ndarray:
    """Undirected non-negative adjacency: magnitude, symmetrized, no self-loops."""
    a = np.abs(np.asarray(weight, dtype=float))
    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 0.0)
    return a


def threshold_proportional(adj: np.ndarray, density: float) -> np.ndarray:
    """Keep the strongest ``density`` fraction of edges, zero the rest.

    Several metrics (path length, small-worldness) are only well-behaved on a
    sparse graph, and comparisons must be at *matched density* or differences in
    overall weight masquerade as differences in topology. Returns a binary-able
    weighted graph (surviving weights kept; use ``> 0`` for the binary mask).
    """
    if not 0 < density <= 1:
        raise ValueError(f"density must be in (0, 1], got {density}")
    a = adj.copy()
    np.fill_diagonal(a, 0.0)
    n = a.shape[0]
    iu = np.triu_indices(n, k=1)
    weights = a[iu]
    n_keep = int(round(density * weights.size))
    if n_keep < weights.size and n_keep >= 0:
        if n_keep == 0:
            return np.zeros_like(a)
        thresh = np.sort(weights)[::-1][n_keep - 1]
        a[a < thresh] = 0.0
    return a


# -- individual metrics ----------------------------------------------------
def modularity(adj: np.ndarray, method: str = "louvain") -> float:
    """Modularity Q on the binary thresholded graph (paper-faithful).

    The seRNN paper computes Q on the binarized top-density |W|, not weighted
    Louvain. Weighted Louvain inflates heavy-tailed graphs (e.g. an over-
    sparsified L1 control) and inverted the seRNN-vs-control ranking in the first
    kill test, so we binarize (matching clustering / path-length / small-
    worldness). On binary graphs Louvain and Newman-spectral Q agree to ~0.01,
    and ``bct.modularity_und`` is incompatible with modern NumPy, so we default
    to deterministic binary Louvain.

    ``method``: "louvain" (default) or "newman" (``bct.modularity_und``; falls
    back to louvain if bctpy/NumPy incompatible).
    """
    a = symmetrize(adj)
    a_bin = (a > 0).astype(float)        # binarize: match the paper and the other metrics
    if a_bin.sum() == 0:
        return 0.0

    if method == "newman":
        try:
            import bct  # type: ignore
            _, q = bct.modularity_und(a_bin, gamma=1)
            return float(q)
        except Exception:
            method = "louvain"           # bctpy broken on NumPy 2.x -> binary Louvain

    if not (_HAVE_LOUVAIN and _HAVE_NX):
        raise ImportError("modularity needs networkx + python-louvain (or bctpy)")
    # a_bin edges all have weight 1.0, so weighted Louvain == binary Louvain here.
    g = nx.from_numpy_array(a_bin)
    partition = community_louvain.best_partition(g, weight="weight", random_state=0)
    return float(community_louvain.modularity(partition, g, weight="weight"))


def characteristic_path_length(adj: np.ndarray, binary: bool = True) -> float:
    """Mean shortest-path length over the largest connected component.

    Restricting to the giant component avoids infinities from isolated nodes;
    we report the component this is computed on via :func:`compute_all`.
    """
    if not _HAVE_NX:
        raise ImportError("path length needs networkx")
    a = symmetrize(adj)
    g = nx.from_numpy_array(a)
    if binary:
        g = nx.Graph((u, v) for u, v, w in g.edges(data="weight") if w and w > 0)
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        return float("nan")
    comp_nodes = max(nx.connected_components(g), key=len)
    sub = g.subgraph(comp_nodes)
    if sub.number_of_nodes() < 2:
        return float("nan")
    return float(nx.average_shortest_path_length(sub))


def clustering_coefficient(adj: np.ndarray, binary: bool = True) -> float:
    """Average clustering coefficient (transitivity of neighbourhoods)."""
    if not _HAVE_NX:
        raise ImportError("clustering needs networkx")
    a = symmetrize(adj)
    g = nx.from_numpy_array(a)
    if binary:
        g = nx.Graph((u, v) for u, v, w in g.edges(data="weight") if w and w > 0)
    if g.number_of_nodes() == 0:
        return float("nan")
    return float(nx.average_clustering(g))


def small_worldness(
    adj: np.ndarray, n_random: int = 10, seed: int = 0
) -> float:
    """Small-world coefficient sigma = (C/C_rand) / (L/L_rand).

    sigma > 1 (high clustering *and* short paths relative to a degree-matched
    random graph) is the canonical small-world signature seRNN reports. Averaged
    over ``n_random`` rewired null graphs for stability.
    """
    if not _HAVE_NX:
        raise ImportError("small-worldness needs networkx")
    a = symmetrize(adj)
    g = nx.from_numpy_array(a)
    g = nx.Graph((u, v) for u, v, w in g.edges(data="weight") if w and w > 0)
    if g.number_of_edges() == 0:
        return float("nan")
    comp_nodes = max(nx.connected_components(g), key=len)
    g = g.subgraph(comp_nodes).copy()
    if g.number_of_nodes() < 3:
        return float("nan")

    c = nx.average_clustering(g)
    l = nx.average_shortest_path_length(g)

    rng = np.random.default_rng(seed)
    c_rand, l_rand = [], []
    n_edges = g.number_of_edges()
    for _ in range(n_random):
        # Degree-preserving randomization; fall back to G(n,m) if it can't mix.
        try:
            r = nx.random_reference(g, niter=5, seed=int(rng.integers(1 << 30)))
        except Exception:
            r = nx.gnm_random_graph(g.number_of_nodes(), n_edges, seed=int(rng.integers(1 << 30)))
        if r.number_of_edges() == 0 or not nx.is_connected(r):
            comp = max(nx.connected_components(r), key=len)
            r = r.subgraph(comp).copy()
            if r.number_of_nodes() < 3:
                continue
        c_rand.append(nx.average_clustering(r))
        l_rand.append(nx.average_shortest_path_length(r))
    if not c_rand:
        return float("nan")
    c_r = float(np.mean(c_rand))
    l_r = float(np.mean(l_rand))
    if c_r == 0 or l_r == 0 or l == 0:
        return float("nan")
    return (c / c_r) / (l / l_r)


def spatial_homophily(adj: np.ndarray, distance: np.ndarray) -> float:
    """Weighted correlation between connection strength and *proximity*.

    seRNN connections are homophilic: strong weights sit between nearby units.
    We return ``-corr(|w_ij|, d_ij)`` over node pairs, so a positive value means
    "stronger connections are shorter" (the brain-like direction).
    """
    a = symmetrize(adj)
    d = np.asarray(distance, dtype=float)
    n = a.shape[0]
    iu = np.triu_indices(n, k=1)
    w = a[iu]
    dist = d[iu]
    if w.std() == 0 or dist.std() == 0:
        return float("nan")
    return float(-np.corrcoef(w, dist)[0, 1])


# -- one-call summary ------------------------------------------------------
def compute_all(
    weight: np.ndarray,
    distance: Optional[np.ndarray] = None,
    density: Optional[float] = 0.1,
    n_random: int = 10,
    seed: int = 0,
) -> Dict[str, float]:
    """Full brain-like-structure summary for one network.

    Parameters
    ----------
    weight:
        ``(N, N)`` recurrent weight matrix (signed/directed ok; symmetrized
        internally).
    distance:
        ``(N, N)`` geometry distances; enables the homophily metric.
    density:
        If set, sparsify to this edge density before topology metrics so
        variants are compared at matched density. ``None`` uses the full graph.
    """
    a = symmetrize(weight)
    a_sparse = threshold_proportional(a, density) if density else a

    metrics: Dict[str, float] = {}
    metrics["modularity"] = modularity(a_sparse)
    metrics["clustering"] = clustering_coefficient(a_sparse)
    metrics["path_length"] = characteristic_path_length(a_sparse)
    metrics["small_worldness"] = small_worldness(a_sparse, n_random=n_random, seed=seed)
    if distance is not None:
        metrics["homophily"] = spatial_homophily(a, distance)
    return metrics
