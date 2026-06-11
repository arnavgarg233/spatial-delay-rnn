"""seRNN spatial regularization: L1 weighted by physical cost.

Cost combines distance d_ij (long axons are expensive) and communicability C_ij
(connecting already well-connected regions is redundant, extra-penalized):

    L_spatial = lambda * sum_{i,j} |W_ij| * d_ij * C_ij

with C = expm(A), A = |W| (Crofts-Higham communicability). Weighting modes:
    none            -> plain L1   (non-spatial control)
    distance        -> |W| * d
    communicability -> |W| * d * C   (full seRNN)
"""

from __future__ import annotations

from typing import Literal

import torch

WeightingMode = Literal["none", "distance", "communicability"]


def communicability(abs_weight: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Crofts-Higham communicability ``C = expm(A)`` of a weighted graph.

    Parameters
    ----------
    abs_weight:
        ``(N, N)`` non-negative weighted adjacency (pass ``|W|``).
    normalize:
        If True, symmetrically normalize ``A -> S^-1/2 A S^-1/2`` (S = node
        strength) before the matrix exponential. This is the standard guard
        against ``expm`` overflowing as weights grow, and keeps the penalty on a
        comparable scale across network sizes - important for the scaling sweep.

    Returns the ``(N, N)`` communicability matrix, differentiable in
    ``abs_weight`` via :func:`torch.matrix_exp`.
    """
    a = abs_weight
    if normalize:
        strength = a.sum(dim=1)                       # node strength s_i
        # Audit fix: a dead neuron (strength 0, common under ReLU + sparsity)
        # would otherwise get rsqrt(1e-12)=1e6 and blow up its incoming column.
        # Map zero-strength nodes to factor 0 so they contribute nothing.
        inv_sqrt = torch.where(
            strength > 0,
            torch.rsqrt(strength.clamp_min(1e-12)),
            torch.zeros_like(strength),
        )
        a = inv_sqrt.unsqueeze(1) * a * inv_sqrt.unsqueeze(0)
    return torch.matrix_exp(a)


def spatial_penalty(
    weight: torch.Tensor,
    distance: torch.Tensor,
    mode: WeightingMode = "communicability",
    normalize_comm: bool = True,
) -> torch.Tensor:
    """Scalar spatial regularization term (before multiplying by ``lambda``).

    Parameters
    ----------
    weight:
        ``(N, N)`` recurrent weight matrix.
    distance:
        ``(N, N)`` Euclidean distances from the geometry.
    mode:
        Which weighting to apply (see module docstring).
    normalize_comm:
        Passed through to :func:`communicability` when ``mode`` uses it.

    Returns a scalar tensor; ``sum`` reduction matches seRNN (so the penalty's
    magnitude grows with the network, which the per-size ``lambda`` absorbs).
    """
    abs_w = weight.abs()

    if mode == "none":
        return abs_w.sum()
    if mode == "distance":
        return (abs_w * distance).sum()
    if mode == "communicability":
        comm = communicability(abs_w, normalize=normalize_comm)
        return (abs_w * distance * comm).sum()

    raise ValueError(f"unknown weighting mode: {mode!r}")
