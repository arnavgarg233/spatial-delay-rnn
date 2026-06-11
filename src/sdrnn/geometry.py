"""Neuron placement in physical space and the pairwise distance matrix.

Distance feeds both the spatial penalty (per-connection cost) and the delay
module (distance -> integer lag). Coordinates are either fixed (seRNN baseline:
a static 3-D grid) or learnable (an ``nn.Parameter``). Both expose the same
:meth:`distance_matrix` so callers never branch on the mode.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def grid_coordinates(n: int, dim: int = 3) -> torch.Tensor:
    """Place ``n`` neurons on the most cube-like regular grid in ``dim`` dims.

    seRNN uses a regular lattice (e.g. 100 neurons on a 5x5x4 grid). We pick
    per-axis counts as close to equal as possible and lay the first ``n`` points
    out in row-major order, then center the cloud on the origin and scale so the
    mean nearest-neighbour spacing is ~1 (keeps distance magnitudes comparable
    across network sizes, which matters for the scaling sweep).

    Returns a ``(n, dim)`` float tensor.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    # Per-axis side lengths whose product covers n, as balanced as possible.
    side = max(1, round(n ** (1.0 / dim)))
    sides = [side] * dim
    i = 0
    while math.prod(sides) < n:  # grow axes round-robin until the grid holds n
        sides[i % dim] += 1
        i += 1

    # Row-major enumeration of the lattice, truncated to exactly n points.
    coords = torch.zeros(n, dim)
    for idx in range(n):
        rem = idx
        for d in range(dim):
            stride = math.prod(sides[d + 1 :]) if d + 1 < dim else 1
            coords[idx, d] = rem // stride
            rem = rem % stride

    coords = coords - coords.mean(dim=0, keepdim=True)  # center on origin
    coords = coords / max(side - 1, 1)                  # ~unit lattice spacing
    return coords


class NeuronGeometry(nn.Module):
    """Holds neuron coordinates and yields the pairwise Euclidean distances.

    Parameters
    ----------
    n_neurons:
        Number of recurrent units placed in space.
    dim:
        Embedding dimensionality (3 reproduces the seRNN setup).
    learnable:
        If ``False`` (baseline) coordinates are a fixed buffer. If ``True``
        (our extension) they are a parameter the optimizer can move. A learnable
        geometry recomputes distances from the live coordinates every call, so
        the spatial penalty and delays follow the neurons as they migrate.
    init_coords:
        Optional ``(n_neurons, dim)`` override for the initial placement;
        defaults to :func:`grid_coordinates`.
    jitter:
        Std of optional Gaussian noise added to the initial grid. A learnable
        geometry initialized on a perfectly symmetric lattice can sit at a
        saddle; a touch of jitter breaks the symmetry. Ignored when 0.
    """

    def __init__(
        self,
        n_neurons: int,
        dim: int = 3,
        learnable: bool = False,
        init_coords: Optional[torch.Tensor] = None,
        jitter: float = 0.0,
        normalize_scale: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()
        self.n_neurons = n_neurons
        self.dim = dim
        self.learnable = learnable
        # Anti-collapse: with learnable coords, the distance penalty is trivially
        # minimized by shrinking all neurons toward one point (distance -> 0).
        # Normalizing the cloud to a fixed scale each forward removes that
        # degenerate optimum, so the only way to reduce the penalty is to
        # REARRANGE neurons relatively - the meaningful self-organization. On by
        # default for learnable geometries; a no-op rescale for fixed grids.
        self.normalize_scale = learnable if normalize_scale is None else normalize_scale

        if init_coords is None:
            coords = grid_coordinates(n_neurons, dim)
        else:
            coords = torch.as_tensor(init_coords, dtype=torch.float32).clone()
            if coords.shape != (n_neurons, dim):
                raise ValueError(
                    f"init_coords must be ({n_neurons}, {dim}), got {tuple(coords.shape)}"
                )

        if jitter > 0:
            noise = torch.randn(coords.shape, generator=generator) * jitter
            coords = coords + noise

        if learnable:
            self.coords = nn.Parameter(coords)
        else:
            self.register_buffer("coords", coords)

    def _scaled_coords(self) -> torch.Tensor:
        """Coordinates centered and rescaled to a fixed RMS radius (anti-collapse).

        Centers on the centroid and divides by the RMS distance-from-centroid so
        the cloud always has unit RMS radius. This is a similarity transform: it
        preserves the *relative* arrangement (what we care about) while pinning
        the global scale, blocking the collapse-to-a-point degeneracy. Applied
        only when ``normalize_scale`` is set.
        """
        c = self.coords
        if not self.normalize_scale:
            return c
        c = c - c.mean(dim=0, keepdim=True)
        rms = c.pow(2).sum(dim=1).mean().clamp_min(1e-12).sqrt()
        return c / rms

    def distance_matrix(self) -> torch.Tensor:
        """Return the ``(n, n)`` Euclidean distance matrix for current coords.

        Differentiable w.r.t. the coordinates, so when ``learnable`` is on the
        spatial penalty pushes gradients back into neuron positions. Uses the
        scale-normalized coordinates when ``normalize_scale`` is on.
        """
        # torch.cdist is the numerically stable pairwise-distance primitive.
        c = self._scaled_coords()
        return torch.cdist(c, c, p=2)

    @torch.no_grad()
    def coords_numpy(self):
        """Effective (scale-normalized) coordinates as NumPy, for plotting/analysis.

        Returns the same positions the geometry actually uses for distances, so
        analyses see the real layout (not the un-normalized raw parameters).
        """
        return self._scaled_coords().detach().cpu().numpy()

    def extra_repr(self) -> str:
        return f"n_neurons={self.n_neurons}, dim={self.dim}, learnable={self.learnable}"
