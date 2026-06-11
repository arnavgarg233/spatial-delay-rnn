"""Distance-proportional transmission delays.

A signal from neuron j to i takes tau_ij steps, proportional to Euclidean
distance over a finite conduction velocity:

    tau_ij = clip(round(d_ij / velocity), 1, max_delay)

so unit i receives sum_j W_ij * h_j[t - tau_ij].

Integer delays take few distinct values, so we group edges by lag and do one
masked matmul per distinct lag instead of an O(N^2) per-edge gather.

round() is non-differentiable, so gradients don't flow into coordinates through
the integer lag; lags are recomputed from live coords each forward pass and
treated as a reindexing op. Use fractional_delays for a differentiable lag.
"""

from __future__ import annotations

from typing import Union

import torch


def integer_delays(
    distance: torch.Tensor,
    velocity: float,
    max_delay: int,
    min_delay: int = 1,
) -> torch.Tensor:
    """Map a distance matrix to integer transmission lags.

    Parameters
    ----------
    distance:
        ``(N, N)`` Euclidean distances.
    velocity:
        Conduction speed in lattice-units per timestep. Larger -> shorter lags.
    max_delay:
        Upper clip on the lag (also the depth of history the model must keep).
    min_delay:
        Lower clip; 1 means "at least one step", matching a standard RNN's
        single-step recurrence as the floor.

    Returns an integer ``(N, N)`` tensor, detached (it indexes history buffers).
    """
    if velocity <= 0:
        raise ValueError(f"velocity must be positive, got {velocity}")
    if max_delay < min_delay:
        raise ValueError(f"max_delay ({max_delay}) < min_delay ({min_delay})")

    with torch.no_grad():
        tau = torch.round(distance / velocity).long()
        tau = tau.clamp_(min_delay, max_delay)
    return tau


def fractional_delays(
    distance: torch.Tensor,
    velocity: Union[float, torch.Tensor],
    max_delay: int,
    min_delay: int = 1,
) -> torch.Tensor:
    """Map distances to clipped fractional lags.

    Unlike :func:`integer_delays`, this keeps gradients through the fractional
    lag value. The later floor/ceil indexing is still piecewise constant, but
    interpolation weights remain differentiable with respect to distance and
    velocity inside each integer-lag bin.
    """
    if max_delay < min_delay:
        raise ValueError(f"max_delay ({max_delay}) < min_delay ({min_delay})")
    if isinstance(velocity, torch.Tensor):
        velocity = velocity.clamp_min(1e-6)
    elif velocity <= 0:
        raise ValueError(f"velocity must be positive, got {velocity}")
    return (distance / velocity).clamp(min_delay, max_delay)


class DelayBuffer:
    """Rolling history of hidden states, indexed by per-connection lag.

    Holds the last ``max_delay`` hidden states for a batch and applies a
    delay-grouped recurrent weight matrix in one sweep. The buffer is a plain
    object (not an ``nn.Module``) because it carries no parameters - just the
    transient history for one forward pass / sequence.

    Usage per sequence::

        buf = DelayBuffer(max_delay, batch, n, device, dtype)
        for t in range(T):
            rec = buf.apply(W, tau)      # sum_j W_ij h_j[t - tau_ij]
            h = activation(input_t + rec + bias)
            buf.push(h)
    """

    def __init__(self, max_delay, batch, n, device, dtype):
        self.max_delay = int(max_delay)
        # history[k] holds h from k steps ago (k=1 is the most recent push).
        # Pre-fill with zeros so early steps see "no signal yet" from far units.
        self.history = torch.zeros(self.max_delay + 1, batch, n, device=device, dtype=dtype)

    def push(self, h: torch.Tensor) -> None:
        """Record the newest hidden state, evicting the oldest."""
        # Shift the ring: index 1 becomes the just-computed state.
        self.history = torch.roll(self.history, shifts=1, dims=0)
        self.history[1] = h

    def apply(self, weight: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Delay-weighted recurrent input ``sum_j W_ij h_j[t - tau_ij]``.

        ``weight`` is ``(N, N)`` with ``W_ij`` the strength from j (col) to
        i (row); ``tau`` is the matching integer-lag matrix.
        """
        return self.apply_groups(grouped_delay_weights(weight, tau))

    def apply_groups(self, groups: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        """Apply precomputed ``(delay, masked_weight)`` groups.

        ``tau`` is constant during one forward pass, so callers can build these
        groups once per sequence instead of rebuilding masks at every timestep.
        """
        out = torch.zeros_like(self.history[1])
        for d_int, masked in groups:
            # history[d_int]: (batch, n) hidden state from d steps ago.
            # masked.T so that out[:, i] = sum_j h_j * W_ij.
            out = out + self.history[d_int] @ masked.t()
        return out

    def apply_gather(self, weight: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Delay recurrent input using a single gather/sum over edges.

        This is faster than one matmul per distinct lag when many lags are
        present. It materializes ``(batch, target, source)`` delayed source
        activations, then sums them with ``W[target, source]``.
        """
        batch = self.history.shape[1]
        n = tau.shape[0]
        hist = self.history.permute(1, 2, 0)              # (batch, source, delay)
        hist = hist[:, None, :, :].expand(batch, n, n, -1)
        idx = tau.to(hist.device).unsqueeze(0).expand(batch, n, n).unsqueeze(-1)
        delayed = torch.gather(hist, dim=3, index=idx).squeeze(-1)
        return torch.einsum("bij,ij->bi", delayed, weight)

    def apply_fractional_gather(self, weight: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Interpolated fractional-delay recurrent input via gather/sum."""
        batch = self.history.shape[1]
        n = tau.shape[0]
        lo = torch.floor(tau).long().clamp(1, self.max_delay)
        hi = torch.ceil(tau).long().clamp(1, self.max_delay)
        frac = (tau - lo.to(tau.dtype)).clamp(0.0, 1.0)

        hist = self.history.permute(1, 2, 0)              # (batch, source, delay)
        hist = hist[:, None, :, :].expand(batch, n, n, -1)
        lo_idx = lo.to(hist.device).unsqueeze(0).expand(batch, n, n).unsqueeze(-1)
        hi_idx = hi.to(hist.device).unsqueeze(0).expand(batch, n, n).unsqueeze(-1)
        h_lo = torch.gather(hist, dim=3, index=lo_idx).squeeze(-1)
        h_hi = torch.gather(hist, dim=3, index=hi_idx).squeeze(-1)
        delayed = h_lo * (1.0 - frac).unsqueeze(0) + h_hi * frac.unsqueeze(0)
        return torch.einsum("bij,ij->bi", delayed, weight)


def grouped_delay_weights(weight: torch.Tensor, tau: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
    """Precompute one masked recurrent matrix per distinct integer delay."""
    zero = torch.zeros_like(weight)
    groups = []
    for d in torch.unique(tau):
        groups.append((int(d.item()), torch.where(tau == d, weight, zero)))
    return groups


def grouped_fractional_delay_weights(
    weight: torch.Tensor,
    tau: torch.Tensor,
    min_delay: int,
    max_delay: int,
) -> list[tuple[int, torch.Tensor]]:
    """Precompute interpolated delay groups for fractional lags.

    For edge ``j -> i`` with fractional lag ``tau_ij = lo + frac``, the
    recurrent term is split across adjacent history states:

        W_ij * ((1 - frac) h_j[t - lo] + frac h_j[t - hi]).
    """
    lo = torch.floor(tau).long().clamp(min_delay, max_delay)
    hi = torch.ceil(tau).long().clamp(min_delay, max_delay)
    frac = (tau - lo.to(tau.dtype)).clamp(0.0, 1.0)
    zero = torch.zeros_like(weight)

    groups = []
    for d in range(min_delay, max_delay + 1):
        lo_mask = lo == d
        hi_mask = hi == d
        if not (bool(lo_mask.any()) or bool(hi_mask.any())):
            continue
        masked = zero
        if bool(lo_mask.any()):
            masked = masked + torch.where(lo_mask, weight * (1.0 - frac), zero)
        if bool(hi_mask.any()):
            masked = masked + torch.where(hi_mask, weight * frac, zero)
        groups.append((d, masked))
    return groups
