"""The spatially embedded, delay-coupled RNN.

One module covers the whole ablation lattice from the README - plain RNN,
seRNN baseline, +delay, +movement, and combinations - selected through
:class:`SDRNNConfig` toggles rather than separate model classes, so an ablation
is a config diff and every variant shares identical weight-init and task wiring.

Recurrence (continuous-time leaky form, Euler-discretized - the seRNN style):

    r_t = activation(state)
    rec = W_rec @ r          (delayed: sum_j W_rec_ij r_j[t - tau_ij])
    state <- (1 - alpha) * state + alpha * (W_in @ x_t + rec + b)
    y_t = W_out @ activation(state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import torch
import torch.nn as nn

from sdrnn.delays import (
    DelayBuffer,
    fractional_delays,
    integer_delays,
)
from sdrnn.geometry import NeuronGeometry
from sdrnn.regularization import WeightingMode, spatial_penalty


@dataclass
class SDRNNConfig:
    """Everything that defines a variant. Ablations are diffs on this object."""

    # --- dimensions ---
    input_size: int = 4
    hidden_size: int = 100
    output_size: int = 2
    space_dim: int = 3

    # --- dynamics ---
    alpha: float = 0.2                       # Euler step / leak (dt/tau)
    activation: str = "relu"                 # seRNN uses a positive-rate ReLU
    nonneg_recurrent: bool = False           # optionally constrain |W| like a rate net

    # --- mechanism 0: spatial regularization (the seRNN baseline knob) ---
    reg_mode: WeightingMode = "communicability"  # none | distance | communicability
    reg_lambda: float = 1e-3
    normalize_comm: bool = True

    # --- mechanism 1: distance-proportional delays ---
    use_delays: bool = False
    velocity: float = 1.0                    # lattice-units per timestep
    max_delay: int = 8
    min_delay: int = 1
    delay_interpolation: str = "integer"     # integer | fractional
    learn_velocity: bool = False             # opt-in global conduction speed
    delay_control: str = "distance"          # distance | shuffled | uniform
    uniform_delay: Optional[int] = None      # if None, use rounded mean lag

    # --- mechanism 2: learnable coordinates (neuron movement) ---
    learnable_coords: bool = False
    coord_jitter: float = 0.0                # symmetry-break for learnable grids
    coord_repulsion: float = 0.0             # soft min-distance penalty (anti point-collapse)
    coord_min_dist: float = 0.15             # target minimum neuron separation (scaled coords)

    # --- init / misc ---
    seed: Optional[int] = None
    init_coords: Optional[torch.Tensor] = field(default=None, repr=False)

    def describe(self) -> str:
        """One-line variant tag for logging / filenames."""
        bits = [f"n{self.hidden_size}", self.reg_mode]
        if self.use_delays:
            vtag = "learnv" if self.learn_velocity else f"v{self.velocity}"
            bits.append(f"delay({vtag},max{self.max_delay},{self.delay_interpolation})")
            if self.delay_control != "distance":
                bits.append(self.delay_control)
        if self.learnable_coords:
            bits.append("move")
        return "-".join(bits)


_ACTIVATIONS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


class SDRNN(nn.Module):
    """Spatially embedded recurrent network with optional delays and movement."""

    def __init__(self, config: SDRNNConfig) -> None:
        super().__init__()
        self.config = config
        if config.activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {config.activation!r}")
        if config.delay_interpolation not in {"integer", "fractional"}:
            raise ValueError(f"unknown delay_interpolation {config.delay_interpolation!r}")
        if config.delay_control not in {"distance", "shuffled", "uniform"}:
            raise ValueError(f"unknown delay_control {config.delay_control!r}")
        self.activation = _ACTIVATIONS[config.activation]

        gen = None
        if config.seed is not None:
            gen = torch.Generator().manual_seed(config.seed)

        n = config.hidden_size
        self.input_layer = nn.Linear(config.input_size, n, bias=False)
        self.recurrent = nn.Linear(n, n, bias=True)
        self.output_layer = nn.Linear(n, config.output_size, bias=False)
        self.log_velocity = nn.Parameter(
            torch.tensor(math.log(config.velocity), dtype=torch.float32),
            requires_grad=config.learn_velocity,
        )

        self._init_weights(gen)

        self.geometry = NeuronGeometry(
            n_neurons=n,
            dim=config.space_dim,
            learnable=config.learnable_coords,
            init_coords=config.init_coords,
            jitter=config.coord_jitter,
            generator=gen,
        )
        self.register_buffer(
            "_delay_shuffle_perm",
            self._make_delay_shuffle_perm(n, config.seed),
            persistent=False,
        )

    @staticmethod
    def _make_delay_shuffle_perm(n: int, seed: Optional[int]) -> torch.Tensor:
        """Fixed off-diagonal permutation for same-histogram delay controls."""
        gen = torch.Generator().manual_seed((0 if seed is None else int(seed)) + 1009)
        return torch.randperm(n * n - n, generator=gen)

    # -- initialization ----------------------------------------------------
    def _init_weights(self, gen: Optional[torch.Generator]) -> None:
        n = self.config.hidden_size
        # Recurrent: scaled orthogonal-ish gaussian (spectral radius ~1) is a
        # stable RNN default; the spatial penalty does the sparsification.
        std = 1.0 / (n ** 0.5)
        with torch.no_grad():
            self.recurrent.weight.normal_(0.0, std, generator=gen)
            self.recurrent.bias.zero_()
            self.input_layer.weight.normal_(0.0, std, generator=gen)
            self.output_layer.weight.normal_(0.0, std, generator=gen)

    # -- recurrent weight view (positivity option) -------------------------
    def recurrent_weight(self) -> torch.Tensor:
        """The effective recurrent matrix ``W_rec`` used in the dynamics."""
        w = self.recurrent.weight
        return w.abs() if self.config.nonneg_recurrent else w

    # -- forward -----------------------------------------------------------
    def forward(self, inputs: torch.Tensor, return_hidden: bool = False):
        """Run the network over a sequence.

        Parameters
        ----------
        inputs:
            ``(batch, T, input_size)`` input sequence.
        return_hidden:
            If True also return the ``(batch, T, hidden)`` rate trajectory
            (used by analyses that need the dynamics, not just the readout).

        Returns ``outputs`` ``(batch, T, output_size)`` and, optionally, hidden.
        """
        if inputs.dim() != 3:
            raise ValueError(f"expected (batch, T, input), got {tuple(inputs.shape)}")
        batch, T, _ = inputs.shape
        n = self.config.hidden_size
        device = self.recurrent.weight.device
        dtype = self.recurrent.weight.dtype
        alpha = self.config.alpha
        cfg = self.config

        w_rec = self.recurrent_weight()
        bias = self.recurrent.bias

        # Delay scaffolding (only when enabled).
        if cfg.use_delays:
            velocity = self.current_velocity()
            dist = self.geometry.distance_matrix()
            if cfg.delay_interpolation == "integer":
                tau = integer_delays(
                    dist,
                    velocity=float(velocity.detach().item()),
                    max_delay=cfg.max_delay,
                    min_delay=cfg.min_delay,
                )
                tau = self._apply_delay_control(tau)
            else:
                if cfg.delay_control != "distance":
                    raise ValueError("fractional delays currently support only delay_control='distance'")
                tau = fractional_delays(
                    dist,
                    velocity=velocity,
                    max_delay=cfg.max_delay,
                    min_delay=cfg.min_delay,
                )
            buf = DelayBuffer(cfg.max_delay, batch, n, device, dtype)

        state = torch.zeros(batch, n, device=device, dtype=dtype)
        proj = self.input_layer(inputs)  # (batch, T, n), precompute input drive

        outputs = []
        hiddens = [] if return_hidden else None
        for t in range(T):
            rate = self.activation(state)
            if cfg.use_delays:
                # Push BEFORE apply so history[1] holds the current rate
                # (= f(state_{t-1}), the "one step ago" signal). This makes a
                # tau=1 delay identical to the no-delay path below, removing the
                # off-by-one and the extra-latency confound the audit flagged:
                # otherwise apply()-before-push() made every lag one step too
                # long and handicapped the delay variant vs the baseline.
                buf.push(rate)
                if cfg.delay_interpolation == "integer":
                    rec = buf.apply_gather(w_rec, tau)  # sum_j W_ij rate_j[t - tau_ij]
                else:
                    rec = buf.apply_fractional_gather(w_rec, tau)
            else:
                rec = rate @ w_rec.t()             # standard single-step
            pre = proj[:, t] + rec + bias
            state = (1 - alpha) * state + alpha * pre
            out_rate = self.activation(state)
            outputs.append(self.output_layer(out_rate))
            if return_hidden:
                hiddens.append(out_rate)

        outputs = torch.stack(outputs, dim=1)
        if return_hidden:
            return outputs, torch.stack(hiddens, dim=1)
        return outputs

    def current_velocity(self) -> torch.Tensor:
        """Positive global conduction velocity, optionally learnable."""
        return torch.exp(self.log_velocity).clamp_min(1e-6)

    def _apply_delay_control(self, tau: torch.Tensor) -> torch.Tensor:
        """Optional reviewer controls for isolating distance-proportional delays."""
        cfg = self.config
        if cfg.delay_control == "distance":
            return tau

        n = tau.shape[0]
        off = ~torch.eye(n, dtype=torch.bool, device=tau.device)
        controlled = tau.clone()
        if cfg.delay_control == "shuffled":
            vals = controlled[off]
            perm = self._delay_shuffle_perm.to(vals.device)
            controlled[off] = vals[perm]
        elif cfg.delay_control == "uniform":
            if cfg.uniform_delay is None:
                delay = int(torch.round(controlled[off].float().mean()).item())
            else:
                delay = int(cfg.uniform_delay)
            delay = max(cfg.min_delay, min(cfg.max_delay, delay))
            controlled[off] = delay
        return controlled

    # -- regularization ----------------------------------------------------
    def spatial_regularization(self) -> torch.Tensor:
        """The seRNN spatial penalty (+ optional coord repulsion) for current state."""
        dist = self.geometry.distance_matrix()
        penalty = self.config.reg_lambda * spatial_penalty(
            self.recurrent_weight(),
            dist,
            mode=self.config.reg_mode,
            normalize_comm=self.config.normalize_comm,
        )
        if self.config.learnable_coords and self.config.coord_repulsion > 0:
            penalty = penalty + self.config.coord_repulsion * self._repulsion_penalty(dist)
        return penalty

    def _repulsion_penalty(self, dist: torch.Tensor) -> torch.Tensor:
        """Soft minimum-distance penalty preventing neurons from overlapping.

        Hinge on too-close pairs only: ``mean relu(min_dist - d_ij)^2`` over
        off-diagonal pairs. Inactive for well-separated neurons, so it spreads
        collapsed modules into extended regions without fighting the spatial
        penalty's overall pull. Only meaningful with scale-normalized coords.
        """
        n = dist.shape[0]
        off = ~torch.eye(n, dtype=torch.bool, device=dist.device)
        gap = torch.relu(self.config.coord_min_dist - dist[off])
        return (gap ** 2).mean()

    @torch.no_grad()
    def weight_matrix_numpy(self):
        """Detached ``|W_rec|`` as NumPy - the adjacency for graph metrics."""
        return self.recurrent_weight().detach().abs().cpu().numpy()
