"""Model-mismatch foreign generator for the CV inverse: a delay-coupled RNN whose
functional form does NOT match the inverse's basis, run with a known v_true through
the same velocity-scan + delayed-field residual + velocity-shuffled null machinery
(reused from pinn_inverse.py). Tests whether v survives misspecification.

The foreign generator (mismatch_rnn) is a continuous-time leaky RNN with three
mismatches vs the inverse's relu/integer-delay/additive-linear basis, each scaled
by `mismatch` in [0,1]:
  (1) saturating tanh activation instead of ReLU;
  (2) distributed/fractional delays (Gaussian blend of lags around d_ij/v) instead
      of a single integer lag;
  (3) multiplicative conductance-like gating: drive = (1-m)*lin + m*sigma(beta*h)*lin.
mismatch=0 reduces to the SDRNN basis (sanity: should recover v); 1 = fully foreign.
v_true always sets the delays (delay ∝ distance / v_true).

The inverse sees only h(t) from real MemoryProTask trials: it relu-rectifies the
states, reconstructs an approximate recurrent target with its assumed alpha, and
fits a free kernel per candidate v (kernel marginalized). Observation noise added
to test robustness. Reports recovered v ± error across seeds/noise/mismatch, the
dip shape, and the velocity-shuffled null floor ratio.

  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=3 \
    python scripts/inverse/foreigninv_mismatch.py --smoke
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from sdrnn.geometry import grid_coordinates
from sdrnn.tasks import MemoryProTask

# Reuse the inverse machinery verbatim (same scoring as the tautological version).
from pinn_inverse import (
    lagged_rate,
    normalized_residual,
    fit_kernel_at_velocity,
    scan_freekernel,
    shuffle_geometry,
    corr,
)


# The foreign generator: mismatch_rnn.
def build_distributed_delay_groups(dist, velocity, max_delay, min_delay,
                                    delay_spread, mismatch):
    """Per-target Gaussian-distributed delay weights over integer lag bins.

    For edge j->i the real delay is d_ij/v. We spread its weight over integer
    lags k in [min_delay, max_delay] with a Gaussian kernel centered at d_ij/v
    and std `delay_spread` (scaled by mismatch: at mismatch=0 spread->0 so the
    kernel collapses to the single nearest integer = the SDRNN's integer delay).

    Returns dict: lag k -> (N,N) weight-of-that-lag matrix Wk (rows i, cols j),
    each entry the fraction of edge (i,j)'s signal arriving at lag k. Rows of
    the stacked kernel sum to 1 over k (per edge).
    """
    N = dist.shape[0]
    real_delay = np.clip(dist / velocity, min_delay, max_delay)  # (N,N) fractional
    ks = np.arange(min_delay, max_delay + 1)
    if mismatch <= 0.0:
        # MATCHED basis: single nearest-integer lag (delta), exactly the SDRNN's
        # integer_delays(round(d/v)). No smearing.
        tau = np.clip(np.round(real_delay), min_delay, max_delay).astype(int)
        W = np.zeros((len(ks), N, N), dtype=np.float64)
        for ki, k in enumerate(ks):
            W[ki] = (tau == k).astype(np.float64)
        return {int(k): W[ki] for ki, k in enumerate(ks)}
    # FOREIGN: distributed/fractional delay. Floor spread at 0.5 so the kernel
    # always covers at least the nearest bin (a real fractional/smeared delay,
    # not a degenerate delta), growing with mismatch.
    spread = 0.5 + delay_spread * mismatch
    W = np.zeros((len(ks), N, N), dtype=np.float64)
    for ki, k in enumerate(ks):
        W[ki] = np.exp(-0.5 * ((k - real_delay) / spread) ** 2)
    W /= W.sum(axis=0, keepdims=True) + 1e-12
    return {int(k): W[ki] for ki, k in enumerate(ks)}


@torch.no_grad()
def generate_mismatch_activity(W_rec, dist, velocity, task, device,
                               *, mismatch, alpha_gen, gain, beta_gate,
                               delay_spread, max_delay, min_delay,
                               n_trials, n_repeats, seed):
    """Run the FOREIGN delay-coupled RNN on real task trials; return h-traces.

    Dynamics (continuous-time leaky, Euler step alpha_gen):
        phi(x)  = tanh(gain * x)                       [saturating, mismatch-blended to relu]
        lagged_in_i(t) = sum_k sum_j Wdist_k[i,j] * W_rec[i,j] * phi(h_j)[t-k]
        gate_i  = sigmoid(beta_gate * h_i(t))          [conductance-like local gate]
        drive_i = (1-m)*lagged_in_i + m * gate_i * lagged_in_i
        h_i <- (1-alpha_gen) h_i + alpha_gen (proj_i(t) + drive_i + bias_i)

    At mismatch m=0: phi=relu, delta-delay (nearest int), additive drive
    => the SDRNN basis (sanity). At m=1: tanh, smeared fractional delay,
    multiplicative gate => fully foreign.

    Returns h-states (B,T,N) and proj (B,T,N) as numpy float64. We deliberately
    do NOT return rec_target: the inverse must reconstruct it from h alone using
    its OWN assumed alpha, which is part of the misspecification test.
    """
    m = float(mismatch)
    N = dist.shape[0]
    Wt = torch.as_tensor(W_rec, dtype=torch.float32, device=device)
    bias = torch.zeros(N, dtype=torch.float32, device=device)

    # input layer: random fixed projection (the foreign net has its own readin)
    g_in = torch.Generator().manual_seed(seed + 31)
    W_in = torch.randn(task_input_size(task), N, generator=g_in) / (N ** 0.5)
    W_in = W_in.to(device)

    # distributed-delay group weights (numpy -> torch per lag)
    groups = build_distributed_delay_groups(dist, velocity, max_delay, min_delay,
                                            delay_spread, m)
    group_t = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
               for k, v in groups.items()}
    # effective per-lag connection matrix: Wdist_k * W_rec  (row i, col j)
    Wk = {k: (group_t[k] * Wt) for k in group_t}

    def phi(x):
        # blend relu (matched) -> tanh (foreign) by mismatch
        return (1 - m) * torch.relu(x) + m * torch.tanh(gain * x)

    all_h, all_proj = [], []
    gen = torch.Generator().manual_seed(seed + 4242)
    for rep in range(n_repeats):
        inputs, _, _ = task.generate(n_trials, generator=gen)
        x = inputs.to(device)
        B, T, _ = x.shape
        proj = torch.einsum("btf,fn->btn", x, W_in)  # (B,T,N) input drive
        # history ring of phi(h): index k holds phi(h) from k steps ago
        hist = torch.zeros(max_delay + 1, B, N, device=device)
        h = torch.zeros(B, N, device=device)
        hs = []
        for t in range(T):
            ph = phi(h)
            hist = torch.roll(hist, shifts=1, dims=0)
            hist[1] = ph
            # distributed delayed linear input
            lin = torch.zeros(B, N, device=device)
            for k, Wkk in Wk.items():
                if k <= max_delay:
                    lin = lin + hist[k] @ Wkk.t()  # (B,N): sum_j Wk[i,j] phi(h_j)[t-k]
            gate = torch.sigmoid(beta_gate * h)
            drive = (1 - m) * lin + m * gate * lin
            pre = proj[:, t] + drive + bias
            h = (1 - alpha_gen) * h + alpha_gen * pre
            hs.append(h.clone())
        all_h.append(torch.stack(hs, dim=1))
        all_proj.append(proj)
    H = torch.cat(all_h, dim=0).cpu().numpy().astype(np.float64)
    P = torch.cat(all_proj, dim=0).cpu().numpy().astype(np.float64)
    return H, P


def task_input_size(task):
    # MemoryProTask emits one-hot cue channels + go cue; infer from a sample
    if not hasattr(task, "_inp_size"):
        s, _, _ = task.generate(1)
        task._inp_size = s.shape[-1]
    return task._inp_size


def make_foreign_weight(N, dist, seed, spatial_decay, spectral_radius=1.5):
    """A foreign recurrent kernel: distance-decaying random connectivity.

    Not trained on any task and not the SDRNN's W - a spatially-structured
    random matrix so dynamics are recurrence-driven and distance-coupled (the
    delay must shape activity for v to be identifiable). Scaled to a chosen
    spectral radius; the saturating tanh keeps even sr>1 bounded/stable.
    """
    rng = np.random.default_rng(seed + 5)
    W = rng.standard_normal((N, N))
    W *= np.exp(-dist / (spatial_decay + 1e-9))   # local-ish coupling
    np.fill_diagonal(W, 0.0)
    sr = np.max(np.abs(np.linalg.eigvals(W)))
    W *= spectral_radius / (sr + 1e-9)
    return W.astype(np.float64)


# The inverse (reuses pinn_inverse scoring) applied to foreign activity.
def reconstruct_rec_target_approx(states, proj, alpha_assumed):
    """Inverse's APPROXIMATE recurrent target from foreign states.

    The inverse does NOT know the foreign generator. It assumes a leaky update
    with its own alpha_assumed and NO bias knowledge, and reconstructs
        rec_t ~= (h_t - (1-a) h_{t-1})/a - proj_t.
    For the foreign generator this is a MISSPECIFIED measurement (wrong alpha,
    missing gate/saturation), which is the whole point. (B,T-1,N).
    """
    s_t = states[:, 1:, :]
    s_prev = states[:, :-1, :]
    rec = (s_t - (1 - alpha_assumed) * s_prev) / alpha_assumed - proj[:, 1:, :]
    return rec


def add_obs_noise(states, noise, seed):
    """Add Gaussian observation noise scaled by per-unit state std."""
    if noise <= 0:
        return states
    rng = np.random.default_rng(seed + 99)
    sd = states.std(axis=(0, 1), keepdims=True) + 1e-9
    return states + rng.standard_normal(states.shape) * noise * sd


def run_inverse_on_activity(states, proj, dist, cands, *, alpha_assumed,
                            max_delay, min_delay, ridge, rank, seed,
                            n_null=2):
    """Apply the (reused) free-kernel velocity scan to foreign activity.

    Observable assumed by the inverse: rate = relu(h) (the SDRNN's rate var --
    the STRICT misspecification: the foreign rates may be tanh). Kernel is
    marginalized (free fit), so this tests VELOCITY recovery, as instructed.

    Returns residual curves (true & shuffled geom), v_hat, floors, interior
    flag, and v_true-localization diagnostics. The shuffled null is averaged
    over `n_null` independent off-diagonal permutations (more robust floor).
    """
    rates = np.maximum(states, 0.0)  # inverse assumes relu rates
    rec_target = reconstruct_rec_target_approx(states, proj, alpha_assumed)

    # TRUE geometry free-kernel scan
    res_true_full, _ = scan_freekernel(
        rates, rec_target, dist, cands, max_delay, min_delay, ridge, rank)
    # SHUFFLED geometry free-kernel scan, averaged over several shuffles (null's
    # best shot: free kernel + best-over-v + averaged so a lucky shuffle does not
    # set the floor).
    shuf_curves = []
    for s in range(n_null):
        d_shuf = shuffle_geometry(dist, seed * 17 + s)
        rs, _ = scan_freekernel(
            rates, rec_target, d_shuf, cands, max_delay, min_delay, ridge, rank)
        shuf_curves.append(rs)
    res_shuf_full = np.mean(shuf_curves, axis=0)
    # most-favourable single shuffle floor (hardest null): lowest min over shuffles
    floor_shuf_best = float(min(c.min() for c in shuf_curves))

    argmin = int(np.argmin(res_true_full))
    v_hat = float(cands[argmin])
    interior = 0 < argmin < len(cands) - 1
    floor_true = float(res_true_full.min())
    floor_shuf = float(res_shuf_full.min())
    ratio = floor_shuf / (floor_true + 1e-12)
    ratio_best = floor_shuf_best / (floor_true + 1e-12)
    dip = float((res_true_full.max() - floor_true) / (res_true_full.max() + 1e-12))
    # v_true-resolved: residual at the candidate nearest v_true on true vs shuf
    i_vt = int(np.argmin(np.abs(cands - cands[len(cands) // 2])))  # set by caller grid
    return {
        "v_hat": v_hat, "interior": interior,
        "floor_true": floor_true, "floor_shuf": floor_shuf,
        "floor_shuf_best": floor_shuf_best,
        "ratio_shuf_over_true": ratio, "ratio_best_shuf_over_true": ratio_best,
        "dip": dip,
        "res_true_full": res_true_full.tolist(),
        "res_shuf_full": res_shuf_full.tolist(),
        "argmin": argmin,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--v-true", type=float, default=0.08, dest="v_true")
    ap.add_argument("--max-delay", type=int, default=14, dest="max_delay")
    ap.add_argument("--min-delay", type=int, default=1, dest="min_delay")
    ap.add_argument("--n-trials", type=int, default=64, dest="n_trials")
    ap.add_argument("--n-repeats", type=int, default=6, dest="n_repeats")
    ap.add_argument("--n-cands", type=int, default=15, dest="n_cands")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--rank", type=int, default=8, dest="rank")
    # foreign-generator knobs
    ap.add_argument("--alpha-gen", type=float, default=0.2, dest="alpha_gen")
    ap.add_argument("--alpha-assumed", type=float, default=0.2, dest="alpha_assumed")
    ap.add_argument("--gain", type=float, default=1.5)
    ap.add_argument("--beta-gate", type=float, default=2.0, dest="beta_gate")
    ap.add_argument("--delay-spread", type=float, default=2.0, dest="delay_spread")
    ap.add_argument("--spatial-decay", type=float, default=2.0, dest="spatial_decay")
    ap.add_argument("--spectral-radius", type=float, default=1.5, dest="spectral_radius")
    ap.add_argument("--mismatch", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.05, 0.15])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "results" / "inverse" / "foreigninv_mismatch.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = 2
        args.mismatch = [0.0, 1.0]
        args.noise = [0.0, 0.1]
        args.n_repeats = 3
        args.n_trials = 32
        args.n_cands = 11

    device = torch.device(args.device if args.device != "mps" or
                          torch.backends.mps.is_available() else "cpu")
    cands = np.geomspace(args.v_true / 3.5, args.v_true * 3.5, args.n_cands)
    task = MemoryProTask(n_choices=4, cue_steps=2, delay_steps=8,
                         response_steps=2, noise=0.2)

    print(f"FOREIGN-INV mismatch_rnn  v_true={args.v_true} hidden={args.hidden} "
          f"seeds={args.seeds} device={device}")
    print(f"mismatch grid={args.mismatch}  noise grid={args.noise}")
    print(f"candidates={np.round(cands,4).tolist()}", flush=True)

    # fixed geometry (grid) shared across seeds; coords-> distance matrix
    coords = grid_coordinates(args.hidden, dim=3).numpy()
    dist0 = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    t0 = time.time()
    cells = []  # one per (mismatch, noise): aggregated over seeds
    per_run = []  # one per (mismatch, noise, seed)
    for m in args.mismatch:
        for noise in args.noise:
            vhats, errs, ratios, floors_t, floors_s, dips, interiors = \
                [], [], [], [], [], [], []
            for seed in range(args.seeds):
                # foreign weight depends on seed (different draws); same geometry
                W_rec = make_foreign_weight(args.hidden, dist0, seed,
                                            args.spatial_decay,
                                            args.spectral_radius)
                H, P = generate_mismatch_activity(
                    W_rec, dist0, args.v_true, task, device,
                    mismatch=m, alpha_gen=args.alpha_gen, gain=args.gain,
                    beta_gate=args.beta_gate, delay_spread=args.delay_spread,
                    max_delay=args.max_delay, min_delay=args.min_delay,
                    n_trials=args.n_trials, n_repeats=args.n_repeats, seed=seed)
                if not np.all(np.isfinite(H)):
                    print(f"  [m={m} noise={noise} seed={seed}] NON-FINITE activity"
                          f" - generator diverged; skipping", flush=True)
                    continue
                Hn = add_obs_noise(H, noise, seed)
                r = run_inverse_on_activity(
                    Hn, P, dist0, cands, alpha_assumed=args.alpha_assumed,
                    max_delay=args.max_delay, min_delay=args.min_delay,
                    ridge=args.ridge, rank=args.rank, seed=seed)
                err = abs(r["v_hat"] / args.v_true - 1.0)
                vhats.append(r["v_hat"]); errs.append(err)
                ratios.append(r["ratio_best_shuf_over_true"])  # hardest null
                floors_t.append(r["floor_true"]); floors_s.append(r["floor_shuf_best"])
                dips.append(r["dip"]); interiors.append(r["interior"])
                per_run.append({"mismatch": m, "noise": noise, "seed": seed,
                                **{k: r[k] for k in
                                   ("v_hat", "interior", "floor_true",
                                    "floor_shuf", "floor_shuf_best",
                                    "ratio_shuf_over_true",
                                    "ratio_best_shuf_over_true", "dip",
                                    "argmin", "res_true_full", "res_shuf_full")},
                                "err": err})
            if not vhats:
                continue
            vhats = np.array(vhats); errs = np.array(errs); ratios = np.array(ratios)
            cell = {
                "mismatch": m, "noise": noise, "n_seeds": len(vhats),
                "vhat_mean": float(vhats.mean()), "vhat_std": float(vhats.std()),
                "err_mean": float(errs.mean()), "err_std": float(errs.std()),
                "ratio_mean": float(ratios.mean()), "ratio_std": float(ratios.std()),
                "ratio_min": float(ratios.min()),
                "floor_true_mean": float(np.mean(floors_t)),
                "floor_shuf_mean": float(np.mean(floors_s)),
                "dip_mean": float(np.mean(dips)),
                "interior_frac": float(np.mean(interiors)),
                "vhats": vhats.tolist(),
            }
            cells.append(cell)
            print(f"  m={m:.2f} noise={noise:.2f} | v_hat={cell['vhat_mean']:.4f}"
                  f"±{cell['vhat_std']:.4f} (true {args.v_true}) "
                  f"err={cell['err_mean']*100:5.1f}±{cell['err_std']*100:4.1f}%  "
                  f"null_ratio={cell['ratio_mean']:.1f}x  dip={cell['dip_mean']:.3f}  "
                  f"interior={cell['interior_frac']:.2f}  n={cell['n_seeds']}",
                  flush=True)

    elapsed = time.time() - t0

    # verdict. m=0 (matched basis) should recover well; foreign cells are the test.
    def cell_at(m, noise):
        for c in cells:
            if abs(c["mismatch"] - m) < 1e-9 and abs(c["noise"] - noise) < 1e-9:
                return c
        return None

    foreign_cells = [c for c in cells if c["mismatch"] >= 0.99]  # fully foreign
    # recovery criterion: err<35% AND true-geom floor below the HARDEST shuffled
    # floor (best of 3 shuffles) by >1.5x AND interior min on majority of seeds.
    def passes(c):
        return (c["err_mean"] < 0.35 and c["ratio_mean"] > 1.5 and
                c["interior_frac"] >= 0.5)
    clean_foreign = cell_at(1.0, args.noise[0]) if args.mismatch and 1.0 in [round(x,6) for x in args.mismatch] else None
    # find the highest-mismatch cell at lowest noise that still passes
    lowest_noise = min(args.noise)
    foreign_low_noise = [c for c in cells if c["noise"] == lowest_noise
                         and c["mismatch"] >= 0.99]
    recovered_clean_foreign = bool(foreign_low_noise and passes(foreign_low_noise[0]))

    # degradation: max mismatch tolerated at lowest noise
    tol = [c["mismatch"] for c in cells if c["noise"] == lowest_noise and passes(c)]
    max_mismatch_ok = max(tol) if tol else None

    summary = {
        "generator": "mismatch_rnn (tanh-saturating + distributed/fractional "
                     "delays + multiplicative conductance gate), distance-decay "
                     "random foreign W, NOT a trained SDRNN and NOT the inverse's basis",
        "non_tautological": "yes - generator functional form differs from the "
                            "inverse's relu/integer-delay/additive linear basis; "
                            "v enters only via delay=distance/v",
        "v_true": args.v_true, "hidden": args.hidden, "seeds": args.seeds,
        "candidates": cands.tolist(),
        "alpha_gen": args.alpha_gen, "alpha_assumed": args.alpha_assumed,
        "gain": args.gain, "beta_gate": args.beta_gate,
        "delay_spread": args.delay_spread, "spatial_decay": args.spatial_decay,
        "spectral_radius": args.spectral_radius,
        "null_floor_def": "ratio = best(min-over-v over 2 shuffles) / true(min-over-v); >1 means true geom unreachable by any shuffled geometry",
        "mismatch_grid": list(args.mismatch), "noise_grid": list(args.noise),
        "max_delay": args.max_delay, "ridge": args.ridge, "rank": args.rank,
        "recovered_clean_foreign": recovered_clean_foreign,
        "max_mismatch_ok_at_low_noise": max_mismatch_ok,
        "cells": cells, "per_run": per_run,
        "elapsed_sec": elapsed,
    }
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(summary, indent=2))
    print(f"\nelapsed {elapsed:.1f}s   wrote {outpath}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
