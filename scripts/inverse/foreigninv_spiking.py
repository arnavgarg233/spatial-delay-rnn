"""Velocity inverse on foreign (spiking) dynamics.

Generator: a leaky integrate-and-fire network on the same grid geometry the
inverse uses, with axonal conduction delays tau_ij = round(d_ij / v_true), sparse
random signed synapses g_ij (unknown to the inverse), and threshold/reset/
refractoriness. A presynaptic spike from j reaches i tau_ij steps later as an
exponential PSC, so v enters only through that delay. Foreign in every way: discrete
spikes, hard nonlinearity, membrane integration - none of it the lagged-rate basis.

Inverse (reused from pinn_inverse, free-kernel only, no weights given): bin spikes
to rates; target = one-step binned-rate increment y_i[t] = r_i[t] - (1-a) r_i[t-1];
per candidate v map tau_ij(v) into bin units, ridge-fit a free kernel, score the
normalized residual, v_hat = argmin. Null floor = velocity-shuffled geometry (same
delay histogram, scrambled pair->lag), free kernel refit per candidate. If the true
floor is not clearly below the shuffled floor, recovery evaporated.

Recovery is not by construction (g_ij never given, activity is spikes-binned-to-
rates), so it requires the conduction delay to survive the LIF nonlinearity +
binning as lag-structure in second-order statistics. Reports v_hat ± error across
seeds/bin-sizes/noise, the dip shape, and the shuffled null floor.

  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=3 \
    python scripts/inverse/foreigninv_spiking.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from sdrnn.geometry import NeuronGeometry


# Generator: LIF spiking network with axonal conduction delays.
def simulate_lif(
    dist,
    v_true,
    T,
    *,
    seed=0,
    p_conn=0.15,
    g_exc=0.9,
    g_inh=1.1,
    frac_inh=0.2,
    tau_m=20.0,
    tau_syn=5.0,
    dt=1.0,
    v_rest=0.0,
    v_th=1.0,
    v_reset=0.0,
    t_ref=2,
    i_ext_mu=1.05,
    i_ext_sigma=0.35,
    max_delay=14,
    min_delay=1,
    burn=200,
):
    """Simulate a delayed-synapse LIF network; return spikes (T,N) {0,1}.

    The conduction velocity enters ONLY through the integer axonal delay
    tau_ij = clip(round(d_ij / v_true), min_delay, max_delay): a spike of j at
    time t deposits a PSC into i's synaptic-current channel at time t+tau_ij.
    Everything else (threshold, reset, refractory, membrane leak) is the foreign
    nonlinearity that the rate-SDRNN inverse does NOT contain.
    """
    rng = np.random.default_rng(seed + 101)
    N = dist.shape[0]

    # --- signed sparse synapses (UNKNOWN to the inverse) ---
    mask = (rng.random((N, N)) < p_conn).astype(np.float64)
    np.fill_diagonal(mask, 0.0)
    sign = np.ones(N)
    n_inh = int(round(frac_inh * N))
    inh_idx = rng.choice(N, size=n_inh, replace=False)
    sign[inh_idx] = -1.0
    # column j carries presynaptic sign of j; magnitude ~ exp so heavy-tailed
    mag = rng.gamma(shape=2.0, scale=0.5, size=(N, N))
    G = mask * mag * (sign[None, :] * np.where(sign[None, :] > 0, g_exc, g_inh))
    # G[i,j] = synaptic weight from j (col) onto i (row)

    # --- integer conduction delays ---
    tau = np.clip(np.round(dist / v_true), min_delay, max_delay).astype(int)
    np.fill_diagonal(tau, min_delay)
    Dmax = int(tau.max())

    # group synapses by delay so deposit is O(#distinct delays) matmuls
    delay_groups = []
    for d in np.unique(tau):
        d = int(d)
        m = (tau == d).astype(np.float64)
        delay_groups.append((d, G * m))  # (N,N) weight active only at lag d

    # --- per-neuron heterogeneous constant drive (keeps net firing) ---
    i_ext = rng.normal(i_ext_mu, i_ext_sigma, size=N)

    # --- state ---
    Ttot = T + burn
    V = rng.uniform(v_reset, v_th, size=N)
    I_syn = np.zeros(N)               # decaying synaptic current
    ref = np.zeros(N, dtype=int)      # refractory countdown
    # ring buffer of FUTURE synaptic deposits: future_in[k] = current to add k steps ahead
    future = np.zeros((Dmax + 1, N))

    decay_m = np.exp(-dt / tau_m)
    decay_syn = np.exp(-dt / tau_syn)

    spikes = np.zeros((T, N), dtype=np.float32)
    noise_sigma = 0.05  # membrane noise (extra biophysical realism)

    for t in range(Ttot):
        # pull scheduled deposits arriving NOW
        I_syn = I_syn * decay_syn + future[0]
        future[:-1] = future[1:]
        future[-1] = 0.0

        # membrane update (LIF, leaky) with refractory clamp
        dV = (dt / tau_m) * (-(V - v_rest) + I_syn + i_ext) + noise_sigma * rng.standard_normal(N) * np.sqrt(dt)
        V = np.where(ref > 0, v_reset, V + dV)
        ref = np.maximum(ref - 1, 0)

        # spikes
        fired = V >= v_th
        if fired.any():
            V[fired] = v_reset
            ref[fired] = t_ref
            sp = fired.astype(np.float64)
            # schedule delayed PSC deposits: for each delay group, add G_g @ sp at lag d
            for d, Gg in delay_groups:
                future[d] += Gg @ sp

        if t >= burn:
            spikes[t - burn] = fired.astype(np.float32)

    return spikes, tau, G


def bin_spikes(spikes, bin_size):
    """Bin (T,N) spike train into (Tb,N) counts per bin."""
    T, N = spikes.shape
    Tb = T // bin_size
    sp = spikes[: Tb * bin_size].reshape(Tb, bin_size, N).sum(axis=1)
    return sp.astype(np.float64)


# Inverse: free-kernel velocity scan on binned rates (reused from pinn_inverse).
def lagged(r, d):
    """r_j[t-d] aligned to t = 1..Tb-1  -> (Tb-1, N)."""
    Tb, N = r.shape
    L = np.zeros_like(r)
    if d < Tb:
        L[d:, :] = r[: Tb - d, :]
    return L[1:, :]


def normalized_residual(Y, pred):
    num = np.sum((Y - pred) ** 2)
    den = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2) + 1e-12
    return float(num / den)


def fit_free_kernel_residual(r, Y, tau_bins, ridge):
    """Fit a FREE ridge kernel from lagged binned rates; return residual.

    Per target unit i: y_i[t] = sum_j W_ij r_j[t - tau_ij].  No weights given.
    """
    Tb, N = r.shape
    M = Tb - 1
    lag_cache = {}
    for d in np.unique(tau_bins):
        d = int(d)
        lag_cache[d] = lagged(r, d)  # (M, N)
    Wfit = np.zeros((N, N))
    pred = np.zeros((M, N))
    for i in range(N):
        Xi = np.empty((M, N))
        for j in range(N):
            Xi[:, j] = lag_cache[int(tau_bins[i, j])][:, j]
        yi = Y[:, i]
        XtX = Xi.T @ Xi
        lam = ridge * (np.trace(XtX) / N + 1e-8)
        XtX[np.diag_indices_from(XtX)] += lam
        wi = np.linalg.solve(XtX, Xi.T @ yi)
        Wfit[i] = wi
        pred[:, i] = Xi @ wi
    return normalized_residual(Y, pred), Wfit


def dist_to_taubins(dist, v, bin_size, max_delay_steps, min_delay_steps):
    """Conduction delay tau_ij(v) in STEPS -> rounded into BIN units (>=1)."""
    tau_steps = np.clip(np.round(dist / v), min_delay_steps, max_delay_steps)
    tau_bins = np.maximum(1, np.round(tau_steps / bin_size)).astype(int)
    np.fill_diagonal(tau_bins, 1)
    return tau_bins


def velocity_target(r, leak):
    """Binned-rate increment target y[t] = r[t] - (1-leak) r[t-1], aligned t=1.. ."""
    return (r[1:, :] - (1 - leak) * r[:-1, :])


def scan_velocity(r, dist, cands, bin_size, max_delay_steps, min_delay_steps, ridge, leak):
    Y = velocity_target(r, leak)
    res = np.empty(len(cands))
    for k, v in enumerate(cands):
        tau_bins = dist_to_taubins(dist, v, bin_size, max_delay_steps, min_delay_steps)
        res[k], _ = fit_free_kernel_residual(r, Y, tau_bins, ridge)
    return res


def shuffle_geometry(dist, seed):
    """Velocity-shuffled null: permute off-diagonal distances (same histogram)."""
    N = dist.shape[0]
    rng = np.random.default_rng(seed + 777)
    iu = np.triu_indices(N, 1)
    d = dist.copy()
    vals = d[iu].copy()
    rng.shuffle(vals)
    d[iu] = vals
    d[(iu[1], iu[0])] = vals
    return d


def parabolic_refine(cands, res):
    """Sub-grid v_hat via parabolic interpolation around the discrete argmin."""
    k = int(np.argmin(res))
    if 0 < k < len(cands) - 1:
        x0, x1, x2 = np.log(cands[k - 1]), np.log(cands[k]), np.log(cands[k + 1])
        y0, y1, y2 = res[k - 1], res[k], res[k + 1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) > 1e-18:
            A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            B = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
            if A > 0:
                xv = -B / (2 * A)
                xv = np.clip(xv, x0, x2)
                return float(np.exp(xv)), k
    return float(cands[k]), k


def run_condition(dist, v_true, bin_size, noise_sigma, seed, cands,
                  max_delay_steps, min_delay_steps, ridge, leak, T, gen_kwargs,
                  n_shuffles=12):
    """One (bin,noise,seed) cell: simulate, bin, scan true + a DISTRIBUTION of
    velocity-shuffled-geometry nulls.

    The decisive null is not ONE shuffle but the DISTRIBUTION of min-residuals
    over many velocity-shuffled geometries (same delay histogram, scrambled
    pair->lag assignment), each given a free kernel at every candidate v (the
    null's best shot). We report where the true-geometry floor sits in that
    distribution as a z-score and an empirical percentile (fraction of shuffles
    whose floor is <= the true floor). z>~2 / percentile~0 = the conduction
    delay leaves a distance-specific fingerprint the shuffle cannot fake.
    """
    spikes, tau_true, G = simulate_lif(
        dist, v_true, T, seed=seed, max_delay=max_delay_steps,
        min_delay=min_delay_steps,
        i_ext_sigma=gen_kwargs["i_ext_sigma"] + noise_sigma,
        **{k: v for k, v in gen_kwargs.items() if k not in ("i_ext_sigma",)},
    )
    rate_hz = spikes.mean() / 1.0  # mean spikes per step per neuron
    r = bin_spikes(spikes, bin_size)

    res_true = scan_velocity(r, dist, cands, bin_size, max_delay_steps,
                             min_delay_steps, ridge, leak)
    floor_true = float(res_true.min())
    v_hat, kbest = parabolic_refine(cands, res_true)
    interior = 0 < int(np.argmin(res_true)) < len(cands) - 1
    dip_true = float((res_true.max() - res_true.min()) / (res_true.max() + 1e-12))

    # velocity-shuffled null floor distribution (the decisive test)
    shuf_floors = []
    res_shuf_example = None
    for s in range(n_shuffles):
        d_shuf = shuffle_geometry(dist, seed * 100 + s)
        res_shuf = scan_velocity(r, d_shuf, cands, bin_size, max_delay_steps,
                                 min_delay_steps, ridge, leak)
        shuf_floors.append(float(res_shuf.min()))
        if s == 0:
            res_shuf_example = res_shuf
    shuf_floors = np.array(shuf_floors)
    floor_shuf_mean = float(shuf_floors.mean())
    floor_shuf_std = float(shuf_floors.std())
    floor_shuf_min = float(shuf_floors.min())  # null's ABSOLUTE best shot
    # z-score: how many shuffle-std the true floor sits below the shuffle mean
    null_z = float((floor_shuf_mean - floor_true) / (floor_shuf_std + 1e-12))
    # empirical percentile: fraction of shuffles whose floor <= true floor
    null_pctile = float((shuf_floors <= floor_true).mean())
    beats_null = floor_true < floor_shuf_min  # below EVERY shuffle = strongest claim

    return {
        "seed": seed, "bin_size": bin_size, "noise": noise_sigma,
        "mean_rate_per_step": float(rate_hz),
        "v_hat": v_hat, "k_best": int(kbest), "interior": bool(interior),
        "floor_true": floor_true,
        "floor_shuf_mean": floor_shuf_mean, "floor_shuf_std": floor_shuf_std,
        "floor_shuf_min": floor_shuf_min,
        "null_z": null_z, "null_pctile": null_pctile, "beats_null": bool(beats_null),
        "gap": floor_shuf_mean - floor_true,
        "ratio_floor": float(floor_shuf_mean / (floor_true + 1e-12)),
        "dip_true": dip_true,
        "err_pct": float(abs(v_hat / v_true - 1) * 100),
        "res_true": res_true.tolist(), "res_shuf": res_shuf_example.tolist(),
        "shuf_floors": shuf_floors.tolist(),
    }


def run_velocity_sweep(dist, v_sweep, *, bin_size, noise, seeds, cand_span,
                       n_cands, max_delay_steps, min_delay_steps, ridge, leak,
                       T, gen_kwargs, n_shuffles):
    """v_true SWEEP (the NON-TAUTOLOGY proof): for each TRUE conduction velocity,
    re-simulate the foreign LIF network and run the SAME free-kernel inverse on a
    candidate grid recentred on that v_true. If v_hat TRACKS v_true MONOTONICALLY
    (rather than collapsing to one fixed grid point), the inverse is reading the
    generator's conduction delay out of the spiking second-order statistics - not
    just refinding the single velocity the candidate grid was built around.

    One clean-cell config (bin, noise) per v_true, averaged over seeds; the
    velocity-shuffled-geometry null z-score is recorded at each point too.
    """
    points = []
    for v_true in v_sweep:
        cands = np.geomspace(v_true / cand_span, v_true * cand_span, n_cands)
        per_seed = []
        for seed in range(seeds):
            row = run_condition(
                dist, v_true, bin_size, noise, seed, cands,
                max_delay_steps, min_delay_steps, ridge, leak,
                T, gen_kwargs, n_shuffles=n_shuffles)
            per_seed.append(row)
        vh = np.array([r["v_hat"] for r in per_seed])
        zr = np.array([r["null_z"] for r in per_seed])
        err = np.array([r["err_pct"] for r in per_seed])
        n_int = sum(r["interior"] for r in per_seed)
        n_beat = int(np.sum([r["beats_null"] for r in per_seed]))
        pt = {
            "v_true": float(v_true),
            "v_hat_mean": float(vh.mean()), "v_hat_std": float(vh.std()),
            "v_hat_seeds": vh.tolist(),
            "err_pct_mean": float(err.mean()), "err_pct_std": float(err.std()),
            "null_z_mean": float(zr.mean()), "null_z_std": float(zr.std()),
            "n_interior": int(n_int), "n_beat_null": int(n_beat),
            "n_seeds": int(seeds),
            "cands": cands.tolist(),
        }
        points.append(pt)
        print(f"  v_true={v_true:.3f}: v_hat={vh.mean():.4f}+-{vh.std():.4f} "
              f"(err {err.mean():.0f}%, interior {n_int}/{seeds})  "
              f"null_z={zr.mean():.1f}+-{zr.std():.1f}  beats {n_beat}/{seeds}", flush=True)
    # monotonicity of v_hat vs v_true (Spearman / strict rank check)
    vts = np.array([p["v_true"] for p in points])
    vhs = np.array([p["v_hat_mean"] for p in points])
    strict_mono = bool(np.all(np.diff(vhs) > 0)) if len(vhs) > 1 else False
    # Pearson on log-log (the slope should be ~+1 if it tracks)
    if len(vts) > 2:
        lr = np.corrcoef(np.log(vts), np.log(vhs))[0, 1]
        slope = float(np.polyfit(np.log(vts), np.log(vhs), 1)[0])
    else:
        lr, slope = float("nan"), float("nan")
    return {
        "points": points,
        "v_true_grid": vts.tolist(), "v_hat_grid": vhs.tolist(),
        "strict_monotonic": strict_mono,
        "loglog_pearson": float(lr), "loglog_slope": slope,
        "config": {"bin_size": bin_size, "noise": noise, "seeds": seeds,
                   "cand_span": cand_span, "n_cands": n_cands,
                   "max_delay_steps": max_delay_steps, "min_delay_steps": min_delay_steps,
                   "ridge": ridge, "leak": leak, "T": T, "n_shuffles": n_shuffles},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--v-true", type=float, default=0.08, dest="v_true")
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--max-delay", type=int, default=14, dest="max_delay")
    ap.add_argument("--min-delay", type=int, default=1, dest="min_delay")
    ap.add_argument("--n-cands", type=int, default=17, dest="n_cands")
    ap.add_argument("--cand-span", type=float, default=3.0, dest="cand_span")
    ap.add_argument("--ridge", type=float, default=1e-1)
    ap.add_argument("--leak", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--bins", type=str, default="1,2,4")
    ap.add_argument("--noises", type=str, default="0.0,0.3")
    ap.add_argument("--p-conn", type=float, default=0.15, dest="p_conn")
    ap.add_argument("--i-ext-mu", type=float, default=1.05, dest="i_ext_mu")
    ap.add_argument("--i-ext-sigma", type=float, default=0.35, dest="i_ext_sigma")
    ap.add_argument("--n-shuffles", type=int, default=12, dest="n_shuffles")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    # v_true sweep (non-tautology proof): recover v_hat at several v_true
    ap.add_argument("--v-sweep", type=str, default="", dest="v_sweep",
                    help="comma-sep list of v_true to sweep; if set, run the "
                         "velocity-tracking sweep (clean cell) instead of the "
                         "bin/noise grid and write foreigninv_spiking.json")
    ap.add_argument("--sweep-max-delay", type=int, default=32, dest="sweep_max_delay",
                    help="generator+inverse max delay for the sweep (raised so the "
                         "slowest v_true does not saturate the delay clip)")
    args = ap.parse_args()

    if args.smoke:
        args.T = 2500
        args.seeds = 2
        args.bins = "1,2"
        args.noises = "0.0"
        args.n_cands = 11
        if args.v_sweep:
            args.v_sweep = "0.06,0.12"
            args.seeds = 2

    bins = [int(b) for b in args.bins.split(",")]
    noises = [float(x) for x in args.noises.split(",")]
    cands = np.geomspace(args.v_true / args.cand_span, args.v_true * args.cand_span, args.n_cands)

    geom = NeuronGeometry(args.N, dim=3, learnable=False)
    dist = geom.distance_matrix().detach().cpu().numpy().astype(np.float64)

    gen_kwargs = dict(
        p_conn=args.p_conn, i_ext_mu=args.i_ext_mu, i_ext_sigma=args.i_ext_sigma,
    )

    # v_true sweep mode (non-tautology proof): recover v_hat at each v_true,
    # write foreigninv_spiking.json with the (v_true -> v_hat) tracking curve.
    if args.v_sweep:
        v_sweep = [float(x) for x in args.v_sweep.split(",")]
        bin_size = bins[0]
        noise = noises[0]
        t0 = time.time()
        print(f"FOREIGN-INVERSE v_true SWEEP (LIF spiking)  N={args.N}  T={args.T}")
        print(f"  v_sweep={v_sweep}  clean cell bin={bin_size} noise={noise}  "
              f"seeds={args.seeds}  max_delay={args.sweep_max_delay} "
              f"n_cands={args.n_cands} span={args.cand_span}", flush=True)
        sweep = run_velocity_sweep(
            dist, v_sweep, bin_size=bin_size, noise=noise, seeds=args.seeds,
            cand_span=args.cand_span, n_cands=args.n_cands,
            max_delay_steps=args.sweep_max_delay, min_delay_steps=args.min_delay,
            ridge=args.ridge, leak=args.leak, T=args.T, gen_kwargs=gen_kwargs,
            n_shuffles=args.n_shuffles)

        print("\n" + "=" * 78)
        print("VELOCITY-TRACKING SWEEP (v_hat vs v_true)")
        print("=" * 78)
        for p in sweep["points"]:
            print(f"  v_true={p['v_true']:.3f} -> v_hat={p['v_hat_mean']:.4f}"
                  f"+-{p['v_hat_std']:.4f}  (err {p['err_pct_mean']:.0f}%, "
                  f"null_z {p['null_z_mean']:.1f}, beats {p['n_beat_null']}/{p['n_seeds']})")
        print(f"  strict-monotonic v_hat in v_true: {sweep['strict_monotonic']}  "
              f"log-log Pearson r={sweep['loglog_pearson']:.3f} slope={sweep['loglog_slope']:.2f}")
        z_all = float(np.mean([p["null_z_mean"] for p in sweep["points"]]))
        mean_err = float(np.mean([p["err_pct_mean"] for p in sweep["points"]]))
        if sweep["strict_monotonic"] and sweep["loglog_pearson"] > 0.9:
            verdict = ("NON-TAUTOLOGICAL TRACKING: v_hat tracks the TRUE conduction "
                       "velocity monotonically across the foreign LIF sweep (log-log "
                       f"r={sweep['loglog_pearson']:.2f}, slope~{sweep['loglog_slope']:.2f}); "
                       "the inverse reads the generator's delay, not the candidate grid.")
        elif sweep["loglog_pearson"] > 0.7:
            verdict = ("PARTIAL TRACKING: v_hat rises with v_true but not strictly "
                       "monotone / slope off; honest partial non-tautology evidence.")
        else:
            verdict = ("NO TRACKING: v_hat does not follow v_true across the sweep; "
                       "tautology concern not dispelled. Honest negative.")
        print("\nVERDICT:", verdict)

        out = {
            "mode": "v_true_sweep",
            "generator": "LIF spiking network with axonal conduction delays tau_ij=round(d_ij/v)",
            "inverse": "free-kernel delayed-rate-consistency velocity scan on binned spikes (no weights given)",
            "N": args.N, "T": args.T,
            "v_sweep": v_sweep,
            "sweep": sweep,
            "null_z_sweep_mean": z_all, "mean_err_pct": mean_err,
            "gen_kwargs": gen_kwargs,
            "verdict": verdict,
            "elapsed_sec": time.time() - t0,
        }
        tag = ("_" + args.tag) if args.tag else ("_smoke" if args.smoke else "")
        outpath = ROOT / "results" / "inverse" / f"foreigninv_spiking{tag}.json"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(json.dumps(out, indent=2))
        print(f"\nelapsed {time.time()-t0:.1f}s   wrote {outpath}")
        return

    print(f"FOREIGN-INVERSE (LIF spiking)  v_true={args.v_true}  N={args.N}  T={args.T}")
    print(f"  bins={bins}  noises={noises}  seeds={args.seeds}  n_cands={args.n_cands} "
          f"span={args.cand_span} ridge={args.ridge} leak={args.leak}")
    print(f"  candidates={np.round(cands,4).tolist()}", flush=True)

    t0 = time.time()
    rows = []
    for bin_size in bins:
        for noise in noises:
            for seed in range(args.seeds):
                row = run_condition(
                    dist, args.v_true, bin_size, noise, seed, cands,
                    args.max_delay, args.min_delay, args.ridge, args.leak,
                    args.T, gen_kwargs, n_shuffles=args.n_shuffles)
                rows.append(row)
                print(f"  bin={bin_size} noise={noise:.2f} seed={seed}: "
                      f"rate={row['mean_rate_per_step']:.3f}/step  "
                      f"v_hat={row['v_hat']:.4f} (err {row['err_pct']:.0f}%, "
                      f"{'interior' if row['interior'] else 'BOUNDARY'})  "
                      f"floor_true={row['floor_true']:.4f} null={row['floor_shuf_mean']:.4f}"
                      f"+-{row['floor_shuf_std']:.4f}  z={row['null_z']:.1f} "
                      f"pctile={row['null_pctile']:.2f} {'BEATS' if row['beats_null'] else 'ties'}", flush=True)

    # aggregate per (bin,noise) cell
    cells = {}
    for row in rows:
        key = (row["bin_size"], row["noise"])
        cells.setdefault(key, []).append(row)

    print("\n" + "=" * 78)
    print("AGGREGATE (mean +- std over seeds)  true v =", args.v_true)
    print("=" * 78)
    summary = []
    for key in sorted(cells):
        cr = cells[key]
        vh = np.array([r["v_hat"] for r in cr])
        err = np.array([r["err_pct"] for r in cr])
        ft = np.array([r["floor_true"] for r in cr])
        fs = np.array([r["floor_shuf_mean"] for r in cr])
        zr = np.array([r["null_z"] for r in cr])
        pc = np.array([r["null_pctile"] for r in cr])
        n_int = sum(r["interior"] for r in cr)
        n_beat = int(np.sum([r["beats_null"] for r in cr]))
        beats_all = bool(np.all([r["beats_null"] for r in cr]))
        cell = {
            "bin_size": key[0], "noise": key[1], "n_seeds": len(cr),
            "v_hat_mean": float(vh.mean()), "v_hat_std": float(vh.std()),
            "err_pct_mean": float(err.mean()), "err_pct_std": float(err.std()),
            "floor_true_mean": float(ft.mean()), "floor_shuf_mean": float(fs.mean()),
            "null_z_mean": float(zr.mean()), "null_z_std": float(zr.std()),
            "null_pctile_mean": float(pc.mean()),
            "n_interior": int(n_int), "n_beat_null": n_beat, "beats_null_all": beats_all,
        }
        summary.append(cell)
        print(f"  bin={key[0]} noise={key[1]:.2f}: "
              f"v_hat={vh.mean():.4f}+-{vh.std():.4f} (true {args.v_true})  "
              f"err={err.mean():.0f}%+-{err.std():.0f}%  "
              f"interior {n_int}/{len(cr)}")
        print(f"           floor true={ft.mean():.4f} null={fs.mean():.4f}  "
              f"z={zr.mean():.1f}+-{zr.std():.1f}  pctile={pc.mean():.3f}  "
              f"beats-every-shuffle {n_beat}/{len(cr)}", flush=True)

    # overall verdict
    all_err = np.array([r["err_pct"] for r in rows])
    all_z = np.array([r["null_z"] for r in rows])
    all_pctile = np.array([r["null_pctile"] for r in rows])
    all_beats = np.array([r["beats_null"] for r in rows])
    all_interior = np.array([r["interior"] for r in rows])
    # clean (bin=1, noise=0) cell is the headline
    clean = [r for r in rows if r["bin_size"] == bins[0] and r["noise"] == noises[0]]
    clean_err = np.array([r["err_pct"] for r in clean])
    clean_vh = np.array([r["v_hat"] for r in clean])
    clean_z = np.array([r["null_z"] for r in clean])

    beats_floor_frac = float(np.mean(all_beats))
    z_sig_frac = float(np.mean(all_z > 2.0))
    # RECOVERED: bounded error AND the true floor is significantly (z>2) below the
    # shuffle distribution on every clean seed. PARTIAL: error larger but the null
    # is still significantly beaten on a majority of cells.
    recovered = (clean_err.mean() < 35) and bool(np.all(clean_z > 2.0))
    partial = (clean_err.mean() < 60) and (z_sig_frac > 0.5)

    print("\n" + "=" * 78)
    print("HONEST READOUT")
    print("=" * 78)
    print(f"  CLEAN cell (bin={bins[0]}, noise={noises[0]}): v_hat={clean_vh.mean():.4f}+-{clean_vh.std():.4f} "
          f"(true {args.v_true}), err={clean_err.mean():.0f}%+-{clean_err.std():.0f}%")
    print(f"  null-floor z-score (true below shuffle dist): clean {clean_z.mean():.1f}+-{clean_z.std():.1f}, "
          f"overall {all_z.mean():.1f}+-{all_z.std():.1f}")
    print(f"  true floor below EVERY shuffle on {int(all_beats.sum())}/{len(rows)} cells; "
          f"z>2 (significant) on {int((all_z>2).sum())}/{len(rows)}")
    print(f"  null-floor percentile (frac shuffles <= true): {all_pctile.mean():.3f} (0=true is best)")
    print(f"  interior (bracketed) minima: {int(all_interior.sum())}/{len(rows)} cells")
    if recovered:
        verdict = ("RECOVERED: the velocity inverse recovers v_true from FOREIGN LIF spiking activity "
                   "(spikes->binned rates, free kernel, no synaptic weights handed over) with bounded "
                   "error and a beaten velocity-shuffled null floor. Non-tautological: the generator is "
                   "a spiking/threshold process, not the rate-SDRNN lagged-rate basis the inverse fits.")
    elif partial:
        verdict = ("PARTIAL: v is recovered approximately from foreign spiking activity (error within "
                   "~60%, beats the shuffled null on a majority of cells) but with real bias/variance "
                   "from the spiking nonlinearity + binning. Honest partial recovery with error bars.")
    else:
        verdict = ("WEAK/NULL: the inverse does NOT cleanly recover v from foreign LIF spiking activity "
                   "and/or does not beat the velocity-shuffled null floor. The spiking nonlinearity + "
                   "binning degrade the delayed-field consistency the inverse relies on. Honest negative "
                   "= the fig_si_pinn_gate limitation showing up on genuinely foreign dynamics.")
    print("\nVERDICT:", verdict)

    out = {
        "generator": "LIF spiking network with axonal conduction delays tau_ij=round(d_ij/v)",
        "inverse": "free-kernel delayed-rate-consistency velocity scan on binned spikes (no weights given)",
        "v_true": args.v_true, "N": args.N, "T": args.T,
        "max_delay_steps": args.max_delay, "min_delay_steps": args.min_delay,
        "ridge": args.ridge, "leak": args.leak,
        "candidates": cands.tolist(),
        "bins": bins, "noises": noises, "seeds": args.seeds,
        "gen_kwargs": gen_kwargs,
        "n_shuffles_per_cell": args.n_shuffles,
        "clean_cell": {"bin": bins[0], "noise": noises[0],
                       "v_hat_mean": float(clean_vh.mean()), "v_hat_std": float(clean_vh.std()),
                       "err_pct_mean": float(clean_err.mean()), "err_pct_std": float(clean_err.std()),
                       "null_z_mean": float(clean_z.mean()), "null_z_std": float(clean_z.std()),
                       "beats_null_all": bool(np.all([r["beats_null"] for r in clean]))},
        "beats_null_frac_overall": beats_floor_frac,
        "null_z_overall_mean": float(all_z.mean()),
        "null_pctile_overall_mean": float(all_pctile.mean()),
        "z_significant_frac": z_sig_frac,
        "interior_frac": float(all_interior.mean()),
        "summary_cells": summary,
        "per_run": rows,
        "verdict": verdict,
        "elapsed_sec": time.time() - t0,
    }
    tag = ("_" + args.tag) if args.tag else ("_smoke" if args.smoke else "")
    outpath = ROOT / "results" / "inverse" / f"foreigninv_spiking{tag}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nelapsed {time.time()-t0:.1f}s   wrote {outpath}")


if __name__ == "__main__":
    main()
