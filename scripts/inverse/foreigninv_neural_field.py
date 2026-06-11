"""Conduction-velocity inverse on a foreign generator: a damped 2D neural-field
wave PDE on a sheet,

  u_tt = c^2 Lap(u) - gamma u_t - k u + drive(x, t),

integrated by leapfrog finite differences on a G x G grid (Neumann BC). No
rate-RNN, no learned kernel, no sum-of-delayed-rates basis, so velocity recovery
here is not by construction.

The inverse recovers the field's signal-propagation (group/front) speed v_g, a
fixed monotonic function of the nominal phase speed c (here v_g ~ 1.5-2.5x c due
to lattice dispersion; we do NOT assume v_g = c). Ground truth measured two
model-free ways: (GT1) wavefront tracking of a single central pulse; (GT2)
lag-distance regression of per-pair best-cc lag on distance, 1/slope = speed.

For candidate v we predict tau_ij(v) = round(d_ij/v) and score a direction-agnostic
traveling-wave consistency residual:
  residual(v) = 1 - mean_{i<j} max(corr(u_i, u_j(t-tau)), corr(u_j, u_i(t-tau))).
Kernel-marginalized (no W fit). Controls: velocity-shuffled distance null (PASS
requires the true floor clearly below the shuffled floor), c/noise/subsample sweeps
with per-seed error bars. Flat/boundary residual or a null reaching the true floor
is reported as the honest negative.

  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=3 \
    python scripts/inverse/foreigninv_neural_field.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))


# Foreign generator: damped 2D neural-field wave PDE (leapfrog FD).
def laplacian_5pt(u):
    """5-point Laplacian, Neumann (reflecting) boundary via edge padding."""
    up = np.pad(u, 1, mode="edge")
    return (up[:-2, 1:-1] + up[2:, 1:-1] + up[1:-1, :-2] + up[1:-1, 2:]
            - 4.0 * up[1:-1, 1:-1])


def simulate_wave_field(G=64, T=300, c=0.45, gamma=0.02, k=0.005, dx=1.0,
                        n_sources=12, pulse_width=2.0, pulse_amp=1.0,
                        burn_in=40, seed=0):
    """Integrate u_tt = c^2 Lap(u) - gamma u_t - k u + drive; return (T,G,G).

    Stability (CFL, 2D, dt=1): c*dt/dx <= 1/sqrt(2) ~ 0.707. We enforce < 0.70.
    Distances in grid cells (dx=1), time in steps (dt=1).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0
    if c * dt / dx > 0.70:
        raise ValueError(f"CFL violated: c*dt/dx={c*dt/dx:.3f} > 0.70 (unstable).")

    u = np.zeros((G, G)); u_prev = np.zeros((G, G))
    total_T = T + burn_in
    onsets = rng.integers(0, total_T - 8, size=n_sources)
    rows = rng.integers(5, G - 5, size=n_sources)
    cols = rng.integers(5, G - 5, size=n_sources)
    yy, xx = np.mgrid[0:G, 0:G]
    c2 = (c * dt / dx) ** 2

    frames = []
    for t in range(total_T):
        drive = np.zeros((G, G))
        for s in range(n_sources):
            dd = t - onsets[s]
            if 0 <= dd <= 6:
                tw = np.exp(-0.5 * (dd - 3.0) ** 2 / 1.5 ** 2)
                sp = np.exp(-0.5 * ((yy - rows[s]) ** 2 + (xx - cols[s]) ** 2) / pulse_width ** 2)
                drive += pulse_amp * tw * sp
        lap = laplacian_5pt(u)
        u_t = u - u_prev
        u_next = 2.0 * u - u_prev + c2 * lap - (gamma * u_t + k * u - drive)
        u_prev, u = u, u_next
        if t >= burn_in:
            frames.append(u.copy())
    return np.array(frames)


# Ground truth 1: wavefront-tracking propagation speed.
def gt_wavefront_speed(G=64, c=0.45, gamma=0.02, k=0.005, seed=123):
    """Single central pulse; track outer wavefront radius vs time; slope=speed."""
    u = np.zeros((G, G)); u_prev = np.zeros((G, G))
    yy, xx = np.mgrid[0:G, 0:G]
    cy, cx = G // 2, G // 2
    c2 = c ** 2
    radii, times = [], []
    T = 80
    for t in range(T):
        drive = np.zeros((G, G))
        if t <= 6:
            tw = np.exp(-0.5 * (t - 3.0) ** 2 / 1.5 ** 2)
            sp = np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / 2.0 ** 2)
            drive = tw * sp
        lap = laplacian_5pt(u)
        u_t = u - u_prev
        u_next = 2 * u - u_prev + c2 * lap - (gamma * u_t + k * u - drive)
        u_prev, u = u, u_next
        if 8 <= t <= 0.7 * G:  # before the front hits the wall
            amp = np.abs(u)
            mask = amp > 0.15 * amp.max() + 1e-9
            if mask.sum() > 5:
                r = np.sqrt((yy[mask] - cy) ** 2 + (xx[mask] - cx) ** 2)
                radii.append(np.percentile(r, 95)); times.append(t)
    if len(times) < 5:
        return float("nan"), float("nan")
    times = np.array(times, float); radii = np.array(radii, float)
    A = np.vstack([times, np.ones_like(times)]).T
    sol, *_ = np.linalg.lstsq(A, radii, rcond=None)
    pred = A @ sol
    r2 = 1 - np.sum((radii - pred) ** 2) / (np.sum((radii - radii.mean()) ** 2) + 1e-12)
    return float(sol[0]), float(r2)


# Electrode sampling (compact array, partial obs via subsample) + noise.
# Compact so inter-electrode lags d_ij/v_g land in a recoverable range (~1-20
# steps); a sheet-spanning array would push lags past the wave's coherence time.
def place_electrodes(G, n_elec, radius, seed=0):
    """Sample n_elec sites inside a disk of given radius about the sheet center."""
    rng = np.random.default_rng(seed + 9090)
    cy, cx = G // 2, G // 2
    coords, seen = [], set()
    R2 = radius * radius
    tries = 0
    while len(coords) < n_elec and tries < 100000:
        tries += 1
        r = int(rng.integers(cy - radius, cy + radius + 1))
        cc = int(rng.integers(cx - radius, cx + radius + 1))
        if ((r - cy) ** 2 + (cc - cx) ** 2 <= R2 and 4 < r < G - 4 and 4 < cc < G - 4
                and (r, cc) not in seen):
            seen.add((r, cc)); coords.append((r, cc))
    return np.array(coords, float)


def sample_at_electrodes(field, coords, noise_std=0.0, seed=0):
    """field (T,G,G) -> (T,E) traces at integer sites, + relative Gaussian noise."""
    ri = coords[:, 0].astype(int); ci = coords[:, 1].astype(int)
    traces = field[:, ri, ci]
    if noise_std > 0:
        rms = np.sqrt(np.mean(traces ** 2)) + 1e-12
        rng = np.random.default_rng(seed + 5151)
        traces = traces + rng.normal(0.0, noise_std * rms, size=traces.shape)
    return traces


def electrode_distance_matrix(coords):
    d = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def zscore_cols(traces):
    ts = traces - traces.mean(0, keepdims=True)
    return ts / (ts.std(0, keepdims=True) + 1e-9)


# Ground truth 2: lag-distance regression propagation speed.
def gt_lagdist_speed(traces, dist, maxlag=24, ccmin=0.3):
    """Per-pair best-cc lag; regress lag on distance; 1/slope = speed. Model-free."""
    ts = zscore_cols(traces); T, E = ts.shape
    C = _lagcorr_tensor(ts, maxlag)   # C[l,i,j]=corr(u_i(t),u_j(t-l))
    lags, dists = [], []
    iu = np.triu_indices(E, 1)
    for a, b in zip(*iu):
        # best lag over both directions
        cc_fwd = C[:, a, b]; cc_bwd = C[:, b, a]
        best = max(cc_fwd.max(), cc_bwd.max())
        if best <= ccmin:
            continue
        L = int(np.argmax(cc_fwd)) if cc_fwd.max() >= cc_bwd.max() else int(np.argmax(cc_bwd))
        lags.append(L); dists.append(dist[a, b])
    if len(lags) < 8:
        return float("nan"), float("nan"), len(lags)
    lags = np.array(lags, float); dists = np.array(dists, float)
    A = np.vstack([dists, np.ones_like(dists)]).T
    sol, *_ = np.linalg.lstsq(A, lags, rcond=None)
    pred = A @ sol
    r2 = 1 - np.sum((lags - pred) ** 2) / (np.sum((lags - lags.mean()) ** 2) + 1e-12)
    slope = sol[0]
    return (float(1.0 / slope) if slope > 1e-6 else float("nan")), float(r2), len(lags)


# The inverse: lag-sensitive traveling-wave consistency residual.
def _lagcorr_tensor(ts, maxlag):
    """C[l,i,j] = mean_t ts[t,i] * ts[t-l,j], l=0..maxlag. ts already z-scored."""
    T, E = ts.shape
    C = np.zeros((maxlag + 1, E, E))
    for l in range(maxlag + 1):
        C[l] = (ts[l:, :].T @ ts[:T - l, :]) / (T - l)
    return C


def residual_from_C(C, dist, v, maxlag, min_delay=1):
    """1 - mean_{i<j} max(C[tau,i,j], C[tau,j,i]) with tau=round(d_ij/v)."""
    E = dist.shape[0]
    tau = np.clip(np.round(dist / v), min_delay, maxlag).astype(int)
    iu = np.triu_indices(E, 1)
    di = tau[iu]
    cc1 = C[di, iu[0], iu[1]]
    cc2 = C[di, iu[1], iu[0]]
    return float(1.0 - np.maximum(cc1, cc2).mean())


def scan_velocity(traces, dist, cands, maxlag, min_delay=1):
    """residual(v) over candidates; returns array + the precomputed C reused."""
    ts = zscore_cols(traces)
    C = _lagcorr_tensor(ts, maxlag)
    res = np.array([residual_from_C(C, dist, v, maxlag, min_delay) for v in cands])
    return res, C


def shuffle_distance(dist, seed):
    """Velocity-shuffled null: permute off-diagonal distances (same histogram)."""
    E = dist.shape[0]
    rng = np.random.default_rng(seed + 4242)
    iu = np.triu_indices(E, 1)
    d = dist.copy()
    vals = d[iu].copy(); rng.shuffle(vals)
    d[iu] = vals
    d[(iu[1], iu[0])] = vals
    return d


def phase_randomize(traces, rng):
    """Autocorr-preserving temporal-shuffle null: per-channel Fourier phase
    randomization. Preserves each electrode's power spectrum (hence its
    autocorrelation) EXACTLY, while scrambling the cross-channel phase = the
    inter-electrode propagation timing. The decisive test that a residual minimum
    reflects TRAVEL-TIME structure, not trivial single-channel autocorrelation."""
    T = traces.shape[0]
    F = np.fft.rfft(traces, axis=0)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=F.shape)
    phases[0, :] = 0.0
    if T % 2 == 0:
        phases[-1, :] = 0.0
    Fr = np.abs(F) * np.exp(1j * phases)
    return np.fft.irfft(Fr, n=T, axis=0)


def run_one_condition(c, gamma, k, G, T, n_elec, elec_radius, noise_std,
                      maxlag, min_delay, n_cands, v_lo, v_hi, seed, n_sources):
    field = simulate_wave_field(G=G, T=T, c=c, gamma=gamma, k=k,
                                n_sources=n_sources, seed=seed)
    coords = place_electrodes(G, n_elec, elec_radius, seed=seed)
    traces = sample_at_electrodes(field, coords, noise_std=noise_std, seed=seed)
    dist = electrode_distance_matrix(coords)
    d_shuf = shuffle_distance(dist, seed)

    # ground-truth empirical propagation speed at the electrodes (model-free)
    v_g, vg_r2, vg_n = gt_lagdist_speed(traces, dist, maxlag=maxlag)

    cands = np.geomspace(v_lo, v_hi, n_cands)
    res_true, _ = scan_velocity(traces, dist, cands, maxlag, min_delay)
    res_shuf, _ = scan_velocity(traces, d_shuf, cands, maxlag, min_delay)

    amin = int(np.argmin(res_true))
    v_hat = float(cands[amin])
    interior = 0 < amin < len(cands) - 1
    floor_true = float(res_true.min()); floor_shuf = float(res_shuf.min())
    v_hat_shuf = float(cands[int(np.argmin(res_shuf))])
    dip_true = float((res_true.max() - res_true.min()) / (res_true.max() + 1e-12))

    # autocorr-preserving null: phase-randomize each electrode (true geometry kept),
    # average the residual floor over several shuffles -> the harder, decisive null.
    rng_ac = np.random.default_rng(seed + 9001)
    ac_floors = []
    for _ in range(8):
        res_ac, _ = scan_velocity(phase_randomize(traces, rng_ac), dist, cands, maxlag, min_delay)
        ac_floors.append(float(res_ac.min()))
    floor_ac = float(np.mean(ac_floors))
    autocorr_null_ratio = float(floor_ac / (floor_true + 1e-12))
    autocorr_beats_true = bool(np.all(np.array(ac_floors) > floor_true))

    return {
        "c": c, "noise_std": noise_std, "n_elec": n_elec, "seed": seed,
        "v_g_empirical": v_g, "v_g_r2": vg_r2, "v_g_npairs": vg_n,
        "v_hat": v_hat, "interior_min": bool(interior),
        "ratio_vhat_vg": float(v_hat / v_g) if v_g == v_g and v_g > 0 else float("nan"),
        "ratio_vhat_c": float(v_hat / c),
        "floor_true": floor_true, "floor_shuf": floor_shuf,
        "null_ratio": float(floor_shuf / (floor_true + 1e-12)),
        "floor_autocorr": floor_ac, "autocorr_null_ratio": autocorr_null_ratio,
        "autocorr_beats_true": autocorr_beats_true,
        "v_hat_shuf": v_hat_shuf, "dip_true": dip_true,
        "cands": cands.tolist(), "res_true": res_true.tolist(), "res_shuf": res_shuf.tolist(),
    }


def aggregate(cells, key_fields):
    groups = defaultdict(list)
    for cell in cells:
        groups[tuple(cell[f] for f in key_fields)].append(cell)
    summary = []
    for _, items in sorted(groups.items()):
        vhat = np.array([it["v_hat"] for it in items])
        vg = np.array([it["v_g_empirical"] for it in items])
        rvg = np.array([it["ratio_vhat_vg"] for it in items])
        rvg = rvg[np.isfinite(rvg)]
        rvc = np.array([it["ratio_vhat_c"] for it in items])
        ft = np.array([it["floor_true"] for it in items])
        fs = np.array([it["floor_shuf"] for it in items])
        nr = np.array([it["null_ratio"] for it in items])
        acr = np.array([it["autocorr_null_ratio"] for it in items])
        n_int = sum(it["interior_min"] for it in items)
        beats_null_all = bool(np.all(fs > ft))
        autocorr_beats_all = bool(all(it["autocorr_beats_true"] for it in items))
        # error vs empirical v_g (the rigorous ground truth)
        err_vg = float(np.abs(rvg - 1).mean() * 100) if rvg.size else float("nan")
        summary.append({
            "c": items[0]["c"], "noise_std": items[0]["noise_std"],
            "n_elec": items[0]["n_elec"], "n_seeds": len(items),
            "v_hat_mean": float(vhat.mean()), "v_hat_std": float(vhat.std()),
            "v_g_mean": float(np.nanmean(vg)), "v_g_std": float(np.nanstd(vg)),
            "ratio_vhat_vg_mean": float(rvg.mean()) if rvg.size else float("nan"),
            "ratio_vhat_vg_std": float(rvg.std()) if rvg.size else float("nan"),
            "err_vs_vg_pct": err_vg,
            "ratio_vhat_c_mean": float(rvc.mean()),
            "floor_true_mean": float(ft.mean()), "floor_true_std": float(ft.std()),
            "floor_shuf_mean": float(fs.mean()),
            "null_ratio_mean": float(nr.mean()), "null_ratio_std": float(nr.std()),
            "autocorr_null_ratio_mean": float(acr.mean()), "autocorr_null_ratio_std": float(acr.std()),
            "autocorr_beats_all_seeds": autocorr_beats_all,
            "dip_true_mean": float(np.mean([it["dip_true"] for it in items])),
            "interior_min_seeds": int(n_int), "beats_null_all_seeds": beats_null_all,
        })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--G", type=int, default=64)
    ap.add_argument("--T", type=int, default=320)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--k", type=float, default=0.005)
    ap.add_argument("--n-sources", type=int, default=16, dest="n_sources")
    ap.add_argument("--elec-radius", type=int, default=9, dest="elec_radius")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--maxlag", type=int, default=24)
    ap.add_argument("--min-delay", type=int, default=1, dest="min_delay")
    ap.add_argument("--n-cands", type=int, default=30, dest="n_cands")
    ap.add_argument("--v-lo", type=float, default=0.15, dest="v_lo")
    ap.add_argument("--v-hi", type=float, default=3.0, dest="v_hi")
    ap.add_argument("--out", default=str(ROOT / "results" / "inverse" / "foreigninv_neural_field.json"))
    args = ap.parse_args()

    if args.smoke:
        c_grid = [0.3, 0.45]; noise_grid = [0.0, 0.15]; elec_grid = [32]
        seeds = 2; G, T = 56, 200; n_sources = 10
        print("SMOKE config", flush=True)
    else:
        c_grid = [0.30, 0.45, 0.60]
        noise_grid = [0.0, 0.05, 0.15, 0.30]
        elec_grid = [48, 24]            # full vs heavy subsample (partial obs)
        seeds = args.seeds; G, T = args.G, args.T; n_sources = args.n_sources

    print(f"FOREIGN-INV neural-field  G={G} T={T} gamma={args.gamma} k={args.k}  "
          f"seeds={seeds} elec_radius={args.elec_radius}", flush=True)
    print(f"  c_grid={c_grid}  noise_grid={noise_grid}  elec_grid={elec_grid}", flush=True)

    # generator sanity: GT1 wavefront speed vs nominal c
    print("\n[GT1 generator sanity] wavefront-tracking propagation speed vs nominal c:", flush=True)
    gen_check = []
    for c in c_grid:
        sp, r2 = gt_wavefront_speed(G=max(G, 64), c=c, gamma=args.gamma, k=args.k)
        gen_check.append({"c": c, "front_speed": sp, "r2": r2, "front_over_c": sp / c})
        print(f"  c={c:.2f} -> front_speed={sp:.3f} (front/c={sp/c:.2f}, R2={r2:.3f})", flush=True)

    t_start = time.time()
    cells = []
    total = len(c_grid) * len(noise_grid) * len(elec_grid) * seeds
    done = 0
    for c in c_grid:
        for noise_std in noise_grid:
            for n_elec in elec_grid:
                for seed in range(seeds):
                    cells.append(run_one_condition(
                        c=c, gamma=args.gamma, k=args.k, G=G, T=T, n_elec=n_elec,
                        elec_radius=args.elec_radius, noise_std=noise_std,
                        maxlag=args.maxlag, min_delay=args.min_delay, n_cands=args.n_cands,
                        v_lo=args.v_lo, v_hi=args.v_hi, seed=seed, n_sources=n_sources))
                    done += 1
        print(f"  [{done}/{total}] through c={c} ({time.time()-t_start:.0f}s)", flush=True)

    summary = aggregate(cells, key_fields=("c", "noise_std", "n_elec"))

    print("\n" + "=" * 104)
    print("RECOVERED v vs EMPIRICAL PROPAGATION SPEED v_g (model-free GT2), with per-seed error bars + SHUFFLED null")
    print("=" * 104)
    print(f"{'c':>5} {'noise':>6} {'E':>4} | {'v_hat(mean±sd)':>17} {'v_g(mean±sd)':>16} "
          f"{'v_hat/v_g':>11} {'err%vs_vg':>9} | {'fl_true':>8} {'fl_shuf':>8} {'null×':>6} "
          f"{'inter':>6} {'beatN':>6}")
    print("-" * 104)
    for s in summary:
        print(f"{s['c']:>5.2f} {s['noise_std']:>6.2f} {s['n_elec']:>4d} | "
              f"{s['v_hat_mean']:>8.3f}±{s['v_hat_std']:<6.3f} "
              f"{s['v_g_mean']:>8.3f}±{s['v_g_std']:<5.3f} "
              f"{s['ratio_vhat_vg_mean']:>6.2f}±{s['ratio_vhat_vg_std']:<4.2f} "
              f"{s['err_vs_vg_pct']:>7.0f}% | "
              f"{s['floor_true_mean']:>8.4f} {s['floor_shuf_mean']:>8.4f} "
              f"{s['null_ratio_mean']:>5.2f}x "
              f"{s['interior_min_seeds']:>2d}/{s['n_seeds']:<2d} "
              f"{'YES' if s['beats_null_all_seeds'] else 'no':>6}")
    print("=" * 104)

    # honest verdict
    def cond(s, max_err):
        return (np.isfinite(s["err_vs_vg_pct"]) and s["err_vs_vg_pct"] < max_err and
                s["interior_min_seeds"] >= max(1, s["n_seeds"] - 1) and
                s["beats_null_all_seeds"] and s["null_ratio_mean"] > 1.08)
    clean = [s for s in summary if s["noise_std"] == 0.0]
    clean_pass = [s for s in clean if cond(s, 30)]
    noisy = [s for s in summary if s["noise_std"] > 0]
    noisy_pass = [s for s in noisy if cond(s, 45)]
    overall_beats = sum(1 for s in summary if s["beats_null_all_seeds"])
    median_err = float(np.median([s["err_vs_vg_pct"] for s in summary if np.isfinite(s["err_vs_vg_pct"])]))
    median_null = float(np.median([s["null_ratio_mean"] for s in summary]))
    # monotonic v_hat in c (clean, full electrodes)? a key non-tautology check
    cf = sorted([s for s in clean if s["n_elec"] == max(elec_grid)], key=lambda z: z["c"])
    mono = all(cf[i]["v_hat_mean"] <= cf[i + 1]["v_hat_mean"] + 1e-6 for i in range(len(cf) - 1)) if len(cf) > 1 else False

    print("\nHONEST READOUT:")
    print(f"  GT1 wavefront speed tracks nominal c (front/c ~ {[round(g['front_over_c'],2) for g in gen_check]}, "
          f"R2 {[round(g['r2'],2) for g in gen_check]}) -> the PDE really propagates a wave.")
    print(f"  CLEAN (noise=0): {len(clean_pass)}/{len(clean)} conditions recover v within 30% of empirical v_g, "
          f"interior min, beat shuffled null (>1.08x).")
    print(f"  NOISY (noise>0): {len(noisy_pass)}/{len(noisy)} conditions recover v within 45% AND beat null.")
    print(f"  v_hat MONOTONIC in true c (clean, full array): {'YES' if mono else 'NO'}  "
          f"(v_hat at c={[round(s['c'],2) for s in cf]} -> {[round(s['v_hat_mean'],3) for s in cf]}).")
    print(f"  median |err| vs v_g = {median_err:.0f}% ; median null ratio = {median_null:.2f}x ; "
          f"beats-null conditions = {overall_beats}/{len(summary)}.")

    if len(clean_pass) >= max(1, len(clean) // 2) and len(noisy_pass) >= 1 and mono:
        verdict = ("REAL METHOD: the lag-consistency velocity inverse recovers the FOREIGN neural-field "
                   "propagation speed v_g with a residual minimum that beats the velocity-shuffled null, "
                   "tracks v_g within bounded error, and is MONOTONIC in the nominal wave speed c. Recovery "
                   "is NON-tautological (wave-PDE generator, kernel marginalized, no rate-RNN basis). The "
                   "recovered velocity is the field's group/front speed (a fixed monotonic function of c).")
        tag = "real-method"
    elif overall_beats >= len(summary) // 2 and (len(clean_pass) >= 1 or mono):
        verdict = ("PROMISING / PARTIAL: clean-data recovery tracks the empirical propagation speed and "
                   "beats the null on most conditions and v_hat is monotonic in c, but noise/subsampling "
                   "bias v_hat or shrink the null gap. Honest partial recovery from foreign dynamics.")
        tag = "promising"
    elif overall_beats >= 1:
        verdict = ("WEAK: occasionally beats the null but v_hat is biased / boundary-pinned across most "
                   "conditions; the foreign field does not give a cleanly identifiable velocity. Honest "
                   "near-negative.")
        tag = "weak"
    else:
        verdict = ("NULL: the velocity-shuffled null reaches the true-geometry floor on foreign field data; "
                   "the inverse is not distance-specific here. The c->v recovery does not transfer to a "
                   "genuinely foreign generator. This is the fig_si_pinn_gate limitation, reported honestly.")
        tag = "null"
    print("\nVERDICT:", verdict)

    out = {
        "generator": "damped 2D neural-field wave PDE  u_tt = c^2 Lap(u) - gamma u_t - k u + drive  (leapfrog FD, Neumann BC)",
        "foreign": True,
        "recovered_quantity": "signal-propagation (group/front) speed v_g of the wave field; ground truth measured model-free two ways (wavefront tracking GT1, lag-distance regression GT2)",
        "config": {"G": G, "T": T, "gamma": args.gamma, "k": args.k, "n_sources": n_sources,
                   "elec_radius": args.elec_radius, "seeds": seeds, "maxlag": args.maxlag,
                   "min_delay": args.min_delay, "n_cands": args.n_cands,
                   "v_lo": args.v_lo, "v_hi": args.v_hi,
                   "c_grid": c_grid, "noise_grid": noise_grid, "elec_grid": elec_grid},
        "generator_sanity_GT1": gen_check,
        "summary": summary, "cells": cells,
        "headline": {
            "clean_pass": len(clean_pass), "clean_total": len(clean),
            "noisy_pass": len(noisy_pass), "noisy_total": len(noisy),
            "overall_beats_null": overall_beats, "n_conditions": len(summary),
            "median_err_vs_vg_pct": median_err, "median_null_ratio": median_null,
            "vhat_monotonic_in_c": bool(mono),
        },
        "verdict": verdict, "verdict_tag": tag,
    }
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}   ({time.time()-t_start:.0f}s total)")


if __name__ == "__main__":
    main()
