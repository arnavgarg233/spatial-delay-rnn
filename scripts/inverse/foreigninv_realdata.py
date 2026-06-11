"""Conduction-velocity inverse on real published cortical traveling-wave data.

Source: ScaleSymmetry/Traveling-wave-analysis (Alexander et al.; PLoS ONE
10.1371/journal.pone.0148413, PLoS Comput Biol 10.1371/journal.pcbi.1007316).
Files fetched over HTTP (see fetch_data()): xyz_coordinates.pkl (64,3 mm),
avref_timeseries.pkl (26,1550,64) and avref_timeseries_stim.pkl (32,1550,64),
fs=500 Hz; dominant wave is alpha-band ~9.2 Hz. Real cortical field activity, not
a rate-SDRNN, so velocity recovery is not by construction.

The inverse refits a free kernel at every candidate v (kernel marginalized) and
asks where the delayed-field model best explains the data. The repo documents no
m/s speed, so the independent reference is the standard phase-gradient method
(Hilbert phase, planar fit phase ~ k.x, speed = 2pi f/|k|) - a different estimator,
so agreement is informative. Literature anchor: cortical alpha waves ~0.1-0.8 m/s.

Per dataset: reference v_phasegrad (IQR), recovered v_hat with a bootstrap CI over
trials, the residual(v) shape (interior/bracketed?), and the velocity-shuffled null
floor. Flat/boundary residual or a null matching the true floor is reported as a
failure, not glossed over.

  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=3 \
    python scripts/inverse/foreigninv_realdata.py --smoke
"""

import argparse
import json
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))

# Reuse the inverse machinery from pinn_inverse (free/low-rank kernel fit,
# normalized residual, velocity-shuffled null).
sys.path.insert(0, str(ROOT / "scripts" / "inverse"))
from pinn_inverse import (  # noqa: E402
    fit_kernel_at_velocity,
    normalized_residual,
    shuffle_geometry,
)

DATA_DIR = ROOT / "results" / "_foreign_cache"
RAW_BASE = "https://raw.githubusercontent.com/ScaleSymmetry/Traveling-wave-analysis/main/"
FILES = ["xyz_coordinates.pkl", "avref_timeseries.pkl", "avref_timeseries_stim.pkl"]
FS = 500.0       # Hz  (tdel = 2 ms, from the repo notebooks)
F_WAVE = 9.2     # Hz  (alpha traveling wave centre frequency, from the repo notebooks)
MM_PER_M = 1000.0


# Data acquisition (HTTP, cached locally). If a file cannot be fetched, bail.
def fetch_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in FILES:
        dest = DATA_DIR / f
        if not dest.exists() or dest.stat().st_size < 1000:
            url = RAW_BASE + f
            print(f"  fetching {f} ...", flush=True)
            urllib.request.urlretrieve(url, dest)
        with open(dest, "rb") as fh:
            out[f] = np.asarray(pickle.load(fh))
        print(f"  {f}: shape {out[f].shape}", flush=True)
    return out


# Independent reference v: phase-gradient (standard traveling-wave method,
# different from the delayed-field residual). Returns m/s samples.
def phasegrad_reference(ts, xyz, fs=FS, f0=F_WAVE, r2_min=0.5):
    """ts (C,T,S) field, xyz (S,3) mm. Speed = 2*pi*f0/|k|, k from planar phase fit."""
    from scipy.signal import butter, filtfilt, hilbert

    C, T, S = ts.shape
    lo, hi = (f0 - 2) / (fs / 2), (f0 + 2) / (fs / 2)
    b, a = butter(3, [lo, hi], btype="band")
    X = xyz - xyz.mean(0)
    G = np.c_[X, np.ones(S)]
    Gpinv = np.linalg.pinv(G)
    speeds, r2s = [], []
    for c in range(C):
        filt = filtfilt(b, a, ts[c], axis=0)
        ph = np.angle(hilbert(filt, axis=0))
        for t in range(300, T - 300, 10):
            p = np.unwrap(ph[t] - ph[t].mean())
            coef = Gpinv @ p
            k = coef[:3]
            pred = G @ coef
            ss_res = np.sum((p - pred) ** 2)
            ss_tot = np.sum((p - p.mean()) ** 2) + 1e-12
            r2 = 1 - ss_res / ss_tot
            kn = np.linalg.norm(k)
            if kn > 1e-6 and r2 > r2_min:
                speeds.append(2 * np.pi * f0 / kn / MM_PER_M)  # m/s
                r2s.append(r2)
    speeds = np.array(speeds)
    return speeds, np.array(r2s)


# Delayed-field inverse adapted to foreign field data. Assumed model:
#   x_i(t) ~ sum_j W_ij x_j(t - tau_ij(v)),  tau_ij = clip(round(d/v), min, max).
# Band-limit to the wave band, z-score per channel, treat trials as the batch dim.
# rec_target is the field itself (no leaky-integrator to invert; not our model).
# Velocity units: v in mm per sample; physical speed = v * fs / 1000 (m/s).
def preprocess_field(ts, fs=FS, f0=F_WAVE, band=(4.0, 16.0)):
    """Band-limit to the wave band and z-score per channel. Return (C,T,S)."""
    from scipy.signal import butter, filtfilt

    b, a = butter(3, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    out = filtfilt(b, a, ts, axis=1)
    mu = out.mean(axis=1, keepdims=True)
    sd = out.std(axis=1, keepdims=True) + 1e-8
    return (out - mu) / sd


def scan_freekernel_field(rates, dist, cands, max_delay, min_delay, ridge, rank):
    """Free/low-rank kernel residual(v) on a geometry. rates=(B,T,S).

    rec_target is the field at t=1..T-1 (same alignment pinn_inverse uses). At
    each v we refit a free kernel (no W handed over) and score the normalized
    residual. Returns (res_full[v], res_lr[v]).
    """
    rec_target = rates[:, 1:, :]  # (B, T-1, S) - predict the field itself
    res_full, res_lr = [], []
    for c in cands:
        tau = np.clip(np.round(dist / c), min_delay, max_delay).astype(int)
        fk = fit_kernel_at_velocity(rates, rec_target, tau, ridge, rank=rank)
        res_full.append(fk["res_full"])
        res_lr.append(fk.get("res_lr", np.nan))
    return np.array(res_full), np.array(res_lr)


def v_to_mps(v_samp):
    """mm-per-sample -> m/s."""
    return v_samp * FS / MM_PER_M


def mps_to_v(v_mps):
    return v_mps * MM_PER_M / FS


def run_dataset(name, ts, xyz, args, rng):
    """Run reference + inverse + null on one foreign dataset; return result dict."""
    print(f"\n==== DATASET: {name}  ts{ts.shape}  ====", flush=True)
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)  # (S,S) mm

    # --- independent reference velocity (phase-gradient) ---
    ref_speeds, ref_r2 = phasegrad_reference(ts, xyz)
    v_ref = float(np.median(ref_speeds)) if len(ref_speeds) else float("nan")
    ref_iqr = ([float(np.percentile(ref_speeds, 25)), float(np.percentile(ref_speeds, 75))]
               if len(ref_speeds) else [float("nan")] * 2)
    print(f"  REFERENCE v_phasegrad: {v_ref:.3f} m/s  IQR {np.round(ref_iqr,3).tolist()}  "
          f"(n={len(ref_speeds)} planar fits, mean r2={ref_r2.mean() if len(ref_r2) else float('nan'):.2f})",
          flush=True)

    # preprocess field for the delayed-field inverse
    rates = preprocess_field(ts)  # (C,T,S)
    C, T, S = rates.shape

    # candidate velocities: a FIXED physical window (m/s) wide enough to bracket
    # both the phase-gradient reference (~0.08 m/s) AND any interior residual
    # minimum, and to expose the high-v lag->1 collapse. A reference-relative
    # window can clip the true minimum (the wide-scan diagnostic showed the
    # free-kernel min sits near ~0.34 m/s, well above 4x the reference), so we
    # do NOT center on the reference. Range covers the literature cortical band.
    cands_mps = np.geomspace(args.v_lo, args.v_hi, args.n_cands)
    cands = mps_to_v(cands_mps)
    print(f"  v candidates (m/s): {np.round(cands_mps, 3).tolist()}", flush=True)
    print(f"  max_delay={args.max_delay}  ridge={args.ridge}  rank={args.rank}  "
          f"dist(mm) med={np.median(dist[dist>0]):.1f} max={dist.max():.1f}", flush=True)

    # inverse on true geometry (free kernel, kernel marginalized)
    res_true, res_true_lr = scan_freekernel_field(
        rates, dist, cands, args.max_delay, args.min_delay, args.ridge, args.rank)
    argmin = int(np.argmin(res_true))
    v_hat_mps = float(cands_mps[argmin])
    interior = 0 < argmin < len(cands) - 1
    floor_true = float(res_true.min())
    dip = float((res_true.max() - res_true.min()) / (res_true.max() + 1e-12))
    print(f"  INVERSE v_hat: {v_hat_mps:.3f} m/s  floor={floor_true:.4f}  dip={dip:.3f}  "
          f"interior_min={interior}", flush=True)

    # bootstrap CI over trials for v_hat
    boot_vhat = []
    n_boot = args.n_boot
    for _ in range(n_boot):
        idx = rng.integers(0, C, size=C)
        rb = rates[idx]
        rf, _ = scan_freekernel_field(
            rb, dist, cands, args.max_delay, args.min_delay, args.ridge, args.rank)
        boot_vhat.append(cands_mps[int(np.argmin(rf))])
    boot_vhat = np.array(boot_vhat)
    vhat_lo, vhat_hi = float(np.percentile(boot_vhat, 16)), float(np.percentile(boot_vhat, 84))
    vhat_med = float(np.median(boot_vhat))
    print(f"  INVERSE v_hat bootstrap: median {vhat_med:.3f} m/s  "
          f"68% CI [{vhat_lo:.3f}, {vhat_hi:.3f}]  (n_boot={n_boot})", flush=True)

    # velocity-shuffled null floor (free kernel = null's best shot). Keep the full
    # null curve per shuffle: the distance-specific signal is the velocity where
    # true geometry most beats the per-velocity null. That gap-max velocity is more
    # robust to the high-v lag->1 collapse than comparing global floors (the
    # collapse lowers both floors, so the floor gap understates specificity).
    null_curves, null_floors, null_vhats = [], [], []
    for s in range(args.n_shuffles):
        d_shuf = shuffle_geometry(dist, s)
        rfn, _ = scan_freekernel_field(
            rates, d_shuf, cands, args.max_delay, args.min_delay, args.ridge, args.rank)
        null_curves.append(rfn)
        null_floors.append(float(rfn.min()))
        null_vhats.append(float(cands_mps[int(np.argmin(rfn))]))
    null_curves = np.array(null_curves)              # (n_shuffles, n_cands)
    null_mean_curve = null_curves.mean(axis=0)        # per-velocity null mean
    null_floors = np.array(null_floors)
    floor_null_mean = float(null_floors.mean())
    floor_null_std = float(null_floors.std())
    gap = floor_null_mean - floor_true                 # >0 => true geom fits better
    ratio = floor_null_mean / (floor_true + 1e-12)
    null_beats = bool(np.all(null_floors > floor_true))
    # per-velocity relative gap (null - true)/null; positive = true beats null here.
    relgap_curve = (null_mean_curve - res_true) / (null_mean_curve + 1e-12)
    gapmax_idx = int(np.argmax(relgap_curve))
    v_gapmax_mps = float(cands_mps[gapmax_idx])
    relgap_max = float(relgap_curve[gapmax_idx])
    print(f"  NULL FLOOR (vel-shuffled, free kernel): {floor_null_mean:.4f} ± {floor_null_std:.4f}  "
          f"(true floor {floor_true:.4f})  gap={gap:.4f}  ratio={ratio:.3f}x  "
          f"true<null all-shuffles={null_beats}", flush=True)
    print(f"  GAP-MAX velocity (true beats null MOST): {v_gapmax_mps:.3f} m/s  "
          f"rel-gap={relgap_max*100:.1f}%  (vs reference {v_ref:.3f} m/s)", flush=True)

    # honest error vs reference
    err_pct = float(abs(v_hat_mps / v_ref - 1) * 100) if np.isfinite(v_ref) else float("nan")
    return {
        "dataset": name,
        "n_trials": int(C), "n_sensors": int(S), "n_samples": int(T),
        "fs_hz": FS, "f_wave_hz": F_WAVE,
        "dist_mm_median": float(np.median(dist[dist > 0])), "dist_mm_max": float(dist.max()),
        "v_ref_mps": v_ref, "v_ref_iqr_mps": ref_iqr, "n_ref_fits": int(len(ref_speeds)),
        "ref_mean_r2": float(ref_r2.mean()) if len(ref_r2) else None,
        "candidates_mps": cands_mps.tolist(),
        "res_true": res_true.tolist(),
        "v_hat_mps": v_hat_mps, "floor_true": floor_true, "dip": dip,
        "interior_min": bool(interior),
        "v_hat_boot_median_mps": vhat_med,
        "v_hat_boot_ci68_mps": [vhat_lo, vhat_hi],
        "err_vs_ref_pct": err_pct,
        "null_floor_mean": floor_null_mean, "null_floor_std": floor_null_std,
        "null_floors": null_floors.tolist(),
        "null_mean_curve": null_mean_curve.tolist(),
        "relgap_curve": relgap_curve.tolist(),
        "null_gap": gap, "null_ratio": ratio, "null_beats_all": null_beats,
        "null_vhats_mps": null_vhats,
        "v_gapmax_mps": v_gapmax_mps, "relgap_max": relgap_max,
        "err_gapmax_vs_ref_pct": (float(abs(v_gapmax_mps / v_ref - 1) * 100)
                                  if np.isfinite(v_ref) else float("nan")),
    }


def verdict_for(r):
    """Per-dataset honest verdict string + booleans."""
    interior = r["interior_min"]
    null_beats = r["null_beats_all"]
    null_clear = r["null_ratio"] > 1.5 and null_beats
    err = r["err_vs_ref_pct"]
    # recovery considered "partial" if v_hat within 2x (factor) of reference and
    # the bootstrap CI brackets or nearly brackets the reference.
    lo, hi = r["v_hat_boot_ci68_mps"]
    ref = r["v_ref_mps"]
    ci_brackets = (lo <= ref <= hi) if np.isfinite(ref) else False
    within_2x = (np.isfinite(err) and 0.5 <= (r["v_hat_mps"] / ref) <= 2.0) if np.isfinite(ref) else False
    # same broad order-of-magnitude band (within ~5x) - both estimators land in
    # the literature cortical range even if they disagree by a few-fold.
    within_5x = (np.isfinite(err) and 0.2 <= (r["v_hat_mps"] / ref) <= 5.0) if np.isfinite(ref) else False
    relgap_max = r.get("relgap_max", 0.0)
    if interior and null_clear and (ci_brackets or within_2x):
        v = "RECOVERED"
    elif (interior or null_beats) and within_2x:
        v = "PARTIAL"
    elif interior and within_5x and relgap_max > 0.03:
        v = ("PARTIAL-ORDER-OF-MAGNITUDE (interior min in the right cortical band but "
             "few-fold off the phase-gradient reference; modest distance-specific gap)")
    elif null_beats and not interior:
        v = "NULL-BEATEN-ONLY (v not bracketed: identified from below, unbounded above)"
    else:
        v = "FAILED"
    return v, {"interior": interior, "null_beats": null_beats, "null_clear": null_clear,
               "ci_brackets_ref": bool(ci_brackets), "within_2x": bool(within_2x)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-delay", type=int, default=40, dest="max_delay")
    ap.add_argument("--min-delay", type=int, default=1, dest="min_delay")
    ap.add_argument("--n-cands", type=int, default=17, dest="n_cands")
    ap.add_argument("--v-lo", type=float, default=0.02, dest="v_lo")  # m/s scan floor
    ap.add_argument("--v-hi", type=float, default=1.0, dest="v_hi")   # m/s scan ceiling
    ap.add_argument("--ridge", type=float, default=1e-1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=30, dest="n_boot")
    ap.add_argument("--n-shuffles", type=int, default=8, dest="n_shuffles")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        args.n_cands = 9
        args.n_boot = 4
        args.n_shuffles = 3
        print("SMOKE MODE: reduced cands/boot/shuffles\n", flush=True)

    print("FOREIGN-DATA conduction-velocity INVERSE  (realdata_scout)", flush=True)
    print("source: ScaleSymmetry/Traveling-wave-analysis (cortical alpha traveling waves)", flush=True)
    print("fetching data ...", flush=True)
    data = fetch_data()
    xyz = data["xyz_coordinates.pkl"].astype(np.float64)      # (64,3) mm

    rng = np.random.default_rng(args.seed)
    results = []
    datasets = [("spontaneous", data["avref_timeseries.pkl"].astype(np.float64)),
                ("stimulus", data["avref_timeseries_stim.pkl"].astype(np.float64))]
    if args.smoke:
        datasets = datasets[:1]
        # subsample trials for speed in smoke
        datasets = [(n, ts[: min(8, ts.shape[0])]) for n, ts in datasets]

    for name, ts in datasets:
        r = run_dataset(name, ts, xyz, args, rng)
        v, flags = verdict_for(r)
        r["verdict"] = v
        r["flags"] = flags
        results.append(r)
        print(f"  >> {name} VERDICT: {v}", flush=True)

    print("\n" + "=" * 78)
    print("SUMMARY (foreign real-data velocity recovery)")
    for r in results:
        print(f"  [{r['dataset']:11s}] ref(phase-grad)={r['v_ref_mps']:.3f} m/s  "
              f"v_hat(residual)={r['v_hat_mps']:.3f} (boot {r['v_hat_boot_ci68_mps'][0]:.3f}-"
              f"{r['v_hat_boot_ci68_mps'][1]:.3f}) err={r['err_vs_ref_pct']:.0f}%  "
              f"v_gapmax={r['v_gapmax_mps']:.3f} (relgap {r['relgap_max']*100:.1f}%)  "
              f"null_ratio={r['null_ratio']:.2f}x interior={r['interior_min']}")
        print(f"               -> {r['verdict']}")
    print("=" * 78)

    out = {
        "source": "ScaleSymmetry/Traveling-wave-analysis (cortical alpha traveling waves)",
        "citations": ["10.1371/journal.pone.0148413", "10.1371/journal.pcbi.1007316"],
        "ground_truth_method": "phase-gradient planar fit (independent of SDRNN inverse)",
        "fs_hz": FS, "f_wave_hz": F_WAVE,
        "literature_anchor_mps": "cortical alpha traveling waves ~0.1-0.8 m/s",
        "args": vars(args),
        "smoke": args.smoke,
        "results": results,
    }
    tag = "_smoke" if args.smoke else ""
    outpath = ROOT / "results" / "inverse" / f"foreigninv_realdata{tag}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}", flush=True)


if __name__ == "__main__":
    main()
