"""Conduction-velocity inverse on a real human Utah microelectrode array
(96 chans, 400 um spacing) recorded under propofol-induced LOC; the slow-wave
MUA bursts propagate as non-oscillatory traveling waves.

Source: neurosmiths "Data and Code for STAR Protocols" (Smith lab),
Zarr/Liou/.../Smith, STAR Protocols 2025;6(1):103659, PMC11919625
(STARmain_data_filtered.mat, OSF node ahqej / file uvd5q, ~539 MB).

The inverse is never told a kernel; it refits a free kernel at each candidate v
and asks where the delayed-field model best explains the data. Independent
reference v: burst-latency planar regression (the dataset's own
SpatialLinearRegression), speed = 1/|beta|, a different estimator so agreement is
informative. Literature anchor: cortical slow-wave traveling waves ~0.1-0.8 m/s.

Phase-gradient is the WRONG reference here (sub-wavelength array, ~2pi wrap at
1 Hz -> spuriously tiny v); kept only as a documented-failure control. Decisive
null = autocorrelation-preserving temporal shuffle: each channel gets an
independent circular time-shift, preserving its spectrum but destroying
inter-channel propagation timing. Win = v_hat within ~1.5-2x of v_ref AND beating
that null; otherwise reported as FAILURE/INCONCLUSIVE.

  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=3 \
    python scripts/inverse/foreigninv_intracranial.py --smoke
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "inverse"))

# Reuse the inverse machinery (free/low-rank kernel fit, normalized residual) verbatim.
from pinn_inverse import (  # noqa: E402
    fit_kernel_at_velocity,
    shuffle_geometry,
)

DATA_DIR = ROOT / "results" / "_foreign_cache" / "intracranial"
DATA_FILE = DATA_DIR / "STARmain_data_filtered.mat"
DATA_URL = "https://osf.io/download/uvd5q/"  # OSF node ahqej, STARmain_data_filtered.mat
FS = 2000.0          # Hz (from the file's Fs field)
SPACING_MM = 0.4     # 400 um = 0.04 cm inter-electrode spacing
MM_PER_M = 1000.0

# Utah-array channel->grid map (electrodepinout, verbatim from STARmain.m).
# Corners are -1 (inactive). Column c (0-based) of LFP corresponds to channel c+1,
# whose grid position is where (c+1) appears in this map.
ELECTRODE_MAP = np.array([
    [-1, 96, 93, 92, 90, 88, 85, 83, 81, -1],
    [95, 63, 94, 91, 89, 87, 86, 84, 80, 79],
    [32, 61, 59, 57, 55, 53, 49, 82, 78, 77],
    [30, 64, 60, 58, 56, 51, 47, 45, 76, 75],
    [28, 31, 62, 52, 46, 44, 43, 41, 74, 73],
    [26, 29, 21, 54, 50, 42, 40, 39, 72, 71],
    [24, 27, 25, 19, 15, 48, 38, 37, 70, 69],
    [22, 20, 23, 13, 17,  5, 36, 35, 68, 67],
    [18, 16, 12, 11,  9,  7, 34, 33, 66, 65],
    [-1, 14, 10,  8,  6,  4,  3,  1,  2, -1],
])


# Data acquisition / loading. If the file cannot be fetched, bail rather than fabricate.
def electrode_positions_mm():
    """(96,2) positions in mm. Column c of LFP -> channel c+1 -> grid cell."""
    pos = np.zeros((96, 2))
    for c in range(1, 97):
        r, col = np.argwhere(ELECTRODE_MAP == c)[0]
        pos[c - 1] = [r * SPACING_MM, col * SPACING_MM]
    return pos


def fetch_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size < 1_000_000:
        print(f"  fetching {DATA_FILE.name} (~539 MB) from OSF ...", flush=True)
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print(f"  {DATA_FILE.name}: {DATA_FILE.stat().st_size/1e6:.0f} MB", flush=True)


def load_lfp(t0_s=None, t1_s=None):
    """Load LFP (T,96) in [t0,t1] seconds + Fs. Uses h5py (v7.3 mat = HDF5)."""
    import h5py

    with h5py.File(DATA_FILE, "r") as f:
        fs = float(np.array(f["Fs"]).ravel()[0])
        Tfull = f["LFP"].shape[0]
        i0 = 0 if t0_s is None else int(t0_s * fs)
        i1 = Tfull if t1_s is None else min(Tfull, int(t1_s * fs))
        lfp = np.array(f["LFP"][i0:i1, :])  # (T,96)
    return lfp, fs, (i0, i1, Tfull)


# Burst detection (multiunit-like broadband envelope): defines the windows the
# wave lives in and feeds the burst-latency reference.
def channel_envelope(lfp, band=(10.0, 100.0), fs=FS):
    from scipy.signal import butter, filtfilt, hilbert

    b, a = butter(3, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    hf = filtfilt(b, a, lfp, axis=0)
    return np.abs(hilbert(hf, axis=0))  # (T,S)


def detect_bursts(envc, fs=FS, k_std=1.5, min_gap_s=0.15):
    """Return rising-edge sample indices of mean-envelope bursts, min-gap apart."""
    env = envc.mean(axis=1)
    thr = env.mean() + k_std * env.std()
    above = env > thr
    edges = np.where(np.diff(above.astype(int)) == 1)[0]
    if len(edges) == 0:
        return edges
    keep = [edges[0]]
    min_gap = int(min_gap_s * fs)
    for e in edges[1:]:
        if e - keep[-1] >= min_gap:
            keep.append(e)
    return np.array(keep)


# Reference v #1: burst-latency planar regression (the dataset's own method for
# these non-oscillatory waves). speed = 1/|beta|, beta = ms/mm latency-vs-position.
def burstlatency_reference(envc, pos_mm, edges, fs=FS, win_ms=50.0,
                           r2_min=0.2, v_lo=0.01, v_hi=5.0, min_chans=30):
    win = int(win_ms / 1000.0 * fs)
    T, S = envc.shape
    speeds, r2s, dirs = [], [], []
    for e in edges:
        s0, s1 = e - win, e + win
        if s0 < 0 or s1 >= T:
            continue
        seg = envc[s0:s1, :]                      # (W,S)
        lat = np.argmax(seg, axis=0).astype(float) / fs * 1000.0  # ms
        amp = seg.max(axis=0)
        good = amp > np.percentile(amp, 20)
        if good.sum() < min_chans:
            continue
        X = pos_mm[good] - pos_mm[good].mean(0)
        G = np.c_[X, np.ones(int(good.sum()))]
        y = lat[good] - lat[good].mean()
        coef, *_ = np.linalg.lstsq(G, y, rcond=None)
        beta = coef[:2]                            # ms/mm
        pred = G @ coef
        ssr = np.sum((y - pred) ** 2)
        sst = np.sum(y ** 2) + 1e-12
        r2 = 1 - ssr / sst
        bn = np.linalg.norm(beta)
        if bn < 1e-6:
            continue
        v = 1.0 / bn                               # mm/ms == m/s
        if r2 > r2_min and v_lo < v < v_hi:
            speeds.append(v)
            r2s.append(r2)
            dirs.append(np.arctan2(beta[1], beta[0]))
    return np.array(speeds), np.array(r2s), np.array(dirs)


# Reference v #2 (documented-to-fail control): Hilbert phase-gradient on the slow
# band. Invalid on a sub-wavelength array (spuriously tiny v); kept to show that.
def phasegrad_reference(lfp, pos_mm, f0, bw, fs=FS, r2_min=0.4, tstep=200):
    from scipy.signal import butter, filtfilt, hilbert

    T, S = lfp.shape
    b, a = butter(3, [(f0 - bw) / (fs / 2), (f0 + bw) / (fs / 2)], btype="band")
    ph = np.angle(hilbert(filtfilt(b, a, lfp, axis=0), axis=0))
    X = pos_mm - pos_mm.mean(0)
    G = np.c_[X, np.ones(S)]
    Gp = np.linalg.pinv(G)
    speeds = []
    for t in range(500, T - 500, tstep):
        p = np.unwrap(ph[t] - ph[t].mean())
        coef = Gp @ p
        k = coef[:2]
        pred = G @ coef
        r2 = 1 - np.sum((p - pred) ** 2) / (np.sum((p - p.mean()) ** 2) + 1e-12)
        kn = np.linalg.norm(k)
        if kn > 1e-6 and r2 > r2_min:
            speeds.append(2 * np.pi * f0 / kn / MM_PER_M)  # m/s
    return np.array(speeds)


# Band-limit to the wave band (where the propagation lives) and z-score.
def preprocess_field(lfp, band, fs=FS):
    from scipy.signal import butter, filtfilt

    b, a = butter(3, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    out = filtfilt(b, a, lfp, axis=0)
    mu = out.mean(axis=0, keepdims=True)
    sd = out.std(axis=0, keepdims=True) + 1e-8
    return (out - mu) / sd


def make_segments(field, edges, fs=FS, win_ms=60.0, max_segs=64):
    """Cut (B,T,S) segments centred on bursts: these are where the wave lives."""
    win = int(win_ms / 1000.0 * fs)
    T, S = field.shape
    segs = []
    for e in edges:
        s0, s1 = e - win, e + win
        if s0 < 0 or s1 >= T:
            continue
        segs.append(field[s0:s1, :])
        if len(segs) >= max_segs:
            break
    if not segs:
        return np.zeros((0, 0, S))
    L = min(s.shape[0] for s in segs)
    return np.stack([s[:L] for s in segs], axis=0)  # (B, L, S)


# Velocity-scan units: v is in mm per sample; physical speed = v * fs / 1000 (m/s).
def v_to_mps(v_samp, fs=FS):
    return v_samp * fs / MM_PER_M


def mps_to_v(v_mps, fs=FS):
    return v_mps * MM_PER_M / fs


def scan_freekernel_field(rates, dist, cands, max_delay, min_delay, ridge, rank):
    """Free/low-rank kernel residual(v). rates=(B,T,S); predict field itself."""
    rec_target = rates[:, 1:, :]
    res_full = []
    for c in cands:
        tau = np.clip(np.round(dist / c), min_delay, max_delay).astype(int)
        fk = fit_kernel_at_velocity(rates, rec_target, tau, ridge, rank=rank)
        res_full.append(fk["res_full"])
    return np.array(res_full)


# Hard null: each channel gets an independent circular time-shift, preserving its
# spectrum/autocorrelation but destroying inter-channel propagation timing.
def autocorr_preserving_shuffle(rates, seed):
    """rates (B,T,S) -> independently circularly time-shift each channel."""
    rng = np.random.default_rng(seed + 13337)
    B, T, S = rates.shape
    out = np.empty_like(rates)
    shifts = rng.integers(0, T, size=S)
    for s in range(S):
        out[:, :, s] = np.roll(rates[:, :, s], shifts[s], axis=1)
    return out


# Run the full protocol on one band/window choice.
def run(lfp, pos_mm, args, rng):
    dist = np.linalg.norm(pos_mm[:, None, :] - pos_mm[None, :, :], axis=2)  # (96,96) mm
    print(f"  geometry: {pos_mm.shape[0]} electrodes, spacing {SPACING_MM} mm, "
          f"dist(mm) med={np.median(dist[dist>0]):.2f} max={dist.max():.2f}", flush=True)

    # bursts: where the wave lives, and the reference
    envc = channel_envelope(lfp, band=tuple(args.burst_band))
    edges = detect_bursts(envc, k_std=args.burst_k)
    edges = edges[edges > int(args.skip_s * FS - 0)]  # drop early light-anesthesia part
    print(f"  detected {len(edges)} bursts (band {args.burst_band} Hz, k={args.burst_k}, "
          f"after {args.skip_s}s)", flush=True)

    # reference #1: burst-latency planar regression
    ref_v, ref_r2, ref_dir = burstlatency_reference(envc, pos_mm, edges)
    v_ref = float(np.median(ref_v)) if len(ref_v) else float("nan")
    ref_iqr = ([float(np.percentile(ref_v, 25)), float(np.percentile(ref_v, 75))]
               if len(ref_v) else [float("nan")] * 2)
    print(f"  REFERENCE v_ref (burst-latency planar fit): {v_ref:.3f} m/s  "
          f"IQR {np.round(ref_iqr,3).tolist()}  (n={len(ref_v)}, meanR2="
          f"{ref_r2.mean() if len(ref_r2) else float('nan'):.2f})", flush=True)

    # reference #2 (control, expected to fail): phase-gradient
    pg = phasegrad_reference(lfp, pos_mm, f0=1.0, bw=0.5)
    v_pg = float(np.median(pg)) if len(pg) else float("nan")
    print(f"  (control) phase-gradient @1Hz: {v_pg:.4f} m/s  (n={len(pg)}) "
          f"-- expected spuriously tiny on a sub-wavelength array", flush=True)

    # preprocess + segment for the inverse
    field = preprocess_field(lfp, band=tuple(args.wave_band))
    rates = make_segments(field, edges, win_ms=args.win_ms, max_segs=args.max_segs)
    if rates.shape[0] < 4:
        print("  TOO FEW SEGMENTS - inconclusive on this window.", flush=True)
        return {"error": "too_few_segments", "n_segments": int(rates.shape[0]),
                "v_ref_mps": v_ref}
    B, T, S = rates.shape
    print(f"  inverse input: {B} burst segments x {T} samples x {S} chans "
          f"(wave band {args.wave_band} Hz, win {args.win_ms} ms)", flush=True)

    # candidate velocities (m/s): a fixed physical window bracketing the reference
    # AND exposing the high-v lag->1 collapse. Range covers the cortical band.
    cands_mps = np.geomspace(args.v_lo, args.v_hi, args.n_cands)
    cands = mps_to_v(cands_mps)
    print(f"  v candidates (m/s): {np.round(cands_mps,3).tolist()}", flush=True)
    print(f"  max_delay={args.max_delay} ridge={args.ridge} rank={args.rank}", flush=True)

    # inverse on true geometry
    res_true = scan_freekernel_field(rates, dist, cands, args.max_delay,
                                     args.min_delay, args.ridge, args.rank)
    argmin = int(np.argmin(res_true))
    v_hat = float(cands_mps[argmin])
    interior = 0 < argmin < len(cands) - 1
    floor_true = float(res_true.min())
    dip = float((res_true.max() - res_true.min()) / (res_true.max() + 1e-12))
    print(f"  INVERSE v_hat: {v_hat:.3f} m/s  floor={floor_true:.4f} dip={dip:.3f} "
          f"interior={interior}", flush=True)

    # bootstrap CI over segments for v_hat
    boot = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, B, size=B)
        rb = rates[idx]
        rf = scan_freekernel_field(rb, dist, cands, args.max_delay,
                                   args.min_delay, args.ridge, args.rank)
        boot.append(cands_mps[int(np.argmin(rf))])
    boot = np.array(boot)
    vhat_lo, vhat_hi = float(np.percentile(boot, 16)), float(np.percentile(boot, 84))
    vhat_med = float(np.median(boot))
    print(f"  v_hat bootstrap: median {vhat_med:.3f}  68% CI [{vhat_lo:.3f},{vhat_hi:.3f}] "
          f"(n_boot={args.n_boot})", flush=True)

    # hard null: autocorrelation-preserving temporal shuffle
    null_ac_floors, null_ac_vhats = [], []
    for s in range(args.n_shuffles):
        rs = autocorr_preserving_shuffle(rates, s)
        rfn = scan_freekernel_field(rs, dist, cands, args.max_delay,
                                    args.min_delay, args.ridge, args.rank)
        null_ac_floors.append(float(rfn.min()))
        null_ac_vhats.append(float(cands_mps[int(np.argmin(rfn))]))
    null_ac_floors = np.array(null_ac_floors)
    floor_ac_mean = float(null_ac_floors.mean())
    floor_ac_std = float(null_ac_floors.std())
    gap_ac = floor_ac_mean - floor_true
    ratio_ac = floor_ac_mean / (floor_true + 1e-12)
    beats_ac = bool(np.all(null_ac_floors > floor_true))
    print(f"  HARD NULL (autocorr-preserving temporal shuffle): floor "
          f"{floor_ac_mean:.4f} ± {floor_ac_std:.4f}  (true {floor_true:.4f})  "
          f"gap={gap_ac:.4f} ratio={ratio_ac:.3f}x  true<null all={beats_ac}", flush=True)

    # secondary null: velocity-shuffled geometry (EEG-template control)
    null_geo_floors = []
    for s in range(args.n_shuffles):
        d_shuf = shuffle_geometry(dist, s)
        rfn = scan_freekernel_field(rates, d_shuf, cands, args.max_delay,
                                    args.min_delay, args.ridge, args.rank)
        null_geo_floors.append(float(rfn.min()))
    null_geo_floors = np.array(null_geo_floors)
    floor_geo_mean = float(null_geo_floors.mean())
    gap_geo = floor_geo_mean - floor_true
    beats_geo = bool(np.all(null_geo_floors > floor_true))
    print(f"  (control) geometry-shuffle null floor: {floor_geo_mean:.4f}  "
          f"gap={gap_geo:.4f}  true<null all={beats_geo}", flush=True)

    err_pct = float(abs(v_hat / v_ref - 1) * 100) if np.isfinite(v_ref) else float("nan")
    ratio_v = float(v_hat / v_ref) if np.isfinite(v_ref) else float("nan")
    return {
        "n_segments": int(B), "seg_len": int(T), "n_chans": int(S),
        "fs_hz": FS, "wave_band": list(args.wave_band), "burst_band": list(args.burst_band),
        "win_ms": args.win_ms, "n_bursts": int(len(edges)),
        "dist_mm_median": float(np.median(dist[dist > 0])), "dist_mm_max": float(dist.max()),
        # references
        "v_ref_mps": v_ref, "v_ref_iqr_mps": ref_iqr, "n_ref_fits": int(len(ref_v)),
        "ref_mean_r2": float(ref_r2.mean()) if len(ref_r2) else None,
        "v_phasegrad_control_mps": v_pg,
        # inverse
        "candidates_mps": cands_mps.tolist(), "res_true": res_true.tolist(),
        "v_hat_mps": v_hat, "ratio_vhat_vref": ratio_v, "err_vs_ref_pct": err_pct,
        "floor_true": floor_true, "dip": dip, "interior_min": bool(interior),
        "v_hat_boot_median_mps": vhat_med, "v_hat_boot_ci68_mps": [vhat_lo, vhat_hi],
        # hard null
        "null_ac_floor_mean": floor_ac_mean, "null_ac_floor_std": floor_ac_std,
        "null_ac_floors": null_ac_floors.tolist(), "null_ac_vhats_mps": null_ac_vhats,
        "null_ac_gap": float(gap_ac), "null_ac_ratio": float(ratio_ac),
        "null_ac_beats_all": beats_ac,
        # geometry null (control)
        "null_geo_floor_mean": floor_geo_mean, "null_geo_floors": null_geo_floors.tolist(),
        "null_geo_gap": float(gap_geo), "null_geo_beats_all": beats_geo,
        "_rates_shape": [int(B), int(T), int(S)],
    }


def track_per_segment(lfp, pos_mm, args, rng):
    """Per-GROUP v_hat: does the inverse MOVE with the data or sit at a fixed
    grid point (the EEG failure mode)? A single 60 ms burst is too short to give
    a stable interior estimate, so we group `track_group` consecutive bursts per
    estimate (still a small fraction of all bursts) - enough signal to land
    interior, while still asking whether v_hat varies across independent groups
    of the recording and tracks the matched burst-latency reference per group."""
    dist = np.linalg.norm(pos_mm[:, None, :] - pos_mm[None, :, :], axis=2)
    envc = channel_envelope(lfp, band=tuple(args.burst_band))
    edges = detect_bursts(envc, k_std=args.burst_k)
    edges = edges[edges > int(args.skip_s * FS)]
    field = preprocess_field(lfp, band=tuple(args.wave_band))
    cands_mps = np.geomspace(args.v_lo, args.v_hi, args.n_cands)
    cands = mps_to_v(cands_mps)
    win = int(args.win_ms / 1000.0 * FS)
    T, S = field.shape

    grp = max(1, args.track_group)
    per_vhat, per_vref = [], []
    valid_edges = [e for e in edges if e - win >= 0 and e + win < T]
    n_groups = min(args.track_n, len(valid_edges) // grp)
    for gi in range(n_groups):
        ges = valid_edges[gi * grp:(gi + 1) * grp]
        # group inverse v_hat (segments as batch)
        segs = np.stack([field[e - win:e + win, :] for e in ges], axis=0)  # (grp,2win,S)
        rf = scan_freekernel_field(segs, dist, cands, args.max_delay,
                                   args.min_delay, args.ridge, args.rank)
        per_vhat.append(float(cands_mps[int(np.argmin(rf))]))
        # matched group reference (median burst-latency speed over the group)
        vrefs = []
        for e in ges:
            segenv = envc[e - win:e + win, :]
            lat = np.argmax(segenv, axis=0).astype(float) / FS * 1000.0
            amp = segenv.max(0)
            good = amp > np.percentile(amp, 20)
            if good.sum() >= 30:
                X = pos_mm[good] - pos_mm[good].mean(0)
                G = np.c_[X, np.ones(int(good.sum()))]
                y = lat[good] - lat[good].mean()
                coef, *_ = np.linalg.lstsq(G, y, rcond=None)
                bn = np.linalg.norm(coef[:2])
                if bn > 1e-6:
                    vrefs.append(1.0 / bn)
        per_vref.append(float(np.median(vrefs)) if vrefs else np.nan)
    per_vhat = np.array(per_vhat)
    per_vref = np.array(per_vref)
    ok = np.isfinite(per_vhat) & np.isfinite(per_vref)
    track_corr = float(np.corrcoef(per_vhat[ok], per_vref[ok])[0, 1]) if ok.sum() > 3 else float("nan")
    frac_at_floor = float(np.mean(per_vhat == cands_mps[0])) if len(per_vhat) else float("nan")
    frac_at_ceil = float(np.mean(per_vhat == cands_mps[-1])) if len(per_vhat) else float("nan")
    interior_frac = (float(np.mean((per_vhat > cands_mps[0]) & (per_vhat < cands_mps[-1])))
                     if len(per_vhat) else float("nan"))
    return {
        "n_tracked": int(len(per_vhat)), "track_group": int(grp),
        "per_vhat_mps": per_vhat.tolist(), "per_vref_mps": per_vref.tolist(),
        "track_spearman_pearson": track_corr,
        "vhat_spread_iqr": ([float(np.percentile(per_vhat, 25)), float(np.percentile(per_vhat, 75))]
                            if len(per_vhat) else [float("nan")] * 2),
        "vhat_median_mps": float(np.median(per_vhat)) if len(per_vhat) else float("nan"),
        "frac_at_scan_floor": frac_at_floor, "frac_at_scan_ceil": frac_at_ceil,
        "frac_interior": interior_frac,
    }


def verdict(r, track):
    if "error" in r:
        return "INCONCLUSIVE (insufficient segments)"
    v_hat, v_ref = r["v_hat_mps"], r["v_ref_mps"]
    interior = r["interior_min"]
    beats_ac = r["null_ac_beats_all"]
    ratio_ac = r["null_ac_ratio"]
    within_2x = np.isfinite(v_ref) and 0.5 <= (v_hat / v_ref) <= 2.0
    within_15x = np.isfinite(v_ref) and (1 / 1.5) <= (v_hat / v_ref) <= 1.5
    lo, hi = r["v_hat_boot_ci68_mps"]
    ci_brackets = (lo <= v_ref <= hi) if np.isfinite(v_ref) else False
    null_clear = beats_ac and ratio_ac > 1.05
    tracks = (track.get("frac_interior", 0.0) > 0.5
              and track.get("frac_at_scan_ceil", 1.0) < 0.5)
    if within_15x and null_clear and (interior or ci_brackets):
        v = "RECOVERED (v_hat within 1.5x of v_ref AND beats the hard autocorr null)"
    elif within_2x and null_clear:
        v = "PARTIAL-RECOVERED (within 2x of v_ref AND beats the hard autocorr null)"
    elif within_2x and not beats_ac:
        v = "AMBIGUOUS (within 2x of v_ref but does NOT beat the hard autocorr null - could be shared temporal structure)"
    elif beats_ac and not within_2x:
        v = "NULL-BEATEN-ONLY (beats hard null but v_hat off >2x from v_ref)"
    else:
        v = "FAILED (off >2x from v_ref and/or does not beat the hard null)"
    return v + (f" | tracks per-segment={tracks}" if track else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-s", type=float, default=150.0, dest="skip_s",
                    help="ignore the first N s (light anesthesia, few bursts)")
    ap.add_argument("--burst-band", type=float, nargs=2, default=[10.0, 100.0], dest="burst_band")
    ap.add_argument("--burst-k", type=float, default=1.5, dest="burst_k")
    ap.add_argument("--wave-band", type=float, nargs=2, default=[2.0, 50.0], dest="wave_band")
    ap.add_argument("--win-ms", type=float, default=60.0, dest="win_ms")
    ap.add_argument("--max-segs", type=int, default=64, dest="max_segs")
    ap.add_argument("--max-delay", type=int, default=30, dest="max_delay")
    ap.add_argument("--min-delay", type=int, default=1, dest="min_delay")
    ap.add_argument("--n-cands", type=int, default=17, dest="n_cands")
    ap.add_argument("--v-lo", type=float, default=0.05, dest="v_lo")  # m/s scan floor
    ap.add_argument("--v-hi", type=float, default=2.0, dest="v_hi")   # m/s scan ceiling
    ap.add_argument("--ridge", type=float, default=1e-1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=30, dest="n_boot")
    ap.add_argument("--n-shuffles", type=int, default=8, dest="n_shuffles")
    ap.add_argument("--track-n", type=int, default=20, dest="track_n")
    ap.add_argument("--track-group", type=int, default=5, dest="track_group",
                    help="bursts aggregated per per-group v_hat estimate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        args.n_cands = 9
        args.n_boot = 4
        args.n_shuffles = 3
        args.max_segs = 16
        args.track_n = 4
        args.track_group = 4

    print("INTRACRANIAL conduction-velocity INVERSE (human Utah array, 400 um)", flush=True)
    print("source: neurosmiths/Data-and-Code-for-STAR-Protocols (PMC11919625), "
          "propofol LOC non-oscillatory traveling waves", flush=True)
    fetch_data()
    pos_mm = electrode_positions_mm()

    # load only the burst-active window to keep memory modest (full file is large)
    t0 = max(0.0, args.skip_s - 5.0)
    lfp, fs, idx = load_lfp(t0_s=t0, t1_s=None)
    assert abs(fs - FS) < 1e-6, f"Fs mismatch {fs} vs {FS}"
    # shift edges: detection runs on this loaded window, so reset skip relative to t0
    args.skip_s = max(0.0, args.skip_s - t0)
    print(f"  loaded LFP window [{t0:.0f}s..end]: {lfp.shape} ({lfp.shape[0]/fs:.0f}s)", flush=True)

    rng = np.random.default_rng(args.seed)
    print("\n==== MAIN RUN ====", flush=True)
    r = run(lfp, pos_mm, args, rng)

    track = {}
    if "error" not in r:
        print("\n==== PER-SEGMENT TRACKING ====", flush=True)
        track = track_per_segment(lfp, pos_mm, args, rng)
        print(f"  tracked {track['n_tracked']} groups (x{track['track_group']} bursts): "
              f"v_hat median {track['vhat_median_mps']:.3f} IQR "
              f"{np.round(track['vhat_spread_iqr'],3).tolist()} m/s  "
              f"corr(v_hat,v_ref)={track['track_spearman_pearson']:.3f}  "
              f"interior={track['frac_interior']:.2f} @ceil={track['frac_at_scan_ceil']:.2f} "
              f"@floor={track['frac_at_scan_floor']:.2f}", flush=True)

    v = verdict(r, track)
    print("\n" + "=" * 78)
    print("SUMMARY (intracranial Utah-array velocity recovery)")
    if "error" not in r:
        print(f"  v_ref(burst-latency) = {r['v_ref_mps']:.3f} m/s  IQR "
              f"{np.round(r['v_ref_iqr_mps'],3).tolist()}")
        print(f"  v_hat(inverse)       = {r['v_hat_mps']:.3f} m/s  (boot "
              f"{r['v_hat_boot_ci68_mps'][0]:.3f}-{r['v_hat_boot_ci68_mps'][1]:.3f})  "
              f"ratio {r['ratio_vhat_vref']:.2f}x  err {r['err_vs_ref_pct']:.0f}%")
        print(f"  HARD autocorr null: true<null all-shuffles={r['null_ac_beats_all']}  "
              f"ratio {r['null_ac_ratio']:.3f}x")
        print(f"  interior min={r['interior_min']}  dip={r['dip']:.3f}")
    print(f"  VERDICT: {v}")
    print("=" * 78)

    out = {
        "source": "neurosmiths/Data-and-Code-for-STAR-Protocols (human Utah array, 400um, propofol LOC)",
        "citations": ["PMC11919625 (STAR Protocols)", "OSF ahqej (STARmain_data_filtered.mat)"],
        "ground_truth_method": "burst-latency planar regression (independent of the delayed-field inverse; the dataset's own SpatialLinearRegression approach)",
        "phasegrad_note": "Hilbert phase-gradient is INVALID here (sub-wavelength array, ~2pi wrap at 1 Hz -> spuriously tiny v); kept only as a documented-failure control.",
        "fs_hz": FS, "electrode_spacing_mm": SPACING_MM,
        "literature_anchor_mps": "cortical/anesthesia/slow-wave intracortical traveling waves ~0.1-0.8 m/s",
        "args": vars(args), "smoke": args.smoke,
        "result": r, "tracking": track, "verdict": v,
    }
    tag = "_smoke" if args.smoke else ""
    outpath = ROOT / "results" / "inverse" / f"foreigninv_intracranial{tag}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}", flush=True)


if __name__ == "__main__":
    main()
