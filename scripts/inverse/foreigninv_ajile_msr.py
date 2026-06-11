"""AJILE12 - MORAN SPECTRAL RANDOMIZATION (MSR) null: the FIELD-STANDARD spatial-autocorrelation
-preserving test (deep-res: prefer BrainSMASH/MSR over the spin test, which distorts iEEG point-cloud
distances). Upgrades the cross-subject donor-geometry null with a WITHIN-subject surrogate that preserves
the electrode-layout spatial autocorrelation exactly while breaking the metric->field correspondence.

THE TEST. For each subject we keep the TRUE electrode geometry (distance matrix d -> delays tau=round(d/v))
and ask whether the velocity inverse explains the REAL neural field better than it explains MSR-surrogate
fields that share the field's spatial autocorrelation but have the electrode pattern re-mixed.
  Surrogate: T = V diag(s) V', s in {-1,+1}^k random, V = Moran eigenvectors of the electrode-distance
  graph. T is orthogonal and commutes with the Moran spectrum, so |V' (T x)| = |V' x| per eigenmode =>
  the surrogate field has the SAME spatial autocorrelation (Moran power spectrum) as the real field, but
  the inter-electrode pattern is scrambled. Each electrode time series keeps its own dynamics (T mixes
  across space, not time), so temporal structure is preserved -- only the spatial->metric alignment breaks.
  floor_true  = min over velocity of the one-step delayed-field residual on the REAL field, true geometry.
  floor_surr  = same on each MSR-surrogate field, true geometry.
If floor_true < floor_surr beyond chance, the inverse's fit is SPECIFIC to the real field's metric-aligned
spatial fine-structure, not merely to its spatial smoothness. Population sign-test + t (AJILE template).

Reuses foreigninv_ajile.stream_subject + foreigninv_realdata.{preprocess_field, scan_freekernel_field}.
Needs only: dandi/remfile/pynwb (stream) + numpy/scipy. (MSR is implemented here in numpy; no brainsmash dep.)
"""
import sys, json, time, argparse, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "inverse"))
import foreigninv_realdata as rd
import foreigninv_ajile as aj


def moran_eigvecs(xyz, sigma=None):
    """Moran eigenvectors of the electrode-distance graph (Gaussian spatial weights)."""
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    if sigma is None:
        sigma = np.median(d[d > 0]) + 1e-9
    W = np.exp(-(d ** 2) / (2 * sigma ** 2)); np.fill_diagonal(W, 0.0)
    n = xyz.shape[0]; C = np.eye(n) - np.ones((n, n)) / n
    M = C @ W @ C; M = (M + M.T) / 2.0
    _, V = np.linalg.eigh(M)
    return V


def msr_surrogate(rates, V, rng):
    """Spatial re-mixing T = V diag(s) V' preserving the Moran spectrum (spatial autocorrelation)."""
    s = rng.choice([-1.0, 1.0], size=V.shape[1])
    T = (V * s) @ V.T                       # (k,k) orthogonal, commutes with Moran eigenbasis
    return rates @ T.T                       # mix across electrodes only


def floor_for_field(rates, d, cands_mps, args):
    cands = cands_mps * rd.MM_PER_M / rd.FS
    res, _ = rd.scan_freekernel_field(rates, d, cands, args.max_delay, args.min_delay, args.ridge, args.rank)
    return float(np.min(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=12, dest="n_subjects")
    ap.add_argument("--win-s", type=float, default=40.0, dest="win_s")
    ap.add_argument("--skip-s", type=float, default=120.0, dest="skip_s")
    ap.add_argument("--seg-s", type=float, default=1.0, dest="seg_s")
    ap.add_argument("--max-chan", type=int, default=64, dest="max_chan")
    ap.add_argument("--n-cands", type=int, default=9, dest="n_cands")
    ap.add_argument("--v-lo", type=float, default=0.05, dest="v_lo")
    ap.add_argument("--v-hi", type=float, default=2.0, dest="v_hi")
    ap.add_argument("--max-delay", type=int, default=40, dest="max_delay")
    ap.add_argument("--min-delay", type=int, default=1, dest="min_delay")
    ap.add_argument("--ridge", type=float, default=1e-1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--n-surr", type=int, default=200, dest="n_surr")
    ap.add_argument("--out", default=str(ROOT / "results" / "inverse" / "foreigninv_ajile_msr.json"))
    args = ap.parse_args()

    from dandi.dandiapi import DandiAPIClient
    t0 = time.time()
    with DandiAPIClient() as c:
        ds = c.get_dandiset("000055", "draft")
        assets = [a for a in ds.get_assets() if a.path.endswith(".nwb")]
    by_sub = {}
    for a in assets:
        s = a.path.split("/")[0]
        if s not in by_sub or a.size < by_sub[s].size:
            by_sub[s] = a
    subs = sorted(by_sub)[: args.n_subjects]
    cands = np.geomspace(args.v_lo, args.v_hi, args.n_cands)

    rows = []
    for s in subs:
        try:
            ts, xyz, fs = aj.stream_subject(by_sub[s], args.win_s, args.skip_s, args.seg_s, args.max_chan)
            rd.FS = fs
            rates = rd.preprocess_field(ts)                      # (n_win, T, k)
            d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
            V = moran_eigvecs(xyz)
            floor_true = floor_for_field(rates, d, cands, args)
            rng = np.random.default_rng(hash(s) % (2 ** 32))
            surr = np.array([floor_for_field(msr_surrogate(rates, V, rng), d, cands, args)
                             for _ in range(args.n_surr)])
            gap = float(surr.mean() - floor_true)
            z = float(gap / (surr.std() + 1e-12))
            beats_all = bool((floor_true < surr).all())
            rows.append(dict(sub=s, floor_true=floor_true, surr_mean=float(surr.mean()),
                             surr_std=float(surr.std()), gap=gap, z=z, beats_all=beats_all,
                             frac_beaten=float((floor_true < surr).mean())))
            print(f"  {s}: true={floor_true:.4f} surr={surr.mean():.4f}±{surr.std():.4f} "
                  f"gap={gap:+.4f} z={z:.1f} beats_all={beats_all}", flush=True)
        except Exception as e:
            print(f"  !! {s} failed: {e}", flush=True)

    n = len(rows)
    g = np.array([r["gap"] for r in rows]); npos = int((g > 0).sum())
    nbeat = sum(r["beats_all"] for r in rows)
    t = float(g.mean() / (g.std(ddof=1) / math.sqrt(n) + 1e-12)) if n > 1 else float("nan")
    out = dict(config=vars(args), n_subjects=n, n_surr=args.n_surr,
               minutes=round((time.time() - t0) / 60, 1), rows=rows,
               gap_mean=float(g.mean()), gap_t=t, n_pos_gap=npos, n_beats_all=nbeat,
               verdict=("MSR-ROBUST: true metric beats spatial-autocorrelation-matched surrogate fields "
                        f"({npos}/{n} pos, {nbeat}/{n} beat-all, t={t:.2f})"
                        if (npos >= n - 1 and g.mean() > 0) else
                        "MSR-NULL: true ~ spatial-autocorr surrogates (effect is a smoothness artifact)"))
    print("\n=== AJILE12 MORAN-SPECTRAL-RANDOMIZATION NULL ===")
    print(f"  n={n} | gap>0: {npos}/{n} | beats-all: {nbeat}/{n} | mean_gap={g.mean():+.4f} t={t:.2f}")
    print("VERDICT:", out["verdict"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
