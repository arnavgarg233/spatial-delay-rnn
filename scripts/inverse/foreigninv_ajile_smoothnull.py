"""AJILE12 SMOOTHNESS-PRESERVING null (the test that actually decides the AJILE paragraph).

The label-shuffle null (foreigninv_ajile.py) breaks the metric AND destroys spatial smoothness, so
"true beats shuffle" is confounded with "smooth geometry fits the smooth field better." This test
removes that confound with a CROSS-SUBJECT GEOMETRY null: run each subject's neural field through
every OTHER subject's REAL electrode geometry (matched to a common channel count). The donor geometries
are equally smooth, real, and distance-matched - but they are the WRONG metric for this brain's field.

  true_beats_donors : is subject A's residual floor with ITS OWN geometry lower than with all 11 donors?
Population sign-test across subjects. If true beats the smooth donors -> the EXACT metric matters, not
smoothness. If true ~ donors -> the AJILE effect was a smoothness artifact (graveyard).

Reuses foreigninv_ajile.stream_subject + foreigninv_realdata.{preprocess_field, scan_freekernel_field}.
"""
import sys, json, time, argparse, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "inverse"))
import foreigninv_realdata as rd
import foreigninv_ajile as aj


def floor_for_geometry(rates, xyz, cands_mps, args):
    """min residual over velocity candidates for a given (rates, geometry)."""
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    cands = cands_mps * rd.MM_PER_M / rd.FS                    # m/s -> samples (mps_to_v)
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
    ap.add_argument("--reps", type=int, default=6, dest="reps", help="random subsamples per donor for a robust null")
    ap.add_argument("--out", default=str(ROOT / "results" / "inverse" / "foreigninv_ajile_smoothnull.json"))
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

    # stream every subject once: cache (rates, xyz)
    cache = {}
    for s in subs:
        try:
            ts, xyz, fs = aj.stream_subject(by_sub[s], args.win_s, args.skip_s, args.seg_s, args.max_chan)
            rd.FS = fs
            cache[s] = (rd.preprocess_field(ts), xyz, fs)
            print(f"  streamed {s}: rates{cache[s][0].shape} xyz{xyz.shape}", flush=True)
        except Exception as e:
            print(f"  !! {s} stream failed: {e}", flush=True)
    subs = [s for s in subs if s in cache]
    k = min(cache[s][1].shape[0] for s in subs)              # common channel count
    cands = np.geomspace(args.v_lo, args.v_hi, args.n_cands)
    print(f"\ncommon channel count k={k} | subjects={len(subs)} | cross-subject smoothness null\n", flush=True)

    rows = []
    for A in subs:
        ratesA, xyzA, fsA = cache[A]
        rd.FS = fsA
        selA = np.random.default_rng(0).permutation(xyzA.shape[0])[:k]; ratesA_k = ratesA[:, :, selA]; xyzA = xyzA[selA]                          # first k channels of A's field
        floor_true = floor_for_geometry(ratesA_k, xyzA[:k], cands, args)
        donor_floors = []
        rng = np.random.default_rng(hash(A) % (2**32))
        for B in subs:
            if B == A:
                continue
            xyzB = cache[B][1]; nB = xyzB.shape[0]
            for _ in range(args.reps):
                sel = rng.permutation(nB)[:k]                    # random k-subsample of donor B's electrodes
                donor_floors.append(floor_for_geometry(ratesA_k, xyzB[sel], cands, args))
        donor_floors = np.array(donor_floors)
        beats_all = bool((floor_true < donor_floors).all())
        gap = float(donor_floors.mean() - floor_true)
        z = float((donor_floors.mean() - floor_true) / (donor_floors.std() + 1e-12))
        rows.append(dict(sub=A, floor_true=floor_true, donor_mean=float(donor_floors.mean()),
                         donor_std=float(donor_floors.std()), gap=gap, z=z, beats_all=beats_all,
                         frac_donors_beaten=float((floor_true < donor_floors).mean())))
        print(f"  {A}: true={floor_true:.4f} donors={donor_floors.mean():.4f}±{donor_floors.std():.4f} "
              f"gap={gap:+.4f} z={z:.1f} beats_all={beats_all}", flush=True)

    n = len(rows)
    g = np.array([r["gap"] for r in rows]); b = sum(r["beats_all"] for r in rows); npos = int((g > 0).sum())
    t = float(g.mean() / (g.std(ddof=1) / math.sqrt(n) + 1e-12)) if n > 1 else float("nan")
    out = dict(config=vars(args), n_subjects=n, minutes=round((time.time() - t0) / 60, 1), rows=rows,
               gap_mean=float(g.mean()), gap_t=t, n_beats_all_donors=b, n_pos_gap=npos,
               verdict=("SMOOTHNESS-ROBUST: true metric beats smooth cross-subject geometries (%d/%d beat-all, t=%.2f)" % (b, n, t)
                        if (b >= n - 1 and g.mean() > 0) else
                        "SMOOTHNESS-CONFOUNDED: true ~ smooth donors -> AJILE is a smoothness artifact (graveyard)"))
    print("\n=== CROSS-SUBJECT SMOOTHNESS NULL ===")
    print(f"  n={n} | true-beats-all-donors: {b}/{n} | gap>0: {npos}/{n} | mean_gap={g.mean():+.4f} t={t:.2f}")
    print("VERDICT:", out["verdict"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
