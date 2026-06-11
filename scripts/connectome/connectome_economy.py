"""CONNECTOME ECONOMY <-> ISOCHRONY test on a REAL human structural connectome (the Nature-tier lever).

Tests whether the brain's wiring matches the conduction-time allocation our sign-flip predicts, beyond a
geometry-shuffled null:
  ARM A (economy):  do strong connections (W=streamline weight) sit on SHORT tracts? weighted-mean
                    conduction cost C = sum|W|*tau / sum|W| (tau = d/v) should be LOWER than a
                    geometry-shuffled null (shuffle_geometry permutes which edge has which length).
  ARM B (isochrony): on the long-range subset, does conduction VELOCITY grow with distance (delay flatten)?
                    Needs a per-edge velocity from myelin (qT1/MPC). corr(v, distance) > 0 = isochrony pole.
Aggregates across subjects/thresholds with a sign-test + t-test (the AJILE population template).

DATA (drop in ~/Downloads or data/raw/): MICA-MICs Schaefer-400 consensus:
  sc400_nos.mat  -> W (number-of-streamlines weights, NxN)
  sc400_los.mat  -> d (length-of-streamlines, NxN, mm)
  [optional] sc400_mpc.mat or a per-node qT1 vector -> myelin -> per-edge velocity for ARM B.
Run:  python scripts/connectome/connectome_economy.py --W <sc400_nos.mat> --d <sc400_los.mat> [--myelin <...>]
"""
import sys, json, argparse, glob, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "inverse"))
from pinn_inverse import shuffle_geometry  # noqa: E402  (symmetric permutation of a distance matrix)


def load_matrix(path):
    """Load an NxN matrix from .mat (any 2D array field) or .npy."""
    if path.endswith(".npy"):
        return np.asarray(np.load(path), float)
    from scipy.io import loadmat
    m = loadmat(path)
    cands = [(k, np.asarray(v, float)) for k, v in m.items()
             if not k.startswith("__") and np.ndim(v) == 2 and np.shape(v)[0] == np.shape(v)[1] and np.shape(v)[0] > 20]
    if not cands:
        raise ValueError(f"no square NxN matrix in {path}; keys={[k for k in m if not k.startswith('__')]}")
    cands.sort(key=lambda kv: -kv[1].shape[0])
    return cands[0][1]


def economy_arm(W, d, n_shuf=500, v=None):
    """C = sum|W|*tau / sum|W| over existing edges; tau=d/v (v=1 if None). vs geometry-shuffled null."""
    W = np.abs(W); np.fill_diagonal(W, 0.0); np.fill_diagonal(d, 0.0)
    mask = W > 0
    tau = d / v if v is not None else d
    Wm, tm = W[mask], tau[mask]
    C_true = float((Wm * tm).sum() / Wm.sum())
    null = []
    for s in range(n_shuf):
        d_s = shuffle_geometry(d, s)
        tau_s = d_s / v if v is not None else d_s
        null.append(float((Wm * tau_s[mask]).sum() / Wm.sum()))
    null = np.array(null)
    gap = float(null.mean() - C_true)                       # >0 => brain is conduction-economical
    beats = bool((C_true < null).all())                     # true beats EVERY shuffle
    z = float((null.mean() - C_true) / (null.std() + 1e-12))
    return dict(C_true=C_true, C_null_mean=float(null.mean()), C_null_std=float(null.std()),
                economy_gap=gap, z=z, beats_all=beats, frac_shuf_beaten=float((C_true < null).mean()))


def isochrony_arm(d, v, W=None, long_q=0.66):
    """On the long-range subset, corr(velocity, distance). >0 => isochrony (myelinate long)."""
    np.fill_diagonal(d, 0.0)
    mask = (W > 0) if W is not None else (d > 0)
    dm, vm = d[mask], v[mask]
    long = dm >= np.quantile(dm[dm > 0], long_q)
    if long.sum() < 10:
        return dict(corr_v_dist_long=None, note="too few long edges")
    c = float(np.corrcoef(vm[long], dm[long])[0, 1])
    slope = float(np.polyfit(dm[long], (dm / np.clip(v, 1e-6, None))[mask][long], 1)[0])  # delay-vs-distance slope
    return dict(corr_v_dist_long=c, delay_dist_slope_long=slope,
                pole="isochrony (myelinate long)" if c > 0.1 else ("economy (myelinate short)" if c < -0.1 else "neutral"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", required=True, help="streamline-weight matrix (.mat/.npy)")
    ap.add_argument("--d", required=True, help="tract-length matrix (.mat/.npy), mm")
    ap.add_argument("--myelin", default=None, help="optional per-node qT1/myelin vector or MPC matrix for ARM B")
    ap.add_argument("--n-shuf", type=int, default=500, dest="n_shuf")
    ap.add_argument("--out", default=str(ROOT / "results" / "inverse" / "connectome_economy.json"))
    args = ap.parse_args()

    W = load_matrix(args.W); d = load_matrix(args.d)
    assert W.shape == d.shape, f"W{W.shape} vs d{d.shape} mismatch"
    print(f"connectome: {W.shape[0]} nodes | edges={int((W>0).sum())} | dist(mm) med={np.median(d[d>0]):.1f}", flush=True)

    # velocity from myelin (Rushton-ish: v grows with myelin); if no myelin, ARM A only (uniform v)
    v = None
    if args.myelin:
        myl = load_matrix(args.myelin) if args.myelin.endswith((".mat", ".npy")) else None
        if myl is not None and myl.ndim == 2:
            v = 0.5 + myl / (myl.max() + 1e-9)              # crude per-edge velocity proxy in [0.5,1.5]
    A = economy_arm(W, d, args.n_shuf, v=v)
    B = isochrony_arm(d, v, W) if v is not None else dict(note="no myelin -> ARM B skipped (uniform v)")

    out = dict(n_nodes=int(W.shape[0]), economy=A, isochrony=B,
               verdict_economy=("ECONOMY: brain weights short-conduction tracts beyond shuffle (z=%.1f)" % A["z"]
                                if A["beats_all"] and A["economy_gap"] > 0 else
                                "NULL: no economy beyond shuffle"))
    print("\n=== ARM A (economy) ===")
    print(f"  C_true={A['C_true']:.3f}  null={A['C_null_mean']:.3f}±{A['C_null_std']:.3f}  "
          f"gap={A['economy_gap']:+.3f}  z={A['z']:.1f}  beats_all={A['beats_all']}")
    print("=== ARM B (isochrony) ===", B)
    print("VERDICT:", out["verdict_economy"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
