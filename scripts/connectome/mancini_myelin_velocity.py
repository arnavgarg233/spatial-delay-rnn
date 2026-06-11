"""MYELIN-VELOCITY ARM on the Mancini multi-shell connectome (the real predicted relationship,
proper spatial null). cmat_lau.mat = 14 subjects x 7 measures x 463 x 463, measures =
[nos, len, ad, g, mtv, cv, delay] (number-of-streamlines weight, tract length mm, axon diameter,
g-ratio, myelin-volume, CONDUCTION VELOCITY, delay).

THE FRAMEWORK PREDICTION (conduction-time economy): the brain should invest conduction velocity
(via myelin) to MINIMIZE total conduction time  C = sum_edges |W| * len / cv.  Equivalently, high cv
should sit on the edges where the conduction-time cost |W|*len is largest -- so reallocating cv across
edges should RAISE C.

DECISIVE TEST (vs a PROPER spatial null, not a plain shuffle): permute cv across edges WITHIN length
deciles. This preserves the cv-vs-length marginal (the gross "long axons are myelinated" trend and the
dominant spatial-autocorrelation axis) and asks whether, AT MATCHED LENGTH, cv is still allocated to the
high-|W| edges that make C economical. If C_true < C_null beyond this null, the velocity allocation is
conduction-time-economical for a reason smoothness/length alone cannot explain. Aggregated across the 14
subjects with a sign test + t (the AJILE population template). Secondary: corr(cv, mtv) (myelin drives
velocity -- the mechanistic link) and corr(cv, len) (the economy<->isochrony sign).

Run: python scripts/connectome/mancini_myelin_velocity.py [--data <cmat_lau.mat>] [--nperm 2000]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "mancini" / "cmat_lau.mat"  # place the Mancini cmat_lau.mat here
MEAS = {"nos": 0, "len": 1, "ad": 2, "g": 3, "mtv": 4, "cv": 5, "delay": 6}


def stratified_perm(cv, length, nbins, rng):
    """Permute cv WITHIN length-deciles (preserve the cv-length marginal / spatial-autocorr axis)."""
    out = cv.copy()
    qs = np.quantile(length, np.linspace(0, 1, nbins + 1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    for b in range(nbins):
        m = (length >= qs[b]) & (length < qs[b + 1])
        idx = np.where(m)[0]
        if idx.size > 1:
            out[idx] = cv[idx][rng.permutation(idx.size)]
    return out


def cost(W, length, cv):
    return float((W * length / np.clip(cv, 1e-9, None)).sum())


def run_subject(W, length, cv, mtv, nperm, nbins, seed):
    m = (W > 0) & (length > 0) & (cv > 0)
    iu = np.triu_indices_from(W, k=1)
    sel = m[iu]
    Wm, lm, cvm, mtvm = W[iu][sel], length[iu][sel], cv[iu][sel], mtv[iu][sel]
    C_true = cost(Wm, lm, cvm)
    rng = np.random.default_rng(seed)
    # PROPER null: cv permuted within length deciles
    null_strat = np.array([cost(Wm, lm, stratified_perm(cvm, lm, nbins, rng)) for _ in range(nperm)])
    # weak floor: cv permuted across ALL edges
    null_plain = np.array([cost(Wm, lm, cvm[rng.permutation(cvm.size)]) for _ in range(nperm)])
    z_strat = (null_strat.mean() - C_true) / (null_strat.std() + 1e-12)   # >0 => true is cheaper
    z_plain = (null_plain.mean() - C_true) / (null_plain.std() + 1e-12)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else float("nan")
    return dict(
        n_edges=int(sel.sum()), C_true=C_true,
        C_null_strat=float(null_strat.mean()), C_null_plain=float(null_plain.mean()),
        economy_gap_strat=float(null_strat.mean() - C_true),     # >0 => economical beyond the spatial null
        z_strat=float(z_strat), z_plain=float(z_plain),
        beats_all_strat=bool((C_true < null_strat).all()),
        frac_beat_strat=float((C_true < null_strat).mean()),
        corr_cv_mtv=corr(cvm, mtvm), corr_cv_len=corr(cvm, lm), corr_cv_W=corr(cvm, Wm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--nbins", type=int, default=10, help="length deciles for the stratified null")
    ap.add_argument("--out", default=str(ROOT / "results" / "connectome" / "mancini_myelin_velocity.json"))
    args = ap.parse_args()

    from scipy.io import loadmat
    cmat = loadmat(args.data)["cmat"]            # (14, 7, 463, 463)
    S = cmat.shape[0]
    print(f"Mancini cmat {cmat.shape} | {S} subjects | proper spatial null = cv shuffled within length deciles\n", flush=True)

    rows = []
    for s in range(S):
        W = cmat[s, MEAS["nos"]].astype(float)
        length = cmat[s, MEAS["len"]].astype(float)
        cv = cmat[s, MEAS["cv"]].astype(float)
        mtv = cmat[s, MEAS["mtv"]].astype(float)
        r = run_subject(W, length, cv, mtv, args.nperm, args.nbins, seed=s)
        rows.append(r)
        print(f"  sub-{s+1:02d}: edges={r['n_edges']:5d}  econ_gap_strat={r['economy_gap_strat']:+.3e} "
              f"z_strat={r['z_strat']:+5.1f} beats_all={r['beats_all_strat']}  "
              f"corr(cv,mtv)={r['corr_cv_mtv']:+.2f} corr(cv,len)={r['corr_cv_len']:+.2f}", flush=True)

    z = np.array([r["z_strat"] for r in rows])
    gaps = np.array([r["economy_gap_strat"] for r in rows])
    npos = int((gaps > 0).sum()); nbeat = sum(r["beats_all_strat"] for r in rows)
    t = float(z.mean() / (z.std(ddof=1) / np.sqrt(len(z)) + 1e-12))
    cv_mtv = np.array([r["corr_cv_mtv"] for r in rows]); cv_len = np.array([r["corr_cv_len"] for r in rows])
    pole = ("ECONOMY (faster on short edges)" if cv_len.mean() < -0.05 else
            "ISOCHRONY (faster on long edges)" if cv_len.mean() > 0.05 else "neutral cv-length")
    economical = (npos >= len(rows) - 1) and z.mean() > 0
    out = dict(
        n_subjects=len(rows), nperm=args.nperm, nbins=args.nbins,
        z_strat_mean=float(z.mean()), z_strat_t=t, n_pos_gap=npos, n_beats_all=nbeat,
        corr_cv_mtv_mean=float(cv_mtv.mean()), corr_cv_len_mean=float(cv_len.mean()), pole=pole,
        verdict=("CONDUCTION-TIME ECONOMY IN REAL CV ALLOCATION beyond a length-stratified spatial null "
                 f"({npos}/{len(rows)} pos, z~{z.mean():.1f}, t={t:.1f})" if economical else
                 "NULL: real cv allocation not economical beyond the spatial null"),
        rows=rows)
    print("\n=== MYELIN-VELOCITY ARM (Mancini, 14 subjects) ===")
    print(f"  conduction-time economy vs length-stratified null: {npos}/{len(rows)} positive, "
          f"beats-all {nbeat}/{len(rows)}, mean z={z.mean():.1f}, t={t:.1f}")
    print(f"  corr(cv, myelin)={cv_mtv.mean():+.2f}  (mechanistic link)   corr(cv, length)={cv_len.mean():+.2f}  -> {pole}")
    print("  VERDICT:", out["verdict"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("  wrote", args.out)


if __name__ == "__main__":
    main()
