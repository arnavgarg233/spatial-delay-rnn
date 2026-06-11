"""THE ECONOMY<->ISOCHRONY SIGN-FLIP, made formal (honest: one unconditional pole + one conditional law).

Per-edge conduction velocity v_k on edges of length d_k, under a shared speed budget sum_k v_k = B.
tau_k = d_k / v_k is the arrival lag. Two objectives, two optima (both KKT-exact, verified symbolically
+ numerically):

  SYNCHRONY / ISOCHRONY pole (UNCONDITIONAL theorem):
      minimize  Var_k(tau_k)   s.t.  sum_k v_k = B
      Optimum:  v_k = B * d_k / sum_j d_j   (v PROPORTIONAL TO d) -> tau_k = const, Var = 0, corr(v,d) = +1.
      Proof: Var(tau) >= 0 with equality iff all tau_k equal; tau_k = c forces v_k = d_k/c, and the budget
      fixes c = sum_j d_j / B. So v ∝ d is the global minimizer, achieving Var=0 exactly. This is the
      never-formalized version of the 20-year qualitative isochrony claim (Salami 2003; Kimura 2009;
      Talidou/Lefebvre 2021).

  ECONOMY pole (CONDITIONAL law -- the load-bearing honesty):
      minimize  C = sum_k |W_k| d_k / v_k   s.t.  sum_k v_k = B          (weighted conduction-time cost)
      Lagrangian KKT:  v_k = sqrt(|W_k| d_k / lambda)  =>  v_k PROPORTIONAL TO sqrt(|W_k| d_k).
      If the coupling mass decays with distance as |W_k| ~ d_k^(-p), then v_k ~ d_k^((1-p)/2), so
          corr(v, d) > 0  iff  p < 1   (myelinate long, isochrony-like)
          corr(v, d) < 0  iff  p > 1   (myelinate short, ECONOMY)   -- sign-flip EXACTLY at p = 1.
      The seRNN's measured -0.71 (pareto_dissociation w=0) is NOT a clean theorem of the budget LP: the
      naive sqrt law on a flat coupling field gives POSITIVE corr. The negative sign is produced by the
      spatial penalty driving p>1 (steep |W|-vs-d decay) PLUS lag discretization clip(round(d/v)) -- an
      EMPIRICAL property of the trained net, not a law of nature. We state it as a conditional law and
      verify the exact p=1 crossing numerically.

Run: python scripts/theory/signflip_theorem.py
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "theory" / "signflip_theorem.json"


def synchrony_pole(d, B):
    """min Var(tau=d/v) s.t. sum v = B -> v = B d / sum d, tau const, corr(v,d)=+1."""
    v = B * d / d.sum()
    tau = d / v
    return dict(v=v, tau=tau, var_tau=float(np.var(tau)), corr_v_d=float(np.corrcoef(v, d)[0, 1]))


def economy_pole(d, W, B):
    """min sum |W| d / v s.t. sum v = B -> v ∝ sqrt(|W| d) (KKT)."""
    a = np.abs(W) * d
    v_unnorm = np.sqrt(np.clip(a, 1e-12, None))
    v = B * v_unnorm / v_unnorm.sum()
    return dict(v=v, corr_v_d=float(np.corrcoef(v, d)[0, 1]),
                C=float((np.abs(W) * d / v).sum()))


def verify_kkt_economy(d, W, B, eps=1e-4):
    """Numerically confirm v ∝ sqrt(|W| d) is the constrained minimizer of C (gradient on the simplex)."""
    r = economy_pole(d, W, B); v = r["v"]
    C0 = (np.abs(W) * d / v).sum()
    rng = np.random.default_rng(0); worse = 0
    for _ in range(2000):
        i, j = rng.integers(0, len(v), 2)
        if i == j:
            continue
        vp = v.copy(); step = eps * B
        vp[i] += step; vp[j] -= step
        if (vp > 0).all() and (np.abs(W) * d / vp).sum() < C0 - 1e-9:
            worse += 1
    return worse == 0, C0


def main():
    rng = np.random.default_rng(7)
    n = 400
    d = np.sort(0.5 + 9.5 * rng.random(n))           # edge lengths in [0.5, 10]
    B = float(n)                                      # speed budget

    # --- SYNCHRONY pole (unconditional) ---
    syn = synchrony_pole(d, B)

    # --- ECONOMY pole: sweep the coupling-decay exponent p (|W| ~ d^-p) to find the sign-flip ---
    ps = np.round(np.arange(0.0, 2.01, 0.25), 2)
    sweep = []
    for p in ps:
        W = d ** (-p)
        e = economy_pole(d, W, B)
        sweep.append(dict(p=float(p), corr_v_d=e["corr_v_d"], slope_exp=float((1 - p) / 2)))
    # exact crossing: corr changes sign between the p straddling 1.0
    crossing = None
    for k in range(len(sweep) - 1):
        if sweep[k]["corr_v_d"] * sweep[k + 1]["corr_v_d"] < 0:
            a, b = sweep[k], sweep[k + 1]
            crossing = a["p"] + (b["p"] - a["p"]) * (0 - a["corr_v_d"]) / (b["corr_v_d"] - a["corr_v_d"])
    kkt_ok, C_opt = verify_kkt_economy(d, d ** (-1.5), B)

    out = dict(
        n_edges=n, budget=B,
        synchrony_pole=dict(corr_v_d=syn["corr_v_d"], var_tau=syn["var_tau"],
                            statement="v ∝ d is the global minimizer of Var(tau); tau=const, corr(v,d)=+1 (UNCONDITIONAL)"),
        economy_pole=dict(law="v ∝ sqrt(|W| d); if |W| ~ d^-p then v ~ d^((1-p)/2)",
                          sign_flip_at_p=1.0, kkt_minimizer_verified=bool(kkt_ok),
                          numeric_crossing_p=(round(float(crossing), 3) if crossing is not None else None),
                          sweep=sweep),
        honest_framing=("SYNCHRONY pole = unconditional theorem (v∝d). ECONOMY pole = CONDITIONAL law: "
                        "corr(v,d)<0 iff p>1 (coupling mass decays faster than 1/distance). The seRNN's "
                        "-0.71 holds because the spatial penalty puts it at p>1 -- an empirical property, "
                        "not a law of nature."),
    )
    print("=== SIGN-FLIP THEOREM ===")
    print(f"SYNCHRONY pole : v∝d -> Var(tau)={syn['var_tau']:.2e}, corr(v,d)={syn['corr_v_d']:+.3f}  (UNCONDITIONAL)")
    print("ECONOMY pole sweep |W|~d^-p  ->  corr(v,d):")
    for s in sweep:
        print(f"   p={s['p']:.2f}  corr(v,d)={s['corr_v_d']:+.3f}  (v~d^{s['slope_exp']:+.2f})")
    print(f"sign-flip crossing at p = {out['economy_pole']['numeric_crossing_p']}  (theory: p=1)")
    print(f"KKT minimizer (v∝sqrt(|W|d)) numerically optimal: {kkt_ok}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
