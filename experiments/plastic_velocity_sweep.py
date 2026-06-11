"""GRADED-DIFFICULTY hardening of the plastic-velocity result.

plastic_velocity.py shows plastic-v (0.85) beats matched-budget uniform-v (0.50=chance) at ONE
tight deadline -- a clean but BINARY solve-vs-fail dissociation. A reviewer will ask: does the
advantage only exist at the cliff where uniform collapses to chance, or is it a graded law?

This sweeps the DEADLINE (the time budget the signal has to cross the long bridges):
  - loose deadline  -> even matched-budget uniform delivers in time -> gap(plastic-uniform) ~ 0
  - tight deadline  -> only a net that myelinates the LONG bridges (plastic-v, isochrony pole)
                       delivers -> gap large
If the gap grows monotonically as the deadline tightens, the binary result becomes a DOSE-RESPONSE
curve: "the harder the conduction-time problem, the more velocity ALLOCATION (not budget) matters."
That is far more scrutiny-proof than solve-vs-fail, and it is the isochrony pole of the two-time-
economies story shown as a graded law.

Reuses experiments/plastic_velocity.py:run_seed verbatim (same matched-myelin uniform control).
"""
import sys, json, time, math, argparse
sys.path.insert(0, "experiments")
import numpy as np
import torch
import plastic_velocity as pv


BASE_CFG = dict(
    N=64, K=3, R=0.5, gap=2.6, spread=1.0, n_bridge=4,
    v_min=0.16, v_max=3.0, v0=1.0,
    min_delay=1, max_delay=24,
    alpha=0.4, T=16, read_window=3,           # deadline is swept below
    steps=400, lr=5e-3, lr_v=0.1, l1=1e-3, lam_M=1.2e-3,
    imp_subsample=None,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", type=str, default=pv.DEV)
    ap.add_argument("--deadlines", type=str, default="4,5,6,8,10,12",
                    help="comma list of deadline values (tight->loose). read_window stays 3.")
    ap.add_argument("--out", type=str, default="results/experiments/plastic_velocity_sweep.json")
    args = ap.parse_args()

    deadlines = [int(x) for x in args.deadlines.split(",")]
    print(f"device: {args.device} | torch {torch.__version__}", flush=True)
    print(f"deadline sweep {deadlines}  seeds={args.seeds} steps={args.steps}", flush=True)

    t0 = time.time()
    rows = []  # one per (deadline, seed)
    for dl in deadlines:
        cfg = dict(BASE_CFG)
        cfg["deadline"] = dl
        cfg["steps"] = args.steps
        # keep read_window < deadline so the window is well-formed
        cfg["read_window"] = min(3, max(1, dl - 1))
        for s in range(args.seeds):
            r = pv.run_seed(s, cfg, args.device)
            rec = dict(deadline=dl, seed=s,
                       plastic_ho=r["plastic_ho"], uniform_ho=r["uniform_ho"],
                       shuffle_ho=r["shuffle_ho"], gap=r["gap"],
                       gap_vs_shuffle=r["gap_vs_shuffle"],
                       corr_v_len=r["corr_v_len"], corr_tau_dist_plastic=r["corr_tau_dist_plastic"],
                       corr_tau_dist_uniform=r["corr_tau_dist_uniform"],
                       v_bridge=r["v_bridge"], v_nonbridge=r["v_nonbridge"],
                       M_star=r["M_star"], M_uniform=r["M_uniform"], v_uniform=r["v_uniform"])
            rows.append(rec)
            print(f"  deadline={dl:2d} seed{s}: plastic={r['plastic_ho']:.3f} "
                  f"uniform={r['uniform_ho']:.3f} gap={r['gap']:+.3f} "
                  f"corr(v,len)={r['corr_v_len']:+.3f} corr(tau,dist)p={r['corr_tau_dist_plastic']:.3f}",
                  flush=True)

    # summarize gap vs deadline
    summary = []
    for dl in deadlines:
        sub = [r for r in rows if r["deadline"] == dl]
        gaps = np.array([r["gap"] for r in sub])
        summary.append(dict(
            deadline=dl,
            plastic_ho=float(np.mean([r["plastic_ho"] for r in sub])),
            uniform_ho=float(np.mean([r["uniform_ho"] for r in sub])),
            gap_mean=float(gaps.mean()), gap_sd=float(gaps.std()),
            gap_t=float(gaps.mean() / (gaps.std() / math.sqrt(len(gaps)) + 1e-9)),
            corr_v_len=float(np.mean([r["corr_v_len"] for r in sub])),
            corr_tau_dist_plastic=float(np.mean([r["corr_tau_dist_plastic"] for r in sub])),
        ))

    # is the gap monotone in difficulty? (tighter deadline = harder = bigger gap)
    gaps_by_tight = [s["gap_mean"] for s in sorted(summary, key=lambda x: x["deadline"])]
    # tighter (smaller deadline) should have LARGER gap -> gaps should be DECREASING in deadline
    monotone = all(gaps_by_tight[i] >= gaps_by_tight[i + 1] - 0.03 for i in range(len(gaps_by_tight) - 1))
    span = float(max(gaps_by_tight) - min(gaps_by_tight))
    out = dict(config=BASE_CFG, deadlines=deadlines, seeds=args.seeds,
               minutes=round((time.time() - t0) / 60, 2),
               rows=rows, summary=summary,
               gap_monotone_in_difficulty=bool(monotone), gap_span=span,
               verdict=("DOSE-RESPONSE (gap grows as deadline tightens)" if (monotone and span > 0.1)
                        else ("graded but noisy" if span > 0.1 else "flat (gap ~ constant)")))
    print("\n=== GAP vs DEADLINE (tight->loose) ===")
    for s in sorted(summary, key=lambda x: x["deadline"]):
        print(f"  deadline {s['deadline']:2d}: plastic {s['plastic_ho']:.3f}  uniform {s['uniform_ho']:.3f}  "
              f"gap {s['gap_mean']:+.3f} ± {s['gap_sd']:.3f} (t={s['gap_t']:.1f})  corr(v,len)={s['corr_v_len']:+.2f}")
    print("VERDICT:", out["verdict"], "| gap_span", round(span, 3))
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out, "in", out["minutes"], "min")


if __name__ == "__main__":
    main()
