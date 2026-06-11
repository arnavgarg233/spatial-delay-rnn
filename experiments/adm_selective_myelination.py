"""ADM KILLER-PREDICTION: task-driven velocity changes land SELECTIVELY on task-ACTIVATED
(high-edge-importance) pathways, not globally.

This is the in-silico test of the falsifiable prediction that Bacmeister (Nat Neurosci 2022)
already confirms CAUSALLY in mouse motor cortex: motor-learning-induced myelination is targeted
to axons that were ACTIVATED by the learning, not deposited uniformly; and the change follows a
biphasic SLOW(remodel)-then-FAST(add myelin / speed up) trajectory (Myrf-cKO blocks the gain ->
causal). Our spatially-embedded delay-coupled RNN with LEARNABLE per-edge conduction velocity
v_ij under a myelin budget is the computational analogue: training = "learning", Delta-v_ij =
"myelin added on edge ij", edge-importance = "was this axon activated/required by the task".

THE PREDICTION (pre-registered sign + the null that would falsify it):
  P1  corr(Delta-v_ij, importance_ij) > 0           velocity gain concentrates on important edges
  P2  SHUFFLED-importance control collapses to ~0    (re-label importance across edges -> the
                                                      alignment is destroyed; this is the decisive
                                                      null. If the shuffle is just as correlated,
                                                      the "selectivity" was a length artifact.)
  P3  partial corr(Delta-v, importance | length) > 0 selectivity SURVIVES partialling out edge
                                                      length -> it is IMPORTANCE-targeted, not just
                                                      "make the long bridges fast" (kills the
                                                      distance confound; this is what makes it a
                                                      real ADM result and not the economy pole).
  P4  biphasic trajectory: |Delta-v| on important   the Bacmeister slow-then-speed signature:
      edges stays ~flat early then accelerates,      velocity on important edges lags then surges,
      AND the importance-targeting (P1 corr) GROWS    and targeting sharpens over training.
      over training epochs.

A NULL (P1<=0 or shuffle just as high or partial<=0) is reported honestly: it would mean velocity
plasticity is NOT importance-targeted in this model and the ADM mapping fails.

Reuses experiments/plastic_velocity.py verbatim: build_graph (two-cluster + long bridges),
PlasticVelocityRNN (fractional delays -> v gradient-learnable), train_model, edge_importance
(per-edge accuracy-drop ablation = "task activation"), eval_acc.
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import torch

import plastic_velocity as pv


# ----------------------------------------------------------------------------------------
# stats helpers
# ----------------------------------------------------------------------------------------
def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pearson(ra, rb)


def partial_corr(x, y, z):
    """corr(x, y) controlling for z, via residualization (least squares regress out z)."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    Z = np.column_stack([np.ones_like(z), z])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ bx
    ry = y - Z @ by
    return pearson(rx, ry)


# ----------------------------------------------------------------------------------------
# One seed: train a plastic-v net, measure importance-targeting of the velocity change.
# ----------------------------------------------------------------------------------------
def run_seed(seed, cfg, device, n_shuffle=200, epoch_chunks=8):
    g = pv.build_graph(cfg["N"], seed, cfg["R"], cfg["v0"], cfg["gap"], cfg["spread"],
                       cfg["min_delay"], cfg["max_delay"], cfg["n_bridge"])
    d = g["d"]
    Tmax, rw, dd = cfg["T"], cfg["read_window"], cfg["deadline"]

    m = pv.PlasticVelocityRNN(g, cfg["K"], velocity_mode="plastic", v_min=cfg["v_min"],
                              v_max=cfg["v_max"], v0=cfg["v0"], min_delay=cfg["min_delay"],
                              max_delay=cfg["max_delay"], alpha=cfg["alpha"], seed=seed).to(device)
    v_init = m.edge_velocity().detach().clone()

    # guard the integer-lag null: velocity gradient must be alive at init
    code = pv.make_codes(16, cfg["K"], torch.Generator().manual_seed(1), device)
    logits = m(code, Tmax, rw, dd)
    gv = torch.autograd.grad(logits.sum(), m.g)[0]
    v_grad_nonzero = int((gv.abs() > 1e-9).sum())

    dl_len = d[m.ui[0].to(d.device), m.ui[1].to(d.device)].cpu().numpy()  # (E,) edge length
    is_bridge = g["bridge_mask"][m.ui[0].cpu(), m.ui[1].cpu()].cpu().numpy().astype(bool)

    # ---- P4 trajectory: train in chunks, log Delta-v and the targeting corr per chunk ----
    total_steps = cfg["steps"]
    per = max(1, total_steps // epoch_chunks)
    traj = []
    # importance is expensive; measure it once on the FINAL model (the "task-activation"
    # fingerprint of the converged net). For the trajectory we track Delta-v growth + its
    # alignment to the FINAL importance vector (does targeting sharpen toward the solution?).
    acc0 = pv.eval_acc(m, Tmax, rw, dd, device, seed)
    steps_done = 0
    snapshots = []  # (steps_done, v_current)
    snapshots.append((0, v_init.cpu().numpy().copy()))
    while steps_done < total_steps:
        chunk = min(per, total_steps - steps_done)
        pv.train_model(m, Tmax, rw, dd, chunk, cfg["lr"], cfg["lr_v"], cfg["l1"],
                       cfg["lam_M"], device, seed)
        steps_done += chunk
        v_now = m.edge_velocity().detach().cpu().numpy().copy()
        acc_now = pv.eval_acc(m, Tmax, rw, dd, device, seed)
        snapshots.append((steps_done, v_now))
        traj.append(dict(step=steps_done, acc=acc_now,
                         dv_mean=float(np.abs(v_now - v_init.cpu().numpy()).mean())))

    v_final = m.edge_velocity().detach().clone()
    dv = (v_final - v_init).cpu().numpy()              # Delta-v_ij  (the "myelin added")
    acc_final = pv.eval_acc(m, Tmax, rw, dd, device, seed)

    # ---- edge importance = task activation (accuracy drop when edge W is ablated) ----
    # full-edge ablation (dense importance vector); smaller eval batch keeps the
    # ~E*nshot forward passes tractable without changing the importance ranking.
    imp = pv.edge_importance(m, Tmax, rw, dd, device, seed,
                             batch=cfg.get("imp_batch", 256),
                             subsample=cfg.get("imp_subsample")).cpu().numpy()

    # =========================== P1: targeting correlation ===========================
    corr_dv_imp = pearson(dv, imp)
    corr_dv_imp_sp = spearman(dv, imp)

    # =========================== P2: shuffled-importance null ===========================
    rng = np.random.default_rng(777 + seed)
    shuf_corrs = []
    for _ in range(n_shuffle):
        perm = rng.permutation(len(imp))
        shuf_corrs.append(pearson(dv, imp[perm]))
    shuf_corrs = np.array(shuf_corrs)
    shuf_mean = float(shuf_corrs.mean())
    shuf_sd = float(shuf_corrs.std() + 1e-12)
    # one-sided permutation p: fraction of shuffles >= observed
    p_perm = float((shuf_corrs >= corr_dv_imp).mean())
    z_vs_shuffle = (corr_dv_imp - shuf_mean) / shuf_sd

    # =========================== P3: partial corr | length ===========================
    pcorr_dv_imp_len = partial_corr(dv, imp, dl_len)
    # also partial out length AND bridge membership (the two structural confounds)
    Zconf = np.column_stack([dl_len, is_bridge.astype(float)])
    # residualize manually for the two-covariate case
    def resid(y, Z):
        Zc = np.column_stack([np.ones(len(y)), Z])
        b, *_ = np.linalg.lstsq(Zc, y, rcond=None)
        return y - Zc @ b
    pcorr_dv_imp_full = pearson(resid(dv, Zconf), resid(imp, Zconf))

    # =========================== P4: biphasic / sharpening ===========================
    # targeting corr at each snapshot (Delta-v from init) vs FINAL importance
    sharpen = []
    for (st, v_now) in snapshots[1:]:
        dvn = v_now - v_init.cpu().numpy()
        sharpen.append(dict(step=int(st), corr=pearson(dvn, imp)))
    # biphasic: split |Delta-v| growth on important vs unimportant edges over time.
    # rank-based top-quartile so the complement is NEVER empty even with tied importances.
    order = np.argsort(np.argsort(imp))               # 0..E-1 ascending rank
    k_hi = max(1, int(round(0.25 * len(imp))))
    imp_hi = order >= (len(imp) - k_hi)               # top-quartile important edges
    dv_traj_hi, dv_traj_lo, steps_axis = [], [], []
    for (st, v_now) in snapshots:
        dvn = np.abs(v_now - v_init.cpu().numpy())
        steps_axis.append(int(st))
        dv_traj_hi.append(float(dvn[imp_hi].mean()) if imp_hi.any() else 0.0)
        dv_traj_lo.append(float(dvn[~imp_hi].mean()) if (~imp_hi).any() else 0.0)

    # contrast: mean Delta-v on important vs unimportant edges (effect size)
    dv_imp_hi = float(dv[imp_hi].mean()) if imp_hi.any() else float("nan")
    dv_imp_lo = float(dv[~imp_hi].mean()) if (~imp_hi).any() else float("nan")

    return dict(
        seed=seed, n_edges=int(len(imp)), v_grad_nonzero=v_grad_nonzero,
        acc0=acc0, acc_final=acc_final,
        corr_dv_imp=corr_dv_imp, corr_dv_imp_spearman=corr_dv_imp_sp,
        shuffle_mean=shuf_mean, shuffle_sd=shuf_sd, shuffle_p=p_perm,
        z_vs_shuffle=float(z_vs_shuffle),
        partial_corr_dv_imp_given_len=pcorr_dv_imp_len,
        partial_corr_dv_imp_given_len_bridge=pcorr_dv_imp_full,
        corr_dv_len=pearson(dv, dl_len),
        corr_imp_len=pearson(imp, dl_len),
        dv_imp_hi=dv_imp_hi, dv_imp_lo=dv_imp_lo,
        dv_hi_minus_lo=dv_imp_hi - dv_imp_lo,
        sharpen=sharpen, traj=traj,
        biphasic=dict(steps=steps_axis, dv_important=dv_traj_hi, dv_unimportant=dv_traj_lo),
    )


def aggregate(rows, n_seeds):
    def arr(k): return np.array([r[k] for r in rows], float)
    def stats(a):
        m = float(a.mean()); sd = float(a.std() + 1e-12)
        return dict(mean=m, sd=sd, t=float(m / (sd / math.sqrt(len(a)))), wins=int((a > 0).sum()))
    out = dict(
        seeds=n_seeds,
        v_grad_alive=bool((arr("v_grad_nonzero") > 0).all()),
        acc_final=float(arr("acc_final").mean()),
        # P1
        corr_dv_imp=stats(arr("corr_dv_imp")),
        corr_dv_imp_spearman=stats(arr("corr_dv_imp_spearman")),
        # P2
        shuffle_mean=float(arr("shuffle_mean").mean()),
        z_vs_shuffle=stats(arr("z_vs_shuffle")),
        shuffle_p_max=float(arr("shuffle_p").max()),
        # P3
        partial_given_len=stats(arr("partial_corr_dv_imp_given_len")),
        partial_given_len_bridge=stats(arr("partial_corr_dv_imp_given_len_bridge")),
        # confound diagnostics
        corr_dv_len=float(arr("corr_dv_len").mean()),
        corr_imp_len=float(arr("corr_imp_len").mean()),
        # P4 effect size
        dv_hi_minus_lo=stats(arr("dv_hi_minus_lo")),
    )
    # VERDICT: ADM-confirmed requires P1>0 (t>2, >=75% wins) AND beats shuffle (z>2, p small)
    # AND survives length partialling (P3 t>2).
    p1 = out["corr_dv_imp"]
    p2z = out["z_vs_shuffle"]
    p3 = out["partial_given_len"]
    adm = (p1["t"] > 2 and p1["wins"] >= max(1, int(0.75 * n_seeds))
           and p2z["mean"] > 2 and out["shuffle_p_max"] < 0.05
           and p3["t"] > 2 and p3["wins"] >= max(1, int(0.75 * n_seeds))
           and out["v_grad_alive"])
    out["verdict"] = (
        "ADM-CONFIRMED: velocity change is importance-targeted, survives shuffle + length"
        if adm else
        "PARTIAL: importance-targeting present but a control is weak"
        if (p1["t"] > 2 and p2z["mean"] > 2) else
        "NULL: velocity change is NOT importance-targeted (ADM mapping fails)"
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--device", type=str, default=pv.DEV)
    ap.add_argument("--n_shuffle", type=int, default=200)
    ap.add_argument("--imp_subsample", type=int, default=None,
                    help="ablate only this many random edges (+bridges) for importance; None=all")
    ap.add_argument("--imp_batch", type=int, default=128,
                    help="eval batch used inside the edge-importance ablation loop")
    ap.add_argument("--out", type=str, default="results/experiments/adm_selective_myelination.json")
    args = ap.parse_args()

    cfg = dict(
        N=args.N, K=3, R=0.5, gap=2.6, spread=1.0, n_bridge=4,
        v_min=0.16, v_max=3.0, v0=1.0,
        min_delay=1, max_delay=24,
        alpha=0.4, T=16, read_window=3, deadline=6,
        steps=args.steps, lr=5e-3, lr_v=0.1, l1=1e-3, lam_M=1.2e-3,
        imp_subsample=args.imp_subsample, imp_batch=args.imp_batch,
    )
    print(f"device: {args.device} | torch {torch.__version__}", flush=True)
    print(f"ADM selective-myelination test | seeds={args.seeds} steps={args.steps} "
          f"n_shuffle={args.n_shuffle}", flush=True)
    print(f"config: {cfg}", flush=True)

    rows, t0 = [], time.time()
    for s in range(args.seeds):
        print(f"=== seed {s+1}/{args.seeds} ===", flush=True)
        r = run_seed(s, cfg, args.device, n_shuffle=args.n_shuffle)
        rows.append(r)
        print(f"  acc {r['acc0']:.3f}->{r['acc_final']:.3f} | "
              f"corr(dv,imp)={r['corr_dv_imp']:+.3f} (sp {r['corr_dv_imp_spearman']:+.3f}) "
              f"shuffle={r['shuffle_mean']:+.3f} z={r['z_vs_shuffle']:.2f} p={r['shuffle_p']:.3f}",
              flush=True)
        print(f"  partial(dv,imp|len)={r['partial_corr_dv_imp_given_len']:+.3f} "
              f"partial(|len,bridge)={r['partial_corr_dv_imp_given_len_bridge']:+.3f} | "
              f"dv_hi={r['dv_imp_hi']:+.3f} dv_lo={r['dv_imp_lo']:+.3f} "
              f"(corr dv,len={r['corr_dv_len']:+.2f} imp,len={r['corr_imp_len']:+.2f})", flush=True)

    summary = aggregate(rows, args.seeds)
    summary["config"] = cfg
    summary["device"] = args.device
    summary["minutes"] = round((time.time() - t0) / 60, 2)
    summary["rows"] = rows

    print("\n=== ADM SELECTIVE-MYELINATION SUMMARY ===")
    print(f"v_grad_alive={summary['v_grad_alive']} acc_final={summary['acc_final']:.3f}")
    print(f"P1 corr(dv,imp): mean={summary['corr_dv_imp']['mean']:+.3f} "
          f"t={summary['corr_dv_imp']['t']:.2f} wins={summary['corr_dv_imp']['wins']}/{args.seeds}")
    print(f"P2 shuffle: mean_corr={summary['shuffle_mean']:+.3f} "
          f"z_vs_shuffle={summary['z_vs_shuffle']['mean']:.2f} p_max={summary['shuffle_p_max']:.3f}")
    print(f"P3 partial|len: mean={summary['partial_given_len']['mean']:+.3f} "
          f"t={summary['partial_given_len']['t']:.2f} | "
          f"partial|len,bridge mean={summary['partial_given_len_bridge']['mean']:+.3f} "
          f"t={summary['partial_given_len_bridge']['t']:.2f}")
    print(f"   confounds: corr(dv,len)={summary['corr_dv_len']:+.2f} "
          f"corr(imp,len)={summary['corr_imp_len']:+.2f}")
    print(f"P4 dv(important)-dv(unimportant): mean={summary['dv_hi_minus_lo']['mean']:+.3f} "
          f"t={summary['dv_hi_minus_lo']['t']:.2f}")
    print("VERDICT:", summary["verdict"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", args.out, "in", summary["minutes"], "min")


if __name__ == "__main__":
    main()
