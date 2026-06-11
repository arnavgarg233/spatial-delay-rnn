"""PINN velocity inverse on natural task activity, beyond the partly-tautological
impulse version (which used clean impulses + the known kernel W, so residual at
v_true was ~0 by construction).

Here activity is natural (hidden states recorded while the trained net performs
MemoryProTask) and we report three pieces of evidence, ranked by how tautological:

  (A) Localization with known W: scan v, residual(v). v_hat~v_true is partly by
      construction; the informative part is whether the dip is sharp near v_true
      despite the strong autocorrelation of natural activity.
  (B) Shuffled-null floor (the distance-specific test): velocity-shuffled geometry
      (same delay histogram, scrambled pair->lag), null refits a free kernel at
      every v (its best shot). Claim: the shuffled geometry cannot reach the
      true-geometry residual floor. No clear gap => the result evaporates.
  (C) Low-rank kernel recovery (de-tautologized): without W, fit a rank-r kernel
      per v from lagged rates, score residual, and at v_hat measure corr(W_r,
      W_true). Reports identifiability honestly (flat residual under a free kernel
      = collinear lagged rates = a publishable negative).

  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=2 \
    python scripts/inverse/pinn_inverse.py --device mps
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from sdrnn.model import SDRNNConfig
from sdrnn.tasks import MemoryProTask
from sdrnn.train import TrainConfig, resolve_device, train


# Record natural task-driven activity. We replay the forward pass by hand (the
# public forward only returns rates) to capture the full state trajectory needed
# to invert the leaky update, driving with real MemoryProTask trials.
@torch.no_grad()
def record_task_activity(model, task, device, n_trials=64, n_repeats=6, seed=0):
    """Run the trained net on real task trials; return state-space traces.

    Returns rates,states,proj (each (B,T,N)) plus alpha, W, bias. We tile a few
    independent batches of task trials end-to-end in the batch dim to get enough
    samples (T is short for MemoryPro, so batch carries the statistics).
    """
    from sdrnn.delays import DelayBuffer, integer_delays

    model.eval()
    cfg = model.config
    n = cfg.hidden_size
    alpha = cfg.alpha
    w_rec = model.recurrent_weight()
    bias = model.recurrent.bias

    velocity = model.current_velocity()
    dist = model.geometry.distance_matrix()
    tau = integer_delays(dist, float(velocity.detach().item()), cfg.max_delay, cfg.min_delay)
    tau = model._apply_delay_control(tau)

    all_rates, all_states, all_proj = [], [], []
    gen = torch.Generator().manual_seed(seed + 4242)
    for rep in range(n_repeats):
        inputs, _, _ = task.generate(n_trials, generator=gen)
        x = inputs.to(device)
        B, T, _ = x.shape
        proj = model.input_layer(x)  # (B,T,N)
        buf = DelayBuffer(cfg.max_delay, B, n, device, w_rec.dtype)
        state = torch.zeros(B, n, device=device, dtype=w_rec.dtype)
        states, rates = [], []
        for t in range(T):
            rate = torch.relu(state)
            buf.push(rate)
            rec = buf.apply_gather(w_rec, tau)
            pre = proj[:, t] + rec + bias
            state = (1 - alpha) * state + alpha * pre
            rates.append(torch.relu(state))
            states.append(state)
        all_rates.append(torch.stack(rates, dim=1))
        all_states.append(torch.stack(states, dim=1))
        all_proj.append(proj)
    rates = torch.cat(all_rates, dim=0)
    states = torch.cat(all_states, dim=0)
    proj = torch.cat(all_proj, dim=0)
    return (rates.cpu().numpy(), states.cpu().numpy(), proj.cpu().numpy(),
            float(alpha), w_rec.detach().cpu().numpy().astype(np.float64),
            bias.detach().cpu().numpy().astype(np.float64))


def reconstruct_rec_target(states, proj, alpha, bias):
    """Exact recurrent drive rec_t = (state_t-(1-a)state_{t-1})/a - proj_t - bias.

    Returns (B, T-1, N) aligned to t=1..T-1. This inversion is a *measurement*
    using the known leaky update; it is fair (we always know our own integrator).
    """
    s_t = states[:, 1:, :]
    s_prev = states[:, :-1, :]
    rec = (s_t - (1 - alpha) * s_prev) / alpha - proj[:, 1:, :] - bias[None, None, :]
    return rec  # (B, T-1, N)


# Lagged-rate machinery (shared by known-W scoring and free-kernel fitting).
def lagged_rate(rates, d):
    """rate_j(t - d) aligned to t=1..T-1, shape (B*(T-1), N)."""
    B, T, N = rates.shape
    Ld = np.zeros_like(rates)
    if d < T:
        Ld[:, d:, :] = rates[:, : T - d, :]
    return Ld[:, 1:, :].reshape(B * (T - 1), N)


def predict_rec_knownW(rates, tau, W, lag_cache):
    """rec_hat_i(t) = sum_j W_ij rate_j(t-tau_ij) with W fixed. Shape (M,N)."""
    B, T, N = rates.shape
    M = B * (T - 1)
    pred = np.zeros((M, N), dtype=np.float64)
    for d in np.unique(tau):
        d = int(d)
        if d not in lag_cache:
            lag_cache[d] = lagged_rate(rates, d)
        Ld = lag_cache[d]
        mask = (tau == d).astype(np.float64)
        pred += Ld @ (W * mask).T
    return pred


def normalized_residual(Y, pred):
    num = np.sum((Y - pred) ** 2)
    den = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2) + 1e-12
    return float(num / den)


def scan_knownW(rates, rec_target, dist, W, cands, max_delay, min_delay):
    """Known-kernel residual(v) on a given geometry."""
    B, T, N = rates.shape
    Y = rec_target.reshape(B * (T - 1), N)
    lag_cache = {}
    res = []
    for c in cands:
        tau = np.clip(np.round(dist / c), min_delay, max_delay).astype(int)
        res.append(normalized_residual(Y, predict_rec_knownW(rates, tau, W, lag_cache)))
    return np.array(res)


# Free/low-rank kernel fit: per candidate v, fit a kernel explaining rec_target
# from lagged rates without being told W (the genuine inverse, and the null's best
# shot). Per target unit i, design column j = rate_j(t-tau_ij), ridge-fit w_i;
# optionally truncate the stacked solution to rank r via SVD and re-score.
def fit_kernel_at_velocity(rates, rec_target, tau, ridge, rank=None):
    """Fit (optionally rank-r) kernel; return (residual, W_fit (N,N) or None).

    Full fit residual measures identifiability of v under a free kernel.
    If rank is given, also truncate W_fit to rank r and report that residual +
    the truncated kernel (for corr-with-true-W).
    """
    B, T, N = rates.shape
    Y = rec_target.reshape(B * (T - 1), N)
    # cache lagged rates per distinct delay
    lag_cache = {}
    for d in np.unique(tau):
        d = int(d)
        lag_cache[d] = lagged_rate(rates, d)
    Wfit = np.zeros((N, N), dtype=np.float64)
    # ridge scale from feature energy (per-unit design has same columns set)
    for i in range(N):
        Xi = np.empty((Y.shape[0], N), dtype=np.float64)
        for j in range(N):
            Xi[:, j] = lag_cache[int(tau[i, j])][:, j]
        yi = Y[:, i]
        XtX = Xi.T @ Xi
        lam = ridge * (np.trace(XtX) / N + 1e-8)
        XtX[np.diag_indices_from(XtX)] += lam
        Wfit[i] = np.linalg.solve(XtX, Xi.T @ yi)
    # full-rank residual
    res_full = _kernel_residual(rates, Y, tau, Wfit, lag_cache)
    out = {"res_full": res_full, "W_full": Wfit}
    if rank is not None:
        U, S, Vt = np.linalg.svd(Wfit, full_matrices=False)
        r = min(rank, len(S))
        W_lr = (U[:, :r] * S[:r]) @ Vt[:r]
        res_lr = _kernel_residual(rates, Y, tau, W_lr, lag_cache)
        out["res_lr"] = res_lr
        out["W_lr"] = W_lr
        out["sv"] = S.tolist()
    return out


def _kernel_residual(rates, Y, tau, W, lag_cache):
    B, T, N = rates.shape
    M = Y.shape[0]
    pred = np.zeros((M, N), dtype=np.float64)
    for d in np.unique(tau):
        d = int(d)
        Ld = lag_cache[int(d)]
        mask = (tau == d).astype(np.float64)
        pred += Ld @ (W * mask).T
    return normalized_residual(Y, pred)


def scan_freekernel(rates, rec_target, dist, cands, max_delay, min_delay, ridge, rank):
    """Free/low-rank kernel residual(v) on a geometry; return arrays + extras."""
    res_full, res_lr, kern_corr_full = [], [], []
    W_at_best = None
    for c in cands:
        tau = np.clip(np.round(dist / c), min_delay, max_delay).astype(int)
        fk = fit_kernel_at_velocity(rates, rec_target, tau, ridge, rank=rank)
        res_full.append(fk["res_full"])
        res_lr.append(fk.get("res_lr", np.nan))
    return np.array(res_full), np.array(res_lr)


def shuffle_geometry(dist, seed):
    """Velocity-shuffled null: permute off-diagonal distances (same histogram)."""
    N = dist.shape[0]
    rng = np.random.default_rng(seed + 777)
    iu = np.triu_indices(N, 1)
    d = dist.copy()
    vals = d[iu].copy()
    rng.shuffle(vals)
    d[iu] = vals
    d[(iu[1], iu[0])] = vals
    return d


def corr(a, b):
    a = a.ravel(); b = b.ravel()
    a = a - a.mean(); b = b - b.mean()
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float((a @ b) / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--v-true", type=float, default=0.08, dest="v_true")
    ap.add_argument("--max-delay", type=int, default=14, dest="max_delay")
    ap.add_argument("--n-trials", type=int, default=64, dest="n_trials")
    ap.add_argument("--n-repeats", type=int, default=6, dest="n_repeats")
    ap.add_argument("--n-cands", type=int, default=15, dest="n_cands")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--rank", type=int, default=8, dest="rank")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # MemoryProTask: longer delay so there is real temporal structure to time.
    task = MemoryProTask(n_choices=4, cue_steps=2, delay_steps=8, response_steps=2, noise=0.2)
    device = resolve_device(args.device)
    cands = np.geomspace(args.v_true / 3.5, args.v_true * 3.5, args.n_cands)

    print(f"PUSH-PINN  v_true={args.v_true}  hidden={args.hidden}  steps={args.steps}  "
          f"seeds={args.seeds}  device={device}  rank={args.rank}")
    print(f"task=MemoryPro(delay=8) NATURAL activity  candidates={np.round(cands,4).tolist()}\n",
          flush=True)

    accs = []
    per_seed = []
    # aggregates
    vhat_known, vhat_free = [], []
    floor_true, floor_shuf_known, floor_shuf_free = [], [], []
    dip_known_true, dip_known_shuf = [], []
    vhat_free_shuf = []
    kern_corr_lr = []
    interior_free, ratio_floor_list = [], []

    for seed in range(args.seeds):
        cfg = SDRNNConfig(hidden_size=args.hidden, reg_mode="communicability", reg_lambda=0.01,
                          use_delays=True, velocity=args.v_true, max_delay=args.max_delay,
                          delay_control="distance", seed=seed)
        tc = TrainConfig(steps=args.steps, batch_size=128, eval_every=args.steps,
                         device=args.device, seed=seed, log=False)
        model, result = train(cfg, task, tc)
        accs.append(result.final_accuracy)

        rates, states, proj, alpha, W, bias = record_task_activity(
            model, task, device, n_trials=args.n_trials, n_repeats=args.n_repeats, seed=seed)
        rec_target = reconstruct_rec_target(states, proj, alpha, bias)
        dist = model.geometry.distance_matrix().detach().cpu().numpy()
        d_shuf = shuffle_geometry(dist, seed)

        # (A) known-W localization on true geometry, natural activity
        res_kt = scan_knownW(rates, rec_target, dist, W, cands, args.max_delay, cfg.min_delay)
        v_kt = float(cands[int(np.argmin(res_kt))])
        vhat_known.append(v_kt)
        floor_true.append(float(res_kt.min()))
        dip_known_true.append(float((res_kt.max() - res_kt.min()) / (res_kt.max() + 1e-12)))

        # (B) shuffled-null floor: known-W and free-kernel (best case)
        res_ks = scan_knownW(rates, rec_target, d_shuf, W, cands, args.max_delay, cfg.min_delay)
        floor_shuf_known.append(float(res_ks.min()))
        dip_known_shuf.append(float((res_ks.max() - res_ks.min()) / (res_ks.max() + 1e-12)))

        # free-kernel on shuffled geometry: the null's BEST shot at any velocity
        res_fs_full, _ = scan_freekernel(
            rates, rec_target, d_shuf, cands, args.max_delay, cfg.min_delay, args.ridge, args.rank)
        floor_shuf_free.append(float(res_fs_full.min()))
        v_fs = float(cands[int(np.argmin(res_fs_full))])
        vhat_free_shuf.append(v_fs)

        # (C) low-rank / free kernel recovery on true geometry
        res_ft_full, res_ft_lr = scan_freekernel(
            rates, rec_target, dist, cands, args.max_delay, cfg.min_delay, args.ridge, args.rank)
        argmin_ft = int(np.argmin(res_ft_full))
        v_ft = float(cands[argmin_ft])
        # HONESTY: is the free-kernel minimum INTERIOR (bracketed below+above) or
        # at the high-velocity boundary? A boundary min => v unbounded from above
        # (high-v delays collapse to lag=1, a free kernel fits the AR(1) drive).
        interior_ft = 0 < argmin_ft < len(cands) - 1
        # ratio of shuffled free floor to true free floor (distance-specificity)
        ratio_floor_free = float(res_fs_full.min() / (res_ft_full.min() + 1e-12))
        vhat_free.append(v_ft)
        # refit the kernel AT v_hat_free to measure corr with true W (rank-r)
        tau_at = np.clip(np.round(dist / v_ft), cfg.min_delay, args.max_delay).astype(int)
        fk = fit_kernel_at_velocity(rates, rec_target, tau_at, args.ridge, rank=args.rank)
        kc_lr = corr(fk["W_lr"], W)
        kc_full = corr(fk["W_full"], W)
        kern_corr_lr.append(kc_lr)
        interior_free.append(interior_ft)
        ratio_floor_list.append(ratio_floor_free)

        per_seed.append({
            "seed": seed, "acc": result.final_accuracy,
            "v_known_true": v_kt, "ratio_known_true": v_kt / args.v_true,
            "floor_true": floor_true[-1], "dip_known_true": dip_known_true[-1],
            "floor_shuf_known": floor_shuf_known[-1], "dip_known_shuf": dip_known_shuf[-1],
            "floor_shuf_free": floor_shuf_free[-1], "v_free_shuf": v_fs,
            "v_free_true": v_ft, "ratio_free_true": v_ft / args.v_true,
            "floor_free_true_full": float(res_ft_full.min()),
            "floor_free_true_lr": float(np.nanmin(res_ft_lr)),
            "interior_min_free": bool(interior_ft),
            "ratio_floor_free_shuf_over_true": ratio_floor_free,
            "kern_corr_lr": kc_lr, "kern_corr_full": kc_full,
            "res_known_true": res_kt.tolist(), "res_known_shuf": res_ks.tolist(),
            "res_free_true_full": res_ft_full.tolist(), "res_free_true_lr": res_ft_lr.tolist(),
            "res_free_shuf_full": res_fs_full.tolist(),
        })
        print(f"  seed {seed}: acc={result.final_accuracy:.3f}", flush=True)
        print(f"    (A) known-W  TRUE: v_hat={v_kt:.4f} (ratio {v_kt/args.v_true:.2f})  "
              f"floor={floor_true[-1]:.4f}  dip={dip_known_true[-1]:.3f}", flush=True)
        print(f"    (B) shuffled floor: known-W={floor_shuf_known[-1]:.4f} (dip {dip_known_shuf[-1]:.3f})  "
              f"free-kernel(best)={floor_shuf_free[-1]:.4f} @v={v_fs:.4f}", flush=True)
        print(f"    (C) free-kernel TRUE: v_hat={v_ft:.4f} (ratio {v_ft/args.v_true:.2f})  "
              f"floor_full={res_ft_full.min():.4f} floor_lr={np.nanmin(res_ft_lr):.4f}  "
              f"corr(W_lr,W_true)={kc_lr:.3f} corr(W_full,W_true)={kc_full:.3f}", flush=True)

    # aggregate
    vk = np.array(vhat_known); vf = np.array(vhat_free)
    err_known = np.abs(vk / args.v_true - 1)
    err_free = np.abs(vf / args.v_true - 1)
    ft = np.array(floor_true); fsk = np.array(floor_shuf_known); fsf = np.array(floor_shuf_free)
    # distance-specific gap: how much LOWER the true-geom floor is than the
    # shuffled-geom floor (free kernel = null's best shot). >0 and clear = real.
    gap_known = fsk - ft
    gap_free = fsf - ft

    print("\n" + "=" * 74)
    print(f"matched accuracy:                {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print("-- (A) LOCALIZATION (known W, natural task activity) - partly by construction --")
    print(f"  v_hat known-W TRUE:            {vk.mean():.4f} ± {vk.std():.4f}  (true {args.v_true})  "
          f"err {err_known.mean()*100:.1f}%  ratios {np.round(vk/args.v_true,2).tolist()}")
    print(f"  residual dip (true vs shuf):   {np.mean(dip_known_true):.3f} ± {np.std(dip_known_true):.3f}  "
          f"| shuf {np.mean(dip_known_shuf):.3f} ± {np.std(dip_known_shuf):.3f}")
    print("-- (B) SHUFFLED-NULL FLOOR (the distance-specific test) --")
    print(f"  min residual TRUE geometry:    {ft.mean():.4f} ± {ft.std():.4f}")
    print(f"  min residual SHUF (known W):   {fsk.mean():.4f} ± {fsk.std():.4f}   gap {gap_known.mean():.4f}")
    print(f"  min residual SHUF (free kern): {fsf.mean():.4f} ± {fsf.std():.4f}   gap {gap_free.mean():.4f}  <-- null best shot")
    print("-- (C) FREE / LOW-RANK KERNEL RECOVERY (de-tautologized inverse) --")
    print(f"  v_hat free-kernel TRUE:        {vf.mean():.4f} ± {vf.std():.4f}  err {err_free.mean()*100:.1f}%  "
          f"ratios {np.round(vf/args.v_true,2).tolist()}")
    print(f"  free-kernel min INTERIOR (bracketed, not high-v boundary): "
          f"{sum(interior_free)}/{len(interior_free)} seeds")
    print(f"  floor ratio shuf/true (free kernel, distance-specificity): "
          f"{np.mean(ratio_floor_list):.1f}x ± {np.std(ratio_floor_list):.1f}")
    print(f"  corr(recovered rank-{args.rank} W, true W): {np.mean(kern_corr_lr):.3f} ± {np.std(kern_corr_lr):.3f}")
    print("=" * 74)

    # verdicts (demand clear distance-vs-shuffled separation)
    loc_ok = err_known.mean() < 0.35
    # the FREE-kernel null is the toughest: require true floor clearly below it.
    null_beat = (gap_free.mean() > 0) and np.all(fsf > ft)
    null_beat_clear = gap_free.mean() > 0.5 * ft.mean() and np.all(gap_free > 0)
    free_ident = err_free.mean() < 0.5
    kern_ok = np.mean(kern_corr_lr) > 0.3

    print("HONEST READOUT:")
    print(f"  [A] known-W localizes near v_true: {'YES' if loc_ok else 'NO'} "
          f"(PARTLY BY CONSTRUCTION - known operator; the informative part is the dip shape).")
    print(f"  [B] true-geom floor below SHUFFLED free-kernel floor on every seed: "
          f"{'YES' if null_beat else 'NO'}  (clear separation: {'YES' if null_beat_clear else 'NO'}).")
    print(f"      -> this is the distance-specific, NON-tautological evidence. gap_free="
          f"{gap_free.mean():.4f} vs true floor {ft.mean():.4f}.")
    n_int = sum(interior_free)
    print(f"  [C] free-kernel (no W given) still identifies v within 50%: {'YES' if free_ident else 'NO'}; "
          f"rank-{args.rank} kernel corr>0.3: {'YES' if kern_ok else 'NO'} "
          f"(corr={np.mean(kern_corr_lr):.3f}).")
    print(f"      CAVEAT: free-kernel min is interior (truly bracketed, not just high-v boundary) on "
          f"{n_int}/{len(interior_free)} seeds. A boundary min means v is identified from BELOW "
          f"(long delays rejected) but NOT bounded above - honest partial identifiability.")

    if null_beat_clear and loc_ok:
        verdict = ("STRENGTHENED: on NATURAL task activity the true-geometry residual floor is clearly "
                   "below the velocity-shuffled free-kernel floor on all seeds (distance-specific, not "
                   "tautological). v_hat~v_true under known-W is partly by construction but the dip is "
                   "sharp on natural activity. Kernel identifiability is the honest caveat (see [C]).")
    elif null_beat:
        verdict = ("PARTIAL: true floor below shuffled free-kernel floor every seed but the gap is "
                   "small - distance-specificity is present but not dramatic on natural activity. "
                   "Report the gap honestly; this is weaker than the impulse version's clean dip.")
    else:
        verdict = ("EVAPORATED: the velocity-shuffled free-kernel null reaches the true-geometry "
                   "residual floor - the inverse is NOT distance-specific on natural activity once the "
                   "null is allowed to refit its kernel. The earlier 0%-error was construction, not "
                   "identifiability. Honest negative.")
    print("\nVERDICT:", verdict)

    out = {
        "v_true": args.v_true, "hidden": args.hidden, "steps": args.steps,
        "task": "MemoryPro(delay=8) natural activity", "rank": args.rank,
        "candidates": cands.tolist(),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        # (A)
        "vhat_known_mean": float(vk.mean()), "vhat_known_std": float(vk.std()),
        "err_known_pct": float(err_known.mean() * 100),
        "dip_known_true_mean": float(np.mean(dip_known_true)),
        "dip_known_shuf_mean": float(np.mean(dip_known_shuf)),
        # (B)
        "floor_true_mean": float(ft.mean()), "floor_true_std": float(ft.std()),
        "floor_shuf_known_mean": float(fsk.mean()),
        "floor_shuf_free_mean": float(fsf.mean()),
        "gap_free_mean": float(gap_free.mean()), "gap_free_min": float(gap_free.min()),
        "null_beat": bool(null_beat), "null_beat_clear": bool(null_beat_clear),
        # (C)
        "vhat_free_mean": float(vf.mean()), "vhat_free_std": float(vf.std()),
        "err_free_pct": float(err_free.mean() * 100),
        "kern_corr_lr_mean": float(np.mean(kern_corr_lr)), "kern_corr_lr_std": float(np.std(kern_corr_lr)),
        "free_ident": bool(free_ident), "kern_ok": bool(kern_ok),
        "free_interior_min_seeds": int(sum(interior_free)),
        "floor_ratio_shuf_over_true_free_mean": float(np.mean(ratio_floor_list)),
        "loc_ok": bool(loc_ok),
        "verdict": verdict, "per_seed": per_seed,
    }
    outpath = ROOT / "results" / "inverse" / "pinn_inverse.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
