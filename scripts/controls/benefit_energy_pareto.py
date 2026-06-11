"""Energy-Pareto: accuracy-per-conduction-energy tradeoff.

The established economy is matched-accuracy: at ~equal accuracy, distance-delay
nets place weight to minimize C = sum |W_ij| * tau_ij below a histogram-matched
shuffled control. Here we instead put C under an explicit budget penalty
(loss = task_loss + reg + beta*C) and sweep beta to trace an accuracy-vs-cost
Pareto frontier per condition, asking whether distance's frontier dominates
shuffled's - i.e. at matched budget C, does distance reach higher accuracy?

This is not the matched-accuracy economy restated: a real benefit needs the two
frontiers genuinely separated in the (C, accuracy) plane (a gap at fixed C beyond
seed noise), not merely shifted along the same curve. So we report the gap at
fixed budgets paired by seed, require it to exceed per-seed noise, check beta
actually moves C, and only call it a benefit if distance dominates.

The penalty uses tau_geom (true geometric delay) for BOTH conditions, since
conduction energy is a physical travel-time cost set by geometry, not by the
shuffled net's scrambled delays. This keeps C a shared currency so "matched
budget" is well defined; C with each net's own dynamic tau is recorded too.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from sdrnn.model import SDRNN, SDRNNConfig
from sdrnn.train import TrainConfig, resolve_device
from sdrnn.tasks import MemoryProTask
from sdrnn.delays import integer_delays

# Conduction-velocity / max-delay are MODULE-LEVEL but overridable from main()
# (the proper run uses a budget-stressing regime with a wider, un-clamped tau
# spread than the first run's VEL=0.08/MAXD=14, which clamped ~26% of edges at
# the ceiling and so compressed the high-tau tail distance can exploit).
VEL, MAXD = 0.08, 14


# Conduction cost C = sum_ij |W_ij| * tau_ij, differentiable in W (tau detached
# -> a weighted-L1 penalty on W).
def conduction_cost(model: SDRNN, tau: torch.Tensor) -> torch.Tensor:
    """Differentiable C = sum |W_rec| * tau (tau fixed/detached)."""
    return (model.recurrent_weight().abs() * tau).sum()


def geom_tau(model: SDRNN) -> torch.Tensor:
    """TRUE geometric integer delay matrix from the (fixed) geometry."""
    dist = model.geometry.distance_matrix().detach()
    return integer_delays(dist, VEL, MAXD).to(dist.dtype)


def train_budget(cfg: SDRNNConfig, task, tc: TrainConfig, beta: float):
    """Train one net with an explicit conduction-energy penalty beta * C.

    Returns dict with final accuracy and achieved conduction cost C (measured on
    the geometric tau), plus secondary readouts.
    """
    device = resolve_device(tc.device)
    cfg.input_size = task.input_size
    cfg.output_size = task.output_size

    torch.manual_seed(tc.seed)
    data_gen = torch.Generator().manual_seed(tc.seed + 1)
    model = SDRNN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr)

    # tau_geom is fixed (geometry is fixed): compute once, reuse as the penalty
    # weighting AND the cost metric. Same edge-by-edge tau for distance/shuffled.
    tau = geom_tau(model).to(device)

    for step in range(1, tc.steps + 1):
        model.train()
        inputs, targets, mask = task.generate(tc.batch_size, generator=data_gen)
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        outputs = model(inputs)
        task_loss = task.loss(outputs, targets, mask)
        reg = model.spatial_regularization()
        loss = task_loss + reg
        if beta > 0:
            loss = loss + beta * conduction_cost(model, tau)
        opt.zero_grad()
        loss.backward()
        if tc.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()

    # eval
    model.eval()
    gen = torch.Generator().manual_seed(12345)
    inp, tgt, msk = task.generate(tc.eval_batch, generator=gen)
    inp, tgt, msk = inp.to(device), tgt.to(device), msk.to(device)
    with torch.no_grad():
        out = model(inp)
        acc = float(task.accuracy(out, tgt, msk))
        loss_val = float(task.loss(out, tgt, msk).item())
        W = model.recurrent_weight().abs().detach().cpu().numpy()
    tau_np = tau.detach().cpu().numpy()
    C_geom = float((W * tau_np).sum())          # PRIMARY: cost on geometric tau
    Wsum = float(W.sum())
    wmean_tau = C_geom / (Wsum + 1e-9)

    return {
        "acc": acc, "loss": loss_val,
        "C": C_geom,                 # the shared physical conduction energy
        "Wsum": Wsum, "wmean_tau": wmean_tau,
    }


# Pareto-frontier analysis: at fixed budgets C, interpolate accuracy per seed.
def per_seed_curve(rows):
    """rows: list of dicts (one per beta) for ONE seed/condition.

    Returns (C_array, acc_array) sorted by ascending C, deduped, monotone-
    upper-envelope-free (raw points; the frontier is taken implicitly via
    interpolation on the achieved (C, acc) cloud).
    """
    pts = sorted([(r["C"], r["acc"]) for r in rows], key=lambda t: t[0])
    Cs = np.array([p[0] for p in pts], float)
    As = np.array([p[1] for p in pts], float)
    return Cs, As


def acc_at_budget(Cs, As, budget):
    """Accuracy achievable at conduction budget <= `budget` for one seed.

    Pareto/operational reading: with a budget you may pick ANY trained net whose
    cost C <= budget, and you would pick the most accurate such net. So the
    achievable accuracy at `budget` is max(acc over points with C <= budget).
    Returns nan if no trained point fits under the budget for this seed.
    """
    feasible = As[Cs <= budget + 1e-9]
    if feasible.size == 0:
        return float("nan")
    return float(feasible.max())


def agg(xs):
    a = np.array([x for x in xs if not (isinstance(x, float) and math.isnan(x))], float)
    if a.size == 0:
        return (float("nan"), float("nan"), 0)
    return (float(a.mean()), float(a.std()), int(a.size))


def _envelope(rows):
    """(C, acc) points -> monotone non-decreasing upper envelope, sorted by C.

    For each cost keep the BEST accuracy achievable at <= that cost (operational
    Pareto: more budget never hurts). Mirrors benefit_energy_pareto_analyze.py.
    """
    pts = sorted([(x["C"], x["acc"]) for x in rows])
    Cs, As, best = [], [], -1.0
    for c, a in pts:
        best = max(best, a)
        Cs.append(c); As.append(best)
    return np.array(Cs, float), np.array(As, float)


def _interp_env(Cs, As, grid):
    """Frontier accuracy at each grid cost. NaN below cheapest achieved point
    (never invent an infeasible cheap-but-accurate net); flat-extrapolate above."""
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        if g < Cs[0]:
            out[i] = np.nan
        elif g >= Cs[-1]:
            out[i] = As[-1]
        else:
            out[i] = np.interp(g, Cs, As)
    return out


def _fair_frontier(results, conds, n_seeds, chance):
    """Paired-per-seed matched-C frontier on a common log-spaced grid.

    Returns dict with: overlap window, a printable table at sampled grid points
    (mean/SD gap, paired wins, mean per-cond acc, whether SOLVABLE = both conds
    above chance there), and full-window / solvable-range dominance fractions.
    """
    seeds = [s for s in range(n_seeds) if results["distance"][s] and results["shuffled"][s]]
    los, his = [], []
    for s in seeds:
        dC, _ = _envelope(results["distance"][s])
        sC, _ = _envelope(results["shuffled"][s])
        los.append(max(dC[0], sC[0]))
        his.append(min(dC[-1], sC[-1]))
    lo, hi = max(los), min(his)
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), 40))

    gaps = np.full((len(seeds), len(grid)), np.nan)
    dacc = np.full((len(seeds), len(grid)), np.nan)
    sacc = np.full((len(seeds), len(grid)), np.nan)
    for si, s in enumerate(seeds):
        dC, dA = _envelope(results["distance"][s])
        sC, sA = _envelope(results["shuffled"][s])
        da = _interp_env(dC, dA, grid)
        sa = _interp_env(sC, sA, grid)
        dacc[si] = da; sacc[si] = sa
        gaps[si] = da - sa

    mean_gap = np.nanmean(gaps, axis=0)
    sd_gap = np.nanstd(gaps, axis=0)
    dist_acc = np.nanmean(dacc, axis=0)
    shuf_acc = np.nanmean(sacc, axis=0)
    wins = np.nansum(gaps > 0, axis=0)
    n_at = np.sum(~np.isnan(gaps), axis=0)
    # SOLVABLE grid point: both conditions clear chance by a margin (so a gap is
    # a real task difference, not two near-chance nets differing by noise).
    solvable = (dist_acc > chance + 0.02) & (shuf_acc > chance + 0.02)

    def _dom(mask):
        m = mean_gap[mask]
        m = m[~np.isnan(m)]
        if m.size == 0:
            return {"frac_pos": 0.0, "frac_neg": 0.0, "frac_tie": 0.0,
                    "mean_signed": float("nan"), "n_grid": 0}
        pos = float(np.mean(m > 0.005)); neg = float(np.mean(m < -0.005))
        return {"frac_pos": pos, "frac_neg": neg, "frac_tie": float(1.0 - pos - neg),
                "mean_signed": float(np.mean(m)), "n_grid": int(m.size)}

    table = []
    for gi in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]:
        table.append({
            "C": float(grid[gi]), "mean_gap": float(mean_gap[gi]),
            "sd_gap": float(sd_gap[gi]), "wins": int(wins[gi]), "n": int(n_at[gi]),
            "dist_acc": float(dist_acc[gi]), "shuf_acc": float(shuf_acc[gi]),
            "solvable": bool(solvable[gi]),
        })
    return {
        "overlap": [float(lo), float(hi)], "chance": float(chance),
        "grid_C": [float(x) for x in grid],
        "mean_gap": [float(x) for x in mean_gap],
        "sd_gap": [float(x) for x in sd_gap],
        "dist_acc": [float(x) for x in dist_acc],
        "shuf_acc": [float(x) for x in shuf_acc],
        "solvable_mask": [bool(x) for x in solvable],
        "table": table,
        "full": _dom(np.ones(len(grid), bool)),
        "solvable": _dom(solvable),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--device", default="mps")
    # DENSE beta grid (12 points) for a smooth frontier; spans degenerate (0,
    # cost ignored) to a heavy penalty that crushes C to near-zero.
    ap.add_argument("--betas",
                    default="0,1e-4,3e-4,6e-4,1e-3,1.6e-3,2.5e-3,4e-3,6e-3,9e-3,1.4e-2,2e-2",
                    help="comma list of conduction-penalty weights beta")
    ap.add_argument("--reg-lambda", type=float, default=0.01, dest="reg_lambda")
    ap.add_argument("--n-choices", type=int, default=6, dest="n_choices")
    ap.add_argument("--delay-steps", type=int, default=10, dest="delay_steps")
    ap.add_argument("--noise", type=float, default=0.4)
    # Budget-stressing geometry: lower velocity + higher max_delay => wider,
    # un-clamped geometric tau spread (the regime where distance can best save
    # energy by pruning expensive far edges). VEL=0.07/MAXD=25 => tau in [5,25],
    # 16 distinct lags, ~0% clamped (vs first run 0.08/14: 26% clamped).
    ap.add_argument("--velocity", type=float, default=0.07)
    ap.add_argument("--max-delay", type=int, default=25, dest="max_delay")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    # Override the module-level conduction geometry from CLI so the penalty tau
    # AND the model's dynamic tau both use the budget-stressing regime.
    global VEL, MAXD
    VEL, MAXD = args.velocity, args.max_delay

    torch.set_num_threads(args.threads)
    betas = [float(x) for x in args.betas.split(",")]
    conds = ["distance", "shuffled"]

    # Harder MemoryPro so the budget pressure has ROOM to trade accuracy for
    # cheaper wiring (a saturated task would flatten the frontier and hide any
    # gap). More choices + longer delay + input noise.
    task = MemoryProTask(n_choices=args.n_choices, cue_steps=2,
                         delay_steps=args.delay_steps, response_steps=2, noise=args.noise)

    print(f"ENERGY-PARETO  hidden={args.hidden} steps={args.steps} seeds={args.seeds} "
          f"betas={betas}", flush=True)
    print(f"task: MemoryPro n_choices={args.n_choices} delay={args.delay_steps} "
          f"noise={args.noise} | reg_lambda={args.reg_lambda} v={VEL} max_delay={MAXD}\n",
          flush=True)

    # results[cond][seed] = list of per-beta dicts (each carries its beta)
    results = {c: {s: [] for s in range(args.seeds)} for c in conds}

    for s in range(args.seeds):
        for c in conds:
            for beta in betas:
                cfg = SDRNNConfig(hidden_size=args.hidden, reg_mode="communicability",
                                  reg_lambda=args.reg_lambda, use_delays=True,
                                  velocity=VEL, max_delay=MAXD, delay_control=c, seed=s)
                tc = TrainConfig(steps=args.steps, batch_size=128, eval_every=args.steps,
                                 device=args.device, seed=s, log=False)
                m = train_budget(cfg, task, tc, beta)
                m["beta"] = beta
                results[c][s].append(m)
                print(f"  seed {s} {c:8s} beta={beta:<8.1e}: acc={m['acc']:.3f}  "
                      f"C={m['C']:8.1f}  wmean_tau={m['wmean_tau']:.3f}  "
                      f"Wsum={m['Wsum']:.1f}", flush=True)
            print("", flush=True)
        # save partial after each seed
        _dump(args, betas, conds, results)

    # frontier
    print("=" * 78)
    print("ACCURACY vs CONDUCTION-COST (mean over seeds, per beta):")
    print(f"{'beta':>9} | {'distance: C':>14} {'acc':>7} | {'shuffled: C':>14} {'acc':>7}")
    for bi, beta in enumerate(betas):
        dC = agg([results["distance"][s][bi]["C"] for s in range(args.seeds)])
        dA = agg([results["distance"][s][bi]["acc"] for s in range(args.seeds)])
        sC = agg([results["shuffled"][s][bi]["C"] for s in range(args.seeds)])
        sA = agg([results["shuffled"][s][bi]["acc"] for s in range(args.seeds)])
        print(f"{beta:>9.1e} | {dC[0]:>9.1f}+/-{dC[1]:<4.0f} {dA[0]:>6.3f} | "
              f"{sC[0]:>9.1f}+/-{sC[1]:<4.0f} {sA[0]:>6.3f}", flush=True)
    print("=" * 78)

    # sanity: does beta actually move C? (a degenerate sweep would invalidate it)
    allC = [results[c][s][bi]["C"] for c in conds for s in range(args.seeds)
            for bi in range(len(betas))]
    C_lo, C_hi = float(min(allC)), float(max(allC))
    print(f"C range across whole sweep: [{C_lo:.1f}, {C_hi:.1f}]  "
          f"(ratio {C_hi / max(C_lo,1e-9):.2f}x)  -> "
          f"{'REAL sweep' if C_hi > 1.5 * C_lo else 'DEGENERATE (beta barely moves C)'}",
          flush=True)

    # accuracy gap at fixed budgets. Log-space budgets across the overlap window so
    # each sits in the populated part of the frontier (C spans orders of magnitude;
    # linear spacing piles them in the high-C tail where only beta=0 lives). Overlap
    # window: max over seeds/conds of min-C up to min over seeds/conds of max-C.
    min_reach = max(min(r["C"] for r in results[c][s])
                    for c in conds for s in range(args.seeds))
    max_reach = min(max(r["C"] for r in results[c][s])
                    for c in conds for s in range(args.seeds))
    lo_l, hi_l = math.log(max(min_reach, 1e-6)), math.log(max(max_reach, 1e-6))
    budgets = [math.exp(lo_l + f * (hi_l - lo_l)) for f in (0.25, 0.5, 0.75)]

    print("\nACCURACY GAP AT MATCHED CONDUCTION BUDGET  (paired per seed)")
    print(f"overlap window of achievable C: [{min_reach:.1f}, {max_reach:.1f}]")
    print(f"{'budget C':>10} | {'dist acc':>16} | {'shuf acc':>16} | "
          f"{'gap(d-s)':>16} | wins")
    gap_report = []
    for budget in budgets:
        d_per_seed, s_per_seed, diffs = [], [], []
        for seed in range(args.seeds):
            dC, dA = per_seed_curve(results["distance"][seed])
            sC, sA = per_seed_curve(results["shuffled"][seed])
            da = acc_at_budget(dC, dA, budget)
            sa = acc_at_budget(sC, sA, budget)
            d_per_seed.append(da)
            s_per_seed.append(sa)
            if not (math.isnan(da) or math.isnan(sa)):
                diffs.append(da - sa)
        dm = agg(d_per_seed); sm = agg(s_per_seed); gm = agg(diffs)
        wins = sum(1 for x in diffs if x > 0)
        losses = sum(1 for x in diffs if x < 0)
        gap_report.append({
            "budget": budget, "dist_acc": dm, "shuf_acc": sm, "gap": gm,
            "diffs": diffs, "wins": wins, "losses": losses, "n": len(diffs),
        })
        print(f"{budget:>10.1f} | {dm[0]:>7.3f}+/-{dm[1]:<6.3f} | "
              f"{sm[0]:>7.3f}+/-{sm[1]:<6.3f} | {gm[0]:>+7.3f}+/-{gm[1]:<6.3f} | "
              f"{wins}/{len(diffs)}", flush=True)

    # matched-beta paired gap (interpolation-free). At a fixed beta both conditions
    # feel the same penalty and converge to nearly identical C (per-beta C differs by
    # << its spread), so the per-beta accuracy difference is a matched-budget gap with
    # no interpolation - paired by seed. We record the per-beta C separation to prove
    # the budgets really are matched (small |C_dist - C_shuf| relative to C).
    print("\nMATCHED-beta PAIRED accuracy gap (no interpolation; paired by seed)")
    print(f"{'beta':>9} | {'C(d)':>8} {'C(s)':>8} {'|dC|/C':>7} | "
          f"{'gap(d-s) acc':>18} | wins")
    beta_gap_report = []
    for bi, beta in enumerate(betas):
        diffs = [results["distance"][s][bi]["acc"] - results["shuffled"][s][bi]["acc"]
                 for s in range(args.seeds)]
        Cd = np.mean([results["distance"][s][bi]["C"] for s in range(args.seeds)])
        Cs = np.mean([results["shuffled"][s][bi]["C"] for s in range(args.seeds)])
        gm = agg(diffs)
        wins = sum(1 for x in diffs if x > 0)
        rel_sep = abs(Cd - Cs) / (0.5 * (Cd + Cs) + 1e-9)
        beta_gap_report.append({
            "beta": beta, "C_dist": float(Cd), "C_shuf": float(Cs),
            "rel_C_sep": float(rel_sep), "gap": gm, "diffs": diffs,
            "wins": wins, "n": len(diffs),
        })
        print(f"{beta:>9.1e} | {Cd:>8.1f} {Cs:>8.1f} {rel_sep:>6.2%} | "
              f"{gm[0]:>+8.3f}+/-{gm[1]:<7.3f} | {wins}/{len(diffs)}", flush=True)

    # Fair frontier (the decisive part). Per seed, build the accuracy-vs-cost
    # monotone upper envelope (more budget never hurts), interpolate onto a common
    # dense log-C grid over the per-seed overlap window, and pair-difference
    # distance - shuffled at each grid cost.
    # This removes the beta-grid artifact (same beta -> different C per condition)
    # and gives an honest matched-C gap with per-seed error bars. We report the
    # FULL frontier (including crossings) and separately the SOLVABLE range, where
    # "solvable" = both conditions are above chance (1/n_choices) so a gap is
    # task-meaningful rather than two near-chance nets differing by noise.
    chance = 1.0 / args.n_choices
    fair = _fair_frontier(results, conds, args.seeds, chance)
    print("\n" + "=" * 78)
    print("FAIR FRONTIER (paired per seed, common matched-C grid)")
    print(f"overlap window C: [{fair['overlap'][0]:.1f}, {fair['overlap'][1]:.1f}]"
          f"  chance={chance:.3f}")
    print(f"{'C':>9} | {'mean gap':>9} {'SD':>7} | {'wins/n':>7} | "
          f"{'dist acc':>8} {'shuf acc':>8} | solvable")
    for row in fair["table"]:
        print(f"{row['C']:>9.1f} | {row['mean_gap']:>+9.3f} {row['sd_gap']:>7.3f} | "
              f"{row['wins']:>3}/{row['n']:<3} | {row['dist_acc']:>8.3f} "
              f"{row['shuf_acc']:>8.3f} | {str(row['solvable'])}", flush=True)
    print(f"\nover FULL overlap window:    dist-above {fair['full']['frac_pos']:.0%}"
          f"  shuf-above {fair['full']['frac_neg']:.0%}  tied {fair['full']['frac_tie']:.0%}"
          f"  mean signed gap {fair['full']['mean_signed']:+.4f}")
    sv = fair["solvable"]
    if sv["n_grid"] > 0:
        print(f"over SOLVABLE (above-chance) range: dist-above {sv['frac_pos']:.0%}"
              f"  shuf-above {sv['frac_neg']:.0%}  tied {sv['frac_tie']:.0%}"
              f"  mean signed gap {sv['mean_signed']:+.4f}  ({sv['n_grid']} grid pts)")
    else:
        print("SOLVABLE range: EMPTY (no matched-C grid point where both conds beat chance)")

    # Verdict. Real benefit requires a non-degenerate sweep AND, at >=1 operating
    # point, a positive mean gap beyond per-seed noise (gap > 1 SD of the paired
    # diffs AND > 0.01) AND a consistent paired win count. Evidence from either the
    # interpolated fixed-budget frontier or the matched-beta comparison (headline).
    real_sweep = C_hi > 1.5 * C_lo

    def _passes(g, require_matched=False):
        mean_gap, sd_gap, n = g["gap"]
        if n < 2 or math.isnan(mean_gap):
            return False
        beats_noise = mean_gap > sd_gap and mean_gap > 0.01
        consistent = g["wins"] >= max(2, int(0.75 * n))
        budget_matched = (not require_matched) or g.get("rel_C_sep", 0.0) < 0.20
        return beats_noise and consistent and budget_matched

    benefit_budgets = [g["budget"] for g in gap_report if _passes(g)]
    # matched-beta points that pass AND are genuinely budget-matched (small rel sep)
    benefit_betas = [g["beta"] for g in beta_gap_report if _passes(g, require_matched=True)]

    # HEADLINE criterion (fair frontier over the SOLVABLE range). A CLEAN energy
    # benefit = distance strictly above across most of the above-chance range with
    # little/no reverse crossing, and a gap beyond per-seed noise somewhere there.
    # Anything weaker (mid-band-only edge, or interleaving frontiers) is MARGINAL;
    # frontiers on top of each other / shuffled-above is NULL.
    sv = fair["solvable"]
    any_beyond_noise = any(
        (not math.isnan(r["mean_gap"])) and r["mean_gap"] > r["sd_gap"]
        and r["mean_gap"] > 0.01 and r["solvable"]
        for r in fair["table"]
    )
    clean_benefit = (
        real_sweep and sv["n_grid"] >= 3
        and sv["frac_pos"] >= 0.80 and sv["frac_neg"] <= 0.10
        and sv["mean_signed"] > 0.01 and any_beyond_noise
    )
    marginal = (
        real_sweep and not clean_benefit
        and (any_beyond_noise or (sv["n_grid"] > 0 and sv["mean_signed"] > 0.005))
    )
    if clean_benefit:
        label = "CLEAN-BENEFIT"
    elif marginal:
        label = "MARGINAL"
    else:
        label = "NULL"
    distance_dominates = clean_benefit  # keep the old key honest

    print("\n" + "=" * 78)
    print("VERDICT")
    print(f"  real (non-degenerate) cost sweep:        {real_sweep}")
    print(f"  interp. budgets with distance>shuffled beyond noise: "
          f"{len(benefit_budgets)}/{len(gap_report)}")
    print(f"  matched-beta points with distance>shuffled beyond noise: "
          f"{len(benefit_betas)}/{len(beta_gap_report)}")
    print(f"  FAIR solvable range: dist-above {sv['frac_pos']:.0%}, "
          f"shuf-above {sv['frac_neg']:.0%}, mean gap {sv['mean_signed']:+.4f}, "
          f"beyond-noise pt: {any_beyond_noise}")
    print(f"  >>> HONEST VERDICT: {label} <<<")
    print("=" * 78, flush=True)

    out = {
        "args": vars(args), "betas": betas, "conds": conds, "VEL": VEL, "MAXD": MAXD,
        "results": results,
        "C_range": [C_lo, C_hi], "real_sweep": real_sweep,
        "overlap_window": [min_reach, max_reach],
        "gap_report": gap_report,
        "beta_gap_report": beta_gap_report,
        "fair_frontier": fair,
        "verdict": {
            "real_sweep": real_sweep,
            "benefit_budgets": benefit_budgets,
            "benefit_betas": benefit_betas,
            "clean_benefit": clean_benefit,
            "marginal": marginal,
            "any_beyond_noise": any_beyond_noise,
            "solvable_frac_pos": sv["frac_pos"],
            "solvable_frac_neg": sv["frac_neg"],
            "solvable_mean_signed_gap": sv["mean_signed"],
            "distance_dominates": distance_dominates,
            "label": label,
        },
    }
    _dump(args, betas, conds, results, final=out)
    print(f"wrote results/controls/{'energy_pareto.json' if not args.tag else f'energy_pareto_{args.tag}.json'}",
          flush=True)


def _dump(args, betas, conds, results, final=None):
    outdir = ROOT / "results" / "controls"
    outdir.mkdir(parents=True, exist_ok=True)
    name = "energy_pareto.json" if not args.tag else f"energy_pareto_{args.tag}.json"
    payload = final if final is not None else {
        "args": vars(args), "betas": betas, "conds": conds,
        "results": results, "partial": True,
    }
    (outdir / name).write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
