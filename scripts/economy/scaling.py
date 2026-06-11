"""Conduction-economy saving vs network size N, at matched dose and accuracy.

Earlier runs only measured the saving at a single N and under a fixed velocity --
but coordinates are scaled to ~unit nearest-neighbour spacing, so grid diameter
(and thus the delay dose) drifts with N, confounding any naive saving-vs-N read.

This sweep controls for that:
  * N in {48, 96, 144}, 4 seeds each.
  * Matched dose: velocity = diameter(N) / target_max_lag, so the longest path is
    ~target_max_lag steps at every N. max_delay has headroom so the clip is never
    binding (saturation reported per cell).
  * Matched density: fixed average degree across N.
  * Metric: tau_bar = sum|W|*tau/sum|W|, scored against the physical distance delay
    field for both conditions (real conduction time, independent of training delays).
      SAVING     = tau_bar(shuffled) - tau_bar(distance)   (>0 => distance economical)
      REL_SAVING = SAVING / tau_bar(shuffled)
  * Accuracy gate: only cells with |acc_dist - acc_shuf| <= 0.03 enter the fit.

H1: paired SAVING > 0 at each N, matched accuracy. H2: REL_SAVING ~ a + b*log10(N)
(b>0 strengthens, b~0 scale-invariant, b<0 washes out; direction reported).

  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=2 \
  python scripts/economy/scaling.py --device mps
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))

import argparse
import json
import time

import numpy as np
import torch

from sdrnn.geometry import grid_coordinates
from sdrnn.model import SDRNNConfig
from sdrnn.tasks import MemoryProTask
from sdrnn.train import TrainConfig, train
from sdrnn.delays import integer_delays


# size-invariant matching (mirrors the scaling_sweep helpers)
def matched_delay_params(size, dim, target_max_lag):
    """velocity s.t. the longest path is ~target_max_lag steps at this N; headroom on clip."""
    coords = grid_coordinates(size, dim)
    diameter = float(torch.cdist(coords, coords).max())
    velocity = diameter / target_max_lag
    return velocity, target_max_lag + 4


def matched_density(size, target_degree):
    return min(0.5, target_degree / (size - 1))


# primary metric, scored against the physical distance delay field
def physical_taubar(model, velocity, max_delay):
    W = model.weight_matrix_numpy()                              # |W_rec|
    dist = model.geometry.distance_matrix().detach().cpu().numpy()
    tau = integer_delays(torch.tensor(dist), velocity, max_delay).numpy().astype(float)
    n = W.shape[0]
    off = ~np.eye(n, dtype=bool)
    sw = float(W[off].sum())
    C = float((W[off] * tau[off]).sum())
    tb = C / sw if sw > 0 else float("nan")
    # clip-saturation diagnostic on the physical field
    sat = float((tau[off] >= max_delay).mean())
    return tb, C, sw, sat


CONDITIONS = [
    ("distance", dict(use_delays=True, delay_control="distance")),
    ("shuffled", dict(use_delays=True, delay_control="shuffled")),
]


# stats
def paired_t(diff):
    diff = np.asarray(diff, float)
    n = len(diff)
    if n < 2 or diff.std(ddof=1) == 0:
        return float("inf") if diff.mean() != 0 else 0.0
    return float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n)))


def cohens_d_paired(diff):
    diff = np.asarray(diff, float)
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return float("nan")
    return float(diff.mean() / diff.std(ddof=1))


def boot_ci_mean(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    bs = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_slope_ci(x, y, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = np.arange(len(x))
    slopes = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if np.ptp(x[b]) == 0:
            continue
        slopes.append(np.polyfit(x[b], y[b], 1)[0])
    slopes = np.array(slopes)
    return float(np.polyfit(x, y, 1)[0]), float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[48, 96, 144])
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--reg-lambda", type=float, default=0.01, dest="reg_lambda")
    ap.add_argument("--target-degree", type=float, default=10.0, dest="target_degree")
    ap.add_argument("--target-max-lag", type=int, default=12, dest="target_max_lag")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--output", default=str(ROOT / "results" / "economy" / "scaling.json"))
    args = ap.parse_args()

    task = MemoryProTask(n_choices=4, delay_steps=6, response_steps=2, noise=0.2)
    records = []
    t0 = time.time()
    print(f"FS SCALING (saving vs N, matched dose)  sizes={args.sizes} dim={args.dim} "
          f"seeds={args.seeds} steps={args.steps} target_max_lag={args.target_max_lag}\n", flush=True)

    for size in args.sizes:
        density = matched_density(size, args.target_degree)
        v, md = matched_delay_params(size, args.dim, args.target_max_lag)
        print(f"-- N={size}  density={density:.4f}  velocity={v:.4f}  max_delay={md} "
              f"(target longest path ~{args.target_max_lag} steps) --", flush=True)
        for cond, ov in CONDITIONS:
            for seed in range(args.seeds):
                # The recurrent matrix is dense (no density knob) and tau_bar uses the
                # full |W| field, so density-matching doesn't apply here; we report the
                # degree-matched density only for the record.
                cfg = SDRNNConfig(hidden_size=size, space_dim=args.dim,
                                  reg_mode="communicability", reg_lambda=args.reg_lambda,
                                  velocity=v, max_delay=md, seed=seed, **ov)
                tc = TrainConfig(steps=args.steps, batch_size=args.batch,
                                 eval_every=args.steps, device=args.device, seed=seed, log=False)
                model, result = train(cfg, task, tc)
                tb, C, sw, sat = physical_taubar(model, v, md)
                rec = dict(dim=args.dim, size=size, density=density, velocity=v, max_delay=md,
                           condition=cond, seed=seed, acc=result.final_accuracy,
                           taubar=tb, C=C, sumW=sw, clip_sat=sat, wall_s=result.wall_seconds)
                records.append(rec)
                print(f"   N={size:>4} {cond:8s} seed={seed} acc={result.final_accuracy:.3f} "
                      f"taubar={tb:.4f} clip_sat={sat:.3f} ({result.wall_seconds:.0f}s)", flush=True)
                _write(args, records)

    analyze(records, args)
    print(f"\nDONE {len(records)} cells in {(time.time()-t0)/60:.1f} min", flush=True)


def _write(args, records):
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "records": records}, indent=1))


def analyze(records, args):
    print("\n" + "=" * 78)
    print("ANALYSIS: conduction-economy SAVING vs N (distance vs shuffled, matched dose)")
    print("=" * 78)
    sizes = sorted({r["size"] for r in records})
    pts_x, pts_y = [], []          # seed-level points for slope CI
    cell_x, cell_y = [], []        # cell means for R^2
    rows = []
    for N in sizes:
        dist = {r["seed"]: r for r in records if r["size"] == N and r["condition"] == "distance"}
        shuf = {r["seed"]: r for r in records if r["size"] == N and r["condition"] == "shuffled"}
        seeds = sorted(set(dist) & set(shuf))
        d_tb = np.array([dist[s]["taubar"] for s in seeds])
        s_tb = np.array([shuf[s]["taubar"] for s in seeds])
        d_ac = np.array([dist[s]["acc"] for s in seeds])
        s_ac = np.array([shuf[s]["acc"] for s in seeds])
        save = s_tb - d_tb                 # paired SAVING (>0 => distance economical)
        rel = save / s_tb
        acc_gap = abs(d_ac.mean() - s_ac.mean())
        matched = acc_gap <= 0.03
        t = paired_t(save)
        cd = cohens_d_paired(save)
        lo, hi = boot_ci_mean(save) if len(save) > 1 else (float("nan"), float("nan"))
        rlo, rhi = boot_ci_mean(rel) if len(rel) > 1 else (float("nan"), float("nan"))
        beats = save.mean() > 0 and lo > 0
        print(f"\nN={N}  seeds={len(seeds)}  acc d/s={d_ac.mean():.3f}/{s_ac.mean():.3f}"
              f"{'   [ACC-UNMATCHED]' if not matched else ''}")
        print(f"   taubar  dist={d_tb.mean():.4f}  shuf={s_tb.mean():.4f}")
        print(f"   SAVING     = {save.mean():+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
              f"paired_t={t:+.2f}  d={cd:+.2f}")
        print(f"   REL_SAVING = {rel.mean():+.4f}  95%CI[{rlo:+.4f},{rhi:+.4f}]")
        print(f"   beats-shuffled (SAVING>0, CI excl 0): {'YES' if beats else 'no'}")
        rows.append(dict(N=N, n_seeds=len(seeds), acc_d=float(d_ac.mean()), acc_s=float(s_ac.mean()),
                         matched=bool(matched), taubar_d=float(d_tb.mean()), taubar_s=float(s_tb.mean()),
                         saving=float(save.mean()), saving_ci=[lo, hi], paired_t=t, cohens_d=cd,
                         rel_saving=float(rel.mean()), rel_saving_ci=[rlo, rhi], beats=bool(beats)))
        if matched:
            for r in rel:
                pts_x.append(np.log10(N)); pts_y.append(r)
            cell_x.append(np.log10(N)); cell_y.append(float(rel.mean()))

    summary = dict(per_N=rows)
    if len(set(cell_x)) >= 2:
        b, blo, bhi = boot_slope_ci(pts_x, pts_y)
        coef = np.polyfit(cell_x, cell_y, 1)
        pred = np.polyval(coef, cell_x)
        ss_res = float(np.sum((np.array(cell_y) - pred) ** 2))
        ss_tot = float(np.sum((np.array(cell_y) - np.mean(cell_y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        verdict = ("STRENGTHENS with scale (b>0)" if blo > 0 else
                   "WASHES OUT with scale (b<0)" if bhi < 0 else
                   "SCALE-INVARIANT constant (slope CI spans 0)")
        print("\n" + "-" * 78)
        print(f"H2 SCALING FIT  REL_SAVING ~ a + b*log10(N)")
        print(f"   slope b = {b:+.4f}  95%CI[{blo:+.4f},{bhi:+.4f}]   R^2(cell means)={r2:.3f}")
        print(f"   VERDICT: {verdict}")
        summary.update(slope=b, slope_ci=[blo, bhi], r2=r2, verdict=verdict)
    out = Path(args.output)
    payload = json.loads(out.read_text())
    payload["summary"] = summary
    out.write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
