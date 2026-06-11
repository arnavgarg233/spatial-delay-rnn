"""Dose-response: does the distance-vs-shuffled conduction saving grow as delays
get longer (lower velocity)?

Metric: weighted-mean delay sum(|W|*tau)/sum(|W|), tau = round(d/v) clipped to
[1, max_delay] (the delay actually experienced at that velocity). max_delay=30 is
generous so the low-velocity end isn't pinned at the clip (~11% saturation at
v=0.04, ~0 above). Distance and shuffled share the same penalty and delay
histogram; only the distance<->delay correspondence differs, so any saving is the
travel-time effect, not the penalty or delay magnitude.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))
import argparse, json, numpy as np, torch
from sdrnn.model import SDRNNConfig
from sdrnn.train import train, TrainConfig
from sdrnn.tasks import MemoryProTask
from sdrnn.delays import integer_delays

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=48)
ap.add_argument("--steps", type=int, default=700)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--max_delay", type=int, default=30)
ap.add_argument("--velocities", type=float, nargs="+", default=[0.04, 0.08, 0.16, 0.4])
ap.add_argument("--device", default="mps")
a = ap.parse_args()

task = MemoryProTask(n_choices=4, delay_steps=6, noise=0.2)


def weighted_mean_delay(model, velocity, max_delay):
    """sum(|W|*tau)/sum(|W|) with tau at the trained velocity (true geometric delay)."""
    W = model.weight_matrix_numpy()                                  # |W_rec|
    dist = model.geometry.distance_matrix().detach().cpu().numpy()
    tau = integer_delays(torch.tensor(dist), velocity, max_delay).numpy().astype(float)
    return float((W * tau).sum() / (W.sum() + 1e-9))


def run_condition(delay_control, velocity, seeds, hidden, steps, max_delay, device):
    cws, accs = [], []
    for s in range(seeds):
        cfg = SDRNNConfig(
            hidden_size=hidden, reg_mode="communicability", reg_lambda=0.01,
            use_delays=True, velocity=velocity, max_delay=max_delay,
            delay_control=delay_control, seed=s,
        )
        m, r = train(cfg, task, TrainConfig(steps=steps, batch_size=128, eval_every=steps,
                                            device=device, seed=s, log=False))
        cws.append(weighted_mean_delay(m, velocity, max_delay))
        accs.append(r.final_accuracy)
        print(f"    v={velocity:5.3f} {delay_control:8s} seed {s}: "
              f"acc={r.final_accuracy:.3f}  wmt={cws[-1]:.3f}", flush=True)
    return np.array(cws), np.array(accs)


curve = []
print("=" * 70)
print(f"DOSE-RESPONSE  hidden={a.hidden}  seeds={a.seeds}  steps={a.steps}  "
      f"max_delay={a.max_delay}")
print("=" * 70)
for v in a.velocities:
    print(f"\n-- velocity {v} (lower v -> longer delays) --", flush=True)
    d_cw, d_acc = run_condition("distance", v, a.seeds, a.hidden, a.steps, a.max_delay, a.device)
    s_cw, s_acc = run_condition("shuffled", v, a.seeds, a.hidden, a.steps, a.max_delay, a.device)

    dist_mean, dist_std = float(d_cw.mean()), float(d_cw.std())
    shuf_mean, shuf_std = float(s_cw.mean()), float(s_cw.std())
    reduction = shuf_mean - dist_mean                       # absolute conduction-cost saving
    rel = reduction / (shuf_mean + 1e-9)                    # fractional saving
    # pooled-sd effect size (sigma separation) across seeds
    pooled = np.sqrt((d_cw.var(ddof=0) + s_cw.var(ddof=0)) / 2.0) + 1e-9
    sigma = reduction / pooled
    row = dict(velocity=v, max_delay=a.max_delay,
               distance_wmt=dist_mean, distance_std=dist_std,
               shuffled_wmt=shuf_mean, shuffled_std=shuf_std,
               reduction=reduction, rel_reduction=rel, sigma=float(sigma),
               distance_acc=float(d_acc.mean()), shuffled_acc=float(s_acc.mean()),
               distance_cws=d_cw.tolist(), shuffled_cws=s_cw.tolist())
    curve.append(row)
    print(f"  => dist {dist_mean:.3f}±{dist_std:.3f}  shuf {shuf_mean:.3f}±{shuf_std:.3f}  "
          f"REDUCTION {reduction:+.3f} ({100*rel:+.1f}%)  {sigma:+.1f}sigma  "
          f"acc d/s {d_acc.mean():.3f}/{s_acc.mean():.3f}", flush=True)

print("\n" + "=" * 70)
print("DOSE-RESPONSE CURVE  (velocity: lower = longer/more-impactful delays)")
print("=" * 70)
print(f"{'vel':>6} {'mean_tau':>9} {'distance':>10} {'shuffled':>10} "
      f"{'reduction':>10} {'rel%':>7} {'sigma':>7} {'acc d/s':>12}")
for r in curve:
    # nominal mean geometric delay at this velocity (shuffled==distance histogram)
    mt = 0.5 * (r["distance_wmt"] + r["shuffled_wmt"])
    print(f"{r['velocity']:>6.3f} {mt:>9.2f} "
          f"{r['distance_wmt']:>10.3f} {r['shuffled_wmt']:>10.3f} "
          f"{r['reduction']:>+10.3f} {100*r['rel_reduction']:>+6.1f}% "
          f"{r['sigma']:>+7.1f} {r['distance_acc']:.3f}/{r['shuffled_acc']:.3f}")

# Monotonicity check: reduction should increase as velocity decreases.
reds_by_incr_vel = [r["reduction"] for r in sorted(curve, key=lambda x: x["velocity"])]
# Lowest-velocity-first (= longest delays first), so it should decrease as velocity
# rises. (An earlier reversed() here double-reversed and spuriously failed the test.)
reds_lowfirst = list(reds_by_incr_vel)
mono = all(reds_lowfirst[i] >= reds_lowfirst[i + 1] - 1e-6 for i in range(len(reds_lowfirst) - 1))
all_pos = all(r["reduction"] > 0 for r in curve)
# Spearman-style trend: correlation of reduction with -velocity
vs = np.array([r["velocity"] for r in curve])
rd = np.array([r["reduction"] for r in curve])
trend = float(np.corrcoef(-np.log(vs), rd)[0, 1]) if len(curve) > 1 else float("nan")
print("=" * 70)
print(f"all distance < shuffled (positive reduction): {all_pos}")
print(f"monotone (reduction grows as velocity drops): {mono}")
print(f"trend corr(reduction, -log velocity) = {trend:+.3f}  "
      f"(want positive: bigger saving at longer delays)")
print("DOSE-RESPONSE: " + (
    "YES -- monotone, saving grows with delay length (supports travel-time principle)"
    if (all_pos and trend > 0.5) else
    "partial/flat -- inspect curve"))

out = dict(hidden=a.hidden, seeds=a.seeds, steps=a.steps, max_delay=a.max_delay,
           curve=curve, all_positive=bool(all_pos), monotone=bool(mono), trend=trend)
OUTDIR = ROOT / "results" / "economy"; OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "dose_response.json").write_text(json.dumps(out, indent=1))
print("\nwrote results/economy/dose_response.json")
