"""Conduction-cost controls: distance vs shuffled and uniform, with paired stats.

tau is the geometric delay round(d/v) clipped to [1, max_delay], from the fixed
geometry; only the learned |W| differs across conditions. We compare the
weighted-mean delay sum(|W|*tau)/sum(|W|) and the raw total sum(|W|*tau), with
per-seed paired sign and t tests, at matched accuracy.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))
import argparse, json, math, numpy as np, torch
from sdrnn.model import SDRNNConfig
from sdrnn.train import train, TrainConfig
from sdrnn.tasks import MemoryProTask
from sdrnn.delays import integer_delays

VEL, MAXD = 0.08, 14

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=56)
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--device", default="mps")
a = ap.parse_args()

task = MemoryProTask(n_choices=4, delay_steps=6, noise=0.2)

# no-delay trains without delays but we still score its learned |W| against the
# geometric tau -- the "delays don't shape wiring" reference.
conds = [
    ("no-delay", dict(use_delays=False)),
    ("distance", dict(use_delays=True, delay_control="distance")),
    ("shuffled", dict(use_delays=True, delay_control="shuffled")),
    ("uniform",  dict(use_delays=True, delay_control="uniform")),
]

# Per-seed records: cond -> list over seeds of (weighted_mean, raw_total, acc)
wmean = {c: [] for c, _ in conds}
rtot  = {c: [] for c, _ in conds}
accs  = {c: [] for c, _ in conds}

for s in range(a.seeds):
    for name, ov in conds:
        cfg = SDRNNConfig(hidden_size=a.hidden, reg_mode="communicability", reg_lambda=0.01,
                          velocity=VEL, max_delay=MAXD, seed=s, **ov)
        m, r = train(cfg, task, TrainConfig(steps=a.steps, batch_size=128, eval_every=a.steps,
                                            device=a.device, seed=s, log=False))
        W = m.weight_matrix_numpy()                                   # |W_rec|, (N,N)
        dist = m.geometry.distance_matrix().detach().cpu().numpy()    # fixed geometry
        tau = integer_delays(torch.tensor(dist), VEL, MAXD).numpy().astype(float)  # geometric delay
        wsum = W.sum() + 1e-9
        wm = float((W * tau).sum() / wsum)
        rt = float((W * tau).sum())
        wmean[name].append(wm)
        rtot[name].append(rt)
        accs[name].append(float(r.final_accuracy))
        print(f"  seed {s} {name:9s}: acc={r.final_accuracy:.3f}  wmean_tau={wm:.3f}  raw={rt:.1f}", flush=True)
    print("", flush=True)


def summ(d):
    return {c: (float(np.mean(v)), float(np.std(v))) for c, v in d.items()}


def sign_test(diffs):
    """Two-sided exact binomial sign test on paired diffs (H0: P(+)=0.5) -> (n_neg, n_pos, p)."""
    diffs = [x for x in diffs if x != 0.0]
    n = len(diffs)
    n_neg = sum(1 for x in diffs if x < 0)   # distance lower than control = win
    n_pos = n - n_neg
    k = min(n_neg, n_pos)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) if n > 0 else 1.0
    p = min(1.0, 2 * tail)
    return n_neg, n_pos, p


def paired_t(diffs):
    d = np.asarray(diffs, float)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    mean, sd = d.mean(), d.std(ddof=1)
    if sd == 0:
        return float("inf") * (1 if mean > 0 else -1), 0.0
    t = mean / (sd / math.sqrt(n))
    return float(t), float(mean)


W_ = summ(wmean); R_ = summ(rtot); A_ = summ(accs)
order = ["no-delay", "distance", "shuffled", "uniform"]

print("=" * 70)
print("WEIGHTED-MEAN conduction delay  (lower = less conduction time):")
for c in order:
    print(f"  {c:9s}: {W_[c][0]:.3f} ± {W_[c][1]:.3f}   acc={A_[c][0]:.4f} ± {A_[c][1]:.4f}")
print("-" * 70)
print("RAW total cost sum(|W|*tau)  (lower = cheaper delay-weighted wiring):")
for c in order:
    print(f"  {c:9s}: {R_[c][0]:.2f} ± {R_[c][1]:.2f}")
print("=" * 70)

# Paired per-seed differences: distance - control (negative = distance wins).
results = {"params": dict(hidden=a.hidden, steps=a.steps, seeds=a.seeds, velocity=VEL, max_delay=MAXD),
           "weighted_mean": {c: wmean[c] for c in order},
           "raw_total": {c: rtot[c] for c in order},
           "accuracy": {c: accs[c] for c in order},
           "summary_weighted_mean": W_, "summary_raw_total": R_, "summary_accuracy": A_,
           "paired": {}}

for metric_name, store in [("weighted_mean", wmean), ("raw_total", rtot)]:
    print(f"\nPAIRED (per-seed) distance - control on {metric_name}  (negative = distance wins):")
    for ctrl in ["shuffled", "uniform"]:
        diffs = [store["distance"][s] - store[ctrl][s] for s in range(a.seeds)]
        n_neg, n_pos, p_sign = sign_test(diffs)
        t, mean_d = paired_t(diffs)
        wins = n_neg  # negative diff = distance lower = win
        results["paired"][f"{metric_name}_distance_minus_{ctrl}"] = dict(
            diffs=diffs, mean=mean_d, sign_neg=n_neg, sign_pos=n_pos, sign_p=p_sign, t=t)
        dstr = ", ".join(f"{x:+.3f}" for x in diffs)
        print(f"  vs {ctrl:9s}: diffs=[{dstr}]  mean={mean_d:+.4f}  "
              f"distance_wins={wins}/{a.seeds}  sign_p={p_sign:.4f}  t={t:.2f}")

# Accuracy-matched check: max pairwise gap in mean accuracy.
acc_means = [A_[c][0] for c in order]
acc_gap = max(acc_means) - min(acc_means)
print("\n" + "=" * 70)
print(f"ACCURACY MATCHED: range across conditions = {acc_gap:.4f} "
      f"(min {min(acc_means):.4f}, max {max(acc_means):.4f}) -> "
      f"{'MATCHED' if acc_gap < 0.02 else 'CHECK - gap > 0.02'}")
results["acc_gap"] = acc_gap

# Verdict.
wm_beats_shuf = W_["distance"][0] < W_["shuffled"][0]
wm_beats_unif = W_["distance"][0] < W_["uniform"][0]
rt_beats_shuf = R_["distance"][0] < R_["shuffled"][0]
rt_beats_unif = R_["distance"][0] < R_["uniform"][0]
print(f"weighted-mean: distance < shuffled {wm_beats_shuf}, distance < uniform {wm_beats_unif}")
print(f"raw total:     distance < shuffled {rt_beats_shuf}, distance < uniform {rt_beats_unif}")
results["verdict"] = dict(wm_beats_shuf=wm_beats_shuf, wm_beats_unif=wm_beats_unif,
                          rt_beats_shuf=rt_beats_shuf, rt_beats_unif=rt_beats_unif)
print("=" * 70)

import os
OUTDIR = ROOT / "results" / "economy"
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "conduction_cost.json").write_text(json.dumps(results, indent=1))
print("wrote results/economy/conduction_cost.json")
