"""Lag dose-response: does the conduction-cost saving scale with delay-demand?

The per-task saving (shuffled - distance, in |W|-weighted mean lag tau and raw
sum|W|*tau) is correlated against a functional delay-demand axis = the routing lag
of the task:

    memorypro_easy     demand 0   (blank hold; delay is costly-only, not routing)
    delayedcopy_lag2   demand 2
    delayedcopy_lag4   demand 4
    delayedcopy_lag6   demand 6
    delayedmatch_lag4  demand 4   (2nd delay-meaningful family: align sample to probe)

Accuracy is a gate, not the claim -- the cost comparison is only fair at matched
accuracy (acc-gate, not budget-gate). Conduction cost is scored against the same
fixed geometric tau for every condition. Checkpoints after every train.

  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=2 \
    python scripts/economy/lag_dose_response.py --device mps --seeds 4
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))
import argparse, json, math, os, time, numpy as np, torch
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F
from sdrnn.model import SDRNNConfig
from sdrnn.train import train, TrainConfig
from sdrnn.tasks import MemoryProTask, DelayedCopyTask
from sdrnn.delays import integer_delays

VEL, MAXD = 0.08, 14
(ROOT / "results" / "economy").mkdir(parents=True, exist_ok=True)
OUT = str(ROOT / "results" / "economy" / "lag_dose_response.json")


# Delayed match-to-sample: a 2nd delay-meaningful family (not copy). Show a sample,
# then `lag` blank steps, then a probe; report match/non-match at a go signal after
# the probe. The net must route the sample forward `lag` steps to align with the
# probe -- a temporal-alignment demand scaling with lag.
@dataclass
class DelayedMatchTask:
    n_symbols: int = 4
    lag: int = 4
    response_steps: int = 2
    noise: float = 0.0

    @property
    def input_size(self) -> int:
        return self.n_symbols + 1   # symbol channels (sample & probe) + go channel

    @property
    def output_size(self) -> int:
        return 2                    # non-match / match

    @property
    def seq_len(self) -> int:
        return 1 + self.lag + 1 + self.response_steps

    def generate(self, batch, generator=None):
        T = self.seq_len
        go_ch = self.n_symbols
        inputs = torch.zeros(batch, T, self.input_size)
        targets = torch.zeros(batch, T, dtype=torch.long)
        mask = torch.zeros(batch, T)

        sample = torch.randint(0, self.n_symbols, (batch,), generator=generator)
        is_match = (torch.rand(batch, generator=generator) < 0.5)
        offset = torch.randint(1, self.n_symbols, (batch,), generator=generator)
        probe = torch.where(is_match, sample, (sample + offset) % self.n_symbols)

        rows = torch.arange(batch)
        sample_t = 0
        probe_t = 1 + self.lag
        resp_start = probe_t + 1
        resp_end = resp_start + self.response_steps

        inputs[rows, sample_t, sample] = 1.0          # sample at t=0
        inputs[rows, probe_t, probe] = 1.0            # probe after the lag
        inputs[:, resp_start:resp_end, go_ch] = 1.0   # go signal
        targets[:, resp_start:resp_end] = is_match.long()[:, None]
        mask[:, resp_start:resp_end] = 1.0

        if self.noise > 0:
            inputs = inputs + self.noise * torch.randn(inputs.shape, generator=generator)
        return inputs, targets, mask

    def loss(self, outputs, targets, mask):
        b, T, c = outputs.shape
        per = F.cross_entropy(outputs.reshape(b * T, c), targets.reshape(b * T),
                              reduction="none").reshape(b, T)
        return (per * mask).sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def accuracy(self, outputs, targets, mask):
        pred = outputs.argmax(-1)
        return (((pred == targets).float() * mask).sum() / mask.sum().clamp_min(1.0)).item()


ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=48)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--device", default="mps")
ap.add_argument("--tasks", default="all")
a = ap.parse_args()

# demand = temporal-routing demand: copy/match -> lag (route forward `lag` steps);
# memorypro_easy -> 0 (blank hold, not routing). Budgets sized so accuracy matches
# near ceiling for all conditions; judged only at matched high accuracy.
TASKS = {
    "memorypro_easy": dict(
        factory=lambda: MemoryProTask(n_choices=4, delay_steps=4, noise=0.1),
        steps=1200, acc_target=0.95, demand=0, family="costly-only"),
    # Step budgets calibrated on the fn_multitask recipe (lag3 @3000 steps,
    # hidden=48, reg_lambda=0.01). Delay variants converge slower; longer lags get
    # more steps. acc_target is above chance (0.25 copy, 0.50 match), not ceiling --
    # the fair comparison only needs distance and shuffled matched.
    "delayedcopy_lag2": dict(
        factory=lambda: DelayedCopyTask(n_symbols=4, lag=2, seq_len=16, noise=0.0),
        steps=3000, acc_target=0.80, demand=2, family="meaningful"),
    "delayedcopy_lag4": dict(
        factory=lambda: DelayedCopyTask(n_symbols=4, lag=4, seq_len=18, noise=0.0),
        steps=4000, acc_target=0.70, demand=4, family="meaningful"),
    "delayedcopy_lag6": dict(
        factory=lambda: DelayedCopyTask(n_symbols=4, lag=6, seq_len=20, noise=0.0),
        steps=5000, acc_target=0.55, demand=6, family="meaningful"),
    "delayedmatch_lag4": dict(
        factory=lambda: DelayedMatchTask(n_symbols=4, lag=4, response_steps=2, noise=0.0),
        steps=3500, acc_target=0.85, demand=4, family="meaningful"),
}
if a.tasks != "all":
    keep = set(a.tasks.split(","))
    TASKS = {k: v for k, v in TASKS.items() if k in keep}

CONDS = [
    ("no-delay", dict(use_delays=False)),
    ("distance", dict(use_delays=True, delay_control="distance")),
    ("shuffled", dict(use_delays=True, delay_control="shuffled")),
]
ORDER = ["no-delay", "distance", "shuffled"]


def conduction_cost(m):
    """(weighted_mean tau, raw sum|W|*tau) vs the SAME fixed geometric tau."""
    W = m.weight_matrix_numpy()
    dist = m.geometry.distance_matrix().detach().cpu().numpy()
    tau = integer_delays(torch.tensor(dist), VEL, MAXD).numpy().astype(float)
    wsum = W.sum() + 1e-9
    return float((W * tau).sum() / wsum), float((W * tau).sum())


def sign_test(diffs):
    diffs = [x for x in diffs if x != 0.0]
    n = len(diffs)
    n_neg = sum(1 for x in diffs if x < 0)   # distance wins (lower cost)
    n_pos = n - n_neg
    k = min(n_neg, n_pos)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) if n > 0 else 1.0
    return n_neg, n_pos, min(1.0, 2 * tail)


def paired_t(diffs):
    d = np.asarray(diffs, float); n = len(d)
    if n < 2:
        return float("nan"), float(d.mean()) if n else float("nan")
    mean, sd = d.mean(), d.std(ddof=1)
    if sd == 0:
        return (float("inf") if mean < 0 else float("-inf")), float(mean)
    return float(mean / (sd / math.sqrt(n))), float(mean)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(v):
        order = np.argsort(v); r = np.empty(len(v)); r[order] = np.arange(len(v))
        return r
    return pearson(rank(np.asarray(x, float)), rank(np.asarray(y, float)))


def load_ckpt():
    if os.path.exists(OUT):
        try:
            return json.load(open(OUT))
        except Exception:
            pass
    return {"params": dict(hidden=a.hidden, seeds=a.seeds, velocity=VEL, max_delay=MAXD),
            "runs": {}, "tasks": {}}


def save(results):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)


results = load_ckpt()
results.setdefault("runs", {})

# train all (task, seed, cond), checkpointing each
for task_name, spec in TASKS.items():
    print("\n" + "#" * 72)
    print(f"# TASK: {task_name}  demand={spec['demand']} family={spec['family']} "
          f"steps={spec['steps']} acc_target={spec['acc_target']}")
    print("#" * 72, flush=True)
    for s in range(a.seeds):
        for name, ov in CONDS:
            key = f"{task_name}|{s}|{name}"
            if key in results["runs"]:
                r = results["runs"][key]
                print(f"  [resume] {key}: acc={r['acc']:.3f} wmean={r['wmean']:.3f}", flush=True)
                continue
            t0 = time.time()
            task = spec["factory"]()
            cfg = SDRNNConfig(hidden_size=a.hidden, reg_mode="communicability", reg_lambda=0.01,
                              velocity=VEL, max_delay=MAXD, seed=s, **ov)
            m, r = train(cfg, task, TrainConfig(steps=spec["steps"], batch_size=128,
                                                eval_every=spec["steps"], device=a.device,
                                                seed=s, log=False))
            wm, rt = conduction_cost(m)
            results["runs"][key] = dict(acc=float(r.final_accuracy), wmean=wm, raw=rt,
                                        secs=round(time.time() - t0, 1))
            save(results)
            print(f"  [{task_name}] seed {s} {name:9s}: acc={r.final_accuracy:.3f}  "
                  f"wmean_tau={wm:.3f}  raw={rt:.1f}  ({time.time()-t0:.0f}s)", flush=True)
        print("", flush=True)

# per-task analysis
results["tasks"] = {}
for task_name, spec in TASKS.items():
    wmean = {c: [] for c in ORDER}; rtot = {c: [] for c in ORDER}; accs = {c: [] for c in ORDER}
    for s in range(a.seeds):
        if not all(f"{task_name}|{s}|{c}" in results["runs"] for c in ORDER):
            continue
        for c in ORDER:
            r = results["runs"][f"{task_name}|{s}|{c}"]
            wmean[c].append(r["wmean"]); rtot[c].append(r["raw"]); accs[c].append(r["acc"])
    n_have = min(len(accs[c]) for c in ORDER)
    if n_have == 0:
        continue
    W_ = {c: (float(np.mean(wmean[c])), float(np.std(wmean[c]))) for c in ORDER}
    R_ = {c: (float(np.mean(rtot[c])), float(np.std(rtot[c]))) for c in ORDER}
    A_ = {c: (float(np.mean(accs[c])), float(np.std(accs[c]))) for c in ORDER}

    print("\n" + "-" * 72)
    print(f"[{task_name}] demand={spec['demand']}  ACC(gate) | WMEAN tau | RAW sum|W|*tau  n={n_have}")
    for c in ORDER:
        print(f"  {c:9s}: acc={A_[c][0]:.4f}±{A_[c][1]:.4f}  "
              f"wmean={W_[c][0]:.3f}±{W_[c][1]:.3f}  raw={R_[c][0]:.1f}±{R_[c][1]:.1f}")

    rec = {"demand": spec["demand"], "family": spec["family"], "n": n_have,
           "summary_accuracy": A_, "summary_weighted_mean": W_, "summary_raw_total": R_,
           "paired": {}}
    for metric_name, store in [("weighted_mean", wmean), ("raw_total", rtot)]:
        diffs = [store["distance"][s] - store["shuffled"][s] for s in range(n_have)]
        n_neg, n_pos, p_sign = sign_test(diffs)
        t, mean_d = paired_t(diffs)
        rec["paired"][metric_name] = dict(diffs=diffs, mean_dist_minus_shuf=mean_d,
                                          saving_shuf_minus_dist=-mean_d,
                                          sign_neg=n_neg, sign_pos=n_pos, sign_p=p_sign, t=t)
        dstr = ", ".join(f"{x:+.3f}" for x in diffs)
        print(f"  PAIRED {metric_name:12s} dist-shuf: [{dstr}]  mean={mean_d:+.4f}  "
              f"saving={-mean_d:+.4f}  dist_wins={n_neg}/{n_have}  t={t:.2f}  sign_p={p_sign:.4f}")

    acc_gap = abs(A_["distance"][0] - A_["shuffled"][0])
    acc_high = min(A_["distance"][0], A_["shuffled"][0]) >= spec["acc_target"]
    acc_matched = acc_gap < 0.03
    wm = rec["paired"]["weighted_mean"]; rt = rec["paired"]["raw_total"]
    wm_beats = W_["distance"][0] < W_["shuffled"][0]
    rt_beats = R_["distance"][0] < R_["shuffled"][0]
    majority = wm["sign_neg"] > wm["sign_pos"]
    cost_spine = wm_beats and rt_beats and majority
    clean = cost_spine and acc_matched and acc_high

    rec.update(acc_gap=acc_gap, acc_high=acc_high, acc_matched=acc_matched,
               cost_spine_holds=cost_spine, clean=clean)
    print(f"  ACC gate: dist={A_['distance'][0]:.3f} shuf={A_['shuffled'][0]:.3f} "
          f"gap={acc_gap:.4f} matched={acc_matched} high(>={spec['acc_target']})={acc_high}")
    print(f"  COST spine: wmean d<s={wm_beats} raw d<s={rt_beats} majority_win={majority} "
          f"-> COST_SPINE={cost_spine}")
    print(f"  VERDICT [{task_name}]: CLEAN(spine @ matched high acc)={clean}")
    results["tasks"][task_name] = rec
    save(results)

# the claim: saving scales with functional demand
print("\n" + "=" * 72)
print("DOSE-RESPONSE: per-task saving (shuffled-distance, wmean tau) vs delay-demand")
print("=" * 72)
rows = []
for tn, spec in TASKS.items():
    tr = results["tasks"].get(tn)
    if tr is None:
        print(f"  {tn:18s}: (no data)"); continue
    save_wm = tr["paired"]["weighted_mean"]["saving_shuf_minus_dist"]
    save_rt = tr["paired"]["raw_total"]["saving_shuf_minus_dist"]
    wins = tr["paired"]["weighted_mean"]["sign_neg"]
    rows.append(dict(task=tn, demand=tr["demand"], family=tr["family"],
                     saving_wmean=save_wm, saving_raw=save_rt,
                     wins=wins, n=tr["n"], clean=tr["clean"],
                     acc_d=tr["summary_accuracy"]["distance"][0],
                     acc_s=tr["summary_accuracy"]["shuffled"][0],
                     sign_p=tr["paired"]["weighted_mean"]["sign_p"]))
rows.sort(key=lambda r: r["demand"])
for r in rows:
    print(f"  {r['task']:18s} demand={r['demand']} ({r['family']:11s}): "
          f"saving_wmean={r['saving_wmean']:+.3f}  saving_raw={r['saving_raw']:+8.1f}  "
          f"dist_wins={r['wins']}/{r['n']}  p={r['sign_p']:.3f}  "
          f"acc d/s={r['acc_d']:.2f}/{r['acc_s']:.2f}  CLEAN={r['clean']}")

if len(rows) >= 3:
    demand = [r["demand"] for r in rows]
    sv_wm = [r["saving_wmean"] for r in rows]
    sv_rt = [r["saving_raw"] for r in rows]
    pr_wm = pearson(demand, sv_wm); sp_wm = spearman(demand, sv_wm)
    pr_rt = pearson(demand, sv_rt); sp_rt = spearman(demand, sv_rt)
    # within delay-meaningful family only (the cleanest dose-response):
    mrows = [r for r in rows if r["family"] == "meaningful"]
    if len(mrows) >= 3:
        md = [r["demand"] for r in mrows]; ms = [r["saving_wmean"] for r in mrows]
        pr_m = pearson(md, ms); sp_m = spearman(md, ms)
    else:
        pr_m = sp_m = float("nan")
    print("\n  CORRELATION  saving_vs_demand (all tasks):")
    print(f"    wmean: Pearson r={pr_wm:+.3f}  Spearman rho={sp_wm:+.3f}")
    print(f"    raw  : Pearson r={pr_rt:+.3f}  Spearman rho={sp_rt:+.3f}")
    print(f"  CORRELATION  within DELAY-MEANINGFUL only (wmean): "
          f"Pearson r={pr_m:+.3f}  Spearman rho={sp_m:+.3f}")
    results["dose_response"] = dict(
        rows=rows, pearson_wmean=pr_wm, spearman_wmean=sp_wm,
        pearson_raw=pr_rt, spearman_raw=sp_rt,
        pearson_wmean_meaningful=pr_m, spearman_wmean_meaningful=sp_m)
    # costly-only vs meaningful contrast
    co = [r for r in rows if r["family"] == "costly-only"]
    me = [r for r in rows if r["family"] == "meaningful"]
    if co and me:
        co_mean = float(np.mean([r["saving_wmean"] for r in co]))
        me_mean = float(np.mean([r["saving_wmean"] for r in me]))
        print(f"\n  CONTRAST mean saving_wmean: costly-only={co_mean:+.3f}  "
              f"meaningful={me_mean:+.3f}  (predict meaningful >> costly-only)")
        results["dose_response"]["mean_saving_costly_only"] = co_mean
        results["dose_response"]["mean_saving_meaningful"] = me_mean

n_clean = sum(1 for r in rows if r["clean"])
print(f"\nCLEAN cost-spine on {n_clean}/{len(rows)} tasks.")
print("=" * 72)
save(results)
print(f"wrote {OUT}")
