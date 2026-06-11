"""Phase diagram of the conduction-time economy over (task demand) x (delay spread).

We map a 2D grid and ask where the distance-vs-shuffled saving turns on:
  * demand axis = input noise on MemoryPro. Clean input just latches the cue;
    noisy input forces the recurrent weights to integrate and become load-bearing.
    Both distance and shuffled stay accuracy-matched at every level (unlike
    DelayedCopy lag, which mechanically breaks matching).
  * delay axis = conduction velocity (sets the geometric delay spread). Lower v ->
    longer, more dispersed lags -> more conduction time to save.

Order parameter: SAVING = wmean_tau(shuffled) - wmean_tau(distance), paired per
seed; >0 means distance is cheaper than the matched scramble. Cells with a
distance-vs-shuffled accuracy gap >= 0.03 are flagged ACC-GAP and excluded.

kappa = -spearman(g_edge, tau_edge) over off-diagonal edges, with the costate
leverage g_ij = |dL_task/dW_ij| * |W_ij| (first-order task-loss sensitivity to
edge ij). kappa>0 means task value sits on short-delay edges; we test whether it
rises with demand and tracks where SAVING turns on.

Checkpointed per (cell, seed, cond) so a crash resumes.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))
import argparse, json, math, os, time
import numpy as np
import torch

from sdrnn.model import SDRNN, SDRNNConfig
from sdrnn.train import train, TrainConfig
from sdrnn.tasks import MemoryProTask
from sdrnn.delays import integer_delays

OUT = str(ROOT / "results" / "economy" / "phase_transition.json")

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=48)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="mps")
# DEMAND axis = input noise (ambiguity -> integration demand). 0 = trivial latch.
ap.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8])
# DELAY axis = conduction velocity (delay spread). lower = longer lags.
ap.add_argument("--vels", type=float, nargs="+", default=[0.05, 0.10])
ap.add_argument("--max_delay", type=int, default=16)
ap.add_argument("--delay_steps", type=int, default=6)   # blank hold length (fixed)
ap.add_argument("--n_choices", type=int, default=4)
ap.add_argument("--steps", type=int, default=2500)
ap.add_argument("--quick", action="store_true", help="tiny smoke grid")
a = ap.parse_args()

if a.quick:
    a.noises = [0.0, 0.6]; a.vels = [0.05]; a.seeds = 2; a.steps = 700; a.hidden = 40

CONDS = [
    ("no-delay", dict(use_delays=False)),
    ("distance", dict(use_delays=True, delay_control="distance")),
    ("shuffled", dict(use_delays=True, delay_control="shuffled")),
]
ORDER = ["no-delay", "distance", "shuffled"]


def make_task(noise):
    return MemoryProTask(n_choices=a.n_choices, cue_steps=2, delay_steps=a.delay_steps,
                         response_steps=2, noise=noise)


# metrics
def geometric_tau(m, velocity, max_delay):
    dist = m.geometry.distance_matrix().detach().cpu().numpy()
    return integer_delays(torch.tensor(dist), velocity, max_delay).numpy().astype(float)


def conduction_cost(m, velocity, max_delay):
    """(weighted_mean tau, raw sum|W|*tau) vs the SAME fixed geometric tau."""
    W = m.weight_matrix_numpy()
    tau = geometric_tau(m, velocity, max_delay)
    wsum = W.sum() + 1e-9
    return float((W * tau).sum() / wsum), float((W * tau).sum())


def costate_leverage_kappa(m, task, velocity, max_delay, device, batch=256, seed=999):
    """kappa = -spearman(g_edge, tau_edge), g_ij = |dL_task/dW_ij|*|W_ij| (costate)."""
    m.eval()
    for p in m.parameters():
        p.requires_grad_(True)
    m.zero_grad(set_to_none=True)
    gen = torch.Generator().manual_seed(seed)
    inputs, targets, mask = task.generate(batch, generator=gen)
    inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
    outputs = m(inputs)
    loss = task.loss(outputs, targets, mask)        # TASK loss only -> pure costate
    grad = torch.autograd.grad(loss, m.recurrent.weight, retain_graph=False)[0]
    g = (grad.detach().abs() * m.recurrent.weight.detach().abs()).cpu().numpy()
    tau = geometric_tau(m, velocity, max_delay)
    n = g.shape[0]
    off = ~np.eye(n, dtype=bool)
    kappa = -spearman(g[off], tau[off])
    m.zero_grad(set_to_none=True)
    return float(kappa)


# stats
def sign_test(diffs):
    diffs = [x for x in diffs if x != 0.0]
    n = len(diffs)
    n_pos = sum(1 for x in diffs if x > 0)      # SAVING>0 => distance wins
    n_neg = n - n_pos
    k = min(n_neg, n_pos)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) if n > 0 else 1.0
    return n_pos, n_neg, min(1.0, 2 * tail)


def paired_t(diffs):
    d = np.asarray(diffs, float); n = len(d)
    if n < 2:
        return float("nan"), float(d.mean()) if n else float("nan")
    mean, sd = d.mean(), d.std(ddof=1)
    if sd == 0:
        return (float("inf") if mean > 0 else float("-inf")), float(mean)
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


# checkpoint
def load_ckpt():
    if os.path.exists(OUT):
        try:
            return json.load(open(OUT))
        except Exception:
            pass
    return {"params": dict(hidden=a.hidden, seeds=a.seeds, steps=a.steps, noises=a.noises,
                           vels=a.vels, max_delay=a.max_delay, delay_steps=a.delay_steps,
                           n_choices=a.n_choices, task="MemoryPro", demand_axis="input_noise"),
            "runs": {}}


def save(results):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)


results = load_ckpt()
results.setdefault("runs", {})

# train grid
t_start = time.time()
for vel in a.vels:
    for noise in a.noises:
        print("\n" + "#" * 74)
        print(f"# CELL  noise={noise}  velocity={vel}  (demand=input noise, delay=velocity)")
        print("#" * 74, flush=True)
        for s in range(a.seeds):
            for name, ov in CONDS:
                key = f"v{vel}|noise{noise}|{s}|{name}"
                if key in results["runs"]:
                    r = results["runs"][key]
                    print(f"  [resume] {key}: acc={r['acc']:.3f} wmean={r['wmean']:.3f} "
                          f"kappa={r.get('kappa', float('nan')):.3f}", flush=True)
                    continue
                t0 = time.time()
                task = make_task(noise)
                cfg = SDRNNConfig(hidden_size=a.hidden, reg_mode="communicability",
                                  reg_lambda=0.01, velocity=vel, max_delay=a.max_delay,
                                  seed=s, **ov)
                m, r = train(cfg, task, TrainConfig(steps=a.steps, batch_size=128,
                                                    eval_every=a.steps, device=a.device,
                                                    seed=s, log=False))
                wm, rt = conduction_cost(m, vel, a.max_delay)
                try:
                    kappa = costate_leverage_kappa(m, task, vel, a.max_delay,
                                                   m.recurrent.weight.device)
                except Exception as e:
                    print(f"    kappa failed: {e}", flush=True)
                    kappa = float("nan")
                results["runs"][key] = dict(acc=float(r.final_accuracy), wmean=wm, raw=rt,
                                            kappa=kappa, secs=round(time.time() - t0, 1))
                save(results)
                print(f"  noise{noise} v{vel} seed{s} {name:9s}: acc={r.final_accuracy:.3f}  "
                      f"wmean={wm:.3f}  raw={rt:.1f}  kappa={kappa:+.3f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            print("", flush=True)

# analysis
print("\n" + "=" * 80)
print("PHASE DIAGRAM  (demand=input noise rows, delay=velocity cols)")
print("Order parameter SAVING = wmean(shuffled) - wmean(distance)  [>0 = economy ON]")
print("=" * 80)

cells = {}
for vel in a.vels:
    for noise in a.noises:
        if not any(f"v{vel}|noise{noise}|{s}|distance" in results["runs"] for s in range(a.seeds)):
            continue
        per = {c: dict(acc=[], wmean=[], raw=[], kappa=[]) for c in ORDER}
        for s in range(a.seeds):
            if not all(f"v{vel}|noise{noise}|{s}|{c}" in results["runs"] for c in ORDER):
                continue
            for c in ORDER:
                r = results["runs"][f"v{vel}|noise{noise}|{s}|{c}"]
                per[c]["acc"].append(r["acc"]); per[c]["wmean"].append(r["wmean"])
                per[c]["raw"].append(r["raw"]); per[c]["kappa"].append(r["kappa"])
        sv_wm, sv_rt = [], []
        for s in range(a.seeds):
            kd = f"v{vel}|noise{noise}|{s}|distance"; ks = f"v{vel}|noise{noise}|{s}|shuffled"
            if kd in results["runs"] and ks in results["runs"]:
                sv_wm.append(results["runs"][ks]["wmean"] - results["runs"][kd]["wmean"])
                sv_rt.append(results["runs"][ks]["raw"] - results["runs"][kd]["raw"])
        if not sv_wm:
            continue
        n_pos, n_neg, p_sign = sign_test(sv_wm)
        t_wm, mean_wm = paired_t(sv_wm)
        t_rt, mean_rt = paired_t(sv_rt)
        acc_d = float(np.mean(per["distance"]["acc"])); acc_s = float(np.mean(per["shuffled"]["acc"]))
        acc_gap = abs(acc_d - acc_s)
        kappa_d = float(np.nanmean(per["distance"]["kappa"]))
        kappa_s = float(np.nanmean(per["shuffled"]["kappa"]))
        matched = acc_gap < 0.03
        economy_on = (mean_wm > 0) and (n_pos == len(sv_wm)) and matched
        cells[(vel, noise)] = dict(
            saving_wmean=mean_wm, saving_wmean_t=t_wm, saving_raw=mean_rt, saving_raw_t=t_rt,
            sign_pos=n_pos, sign_neg=n_neg, sign_p=p_sign, n_seeds=len(sv_wm),
            acc_distance=acc_d, acc_shuffled=acc_s, acc_gap=acc_gap, matched=matched,
            kappa_distance=kappa_d, kappa_shuffled=kappa_s, economy_on=economy_on,
            saving_per_seed=sv_wm)
        flag = "ON " if economy_on else ("off" if matched else "ACC-GAP!")
        print(f"  v={vel:<5} noise={noise:<4} | SAVING_wm={mean_wm:+.3f} (t={t_wm:+.2f}, "
              f"{n_pos}/{len(sv_wm)}+) raw={mean_rt:+.1f} | acc d={acc_d:.3f}/s={acc_s:.3f} "
              f"(gap={acc_gap:.3f}) | kappa_d={kappa_d:+.3f} | [{flag}]")

# does kappa track the transition?
print("\n" + "=" * 80)
print("DOES kappa TRACK THE TRANSITION?  (per velocity column, across demand/noise)")
print("=" * 80)
kappa_track = {}
for vel in a.vels:
    noises_here = [nz for nz in a.noises if (vel, nz) in cells]
    if len(noises_here) < 2:
        continue
    sv = [cells[(vel, nz)]["saving_wmean"] for nz in noises_here]
    kp = [cells[(vel, nz)]["kappa_distance"] for nz in noises_here]
    dm = [float(nz) for nz in noises_here]
    r_sv_dm = pearson(dm, sv)
    r_kp_dm = pearson(dm, kp)
    r_sv_kp = pearson(kp, sv); sr_sv_kp = spearman(kp, sv)
    kappa_track[vel] = dict(noises=noises_here, saving=sv, kappa=kp,
                            corr_saving_vs_demand=r_sv_dm, corr_kappa_vs_demand=r_kp_dm,
                            corr_saving_vs_kappa=r_sv_kp, spearman_saving_vs_kappa=sr_sv_kp)
    print(f"  v={vel}:  corr(saving, demand)={r_sv_dm:+.3f}  corr(kappa, demand)={r_kp_dm:+.3f}  "
          f"corr(saving, kappa)={r_sv_kp:+.3f} (sp {sr_sv_kp:+.3f})")
    print(f"           noises={noises_here}")
    print(f"           saving={[round(x,3) for x in sv]}")
    print(f"           kappa ={[round(x,3) for x in kp]}")

all_sv = [c["saving_wmean"] for c in cells.values()]
all_kp = [c["kappa_distance"] for c in cells.values()]
pooled_r = pearson(all_kp, all_sv); pooled_sr = spearman(all_kp, all_sv)
print(f"\nPOOLED across all {len(cells)} cells: corr(saving, kappa)={pooled_r:+.3f} "
      f"(spearman {pooled_sr:+.3f})")

# verdict
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
on_cells = [(v, nz) for (v, nz), c in cells.items() if c["economy_on"]]
off_cells = [(v, nz) for (v, nz), c in cells.items() if not c["economy_on"] and c["matched"]]
accgap_cells = [(v, nz) for (v, nz), c in cells.items() if not c["matched"]]
print(f"  economy ON  cells ({len(on_cells)}): {sorted(on_cells)}")
print(f"  economy off cells ({len(off_cells)}): {sorted(off_cells)}")
if accgap_cells:
    print(f"  ACC-GAP (excluded) cells ({len(accgap_cells)}): {sorted(accgap_cells)}")
on_noises = sorted({nz for (v, nz) in on_cells})
off_noises = sorted({nz for (v, nz) in off_cells})
print(f"  demand levels with >=1 ON cell : {on_noises}")
print(f"  demand levels only off         : {[nz for nz in off_noises if nz not in on_noises]}")

results["cells"] = {f"v{v}|noise{nz}": c for (v, nz), c in cells.items()}
results["kappa_track"] = {f"v{v}": d for v, d in kappa_track.items()}
results["pooled"] = dict(corr_saving_kappa=pooled_r, spearman_saving_kappa=pooled_sr,
                         on_cells=[f"v{v}|noise{nz}" for (v, nz) in on_cells],
                         off_cells=[f"v{v}|noise{nz}" for (v, nz) in off_cells],
                         accgap_cells=[f"v{v}|noise{nz}" for (v, nz) in accgap_cells],
                         on_noises=on_noises)
save(results)
print(f"\nwrote {OUT}   (total {time.time()-t_start:.0f}s)")
