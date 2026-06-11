"""Dose-response (reservoir, regime A): is the conduction-time saving linear in delay-spread?

Saving = wmean_tau(shuffled) - wmean_tau(distance). Spread is set by velocity (lower v -> larger
lags -> larger spread). Sweep velocity in the trained-gain reservoir and fit Saving = B0*s.
Reuses train_one from law_reservoir.py, so dynamics, shuffle, gain pressure and cost are identical.

  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=3 \
    python scripts/law/law_reservoir_dose.py --device cpu
"""
import sys, os, json, math, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "law"))
import numpy as np
import torch

import importlib.util
spec = importlib.util.spec_from_file_location(
    "law_reservoir", str(ROOT / "scripts" / "law" / "law_reservoir.py"))
LR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LR)
from sdrnn.tasks import DelayedCopyTask

OUT = str(ROOT / "results" / "law" / "law_reservoir_dose.json")


def realized_spread(n, dim, velocity, max_delay, seed):
    """Std of the off-diagonal geometric integer lags = the delay-spread dose."""
    _, _, tau = LR.make_geometry(n, dim, velocity, max_delay, seed)
    off = ~torch.eye(n, dtype=torch.bool)
    v = tau[off].float()
    return float(v.std().item()), float(v.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--spectral_radius", type=float, default=0.95)
    ap.add_argument("--leak", type=float, default=0.6)
    ap.add_argument("--max_delay", type=int, default=24)
    ap.add_argument("--velocities", default="0.05,0.08,0.12,0.20")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--reg_lambda", type=float, default=1e-3)
    ap.add_argument("--lag", type=int, default=6)
    ap.add_argument("--seq_len", type=int, default=24)
    ap.add_argument("--n_symbols", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    device = torch.device(a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    task = DelayedCopyTask(n_symbols=a.n_symbols, lag=a.lag, seq_len=a.seq_len, noise=0.0)
    vels = [float(v) for v in a.velocities.split(",")]

    rows = []
    t0 = time.time()
    for vel in vels:
        spreads, means = [], []
        savings_wm, savings_raw = [], []
        accs_d, accs_s = [], []
        per_seed = []
        for seed in range(a.seeds):
            s_std, s_mean = realized_spread(a.n, a.dim, vel, a.max_delay, seed)
            rd = LR.train_one(task, a.n, a.density, a.spectral_radius, a.leak, vel,
                              a.max_delay, a.dim, "distance", seed, "A", a.steps, a.lr,
                              a.batch, device, a.reg_lambda, 1.0, 24000)
            rs = LR.train_one(task, a.n, a.density, a.spectral_radius, a.leak, vel,
                              a.max_delay, a.dim, "shuffled", seed, "A", a.steps, a.lr,
                              a.batch, device, a.reg_lambda, 1.0, 24000)
            gd, gs = rd["cost"]["gain"], rs["cost"]["gain"]
            sav_wm = gs["wmean_tau"] - gd["wmean_tau"]      # shuffled - distance (economy>0)
            sav_raw = gs["raw_cost"] - gd["raw_cost"]
            spreads.append(s_std); means.append(s_mean)
            savings_wm.append(sav_wm); savings_raw.append(sav_raw)
            accs_d.append(rd["acc"]); accs_s.append(rs["acc"])
            per_seed.append(dict(seed=seed, spread=s_std, mean_tau=s_mean,
                                 saving_wmean=sav_wm, saving_raw=sav_raw,
                                 acc_distance=rd["acc"], acc_shuffled=rs["acc"],
                                 wmean_distance=gd["wmean_tau"], wmean_shuffled=gs["wmean_tau"]))
            print(f"[vel={vel}] seed={seed} spread={s_std:.3f} meanTau={s_mean:.2f} "
                  f"acc(d/s)={rd['acc']:.3f}/{rs['acc']:.3f} "
                  f"saving_wmean={sav_wm:+.4f} saving_raw={sav_raw:+.1f}", flush=True)
        rows.append(dict(velocity=vel,
                         spread_mean=float(np.mean(spreads)),
                         tau_mean=float(np.mean(means)),
                         saving_wmean_mean=float(np.mean(savings_wm)),
                         saving_wmean_std=float(np.std(savings_wm)),
                         saving_raw_mean=float(np.mean(savings_raw)),
                         acc_distance=float(np.mean(accs_d)),
                         acc_shuffled=float(np.mean(accs_s)),
                         per_seed=per_seed))
        print(f"  vel={vel}: spread={np.mean(spreads):.3f} "
              f"saving_wmean={np.mean(savings_wm):+.4f}+/-{np.std(savings_wm):.4f}\n", flush=True)

    # linear fit Saving_wmean = B0 * spread (through origin) and with intercept.
    s = np.array([r["spread_mean"] for r in rows])
    y = np.array([r["saving_wmean_mean"] for r in rows])
    B0_origin = float((s @ y) / (s @ s)) if (s @ s) > 0 else float("nan")
    # with-intercept fit + R^2
    A = np.vstack([s, np.ones_like(s)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pear = float(np.corrcoef(s, y)[0, 1]) if len(s) > 1 else float("nan")

    result = dict(meta=dict(vars(a)), rows=rows,
                  fit=dict(B0_through_origin=B0_origin, slope=slope,
                           intercept=intercept, r2=r2, pearson=pear,
                           spreads=s.tolist(), savings_wmean=y.tolist()),
                  wall_seconds=time.time() - t0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print("=" * 64)
    print(f"DOSE-RESPONSE FIT  Saving_wmean ~ B0*spread")
    print(f"  through-origin B0 = {B0_origin:.4f}")
    print(f"  slope={slope:.4f} intercept={intercept:.4f} R^2={r2:.3f} pearson={pear:.3f}")
    for r in rows:
        print(f"  spread={r['spread_mean']:.3f} -> saving_wmean={r['saving_wmean_mean']:+.4f} "
              f"(acc d/s {r['acc_distance']:.3f}/{r['acc_shuffled']:.3f})")
    print(f"wrote {a.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
