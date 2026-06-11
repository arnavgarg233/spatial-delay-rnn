"""Task/value vs synchrony make opposite velocity-allocation laws.

A learnable per-edge velocity network under a task objective speeds up short,
high-|W| edges (corr(v,dist) < 0; rev_myelin.py). Under a synchrony objective it
should instead speed up long edges to equalize arrival times (corr(v,dist) > 0).
Sweeping the trade-off weight w traces a Pareto front between throughput and
synchrony.

Same velocity budget mu*mean_offdiag(v) across all objectives (bare-speed, not
|W|-weighted) so synchrony can't cheat by globally slowing down; every objective
gets the same envelope and must decide where to spend it. Per-edge fractional
delay tau = clip(dist/v, 1, max). Losses:
  L_task = masked CE on MemoryPro
  L_sync = Var_offdiag(tau)              (low Var = synchronous = isochrony)
  L_bud  = mu * mean_offdiag(v)
  mixed  = (1-w)*L_task/s_task + w*L_sync/s_sync + L_bud
L_task/L_sync are normalized by init scales so w is interpretable; w=0 (pure task)
and w=1 (pure synchrony) are the anchors. Pass bar: sign flip in corr(v,dist) from
task to synchrony, a real accuracy<->Var(tau) trade-off, and distance-specific
(shuffled doesn't reproduce the flip).

  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=2 \
  python scripts/economy/pareto_dissociation.py --device mps
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from sdrnn.model import SDRNNConfig
from sdrnn.train import TrainConfig, resolve_device
from sdrnn.tasks import MemoryProTask

# reuse the learnable-velocity model + analysis from the myelin rig
sys.path.insert(0, str(ROOT / "scripts" / "archive"))
from rev_myelin import MyelinSDRNN  # noqa: E402


# Losses on the per-edge velocity / arrival-time field.
def _offdiag(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return x[off]


def arrival_tau(model: MyelinSDRNN) -> torch.Tensor:
    """Differentiable per-edge arrival time tau_ij = clip(dist/v, 1, max_delay).

    Uses the same geometry control (distance vs shuffled) the forward pass uses, so
    synchrony is defined on exactly the edges the dynamics see.
    """
    cfg = model.config
    dist = model.geometry.distance_matrix()
    dist_used = model._apply_geom_control(dist)
    v = model.velocity_matrix()
    return (dist_used / v).clamp(cfg.min_delay, cfg.max_delay)


def synchrony_loss(model: MyelinSDRNN) -> torch.Tensor:
    """Variance of arrival times across edges. Low = synchronous (isochrony)."""
    tau = _offdiag(arrival_tau(model))
    return tau.var(unbiased=False)


def speed_budget(model: MyelinSDRNN, mu: float) -> torch.Tensor:
    """Shared velocity envelope: pay mu per unit mean conduction speed."""
    v = model.velocity_matrix()
    return mu * _offdiag(v).mean()


# Train one network at trade-off weight w; returns model + eval metrics.
# Loss = (1-w)*L_task/s_task + w*L_sync/s_sync + mu*mean(v). s_task/s_sync are
# fixed normalizers (passed in) so w is comparable.
def train_pareto(cfg, task, tc, w, mu, s_task, s_sync):
    device = resolve_device(tc.device)
    cfg.input_size = task.input_size
    cfg.output_size = task.output_size
    torch.manual_seed(tc.seed)
    data_gen = torch.Generator().manual_seed(tc.seed + 1)
    model = MyelinSDRNN(cfg, vel_param="edge").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr)

    for step in range(1, tc.steps + 1):
        model.train()
        inputs, targets, mask = task.generate(tc.batch_size, generator=data_gen)
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        outputs = model(inputs)
        l_task = task.loss(outputs, targets, mask) + model.spatial_regularization()
        l_sync = synchrony_loss(model)
        loss = (1.0 - w) * (l_task / s_task) + w * (l_sync / s_sync)
        if mu > 0:
            loss = loss + speed_budget(model, mu)
        opt.zero_grad()
        loss.backward()
        if tc.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()

    return eval_model(model, task, tc, device)


def eval_model(model, task, tc, device):
    model.eval()
    gen = torch.Generator().manual_seed(12345)
    inp, tgt, msk = task.generate(tc.eval_batch, generator=gen)
    inp, tgt, msk = inp.to(device), tgt.to(device), msk.to(device)
    with torch.no_grad():
        out = model(inp)
        acc = float(task.accuracy(out, tgt, msk))
    m = analyze_velocity(model)
    m["acc"] = acc
    return model, m


# Velocity-allocation analysis (the dissociation read-out).
def analyze_velocity(model: MyelinSDRNN):
    n = model.config.hidden_size
    off = ~np.eye(n, dtype=bool)
    W = model.weight_matrix_numpy()
    dist = model.geometry.distance_matrix().detach().cpu().numpy()
    v = model.velocity_matrix().detach().cpu().numpy()

    w = W[off]; d = dist[off]; vv = v[off]
    wd = w * d

    def corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    # arrival-time variance (synchrony axis): on TRUE distances of each edge slot
    tau = np.clip(d / vv, model.config.min_delay, model.config.max_delay)
    var_tau = float(np.var(tau))
    cv_tau = float(np.std(tau) / (np.mean(tau) + 1e-12))

    return {
        "r_v_d": corr(vv, d),           # the dissociation metric (sign flips)
        "r_v_w": corr(vv, w),
        "r_v_wd": corr(vv, wd),
        "var_tau": var_tau,             # synchrony axis (lower = more synchronous)
        "cv_tau": cv_tau,
        "v_mean": float(vv.mean()), "v_std": float(vv.std()),
        "v_min": float(vv.min()), "v_max": float(vv.max()),
    }


# Calibrate normalizers s_task, s_sync from init-time loss magnitudes so the sweep
# weight w is meaningful.
def calibrate_scales(cfg_factory, task, tc, mu):
    """Fixed scales from L_task and L_sync at init (cheap, seed-stable). Only the
    ratio matters -- it puts the two losses on a common footing."""
    device = resolve_device(tc.device)
    cfg = cfg_factory(0, "distance")
    cfg.input_size = task.input_size
    cfg.output_size = task.output_size
    torch.manual_seed(tc.seed)
    gen = torch.Generator().manual_seed(tc.seed + 1)
    model = MyelinSDRNN(cfg, vel_param="edge").to(device)
    inp, tgt, msk = task.generate(tc.batch_size, generator=gen)
    inp, tgt, msk = inp.to(device), tgt.to(device), msk.to(device)
    with torch.no_grad():
        out = model(inp)
        s_task = float(task.loss(out, tgt, msk) + model.spatial_regularization())
        s_sync = float(synchrony_loss(model))
    return max(s_task, 1e-3), max(s_sync, 1e-6)


def agg(rows, key):
    a = np.array([r[key] for r in rows if key in r and not np.isnan(r[key])], float)
    return (float(a.mean()), float(a.std())) if a.size else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=56)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--velocity", type=float, default=0.08)
    ap.add_argument("--max-delay", type=int, default=14, dest="max_delay")
    ap.add_argument("--mu", type=float, default=0.05, help="shared speed budget weight")
    ap.add_argument("--weights", default="0,0.25,0.5,0.75,1.0",
                    help="comma list of synchrony trade-off weights w")
    ap.add_argument("--controls", default="distance,shuffled",
                    help="comma list of geometry controls to run")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    torch.set_num_threads(2)
    ws = [float(x) for x in args.weights.split(",")]
    controls = args.controls.split(",")
    task = MemoryProTask(n_choices=4, delay_steps=6, response_steps=2, noise=0.2)

    def cfg_factory(seed, ctrl):
        return SDRNNConfig(
            hidden_size=args.hidden, reg_mode="communicability", reg_lambda=0.01,
            use_delays=True, velocity=args.velocity, max_delay=args.max_delay,
            delay_interpolation="fractional", delay_control=ctrl, seed=seed,
        )

    print(f"NS-PARETO (task vs synchrony velocity allocation)  hidden={args.hidden} "
          f"steps={args.steps} seeds={args.seeds} v0={args.velocity} max_delay={args.max_delay}")
    print(f"shared speed budget mu={args.mu}  weights={ws}  controls={controls}\n")

    tc0 = TrainConfig(steps=args.steps, batch_size=128, eval_every=args.steps,
                      device=args.device, seed=0, log=False)
    s_task, s_sync = calibrate_scales(cfg_factory, task, tc0, args.mu)
    print(f"normalizers: s_task={s_task:.4f}  s_sync={s_sync:.4f}\n")

    # results[ctrl][w] = list of per-seed metric dicts
    results = {c: {f"{w}": [] for w in ws} for c in controls}
    OUTPATH = ROOT / "results" / "economy" / "pareto_dissociation.json"
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    for ctrl in controls:
        for w in ws:
            rows = []
            for s in range(args.seeds):
                cfg = cfg_factory(s, ctrl)
                tc = TrainConfig(steps=args.steps, batch_size=128, eval_every=args.steps,
                                 device=args.device, seed=s, log=False)
                _, m = train_pareto(cfg, task, tc, w, args.mu, s_task, s_sync)
                rows.append(m)
                print(f"  [{ctrl:8s} w={w:.2f}] seed {s}: acc={m['acc']:.3f}  "
                      f"corr(v,dist)={m['r_v_d']:+.3f}  Var(tau)={m['var_tau']:.3f}  "
                      f"CV(tau)={m['cv_tau']:.3f}  v[{m['v_min']:.3f},{m['v_max']:.3f}]",
                      flush=True)
            results[ctrl][f"{w}"] = rows
        # save partial after each control
        OUTPATH.write_text(json.dumps(
            {"args": vars(args), "s_task": s_task, "s_sync": s_sync,
             "weights": ws, "results": results}, indent=1))
        print()

    # Pareto / dissociation summary
    print("=" * 86)
    print("DISSOCIATION + PARETO FRONT  (distance condition)")
    print(f"{'w':>6}{'acc':>16}{'corr(v,dist)':>18}{'Var(tau)':>16}{'CV(tau)':>14}")
    for w in ws:
        r = results["distance"][f"{w}"]
        a = agg(r, "acc"); c = agg(r, "r_v_d"); vt = agg(r, "var_tau"); cv = agg(r, "cv_tau")
        print(f"{w:>6.2f}{a[0]:>9.3f}+/-{a[1]:<5.3f}{c[0]:>11.3f}+/-{c[1]:<5.3f}"
              f"{vt[0]:>9.3f}+/-{vt[1]:<5.3f}{cv[0]:>8.3f}+/-{cv[1]:<5.3f}")
    print("=" * 86)

    if "shuffled" in controls:
        print("\nDISTANCE-SPECIFICITY  corr(v,dist): distance vs shuffled (margin)")
        for w in ws:
            dd = agg(results["distance"][f"{w}"], "r_v_d")
            ss = agg(results["shuffled"][f"{w}"], "r_v_d")
            print(f"  w={w:.2f}: distance={dd[0]:+.3f}+/-{dd[1]:.3f}  "
                  f"shuffled={ss[0]:+.3f}+/-{ss[1]:.3f}  margin={dd[0]-ss[0]:+.3f}")

    # verdict
    w_task = f"{ws[0]}"        # pure task (w=0)
    w_sync = f"{ws[-1]}"       # pure synchrony (w=1)
    c_task = agg(results["distance"][w_task], "r_v_d")
    c_sync = agg(results["distance"][w_sync], "r_v_d")
    a_task = agg(results["distance"][w_task], "acc")
    a_sync = agg(results["distance"][w_sync], "acc")
    vt_task = agg(results["distance"][w_task], "var_tau")
    vt_sync = agg(results["distance"][w_sync], "var_tau")

    sign_flip = (c_task[0] < -0.05) and (c_sync[0] > 0.05)
    trade_off = (a_sync[0] < a_task[0] - 0.02) and (vt_sync[0] < vt_task[0])
    dist_specific = True
    if "shuffled" in controls:
        st = agg(results["shuffled"][w_task], "r_v_d")
        ss = agg(results["shuffled"][w_sync], "r_v_d")
        # the sign-flip should be much weaker / absent in shuffled
        dist_specific = ((c_task[0] - st[0]) < -0.10) and ((c_sync[0] - ss[0]) > 0.10)

    print("\nVERDICT:")
    print(f"  pure TASK   (w={ws[0]}): corr(v,dist)={c_task[0]:+.3f}  acc={a_task[0]:.3f}  Var(tau)={vt_task[0]:.3f}")
    print(f"  pure SYNC   (w={ws[-1]}): corr(v,dist)={c_sync[0]:+.3f}  acc={a_sync[0]:.3f}  Var(tau)={vt_sync[0]:.3f}")
    print(f"  SIGN FLIP (task<0, sync>0):        {sign_flip}")
    print(f"  REAL TRADE-OFF (acc down, Var down): {trade_off}")
    print(f"  DISTANCE-SPECIFIC (vs shuffled):     {dist_specific}")
    clean = bool(sign_flip and trade_off and dist_specific)
    print(f"  CLEAN DISSOCIATION + PARETO:         {clean}")

    out = {
        "args": vars(args), "s_task": s_task, "s_sync": s_sync, "weights": ws,
        "results": results,
        "verdict": {
            "corr_task": c_task, "corr_sync": c_sync,
            "acc_task": a_task, "acc_sync": a_sync,
            "var_task": vt_task, "var_sync": vt_sync,
            "sign_flip": sign_flip, "trade_off": trade_off,
            "dist_specific": dist_specific, "clean": clean,
        },
    }
    OUTPATH.write_text(json.dumps(out, indent=1))
    print("\nwrote results/economy/pareto_dissociation.json")


if __name__ == "__main__":
    main()
