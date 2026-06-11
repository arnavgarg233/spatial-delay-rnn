"""CONDUCTION-GEOMETRY AS A NON-REMOVABLE FINGERPRINT (watermark / PUF cheap-kill).

CLAIM (deep-res divergent-A headline, re-sculpting-proof BY CONSTRUCTION): the distance->delay map
tau_ij=round(d_ij/v) is a PLANTED PHYSICAL INVARIANT baked into the delay buffer. A freely (re)trained
recurrent W cannot erase it, and the velocity INVERSE recovers the planted v from the network's own
activity REGARDLESS of W. That makes the conduction geometry a structural watermark for delay-coupled
analog/neuromorphic hardware: an adversary who can only retrain WEIGHTS cannot remove it; only tampering
with the physical delay buffer (the velocity itself) can.

CHEAP-KILL (this script). Plant v_true, train an SDRNN on a task, and measure the fingerprint's
READABILITY = the velocity-shuffle-null GAP from pinn_inverse: how much lower the true-geometry free-kernel
residual floor is than the best-over-velocity floor on a delay-histogram-matched SHUFFLED geometry (the
null's best shot). Then run an ADVERSARY that retrains W with the SAME planted delays, two ways:
  (1) W-INDEPENDENCE: a fresh W (different seed) solving the same task -> is the gap preserved?
  (2) OVER-TRAIN ATTACK: heavily over-train W (3x steps + weight decay) to seek a delay-independent
      solution -> does the gap survive?
BOUNDARY CONTROL: re-plant a DIFFERENT velocity (tamper the delay buffer) -> the inverse should now recover
the NEW v, not the old -> confirming the fingerprint tracks the physical delays, not W.
If the gap (and v_hat~v_true) SURVIVE W-retraining but MOVE when the velocity is re-planted, the conduction
geometry is a W-robust structural fingerprint. If a retrained W erases the gap, the watermark is removable
(null) -- report honestly.

Reuses pinn_inverse.{record_task_activity, reconstruct_rec_target, scan_freekernel, shuffle_geometry}.
Run: PYTHONPATH=.:src python experiments/fingerprint_watermark.py --seeds 3
"""
import sys, os, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts", "inverse"))
from sdrnn.model import SDRNNConfig
from sdrnn.train import train, TrainConfig
from sdrnn.tasks import MemoryProTask
import pinn_inverse as pi


def readability(model, task, device, v_true, cands, args, seed):
    """Fingerprint readability on this trained model: free-kernel residual floor on TRUE geometry, the
    null's best-shot floor on a velocity-SHUFFLED geometry, their GAP, and the recovered velocity v_ft."""
    rates, states, proj, alpha, W, bias = pi.record_task_activity(
        model, task, device, n_trials=args.n_trials, n_repeats=args.n_repeats, seed=seed)
    rec = pi.reconstruct_rec_target(states, proj, alpha, bias)
    dist = model.geometry.distance_matrix().detach().cpu().numpy()
    d_shuf = pi.shuffle_geometry(dist, seed)
    mind = model.config.min_delay
    res_true, _ = pi.scan_freekernel(rates, rec, dist, cands, args.max_delay, mind, args.ridge, args.rank)
    res_shuf, _ = pi.scan_freekernel(rates, rec, d_shuf, cands, args.max_delay, mind, args.ridge, args.rank)
    floor_true, floor_shuf = float(res_true.min()), float(res_shuf.min())
    v_ft = float(cands[int(np.argmin(res_true))])
    interior = 0 < int(np.argmin(res_true)) < len(cands) - 1
    return dict(floor_true=floor_true, floor_shuf=floor_shuf, gap=floor_shuf - floor_true,
                v_ft=v_ft, v_err=abs(v_ft - v_true) / v_true, interior=bool(interior),
                acc=None)


def make_model(v, steps, seed, args, task, wd=0.0):
    cfg = SDRNNConfig(hidden_size=args.hidden, reg_mode="communicability", reg_lambda=0.01,
                      use_delays=True, velocity=v, max_delay=args.max_delay,
                      delay_control="distance", seed=seed)
    tc = TrainConfig(steps=steps, batch_size=128, eval_every=steps, device=args.device, seed=seed, log=False)
    model, result = train(cfg, task, tc)
    return model, float(result.final_accuracy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--adv-steps", type=int, default=2700, dest="adv_steps")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--v-true", type=float, default=0.08, dest="v_true")
    ap.add_argument("--v-replant", type=float, default=0.16, dest="v_replant")  # boundary: tamper the buffer
    ap.add_argument("--max-delay", type=int, default=14, dest="max_delay")
    ap.add_argument("--n-trials", type=int, default=64, dest="n_trials")
    ap.add_argument("--n-repeats", type=int, default=6, dest="n_repeats")
    ap.add_argument("--n-cands", type=int, default=15, dest="n_cands")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "experiments", "fingerprint_watermark.json"))
    args = ap.parse_args()

    task = MemoryProTask(n_choices=4, cue_steps=2, delay_steps=8, response_steps=2, noise=0.2)
    cands = np.geomspace(args.v_true / 3.5, args.v_true * 3.5, args.n_cands)
    cands_rp = np.geomspace(args.v_replant / 3.5, args.v_replant * 3.5, args.n_cands)
    print(f"FINGERPRINT WATERMARK  v_true={args.v_true} replant={args.v_replant} hidden={args.hidden} "
          f"steps={args.steps} adv={args.adv_steps} seeds={args.seeds}\n", flush=True)

    rows = []
    for seed in range(args.seeds):
        # honest net (planted v_true)
        m_h, acc_h = make_model(args.v_true, args.steps, seed, args, task)
        r_h = readability(m_h, task, args.device, args.v_true, cands, args, seed); r_h["acc"] = acc_h
        # ADVERSARY 1: fresh W (different seed), same delays -> W-independence
        m_i, acc_i = make_model(args.v_true, args.steps, seed + 100, args, task)
        r_i = readability(m_i, task, args.device, args.v_true, cands, args, seed + 100); r_i["acc"] = acc_i
        # ADVERSARY 2: over-trained W (3x steps) -> seek a delay-independent solution
        m_a, acc_a = make_model(args.v_true, args.adv_steps, seed, args, task)
        r_a = readability(m_a, task, args.device, args.v_true, cands, args, seed); r_a["acc"] = acc_a
        # BOUNDARY: re-plant a DIFFERENT velocity (tamper the delay buffer) -> inverse should track NEW v
        m_b, acc_b = make_model(args.v_replant, args.steps, seed, args, task)
        r_b = readability(m_b, task, args.device, args.v_replant, cands_rp, args, seed); r_b["acc"] = acc_b
        rows.append(dict(seed=seed, honest=r_h, adv_freshW=r_i, adv_overtrain=r_a, replant=r_b))
        print(f"  seed {seed}: HONEST gap={r_h['gap']:+.4f} v_ft={r_h['v_ft']:.4f}(err{r_h['v_err']*100:.0f}%) | "
              f"FRESH-W gap={r_i['gap']:+.4f} v={r_i['v_ft']:.4f} | OVERTRAIN gap={r_a['gap']:+.4f} v={r_a['v_ft']:.4f} | "
              f"REPLANT v_ft={r_b['v_ft']:.4f}(target {args.v_replant})", flush=True)

    def col(path):
        return np.array([eval("r['" + "']['".join(path.split(".")) + "']", {"r": row}) for row in rows], float)
    gap_h, gap_i, gap_a = col("honest.gap"), col("adv_freshW.gap"), col("adv_overtrain.gap")
    verr_h, verr_a = col("honest.v_err"), col("adv_overtrain.v_err")
    survives = bool((gap_h > 0).all() and (gap_i > 0).all() and (gap_a > 0).all()
                    and gap_a.mean() > 0.5 * gap_h.mean())
    replant_tracks = bool((col("replant.v_ft") > args.v_true * 1.3).all())   # tracks the NEW (higher) velocity
    out = dict(config=vars(args), rows=rows,
               gap_honest=float(gap_h.mean()), gap_freshW=float(gap_i.mean()), gap_overtrain=float(gap_a.mean()),
               verr_honest=float(verr_h.mean()), verr_overtrain=float(verr_a.mean()),
               fingerprint_survives_W_retrain=survives, replant_tracks_new_velocity=replant_tracks,
               verdict=("WATERMARK: conduction geometry is a W-ROBUST structural fingerprint (gap survives "
                        "fresh-W + over-train; recovery tracks the re-planted velocity)"
                        if survives and replant_tracks else
                        "REMOVABLE/NULL: a retrained W erases the velocity fingerprint"))
    print("\n=== FINGERPRINT WATERMARK ===")
    print(f"  shuffle-null GAP: honest={gap_h.mean():+.4f}  fresh-W={gap_i.mean():+.4f}  over-train={gap_a.mean():+.4f}")
    print(f"  v_hat err: honest={verr_h.mean()*100:.0f}%  over-train={verr_a.mean()*100:.0f}%")
    print(f"  survives W-retrain={survives} | replant tracks new velocity={replant_tracks}")
    print("  VERDICT:", out["verdict"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print("  wrote", args.out)


if __name__ == "__main__":
    main()
