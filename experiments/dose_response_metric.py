"""DOSE-RESPONSE LAW: held-out localization error as a continuous function of delay-graph
metric-INconsistency (triangle-violation rate).

Upgrades the binary ordered-vs-shuffle t-test into a predictive curve: partially entry-shuffle
k% of the off-diagonal delay entries (k = 0,10,25,50,75,100%), measure the resulting triangle-
violation rate epsilon and the held-out source-localization RMSE. If RMSE rises smoothly and
monotonically with epsilon (with an identifiable collapse knee), that IS the empirical converse
the decodability theorem needs -- the metric-distortion dose axis, untested so far.

Reuses experiments/localization.py verbatim (LocRNN, arrival_input, positions, delay_from_positions).
"""
import sys, json, time, math
sys.path.insert(0, "experiments")
import numpy as np
import torch
import localization as loc

DEV = loc.DEV


def partial_shuffle(tau, frac, seed):
    """Scramble a FRACTION `frac` of the off-diagonal delay entries (symmetric); the rest stay
    in their true geometric place. frac=0 -> ordered, frac=1 -> full entry-shuffle."""
    N = tau.shape[0]
    g = torch.Generator().manual_seed(1000 + seed)
    iu = torch.triu_indices(N, N, 1)
    vals = tau[iu[0], iu[1]].clone()
    n = vals.numel()
    k = int(round(frac * n))
    if k >= 2:
        sel = torch.randperm(n, generator=g)[:k]
        vals[sel] = vals[sel][torch.randperm(k, generator=g)]
    out = torch.ones_like(tau)
    out[iu[0], iu[1]] = vals
    out[iu[1], iu[0]] = vals
    out.fill_diagonal_(1)
    return out


def tri_viol_rate(tau, n=8000, seed=0):
    N = tau.shape[0]
    g = torch.Generator().manual_seed(seed)
    i, j, k = (torch.randint(0, N, (n,), generator=g) for _ in range(3))
    ok = (i != j) & (j != k) & (i != k)
    i, j, k = i[ok], j[ok], k[ok]
    lhs = tau[i, k].float(); rhs = (tau[i, j] + tau[j, k]).float()
    return float((lhs > rhs + 1e-6).float().mean())


def run_dose(pos, tau, seed, N, steps, device):
    tau = tau.to(device)
    T = min(int(tau.max().item()) + 3, 40)
    g = torch.Generator().manual_seed(123 + seed)
    perm = torch.randperm(N, generator=g)
    train_src, test_src = perm[: int(0.75 * N)], perm[int(0.75 * N):]
    tgt = pos.to(device)
    m = loc.LocRNN(N, tau, seed=seed).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    def batch(pool, B):
        idx = pool[torch.randint(0, len(pool), (B,))]
        return loc.arrival_input(tau, idx, T, device), tgt[idx]
    for it in range(steps):
        opt.zero_grad()
        X, y = batch(train_src, 128)
        loss = ((m(X) - y) ** 2).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    def err(pool):
        with torch.no_grad():
            X, y = batch(pool, min(256, 4 * len(pool)))
            return float(((m(X) - y) ** 2).sum(-1).mean().sqrt())
    return err(train_src), err(test_src)


if __name__ == "__main__":
    t0 = time.time()
    SEEDS, N, v, maxd, steps = 5, 64, 0.12, 16, 700
    DOSES = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    print(f"dose-response: N={N} v={v} steps={steps} seeds={SEEDS} doses={DOSES} dev={DEV}", flush=True)
    rec = {d: {"triViol": [], "heldout": [], "train": []} for d in DOSES}
    for s in range(SEEDS):
        pos = loc.positions(N, s)
        tau_ord = loc.delay_from_positions(pos, v, maxd)
        for d in DOSES:
            tau = partial_shuffle(tau_ord, d, s)
            tv = tri_viol_rate(tau, seed=s)
            tr, ho = run_dose(pos, tau, s, N, steps, DEV)
            rec[d]["triViol"].append(tv); rec[d]["heldout"].append(ho); rec[d]["train"].append(tr)
            print(f"  seed{s} dose{d:.2f}  triViol={tv:.3f}  train={tr:.3f}  heldout={ho:.3f}", flush=True)

    summary = []
    for d in DOSES:
        summary.append({"dose": d, "triViol": float(np.mean(rec[d]["triViol"])),
                        "heldout_mean": float(np.mean(rec[d]["heldout"])),
                        "heldout_sd": float(np.std(rec[d]["heldout"])),
                        "train_mean": float(np.mean(rec[d]["train"]))})
    hos = [s["heldout_mean"] for s in summary]
    tvs = [s["triViol"] for s in summary]
    monotone = all(hos[i] <= hos[i + 1] + 1e-6 for i in range(len(hos) - 1))
    r = float(np.corrcoef(tvs, hos)[0, 1])
    # knee = largest jump between consecutive doses
    jumps = [hos[i + 1] - hos[i] for i in range(len(hos) - 1)]
    knee_at = DOSES[int(np.argmax(jumps)) + 1]
    out = {"config": {"N": N, "v": v, "maxd": maxd, "steps": steps, "seeds": SEEDS},
           "summary": summary, "monotone": monotone, "corr_heldout_vs_triViol": r,
           "knee_dose": knee_at, "minutes": round((time.time() - t0) / 60, 1),
           "verdict": "DOSE-RESPONSE LAW (smooth monotone collapse)" if (monotone and r > 0.85)
                      else ("monotone but noisy" if monotone else "non-monotone")}
    print("\n=== DOSE-RESPONSE (held-out RMSE vs triangle-violation) ===")
    for srow in summary:
        print(f"  dose {srow['dose']:.2f}  triViol {srow['triViol']:.3f}  ->  heldout {srow['heldout_mean']:.3f} ± {srow['heldout_sd']:.3f}")
    print(f"  corr(heldout, triViol)={r:.3f}  monotone={monotone}  knee@dose={knee_at}")
    print("VERDICT:", out["verdict"])
    json.dump(out, open("results/experiments/dose_response_metric.json", "w"), indent=2)
    print("wrote results/experiments/dose_response_metric.json in", out["minutes"], "min")
