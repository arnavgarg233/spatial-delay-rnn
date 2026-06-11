"""Confound-free temporal-composition test (rank-based).

Earlier temporal tests were confounded:
  - temporal twin: per-hop NRMSE blows up where target variance -> 0 (variance trap)
  - graph-metric : within-hop comparison; ordered's metric makes same-hop pairs equidistant
                   (CV collapses 0.37->0.08), so "which is closer" is ill-posed FOR ordered.
Both artifacts SUPPRESS ordered. This version removes them:
  * metric = SPEARMAN rank-correlation of predicted-vs-true conduction time (immune to the
    absolute-spread / CV-collapse confound),
  * evaluated on a POOL of held-out long-range pairs (hops 3-5 together -> wide true-SP range,
    real discriminability) from HELD-OUT SOURCE nodes the net never trained on,
  * the net is trained only on SHORT-range conduction times (hop<=2); long-range must be
    inferred by COMPOSING legs -> the triangle inequality is exactly what should let ordered
    generalize and entry-shuffle (triangle broken) fail.

Conditions: ordered | shuffle_entry | shuffle_index | none. Reuses graph_metric's graph
machinery (radius graph, make_graph, geodesic) so the controls are identical to the audited run.
"""
import sys, json, math, time
sys.path.insert(0, "experiments")
import numpy as np
import torch
import torch.nn as nn
import graph_metric as gm

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {DEV} | torch {torch.__version__}", flush=True)


def hop_matrix(W, BIG):
    H = torch.where(W < BIG - 1, 1.0, float("inf")); H.fill_diagonal_(0); N = H.shape[0]
    for k in range(N):
        H = torch.minimum(H, H[:, k:k + 1] + H[k:k + 1, :])
    return H


def spearman(a, b):
    a = torch.as_tensor(a, dtype=torch.float); b = torch.as_tensor(b, dtype=torch.float)
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-9); rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


class CondRNN(nn.Module):
    """Delay-coupled RNN on the sparse graph: a pulse from source s propagates along edges
    with their conduction delays; a per-unit head reads predicted SP(unit, s) from each unit's
    activity trajectory summary. Recurrent weights are learnable but masked to real edges."""
    def __init__(self, W, BIG, T, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        N = W.shape[0]
        self.N, self.T = N, T
        edge = (W < BIG - 1) & (~torch.eye(N, dtype=bool, device=W.device))
        self.register_buffer("delay", torch.where(edge, W, torch.zeros_like(W)).long())
        self.register_buffer("mask", edge.float())
        self.Wr = nn.Parameter(torch.randn(N, N) * (0.9 / math.sqrt(max(edge.sum(1).float().mean(), 1))))
        self.alpha = 0.3
        self.head = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, src):                                 # src: (B,) source unit indices
        N, T, B = self.N, self.T, src.shape[0]
        dev = src.device
        Hbuf = torch.zeros(T + 1, B, N, device=dev)
        Wr = self.Wr * self.mask
        cols = torch.arange(N, device=dev).view(1, N).expand(N, N)
        bidx = torch.arange(B, device=dev)
        for t in range(1, T + 1):
            ti = (t - 1 - self.delay).clamp(min=0)
            gathered = Hbuf[ti, :, cols]                    # (N,N,B)
            drive = torch.einsum("ji,jib->jb", Wr, gathered).t()   # (B,N)
            inp = torch.zeros(B, N, device=dev)
            if t == 1:
                inp[bidx, src] = 1.0                        # pulse at source
            Hbuf[t] = (1 - self.alpha) * Hbuf[t - 1] + self.alpha * torch.tanh(drive + inp)
        H = Hbuf[1:]                                        # (T,B,N)
        peak_t = H.abs().argmax(0).float() / T             # arrival-time feature (B,N)
        feat = torch.stack([H[-1], H.mean(0), peak_t], dim=-1)   # (B,N,3)
        return self.head(feat).squeeze(-1)                 # (B,N) predicted SP(unit, src)


def run(control, seed, N, R, v, maxd, steps, device):
    pos = gm.positions(N, seed)
    W, BIG, _ = gm.make_graph(pos, R, v, maxd, control, seed)
    SP, HOP = gm.geodesic(W.float(), float(BIG))
    reach0 = (HOP >= 1) & torch.isfinite(HOP)
    T = min(int(SP[reach0].max().item()) + 4, 60)
    Wd = W.to(device); m = CondRNN(Wd, float(BIG), T, seed=seed).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(7 + seed)
    srcs = torch.randperm(N, generator=g)
    train_s, test_s = srcs[: int(0.7 * N)], srcs[int(0.7 * N):]
    SPd = SP.to(device)
    reach = (HOP >= 1) & torch.isfinite(HOP)
    short = reach & (HOP <= 2)                             # train targets
    longm = reach & (HOP >= 3)                             # held-out test targets
    for it in range(steps):
        b = train_s[torch.randint(0, len(train_s), (min(32, len(train_s)),))]
        pred = m(b.to(device))
        tgt_mask = short[b].to(device)
        tgt = SPd[b]
        loss = (((pred - tgt) ** 2) * tgt_mask).sum() / tgt_mask.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if it % 100 == 0:
            print(f"      [{control}] step {it}/{steps} loss={loss.item():.4f}", flush=True)
    # eval: pooled long-range pairs from HELD-OUT sources, Spearman(pred, true)
    with torch.no_grad():
        pred = m(test_s.to(device)).cpu()
    lm = longm[test_s]
    p, t = pred[lm], SP[test_s][lm]
    rho = spearman(p, t)
    # also report short-range Spearman (sanity: should be high for all)
    sm = short[test_s]
    rho_short = spearman(m(test_s.to(device)).cpu()[sm], SP[test_s][sm]) if sm.sum() > 5 else float("nan")
    print(f"    {control:14s} long-range Spearman={rho:.3f}  (short-range={rho_short:.3f}, n_long={int(lm.sum())})", flush=True)
    return dict(rho_long=rho, rho_short=rho_short, n_long=int(lm.sum()))


if __name__ == "__main__":
    t0 = time.time()
    SEEDS, N, R, v, maxd, steps = 5, 100, 0.24, 0.04, 28, 1200
    print(f"config: N={N} R={R} v={v} steps={steps} seeds={SEEDS}", flush=True)
    res = {c: [] for c in ["ordered", "shuffle_entry", "shuffle_index", "none"]}
    for s in range(SEEDS):
        print(f"=== seed {s + 1}/{SEEDS} ===", flush=True)
        for c in res:
            res[c].append(run(c, s, N, R, v, maxd, steps, DEV))

    def arr(c): return np.array([r["rho_long"] for r in res[c]])
    ro, re_, ri, rn = arr("ordered"), arr("shuffle_entry"), arr("shuffle_index"), arr("none")
    gap = ro - re_
    t = float(gap.mean() / (gap.std(ddof=1) / math.sqrt(len(gap)) + 1e-9))
    wins = int((gap > 0).sum())
    idx_ok = abs(float(ro.mean() - ri.mean())) < 0.1
    signal = gap.mean() > 0.05 and wins >= math.ceil(0.8 * SEEDS) and idx_ok
    verdict = ("SIGNAL: ordered composes long-range conduction time better (rank-robust)"
               if signal else "null (no rank advantage for metric delays)")
    out = {"config": dict(N=N, R=R, v=v, steps=steps, seeds=SEEDS),
           "ordered_rho_long": float(ro.mean()), "entry_rho_long": float(re_.mean()),
           "index_rho_long": float(ri.mean()), "none_rho_long": float(rn.mean()),
           "gap_mean": float(gap.mean()), "gap_t": t, "wins": wins, "seeds": SEEDS,
           "index_tracks_ordered": idx_ok, "verdict": verdict,
           "minutes": round((time.time() - t0) / 60, 1), "device": DEV}
    print("\n=== TEMPORAL (rank-based, confound-free) ===")
    print(f"  ordered={ro.mean():.3f}  entry={re_.mean():.3f}  index={ri.mean():.3f}  none={rn.mean():.3f}")
    print(f"  gap(ord-ent)={gap.mean():+.3f}  t={t:.2f}  wins={wins}/{SEEDS}  idx_ok={idx_ok}")
    print("VERDICT:", verdict)
    with open("results/experiments/temporal_rank.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/experiments/temporal_rank.json in", out["minutes"], "min")
