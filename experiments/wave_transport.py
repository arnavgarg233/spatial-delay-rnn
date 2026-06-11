"""FAIR wave gate v2: force a wave by blocking direct routing, then test metric-specificity.

Why v1 was unfair: full trainable W routed info directly (no wave needed; `none` solved it).
v2 fixes that:
  * LOCAL connectivity only (units couple to spatial neighbours within radius r) -> info MUST
    propagate hop-by-hop across the sheet; there is no long-range weight to route through.
  * NON-UNIFORM 2D geometry -> ordered (delay = round(d/v), constant PHYSICAL velocity) should
    carry a coherent front; `none` (uniform delay over varying distance -> varying physical
    speed) should smear it; entry (scrambled local delays) should smear it more.
  * TRANSPORT task: a length-K code is injected at the LEFT-edge units; a readout at the
    RIGHT-edge units must reconstruct it after the signal has crossed the sheet. With only
    local edges the code can only arrive by travelling -> a wave is required.

Reports, per condition (ordered | shuffle_entry | none):
  (1) transport bit-accuracy on held-out codes,
  (2) wave coherence = R^2 of [peak-activation time per unit] vs [physical distance from source].
GATE PASS if ordered transports better AND shows a more coherent physical-space wave than both.
"""
import json, math, time
import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {DEV} | torch {torch.__version__}", flush=True)


def geometry(N, R, v, maxd, control, seed):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(N, 2, generator=g)                       # non-uniform 2D
    D = torch.cdist(pos, pos)
    edge = (D < R) & (~torch.eye(N, dtype=bool))              # LOCAL connectivity
    tau = torch.round(D / v).clamp(1, maxd).long()
    if control == "ordered":
        td = tau
    elif control == "none":
        td = torch.ones(N, N, dtype=torch.long)
    elif control == "shuffle_entry":
        iu = torch.triu_indices(N, N, 1)
        em = edge[iu[0], iu[1]]
        ev = tau[iu[0], iu[1]].clone()
        idx = torch.where(em)[0]
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        ev2 = ev.clone(); ev2[idx] = ev[perm]                 # permute delays among edges only
        td = torch.zeros(N, N, dtype=torch.long)
        td[iu[0], iu[1]] = ev2; td[iu[1], iu[0]] = ev2
    else:
        raise ValueError(control)
    return pos, edge, td


class LocalWaveRNN(nn.Module):
    def __init__(self, pos, edge, td, K, src, tgt, alpha=0.35, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        N = pos.shape[0]
        self.N, self.alpha, self.K = N, alpha, K
        self.register_buffer("delay", (td * edge.long()))
        self.register_buffer("mask", edge.float())
        self.register_buffer("src", src); self.register_buffer("tgt", tgt)
        self.W = nn.Parameter(torch.randn(N, N) * (0.9 / math.sqrt(max(edge.float().sum(1).mean(), 1))))
        self.Wout = nn.Parameter(torch.randn(len(tgt), K) / math.sqrt(len(tgt)))
        self.bout = nn.Parameter(torch.zeros(K))

    def forward(self, code, T, return_H=False):
        # code: (B,K) in {-1,1} injected one bit/step into the LEFT-edge source units
        B, K = code.shape; N = self.N; dev = code.device
        H = torch.zeros(T + 1, B, N, device=dev)
        Wl = self.W * self.mask
        cols = torch.arange(N, device=dev).view(1, N).expand(N, N)
        for t in range(1, T + 1):
            ti = (t - 1 - self.delay).clamp(min=0)
            gathered = H[ti, :, cols]
            drive = torch.einsum("ji,jib->jb", Wl, gathered).t()
            inp = torch.zeros(B, N, device=dev)
            if t - 1 < K:
                inp[:, self.src] = code[:, t - 1:t]            # broadcast bit to all source units
            H[t] = (1 - self.alpha) * H[t - 1] + self.alpha * torch.tanh(drive + inp)
        read = H[T][:, self.tgt]                                # read the FAR edge
        logits = read @ self.Wout + self.bout
        return (logits, H) if return_H else logits


def wave_r2(H, pos, src_idx):
    env = H.abs().mean(1)[1:]                                  # (T,N)
    tpeak = env.argmax(0).float().cpu().numpy()
    src_pos = pos[src_idx].mean(0)
    dist = (pos - src_pos).norm(dim=1).cpu().numpy()           # PHYSICAL distance from source
    active = env.max(0).values.cpu().numpy() > 1e-3
    if active.sum() < 6: return 0.0
    x, y = dist[active], tpeak[active]
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None); pred = A @ c
    ss = ((y - pred) ** 2).sum(); st = ((y - y.mean()) ** 2).sum() + 1e-9
    return float(max(1 - ss / st, 0.0))


def run(control, seed, N, R, v, maxd, K, steps, device):
    pos, edge, td = geometry(N, R, v, maxd, control, seed)
    src = torch.where(pos[:, 0] < 0.25)[0]; tgt = torch.where(pos[:, 0] > 0.75)[0]
    if len(src) < 2 or len(tgt) < 2:
        return dict(acc=float("nan"), wave_r2=float("nan"))
    diam = int(td[edge].float().mean() * 6) + K + 5
    T = min(diam, 60)
    m = LocalWaveRNN(pos.to(device), edge.to(device), td.to(device), K,
                     src.to(device), tgt.to(device), seed=seed).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3); lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(50 + seed)
    for it in range(steps):
        code = (torch.randint(0, 2, (128, K), generator=g).float() * 2 - 1).to(device)
        loss = lossf(m(code, T), (code > 0).float())
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if it % 150 == 0:
            print(f"      [{control}] step {it}/{steps} loss={loss.item():.4f} (T={T})", flush=True)
    with torch.no_grad():
        code = (torch.randint(0, 2, (256, K), generator=torch.Generator().manual_seed(9)).float() * 2 - 1).to(device)
        logits, H = m(code, T, return_H=True)
        acc = ((logits > 0).float() == (code > 0).float()).float().mean().item()
        r2 = wave_r2(H, pos.to(device), src.to(device))
    print(f"    {control:14s} transport-acc={acc:.3f}  physical-wave_R2={r2:.3f}", flush=True)
    return dict(acc=acc, wave_r2=r2)


if __name__ == "__main__":
    t0 = time.time()
    SEEDS, N, R, v, maxd, K, steps = 3, 60, 0.22, 0.04, 20, 4, 500
    print(f"config: N={N} R={R}(local) v={v} K={K} steps={steps} seeds={SEEDS}", flush=True)
    res = {c: [] for c in ["ordered", "shuffle_entry", "none"]}
    for s in range(SEEDS):
        print(f"=== seed {s + 1}/{SEEDS} ===", flush=True)
        for c in res:
            res[c].append(run(c, s, N, R, v, maxd, K, steps, DEV))

    def mean(c, k): return float(np.nanmean([r[k] for r in res[c]]))
    summ = {c: {"acc": mean(c, "acc"), "wave_r2": mean(c, "wave_r2")} for c in res}
    oa, ea, na = summ["ordered"]["acc"], summ["shuffle_entry"]["acc"], summ["none"]["acc"]
    ow, ew, nw = summ["ordered"]["wave_r2"], summ["shuffle_entry"]["wave_r2"], summ["none"]["wave_r2"]
    needs_wave = na < oa - 0.03                                # did blocking routing make none fail?
    transports = oa > max(ea, na) + 0.03
    coherent = ow > 0.6 and ow > max(ew, nw) + 0.1
    verdict = ("GATE PASS: ordered transports better AND wave is metric-specific -> Idea C alive"
               if transports and coherent else
               "GATE FAIL: " + ("none still solves it (routing not blocked)" if not needs_wave
                                else "ordered no better / no coherent metric-specific wave"))
    out = {"config": dict(N=N, R=R, v=v, K=K, steps=steps, seeds=SEEDS), "summary": summ,
           "verdict": verdict, "minutes": round((time.time() - t0) / 60, 1), "device": DEV}
    print("\n=== WAVE TRANSPORT GATE (v2, fair) ===")
    for c in res:
        print(f"  {c:14s} transport-acc={summ[c]['acc']:.3f}  wave_R2={summ[c]['wave_r2']:.3f}")
    print("VERDICT:", verdict)
    with open("results/experiments/wave_transport_gate.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/experiments/wave_transport_gate.json in", out["minutes"], "min")
