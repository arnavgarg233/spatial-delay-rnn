"""GATE (Idea C, Wave-RNN bridge): does a coherent traveling wave EMERGE in a trained
distance-delay net on a task that NEEDS wave-like memory? And is it metric-specific?

Task (wave-needing): a length-K binary sequence is injected ONE bit/step into the single
edge unit (unit 0). After the input stops the net runs to time T; a readout must reconstruct
the WHOLE sequence from the final state h(T). A traveling wave stores the sequence as a
spatial pattern (bit j sits at position ~ velocity*(T-j)), so reconstruction NEEDS a
coherent constant-velocity front. Scrambled/absent delays cannot lay the bits out in space.

Conditions: ordered (metric tau=round(d/v)) | shuffle_entry (same delay histogram, triangle
broken) | none (all delays=1). We report, per condition:
  (1) WAVE coherence  = R^2 of [peak-activation time at unit n]  vs  [position of unit n]
      (a clean traveling front => arrival time is linear in position => R^2 -> 1)
  (2) task bit-accuracy on held-out sequences.
GATE PASSES if ordered shows a coherent wave (R^2 high) that entry-shuffle/none do NOT.
"""
import json, math, time
import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {DEV} | torch {torch.__version__}", flush=True)


def make_tau(N, v, maxd, control, seed):
    g = torch.Generator().manual_seed(seed)
    posx = torch.sort(torch.rand(N, generator=g)).values        # 1-D line positions
    D = (posx[:, None] - posx[None]).abs()
    tau = torch.round(D / v).clamp(0, maxd).long()
    if control == "ordered":
        return tau, posx
    if control == "none":
        t = torch.ones(N, N, dtype=torch.long); t.fill_diagonal_(0)
        return t, posx
    if control == "shuffle_entry":
        iu = torch.triu_indices(N, N, offset=1)
        vals = tau[iu[0], iu[1]].clone()
        vals = vals[torch.randperm(vals.numel(), generator=g)]
        t = torch.zeros_like(tau)
        t[iu[0], iu[1]] = vals; t[iu[1], iu[0]] = vals
        return t, posx
    raise ValueError(control)


class WaveRNN(nn.Module):
    def __init__(self, N, tau, K, alpha=0.3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.N, self.alpha, self.K = N, alpha, K
        self.register_buffer("tau", tau)
        self.maxd = int(tau.max().item())
        self.W = nn.Parameter(torch.randn(N, N) * (0.9 / math.sqrt(N)))
        self.b = nn.Parameter(torch.zeros(N))
        self.Wout = nn.Parameter(torch.randn(N, K) / math.sqrt(N))
        self.bout = nn.Parameter(torch.zeros(K))

    def forward(self, seq, T, return_H=False):
        # seq: (B,K) in {-1,+1}; inject bit j at unit 0 at step j
        B = seq.shape[0]; N, K = self.N, self.K
        dev = seq.device
        H = torch.zeros(T + 1, B, N, device=dev)
        cols = torch.arange(N, device=dev).view(1, N).expand(N, N)   # source i index
        for t in range(1, T + 1):
            ti = (t - 1 - self.tau).clamp(min=0)                     # (N,N) [j,i]
            gathered = H[ti, :, cols]                                # (N,N,B)
            drive = torch.einsum("ji,jib->jb", self.W, gathered).t() # (B,N)
            inp = torch.zeros(B, N, device=dev)
            if t - 1 < K:
                inp[:, 0] = seq[:, t - 1]
            pre = drive + inp + self.b
            H[t] = (1 - self.alpha) * H[t - 1] + self.alpha * torch.tanh(pre)
        logits = H[T] @ self.Wout + self.bout
        return (logits, H) if return_H else logits


def wave_coherence(H, posx):
    """H: (T+1,B,N) -> R^2 of [peak-activation time per unit] vs [unit position]."""
    env = H.abs().mean(dim=1)                 # (T+1, N) activity envelope
    env = env[1:]                             # drop t=0
    tpeak = env.argmax(dim=0).float().cpu().numpy()   # (N,) time of peak per unit
    x = posx.cpu().numpy()
    # only units that actually activate (avoid dead units flattening the fit)
    active = env.max(dim=0).values.cpu().numpy() > 1e-3
    if active.sum() < 5:
        return 0.0, 0.0
    xa, ta = x[active], tpeak[active]
    A = np.vstack([xa, np.ones_like(xa)]).T
    coef, *_ = np.linalg.lstsq(A, ta, rcond=None)
    pred = A @ coef
    ss_res = ((ta - pred) ** 2).sum(); ss_tot = ((ta - ta.mean()) ** 2).sum() + 1e-9
    r2 = 1 - ss_res / ss_tot
    return float(max(r2, 0.0)), float(coef[0])     # R^2, front slope (steps per unit-position)


def run(control, seed, N, v, maxd, K, T, steps, device):
    tau, posx = make_tau(N, v, maxd, control, seed)
    m = WaveRNN(N, tau.to(device), K, seed=seed).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(100 + seed)
    for it in range(steps):
        seq = (torch.randint(0, 2, (128, K), generator=g).float() * 2 - 1).to(device)
        logits = m(seq, T)
        loss = lossf(logits, (seq > 0).float())
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if it % 100 == 0:
            print(f"      [{control}] step {it}/{steps} loss={loss.item():.4f}", flush=True)
    # eval on held-out sequences
    with torch.no_grad():
        seq = (torch.randint(0, 2, (256, K), generator=torch.Generator().manual_seed(999)).float() * 2 - 1).to(device)
        logits, H = m(seq, T, return_H=True)
        acc = ((logits > 0).float() == (seq > 0).float()).float().mean().item()
        r2, slope = wave_coherence(H, posx)
    print(f"    {control:14s} bit-acc={acc:.3f}  wave_R2={r2:.3f}  front_slope={slope:.2f}", flush=True)
    return dict(acc=acc, wave_r2=r2, front_slope=slope)


if __name__ == "__main__":
    t0 = time.time()
    SEEDS, N, v, maxd, K, T, steps = 2, 32, 0.03, 33, 4, 22, 400
    print(f"config: N={N} v={v} K={K} T={T} steps={steps} seeds={SEEDS}", flush=True)
    res = {c: [] for c in ["ordered", "shuffle_entry", "none"]}
    for s in range(SEEDS):
        print(f"=== seed {s + 1}/{SEEDS} ===", flush=True)
        for c in ["ordered", "shuffle_entry", "none"]:
            res[c].append(run(c, s, N, v, maxd, K, T, steps, DEV))

    def mean(c, k): return float(np.mean([r[k] for r in res[c]]))
    summ = {c: {"wave_r2": mean(c, "wave_r2"), "acc": mean(c, "acc"),
                "front_slope": mean(c, "front_slope")} for c in res}
    ord_r2, ent_r2, non_r2 = summ["ordered"]["wave_r2"], summ["shuffle_entry"]["wave_r2"], summ["none"]["wave_r2"]
    wave_emerges = ord_r2 > 0.6
    metric_specific = ord_r2 > max(ent_r2, non_r2) + 0.15
    verdict = ("GATE PASS: coherent wave emerges AND is metric-specific -> pursue Idea C"
               if wave_emerges and metric_specific else
               "GATE FAIL: " + ("no coherent wave in ordered" if not wave_emerges
                                 else "wave not metric-specific (shuffle/none match it)"))
    out = {"config": dict(N=N, v=v, K=K, T=T, steps=steps, seeds=SEEDS),
           "summary": summ, "ordered_wave_r2": ord_r2, "entry_wave_r2": ent_r2, "none_wave_r2": non_r2,
           "verdict": verdict, "minutes": round((time.time() - t0) / 60, 1), "device": DEV}
    print("\n=== WAVE GATE ===")
    for c in res:
        print(f"  {c:14s} wave_R2={summ[c]['wave_r2']:.3f}  bit-acc={summ[c]['acc']:.3f}")
    print("VERDICT:", verdict)
    with open("results/experiments/wave_memory_gate.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/experiments/wave_memory_gate.json in", out["minutes"], "min")
