"""Large-N scaling of the conduction-time economy (self-contained, CUDA/Colab).

Does distance-delay's conduction saving over a histogram-matched shuffled control
keep growing with N, up to N=512+? Needs only torch + numpy; runs on CUDA (MPS
CPU-fallback matrix_exp is too slow at large N).

Per N, at matched accuracy:
  cost(cond) = sum(|W| * tau_true) / sum(|W|)        (tau_true = true geometry)
  saving     = cost(shuffled) - cost(distance)       (>0 = weight on short routes)
  rel_saving = saving / cost(shuffled)
and the slope of saving / rel_saving vs log10(N).
"""
import json, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, "| torch:", torch.__version__)


def make_batch(B, K, cue, delay, resp, gen):
    T = cue + delay + resp
    y = torch.randint(0, K, (B,), generator=gen)
    x = torch.zeros(B, T, K)
    for b in range(B):
        x[b, :cue, y[b]] = 1.0
    x += 0.05 * torch.randn(B, T, K, generator=gen)
    return x, y, T


def geometry(N, gen):
    pos = torch.rand(N, 3, generator=gen)
    return torch.cdist(pos, pos)


def delay_matrix(D, v, max_delay, control, gen):
    tau = torch.clamp(torch.round(D / v), 1, max_delay).long()
    tau.fill_diagonal_(1)
    if control == "shuffled":
        N = tau.shape[0]
        iu = torch.triu_indices(N, N, 1)
        vals = tau[iu[0], iu[1]]
        vals = vals[torch.randperm(vals.numel(), generator=gen)]
        t2 = tau.clone(); t2[iu[0], iu[1]] = vals; t2[iu[1], iu[0]] = vals; t2.fill_diagonal_(1)
        return t2
    return tau


class DelayRNN(nn.Module):
    def __init__(self, N, K, tau, alpha=0.2):
        super().__init__()
        self.N, self.alpha = N, alpha
        self.W = nn.Parameter(0.6 * torch.randn(N, N) / math.sqrt(N))
        self.Win = nn.Parameter(torch.randn(K, N) / math.sqrt(K))
        self.Wout = nn.Parameter(torch.randn(N, K) / math.sqrt(N))
        self.b = nn.Parameter(torch.zeros(N))
        self.register_buffer("tau", tau)
        self.delays = sorted(set(int(t) for t in tau.unique().tolist()))
        self.register_buffer("mask_stack", torch.stack([(tau == d).float() for d in self.delays]))

    def forward(self, x):
        B, T, K = x.shape; N = self.N
        h = torch.zeros(B, N, device=x.device)
        hist = [torch.zeros(B, N, device=x.device) for _ in range(max(self.delays) + 1)]
        outs = []
        for t in range(T):
            rec = torch.zeros(B, N, device=x.device)
            for gi, d in enumerate(self.delays):
                rec = rec + hist[-d] @ (self.W * self.mask_stack[gi]).t()
            h = (1 - self.alpha) * h + self.alpha * (x[:, t, :] @ self.Win + rec + self.b)
            r = torch.tanh(h); hist.append(r); hist.pop(0)
            outs.append(r @ self.Wout)
        return torch.stack(outs, 1)


def comm_distance_reg(W, D):
    A = W.abs(); deg = A.sum(1).clamp_min(1e-6); Dm = torch.diag(deg.pow(-0.5))
    C = torch.matrix_exp(Dm @ A @ Dm)
    return (C * D * A).sum() / W.numel()


def train_and_cost(N, K, tau_train, tau_true, D, steps, lr, reg, seed, cue=2, delay=8, resp=2, B=64):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    m = DelayRNN(N, K, tau_train.to(DEV)).to(DEV); Dd = D.to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for it in range(steps):
        x, y, T = make_batch(B, K, cue, delay, resp, g); x, y = x.to(DEV), y.to(DEV)
        loss = F.cross_entropy(m(x)[:, -resp:].mean(1), y) + reg * comm_distance_reg(m.W, Dd)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    with torch.no_grad():
        x, y, T = make_batch(512, K, cue, delay, resp, g); x, y = x.to(DEV), y.to(DEV)
        acc = (m(x)[:, -resp:].mean(1).argmax(1) == y).float().mean().item()
        Wabs = m.W.detach().abs(); tt = tau_true.to(DEV).float()
        cost = ((Wabs * tt).sum() / Wabs.sum().clamp_min(1e-9)).item()
    return acc, cost


def run(N_LIST=(64, 128, 256, 512), SEEDS=3, STEPS=350, K=4, V=0.10, MAXD=20, LR=2e-3, REG=0.01, DELAY=8):
    out = {}
    for N in N_LIST:
        gN = torch.Generator().manual_seed(7000 + N)
        D = geometry(N, gN)
        tau_true = delay_matrix(D, V, MAXD, "distance", torch.Generator().manual_seed(1))
        savings, rels, accs_d, accs_s = [], [], [], []
        for sd in range(SEEDS):
            tau_d = delay_matrix(D, V, MAXD, "distance", torch.Generator().manual_seed(1))
            tau_s = delay_matrix(D, V, MAXD, "shuffled", torch.Generator().manual_seed(100 + sd))
            ad, cd = train_and_cost(N, K, tau_d, tau_true, D, STEPS, LR, REG, sd, delay=DELAY)
            as_, cs = train_and_cost(N, K, tau_s, tau_true, D, STEPS, LR, REG, sd, delay=DELAY)
            savings.append(cs - cd); rels.append((cs - cd) / max(cs, 1e-9))
            accs_d.append(ad); accs_s.append(as_)
            print(f"  N={N:4d} seed{sd}: dist cost {cd:.3f} (acc {ad:.3f}) | shuf cost {cs:.3f} (acc {as_:.3f}) "
                  f"| saving {cs-cd:+.3f}", flush=True)
        out[str(N)] = {"saving_mean": float(np.mean(savings)), "saving_std": float(np.std(savings)),
                       "rel_mean": float(np.mean(rels)), "acc_dist": float(np.mean(accs_d)),
                       "acc_shuf": float(np.mean(accs_s)),
                       "acc_matched": bool(abs(np.mean(accs_d) - np.mean(accs_s)) < 0.03)}
        print(f"  -> N={N}: saving={np.mean(savings):+.3f}±{np.std(savings):.3f}  rel={np.mean(rels):.3f}  "
              f"acc_matched={out[str(N)]['acc_matched']}\n", flush=True)
    Ns = np.array([int(n) for n in out]); rel = np.array([out[str(n)]["rel_mean"] for n in Ns])
    sav = np.array([out[str(n)]["saving_mean"] for n in Ns])
    slope_rel = float(np.polyfit(np.log10(Ns), rel, 1)[0]) if len(Ns) > 1 else 0.0
    slope_sav = float(np.polyfit(np.log10(Ns), sav, 1)[0]) if len(Ns) > 1 else 0.0
    out["SUMMARY"] = {"N_list": [int(n) for n in Ns], "rel_saving": [round(float(r), 4) for r in rel],
                      "abs_saving": [round(float(s), 4) for s in sav],
                      "rel_slope_vs_log10N": round(slope_rel, 4), "abs_slope_vs_log10N": round(slope_sav, 4),
                      "law_holds_growing": bool(slope_rel > 0)}
    print("=" * 60)
    print(f"SCALING LAW: rel_saving vs N = {dict(zip([int(n) for n in Ns], [round(float(r),3) for r in rel]))}")
    print(f"  rel slope vs log10(N) = {slope_rel:+.4f}  -> saving {'GROWS' if slope_rel>0 else 'shrinks'} with scale")
    # Write to results/economy/ when run from the repo; fall back to CWD (e.g. a
    # standalone Colab cell with no repo layout).
    from pathlib import Path
    try:
        outdir = Path(__file__).resolve().parents[2] / "results" / "economy"
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / "scaling_large_n.json"
    except (NameError, IndexError):
        outpath = Path("scaling_large_n.json")
    outpath.write_text(json.dumps(out, indent=1))
    print("\nPASTE THIS BACK:")
    print(json.dumps(out["SUMMARY"], indent=1))
    return out


if __name__ == "__main__":
    t0 = time.time()
    # Full sweep to N=1024 (~2-3 hr on a T4). For a quick run drop to:
    #   run(N_LIST=(64,128,256,512), SEEDS=3, STEPS=350)
    run(N_LIST=(64, 128, 256, 512, 768, 1024), SEEDS=3, STEPS=350)
    print(f"\ntotal {time.time()-t0:.0f}s")
