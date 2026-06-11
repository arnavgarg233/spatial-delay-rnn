"""
Self-contained Colab (T4) test of THE breakthrough lead: SOURCE LOCALIZATION / TDOA.

Two independent theory agents converged here. The claim: geometric (ordered) delays
make the population arrival-time field METRIC-CONSISTENT, so a trained network can
localize a source at a HELD-OUT position; the histogram-matched ENTRY-SHUFFLE breaks
metric consistency (triangle inequality violated ~58%), so no source-independent map
inverts it -- and crucially trainable recurrence CANNOT rebuild a global metric
property (unlike the per-task codes it re-sculpted in every prior null).

DECISIVE design choices (from the agents' corrections):
  * ENTRY-shuffle (scramble delay-matrix entries), NOT an index-permutation (a readout
    absorbs index-permutations perfectly -> that was the ITD/motion trap).
  * Target = a PHYSICAL COORDINATE (source position), held-out positions -> tests
    CONSISTENCY/generalization, not Fisher variance (variance framing favors the shuffle).
  * FULL TRAINABLE RECURRENCE + readout (not just a linear readout) -> the real test of
    whether trainable recurrence re-sculpts it like everything else.
  * no-delay control = chance (all arrivals equal -> no source info), confirms delays
    carry the localization.

Benefit = ordered localizes HELD-OUT sources far better than the entry-shuffle, at
matched train error, beyond seed noise, AND beyond the index-permutation sanity (which
should be ~0). Paste into a T4 Colab; downloads colab_localization_results.json.
"""
import json, math, time
import numpy as np
import torch, torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, "| torch", torch.__version__)


def positions(N, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(N, 2, generator=g)  # random 2D layout


def delay_from_positions(pos, v, maxd):
    d = torch.cdist(pos, pos)
    tau = torch.round(d / v).clamp(1, maxd).long()
    tau.fill_diagonal_(1)
    return tau


def make_delays(pos, v, maxd, control, seed):
    """control: 'ordered' (true geometry), 'shuffle_entry' (project's control: scramble
    the off-diagonal delay multiset -> breaks metric consistency), 'shuffle_index'
    (relabel units consistently -> a VALID alt-geometry a readout can absorb; sanity
    that should give ~0 gap), 'none' (all delays 1)."""
    tau = delay_from_positions(pos, v, maxd)
    N = tau.shape[0]
    if control == "ordered":
        return tau, pos
    if control == "none":
        return torch.ones_like(tau), pos
    if control == "shuffle_index":
        g = torch.Generator().manual_seed(7000 + seed)
        perm = torch.randperm(N, generator=g)
        return tau[perm][:, perm], pos[perm]          # KEY FIX: relabel targets too -> pure relabeling control
    if control == "shuffle_entry":
        g = torch.Generator().manual_seed(1000 + seed)
        iu = torch.triu_indices(N, N, offset=1)
        vals = tau[iu[0], iu[1]]
        vals = vals[torch.randperm(vals.numel(), generator=g)]
        out = torch.ones_like(tau)
        out[iu[0], iu[1]] = vals
        out[iu[1], iu[0]] = vals
        out.fill_diagonal_(1)
        return out, pos
    raise ValueError(control)


class LocRNN(nn.Module):
    """Delay-coupled RNN; receives a per-unit arrival pulse, regresses the 2D source
    position. Input weight is a FROZEN identity (unit r is driven by receiver-r's
    arrival) so the geometry is grounded and cannot be relabeled by training."""
    def __init__(self, N, tau, alpha=0.25, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.N, self.alpha = N, alpha
        self.register_buffer("tau", tau)
        self.maxd = int(tau.max().item())
        self.W = nn.Parameter(torch.randn(N, N) * (0.9 / math.sqrt(N)))
        self.b = nn.Parameter(torch.zeros(N))
        self.Wout = nn.Parameter(torch.randn(N, 2) / math.sqrt(N))
        self.bout = nn.Parameter(torch.zeros(2))
        masks = [(tau == d).float() for d in range(1, self.maxd + 1)]
        self.register_buffer("masks", torch.stack(masks) if masks else torch.zeros(0, N, N))

    def forward(self, X):
        # X: (T, B, N) per-unit arrival pulses
        T, B, N = X.shape
        dev = X.device
        hist = torch.zeros(self.maxd + 1, B, N, device=dev)
        phi = torch.tanh
        pooled = torch.zeros(B, N, device=dev)
        for t in range(T):
            rec = torch.zeros(B, N, device=dev)
            for di, dval in enumerate(range(1, self.maxd + 1)):
                m = self.masks[di]
                if m.sum() == 0:
                    continue
                rec = rec + phi(hist[dval]) @ (self.W * m).t()
            h = (1 - self.alpha) * hist[0] + self.alpha * (rec + X[t] + self.b)
            h = torch.clamp(h, -8, 8)
            hist = torch.roll(hist, 1, 0)
            hist[0] = h
            pooled = pooled + h
        pooled = pooled / T
        return pooled @ self.Wout + self.bout  # (B,2) predicted source position


def arrival_input(tau, src_idx, T, device):
    """Vectorized: for trial b with source src_idx[b], unit r gets a pulse at t=tau[src,r]."""
    src_idx = src_idx.to(tau.device)
    B = src_idx.shape[0]; N = tau.shape[0]
    arr = tau[src_idx].clamp(0, T - 1)                         # (B,N) arrival step per receiver
    X = torch.zeros(T, B, N, device=device)
    bidx = torch.arange(B, device=device).view(B, 1).expand(B, N)
    nidx = torch.arange(N, device=device).view(1, N).expand(B, N)
    X[arr, bidx, nidx] = 1.0
    return X


def run_control(pos, control, seed, N, v, maxd, steps=1000, lr=3e-3, device=DEV):
    tau, tgt_pos = make_delays(pos, v, maxd, control, seed)
    tau = tau.to(device)
    T = min(int(tau.max().item()) + 3, 40)
    g = torch.Generator().manual_seed(123 + seed)
    perm = torch.randperm(N, generator=g)
    train_src = perm[: int(0.75 * N)]
    test_src = perm[int(0.75 * N):]                  # HELD-OUT source units
    tgt_all = tgt_pos.to(device)                      # targets (relabeled for shuffle_index)
    m = LocRNN(N, tau, seed=seed).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    def batch(src_pool, B):
        idx = src_pool[torch.randint(0, len(src_pool), (B,))]
        X = arrival_input(tau, idx, T, device)
        return X, tgt_all[idx]
    for it in range(steps):
        opt.zero_grad()
        X, y = batch(train_src, 128)
        pred = m(X)
        loss = ((pred - y) ** 2).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if it % 250 == 0:
            print(f"      [{control}] step {it}/{steps} loss={loss.item():.3f}", flush=True)
    def err(src_pool):
        with torch.no_grad():
            X, y = batch(src_pool, min(256, 4 * len(src_pool)))
            pred = m(X)
            return float(((pred - y) ** 2).sum(-1).mean().sqrt())  # RMSE in position units
    return {"train_rmse": err(train_src), "heldout_rmse": err(test_src)}


def run_seed(seed, N=64, v=0.12, maxd=16, device=DEV):
    pos = positions(N, seed)
    out = {}
    for ctrl in ["ordered", "shuffle_entry", "shuffle_index", "none"]:
        out[ctrl] = run_control(pos, ctrl, seed, N, v, maxd, device=device)
        print(f"    seed {seed} {ctrl:14s} train={out[ctrl]['train_rmse']:.3f} "
              f"heldout={out[ctrl]['heldout_rmse']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    SEEDS = 6
    rows = []
    for s in range(SEEDS):
        print(f"=== seed {s + 1}/{SEEDS} ===", flush=True)
        rows.append(run_seed(s))

    def arr(c, k): return np.array([r[c][k] for r in rows])
    o_ho, e_ho = arr("ordered", "heldout_rmse"), arr("shuffle_entry", "heldout_rmse")
    o_tr, e_tr = arr("ordered", "train_rmse"), arr("shuffle_entry", "train_rmse")
    gap = e_ho - o_ho                       # entry-shuffle minus ordered held-out RMSE (>0 = ordered better)
    idx_gap = arr("shuffle_index", "heldout_rmse") - o_ho   # sanity: should be ~0
    def stats(a):
        m = float(a.mean()); sd = float(a.std() + 1e-9)
        return {"mean": m, "sd": sd, "t": m / (sd / math.sqrt(len(a))), "wins": int((a > 0).sum())}
    res = {
        "ordered_heldout_rmse": float(o_ho.mean()),
        "entryshuffle_heldout_rmse": float(e_ho.mean()),
        "train_rmse_ordered": float(o_tr.mean()), "train_rmse_entryshuffle": float(e_tr.mean()),
        "train_matched": bool(abs(o_tr.mean() - e_tr.mean()) < 0.5 * abs(gap.mean() + 1e-9)),
        "heldout_gap_entryshuf_minus_ordered": stats(gap),
        "index_perm_sanity_gap": stats(idx_gap),   # must be ~0 for the result to be non-circular
        "none_heldout_rmse": float(arr("none", "heldout_rmse").mean()),
        "device": DEV, "seeds": SEEDS, "minutes": round((time.time() - t0) / 60, 1),
    }
    g = res["heldout_gap_entryshuf_minus_ordered"]; idx = res["index_perm_sanity_gap"]
    # SIGNAL only if: ordered beats ENTRY-shuffle on held-out, train matched, AND the
    # index-perm sanity gap is ~0 (a consistent RELABEL generalizes like ordered -> proves
    # it's metric CONSISTENCY, not "any delay change", that entry-shuffle breaks).
    min_wins = max(1, math.ceil(0.85 * SEEDS))
    res["verdict"] = ("SIGNAL (needs adversarial verify)"
                      if g["t"] > 3 and g["wins"] >= min_wins and res["train_matched"]
                      and abs(idx["mean"]) < 0.3 * g["mean"]
                      else "marginal" if g["t"] > 1.5 else "null")
    print("\n=== SOURCE LOCALIZATION VERDICT ===")
    print("ordered held-out RMSE  : %.3f" % res["ordered_heldout_rmse"])
    print("entry-shuffle held-out : %.3f" % res["entryshuffle_heldout_rmse"])
    print("index-perm held-out    : %.3f  (sanity; should ~= ordered)" % (o_ho.mean() + idx["mean"]))
    print("none (chance) held-out : %.3f" % res["none_heldout_rmse"])
    print("gap entryshuf-ordered  : %.3f  t=%.2f  wins=%d/%d  train_matched=%s  idx_sanity=%.3f"
          % (g["mean"], g["t"], g["wins"], SEEDS, res["train_matched"], idx["mean"]))
    print("VERDICT:", res["verdict"])
    with open("colab_localization_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote colab_localization_results.json in", res["minutes"], "min")
    try:
        from google.colab import files
        files.download("colab_localization_results.json")
    except Exception:
        pass
