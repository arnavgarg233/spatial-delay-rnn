"""
Self-contained Colab (T4) test of THE breakthrough lead via a SPATIAL-ORDER RELAY /
ROUTING task. The claim: geometric (ordered) delays make the propagation field
METRIC-CONSISTENT, so a baton fired from region A arrives at B before C exactly when
B is spatially closer to A than C (d(A,B) < d(A,C)). A trained, wiring-cost-penalized
network can therefore learn a POSITION-BASED ordering rule that GENERALIZES to held-out
A/B/C triples. The histogram-matched ENTRY-SHUFFLE preserves the delay multiset but
gives effective propagation speeds huge jitter (effective-speed CV ~0.65 vs ~0.07 for
ordered), so arrival order no longer tracks distance -> no source/triple-independent
rule inverts it, and trainable recurrence CANNOT rebuild a GLOBAL metric property
(unlike the per-task codes it re-sculpted in every prior null incl. ITD).

DECISIVE design choices (the same corrections that the localization template encodes):
  * ENTRY-shuffle (symmetric permutation of the off-diagonal delay multiset), NOT an
    index-permutation (a readout absorbs index-permutations perfectly -> the ITD/motion
    trap). We ALSO run the index-perm as a SANITY that must come out ~0.
  * Target = a PHYSICAL-COORDINATE / GLOBAL-metric property: "which neighbor (B or C) is
    spatially closer to the source A", read out from arrival timing. Held-out triples ->
    tests CONSISTENCY/generalization, not per-trial variance.
  * FULL TRAINABLE RECURRENCE + readout -> the real test of whether trainable recurrence
    re-sculpts it. W is L1/wiring-cost penalized (sparse) -- the regime where geometry is
    load-bearing (coupling mass must sit on the delay metric), per the reservoir-economy law.
  * no-delay control = chance (all arrivals equal -> no order info), confirms delays carry
    the routing.

Benefit = ordered gets the HELD-OUT A/B/C ordering right far more often than the
entry-shuffle, at MATCHED train accuracy, beyond seed noise, AND beyond the index-perm
sanity (which should be ~0). Paste into a T4 Colab; downloads colab_relay_results.json.
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
    """control: 'ordered' (true geometry -> arrival order tracks distance),
    'shuffle_entry' (project's control: symmetric permutation of the off-diagonal delay
    multiset -> breaks metric consistency / effective-speed jitter), 'shuffle_index'
    (relabel units consistently -> a VALID alt-geometry a readout can absorb; sanity that
    should give ~0 gap), 'none' (all delays 1 -> simultaneous arrival, no order info)."""
    tau = delay_from_positions(pos, v, maxd)
    N = tau.shape[0]
    if control == "ordered":
        return tau, pos
    if control == "none":
        return torch.ones_like(tau), pos
    if control == "shuffle_index":
        g = torch.Generator().manual_seed(7000 + seed)
        perm = torch.randperm(N, generator=g)
        return tau[perm][:, perm], pos[perm]          # KEY FIX: relabel positions (the distance/label basis) too -> pure relabeling control
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


def effective_speed_cv(tau, pos):
    """Diagnostic: CV of effective propagation speed d_ij / tau_ij over off-diagonal
    pairs. Ordered ~0.07 (speed ~constant); entry-shuffle ~0.65 (latency jitter)."""
    d = torch.cdist(pos, pos)
    N = tau.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    sp = (d[iu[0], iu[1]] / tau[iu[0], iu[1]].float()).numpy()
    return float(sp.std() / (sp.mean() + 1e-9))


class RelayRNN(nn.Module):
    """Delay-coupled, wiring-cost-penalized RNN. A single source unit A is pulsed; the
    baton propagates through W along the delay graph. The network must report whether
    neighbor B or neighbor C is the spatially-closer one to A, read from arrival timing.
    Input weight is a FROZEN identity (unit r is driven by its own arrival pulse) so the
    geometry is grounded and cannot be relabeled by training; the readout reads the
    population trace at the two probed neighbors B,C."""
    def __init__(self, N, tau, alpha=0.25, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.N, self.alpha = N, alpha
        self.register_buffer("tau", tau)
        self.maxd = int(tau.max().item())
        self.W = nn.Parameter(torch.randn(N, N) * (0.9 / math.sqrt(N)))
        self.b = nn.Parameter(torch.zeros(N))
        # readout consumes the time-resolved trace at B and C (2 units x T_read steps);
        # built lazily in forward via a learned linear map over pooled per-unit features.
        self.Wout = nn.Parameter(torch.randn(2, 1) / math.sqrt(2))
        self.bout = nn.Parameter(torch.zeros(1))
        masks = [(tau == d).float() for d in range(1, self.maxd + 1)]
        self.register_buffer("masks", torch.stack(masks) if masks else torch.zeros(0, N, N))

    def propagate(self, X):
        """X: (T,B,N) per-unit pulse input. Returns full hidden trace H: (T,B,N)."""
        T, B, N = X.shape
        dev = X.device
        hist = torch.zeros(self.maxd + 1, B, N, device=dev)
        phi = torch.tanh
        outs = []
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
            outs.append(h)
        return torch.stack(outs)  # (T,B,N)

    def forward(self, X, bc_idx):
        """bc_idx: (B,2) the [B,C] unit indices probed per trial. Returns logit (B,) for
        'B closer than C'. The readout sees the arrival-weighted trace (time-of-mass) at
        B and C -- a timing feature -- mapped by a learned 2->1 linear head."""
        H = self.propagate(X)                 # (T,B,N)
        T, B, N = H.shape
        dev = H.device
        tgrid = torch.arange(1, T + 1, device=dev).float().view(T, 1, 1)
        a = torch.relu(H)                      # activity (arrival energy)
        mass = a.sum(0) + 1e-6                 # (B,N) total energy per unit
        tom = (a * tgrid).sum(0) / mass        # (B,N) time-of-mass = arrival timing per unit
        bi = bc_idx[:, 0]; ci = bc_idx[:, 1]
        ar = torch.arange(B, device=dev)
        feat = torch.stack([tom[ar, bi], tom[ar, ci]], dim=-1)  # (B,2) [t_B, t_C]
        logit = (feat @ self.Wout + self.bout).squeeze(-1)      # (B,)
        return logit


def relay_input(tau, src_idx, T, device):
    """Vectorized: for each trial b with source unit src_idx[b]=A, every unit r gets a unit
    pulse at its own arrival step tau[A,r] (the baton's wavefront). This is the physical
    relay: the wavefront reaches each unit at its geometric delay from A."""
    src_idx = src_idx.to(tau.device)
    B = src_idx.shape[0]; N = tau.shape[0]
    arr = tau[src_idx].clamp(0, T - 1)        # (B,N) arrival step per unit from A
    X = torch.zeros(T, B, N, device=device)
    bidx = torch.arange(B, device=device).view(B, 1).expand(B, N)
    nidx = torch.arange(N, device=device).view(1, N).expand(B, N)
    X[arr, bidx, nidx] = 1.0
    return X


def run_control(pos, control, seed, N, v, maxd, steps=1500, lr=3e-3, l1=2e-3, device=DEV):
    tau, gpos = make_delays(pos, v, maxd, control, seed)   # gpos relabeled for shuffle_index
    tau = tau.to(device)
    T = min(int(tau.max().item()) + 3, 40)
    d_geo = torch.cdist(gpos, gpos).to(device)   # TRUE spatial distance on the SAME basis as tau (defines the label)

    # held-out split over SOURCE units A: train triples draw A from train pool, test from
    # held-out pool -> the ordering rule must generalize to A's it never saw.
    g = torch.Generator().manual_seed(123 + seed)
    permA = torch.randperm(N, generator=g)
    train_A = permA[: int(0.75 * N)]
    test_A = permA[int(0.75 * N):]             # HELD-OUT sources

    m = RelayRNN(N, tau, seed=seed).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)

    def sample_triples(A_pool, B, gen):
        """Draw B triples (A, b, c) with b,c != A and d(A,b) != d(A,c). Label = 1 if b is
        the closer neighbor (d(A,b) < d(A,c)). Returns src(B,), bc(B,2), label(B,)."""
        A = A_pool[torch.randint(0, len(A_pool), (B,), generator=gen)]
        # pick two distinct other units per trial
        others = torch.randint(0, N, (B, 2), generator=gen)
        # resolve collisions cheaply (rare): nudge any equal/self picks
        for col in range(2):
            bad = (others[:, col] == A) | (others[:, 0] == others[:, 1])
            others[bad, col] = (others[bad, col] + 1) % N
        bad = (others[:, 0] == others[:, 1]) | (others[:, 0] == A) | (others[:, 1] == A)
        others[bad, 1] = (others[bad, 1] + 2) % N
        dAb = d_geo[A, others[:, 0]]
        dAc = d_geo[A, others[:, 1]]
        label = (dAb < dAc).float()            # 1 -> B (col 0) is the closer one
        return A.to(device), others.to(device), label.to(device)

    bce = nn.BCEWithLogitsLoss()
    train_gen = torch.Generator().manual_seed(555 + seed)
    for it in range(steps):
        opt.zero_grad()
        A, bc, y = sample_triples(train_A, 128, train_gen)
        X = relay_input(tau, A, T, device)
        logit = m(X, bc)
        loss = bce(logit, y) + l1 * m.W.abs().mean()   # WIRING-COST (L1) penalty on W
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if it % 300 == 0:
            print(f"      [{control}] step {it}/{steps} loss={loss.item():.3f}", flush=True)

    def acc(A_pool, nshot=8, B=256):
        gen = torch.Generator().manual_seed(9000 + seed)
        accs = []
        with torch.no_grad():
            for _ in range(nshot):
                A, bc, y = sample_triples(A_pool, B, gen)
                X = relay_input(tau, A, T, device)
                pred = (m(X, bc) > 0).float()
                accs.append((pred == y).float().mean().item())
        return float(np.mean(accs))

    # structural sparsity achieved (diagnostic that geometry is load-bearing)
    Wrec = m.W.detach().cpu().numpy().copy()
    np.fill_diagonal(Wrec, 0.0)
    thr = 0.05 * (np.abs(Wrec).max() + 1e-9)
    part = float((np.abs(Wrec) > thr).mean())
    return {"train_acc": acc(train_A), "heldout_acc": acc(test_A),
            "participation": part, "speed_cv": effective_speed_cv(tau.cpu(), gpos)}


def run_seed(seed, N=64, v=0.12, maxd=16, device=DEV):
    pos = positions(N, seed)
    out = {}
    for ctrl in ["ordered", "shuffle_entry", "shuffle_index", "none"]:
        out[ctrl] = run_control(pos, ctrl, seed, N, v, maxd, device=device)
        print(f"    seed {seed} {ctrl:14s} train={out[ctrl]['train_acc']:.3f} "
              f"heldout={out[ctrl]['heldout_acc']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    SEEDS = 8
    rows = []
    for s in range(SEEDS):
        print(f"=== seed {s + 1}/{SEEDS} ===", flush=True)
        rows.append(run_seed(s))
    for s, r in enumerate(rows):
        print(f"seed {s}: ordered train={r['ordered']['train_acc']:.3f} ho={r['ordered']['heldout_acc']:.3f} | "
              f"entry-shuf train={r['shuffle_entry']['train_acc']:.3f} ho={r['shuffle_entry']['heldout_acc']:.3f} | "
              f"idx-shuf ho={r['shuffle_index']['heldout_acc']:.3f} | none ho={r['none']['heldout_acc']:.3f} | "
              f"cv ord={r['ordered']['speed_cv']:.2f} ent={r['shuffle_entry']['speed_cv']:.2f}")

    def arr(c, k): return np.array([r[c][k] for r in rows])
    o_ho, e_ho = arr("ordered", "heldout_acc"), arr("shuffle_entry", "heldout_acc")
    o_tr, e_tr = arr("ordered", "train_acc"), arr("shuffle_entry", "train_acc")
    gap = o_ho - e_ho                       # ordered minus entry-shuffle held-out acc (>0 = ordered better)
    idx_gap = o_ho - arr("shuffle_index", "heldout_acc")   # sanity: should be ~0
    def stats(a):
        m = float(a.mean()); sd = float(a.std() + 1e-9)
        return {"mean": m, "sd": sd, "t": m / (sd / math.sqrt(len(a))), "wins": int((a > 0).sum())}
    res = {
        "ordered_heldout_acc": float(o_ho.mean()),
        "entryshuffle_heldout_acc": float(e_ho.mean()),
        "train_acc_ordered": float(o_tr.mean()), "train_acc_entryshuffle": float(e_tr.mean()),
        # train-matched: ordered does not simply train BETTER (the benefit must be on
        # generalization, not a train-accuracy confound).
        "train_matched": bool(abs(o_tr.mean() - e_tr.mean()) < 0.5 * abs(gap.mean() + 1e-9)),
        "heldout_gap_ordered_minus_entryshuf": stats(gap),
        "index_perm_sanity_gap": stats(idx_gap),   # must be ~0 for the result to be non-circular
        "none_heldout_acc": float(arr("none", "heldout_acc").mean()),
        "participation_ordered": float(arr("ordered", "participation").mean()),
        "participation_entryshuffle": float(arr("shuffle_entry", "participation").mean()),
        "speed_cv_ordered": float(arr("ordered", "speed_cv").mean()),
        "speed_cv_entryshuffle": float(arr("shuffle_entry", "speed_cv").mean()),
        "device": DEV, "seeds": SEEDS, "minutes": round((time.time() - t0) / 60, 1),
    }
    g = res["heldout_gap_ordered_minus_entryshuf"]; idx = res["index_perm_sanity_gap"]
    # SIGNAL only if: ordered beats ENTRY-shuffle on held-out ordering, train matched, AND
    # the index-permutation sanity gap is ~0 (else it's just "delays help", not geometry).
    res["verdict"] = ("SIGNAL (needs adversarial verify)"
                      if g["t"] > 3 and g["wins"] >= 7 and res["train_matched"]
                      and abs(idx["mean"]) < 0.3 * abs(g["mean"] + 1e-9)
                      else "marginal" if g["t"] > 1.5 else "null")
    print("\n=== SPATIAL-ORDER RELAY VERDICT ===")
    print("ordered held-out order-acc  : %.3f" % res["ordered_heldout_acc"])
    print("entry-shuffle held-out acc  : %.3f" % res["entryshuffle_heldout_acc"])
    print("index-perm held-out acc     : %.3f  (sanity; should ~= ordered)" % (o_ho.mean() - idx["mean"]))
    print("none (chance) held-out acc  : %.3f  (should ~0.5)" % res["none_heldout_acc"])
    print("speed CV  ordered=%.2f  entry-shuffle=%.2f" % (res["speed_cv_ordered"], res["speed_cv_entryshuffle"]))
    print("gap ordered-entryshuf       : %.3f  t=%.2f  wins=%d/%d  train_matched=%s  idx_sanity=%.3f"
          % (g["mean"], g["t"], g["wins"], SEEDS, res["train_matched"], idx["mean"]))
    print("VERDICT:", res["verdict"])
    with open("colab_relay_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote colab_relay_results.json in", res["minutes"], "min")
    try:
        from google.colab import files
        files.download("colab_relay_results.json")
    except Exception:
        pass
