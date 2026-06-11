"""
Self-contained Colab (T4) test of THE breakthrough lead: METRIC / PROXIMITY REASONING.

Same converged claim as colab_localization.py, on a task whose GROUND-TRUTH RELATION
IS PHYSICAL DISTANCE. Given a *probe* unit p and two *candidate* units a, b, answer the
transitive-proximity question "is a closer to p than b is?" (and, as a second readout,
rank a probe's nearest neighbors). The answer is a GLOBAL-metric property of the layout.

Why this is the decisive design (from the agents' corrections, mirrored from localization):
  * Ordered tau is a VALID Euclidean distance matrix: arrival-time IS a monotone image of
    physical distance, triangle-consistent, embeddable. So proximity comparisons read out
    of the arrival-time field directly and TRANSFER to held-out probe units.
  * ENTRY-shuffle (scramble the off-diagonal delay multiset symmetrically) breaks metric
    consistency -- it violates the triangle inequality on a large fraction of triples, so
    the delay field corresponds to NO consistent geometry; "a closer to p than b" computed
    from shuffled arrivals is a WRONG relation that does not generalize. A trained readout +
    recurrence must OVERWRITE a globally-inconsistent relation -> sample-complexity wall.
  * shuffle_INDEX (relabel units consistently) is a VALID alternative geometry: the SAME
    proximity relation under a permutation a readout absorbs perfectly -> sanity gap ~0.
    If the ordered-vs-index gap is NOT ~0 the effect is "delays help", not "geometry helps".
  * none (all delays = 1): every unit arrives simultaneously -> no proximity info -> chance.
  * FULL TRAINABLE RECURRENCE + readout (the real test of whether trainable W re-sculpts a
    global metric property; it cannot edit tau, a non-learnable index into history).
  * FEW-SHOT: small train set of probe units to EXPOSE the sample-complexity gap, evaluated
    on HELD-OUT probe units (generalization), at TIGHTLY MATCHED train accuracy.

Benefit = ordered answers proximity on HELD-OUT probes far better than the entry-shuffle,
at matched train accuracy, beyond seed noise, AND beyond the index-perm sanity (~0). Paste
into a T4 Colab; downloads colab_relational_results.json.
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
    the off-diagonal delay multiset symmetrically -> breaks metric consistency),
    'shuffle_index' (relabel units consistently -> a VALID alt-geometry a readout can
    absorb; sanity that should give ~0 gap), 'none' (all delays 1 -> simultaneous)."""
    tau = delay_from_positions(pos, v, maxd)
    N = tau.shape[0]
    if control == "ordered":
        return tau, pos
    if control == "none":
        return torch.ones_like(tau), pos
    if control == "shuffle_index":
        g = torch.Generator().manual_seed(7000 + seed)
        perm = torch.randperm(N, generator=g)
        return tau[perm][:, perm], pos[perm]          # KEY FIX: relabel positions (the ground-truth basis) too -> pure relabeling control
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


def triangle_violation_rate(tau, n_triples=4000, seed=0):
    """Fraction of triples (i,j,k) violating tau[i,k] <= tau[i,j] + tau[j,k] (slack 0).
    Ordered (a metric image) ~0; entry-shuffle large -> not embeddable as a geometry."""
    N = tau.shape[0]
    g = torch.Generator().manual_seed(99 + seed)
    t = tau.float()
    ijk = torch.randint(0, N, (n_triples, 3), generator=g)
    i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    ok = (i != j) & (j != k) & (i != k)
    i, j, k = i[ok], j[ok], k[ok]
    viol = (t[i, k] > t[i, j] + t[j, k] + 1e-6)
    return float(viol.float().mean())


class RelRNN(nn.Module):
    """Delay-coupled RNN for proximity reasoning. A probe unit p is pulsed; its arrival
    wave propagates through the delay graph. The pooled response is read out at the two
    candidate units a,b to score "a closer to p than b". Input weight is a FROZEN identity
    (unit r is driven by its own arrival from p) so geometry is grounded and cannot be
    relabeled by training; only W (recurrence) + the scalar proximity readout are learned.
    """
    def __init__(self, N, tau, alpha=0.25, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.N, self.alpha = N, alpha
        self.register_buffer("tau", tau)
        self.maxd = int(tau.max().item())
        self.W = nn.Parameter(torch.randn(N, N) * (0.9 / math.sqrt(N)))
        self.b = nn.Parameter(torch.zeros(N))
        # per-unit scalar "proximity-to-probe" score: pooled activity -> 1 number/unit
        self.wscore = nn.Parameter(torch.randn(N) / math.sqrt(N))
        self.bscore = nn.Parameter(torch.zeros(1))
        masks = [(tau == d).float() for d in range(1, self.maxd + 1)]
        self.register_buffer("masks", torch.stack(masks) if masks else torch.zeros(0, N, N))

    def field(self, X):
        """Propagate probe pulses; return pooled per-unit activity (B,N)."""
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
        return pooled / T

    def score(self, X):
        """Per-unit proximity-to-probe score (B,N): high = network thinks unit is near probe."""
        pooled = self.field(X)                       # (B,N)
        return pooled * self.wscore + self.bscore    # (B,N)

    def forward(self, X, a_idx, b_idx):
        """Logit for "candidate a is closer to probe than candidate b"."""
        s = self.score(X)                            # (B,N)
        B = s.shape[0]                                # batch (X is (T,B,N))
        ar = torch.arange(B, device=X.device)
        return s[ar, a_idx] - s[ar, b_idx]           # (B,) logit


def probe_input(tau, probe_idx, T, device):
    """Vectorized: for trial b with probe probe_idx[b], unit r gets a pulse at t=tau[probe,r]."""
    probe_idx = probe_idx.to(tau.device)
    B = probe_idx.shape[0]; N = tau.shape[0]
    arr = tau[probe_idx].clamp(0, T - 1)             # (B,N) arrival step per unit
    X = torch.zeros(T, B, N, device=device)
    bidx = torch.arange(B, device=device).view(B, 1).expand(B, N)
    nidx = torch.arange(N, device=device).view(1, N).expand(B, N)
    X[arr, bidx, nidx] = 1.0
    return X


def run_control(pos, control, seed, N, v, maxd, steps=1200, lr=3e-3,
                n_train_probes=10, device=DEV):
    """FEW-SHOT: train on a small set of probe units; test on HELD-OUT probes. Ground truth
    is PHYSICAL distance (pos), identical across controls -- only tau (the network's access
    to geometry) changes. So any held-out gap is about metric consistency of the delay graph."""
    tau, tgt_pos = make_delays(pos, v, maxd, control, seed)   # tgt_pos relabeled for shuffle_index
    tau = tau.to(device)
    T = min(int(tau.max().item()) + 3, 40)
    pos_d = tgt_pos.to(device)
    D = torch.cdist(pos_d, pos_d)                     # TRUE physical distances on the SAME basis as tau (ground truth)

    g = torch.Generator().manual_seed(123 + seed)
    perm = torch.randperm(N, generator=g)
    train_probes = perm[:n_train_probes]
    test_probes = perm[int(0.7 * N):]                 # HELD-OUT probe units (disjoint tail)

    m = RelRNN(N, tau, seed=seed).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    gpool = torch.Generator().manual_seed(555 + seed)

    def batch(probe_pool, B):
        """Each example: probe p, two candidates a,b (a != b != p). Label = a closer than b."""
        pi = probe_pool[torch.randint(0, len(probe_pool), (B,), generator=gpool)]
        a = torch.randint(0, N, (B,), generator=gpool)
        b = torch.randint(0, N, (B,), generator=gpool)
        # avoid degenerate / tied candidates
        bad = (a == b) | (a == pi) | (b == pi)
        for _ in range(8):
            if not bad.any():
                break
            a[bad] = torch.randint(0, N, (int(bad.sum()),), generator=gpool)
            b[bad] = torch.randint(0, N, (int(bad.sum()),), generator=gpool)
            bad = (a == b) | (a == pi) | (b == pi)
        X = probe_input(tau, pi, T, device)
        da = D[pi, a]; db = D[pi, b]                  # true distances probe->candidate
        y = (da < db).float()                        # 1 if a is genuinely closer
        return X, a.to(device), b.to(device), y.to(device)

    for it in range(steps):
        opt.zero_grad()
        X, a, b, y = batch(train_probes, 128)
        logit = m(X, a, b)
        loss = bce(logit, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if it % 300 == 0:
            print(f"      [{control}] step {it}/{steps} loss={loss.item():.3f}", flush=True)

    def acc(probe_pool, B):
        with torch.no_grad():
            X, a, b, y = batch(probe_pool, B)
            logit = m(X, a, b)
            return float(((logit > 0).float() == y).float().mean())

    def ndcg_at_k(probe_pool, k=5, nrep=4):
        """Ranking metric: for each held-out probe, rank ALL units by predicted proximity
        score; nDCG@k against true nearest neighbors. Rewards correct global ordering."""
        with torch.no_grad():
            vals = []
            for _ in range(nrep):
                pi = probe_pool
                X = probe_input(tau, pi, T, device)
                s = m.score(X)                       # (P,N) predicted proximity
                Bp = len(pi)
                # mask self
                s = s.clone()
                s[torch.arange(Bp), pi] = -1e9
                d = D[pi].clone()
                d[torch.arange(Bp), pi] = 1e9
                pred_rank = torch.argsort(s, dim=1, descending=True)   # near-first
                # ideal: smallest true distance first
                true_near = torch.argsort(d, dim=1, descending=False)
                topk = pred_rank[:, :k]
                # relevance = 1 if predicted-topk item is among true k nearest
                true_set = true_near[:, :k]
                rel = torch.zeros(Bp, k, device=s.device)
                for r in range(Bp):
                    ts = set(true_set[r].tolist())
                    for c in range(k):
                        rel[r, c] = 1.0 if topk[r, c].item() in ts else 0.0
                discount = 1.0 / torch.log2(torch.arange(2, k + 2, device=s.device).float())
                dcg = (rel * discount).sum(1)
                idcg = discount.sum()                # all-relevant ideal
                vals.append(float((dcg / idcg).mean()))
            return float(np.mean(vals))

    return {
        "train_acc": acc(train_probes, 512),
        "heldout_acc": acc(test_probes, min(2048, 64 * len(test_probes))),
        "heldout_ndcg": ndcg_at_k(test_probes, k=5),
    }


def run_seed(seed, N=64, v=0.12, maxd=16, device=DEV):
    pos = positions(N, seed)
    out = {}
    # report triangle-violation rate for the geometry diagnostics
    tv = {c: triangle_violation_rate(make_delays(pos, v, maxd, c, seed)[0], seed=seed)
          for c in ["ordered", "shuffle_entry"]}
    for ctrl in ["ordered", "shuffle_entry", "shuffle_index", "none"]:
        out[ctrl] = run_control(pos, ctrl, seed, N, v, maxd, device=device)
        print(f"    seed {seed} {ctrl:14s} train={out[ctrl]['train_acc']:.3f} "
              f"heldout={out[ctrl]['heldout_acc']:.3f} ndcg={out[ctrl]['heldout_ndcg']:.3f}", flush=True)
    out["_tri_viol"] = tv
    return out


if __name__ == "__main__":
    t0 = time.time()
    SEEDS = 8
    rows = []
    for s in range(SEEDS):
        print(f"=== seed {s + 1}/{SEEDS} ===", flush=True)
        rows.append(run_seed(s))
    for s, r in enumerate(rows):
        print(f"seed {s}: ordered tr={r['ordered']['train_acc']:.3f} ho={r['ordered']['heldout_acc']:.3f} "
              f"ndcg={r['ordered']['heldout_ndcg']:.3f} | "
              f"entry-shuf tr={r['shuffle_entry']['train_acc']:.3f} ho={r['shuffle_entry']['heldout_acc']:.3f} "
              f"ndcg={r['shuffle_entry']['heldout_ndcg']:.3f} | "
              f"idx-shuf ho={r['shuffle_index']['heldout_acc']:.3f} | none ho={r['none']['heldout_acc']:.3f} | "
              f"triViol ord={r['_tri_viol']['ordered']:.2f} ent={r['_tri_viol']['shuffle_entry']:.2f}")

    def arr(c, k): return np.array([r[c][k] for r in rows])
    o_ho, e_ho = arr("ordered", "heldout_acc"), arr("shuffle_entry", "heldout_acc")
    o_tr, e_tr = arr("ordered", "train_acc"), arr("shuffle_entry", "train_acc")
    o_nd, e_nd = arr("ordered", "heldout_ndcg"), arr("shuffle_entry", "heldout_ndcg")
    gap = o_ho - e_ho                       # ordered minus entry-shuffle held-out ACC (>0 = ordered better)
    nd_gap = o_nd - e_nd                     # ranking gap (>0 = ordered better)
    idx_gap = o_ho - arr("shuffle_index", "heldout_acc")   # sanity: should be ~0

    def stats(a):
        m = float(a.mean()); sd = float(a.std() + 1e-9)
        return {"mean": m, "sd": sd, "t": m / (sd / math.sqrt(len(a))), "wins": int((a > 0).sum())}

    res = {
        "ordered_heldout_acc": float(o_ho.mean()),
        "entryshuffle_heldout_acc": float(e_ho.mean()),
        "ordered_heldout_ndcg": float(o_nd.mean()),
        "entryshuffle_heldout_ndcg": float(e_nd.mean()),
        "train_acc_ordered": float(o_tr.mean()), "train_acc_entryshuffle": float(e_tr.mean()),
        # train matched if the train-accuracy difference is small vs the held-out gap it must explain
        "train_matched": bool(abs(o_tr.mean() - e_tr.mean()) < 0.5 * abs(gap.mean() + 1e-9)),
        "heldout_gap_ordered_minus_entryshuf": stats(gap),
        "ndcg_gap_ordered_minus_entryshuf": stats(nd_gap),
        "index_perm_sanity_gap": stats(idx_gap),   # must be ~0 for the result to be non-circular
        "none_heldout_acc": float(arr("none", "heldout_acc").mean()),
        "tri_viol_ordered": float(np.mean([r["_tri_viol"]["ordered"] for r in rows])),
        "tri_viol_entryshuffle": float(np.mean([r["_tri_viol"]["shuffle_entry"] for r in rows])),
        "device": DEV, "seeds": SEEDS, "minutes": round((time.time() - t0) / 60, 1),
    }
    g = res["heldout_gap_ordered_minus_entryshuf"]; idx = res["index_perm_sanity_gap"]
    # SIGNAL only if: ordered beats ENTRY-shuffle on held-out, train matched, AND the
    # index-permutation sanity gap is ~0 (else it's just "delays help", not geometry).
    res["verdict"] = ("SIGNAL (needs adversarial verify)"
                      if g["t"] > 3 and g["wins"] >= 7 and res["train_matched"]
                      and abs(idx["mean"]) < 0.3 * abs(g["mean"])
                      else "marginal" if g["t"] > 1.5 else "null")
    print("\n=== METRIC / PROXIMITY REASONING VERDICT ===")
    print("ordered held-out acc   : %.3f  (ndcg@5 %.3f)" % (res["ordered_heldout_acc"], res["ordered_heldout_ndcg"]))
    print("entry-shuffle held-out : %.3f  (ndcg@5 %.3f)" % (res["entryshuffle_heldout_acc"], res["entryshuffle_heldout_ndcg"]))
    print("index-perm held-out    : %.3f  (sanity; should ~= ordered)" % (o_ho.mean() - idx["mean"]))
    print("none (chance) held-out : %.3f" % res["none_heldout_acc"])
    print("triangle-viol  ordered : %.3f   entry-shuffle: %.3f" % (res["tri_viol_ordered"], res["tri_viol_entryshuffle"]))
    print("gap ordered-entryshuf  : %.3f  t=%.2f  wins=%d/%d  train_matched=%s  idx_sanity=%.3f"
          % (g["mean"], g["t"], g["wins"], SEEDS, res["train_matched"], idx["mean"]))
    print("VERDICT:", res["verdict"])
    with open("colab_relational_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote colab_relational_results.json in", res["minutes"], "min")
    try:
        from google.colab import files
        files.download("colab_relational_results.json")
    except Exception:
        pass
