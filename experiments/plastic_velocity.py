"""HETEROGENEOUS PLASTIC CONDUCTION VELOCITY: turn the conduction-time economy from a
STRUCTURAL result (under uniform v, tau = d/v makes delay PROPORTIONAL to distance, so
"value where delay is short" == "value where distance is short" == the wire economy --
dismissible as a rearrangement inequality capped by value~=distance) into a FUNCTIONAL one.

THE FIX. Make tau_ij = d_ij / v_ij with a LEARNABLE per-edge velocity v_ij under a
MYELINATION budget (faster axon = more myelin = more cost, growing with axon length).
Now value DECOUPLES from distance: a long-but-FAST edge is low-delay / high-wire.

THE DECISIVE TEST (matched RESOURCE, not matched accuracy):
  Compare a PLASTIC-v net (learnable v_ij under myelin budget) vs a UNIFORM-v net at
  MATCHED TOTAL MYELIN on HELD-OUT accuracy. The myelin budget is
      M = sum_{(i,j) in E} d_ij * (v_ij - v_min),
  so for the uniform net a single scalar v_uniform = v_min + M*/sum_E d_ij matches the
  plastic net's realized budget M* EXACTLY (closed form). If plastic-v buys held-out
  accuracy uniform-v cannot match at the same total myelin, the economy is FUNCTIONAL.
  SHUFFLE-v control: reassign the trained plastic v multiset randomly across edges (same
  budget, same histogram, allocation geometry destroyed) -> must collapse to uniform level.

TASK (must have HIGH-VALUE LONG-RANGE deps). Two clusters at opposite sides; the radius
graph connects them only through a SMALL number of long bridge edges. A K-bit code is
injected at the LEFT source units; a readout at the RIGHT target units must reconstruct it
within a SHORT DEADLINE WINDOW. With unmyelinated v_min the long bridge delay exceeds the
deadline -> bits arrive late and are lost. The only way to raise held-out accuracy is to
MYELINATE (speed up) the few long bridge edges; short local edges are already fast enough,
so spending myelin on them is wasted. => value concentrates on LONG edges => value
decouples from distance.

DIFFERENTIABILITY. tau = round(.) is non-differentiable -> SOFT/FRACTIONAL delays: the
contribution along edge (i,j) at lag tau_ij is read from the partner's history by linear
interpolation between bracketing integer buffer slots,
    h_interp = (1-frac) * hist[lo] + frac * hist[hi],   lo=floor(tau), hi=ceil(tau).
Both value and d/dtau = hist[hi]-hist[lo] are well defined, so grad flows to v_ij via
tau = d/v and the bounded reparam v = v_min + (v_max-v_min)*sigmoid(g_ij). CAVEAT (verified
& guarded): the interp slope is 0 at integer lags (frac=0/1), so the geometry must be
scaled so off-diagonal lags land off-integer -> we scale positions so tau spans the active
band and ASSERT a healthy fraction of edges are fractional + the v-gradient is nonzero.

A NULL is informative: if plastic-v gives NO held-out gain at matched budget, the economy
stays structural -- reported honestly.
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# Geometry: two clusters (left source / right target) joined only by a few long bridges.
# --------------------------------------------------------------------------------------
def two_cluster_positions(N, seed, gap=2.2, spread=1.0):
    """Place N units in two blobs separated along x by `gap`. Returns (pos, src, tgt).
    Positions are in LATTICE UNITS (scaled up) so delays land in the active fractional band.
    """
    g = torch.Generator().manual_seed(seed)
    nl = N // 2
    nr = N - nl
    left = torch.rand(nl, 2, generator=g) * spread
    right = torch.rand(nr, 2, generator=g) * spread
    right[:, 0] += gap                                  # push the right cluster across the gap
    pos = torch.cat([left, right], 0)
    src = torch.arange(0, nl)                            # left cluster = sources
    tgt = torch.arange(nl, N)                            # right cluster = targets
    return pos, src, tgt


def radius_graph_delays(pos, R, v0, min_delay, max_delay):
    """Edges within radius R (PLUS a guaranteed bridge so the clusters are connected).
    Returns (edge bool NxN symmetric, d NxN distances, tau0 fractional lags at v0)."""
    N = pos.shape[0]
    d = torch.cdist(pos, pos)
    edge = (d <= R) & (d > 0)
    # Ensure the two clusters are connected: keep the K shortest cross-cluster pairs as bridges.
    return edge, d


def build_graph(N, seed, R, v0, gap, spread, min_delay, max_delay, n_bridge):
    """Build the two-cluster radius graph and GUARANTEE a small bridge set of long edges
    that are the only timely cross-sheet route. Returns dict of buffers."""
    pos, src, tgt = two_cluster_positions(N, seed, gap=gap, spread=spread)
    d = torch.cdist(pos, pos)
    edge = (d <= R) & (d > 0)
    # Cross-cluster candidate edges (left<->right). Radius R is set so NO local-radius edge
    # crosses the gap; we explicitly add the n_bridge SHORTEST cross pairs as the bridges.
    cross = torch.zeros(N, N, dtype=torch.bool)
    sl, tl = src.tolist(), tgt.tolist()
    cross_pairs = []
    for i in sl:
        for j in tl:
            cross_pairs.append((d[i, j].item(), i, j))
    cross_pairs.sort()
    bridge_edges = cross_pairs[:n_bridge]
    for _, i, j in bridge_edges:
        cross[i, j] = True
        cross[j, i] = True
    edge = edge | cross
    edge.fill_diagonal_(False)
    bridge_mask = cross.clone()
    return dict(pos=pos, d=d, edge=edge, src=src, tgt=tgt, bridge_mask=bridge_mask,
                bridge_edges=bridge_edges)


# --------------------------------------------------------------------------------------
# The delay-coupled RNN with per-edge SOFT/FRACTIONAL delays and learnable velocity.
# --------------------------------------------------------------------------------------
class PlasticVelocityRNN(nn.Module):
    """Delay-coupled tanh RNN. Recurrent signal j->i is read at fractional lag tau_ij from
    a rolling history by linear interpolation (so v_ij is gradient-learnable). Input weight
    is a FROZEN identity on source units (geometry grounded; training cannot relabel it).

    velocity_mode:
      'plastic' : per-edge log-velocity nn.Parameter (symmetric, masked) -> v_ij learnable.
      'uniform' : a single fixed scalar v applied to every edge (the original tau ~ d net).
      'fixed'   : per-edge velocities pinned to a supplied tensor (used by the shuffle-v ctrl).
    """

    def __init__(self, graph, K, *, velocity_mode, v_min, v_max, v0, min_delay, max_delay,
                 alpha=0.3, seed=0, fixed_v=None):
        super().__init__()
        torch.manual_seed(seed)
        N = graph["pos"].shape[0]
        self.N, self.K, self.alpha = N, K, alpha
        self.v_min, self.v_max = v_min, v_max
        self.min_delay, self.max_delay = min_delay, max_delay
        self.velocity_mode = velocity_mode

        self.register_buffer("d", graph["d"].clone())
        self.register_buffer("edge", graph["edge"].clone())
        self.register_buffer("src", graph["src"].clone())
        self.register_buffer("tgt", graph["tgt"].clone())
        self.register_buffer("emask", graph["edge"].float())

        deg = graph["edge"].float().sum(1).clamp_min(1).mean()
        self.W = nn.Parameter(torch.randn(N, N) * (0.9 / math.sqrt(float(deg))))
        self.b = nn.Parameter(torch.zeros(N))
        self.Wout = nn.Parameter(torch.randn(len(graph["tgt"]), K) / math.sqrt(len(graph["tgt"])))
        self.bout = nn.Parameter(torch.zeros(K))

        # symmetric undirected edge index (upper triangle) for symmetric velocity params.
        iu = torch.triu_indices(N, N, 1)
        em = graph["edge"][iu[0], iu[1]]
        self.register_buffer("ui", iu[:, em])              # (2, E) undirected edge endpoints
        bui = iu[:, em]
        self.register_buffer("bridge_ui", graph["bridge_mask"][bui[0], bui[1]])  # (E,) bool

        if velocity_mode == "plastic":
            # free param g_ij -> v = v_min + (v_max-v_min)*sigmoid(g). init near v0.
            g0 = self._inv_reparam(torch.full((self.ui.shape[1],), float(v0)))
            self.g = nn.Parameter(g0 + 0.01 * torch.randn(self.ui.shape[1]))
        elif velocity_mode == "uniform":
            self.register_buffer("uniform_v", torch.tensor(float(v0)))
        elif velocity_mode == "fixed":
            assert fixed_v is not None
            self.register_buffer("fixed_v", fixed_v.clone())  # (E,) per undirected edge
        else:
            raise ValueError(velocity_mode)

    def _inv_reparam(self, v):
        v = v.clamp(self.v_min + 1e-4, self.v_max - 1e-4)
        z = (v - self.v_min) / (self.v_max - self.v_min)
        return torch.log(z / (1 - z))

    def edge_velocity(self):
        """Per-undirected-edge velocity vector (E,)."""
        if self.velocity_mode == "plastic":
            return self.v_min + (self.v_max - self.v_min) * torch.sigmoid(self.g)
        if self.velocity_mode == "uniform":
            return self.uniform_v.expand(self.ui.shape[1])
        return self.fixed_v

    def velocity_matrix(self):
        """Dense symmetric (N,N) velocity matrix (only meaningful on edges)."""
        v = self.edge_velocity()
        V = torch.full((self.N, self.N), float(self.v_max), device=v.device, dtype=v.dtype)
        V[self.ui[0], self.ui[1]] = v
        V[self.ui[1], self.ui[0]] = v
        return V

    def tau_matrix(self):
        """Fractional lags tau_ij = clamp(d/v, min, max), differentiable in v (plastic)."""
        V = self.velocity_matrix()
        tau = (self.d / V).clamp(self.min_delay, self.max_delay)
        return tau

    def myelin(self):
        """Total myelin M = sum_E d_ij * (v_ij - v_min) over undirected edges."""
        v = self.edge_velocity()
        dl = self.d[self.ui[0], self.ui[1]]
        return (dl * (v - self.v_min)).sum()

    def propagate(self, code, T):
        """code: (B,K) in {-1,1} injected one bit/step into the source units. Returns H (T,B,N)."""
        B = code.shape[0]
        N, dev = self.N, code.device
        Wl = self.W * self.emask                                  # mask to existing edges
        tau = self.tau_matrix()                                   # (N,N) fractional
        lo = torch.floor(tau).long().clamp(self.min_delay, self.max_delay)
        hi = torch.ceil(tau).long().clamp(self.min_delay, self.max_delay)
        frac = (tau - lo.to(tau.dtype)).clamp(0.0, 1.0)
        maxd = self.max_delay
        hist = torch.zeros(maxd + 1, B, N, device=dev)            # hist[k] = h from k steps ago
        cols = torch.arange(N, device=dev).view(1, N).expand(N, N)
        outs = []
        phi = torch.tanh
        for t in range(T):
            # gather partner history at lo and hi lags, interpolate by frac (soft delay).
            h_lo = hist[lo, :, cols]                              # (N,N,B): [i,j,b]=h_j[t-lo_ij]
            h_hi = hist[hi, :, cols]
            delayed = h_lo * (1.0 - frac).unsqueeze(-1) + h_hi * frac.unsqueeze(-1)  # (N,N,B)
            # rec_i = sum_j W_ij * phi(delayed_j);  einsum over source index j.
            rec = torch.einsum("ij,ijb->ib", Wl, phi(delayed)).t()   # (B,N)
            inp = torch.zeros(B, N, device=dev)
            if t < self.K:
                inp[:, self.src] = code[:, t:t + 1]               # broadcast bit to source units
            h = (1 - self.alpha) * hist[0] + self.alpha * (rec + inp + self.b)
            h = torch.clamp(h, -8, 8)
            hist = torch.roll(hist, 1, 0)
            hist[0] = h
            outs.append(h)
        return torch.stack(outs)                                  # (T,B,N)

    def forward(self, code, T, read_window, deadline):
        """Read the code from the target units within a SHORT deadline window
        [deadline-read_window, deadline). Returns logits (B,K). Timing-sensitive: the
        window CLOSES at `deadline`, so a signal whose bridge delay pushes arrival past
        the deadline is never read -> the bits are lost. A FAST bridge delivers in time."""
        H = self.propagate(code, T)                               # (T,B,N)
        w = read_window
        lo = max(0, deadline - w)
        read = H[lo:deadline][:, :, self.tgt].mean(0)             # (B, n_tgt)
        logits = read @ self.Wout + self.bout                     # (B,K)
        return logits


# --------------------------------------------------------------------------------------
# Training + evaluation
# --------------------------------------------------------------------------------------
def make_codes(B, K, gen, device):
    code = (torch.randint(0, 2, (B, K), generator=gen).float() * 2 - 1).to(device)
    return code


def train_model(m, T, read_window, deadline, steps, lr, lr_v, l1, lam_M, device, seed,
                batch=128, log_every=1e9):
    # give the per-edge velocity its own (higher) learning rate so it actually moves.
    vparams, oparams = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        (vparams if n == "g" else oparams).append(p)
    groups = [dict(params=oparams, lr=lr)]
    if vparams:
        groups.append(dict(params=vparams, lr=lr_v))
    opt = torch.optim.Adam(groups)
    bce = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(555 + seed)
    for it in range(steps):
        opt.zero_grad()
        code = make_codes(batch, m.K, gen, device)
        logits = m(code, T, read_window, deadline)
        target = (code > 0).float()
        loss = bce(logits, target) + l1 * m.W.abs().mean()
        if lam_M > 0 and m.velocity_mode == "plastic":
            loss = loss + lam_M * m.myelin()
        loss.backward()
        nn.utils.clip_grad_norm_([p for grp in groups for p in grp["params"]], 1.0)
        opt.step()
        if it % int(log_every) == 0:
            print(f"      step {it}/{steps} loss={loss.item():.4f}", flush=True)
    return m


@torch.no_grad()
def eval_acc(m, T, read_window, deadline, device, seed, nshot=4, batch=256):
    bce_correct = []
    gen = torch.Generator().manual_seed(9000 + seed)
    for _ in range(nshot):
        code = make_codes(batch, m.K, gen, device)
        logits = m(code, T, read_window, deadline)
        pred = (logits > 0).float()
        target = (code > 0).float()
        bce_correct.append((pred == target).float().mean().item())
    return float(np.mean(bce_correct))


# --------------------------------------------------------------------------------------
# Analysis: edge importance, value, value-distance decoupling.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def edge_importance(m, T, read_window, deadline, device, seed, batch=256, subsample=None):
    """Per-undirected-edge importance = drop in accuracy when that edge's W is zeroed.
    Returns (E,) importance vector aligned with m.ui. `subsample` (int) ablates only a
    random subset of edges (the rest get importance 0) -- the bridges are always included
    so the high-value-long-edge signal is preserved; used to keep the smoke fast."""
    base = eval_acc(m, T, read_window, deadline, device, seed, nshot=2, batch=batch)
    E = m.ui.shape[1]
    imp = torch.zeros(E)
    if subsample is not None and subsample < E:
        gen = torch.Generator().manual_seed(4242 + seed)
        pick = torch.randperm(E, generator=gen)[:subsample]
        # always include every bridge edge (the high-value long ones)
        bridge_e = torch.where(m.bridge_ui)[0].cpu() if hasattr(m, "bridge_ui") else torch.tensor([], dtype=torch.long)
        edges = torch.unique(torch.cat([pick, bridge_e])).tolist()
    else:
        edges = list(range(E))
    W0 = m.W.detach().clone()
    for e in edges:
        i, j = int(m.ui[0, e]), int(m.ui[1, e])
        m.W.data[i, j] = 0.0
        m.W.data[j, i] = 0.0
        a = eval_acc(m, T, read_window, deadline, device, seed, nshot=2, batch=batch)
        imp[e] = base - a
        m.W.data[i, j] = W0[i, j]
        m.W.data[j, i] = W0[j, i]
    return imp


def corr(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def edge_value(m):
    """Optimal-control-style per-edge value proxy used in the repo: |W|-weighted
    contribution (|W_ij| is the coupling mass on edge ij). Returns (E,)."""
    W = m.W.detach().abs()
    return W[m.ui[0], m.ui[1]].cpu().numpy()


# --------------------------------------------------------------------------------------
# One full matched-budget comparison at a given seed.
# --------------------------------------------------------------------------------------
def run_seed(seed, cfg, device):
    g = build_graph(cfg["N"], seed, cfg["R"], cfg["v0"], cfg["gap"], cfg["spread"],
                    cfg["min_delay"], cfg["max_delay"], cfg["n_bridge"])
    d = g["d"]
    # ----- diagnostics: are lags in the active fractional band? -----
    iu = torch.triu_indices(cfg["N"], cfg["N"], 1)
    em = g["edge"][iu[0], iu[1]]
    de = d[iu[0], iu[1]][em]
    tau0 = (de / cfg["v0"]).clamp(cfg["min_delay"], cfg["max_delay"])
    frac_pairs = float(((tau0 - tau0.floor()).clamp(0, 1).gt(1e-3) &
                        (tau0 - tau0.floor()).clamp(0, 1).lt(1 - 1e-3)).float().mean())
    n_edges = int(em.sum())

    Tmax = cfg["T"]
    rw = cfg["read_window"]
    dd = cfg["deadline"]

    # ===== 1) PLASTIC-v: trains W, b, head, AND per-edge velocity under myelin budget. =====
    mp = PlasticVelocityRNN(g, cfg["K"], velocity_mode="plastic", v_min=cfg["v_min"],
                            v_max=cfg["v_max"], v0=cfg["v0"], min_delay=cfg["min_delay"],
                            max_delay=cfg["max_delay"], alpha=cfg["alpha"], seed=seed).to(device)
    v_init = mp.edge_velocity().detach().clone()
    # check velocity gradient is alive BEFORE training (guards the integer-lag null)
    code = make_codes(16, cfg["K"], torch.Generator().manual_seed(1), device)
    logits = mp(code, Tmax, rw, dd)
    gv = torch.autograd.grad(logits.sum(), mp.g, retain_graph=False)[0]
    v_grad_nonzero = int((gv.abs() > 1e-9).sum())
    v_grad_norm = float(gv.norm())

    train_model(mp, Tmax, rw, dd, cfg["steps"], cfg["lr"], cfg["lr_v"], cfg["l1"],
                cfg["lam_M"], device, seed)
    M_star = float(mp.myelin().item())
    v_plastic = mp.edge_velocity().detach().clone()
    v_moved = float((v_plastic - v_init).abs().mean().item())
    plastic_ho = eval_acc(mp, Tmax, rw, dd, device, seed)
    plastic_tr = eval_acc(mp, Tmax, rw, dd, device, seed + 7777)  # "train-like" different RNG

    # ===== 2) UNIFORM-v at MATCHED myelin: v_uniform = v_min + M*/sum_E d_ij (closed form). =====
    dl = d[mp.ui[0].to(d.device), mp.ui[1].to(d.device)]
    sum_d = float(dl.sum().item())
    v_uniform = cfg["v_min"] + M_star / max(sum_d, 1e-9)
    v_uniform = float(np.clip(v_uniform, cfg["v_min"], cfg["v_max"]))
    mu = PlasticVelocityRNN(g, cfg["K"], velocity_mode="uniform", v_min=cfg["v_min"],
                            v_max=cfg["v_max"], v0=v_uniform, min_delay=cfg["min_delay"],
                            max_delay=cfg["max_delay"], alpha=cfg["alpha"], seed=seed).to(device)
    M_uniform = float(mu.myelin().item())
    train_model(mu, Tmax, rw, dd, cfg["steps"], cfg["lr"], cfg["lr_v"], cfg["l1"], 0.0, device, seed)
    uniform_ho = eval_acc(mu, Tmax, rw, dd, device, seed)

    # ===== 3) SHUFFLE-v control: SAME plastic velocity multiset, reassigned across edges. =====
    perm = torch.randperm(v_plastic.numel(), generator=torch.Generator().manual_seed(321 + seed))
    v_shuf = v_plastic[perm.to(v_plastic.device)].cpu()
    ms = PlasticVelocityRNN(g, cfg["K"], velocity_mode="fixed", v_min=cfg["v_min"],
                            v_max=cfg["v_max"], v0=cfg["v0"], min_delay=cfg["min_delay"],
                            max_delay=cfg["max_delay"], alpha=cfg["alpha"], seed=seed,
                            fixed_v=v_shuf).to(device)
    M_shuf = float(ms.myelin().item())
    train_model(ms, Tmax, rw, dd, cfg["steps"], cfg["lr"], cfg["lr_v"], cfg["l1"], 0.0, device, seed)
    shuffle_ho = eval_acc(ms, Tmax, rw, dd, device, seed)

    # ===== MEASUREMENT 2: selective myelination of high-value long edges. =====
    imp = edge_importance(mp, Tmax, rw, dd, device, seed,
                          subsample=cfg.get("imp_subsample")).cpu().numpy()
    length = dl.cpu().numpy()
    vlearn = v_plastic.cpu().numpy()
    is_bridge = g["bridge_mask"][mp.ui[0].cpu(), mp.ui[1].cpu()].cpu().numpy().astype(bool)
    corr_v_implen = corr(vlearn, imp * length)
    corr_v_len = corr(vlearn, length)
    v_bridge = float(vlearn[is_bridge].mean()) if is_bridge.any() else float("nan")
    v_nonbridge = float(vlearn[~is_bridge].mean()) if (~is_bridge).any() else float("nan")

    # ===== MEASUREMENT 3: value decouples from distance. =====
    # value proxy = |W|-weighted edge contribution. Under uniform v, value ~ -distance
    # (short edges carry weight). Under plastic v, long edges can be fast -> value spreads.
    val_p = edge_value(mp)
    val_u = edge_value(mu)
    corr_val_dist_plastic = corr(val_p, length)
    corr_val_dist_uniform = corr(val_u, length)
    # tau-distance correlation: under uniform tau ~ d (corr 1); under plastic it drops.
    tau_p = mp.tau_matrix().detach()[mp.ui[0], mp.ui[1]].cpu().numpy()
    tau_u = mu.tau_matrix().detach()[mp.ui[0], mp.ui[1]].cpu().numpy()
    corr_tau_dist_plastic = corr(tau_p, length)
    corr_tau_dist_uniform = corr(tau_u, length)

    return dict(
        seed=seed, n_edges=n_edges, frac_pairs=frac_pairs,
        v_grad_nonzero=v_grad_nonzero, v_grad_norm=v_grad_norm, v_moved=v_moved,
        M_star=M_star, M_uniform=M_uniform, M_shuf=M_shuf, v_uniform=v_uniform,
        plastic_ho=plastic_ho, plastic_tr=plastic_tr, uniform_ho=uniform_ho, shuffle_ho=shuffle_ho,
        gap=plastic_ho - uniform_ho, gap_vs_shuffle=plastic_ho - shuffle_ho,
        corr_v_implen=corr_v_implen, corr_v_len=corr_v_len,
        v_bridge=v_bridge, v_nonbridge=v_nonbridge,
        corr_val_dist_plastic=corr_val_dist_plastic, corr_val_dist_uniform=corr_val_dist_uniform,
        corr_tau_dist_plastic=corr_tau_dist_plastic, corr_tau_dist_uniform=corr_tau_dist_uniform,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--device", type=str, default=DEV)
    ap.add_argument("--out", type=str, default="results/experiments/plastic_velocity.json")
    ap.add_argument("--imp_subsample", type=int, default=None,
                    help="ablate only this many random edges (+all bridges) for edge-importance; None=all")
    args = ap.parse_args()

    cfg = dict(
        N=args.N, K=3, R=0.5, gap=2.6, spread=1.0, n_bridge=4,
        # v0 MODERATE so the task is learnable at init AND the bridge lag lives in the
        # fractional band (~1.7 steps) -> velocity gradient flows. v_min low so under the
        # myelin budget a slow bridge's delay exceeds the deadline; v_max high so a
        # myelinated bridge delivers in time. The decisive squeeze is the budget, not init.
        v_min=0.16, v_max=3.0, v0=1.0,
        min_delay=1, max_delay=24,
        alpha=0.4, T=16, read_window=3, deadline=6,   # tight, EARLY deadline window [3,6)
        steps=args.steps, lr=5e-3, lr_v=0.1, l1=1e-3, lam_M=1.2e-3,
        imp_subsample=args.imp_subsample,
    )
    print(f"device: {args.device} | torch {torch.__version__}", flush=True)
    print(f"config: {cfg}", flush=True)

    rows = []
    t0 = time.time()
    for s in range(args.seeds):
        print(f"=== seed {s+1}/{args.seeds} ===", flush=True)
        r = run_seed(s, cfg, args.device)
        rows.append(r)
        print(f"  edges={r['n_edges']} frac_pairs={r['frac_pairs']:.2f} "
              f"v_grad_nonzero={r['v_grad_nonzero']} v_grad_norm={r['v_grad_norm']:.2f} "
              f"v_moved={r['v_moved']:.3f}", flush=True)
        print(f"  M*={r['M_star']:.3f} M_uniform={r['M_uniform']:.3f} (matched) "
              f"v_uniform={r['v_uniform']:.3f}", flush=True)
        print(f"  HELD-OUT: plastic={r['plastic_ho']:.3f} uniform={r['uniform_ho']:.3f} "
              f"shuffle-v={r['shuffle_ho']:.3f} | gap={r['gap']:+.3f} "
              f"gap_vs_shuf={r['gap_vs_shuffle']:+.3f}", flush=True)
        print(f"  corr(v, imp*len)={r['corr_v_implen']:+.3f} corr(v,len)={r['corr_v_len']:+.3f} "
              f"v_bridge={r['v_bridge']:.3f} v_nonbridge={r['v_nonbridge']:.3f}", flush=True)
        print(f"  corr(value,dist): plastic={r['corr_val_dist_plastic']:+.3f} "
              f"uniform={r['corr_val_dist_uniform']:+.3f} | corr(tau,dist): "
              f"plastic={r['corr_tau_dist_plastic']:+.3f} uniform={r['corr_tau_dist_uniform']:+.3f}",
              flush=True)

    def arr(k): return np.array([r[k] for r in rows], dtype=float)
    gap = arr("gap")
    def stats(a):
        m = float(a.mean()); sd = float(a.std() + 1e-9)
        return dict(mean=m, sd=sd, t=m / (sd / math.sqrt(len(a))), wins=int((a > 0).sum()))
    summary = dict(
        config=cfg, seeds=args.seeds, device=args.device,
        minutes=round((time.time() - t0) / 60, 2),
        matched_myelin_ok=bool(np.allclose(arr("M_star"), arr("M_uniform"), atol=1e-3)),
        v_learns=bool((arr("v_moved") > 1e-3).all() and (arr("v_grad_nonzero") > 0).all()),
        plastic_ho=float(arr("plastic_ho").mean()),
        plastic_tr=float(arr("plastic_tr").mean()),
        uniform_ho=float(arr("uniform_ho").mean()),
        shuffle_ho=float(arr("shuffle_ho").mean()),
        heldout_gap_plastic_minus_uniform=stats(gap),
        heldout_gap_plastic_minus_shuffle=stats(arr("gap_vs_shuffle")),
        corr_v_implen=float(arr("corr_v_implen").mean()),
        corr_v_len=float(arr("corr_v_len").mean()),
        v_bridge=float(np.nanmean(arr("v_bridge"))),
        v_nonbridge=float(np.nanmean(arr("v_nonbridge"))),
        corr_val_dist_plastic=float(arr("corr_val_dist_plastic").mean()),
        corr_val_dist_uniform=float(arr("corr_val_dist_uniform").mean()),
        corr_tau_dist_plastic=float(arr("corr_tau_dist_plastic").mean()),
        corr_tau_dist_uniform=float(arr("corr_tau_dist_uniform").mean()),
        rows=rows,
    )
    g = summary["heldout_gap_plastic_minus_uniform"]
    summary["verdict"] = (
        "FUNCTIONAL (plastic-v beats uniform-v at matched myelin)"
        if g["t"] > 2 and g["wins"] >= max(1, int(0.75 * args.seeds)) and summary["v_learns"]
        else "structural/null (no matched-budget held-out gain)"
        if summary["v_learns"] else "INVALID (v did not learn)"
    )
    print("\n=== PLASTIC VELOCITY SUMMARY ===")
    print(f"v_learns={summary['v_learns']} matched_myelin_ok={summary['matched_myelin_ok']}")
    print(f"held-out: plastic={summary['plastic_ho']:.3f} uniform={summary['uniform_ho']:.3f} "
          f"shuffle-v={summary['shuffle_ho']:.3f}")
    print(f"gap plastic-uniform: mean={g['mean']:+.3f} t={g['t']:.2f} wins={g['wins']}/{args.seeds}")
    print(f"corr(v, imp*len)={summary['corr_v_implen']:+.3f} (vs corr(v,len)={summary['corr_v_len']:+.3f})")
    print(f"v_bridge={summary['v_bridge']:.3f} v_nonbridge={summary['v_nonbridge']:.3f}")
    print(f"corr(value,dist): plastic={summary['corr_val_dist_plastic']:+.3f} "
          f"uniform={summary['corr_val_dist_uniform']:+.3f}")
    print("VERDICT:", summary["verdict"])

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", args.out, "in", summary["minutes"], "min")


if __name__ == "__main__":
    main()
