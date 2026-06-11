"""THALAMOCORTICAL ISOCHRONY IN-SILICO: the myelinate-LONG pole, made specific.

WHAT SALAMI (PNAS 2003, doi:10.1073/pnas.0937380100) SHOWS, AND WHAT WE REPRODUCE.
The thalamocortical system has near-CONSTANT afferent latency (~2 ms) from thalamus to
cortex DESPITE a WIDE spread of afferent path lengths, because regional myelination makes
the long, variable thalamus->white-matter leg conduct ~10x FASTER than the short, near-
constant intracortical leg (CV 3.28 vs 0.33 m/s). This is the SYNCHRONY / equal-arrival
("isochrony") pole of our two-time-economies sign flip: under a SYNCHRONY objective the net
should myelinate the LONG axons to equalize arrival time, the OPPOSITE of the task/economy
pole (which speeds SHORT, high-|W| edges; corr(v,dist)<0). pareto_dissociation.json already
gives the WEAK generic version (corr(v,dist)=+0.33 over an unstructured radius graph). This
script makes the SHARP, Salami-shaped version: a TWO-LEG architecture in which the long
afferent leg is widely variable and the short relay leg is near-constant, so the equal-
arrival solution REQUIRES a ~10x velocity ratio between the legs -- recovering Salami's
number from an objective, not by construction.

ARCHITECTURE (the load-bearing geometry).
  - One TARGET population (cortical layer-4 analogue): n_tgt units, tight cluster.
  - A SOURCE/thalamus population: n_src units placed at WIDELY VARYING distances d_k from the
    target (uniform in [d_lo, d_hi], e.g. 4..16 lattice units). Each source carries one
    afferent edge to a target unit: the LONG, VARIABLE leg.
  - A short RELAY/intracortical leg: target<->target local edges of near-CONSTANT short
    distance (radius graph inside the cluster), the constant-latency leg.
  Edges thus partition cleanly into LEG_LONG (source->target afferents, variable length) and
  LEG_SHORT (target<->target, ~constant short length). This partition is the per-leg label the
  inverse will try to recover.

OBJECTIVE (synchrony / common arrival, NOT a readout task).
  A pulse is injected at ALL source units at t=0. We want every afferent to ARRIVE at its
  target at the SAME time. Two equivalent, re-sculpting-PROOF readouts of synchrony, both on
  the velocity field directly (no trainable W can rebuild them):
    (S1) L_sync = Var over the long-leg edges of arrival lag tau_ij = d_ij / v_ij.
         Salami's "constant latency" == low Var(tau) on the variable-length afferents.
    (S2) common-arrival peak: cross-edge variance of the per-target first-arrival time of the
         injected pulse measured from the DYNAMICS (a forward pass), so synchrony is read off
         activity, not just the tau formula. (S1) is the gradient driver; (S2) is a dynamics
         CHECK that the formula-level synchrony shows up in real arrivals.
  Velocity budget: the SAME shared-speed envelope as pareto_dissociation,
        L_bud = mu * mean_edge(v_ij),
  so the net cannot cheat by globally slowing the short leg; it must DECIDE WHERE to spend
  speed. The only way to equalize variable-length afferent arrivals under a finite speed
  envelope is to make the LONG afferents FAST and leave the short relay slow => a velocity
  RATIO v_long/v_short that should track the afferent length spread (Salami ~10x).

THE PREDICTION (quantitative, falsifiable).
  Let R_ratio = mean v on LEG_LONG / mean v on LEG_SHORT after synchrony training.
  - PERFECT isochrony on the long leg requires v_long proportional to d (so tau const). With
    afferent lengths spanning [d_lo,d_hi] and a short leg at ~d_short, the equal-tau solution
    pins v_long/v_short ~ (mean d_long)/d_short. For d_long in [4,16] (mean 10) and d_short~1,
    that is ~10x -- the Salami ratio falls out of the GEOMETRY SPREAD, not a tuned constant.
  - corr(v, dist) > 0 on the long leg (myelinate-LONG), and Var(tau_long) drops sharply vs the
    uniform-v / economy-pole baseline.
  CONTROLS:
    * ECONOMY POLE (w=0, pure throughput-style budget min / no synchrony): corr(v,dist) on the
      long leg should be <=0 and R_ratio ~1 -- the sign flip, in-architecture.
    * SHUFFLE-leg null: permute the trained v multiset across all edges (same budget, same
      histogram, leg structure destroyed) -> Var(tau) collapses back up, R_ratio ~1. If shuffle
      reproduces the ratio, the result is a histogram artifact, not allocation -> NULL.
    * UNIFORM-v at matched budget: a single scalar v cannot equalize variable-length arrivals
      (Var(tau) stays high) -> shows allocation, not budget, buys isochrony.

WHY IT IS RE-SCULPTING-PROOF (the named wall from MEMORY).
  The functional-probe nulls in this repo all die because a trainable recurrent W re-sculpts a
  scrambled-delay code back to matched accuracy. Here the objective is DEFINED ON THE VELOCITY
  FIELD / ARRIVAL TIMES, not on a decoded readout: there is no W that can make variable-length
  afferents arrive together except by changing the per-edge velocities. Synchrony of arrival is
  a property of tau = d/v; W reweights contributions but cannot move an arrival time. So the
  myelinate-long allocation is the ONLY degree of freedom that lowers L_sync -- exactly the
  frozen-input / matched-budget-allocation regime where geometry is functional. (This is the
  same reason the deadline plastic-v result survives: arrival timing is upstream of W.)

THE INVERSE HALF (recover the regional CV pattern from activity).
  After synchrony training, inject the pulse, record hidden trajectories, and recover a PER-LEG
  conduction velocity by the repo's delayed-field residual inverse (scan v, fit free kernel,
  pick the v that best predicts next-step field) -- but fit v SEPARATELY on the long-leg edges
  and the short-leg edges (two tau sub-matrices). Predict: recovered v_long/v_short ~ trained
  ratio ~10x, and a GEOMETRY-SHUFFLE null (scramble which source sits at which distance) erases
  the recovered ratio. This mirrors foreigninv_intracranial's scan + geometry-shuffle null,
  extended from one global v to a two-region field -- i.e. our velocity inverse RECOVERS the
  Salami regional-myelination pattern from simulated activity.

A NULL IS INFORMATIVE. If the synchrony objective does NOT drive R_ratio toward the geometry-
implied ~10x (e.g. shuffle reproduces it, or uniform matches Var(tau)), report honestly: the
thalamocortical isochrony is then NOT reproduced as an allocation law in this model.

STATUS: scaffold. The geometry, objective, controls, and inverse are wired; defaults are set
to land afferent lags in the active fractional band. Run with --smoke first to confirm the
velocity gradient is alive and Var(tau) actually drops, then scale seeds/steps.

Run:
  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=2 \
  python experiments/thalamocortical_isochrony.py --device mps --seeds 5 \
    --out results/experiments/thalamocortical_isochrony.json
  python experiments/thalamocortical_isochrony.py --smoke   # fast sanity (1 seed, few steps)
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

# reuse the validated plastic-velocity machinery verbatim (model, soft delays, budget, eval)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import plastic_velocity as pv  # PlasticVelocityRNN, make_codes, corr, DEV

DEV = pv.DEV


# =====================================================================================
# GEOMETRY: two-leg thalamocortical architecture.
#   target cluster (cortex) + source ring (thalamus) at widely varying afferent distance.
# Returns a graph dict in the SAME schema build_graph() produces so PlasticVelocityRNN
# consumes it unchanged, PLUS a per-edge leg label (long afferent vs short relay).
# =====================================================================================
def thalamocortical_geometry(n_tgt, n_src, seed, d_lo, d_hi, tgt_spread, relay_R):
    """Place a tight target cluster and source units at variable afferent distance.

    - target units: ids [0, n_tgt) in a tight blob of size `tgt_spread` (short relay leg).
    - source units: ids [n_tgt, n_tgt+n_src) placed radially OUTWARD at distance d_k drawn
      uniformly in [d_lo, d_hi] from a target anchor -> the LONG, VARIABLE afferent leg.
    Each source k connects to one target (round-robin) by a single afferent edge.
    """
    g = torch.Generator().manual_seed(seed)
    N = n_tgt + n_src
    # target blob around the origin
    tgt_pos = (torch.rand(n_tgt, 2, generator=g) - 0.5) * tgt_spread
    anchor = tgt_pos.mean(0, keepdim=True)
    # sources at variable radius, random angle, around the anchor
    radii = d_lo + (d_hi - d_lo) * torch.rand(n_src, generator=g)        # variable afferent length
    ang = 2 * math.pi * torch.rand(n_src, generator=g)
    src_pos = anchor + torch.stack([radii * torch.cos(ang), radii * torch.sin(ang)], 1)
    pos = torch.cat([tgt_pos, src_pos], 0)
    d = torch.cdist(pos, pos)

    tgt = torch.arange(0, n_tgt)
    src = torch.arange(n_tgt, N)

    # edges: short relay leg = target<->target within relay_R; long leg = each src->one tgt.
    edge = torch.zeros(N, N, dtype=torch.bool)
    relay = (d <= relay_R) & (d > 0)
    relay[n_tgt:, :] = False; relay[:, n_tgt:] = False                    # relay only among targets
    edge |= relay
    leg_long = torch.zeros(N, N, dtype=torch.bool)                        # afferent (variable) leg
    for k in range(n_src):
        s = n_tgt + k
        t = int(tgt[k % n_tgt])
        edge[s, t] = True; edge[t, s] = True
        leg_long[s, t] = True; leg_long[t, s] = True
    edge.fill_diagonal_(False)
    leg_short = relay & ~leg_long

    return dict(pos=pos, d=d, edge=edge, src=src, tgt=tgt,
                bridge_mask=leg_long.clone(),         # reuse bridge_mask slot = long leg
                leg_long=leg_long, leg_short=leg_short, n_tgt=n_tgt, n_src=n_src,
                afferent_radii=radii)


# =====================================================================================
# SYNCHRONY OBJECTIVE on the velocity field (re-sculpting-proof: defined on tau=d/v).
# =====================================================================================
def long_leg_edge_index(m, g):
    """Indices into m.ui (undirected edge list) that are LONG-leg afferents / SHORT-leg relay."""
    ui = m.ui
    ll = g["leg_long"][ui[0].cpu(), ui[1].cpu()]
    ls = g["leg_short"][ui[0].cpu(), ui[1].cpu()]
    return ll, ls


def sync_loss(m, long_idx):
    """L_sync = Var over LONG-leg afferent edges of arrival lag tau_ij = d/v (differentiable).

    Low Var(tau_long) == all variable-length afferents arrive together == Salami isochrony.
    """
    tau = m.tau_matrix()                                   # (N,N), differentiable in v
    tlong = tau[m.ui[0], m.ui[1]][long_idx]
    return tlong.var(unbiased=False)


def speed_budget(m, mu):
    """Shared velocity envelope (same as pareto_dissociation): pay mu per unit mean speed."""
    return mu * m.edge_velocity().mean()


def train_sync(m, g, *, w_sync, mu, steps, lr, lr_v, device, seed, T, log_every=1e9):
    """Train per-edge velocity (and W,b as nuisance) to MINIMIZE arrival-time variance on the
    long leg under a shared speed budget. w_sync trades synchrony against the budget:
      loss = w_sync * L_sync + (1-w_sync)*0 + L_bud .
    w_sync=0 reduces to the ECONOMY/budget pole (no synchrony pressure)."""
    long_idx, _ = long_leg_edge_index(m, g)
    long_idx = long_idx.to(device)
    vparams, oparams = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        (vparams if n == "g" else oparams).append(p)
    groups = [dict(params=oparams, lr=lr)]
    if vparams:
        groups.append(dict(params=vparams, lr=lr_v))
    opt = torch.optim.Adam(groups)
    gen = torch.Generator().manual_seed(555 + seed)
    for it in range(steps):
        opt.zero_grad()
        # drive a pulse through so W,b stay in a sane regime (keeps dynamics bounded);
        # the synchrony gradient flows through tau=d/v, not the readout.
        code = pv.make_codes(32, m.K, gen, device)
        _ = m.propagate(code, T)
        loss = w_sync * sync_loss(m, long_idx)
        if mu > 0:
            loss = loss + speed_budget(m, mu)
        loss.backward()
        nn.utils.clip_grad_norm_([p for grp in groups for p in grp["params"]], 1.0)
        opt.step()
        if it % int(log_every) == 0:
            print(f"      step {it}/{steps} loss={loss.item():.5f}", flush=True)
    return m


# =====================================================================================
# DYNAMICS-LEVEL synchrony check (S2) + per-leg velocity readout.
# =====================================================================================
@torch.no_grad()
def arrival_var_from_dynamics(m, g, T, device, thresh=0.05):
    """Inject a pulse at all source units, measure each target's FIRST-arrival time, return the
    cross-target variance of arrival (the dynamics-level synchrony, independent of the tau formula)."""
    code = torch.ones(1, m.K, device=device)              # constant drive into source units
    H = m.propagate(code, T)[:, 0, :]                      # (T, N)
    tgt = g["tgt"].to(device)
    arr = []
    for t_id in tgt.tolist():
        sig = H[:, t_id].abs()
        peak = sig.max()
        if peak < 1e-6:
            continue
        first = int((sig > thresh * peak).float().argmax().item())
        arr.append(first)
    arr = np.array(arr, float)
    return float(arr.var()) if arr.size > 1 else float("nan")


@torch.no_grad()
def leg_velocity_stats(m, g):
    """Mean trained velocity on the long afferent leg vs short relay leg, and the RATIO."""
    v = m.edge_velocity().detach().cpu().numpy()
    ll, ls = long_leg_edge_index(m, g)
    ll = ll.cpu().numpy(); ls = ls.cpu().numpy()
    length = m.d[m.ui[0], m.ui[1]].detach().cpu().numpy()
    v_long = float(v[ll].mean()) if ll.any() else float("nan")
    v_short = float(v[ls].mean()) if ls.any() else float("nan")
    ratio = v_long / v_short if (v_short and not math.isnan(v_short)) else float("nan")
    # corr(v, dist) restricted to the long leg = the myelinate-LONG sign
    if ll.sum() > 2:
        corr_v_dist_long = pv.corr(v[ll], length[ll])
    else:
        corr_v_dist_long = float("nan")
    return dict(v_long=v_long, v_short=v_short, ratio=ratio, corr_v_dist_long=corr_v_dist_long)


@torch.no_grad()
def tau_var_long(m, g):
    """Var of arrival lag tau=d/v over the long afferent leg (the synchrony axis, formula-level)."""
    tau = m.tau_matrix().detach()
    ll, _ = long_leg_edge_index(m, g)
    tl = tau[m.ui[0], m.ui[1]][ll].cpu().numpy()
    return float(np.var(tl)), float(np.std(tl) / (np.mean(tl) + 1e-12))


# =====================================================================================
# INVERSE: per-leg conduction-velocity recovery from activity (mirror of the ECoG inverse).
#   Record pulse-driven hidden trajectories, scan a candidate velocity for EACH leg
#   separately, pick the v minimizing one-step delayed-field residual. Geometry-shuffle null
#   scrambles which source sits at which distance (destroys the leg's distance structure).
# =====================================================================================
@torch.no_grad()
def record_field(m, T, device, n_trials=8):
    """Pulse-driven hidden trajectories, (n_trials, T, N)."""
    gen = torch.Generator().manual_seed(99)
    codes = pv.make_codes(n_trials, m.K, gen, device)
    H = m.propagate(codes, T)                              # (T, B, N)
    return H.permute(1, 0, 2).cpu().numpy()               # (B, T, N)


def _residual_at_v(field, dist, v, edge_mask, min_delay, max_delay, ridge=1e-2):
    """One-step delayed-field residual at candidate velocity v on a given edge set.
    Predict field[:,1:,:] from delayed neighbor field at tau=round(dist/v); free-kernel ridge fit.
    Returns normalized residual (lower=better-explained). Mirrors scan_freekernel_field."""
    B, T, N = field.shape
    tau = np.clip(np.round(dist / v), min_delay, max_delay).astype(int)
    # build delayed predictor: for each unit i, sum over masked neighbors j of field_j[t-tau_ij]
    Xs, Ys = [], []
    for t in range(max_delay, T):
        pred = np.zeros((B, N))
        for i in range(N):
            js = np.where(edge_mask[i])[0]
            if js.size == 0:
                continue
            for j in js:
                pred[:, i] += field[:, t - tau[i, j], j]
        Xs.append(pred); Ys.append(field[:, t, :])
    X = np.concatenate(Xs, 0); Y = np.concatenate(Ys, 0)   # (M, N)
    # per-unit scalar gain ridge fit (free kernel, rank-1 per unit): a_i = <x_i,y_i>/(<x_i,x_i>+ridge)
    num = (X * Y).sum(0); den = (X * X).sum(0) + ridge
    a = num / den
    res = Y - a[None, :] * X
    ss_res = (res ** 2).sum(); ss_tot = ((Y - Y.mean(0)) ** 2).sum() + 1e-12
    return float(ss_res / ss_tot)


def recover_leg_velocity(field, dist, edge_mask, cands, min_delay, max_delay):
    """Scan candidate velocities, return v that minimizes residual on the masked leg edges."""
    res = [_residual_at_v(field, dist, c, edge_mask, min_delay, max_delay) for c in cands]
    res = np.array(res)
    k = int(np.argmin(res))
    return float(cands[k]), res, (0 < k < len(cands) - 1)


def _residual_at_tau(field, tau, edge_mask, min_delay, max_delay, ridge=1e-2):
    """Same one-step delayed-field residual as _residual_at_v, but for an ARBITRARY per-edge tau
    matrix (so the velocity model need not be a single scalar)."""
    B, T, N = field.shape
    tau = np.clip(tau, min_delay, max_delay).astype(int)
    Xs, Ys = [], []
    for t in range(max_delay, T):
        pred = np.zeros((B, N))
        for i in range(N):
            js = np.where(edge_mask[i])[0]
            if js.size == 0:
                continue
            for j in js:
                pred[:, i] += field[:, t - tau[i, j], j]
        Xs.append(pred); Ys.append(field[:, t, :])
    X = np.concatenate(Xs, 0); Y = np.concatenate(Ys, 0)
    num = (X * Y).sum(0); den = (X * X).sum(0) + ridge
    a = num / den
    res = Y - a[None, :] * X
    ss_res = (res ** 2).sum(); ss_tot = ((Y - Y.mean(0)) ** 2).sum() + 1e-12
    return float(ss_res / ss_tot)


def recover_isochrony_exponent(field, dist, edge_mask, p_cands, v0_cands, d_ref, min_delay, max_delay):
    """Recover the velocity-distance EXPONENT p in v(d)=v0*(d/d_ref)^p on a leg, by the delayed-field
    residual. p=0 => uniform velocity (tau grows with d); p=1 => isochrony (v proportional to d, tau
    approximately constant). For each p the overall speed scale v0 is chosen by an inner 1-D scan
    (p sets the SHAPE of tau vs distance; v0 only shifts the magnitude). Returns the p minimizing the
    residual. The synchrony pole should recover p~1; a geometry-shuffle null should collapse to p~0."""
    dref = max(d_ref, 1e-6)
    res_p = []
    for p in p_cands:
        shape = np.power(np.clip(dist, 1e-6, None) / dref, p)        # d-dependence of velocity
        best = np.inf
        for v0 in v0_cands:
            tau = np.round(dist / np.clip(v0 * shape, 1e-6, None))
            r = _residual_at_tau(field, tau, edge_mask, min_delay, max_delay)
            if r < best:
                best = r
        res_p.append(best)
    res_p = np.array(res_p)
    k = int(np.argmin(res_p))
    return float(p_cands[k]), res_p, (0 < k < len(p_cands) - 1)


def run_inverse(m, g, cands, T, device, seed):
    """Recover v_long, v_short from pulse-driven activity; geometry-shuffle null on the long leg."""
    field = record_field(m, T, device)
    dist = m.d.detach().cpu().numpy()
    long_mask = g["leg_long"].cpu().numpy()
    short_mask = g["leg_short"].cpu().numpy()
    mind, maxd = m.min_delay, m.max_delay
    vhat_long, _, in_long = recover_leg_velocity(field, dist, long_mask, cands, mind, maxd)
    vhat_short, _, in_short = recover_leg_velocity(field, dist, short_mask, cands, mind, maxd)
    # geometry-shuffle null: scramble the afferent distances among sources, re-fit long leg.
    rng = np.random.default_rng(seed + 7)
    dist_shuf = dist.copy()
    srcs = g["src"].cpu().numpy()
    perm = rng.permutation(len(srcs))
    # permute the source rows/cols distance to targets (destroys which source is far/near)
    for a, b in zip(srcs, srcs[perm]):
        dist_shuf[a, :] = dist[b, :]
        dist_shuf[:, a] = dist[:, b]
    vhat_long_null, _, _ = recover_leg_velocity(field, dist_shuf, long_mask, cands, mind, maxd)

    # PARAMETRIC isochrony-exponent inverse (the right model for the v∝d synchrony solution):
    # recover p in v(d)=v0*(d/d_ref)^p on the long leg; isochrony => p~1, geometry-shuffle null => p~0.
    d_long = dist[long_mask]
    d_ref = float(np.median(d_long[d_long > 0])) if (d_long > 0).any() else 1.0
    p_cands = np.linspace(0.0, 1.5, 16)
    v0_cands = np.linspace(cands[0], cands[-1], 8)
    p_hat_long, _, p_interior = recover_isochrony_exponent(field, dist, long_mask, p_cands, v0_cands, d_ref, mind, maxd)
    p_hat_long_null, _, _ = recover_isochrony_exponent(field, dist_shuf, long_mask, p_cands, v0_cands, d_ref, mind, maxd)
    return dict(vhat_long=vhat_long, vhat_short=vhat_short,
                vhat_ratio=(vhat_long / vhat_short if vhat_short else float("nan")),
                vhat_long_null=vhat_long_null, interior_long=in_long, interior_short=in_short,
                p_hat_long=p_hat_long, p_hat_long_null=p_hat_long_null, p_interior=p_interior)


# =====================================================================================
# One seed: train synchrony pole + economy pole + shuffle/uniform controls; run inverse.
# =====================================================================================
def run_seed(seed, cfg, device):
    g = thalamocortical_geometry(cfg["n_tgt"], cfg["n_src"], seed, cfg["d_lo"], cfg["d_hi"],
                                 cfg["tgt_spread"], cfg["relay_R"])
    mk = lambda mode, fv=None: pv.PlasticVelocityRNN(
        g, cfg["K"], velocity_mode=mode, v_min=cfg["v_min"], v_max=cfg["v_max"], v0=cfg["v0"],
        min_delay=cfg["min_delay"], max_delay=cfg["max_delay"], alpha=cfg["alpha"],
        seed=seed, fixed_v=fv).to(device)
    T = cfg["T"]

    # diagnostics: afferent lags in active fractional band? v-gradient alive?
    long_idx, _ = long_leg_edge_index(mk("plastic"), g)
    de = g["d"][g["leg_long"]].reshape(-1)
    tau0 = (de / cfg["v0"]).clamp(cfg["min_delay"], cfg["max_delay"])
    frac_long = float(((tau0 % 1) > 1e-3).float().mean())

    # ---- SYNCHRONY POLE (myelinate-long) ----
    ms = mk("plastic")
    # velocity gradient check on the synchrony loss BEFORE training (guards integer-lag null)
    li = long_idx.to(device)
    gv = torch.autograd.grad(sync_loss(ms, li), ms.g, retain_graph=False)[0]
    v_grad_nonzero = int((gv.abs() > 1e-12).sum()); v_grad_norm = float(gv.norm())
    train_sync(ms, g, w_sync=1.0, mu=cfg["mu"], steps=cfg["steps"], lr=cfg["lr"],
               lr_v=cfg["lr_v"], device=device, seed=seed, T=T)
    sync_stats = leg_velocity_stats(ms, g)
    var_sync, cv_sync = tau_var_long(ms, g)
    adyn_sync = arrival_var_from_dynamics(ms, g, T, device)
    M_sync = float(ms.myelin().item())

    # ---- ECONOMY POLE (w_sync=0: budget only, no synchrony pressure) ----
    me = mk("plastic")
    train_sync(me, g, w_sync=0.0, mu=cfg["mu"], steps=cfg["steps"], lr=cfg["lr"],
               lr_v=cfg["lr_v"], device=device, seed=seed, T=T)
    econ_stats = leg_velocity_stats(me, g)
    var_econ, cv_econ = tau_var_long(me, g)

    # ---- SHUFFLE-leg null: trained sync velocities reassigned across edges ----
    v_sync = ms.edge_velocity().detach()
    perm = torch.randperm(v_sync.numel(), generator=torch.Generator().manual_seed(321 + seed))
    v_shuf = v_sync[perm.to(v_sync.device)].cpu()
    msh = mk("fixed", fv=v_shuf)
    shuf_stats = leg_velocity_stats(msh, g)
    var_shuf, cv_shuf = tau_var_long(msh, g)

    # ---- UNIFORM-v at matched budget: single scalar v, can it equalize arrivals? ----
    dl = g["d"][ms.ui[0].cpu(), ms.ui[1].cpu()]
    v_uniform = float(np.clip(cfg["v_min"] + M_sync / max(float(dl.sum()), 1e-9),
                              cfg["v_min"], cfg["v_max"]))
    mu_ = mk("uniform"); mu_.uniform_v.fill_(v_uniform)
    var_uni, cv_uni = tau_var_long(mu_, g)

    # ---- INVERSE: per-leg velocity recovery from synchrony-pole activity ----
    cands = np.linspace(cfg["v_min"] * 1.1, cfg["v_max"] * 0.95, cfg["n_cands"])
    inv = run_inverse(ms, g, cands, T, device, seed)

    # geometry-implied ideal ratio: equal tau on long leg needs v_long ~ d_long; vs short leg ~ d_short
    mean_d_long = float(g["d"][g["leg_long"]].mean())
    mean_d_short = float(g["d"][g["leg_short"]].mean()) if g["leg_short"].any() else float("nan")
    ideal_ratio = mean_d_long / mean_d_short if mean_d_short else float("nan")

    return dict(
        seed=seed, frac_long=frac_long, v_grad_nonzero=v_grad_nonzero, v_grad_norm=v_grad_norm,
        # synchrony pole (the headline myelinate-long result)
        ratio_sync=sync_stats["ratio"], v_long_sync=sync_stats["v_long"],
        v_short_sync=sync_stats["v_short"], corr_v_dist_long_sync=sync_stats["corr_v_dist_long"],
        var_tau_long_sync=var_sync, cv_tau_long_sync=cv_sync, arrival_var_dyn_sync=adyn_sync,
        # economy pole (sign-flip control)
        ratio_econ=econ_stats["ratio"], corr_v_dist_long_econ=econ_stats["corr_v_dist_long"],
        var_tau_long_econ=var_econ,
        # nulls
        ratio_shuf=shuf_stats["ratio"], var_tau_long_shuf=var_shuf,
        var_tau_long_uniform=var_uni, v_uniform=v_uniform,
        # geometry-implied target
        ideal_ratio=ideal_ratio, mean_d_long=mean_d_long, mean_d_short=mean_d_short,
        # inverse
        vhat_ratio=inv["vhat_ratio"], vhat_long=inv["vhat_long"], vhat_short=inv["vhat_short"],
        vhat_long_null=inv["vhat_long_null"],
        p_hat_long=inv["p_hat_long"], p_hat_long_null=inv["p_hat_long_null"],
    )


DEFAULT_CFG = dict(
    n_tgt=24, n_src=24, K=4,
    d_lo=4.0, d_hi=16.0,            # afferent length spread -> ideal ratio ~ (mean 10)/short
    tgt_spread=1.2, relay_R=0.9,    # short relay leg ~ constant, near-1 lattice unit
    v_min=0.3, v_max=8.0, v0=2.0,   # v0 puts afferent lag ~ (4..16)/2 = 2..8 steps (fractional band)
    min_delay=1, max_delay=24,
    alpha=0.4, T=28,
    mu=0.02, steps=500, lr=5e-3, lr_v=0.15,
    n_cands=24,
)


def stats(a):
    a = np.asarray([x for x in a if not (isinstance(x, float) and math.isnan(x))], float)
    if a.size == 0:
        return dict(mean=float("nan"), sd=float("nan"), t=float("nan"))
    m = float(a.mean()); sd = float(a.std() + 1e-9)
    return dict(mean=m, sd=sd, t=m / (sd / math.sqrt(len(a))), n=len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", type=str, default=DEV)
    ap.add_argument("--out", type=str, default="results/experiments/thalamocortical_isochrony.json")
    ap.add_argument("--smoke", action="store_true", help="1 seed, few steps, sanity only")
    # calibration knobs (un-clip the ratio + land mean_d_short ~1.0-1.5 for a Salami-shaped ideal)
    for k in ["v_max", "v0", "v_min", "d_lo", "d_hi", "relay_R", "tgt_spread", "mu", "alpha", "T"]:
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=None, dest=k)
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    for k in ["v_max", "v0", "v_min", "d_lo", "d_hi", "relay_R", "tgt_spread", "mu", "alpha"]:
        if getattr(args, k, None) is not None:
            cfg[k] = getattr(args, k)
    if getattr(args, "T", None) is not None:
        cfg["T"] = int(args.T)
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.smoke:
        cfg["steps"] = 60; args.seeds = 1
    print(f"device={args.device} torch={torch.__version__}")
    print(f"config={cfg}\n")

    rows, t0 = [], time.time()
    for s in range(args.seeds):
        print(f"=== seed {s+1}/{args.seeds} ===", flush=True)
        r = run_seed(s, cfg, args.device)
        rows.append(r)
        print(f"  frac_long={r['frac_long']:.2f} v_grad_nonzero={r['v_grad_nonzero']} "
              f"grad_norm={r['v_grad_norm']:.3f} ideal_ratio={r['ideal_ratio']:.2f}", flush=True)
        print(f"  SYNC pole: v_long/v_short={r['ratio_sync']:.2f} "
              f"corr(v,dist|long)={r['corr_v_dist_long_sync']:+.3f} "
              f"Var(tau_long)={r['var_tau_long_sync']:.3f} arrival_var_dyn={r['arrival_var_dyn_sync']:.3f}",
              flush=True)
        print(f"  ECON pole: ratio={r['ratio_econ']:.2f} corr(v,dist|long)={r['corr_v_dist_long_econ']:+.3f} "
              f"Var(tau_long)={r['var_tau_long_econ']:.3f}", flush=True)
        print(f"  NULLS: shuffle ratio={r['ratio_shuf']:.2f} Var={r['var_tau_long_shuf']:.3f} | "
              f"uniform Var={r['var_tau_long_uniform']:.3f}", flush=True)
        print(f"  INVERSE: vhat_long/vhat_short={r['vhat_ratio']:.2f} "
              f"(long={r['vhat_long']:.2f} short={r['vhat_short']:.2f} null={r['vhat_long_null']:.2f})",
              flush=True)
        print(f"  ISO-EXP inverse: p_hat_long={r['p_hat_long']:.2f} (null={r['p_hat_long_null']:.2f}) "
              f"[p~1=isochrony recovered, p~0=null]", flush=True)

    def arr(k): return [r[k] for r in rows]
    summary = dict(
        config=cfg, seeds=args.seeds, device=args.device,
        minutes=round((time.time() - t0) / 60, 2),
        v_grad_alive=bool(all(r["v_grad_nonzero"] > 0 for r in rows)),
        ideal_ratio=stats(arr("ideal_ratio")),
        ratio_sync=stats(arr("ratio_sync")),
        ratio_econ=stats(arr("ratio_econ")),
        ratio_shuf=stats(arr("ratio_shuf")),
        corr_v_dist_long_sync=stats(arr("corr_v_dist_long_sync")),
        corr_v_dist_long_econ=stats(arr("corr_v_dist_long_econ")),
        var_tau_long_sync=stats(arr("var_tau_long_sync")),
        var_tau_long_econ=stats(arr("var_tau_long_econ")),
        var_tau_long_shuf=stats(arr("var_tau_long_shuf")),
        var_tau_long_uniform=stats(arr("var_tau_long_uniform")),
        vhat_ratio=stats(arr("vhat_ratio")),
        vhat_long_null=stats(arr("vhat_long_null")),
        p_hat_long=stats(arr("p_hat_long")),
        p_hat_long_null=stats(arr("p_hat_long_null")),
        rows=rows,
    )
    # VERDICT: myelinate-long reproduced if sync ratio >> econ ratio ~1, corr(v,dist|long)>0,
    # Var(tau_long) collapses vs uniform AND vs shuffle, and the inverse recovers the ratio.
    rs, re_ = summary["ratio_sync"], summary["ratio_econ"]
    vs, vu, vsh = summary["var_tau_long_sync"], summary["var_tau_long_uniform"], summary["var_tau_long_shuf"]
    iso = (rs["mean"] > 3.0 and rs["mean"] > 2 * max(re_["mean"], 1.0)
           and summary["corr_v_dist_long_sync"]["mean"] > 0.3
           and vs["mean"] < 0.5 * vu["mean"] and vs["mean"] < 0.5 * vsh["mean"]
           and summary["v_grad_alive"])
    # inverse "recovers" ONLY if a MAJORITY of seeds individually recover -- guards against a single
    # outlier seed inflating the mean (the scalar-per-leg scan is misspecified for the v-proportional-to-d
    # synchrony solution, so most seeds fail; mean-based thresholds give false positives).
    inv_seed_ok = [r["vhat_ratio"] > 1.8 and r["vhat_ratio"] > 1.5 * max(r["vhat_long_null"], 1e-9) for r in rows]
    summary["inv_seeds_recovered"] = f"{sum(inv_seed_ok)}/{len(rows)}"
    inv_ok = sum(inv_seed_ok) >= (len(rows) + 1) // 2
    summary["verdict"] = (
        "THALAMOCORTICAL ISOCHRONY REPRODUCED (myelinate-long pole + inverse recovers regional CV)"
        if iso and inv_ok else
        "PARTIAL (isochrony allocation seen; inverse recovery weak)" if iso else
        "NULL (synchrony objective did not produce the myelinate-long ratio)")

    print("\n=== THALAMOCORTICAL ISOCHRONY SUMMARY ===")
    print(f"v_grad_alive={summary['v_grad_alive']} ideal_ratio~{summary['ideal_ratio']['mean']:.2f}")
    print(f"v_long/v_short: SYNC={rs['mean']:.2f}+/-{rs['sd']:.2f} (t={rs['t']:.1f})  "
          f"ECON={re_['mean']:.2f}  SHUFFLE={summary['ratio_shuf']['mean']:.2f}")
    print(f"corr(v,dist|long): SYNC={summary['corr_v_dist_long_sync']['mean']:+.3f}  "
          f"ECON={summary['corr_v_dist_long_econ']['mean']:+.3f}")
    print(f"Var(tau_long): SYNC={vs['mean']:.3f}  UNIFORM={vu['mean']:.3f}  SHUFFLE={vsh['mean']:.3f}")
    print(f"INVERSE vhat_long/vhat_short={summary['vhat_ratio']['mean']:.2f}  "
          f"(geom-shuffle null={summary['vhat_long_null']['mean']:.2f})")
    print("VERDICT:", summary["verdict"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", args.out, "in", summary["minutes"], "min")


if __name__ == "__main__":
    main()
