"""AUDITORY-BRAINSTEM COINCIDENCE-DETECTOR ISOCHRONY in-silico (a SECOND isochrony system + a knockout).

WHAT THE BIOLOGY SHOWS (Seidl, Rubel & Barria 2014, J Neurosci 34:4914; Stange-Marten 2017 PNAS 114:E4851):
In the avian sound-localization circuit (nucleus magnocellularis -> laminaris, the Jeffress coincidence
detector), the TWO COLLATERALS of a single afferent axon individually tune CONDUCTION VELOCITY so that
signals from the two ears ARRIVE SIMULTANEOUSLY at the detector despite very different path lengths --
the longer (contralateral) branch is myelinated to conduct FASTER (3.69 vs 1.59 m/s; arrival times equal,
457 vs 399 us, p=0.26, despite >1600 um length difference). This is the SYNCHRONY/ISOCHRONY pole of our
two-time-economies sign flip, in a coincidence-detection architecture. The KNOCKOUT: this myelination
specialization is present in ITD-users (gerbil) but ABSENT in non-ITD species (mouse) -- isochrony emerges
ONLY where coincidence/timing is the objective.

WHAT WE REPRODUCE (re-sculpting-PROOF, like the thalamocortical test):
Architecture: n_det coincidence detectors; each receives one IPSI afferent (short branch, length d_ipsi)
and one CONTRA afferent (long branch, length d_contra > d_ipsi, varying across detectors). Per-edge
velocity is learnable under a shared speed budget. Objective:
  COINCIDENCE  L_coinc = mean_detector ( tau_contra - tau_ipsi )^2 ,   tau = d / v ,
i.e. the two branches must arrive together. Because the objective is defined on arrival times tau=d/v
(upstream of the recurrent weights W), no W can equalize variable-length branch arrivals -- per-branch
velocity allocation is the only degree of freedom. PREDICTION:
  - v_contra / v_ipsi  ->  d_contra / d_ipsi  (myelinate the LONG branch faster; the Seidl ratio).
  - corr(v, branch length) > 0 (isochrony pole); arrival difference |tau_contra - tau_ipsi| collapses.
  KNOCKOUT (the gerbil-vs-mouse natural experiment): train the SAME architecture with the coincidence
  objective OFF (budget only) -> NO velocity tuning, v_contra/v_ipsi ~ 1, arrival difference stays large.
  CONTROLS: velocity-shuffle null (trained v permuted across edges) and uniform-v at matched budget
  (a single scalar cannot equalize variable-length branch arrivals).

A NULL IS INFORMATIVE: if the coincidence objective does NOT drive v_contra/v_ipsi toward the length
ratio (or the knockout reproduces it), the auditory-brainstem isochrony is NOT reproduced as an
allocation law here -- report honestly.

Run:
  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=2 PYTHONPATH=.:src \
  python experiments/auditory_brainstem_isochrony.py --device mps --seeds 5
  python experiments/auditory_brainstem_isochrony.py --smoke
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import plastic_velocity as pv          # PlasticVelocityRNN, make_codes, corr, DEV
DEV = pv.DEV


# =====================================================================================
# GEOMETRY: n_det coincidence detectors; each gets an IPSI (short) and a CONTRA (long) afferent.
#   detector ids   : [0, n_det)
#   ipsi sources   : [n_det, 2*n_det)      source i -> detector i, short branch
#   contra sources : [2*n_det, 3*n_det)    source i -> detector i, long branch (length varies)
# Returns a graph dict in build_graph()'s schema (so PlasticVelocityRNN consumes it unchanged)
# plus per-edge leg_ipsi / leg_contra masks.
# =====================================================================================
def coincidence_geometry(n_det, seed, d_ipsi, d_contra_lo, d_contra_hi, det_spread, itd_span=0.0):
    g = torch.Generator().manual_seed(seed)
    N = 3 * n_det
    det_pos = (torch.rand(n_det, 2, generator=g) - 0.5) * det_spread          # tight detector cluster
    # ipsi source: short fixed offset; contra source: long variable offset (random angle)
    ang_i = 2 * math.pi * torch.rand(n_det, generator=g)
    ipsi_pos = det_pos + torch.stack([d_ipsi * torch.cos(ang_i), d_ipsi * torch.sin(ang_i)], 1)
    rc = d_contra_lo + (d_contra_hi - d_contra_lo) * torch.rand(n_det, generator=g)   # variable long-branch length
    ang_c = 2 * math.pi * torch.rand(n_det, generator=g)
    contra_pos = det_pos + torch.stack([rc * torch.cos(ang_c), rc * torch.sin(ang_c)], 1)
    pos = torch.cat([det_pos, ipsi_pos, contra_pos], 0)
    d = torch.cdist(pos, pos)

    edge = torch.zeros(N, N, dtype=torch.bool)
    leg_ipsi = torch.zeros(N, N, dtype=torch.bool)
    leg_contra = torch.zeros(N, N, dtype=torch.bool)
    for i in range(n_det):
        si, sc = n_det + i, 2 * n_det + i
        edge[si, i] = edge[i, si] = True; leg_ipsi[si, i] = leg_ipsi[i, si] = True
        edge[sc, i] = edge[i, sc] = True; leg_contra[sc, i] = leg_contra[i, sc] = True
    edge.fill_diagonal_(False)

    # swept Jeffress ITD place-map (per detector), DECORRELATED from branch-length difference so that
    # no single global velocity (which yields ITD ~ (d_contra-d_ipsi)/v, monotone in length) can hit it.
    itd_target = itd_span * (2 * torch.rand(n_det, generator=g) - 1)
    return dict(pos=pos, d=d, edge=edge,
                src=torch.arange(n_det, N), tgt=torch.arange(0, n_det),
                bridge_mask=leg_contra.clone(),          # reuse bridge_mask slot = long branch
                leg_ipsi=leg_ipsi, leg_contra=leg_contra, n_det=n_det,
                d_ipsi=d_ipsi, contra_radii=rc, itd_target=itd_target)


def branch_pairs(m, g):
    """Per-detector aligned ui-edge indices for the ipsi and contra branches (tau[ui][ii] vs [ci])."""
    ui0 = m.ui[0].cpu().numpy(); ui1 = m.ui[1].cpu().numpy()
    n_det = g["n_det"]
    ipsi = g["leg_ipsi"].cpu().numpy(); contra = g["leg_contra"].cpu().numpy()
    i_of, c_of = {}, {}
    for e in range(len(ui0)):
        a, b = int(ui0[e]), int(ui1[e])
        det = a if a < n_det else (b if b < n_det else -1)
        if det < 0:
            continue
        if ipsi[a, b]:
            i_of[det] = e
        elif contra[a, b]:
            c_of[det] = e
    dets = sorted(set(i_of) & set(c_of))
    ii = torch.tensor([i_of[dd] for dd in dets], dtype=torch.long)
    ci = torch.tensor([c_of[dd] for dd in dets], dtype=torch.long)
    return ii, ci, torch.tensor(dets, dtype=torch.long)


def coincidence_loss(m, ii, ci, itd=None):
    """L = mean over detectors of (tau_contra - tau_ipsi - ITD_target)^2 (differentiable in v).
    itd=None (or zeros) => pure coincidence (equalize arrival). nonzero swept itd => Jeffress place-map."""
    tau = m.tau_matrix()
    te = tau[m.ui[0], m.ui[1]]
    diff = te[ci] - te[ii]
    if itd is not None:
        diff = diff - itd
    return (diff ** 2).mean()


def speed_budget(m, mu):
    return mu * m.edge_velocity().mean()


def train(m, g, ii, ci, *, w_coinc, mu, steps, lr, lr_v, device, seed, T, itd=None):
    """Train per-edge velocity to make the two branches hit the ITD target under a shared speed budget.
    w_coinc=1 = ITD-user (coincidence objective ON); w_coinc=0 = non-ITD knockout (budget only)."""
    ii, ci = ii.to(device), ci.to(device)
    itd = itd.to(device) if itd is not None else None
    vparams, oparams = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        (vparams if n == "g" else oparams).append(p)
    groups = [dict(params=oparams, lr=lr)] + ([dict(params=vparams, lr=lr_v)] if vparams else [])
    opt = torch.optim.Adam(groups)
    gen = torch.Generator().manual_seed(555 + seed)
    for it in range(steps):
        opt.zero_grad()
        _ = m.propagate(pv.make_codes(32, m.K, gen, device), T)     # keep W,b in a sane regime
        loss = w_coinc * coincidence_loss(m, ii, ci, itd)
        if mu > 0:
            loss = loss + speed_budget(m, mu)
        loss.backward()
        nn.utils.clip_grad_norm_([p for grp in groups for p in grp["params"]], 1.0)
        opt.step()
    return m


@torch.no_grad()
def branch_stats(m, g, ii, ci, itd=None):
    v = m.edge_velocity().detach().cpu().numpy()
    length = m.d[m.ui[0], m.ui[1]].detach().cpu().numpy()
    iin, cin = ii.cpu().numpy(), ci.cpu().numpy()
    v_ipsi, v_contra = float(v[iin].mean()), float(v[cin].mean())
    ratio = v_contra / v_ipsi if v_ipsi else float("nan")
    tau = m.tau_matrix().detach(); te = tau[m.ui[0], m.ui[1]].cpu().numpy()
    achieved = te[cin] - te[iin]                                     # achieved ITD per detector
    tgt = itd.cpu().numpy() if itd is not None else np.zeros_like(achieved)
    arr_diff = float(np.mean(np.abs(achieved - tgt)))               # residual to the (possibly swept) ITD target
    arr_diff_norm = arr_diff / (0.5 * (te[cin].mean() + te[iin].mean()) + 1e-9)
    # corr(achieved ITD, target ITD): does the allocation reproduce the swept Jeffress map?
    corr_itd = (float(np.corrcoef(achieved, tgt)[0, 1]) if (itd is not None and achieved.size > 2 and tgt.std() > 1e-9)
                else float("nan"))
    vv = np.concatenate([v[iin], v[cin]]); ll = np.concatenate([length[iin], length[cin]])
    corr_v_len = pv.corr(vv, ll) if vv.size > 2 else float("nan")
    return dict(v_ipsi=v_ipsi, v_contra=v_contra, ratio=ratio, arr_diff=arr_diff,
                arr_diff_norm=arr_diff_norm, corr_v_len=corr_v_len, corr_itd=corr_itd)


def run_seed(seed, cfg, device):
    g = coincidence_geometry(cfg["n_det"], seed, cfg["d_ipsi"], cfg["d_contra_lo"],
                             cfg["d_contra_hi"], cfg["det_spread"], itd_span=cfg.get("itd_span", 0.0))
    mk = lambda mode, fv=None: pv.PlasticVelocityRNN(
        g, cfg["K"], velocity_mode=mode, v_min=cfg["v_min"], v_max=cfg["v_max"], v0=cfg["v0"],
        min_delay=cfg["min_delay"], max_delay=cfg["max_delay"], alpha=cfg["alpha"],
        seed=seed, fixed_v=fv).to(device)
    T = cfg["T"]

    m0 = mk("plastic"); ii, ci, dets = branch_pairs(m0, g)
    itd = g["itd_target"][dets].to(device)                          # swept Jeffress target (zeros if itd_span=0)
    # geometry-implied ideal ratio: equal-arrival needs v_contra/v_ipsi ~ d_contra/d_ipsi
    dC = float(g["d"][g["leg_contra"]].mean()); dI = float(g["d"][g["leg_ipsi"]].mean())
    ideal_ratio = dC / dI if dI else float("nan")
    # velocity-gradient sanity on the coincidence loss (guards integer-lag null)
    gv = torch.autograd.grad(coincidence_loss(m0, ii.to(device), ci.to(device), itd), m0.g)[0]
    v_grad_nonzero = int((gv.abs() > 1e-12).sum())

    # ---- ITD-USER: coincidence objective ON (myelinate-long) ----
    mit = mk("plastic")
    train(mit, g, ii, ci, w_coinc=1.0, mu=cfg["mu"], steps=cfg["steps"], lr=cfg["lr"],
          lr_v=cfg["lr_v"], device=device, seed=seed, T=T, itd=itd)
    s_itd = branch_stats(mit, g, ii, ci, itd)

    # ---- KNOCKOUT: coincidence objective OFF (the non-ITD "mouse") ----
    mko = mk("plastic")
    train(mko, g, ii, ci, w_coinc=0.0, mu=cfg["mu"], steps=cfg["steps"], lr=cfg["lr"],
          lr_v=cfg["lr_v"], device=device, seed=seed, T=T, itd=itd)
    s_ko = branch_stats(mko, g, ii, ci, itd)

    # ---- SHUFFLE null: trained ITD velocities reassigned across edges ----
    v_itd = mit.edge_velocity().detach()
    perm = torch.randperm(v_itd.numel(), generator=torch.Generator().manual_seed(321 + seed))
    msh = mk("fixed", fv=v_itd[perm.to(v_itd.device)].cpu())
    s_sh = branch_stats(msh, g, ii, ci, itd)

    # ---- UNIFORM-v at matched budget (cannot hit a non-monotone swept ITD map) ----
    M = float(mit.myelin().item()); dl = g["d"][mit.ui[0].cpu(), mit.ui[1].cpu()]
    v_uni = float(np.clip(cfg["v_min"] + M / max(float(dl.sum()), 1e-9), cfg["v_min"], cfg["v_max"]))
    mu_ = mk("uniform"); mu_.uniform_v.fill_(v_uni)
    s_uni = branch_stats(mu_, g, ii, ci, itd)

    return dict(
        seed=seed, ideal_ratio=ideal_ratio, v_grad_nonzero=v_grad_nonzero,
        ratio_itd=s_itd["ratio"], corr_v_len_itd=s_itd["corr_v_len"],
        arr_diff_itd=s_itd["arr_diff"], arr_diff_norm_itd=s_itd["arr_diff_norm"],
        ratio_ko=s_ko["ratio"], corr_v_len_ko=s_ko["corr_v_len"], arr_diff_ko=s_ko["arr_diff"],
        ratio_shuf=s_sh["ratio"], arr_diff_shuf=s_sh["arr_diff"],
        arr_diff_uniform=s_uni["arr_diff"], v_uniform=v_uni,
        corr_itd_itd=s_itd["corr_itd"], corr_itd_uniform=s_uni["corr_itd"], corr_itd_ko=s_ko["corr_itd"])


DEFAULT_CFG = dict(
    n_det=24, K=4,
    d_ipsi=2.0, d_contra_lo=5.0, d_contra_hi=14.0,    # ideal ratio ~ (mean 9.5)/2 ~ 4.7
    det_spread=1.0,
    v_min=0.3, v_max=8.0, v0=2.0,
    min_delay=1, max_delay=24, alpha=0.4, T=28,
    mu=0.02, steps=500, lr=5e-3, lr_v=0.15, itd_span=0.0,
)


def stats(a):
    a = np.asarray([x for x in a if not (isinstance(x, float) and math.isnan(x))], float)
    if a.size == 0:
        return dict(mean=float("nan"), sd=float("nan"), t=float("nan"), n=0)
    m = float(a.mean()); sd = float(a.std() + 1e-9)
    return dict(mean=m, sd=sd, t=m / (sd / math.sqrt(len(a))), n=len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", type=str, default=DEV)
    ap.add_argument("--out", type=str, default="results/experiments/auditory_brainstem_isochrony.json")
    ap.add_argument("--smoke", action="store_true")
    for k in ["v_max", "v0", "d_ipsi", "d_contra_lo", "d_contra_hi", "det_spread", "mu", "itd_span"]:
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=None, dest=k)
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    for k in ["v_max", "v0", "d_ipsi", "d_contra_lo", "d_contra_hi", "det_spread", "mu", "itd_span"]:
        if getattr(args, k, None) is not None:
            cfg[k] = getattr(args, k)
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.smoke:
        cfg["steps"] = 60; args.seeds = 1
    print(f"device={args.device} torch={torch.__version__}\nconfig={cfg}\n")

    rows, t0 = [], time.time()
    for s in range(args.seeds):
        print(f"=== seed {s+1}/{args.seeds} ===", flush=True)
        r = run_seed(s, cfg, args.device); rows.append(r)
        print(f"  ideal_ratio={r['ideal_ratio']:.2f} v_grad_nonzero={r['v_grad_nonzero']}", flush=True)
        print(f"  ITD-USER : v_contra/v_ipsi={r['ratio_itd']:.2f} corr(v,len)={r['corr_v_len_itd']:+.3f} "
              f"arr_diff={r['arr_diff_itd']:.3f} (norm {r['arr_diff_norm_itd']:.2f})", flush=True)
        print(f"  KNOCKOUT : v_contra/v_ipsi={r['ratio_ko']:.2f} corr(v,len)={r['corr_v_len_ko']:+.3f} "
              f"arr_diff={r['arr_diff_ko']:.3f}", flush=True)
        print(f"  NULLS    : shuffle ratio={r['ratio_shuf']:.2f} arr_diff={r['arr_diff_shuf']:.3f} | "
              f"uniform arr_diff={r['arr_diff_uniform']:.3f}", flush=True)
        if cfg.get("itd_span", 0.0) > 0:
            print(f"  ITD-MAP  : corr(achieved,target) ITD-user={r['corr_itd_itd']:+.3f} "
                  f"uniform={r['corr_itd_uniform']:+.3f}  [per-edge v hits the swept Jeffress map; global v cannot]",
                  flush=True)

    def arr(k): return [r[k] for r in rows]
    S = dict(config=cfg, seeds=args.seeds, device=args.device, minutes=round((time.time() - t0) / 60, 2),
             ideal_ratio=stats(arr("ideal_ratio")), ratio_itd=stats(arr("ratio_itd")),
             ratio_ko=stats(arr("ratio_ko")), ratio_shuf=stats(arr("ratio_shuf")),
             corr_v_len_itd=stats(arr("corr_v_len_itd")), corr_v_len_ko=stats(arr("corr_v_len_ko")),
             arr_diff_itd=stats(arr("arr_diff_itd")), arr_diff_ko=stats(arr("arr_diff_ko")),
             arr_diff_shuf=stats(arr("arr_diff_shuf")), arr_diff_uniform=stats(arr("arr_diff_uniform")),
             v_grad_alive=bool(all(r["v_grad_nonzero"] > 0 for r in rows)), rows=rows)
    ri, rk = S["ratio_itd"], S["ratio_ko"]; ai, ak = S["arr_diff_itd"], S["arr_diff_uniform"]
    isochrony = (ri["mean"] > 1.5 and ri["mean"] > 1.5 * max(rk["mean"], 1.0)
                 and S["corr_v_len_itd"]["mean"] > 0.3 and ai["mean"] < 0.5 * ak["mean"]
                 and S["v_grad_alive"])
    knockout_ok = rk["mean"] < 1.4 and S["corr_v_len_ko"]["mean"] < 0.2     # no tuning without the objective
    S["verdict"] = (
        "AUDITORY-BRAINSTEM ISOCHRONY REPRODUCED + KNOCKOUT (coincidence objective tunes per-branch velocity; absent without it)"
        if isochrony and knockout_ok else
        "PARTIAL (isochrony tuning seen; knockout not clean)" if isochrony else
        "NULL (coincidence objective did not tune per-branch velocity to equalize arrival)")
    print("\n=== AUDITORY-BRAINSTEM COINCIDENCE ISOCHRONY ===")
    print(f"ideal_ratio~{S['ideal_ratio']['mean']:.2f}  v_grad_alive={S['v_grad_alive']}")
    print(f"v_contra/v_ipsi: ITD-USER={ri['mean']:.2f}+/-{ri['sd']:.2f} (t={ri['t']:.1f})  KNOCKOUT={rk['mean']:.2f}  SHUFFLE={S['ratio_shuf']['mean']:.2f}")
    print(f"corr(v,len): ITD={S['corr_v_len_itd']['mean']:+.3f}  KO={S['corr_v_len_ko']['mean']:+.3f}")
    print(f"arrival diff: ITD={ai['mean']:.3f}  UNIFORM={ak['mean']:.3f}  SHUFFLE={S['arr_diff_shuf']['mean']:.3f}")
    print("VERDICT:", S["verdict"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(S, open(args.out, "w"), indent=2, default=float)
    print("wrote", args.out, "in", S["minutes"], "min")


if __name__ == "__main__":
    main()
