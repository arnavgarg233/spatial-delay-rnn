"""Does the conduction-time economy reproduce in Kuramoto phase oscillators with delays?

Phase oscillators on a 2-D sheet:
    dtheta_i/dt = omega_i + sum_j K_ij * sin( theta_j(t - tau_ij) - theta_i(t) )
with tau_ij = round(d_ij/velocity) clipped to [1, max_delay].

Target is a planar travelling phase wave phi_i = k.pos_i; coherence is the pattern order
parameter R = |mean_i exp(i(theta_i - phi_i))|. (Global sync phi==0 is the wrong target here:
it rewards long-range wiring and inverts the economy.) Optimize K>=0 to reach target R while
paying the conduction cost lambda*sum K_ij*tau_ij, then compare distance vs a histogram-matched
shuffled control (same tau multiset, off-diagonal permutation) at matched R. Both conditions
minimize the same objective with the same tau multiset; only the distance<->delay
correspondence differs.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))
import argparse, json, math, os
import numpy as np
import torch

from sdrnn.geometry import grid_coordinates
from sdrnn.delays import integer_delays

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=48, help="number of oscillators")
ap.add_argument("--velocity", type=float, default=0.08)
ap.add_argument("--max_delay", type=int, default=14)
ap.add_argument("--dt", type=float, default=0.15, help="integrator step")
ap.add_argument("--T", type=int, default=120, help="rollout steps for sync eval")
ap.add_argument("--opt_steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=0.06)
ap.add_argument("--cost_lambda", type=float, default=0.10,
                help="weight on the conduction-time cost sum(K*tau) in the objective")
ap.add_argument("--budget", type=float, default=10.0,
                help="target total coupling mass sum(K); a soft budget pins total mass so the "
                     "conduction-cost term must REALLOCATE (anti-sort onto short tau) rather "
                     "than uniformly shrink -- the KKT/HLP reallocation the law describes")
ap.add_argument("--budget_pen", type=float, default=0.02,
                help="weight on (sum(K)-budget)^2 budget constraint")
ap.add_argument("--target_R", type=float, default=0.90, help="target pattern order parameter")
ap.add_argument("--omega_spread", type=float, default=0.15,
                help="std of intrinsic frequency noise around the pattern")
ap.add_argument("--wave_k", type=float, default=1.3,
                help="spatial wavenumber of the target travelling phase pattern phi=k*x")
ap.add_argument("--sigma", type=float, default=0.5,
                help="length-scale of the distance-decreasing functional demand profile")
ap.add_argument("--demand_scale", type=float, default=0.5,
                help="coupling-strength scale for the demand-aligned (KKT/HLP) allocation")
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--device", default="cpu")  # tiny N; cpu is fastest & deterministic for this
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

if a.smoke:
    a.n, a.opt_steps, a.T, a.seeds = 24, 60, 80, 2

DEV = torch.device(a.device)
torch.set_num_threads(3)
DTYPE = torch.float32


def build_geometry(n, velocity, max_delay):
    """2-D grid positions -> Euclidean distances -> integer conduction delays."""
    coords = grid_coordinates(n, dim=2)
    dist = torch.cdist(coords, coords, p=2).to(DTYPE)
    tau = integer_delays(dist, velocity, max_delay).to(DTYPE)  # (n,n), diag clipped to 1
    return tau, dist, coords


def shuffle_perm(n, seed):
    """Off-diagonal permutation: same delay multiset, distance<->delay correspondence destroyed."""
    gen = torch.Generator().manual_seed(int(seed) + 1009)
    return torch.randperm(n * n - n, generator=gen)


def apply_shuffle(tau, perm):
    n = tau.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    out = tau.clone()
    vals = out[off]
    out[off] = vals[perm]
    return out


def pattern_order_parameter(theta, phi):
    """R = |mean_i exp(i(theta_i - phi_i))|: coherence with target pattern phi (phi==0 = global sync)."""
    return torch.exp(1j * (theta - phi).to(torch.cfloat)).mean(-1).abs()


def roll_kuramoto(K, tau_idx, omega, theta0, phi, dt, T, max_delay, settle_frac=0.5):
    """Integrate delayed Kuramoto; return order parameter time-averaged over the settled half.

    History is a (max_delay+1, n) buffer; per step we gather source phase theta_j at its own
    integer lag tau_ij in one advanced-index op. Differentiable in K.
    """
    n = K.shape[0]
    # history[k] = phases k steps ago (k=0 newest). Constant-history start at theta0.
    hist = theta0.unsqueeze(0).repeat(max_delay + 1, 1)   # (D+1, n)
    theta = theta0.clone()
    # per-edge source index and lag, flat for gather: want hist[tau_ij, j] for each (i,j)
    src = torch.arange(n).unsqueeze(0).expand(n, n)        # (i,j) -> j
    lag = tau_idx.long()                                   # (i,j) -> tau_ij
    Rs = []
    start = int(settle_frac * T)
    for t in range(T):
        # gather delayed source phases: theta_past[i,j] = hist[lag_ij, j]
        theta_past = hist[lag, src]                        # (n, n) advanced indexing
        diff = theta_past - theta.unsqueeze(1)             # theta_j(t-tau) - theta_i(t)
        dtheta = omega + (K * torch.sin(diff)).sum(dim=1)
        theta = theta + dt * dtheta
        hist = torch.roll(hist, shifts=1, dims=0)
        hist = hist.clone(); hist[0] = theta               # clone keeps autograd happy
        if t >= start:
            Rs.append(pattern_order_parameter(theta, phi))
    return torch.stack(Rs).mean()


def optimize_coupling(tau, omega, theta0, phi, a):
    """Optimize K>=0 to reach target R while paying the conduction cost. Returns (K, R).

        minimize_{K>=0}  relu(R* - R(K))  +  lambda * sum_ij K_ij * tau_ij
    K = softplus(raw) enforces K>=0; relu pushes R up to target only (no reward for overshoot).
    """
    n = tau.shape[0]
    raw = torch.nn.Parameter(torch.full((n, n), -2.0, dtype=DTYPE))
    opt = torch.optim.Adam([raw], lr=a.lr)
    eye = torch.eye(n, dtype=torch.bool)
    tau_cost = tau.to(DTYPE)
    for it in range(a.opt_steps):
        opt.zero_grad()
        K = torch.nn.functional.softplus(raw).masked_fill(eye, 0.0)
        R = roll_kuramoto(K, tau, omega, theta0, phi, a.dt, a.T, a.max_delay)
        sync_loss = 2.0 * torch.relu(a.target_R - R)
        cost = (K * tau_cost).sum()
        budget_pen = a.budget_pen * (K.sum() - a.budget) ** 2  # pin total mass -> force reallocation
        loss = sync_loss + a.cost_lambda * cost + budget_pen
        loss.backward()
        opt.step()
    with torch.no_grad():
        K = torch.nn.functional.softplus(raw).masked_fill(eye, 0.0)
        R = roll_kuramoto(K, tau, omega, theta0, phi, a.dt, a.T, a.max_delay)
    return K.detach(), float(R)


def conduction_cost(K, tau):
    """C = sum K*tau, tau_bar = C/sum K. tau is the geometric delay (same multiset both conditions)."""
    Kn = K.detach().cpu().numpy()
    tn = tau.detach().cpu().numpy().astype(float)
    C = float((Kn * tn).sum())
    return C, C / (Kn.sum() + 1e-12)


def hlp_optimal_cost(demand, tau_offdiag):
    """Rearrangement-optimal cost: pair largest mass with smallest tau (anti-sorted)."""
    d = np.sort(np.asarray(demand, float))[::-1]
    t = np.sort(np.asarray(tau_offdiag, float))
    m = min(len(d), len(t))
    return float((d[:m] * t[:m]).sum())


def demand_profile(dist, sigma):
    """Per-pair functional demand (Gaussian of distance): near pairs need tighter coupling to
    hold a smooth phase ramp, so demand decreases with distance. Defined on pairs, not on tau."""
    D = np.asarray(dist, float)
    dem = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(dem, 0.0)
    return dem


def demand_aligned_coupling(dist, scale, sigma, n):
    """KKT/HLP coupling = demand profile scaled to a target mass: geometry-aware but delay-blind
    (does not look at tau), so both delay assignments carry the same allocation."""
    K = torch.tensor(demand_profile(dist, sigma), dtype=DTYPE) * scale
    return K


def run_seed(seed, a, tau_dist, perm, phi, dist, K_demand):
    g = torch.Generator().manual_seed(seed)
    # small omega noise around a common rotation keeps it a real (not trivial) optimization;
    # the wave is carried by coupling + the geometric target, not per-oscillator drives.
    omega = (torch.randn(a.n, generator=g) * a.omega_spread).to(DTYPE)
    # start near the target pattern with a random kick (basin of the desired collective state)
    theta0 = (phi + (torch.rand(a.n, generator=g) * 0.6 - 0.3)).to(DTYPE)
    tau_shuf = apply_shuffle(tau_dist, perm)

    # per-seed jitter on the demand allocation so the margin is measured across different couplings
    eye = torch.eye(a.n, dtype=torch.bool)
    Kjit = (K_demand * (1.0 + 0.15 * torch.randn(a.n, a.n, generator=g))).clamp_min(0.0)
    Kjit = Kjit.masked_fill(eye, 0.0).to(DTYPE)

    out = {"free": {}, "demand": {}}
    for name, tau in [("distance", tau_dist), ("shuffled", tau_shuf)]:
        # (1) free optimizer: gradient descent on pattern coherence + conduction cost
        Kf, Rf = optimize_coupling(tau, omega, theta0, phi, a)
        Cf, tbf = conduction_cost(Kf, tau)
        out["free"][name] = dict(R=float(Rf), C=float(Cf), tau_bar=float(tbf), Ksum=float(Kf.sum()))
        # (2) demand-aligned (KKT/HLP): same Kjit under both conditions; only tau differs, so
        # any cost gap is pure rearrangement
        with torch.no_grad():
            Rd = float(roll_kuramoto(Kjit, tau, omega, theta0, phi, a.dt, a.T, a.max_delay))
        Cd, tbd = conduction_cost(Kjit, tau)
        out["demand"][name] = dict(R=Rd, C=float(Cd), tau_bar=float(tbd), Ksum=float(Kjit.sum()))
        print(f"    seed {seed} {name:8s}: FREE R={Rf:.3f} C={Cf:7.1f} tbar={tbf:.3f} | "
              f"DEMAND R={Rd:.3f} C={Cd:7.1f} tbar={tbd:.3f}", flush=True)
    return out


def _ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std())


def _sign_test(diffs):
    d = [x for x in diffs if x != 0.0]; n = len(d)
    n_neg = sum(1 for x in d if x < 0); n_pos = n - n_neg
    k = min(n_neg, n_pos)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) if n else 1.0
    return n_neg, n_pos, min(1.0, 2 * tail)


def _paired_t(diffs):
    d = np.asarray(diffs, float); n = len(d)
    if n < 2: return float("nan")
    sd = d.std(ddof=1)
    if sd == 0: return float("inf") if d.mean() > 0 else float("-inf")
    return float(d.mean() / (sd / math.sqrt(n)))


def summarize_block(label, rows, paired_C, paired_tbar):
    """Print + return the economy verdict for one allocation mode (free or demand)."""
    print("-" * 76)
    print(f"[{label}] per condition (mean +/- std over seeds):")
    for name in ("distance", "shuffled"):
        Rm, Rs = _ms(rows[name]["R"]); Cm, Cs = _ms(rows[name]["C"])
        Tm, Ts = _ms(rows[name]["tau_bar"])
        print(f"  {name:8s}: R={Rm:.3f}+/-{Rs:.3f}  C={Cm:7.1f}+/-{Cs:5.1f}  "
              f"tau_bar={Tm:.3f}+/-{Ts:.3f}")
    R_gap = abs(_ms(rows["distance"]["R"])[0] - _ms(rows["shuffled"]["R"])[0])
    matched = R_gap < 0.03
    res = {}
    for mn, diffs in [("C", paired_C), ("tau_bar", paired_tbar)]:
        nn_, np_, p = _sign_test(diffs); t = _paired_t(diffs)
        dstr = ", ".join(f"{x:+.3f}" for x in diffs)
        print(f"  paired d-s {mn:7s}: [{dstr}] mean={float(np.mean(diffs)):+.3f} "
              f"wins={nn_}/{len(diffs)} sign_p={p:.3f} t={t:.2f}")
        res[mn] = dict(diffs=[float(x) for x in diffs], mean=float(np.mean(diffs)),
                       wins=int(nn_), n=int(len(diffs)), sign_p=float(p), t=float(t))
    Cd = np.asarray(rows["distance"]["C"]); Cs_ = np.asarray(rows["shuffled"]["C"])
    pooled = math.sqrt((Cd.var(ddof=0) + Cs_.var(ddof=0)) / 2)
    # deterministic allocation -> zero per-seed variance -> sigma separation undefined; report
    # the relative saving instead (an exact every-seed margin, not a noisy effect size)
    rel_saving = float((Cs_.mean() - Cd.mean()) / (Cs_.mean() + 1e-12))
    if pooled < 1e-9:
        sigma_C = float("nan")
        sep_str = f"rel_saving={100*rel_saving:+.1f}% (deterministic; every seed)"
    else:
        sigma_C = float((Cs_.mean() - Cd.mean()) / pooled)
        sep_str = f"sigma_C={sigma_C:+.2f}  rel_saving={100*rel_saving:+.1f}%"
    # require a meaningful margin (>0.5% cheaper) AND lower tau_bar; sub-percent is not an economy
    economy = (rel_saving > 0.005
               and _ms(rows['distance']['tau_bar'])[0] < _ms(rows['shuffled']['tau_bar'])[0])
    print(f"  => R_gap={R_gap:.4f} ({'MATCHED' if matched else 'UNMATCHED'})  "
          f"{sep_str}  ECONOMY(distance<shuffled)={economy}")
    return dict(per_condition={n: {k: [float(x) for x in rows[n][k]] for k in rows[n]}
                               for n in rows},
                paired=res, R_gap=float(R_gap), matched=bool(matched),
                sigma_C=float(sigma_C), rel_saving=rel_saving, economy=bool(economy))


def main():
    print("=" * 76)
    print("LAW TEST: Kuramoto oscillators with conduction delays")
    print(f"  n={a.n}  velocity={a.velocity}  max_delay={a.max_delay}  target_R={a.target_R}")
    print(f"  opt_steps={a.opt_steps}  T={a.T}  dt={a.dt}  seeds={a.seeds}  wave_k={a.wave_k}")
    print(f"  cost_lambda={a.cost_lambda}  budget={a.budget}  sigma={a.sigma}  "
          f"demand_scale={a.demand_scale}")
    print("=" * 76)

    tau_dist, dist, coords = build_geometry(a.n, a.velocity, a.max_delay)
    # target spatial phase pattern: a planar travelling wave phi_i = wave_k * x_i across the sheet
    phi = (coords[:, 0] * a.wave_k).to(DTYPE)
    perm = shuffle_perm(a.n, seed=0)  # one fixed scramble shared across seeds (as in RNN)
    tau_shuf = apply_shuffle(tau_dist, perm)
    off = ~torch.eye(a.n, dtype=torch.bool)

    # demand-aligned (KKT/HLP) coupling: distance-decreasing, delay-blind; identical for both
    # conditions so any cost gap is pure rearrangement.
    K_demand = demand_aligned_coupling(dist.cpu().numpy(), a.demand_scale, a.sigma, a.n)

    free_rows = {"distance": dict(R=[], C=[], tau_bar=[], Ksum=[]),
                 "shuffled": dict(R=[], C=[], tau_bar=[], Ksum=[])}
    dem_rows = {"distance": dict(R=[], C=[], tau_bar=[], Ksum=[]),
                "shuffled": dict(R=[], C=[], tau_bar=[], Ksum=[])}
    free_pC, free_pT, dem_pC, dem_pT = [], [], [], []

    for s in range(a.seeds):
        out = run_seed(s, a, tau_dist, perm, phi, dist, K_demand)
        for mode, rows, pC, pT in [("free", free_rows, free_pC, free_pT),
                                   ("demand", dem_rows, dem_pC, dem_pT)]:
            for name in ("distance", "shuffled"):
                for k in ("R", "C", "tau_bar", "Ksum"):
                    rows[name][k].append(float(out[mode][name][k]))
            pC.append(float(out[mode]["distance"]["C"] - out[mode]["shuffled"]["C"]))
            pT.append(float(out[mode]["distance"]["tau_bar"] - out[mode]["shuffled"]["tau_bar"]))
        print("", flush=True)

    # analytic rearrangement inequality: demand profile on distance vs shuffled delays. No
    # optimizer, no dynamics -- pure cost of carrying the loop-gain mass over the two metrics.
    dem_off = K_demand[off].cpu().numpy()
    tau_d_off = tau_dist[off].cpu().numpy()
    tau_s_off = tau_shuf[off].cpu().numpy()
    Cd_an = float((dem_off * tau_d_off).sum()); Cs_an = float((dem_off * tau_s_off).sum())
    analytic_margin = Cs_an - Cd_an
    analytic_rel = analytic_margin / (Cs_an + 1e-9)

    # dose-response: sweep velocity -> delay-spread; saving should grow linearly (Saving(s)=B0*s).
    # Analytic (demand profile on distance vs shuffled tau), generous max_delay so the long end
    # is not clipped.
    dose = {"velocity": [], "mean_tau": [], "saving": [], "rel": []}
    dem_full = demand_profile(dist.cpu().numpy(), a.sigma)  # (n,n), diag 0
    for vel in [0.30, 0.16, 0.08, 0.05, 0.035]:
        tau_v = integer_delays(dist, vel, 40).to(DTYPE)
        tau_vs = apply_shuffle(tau_v, perm)
        Cd_v = float((dem_full * tau_v.cpu().numpy()).sum())
        Cs_v = float((dem_full * tau_vs.cpu().numpy()).sum())
        dose["velocity"].append(vel)
        dose["mean_tau"].append(float(tau_v[off].float().mean()))
        dose["saving"].append(Cs_v - Cd_v)
        dose["rel"].append((Cs_v - Cd_v) / (Cs_v + 1e-9))
    mt = np.asarray(dose["mean_tau"]); sv = np.asarray(dose["saving"])
    A = np.vstack([mt, np.ones_like(mt)]).T
    slope, intercept = np.linalg.lstsq(A, sv, rcond=None)[0]
    pred = A @ np.array([slope, intercept])
    r2 = float(1 - ((sv - pred) ** 2).sum() / (((sv - sv.mean()) ** 2).sum() + 1e-12))
    dose.update(linear_slope=float(slope), intercept=float(intercept), r2=r2)

    print("=" * 76)
    print("RESULTS")
    free_res = summarize_block("FREE OPTIMIZER (gradient descent on pattern+conduction cost)",
                               free_rows, free_pC, free_pT)
    dem_res = summarize_block("DEMAND-ALIGNED (KKT/HLP coupling: mass on near/high-demand pairs)",
                              dem_rows, dem_pC, dem_pT)
    print("-" * 76)
    print("ANALYTIC rearrangement inequality (demand profile on distance vs shuffled delays):")
    print(f"  distance C={Cd_an:.1f}  shuffled C={Cs_an:.1f}  "
          f"saving={analytic_margin:+.1f} ({100*analytic_rel:+.1f}%)  "
          f"holds={analytic_margin > 0}")
    print("-" * 76)
    print("DOSE-RESPONSE (saving vs delay-spread; lower velocity = longer delays):")
    for v, m, s_, r in zip(dose["velocity"], dose["mean_tau"], dose["saving"], dose["rel"]):
        print(f"  vel={v:5.3f}  mean_tau={m:6.2f}  saving={s_:8.1f}  rel={100*r:5.1f}%")
    print(f"  LINEAR LAW: saving = {dose['linear_slope']:.1f}*mean_tau + {dose['intercept']:.1f}"
          f"   R^2={dose['r2']:.4f}")
    print("=" * 76)

    print("VERDICT:")
    print(f"  analytic law (rearrangement):     distance<shuffled = {analytic_margin>0} "
          f"({100*analytic_rel:+.1f}% cheaper)")
    print(f"  demand-aligned allocation econ.:  {dem_res['economy']} "
          f"(matched R={dem_res['matched']}, {100*dem_res['rel_saving']:+.1f}% cheaper)")
    print(f"  dose-response linear law:         R^2={dose['r2']:.4f}  "
          f"(saving grows linearly in delay-spread)")
    print(f"  free-optimizer econ. (RNN-style): {free_res['economy']} "
          f"(matched R={free_res['matched']}, {free_res['sigma_C']:+.1f} sigma)")
    print("=" * 76)

    result = dict(
        params={k: (float(v) if isinstance(v, (int, float)) else v) for k, v in vars(a).items()},
        free=free_res,
        demand=dem_res,
        analytic=dict(distance_C=Cd_an, shuffled_C=Cs_an, saving=float(analytic_margin),
                      rel_saving=float(analytic_rel), holds=bool(analytic_margin > 0)),
        dose=dose,
    )
    outdir = ROOT / "results" / "law"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "law_kuramoto.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}")
    return result


if __name__ == "__main__":
    main()
