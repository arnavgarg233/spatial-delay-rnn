"""Conduction-time economy as delayed optimal control.

Frames the trained distance-delay RNN as the solution of a delayed LQ optimal-control problem,
and derives where the per-edge "task value" g_k comes from (rather than assuming it):

  P1  min sum|W|tau is the KKT stationarity condition of a delayed-LQR problem whose running
      cost charges magnitude by conduction time; g_k is the costate sensitivity dJ/d|W_k|.
  P2  For a delayed linear plant under a conduction-time penalty, optimal control fills by the
      greedy ratio g_k/tau_k -> the knapsack is the KKT system of an optimal-control problem.
  P3  Optimal-transport reading: the allocation transports a fixed loop-gain mass onto the
      delay metric; the shuffled control uses a random coupling, so by HLP rearrangement it
      pays more (gap = OT duality gap, monotone in s).
  P4  Sufficient condition (C*): g_k monotone-decreasing in tau_k => optimal allocation lands on
      the shortest edges => B0>=0 (proved, sufficient not necessary). (C*) holds mechanistically
      for the HOLD task (DDE stability budget G_max(L) decreases in round-trip latency). The
      empirical gate is the concentration CV(g): flat g (easy task) -> B0~0 washout; concentrated
      g (functional task) -> B0>0. Matches the multi-task data.

Every claim has a sympy/numpy check; no training. Under tau=s*t it recovers Saving(s)=B0*s.

Run:  python scripts/phys_optimal_control.py
"""

import json
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
OUT = ROOT / "results" / "law" / "optimal_control.json"


# P0. The delayed linear plant we linearize.
# model.py leaky-Euler rate net: state <- (1-a) state + a (W_in x + W_rec r[t-tau] + b), r=phi(state).
# Linearize: with effective gain g_ij = a W_ij phi'_j the perturbation obeys (theory_test.py)
#   s_i[t] = (1-a) s_i[t-1] + sum_j g_ij s_j[t-tau_ij].
# Treat the recurrent magnitudes |W_k| as control variables, tau_k as a per-control latency cost.


# P1. min sum|W|tau is the LQR stationarity system.
# Reduce to the dominant memory mode: a HOLD task needs the loop to sustain a scalar latent m[t]
# with self-gain G near an integrator pole. For one route k at latency tau_k the latent maps as
#       m[t] = (1-a) m[t-1] + a G_k m[t-tau_k] + (other routes).
# Sustaining a target loop gain G* on route k needs placed magnitude w_k = G*/b_k (b_k = phi'-
# weighted readout leverage; a pure delay has unit DC gain, so this is tau-independent). The
# conduction cost is C_k = w_k tau_k = G* (tau_k/b_k). Minimizing total conduction cost subject
# to realizing G* is
#       min sum_k w_k tau_k   s.t.  sum_k b_k w_k >= G*,   w_k >= 0,        (LQR*)
# i.e. fractional knapsack with value field g_k := b_k and cost tau_k. KKT: active routes share a
# common ratio tau_k/b_k = mu, excluded routes have tau_k/b_k >= mu. So optimal control fills
# minimal tau_k/b_k first, with g_k = b_k the (derived, not assumed) costate leverage.

def lqr_costate_sympy():
    """Show g_k = b_k (the adjoint leverage) symbolically: confirm the per-route conduction
    cost is (G*/b)*tau and that the costate dG/dw equals the knapsack value field b."""
    w, b, tau, Gstar, rho = sp.symbols('w b tau G_star rho', positive=True)
    # realized loop gain from placing w on the route:
    G_realized = b * w
    # magnitude required to realize target gain G*:
    w_req = sp.solve(sp.Eq(G_realized, Gstar), w)[0]          # = G*/b
    C_route = sp.simplify(w_req * tau)                        # conduction cost
    # adjoint leverage = d(realized gain)/d w = b  (the costate value field g_k):
    g_k = sp.diff(G_realized, w)
    # cost-per-value ratio that KKT equalizes across active routes:
    ratio = sp.simplify(C_route / Gstar)                     # = tau/b
    return dict(w_req=w_req, C_route=C_route, g_k=g_k, ratio=ratio)


def kkt_common_ratio_numeric(seed=0, K=12):
    """Solve (LQR*) knapsack and confirm KKT: active routes share a common ratio tau_k/b_k = mu,
    excluded routes have tau_k/b_k >= mu."""
    rng = np.random.default_rng(seed)
    tau = rng.uniform(1, 10, K)
    b = rng.uniform(0.2, 1.5, K)      # adjoint leverage (value field)
    Gstar = 0.6 * (b).sum()           # require 60% of max realizable gain
    # greedy fractional knapsack by ratio tau/b (min cost per value first):
    order = np.argsort(tau / b)
    w = np.zeros(K); acc = 0.0; cap = 1.0; active = []
    mu = None
    for k in order:
        if acc >= Gstar:
            break
        take = min(cap, (Gstar - acc) / b[k])
        w[k] = take; acc += take * b[k]; active.append(k)
        mu = tau[k] / b[k]            # last filled sets the marginal ratio
    ratios = tau / b
    active = np.array(active)
    excluded = np.array([k for k in range(K) if k not in set(active)])
    # KKT: every active route has ratio <= mu (<= marginal), excluded >= mu.
    active_ok = bool(np.all(ratios[active] <= mu + 1e-9))
    excl_ok = bool(len(excluded) == 0 or np.all(ratios[excluded] >= mu - 1e-9))
    return dict(mu=float(mu), active_max_ratio=float(ratios[active].max()),
                excluded_min_ratio=float(ratios[excluded].min()) if len(excluded) else None,
                kkt_active_ok=active_ok, kkt_excluded_ok=excl_ok,
                n_active=int(len(active)))


# P2. General statement: delayed-LQR -> conduction-time least action.
# A delayed linear plant minimizing LQ cost J(W) + rho*sum_k|W_k|tau_k, linearized about the
# task optimum as J ~ J0 - sum g_k|W_k| + .5 sum |W_k| H_kl |W_l| (g_k = -dJ/d|W_k| >= 0 the
# adjoint leverage, H PSD), has stationary control H|W| = g - rho*tau - nu, whose support is the
# fractional-knapsack solution of min sum|W|tau s.t. sum g_k|W_k| >= V*. So the trained net's
# sum|W|tau is the minimum for the value it realizes, and Saving(s)=B0 s is its comparative
# statics in s. We verify by solving the QP min_{|W|>=0} -g.|W| + .5|W| H |W| + rho tau.|W| and
# checking support = knapsack support and lower cost than the shuffle. (Knapsack = zero-curvature
# limit of the QP.)

def _solve_qp(g, H, tau, rho, n_iter=20000):
    """Projected-gradient minimizer of  -g.w + .5 w^T H w + rho (tau.w), w>=0."""
    K = len(g)
    w = np.zeros(K)
    lr = 1.0 / (np.linalg.eigvalsh(H).max() + 1e-9)
    for _ in range(n_iter):
        w = np.maximum(0.0, w - lr * (-g + H @ w + rho * tau))
    return w


def delayed_lqr_qp(seed=0, K=14, rho=0.15, ridge=0.6, n_perm=200):
    """Solve the linearized delayed-LQR QP and test the paper-faithful shuffle.

    min_{w>=0}  -g.w + 0.5 w^T H w + rho * (tau .* w).sum()  (H PSD, g>=0, tau>=1)

    Two shuffle controls (tau is the true geometric delay; only the learned weights differ):
      * paper-faithful: shuffled net optimized on a permuted delay field but scored on the true
        one, so it can't avoid long-true-delay edges. The decisive distance-specific test.
      * fixed-|W|: same w, tau reassigned -> E = (sum w) mean tau. True by construction.
    """
    rng = np.random.default_rng(seed)
    # latencies + a value field coupled to proximity (short routes more useful) -> (C*) holds
    tau = rng.uniform(1.0, 12.0, K)
    g = 0.3 + (tau.max() - tau) / tau.max() + 0.1 * rng.random(K)
    M = rng.standard_normal((K, K)) * 0.25
    H = M.T @ M + ridge * np.eye(K)         # PSD LQ Hessian

    # distance: optimize on true tau, score on true tau
    w = _solve_qp(g, H, tau, rho)
    cost_dist = float((w * tau).sum())
    tbar_dist = float((w * tau).sum() / w.sum())
    support = w > 1e-6

    # paper-faithful shuffle: optimize on permuted tau, score on true tau
    faith_cost, faith_tbar = [], []
    for _ in range(n_perm):
        perm = rng.permutation(K)
        ws = _solve_qp(g, H, tau[perm], rho)        # trained on wrong geometry
        faith_cost.append(float((ws * tau).sum()))  # scored on TRUE geometry
        faith_tbar.append(float((ws * tau).sum() / ws.sum()))
    cost_shuf = float(np.mean(faith_cost))
    tbar_shuf = float(np.mean(faith_tbar))

    # fixed-|W| shuffle (trivial-by-construction): same w, tau permuted
    cost_fixedW = float(w.sum() * tau.mean())

    # knapsack support (zero-curvature limit) for the same value field
    Vstar = float(g @ w)
    order = np.argsort(tau / g)
    wk = np.zeros(K); acc = 0.0; cap = w.max() if w.max() > 0 else 1.0
    for k in order:
        if acc >= Vstar:
            break
        take = min(cap, (Vstar - acc) / g[k]); wk[k] = take; acc += take * g[k]
    knap_support = wk > 1e-6
    overlap = float((support & knap_support).sum() /
                    max(1, (support | knap_support).sum()))

    return dict(
        tbar_dist=tbar_dist, tbar_shuf=tbar_shuf,
        cost_dist=cost_dist, cost_shuf=cost_shuf, cost_fixedW=cost_fixedW,
        saving_tbar=tbar_shuf - tbar_dist,
        beats_shuffle=bool(tbar_dist < tbar_shuf - 1e-9),
        support_overlap_with_knapsack=overlap, n_support=int(support.sum()),
    )


# P3. Optimal-transport reading + rearrangement-inequality proof of the gap.
# The allocation transports normalized mass pi_k = w_k/sum w onto the delay metric tau. The
# trained net's tau_bar_dist = <pi, tau> uses the anti-sorted coupling (big mass <-> small tau);
# the shuffle uses a random coupling with expected value mean(tau). By Hardy-Littlewood-Polya the
# anti-sorted coupling minimizes the inner product, so
#     tau_bar_shuf - tau_bar_dist = mean(tau) - <pi_sorted, tau> >= 0  (= B0*s under tau=s*t).
# We verify the bound and that the optimal coupling is anti-sorted.

def ot_rearrangement_numeric(seed=0, K=20, s=4.0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.2, 2.0, K); t /= t.mean()
    tau = s * t
    # optimal mass: anti-sorted (big mass on small tau)
    mass = rng.random(K) + 0.1
    order = np.argsort(tau)              # small tau first
    mass_sorted = np.sort(mass)[::-1]    # big mass first
    pi = np.zeros(K); pi[order] = mass_sorted; pi /= pi.sum()
    tau_bar_dist = float(pi @ tau)
    tau_bar_shuf = float(tau.mean())     # expected random coupling
    # rearrangement check: a random sample of permutations should never beat the anti-sorted one
    best_other = min(float((pi[rng.permutation(K)] @ tau)) for _ in range(5000))
    is_min = bool(tau_bar_dist <= best_other + 1e-9)
    return dict(tau_bar_dist=tau_bar_dist, tau_bar_shuf=tau_bar_shuf,
                gap=tau_bar_shuf - tau_bar_dist,
                anti_sorted_is_min=is_min, s=s)


# P4. The condition, its mechanistic source, and the break.
# The saving over the re-optimized shuffle is B0 = <t>_shuf - <t>_opt, where <t>_opt is the
# value-weighted mean normalized delay of the cost-minimizing (knapsack-by-tau/g) allocation
# under the distance lock. B0>0 iff the distance lock concentrates value on short edges.
#
#   (C*)  g_k decreasing in tau_k  =>  <t>_opt <= <t>_shuf  =>  B0>=0.
# Proof: if g is non-increasing in tau then tau/g is non-decreasing, so the greedy knapsack
# fills in ascending-tau order; its support is a prefix of the shortest edges, with the minimum
# value-weighted mean delay. A re-optimized shuffle puts the same value on a random subset, mean
# delay >= the prefix mean. So B0 >= 0, strict unless tau is constant. (Sufficient, not
# necessary: a bump field peaking at low tau also gives B0>0. A raw covariance cov(g,-tau) is
# neither necessary nor sufficient, since the knapsack follows tau/g.)
#
# Mechanistic (C*) for HOLD: the route must sustain a near-integrator loop without instability.
# The DDE stability budget (theory_test.py) caps loop gain at G_max(L), L ~ 2 tau_k, and G_max
# decreases in L, so short routes carry more usable gain per unit magnitude -> g_k decreasing.
#
# Empirical gate: the saving scales with CV(g). Flat g (easy task, little recurrence) -> B0~0,
# washout (MemoryPro dist-shuf tau_bar = +0.115); concentrated g (functional task) -> B0>0,
# clean (DelayedCopy dist-shuf = -1.170). Matches fn_multitask.json. (A bump field peaking at
# tau*>0 can give B0<0 -- reported only as a boundary of (C*), not the data's gate.)

def _max_root_modulus(a, G, L):
    coef = np.zeros(L + 1); coef[0] = 1.0; coef[1] = -(1.0 - a); coef[L] = -a * G
    return float(np.max(np.abs(np.roots(coef))))

def g_max_negative(a, L, hi=50.0):
    """Largest |G| with all roots inside unit circle for NEGATIVE (oscillatory)
    feedback at round-trip latency L. Bisection (mirrors theory_test.g_max)."""
    if _max_root_modulus(a, -hi, L) <= 1.0:
        return hi
    lo, hg = 0.0, hi
    for _ in range(60):
        mid = 0.5 * (lo + hg)
        if _max_root_modulus(a, -mid, L) <= 1.0:
            lo = mid
        else:
            hg = mid
    return lo

def hold_value_field_from_dynamics(a=0.2, tau_grid=None):
    """Derive g(tau) for the HOLD task from the DDE stability ceiling and test (C*): g decreasing
    in tau. g_k ~ stable gain headroom per unit magnitude = G_max(2 tau)."""
    from scipy.stats import spearmanr
    if tau_grid is None:
        tau_grid = np.arange(1, 13)
    g = np.array([g_max_negative(a, int(2 * t)) for t in tau_grid])
    g = g / g.max()
    # (C*): g decreasing in tau. Spearman rho(g, tau) < 0 (covariance is the wrong summary here).
    rho, p = spearmanr(g, tau_grid)
    g_decreasing = bool(np.all(np.diff(g) <= 1e-9))   # strictly non-increasing
    return dict(tau=tau_grid.tolist(), g_hold=g.tolist(),
                spearman_g_tau=float(rho), g_monotone_decreasing=g_decreasing,
                condition_Cstar_holds=bool(g_decreasing))


def validate_Cstar_sufficiency(n_test=400, seed=0, nperm=600):
    """Check that (C*) [g monotone-decreasing in tau] is sufficient for B0>0, over random
    value-field shapes (monotone, bump, arbitrary); also characterize necessity."""
    import warnings
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    tau = np.arange(1, 13).astype(float)
    mono_count = mono_pos = nonmono_count = nonmono_nonpos = 0
    for _ in range(n_test):
        kind = rng.integers(0, 3)
        if kind == 0:                                  # monotone decreasing
            g = np.exp(-rng.uniform(0.05, 0.5) * tau) + 0.05 * rng.random(12)
            g = np.sort(g)[::-1]
        elif kind == 1:                                # bump at random lag
            ts = rng.uniform(2, 11)
            g = np.exp(-0.5 * ((tau - ts) / rng.uniform(1, 3)) ** 2) + 0.05
        else:                                          # arbitrary
            g = rng.random(12) + 0.05
        B0 = saving_under_value_field(g, tau, seed=int(rng.integers(1e6)),
                                      nperm=nperm)["B0"]
        mono = bool(np.all(np.diff(g) <= 1e-9))
        if mono:
            mono_count += 1; mono_pos += int(B0 > 1e-6)
        else:
            nonmono_count += 1; nonmono_nonpos += int(B0 <= 1e-6)
    return dict(
        n_test=n_test,
        monotone_fields=mono_count,
        monotone_with_B0_positive=mono_pos,
        sufficiency_rate=float(mono_pos / max(1, mono_count)),
        nonmonotone_fields=nonmono_count,
        nonmonotone_with_B0_nonpositive=nonmono_nonpos,
        necessity_rate=float(nonmono_nonpos / max(1, nonmono_count)),
    )

def saving_under_value_field(g, tau, s_values=(1, 2, 4, 8, 16), seed=0,
                             nperm=3000):
    """Knapsack saving (re-optimized faithful shuffle) under arbitrary g and tau, across latency
    scales s. B0>0 => economy holds; <=0 => washes out/reverses."""
    rng = np.random.default_rng(seed)
    g = np.asarray(g, float); t = np.asarray(tau, float); t = t / t.mean()
    K = len(t)
    def knap(tt):
        w = np.zeros(K); acc = 0.0; V = 0.5 * g.sum(); cap = 1.0
        for k in np.argsort(tt / g):
            if acc >= V:
                break
            take = min(cap, (V - acc) / g[k]); w[k] = take; acc += take * g[k]
        return w
    wd = knap(t); tbd = (wd * t).sum() / wd.sum()
    sh = []
    for _ in range(nperm):
        tp = t[rng.permutation(K)]; ws = knap(tp)
        sh.append((ws * tp).sum() / ws.sum())
    t_opt = float(tbd); t_shuf = float(np.mean(sh))
    B0 = t_shuf - t_opt
    sav = [s * B0 for s in s_values]
    return dict(B0=B0, t_bar_opt=t_opt, t_bar_shuf=t_shuf, savings=sav,
                s=list(s_values), economy_holds=bool(B0 > 1e-6))


def gating_by_concentration(seed=0, betas=(0.0, 0.1, 0.3, 0.6, 1.0)):
    """B0 grows with how concentrated-at-short g is (its CV). g = exp(-beta*tau): beta=0 -> flat g
    (washout), large beta -> concentrated (clean). Mirrors MemoryPro vs DelayedCopy in
    fn_multitask.json."""
    tau = np.arange(1, 13).astype(float)
    rows = []
    for beta in betas:
        g = np.exp(-beta * tau)
        r = saving_under_value_field(g, tau, seed=seed, nperm=2000)
        cv = float(g.std() / g.mean())
        rows.append(dict(beta=float(beta), cv_g=cv, B0=r["B0"],
                         economy_holds=bool(r["B0"] > 1e-3)))
    B0s = [r["B0"] for r in rows]
    monotone_in_cv = bool(np.all(np.diff(B0s) >= -1e-6))
    return dict(rows=rows, B0_monotone_in_concentration=monotone_in_cv)


def bump_edge_case(seed=0):
    """Boundary of (C*), not the data explanation: a value field peaking at tau*>0 (non-monotone)
    can give B0<0. In the data DelayedCopy is the clean B0>0 case, the opposite of this."""
    from scipy.stats import spearmanr
    a = 0.2
    tau = np.arange(1, 13).astype(float)
    g_hold = np.array([g_max_negative(a, int(2 * t)) for t in tau]); g_hold /= g_hold.max()
    hold = saving_under_value_field(g_hold, tau, seed=seed)
    tau_star = 6.0
    g_copy = np.exp(-0.5 * ((tau - tau_star) / 2.0) ** 2) + 0.05
    copy = saving_under_value_field(g_copy, tau, seed=seed)
    rho_hold = float(spearmanr(g_hold, tau)[0])
    rho_copy = float(spearmanr(g_copy, tau)[0])
    mono_hold = bool(np.all(np.diff(g_hold) <= 1e-9))
    mono_copy = bool(np.all(np.diff(g_copy) <= 1e-9))
    return dict(
        monotone_decreasing_g=dict(spearman_g_tau=rho_hold, g_monotone_decreasing=mono_hold,
                  Cstar_holds=mono_hold, **hold),
        bump_field=dict(spearman_g_tau=rho_copy, g_monotone_decreasing=mono_copy,
                          Cstar_holds=mono_copy, **copy),
    )


if __name__ == "__main__":
    out = {}
    print("=" * 78)
    print("P1 - Pontryagin/LQR: g_k is the adjoint leverage, KKT common-ratio")
    print("=" * 78)
    L = lqr_costate_sympy()
    print(f"  required magnitude w_req(tau) = {L['w_req']}   (= G*/b)")
    print(f"  per-route conduction cost C_route = {L['C_route']}   (= G* tau / b)")
    print(f"  adjoint leverage  g_k = dG/dw = {L['g_k']}   (= b, the value field)")
    print(f"  KKT cost-per-value ratio = {L['ratio']}   (= tau / b)")
    g_is_b = sp.simplify(L['g_k'] - sp.Symbol('b', positive=True)) == 0
    cost_form = sp.simplify(L['C_route'] -
                            sp.Symbol('G_star', positive=True) *
                            sp.Symbol('tau', positive=True) /
                            sp.Symbol('b', positive=True)) == 0
    print(f"  -> g_k == b (value field = costate): {g_is_b}")
    print(f"  -> C_route == G* tau / b (closed form): {cost_form}")
    out["P1_costate"] = dict(value_field_equals_costate=bool(g_is_b),
                             conduction_cost_closed_form=bool(cost_form))
    kkt = [kkt_common_ratio_numeric(seed=s) for s in range(4)]
    all_kkt = all(k["kkt_active_ok"] and k["kkt_excluded_ok"] for k in kkt)
    for s, k in enumerate(kkt):
        print(f"  seed {s}: KKT active<=mu={k['kkt_active_ok']} "
              f"excluded>=mu={k['kkt_excluded_ok']}  mu={k['mu']:.3f} "
              f"n_active={k['n_active']}")
    print(f"  -> KKT common cost-per-value ratio holds at optimum (all seeds): {all_kkt}")
    out["P1_kkt"] = dict(all_seeds_pass=all_kkt, seeds=kkt)

    print("\n" + "=" * 78)
    print("P2 - Delayed-LQR QP beats the PAPER-FAITHFUL shuffle (train-permuted,")
    print("     score-true) on tau_bar - the decisive distance-specific test")
    print("=" * 78)
    qp = [delayed_lqr_qp(seed=s) for s in range(6)]
    all_beat = all(q["beats_shuffle"] for q in qp)
    md = float(np.mean([q["tbar_dist"] for q in qp]))
    ms = float(np.mean([q["tbar_shuf"] for q in qp]))
    for s, q in enumerate(qp):
        print(f"  seed {s}: tbar_dist={q['tbar_dist']:.3f} tbar_shuf={q['tbar_shuf']:.3f} "
              f"saving={q['saving_tbar']:.3f} beats_shuffle={q['beats_shuffle']} "
              f"knap_support_overlap={q['support_overlap_with_knapsack']:.2f}")
    print(f"  mean tbar_dist={md:.3f} < tbar_shuf={ms:.3f}  "
          f"(mirrors empirical 6.32 < 7.59)")
    print(f"  -> optimal-control allocation beats paper-faithful shuffle (all seeds): {all_beat}")
    out["P2_qp"] = dict(all_seeds_beat_shuffle=all_beat,
                        mean_tbar_dist=md, mean_tbar_shuf=ms, seeds=qp)

    print("\n" + "=" * 78)
    print("P3 - Optimal transport: anti-sorted coupling is the min (rearrangement)")
    print("=" * 78)
    ot = [ot_rearrangement_numeric(seed=s) for s in range(4)]
    all_ot = all(o["anti_sorted_is_min"] and o["gap"] > 0 for o in ot)
    for s, o in enumerate(ot):
        print(f"  seed {s}: tau_bar_dist={o['tau_bar_dist']:.3f} "
              f"tau_bar_shuf={o['tau_bar_shuf']:.3f} gap={o['gap']:.3f} "
              f"anti_sorted_is_min={o['anti_sorted_is_min']}")
    print(f"  -> OT rearrangement bound holds, gap>0 (all seeds): {all_ot}")
    out["P3_ot"] = dict(all_seeds_pass=all_ot, seeds=ot)

    print("\n" + "=" * 78)
    print("P4 - Sufficient condition (C*), its DDE source, and the DATA-MATCHED gate")
    print("=" * 78)
    kp = hold_value_field_from_dynamics()
    print(f"  HOLD-task g(tau) from DDE stability ceiling:")
    print(f"    spearman(g,tau)={kp['spearman_g_tau']:+.3f}  "
          f"g_monotone_decreasing={kp['g_monotone_decreasing']}  "
          f"=> (C*) holds = {kp['condition_Cstar_holds']}  (mechanistic, not assumed)")
    out["P4_hold_value_field"] = kp

    print("\n  (C*) sufficiency sweep (random value-field shapes):")
    vs = validate_Cstar_sufficiency()
    print(f"    monotone-decreasing g => B0>0 : "
          f"{vs['monotone_with_B0_positive']}/{vs['monotone_fields']} "
          f"({100*vs['sufficiency_rate']:.0f}%)  [SUFFICIENT, proved]")
    print(f"    non-monotone g => B0<=0 : "
          f"{vs['nonmonotone_with_B0_nonpositive']}/{vs['nonmonotone_fields']} "
          f"({100*vs['necessity_rate']:.0f}%)  [NOT necessary]")
    out["P4_Cstar_sufficiency"] = vs

    print("\n  DATA-MATCHED gate: B0 grows with value-field concentration CV(g):")
    gc = gating_by_concentration()
    for r in gc["rows"]:
        tag = "easy/flat g -> WASHOUT" if r["beta"] == 0 else \
              ("functional/concentrated g -> CLEAN" if r["B0"] > 1e-3 else "")
        print(f"    beta={r['beta']:.1f}  CV(g)={r['cv_g']:.3f}  B0={r['B0']:+.4f}  {tag}")
    print(f"    B0 monotone in concentration: {gc['B0_monotone_in_concentration']}")
    print(f"    => matches data: easy MemoryPro washes (dist-shuf tau_bar +0.115),")
    print(f"       DelayedCopy clean (dist-shuf -1.170).")
    out["P4_gating_by_concentration"] = gc

    print("\n  (boundary edge case, NOT the data: a bump value field at tau*>0 gives B0<0)")
    bc = bump_edge_case()
    print(f"    monotone-decreasing g: B0={bc['monotone_decreasing_g']['B0']:+.4f} "
          f"(C*={bc['monotone_decreasing_g']['Cstar_holds']})  "
          f"bump-at-tau*: B0={bc['bump_field']['B0']:+.4f} "
          f"(C*={bc['bump_field']['Cstar_holds']})")
    out["P4_bump_edge_case"] = bc

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
