"""Does the conduction-time economy generalize to an echo-state reservoir, where the readout
(not BPTT) does the learning?

Delayed leaky-integrator ESN: units in 3D space, state i reads state j from tau_ij=round(d_ij/v)
steps ago. There's no learnable mass on the raw fixed-random recurrent matrix, so we test the
economy on the object that carries the learned allocation, in two regimes:

  Regime A -- trained-gain reservoir (the clean HLP test). Each recurrent edge keeps its fixed
    random structure but gets a learnable nonneg gain g_ij>=0. Train {gains, readout} under
    plain (not distance-weighted) L1 on the gains. Economy object: C = sum |W_rec_ij|*g_ij*tau.
  Regime B -- readout-only pure ESN. Recurrent matrix fully fixed; only a ridge/gradient
    readout learns. Per-unit importance r_i = ||W_out[:,i]||; effective travel-time is
    C = sum_ij (r_i*|W_rec_ij|)*tau_ij. Zero learning in the recurrent weights.

Protocol mirrors the rate-RNN test: delayed-copy task (lag is load-bearing), conditions
{distance, shuffled} share the same tau multiset (shuffled = symmetric off-diagonal
permutation), compare C and weighted-mean lag at matched accuracy, paired per seed.

Run:
  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=3 \
    python scripts/law/law_reservoir.py --device mps
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sdrnn.geometry import grid_coordinates
from sdrnn.delays import integer_delays

from sdrnn.tasks import DelayedCopyTask

OUT = str(ROOT / "results" / "law" / "law_reservoir.json")


# Geometry + delays (positions on a centered grid).
def make_geometry(n, dim, velocity, max_delay, seed, jitter=0.15):
    """Jittered grid positions -> distances -> integer geometric tau."""
    g = torch.Generator().manual_seed(seed + 101)
    coords = grid_coordinates(n, dim)
    coords = coords + jitter * torch.randn(coords.shape, generator=g)
    dist = torch.cdist(coords, coords, p=2)
    tau_geo = integer_delays(dist, velocity, max_delay)  # the TRUE geometric lag
    return coords, dist, tau_geo


def shuffle_offdiag(tau, seed):
    """Same-histogram geometry scramble: symmetric off-diagonal permutation.
    Mirrors sdrnn.model._apply_delay_control('shuffled')."""
    n = tau.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    out = tau.clone()
    g = torch.Generator().manual_seed(seed + 1009)
    perm = torch.randperm(n * n - n, generator=g)
    out[off] = out[off][perm]
    return out


def make_reservoir(n, density, spectral_radius, seed):
    """Sparse fixed random recurrent skeleton, spectral-radius rescaled."""
    g = torch.Generator().manual_seed(seed + 202)
    W = torch.randn(n, n, generator=g)
    mask = (torch.rand(n, n, generator=g) < density).float()
    mask.fill_diagonal_(0.0)  # no self-loops in the recurrent skeleton
    W = W * mask
    # rescale to target spectral radius (echo-state property knob)
    with torch.no_grad():
        eig = torch.linalg.eigvals(W)
        rho = eig.abs().max().item()
        if rho > 1e-6:
            W = W * (spectral_radius / rho)
    return W


# Delayed leaky-integrator reservoir dynamics:
#   x[t] = (1-a) x[t-1] + a * tanh( W_in u[t] + sum_j (W_rec_ij * g_ij) x_j[t-tau_ij] + b )
# g_ij: all-ones in regime B, learnable nonneg in regime A.
class DelayedReservoir(nn.Module):
    def __init__(self, n, in_size, W_rec, tau, leak, seed, train_gain):
        super().__init__()
        self.n = n
        self.leak = leak
        self.max_delay = int(tau.max().item())
        g = torch.Generator().manual_seed(seed + 303)
        # input weights: fixed random (reservoir input scaling), NOT trained.
        W_in = torch.randn(n, in_size, generator=g) * (1.0 / math.sqrt(in_size))
        b = torch.zeros(n)
        self.register_buffer("W_in", W_in)
        self.register_buffer("b", b)
        self.register_buffer("W_rec", W_rec)         # fixed random skeleton
        self.register_buffer("tau", tau.long())
        # per-distinct-lag masks of the fixed skeleton, as buffers so they follow the device
        self._lag_ints = []
        for k, d in enumerate(torch.unique(self.tau)):
            di = int(d.item())
            self.register_buffer(f"_lagmask_{k}", (self.tau == di).to(W_rec.dtype))
            self._lag_ints.append((k, di))
        self.train_gain = train_gain
        if train_gain:
            # learnable nonneg gain per edge (regime A); raw -> softplus, init effective gain ~1
            raw = torch.zeros(n, n)
            self.gain_raw = nn.Parameter(raw)
        # else: gains are implicitly 1 (regime B)

    def effective_gain(self):
        if self.train_gain:
            return F.softplus(self.gain_raw + 0.5413)  # softplus(0.5413)=1.0
        return None

    def forward(self, u):
        # u: (B, T, in_size) -> states X: (B, T, n)
        B, T, _ = u.shape
        device = u.device
        n = self.n
        gain = self.effective_gain()
        if gain is not None:
            Wg = self.W_rec * gain
        else:
            Wg = self.W_rec
        # per-distinct-lag masked recurrent matrices (transposed once for the matmul); looping
        # only over the lags that occur beats a dense (L,n,n) bmm since the skeleton is sparse
        groups = [(di, (Wg * getattr(self, f"_lagmask_{k}")).t()) for k, di in self._lag_ints]
        # delay line as a list of past states (states[s] = x at time s); a list rather than a
        # ring tensor keeps the autograd graph lean. Pre-start reads are zeros.
        zero = torch.zeros(B, n, device=device, dtype=u.dtype)
        states = []
        Win_b = u @ self.W_in.t() + self.b           # (B,T,n) precompute input drive
        for t in range(T):
            rec = None
            for di, Wt in groups:
                src = t - di
                h_src = states[src] if src >= 0 else zero   # x_{t-di}, zeros before t=0
                term = h_src @ Wt
                rec = term if rec is None else rec + term
            pre = Win_b[:, t, :] if rec is None else Win_b[:, t, :] + rec
            prev = states[t - 1] if t >= 1 else zero     # x_{t-1} for the leak term
            x = (1.0 - self.leak) * prev + self.leak * torch.tanh(pre)
            states.append(x)
        return torch.stack(states, dim=1)            # (B, T, n)


class ReservoirNet(nn.Module):
    """Delayed reservoir + linear readout. Regime B: only readout learns; regime A: readout +
    per-edge gains."""
    def __init__(self, task, n, W_rec, tau, leak, seed, train_gain):
        super().__init__()
        self.reservoir = DelayedReservoir(n, task.input_size, W_rec, tau, leak,
                                          seed, train_gain)
        self.readout = nn.Linear(n, task.output_size, bias=True)
        with torch.no_grad():
            self.readout.weight.zero_(); self.readout.bias.zero_()

    def forward(self, u):
        X = self.reservoir(u)
        return self.readout(X)


@torch.no_grad()
def conduction_cost(net, tau_geo):
    """C = sum mass_ij*tau and weighted-mean lag = C/sum mass, against the fixed geometric tau
    for all conditions (only the learned allocation differs). Returns both allocation objects:
      'gain'   : mass = |W_rec| * effective_gain   (regime A)
      'readout': mass = r_i * |W_rec_ij|, r_i = ||W_out[:,i]||  (regime B)
    """
    res = net.reservoir
    Wabs = res.W_rec.detach().abs().cpu()
    tau = tau_geo.float().cpu()

    out = {}
    # gain-allocation object
    gain = res.effective_gain()
    if gain is not None:
        mass_g = Wabs * gain.detach().cpu()
    else:
        mass_g = Wabs.clone()  # pure ESN: all gains 1 (identical dist vs shuf -> control)
    s = mass_g.sum().item()
    raw = (mass_g * tau).sum().item()
    out["gain"] = dict(raw_cost=raw, wmean_tau=raw / max(s, 1e-12), mass_sum=s)

    # readout-effective object
    r = net.readout.weight.detach().abs().cpu()          # (out, n)
    imp = r.norm(dim=0)                                   # (n,) per-unit importance
    # edge i<-j mass = importance(target i) * |W_rec_ij|
    mass_r = imp.unsqueeze(1) * Wabs                      # (i, j)
    s2 = mass_r.sum().item()
    raw2 = (mass_r * tau).sum().item()
    out["readout"] = dict(raw_cost=raw2, wmean_tau=raw2 / max(s2, 1e-12), mass_sum=s2)
    return out


@torch.no_grad()
def evaluate(net, task, device, batch=512):
    net.eval()
    g = torch.Generator().manual_seed(987654)
    x, tgt, mask = task.generate(batch, generator=g)
    x, tgt, mask = x.to(device), tgt.to(device), mask.to(device)
    out = net(x)
    return float(task.accuracy(out, tgt, mask)), float(task.loss(out, tgt, mask).item())


@torch.no_grad()
def ridge_fit_readout(net, task, device, ridge_alpha, n_collect, batch):
    """Closed-form ridge readout (canonical ESN training). Collect reservoir states on driven
    trials, solve W_out by ridge on the scored timesteps. No gradient touches the recurrent
    weights, and the closed-form solve matches distance/shuffled accuracy by construction."""
    net.eval()
    g = torch.Generator().manual_seed(2024)
    feats, tgts = [], []
    collected = 0
    while collected < n_collect:
        x, tgt, mask = task.generate(batch, generator=g)
        X = net.reservoir(x.to(device)).cpu()           # (B,T,n)
        sel = mask.bool()                                # scored timesteps
        feats.append(X[sel]); tgts.append(tgt[sel])
        collected += int(sel.sum().item())
    H = torch.cat(feats, 0)                               # (M, n)
    Y = torch.cat(tgts, 0)                                # (M,)
    C = task.output_size
    Yoh = F.one_hot(Y, C).float()                        # (M, C)
    Hb = torch.cat([H, torch.ones(H.shape[0], 1)], 1)    # bias column
    A = Hb.t() @ Hb + ridge_alpha * torch.eye(Hb.shape[1])
    B = Hb.t() @ Yoh
    Wb = torch.linalg.solve(A, B)                         # (n+1, C)
    Wout = Wb[:-1].t().contiguous()                      # (C, n)
    bout = Wb[-1].contiguous()                           # (C,)
    net.readout.weight.copy_(Wout.to(device))
    net.readout.bias.copy_(bout.to(device))


def train_one(task, n, density, spectral_radius, leak, velocity, max_delay,
              dim, cond, seed, regime, steps, lr, batch, device, reg_lambda,
              ridge_alpha, ridge_collect):
    torch.manual_seed(seed)
    _, _, tau_geo = make_geometry(n, dim, velocity, max_delay, seed)
    if cond == "distance":
        tau = tau_geo.clone()
    elif cond == "shuffled":
        tau = shuffle_offdiag(tau_geo, seed)
    else:
        raise ValueError(cond)

    W_rec = make_reservoir(n, density, spectral_radius, seed)
    train_gain = (regime == "A")
    net = ReservoirNet(task, n, W_rec, tau, leak, seed, train_gain).to(device)

    if regime == "B":
        # canonical ESN: fixed reservoir, closed-form ridge readout
        ridge_fit_readout(net, task, device, ridge_alpha, ridge_collect, batch)
    else:
        # regime A: gradient on {gains, readout} under plain-L1 magnitude pressure
        params = list(net.readout.parameters()) + [net.reservoir.gain_raw]
        opt = torch.optim.Adam(params, lr=lr)
        dgen = torch.Generator().manual_seed(seed + 1)
        for step in range(1, steps + 1):
            net.train()
            x, tgt, mask = task.generate(batch, generator=dgen)
            x, tgt, mask = x.to(device), tgt.to(device), mask.to(device)
            out = net(x)
            loss = task.loss(out, tgt, mask)
            if reg_lambda > 0:
                # plain L1 on gains (not distance-weighted): magnitude pressure, not a tautology
                loss = loss + reg_lambda * net.reservoir.effective_gain().abs().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

    acc, eloss = evaluate(net, task, device)
    cost = conduction_cost(net, tau_geo)
    return dict(acc=acc, loss=eloss, cost=cost)


# stats helpers (mirror conduction_cost.py)
def paired_stats(records, regime_key, metric):
    dist = {r["seed"]: r["cost"][regime_key][metric] for r in records if r["cond"] == "distance"}
    shuf = {r["seed"]: r["cost"][regime_key][metric] for r in records if r["cond"] == "shuffled"}
    seeds = sorted(set(dist) & set(shuf))
    diffs = np.array([dist[s] - shuf[s] for s in seeds], float)
    if len(diffs) == 0:
        return None
    mean = float(diffs.mean())
    if len(diffs) > 1 and diffs.std(ddof=1) > 0:
        t = float(mean / (diffs.std(ddof=1) / math.sqrt(len(diffs))))
    else:
        t = float("nan")
    # exact two-sided sign test
    nz = diffs[diffs != 0]
    nneg = int((nz < 0).sum()); npos = int((nz > 0).sum()); nn = len(nz)
    k = min(nneg, npos)
    p = min(1.0, 2 * sum(math.comb(nn, i) for i in range(k + 1)) / (2 ** nn)) if nn > 0 else 1.0
    return dict(mean_diff=mean, t=t, n=len(diffs),
                n_dist_lt_shuf=int((diffs < 0).sum()), sign_p=p,
                per_seed={int(s): float(dist[s] - shuf[s]) for s in seeds})


def acc_paired(records):
    dist = {r["seed"]: r["acc"] for r in records if r["cond"] == "distance"}
    shuf = {r["seed"]: r["acc"] for r in records if r["cond"] == "shuffled"}
    seeds = sorted(set(dist) & set(shuf))
    diffs = np.array([dist[s] - shuf[s] for s in seeds], float)
    return dict(mean_diff=float(diffs.mean()),
                per_seed={int(s): float(dist[s] - shuf[s]) for s in seeds},
                dist_mean=float(np.mean(list(dist.values()))),
                shuf_mean=float(np.mean(list(shuf.values()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="reservoir units")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--spectral_radius", type=float, default=0.95)
    ap.add_argument("--leak", type=float, default=0.6)
    ap.add_argument("--velocity", type=float, default=0.08)
    ap.add_argument("--max_delay", type=int, default=14)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--reg_lambda", type=float, default=1e-3)
    ap.add_argument("--ridge_alpha", type=float, default=1.0, help="ridge L2 for ESN readout (regime B)")
    ap.add_argument("--ridge_collect", type=int, default=20000, help="scored timesteps for ridge fit")
    ap.add_argument("--regimes", default="A,B", help="A=trained-gain, B=readout-only(ridge)")
    ap.add_argument("--lag", type=int, default=6)
    ap.add_argument("--seq_len", type=int, default=24)
    ap.add_argument("--n_symbols", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    device = torch.device(a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu")

    task = DelayedCopyTask(n_symbols=a.n_symbols, lag=a.lag, seq_len=a.seq_len, noise=0.0)
    CONDS = ["distance", "shuffled"]
    regimes = [r for r in a.regimes.split(",") if r]

    all_results = {"_meta": dict(vars(a)), "regimes": {}}
    t0 = time.time()
    for regime in regimes:
        # the object whose allocation the regime actually trains
        regime_obj = "gain" if regime == "A" else "readout"
        records = []
        print(f"\n##### REGIME {regime} "
              f"({'trained-gain reservoir' if regime=='A' else 'readout-only pure ESN'}) "
              f"-- economy object = '{regime_obj}' #####", flush=True)
        for seed in range(a.seeds):
            for cond in CONDS:
                r = train_one(task, a.n, a.density, a.spectral_radius, a.leak,
                              a.velocity, a.max_delay, a.dim, cond, seed, regime,
                              a.steps, a.lr, a.batch, device, a.reg_lambda,
                              a.ridge_alpha, a.ridge_collect)
                r.update(cond=cond, seed=seed)
                records.append(r)
                gc = r["cost"]["gain"]; rc = r["cost"]["readout"]
                print(f"[{regime}] seed={seed} {cond:9s} acc={r['acc']:.3f} "
                      f"gain[C={gc['raw_cost']:.1f} wm={gc['wmean_tau']:.3f}] "
                      f"readout[C={rc['raw_cost']:.2f} wm={rc['wmean_tau']:.3f}]",
                      flush=True)

        # paired economy on the regime's own object and the other one
        eco = {}
        for obj in ("gain", "readout"):
            eco[obj] = dict(raw_cost=paired_stats(records, obj, "raw_cost"),
                            wmean_tau=paired_stats(records, obj, "wmean_tau"))
        acc = acc_paired(records)
        all_results["regimes"][regime] = dict(
            regime_object=regime_obj, records=records, economy=eco, accuracy=acc)

        print(f"\n=== REGIME {regime} SUMMARY (primary object '{regime_obj}') ===")
        accgap = abs(acc["dist_mean"] - acc["shuf_mean"])
        print(f"  acc: distance={acc['dist_mean']:.3f} shuffled={acc['shuf_mean']:.3f} "
              f"gap={accgap:.3f} {'MATCHED' if accgap < 0.02 else 'CHECK gap>0.02'}")
        for obj in ("gain", "readout"):
            for metric in ("raw_cost", "wmean_tau"):
                st = eco[obj][metric]
                if st is None:
                    continue
                star = " <-- PRIMARY" if obj == regime_obj else ""
                print(f"  ECONOMY[{obj:7s}/{metric:9s}] dist-shuf mean={st['mean_diff']:+.4g} "
                      f"t={st['t']:.2f} dist<shuf {st['n_dist_lt_shuf']}/{st['n']} "
                      f"sign_p={st['sign_p']:.3f}{star}")

    all_results["_meta"]["wall_seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {a.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
