"""Does the conduction-time economy hold beyond RNNs, in a delayed-attention block?

The SDRNN result is that training drives sum|W|*tau below a histogram-matched
shuffled control at matched accuracy. Here we test whether that's architecture-
general with a minimal distance-delayed attention block, asking both (a) economy
(distance < shuffled cost, matched acc) and (b) function (distance acc > shuffled).

Architecture: D feature channels at positions in space_dim-D space (the spatial
units, as neurons are in SDRNN). A causal single head computes the context
c_t = sum_s a_ts v_s over the sequence; the delay enters in the channel-mixing
output projection W_o, with source channel j of the context reaching target i a
distance-proportional lag later:
    y[t,i] = sum_j Wo_ij * c[t - tau_ij, j],   tau_ij = round(dist_ij / v)
A plain transformer is the tau=1 / single-step special case. Conduction cost
C = sum_ij |Wo_ij| * tau_ij is scored against the same fixed geometric tau for
every condition; shuffled = same histogram, geometry scrambled with a fixed seed.

Tasks: long-range delayed-copy (lag load-bearing) and an easy memory hold (wash
control). Attention can reach arbitrarily far, so whether delays matter is empirical.

  PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=2 \
    python scripts/economy/transformer_economy.py --device mps
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

from sdrnn.tasks import DelayedCopyTask, MemoryProTask

OUT = str(ROOT / "results" / "economy" / "transformer_economy.json")

# Geometry + delays (mirror sdrnn.delays / sdrnn.geometry, kept self-contained).
def make_positions(D, space_dim, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(D, space_dim, generator=g)  # uniform in unit cube


def distance_matrix(pos):
    d = torch.cdist(pos, pos)  # (D, D) Euclidean
    return d


def integer_tau(dist, velocity, max_delay, min_delay=1):
    with torch.no_grad():
        tau = torch.round(dist / velocity).long().clamp_(min_delay, max_delay)
    return tau


def shuffle_offdiag(tau, seed):
    """Histogram-matched geometry-scramble: permute off-diagonal lags, fixed seed.
    Mirrors SDRNN._make_delay_shuffle_perm / _apply_delay_control('shuffled')."""
    D = tau.shape[0]
    off = ~torch.eye(D, dtype=torch.bool)
    out = tau.clone()
    g = torch.Generator().manual_seed(seed + 1009)
    perm = torch.randperm(D * D - D, generator=g)
    vals = out[off]
    out[off] = vals[perm]
    return out


# Delayed-attention block
class DelayedAttention(nn.Module):
    """One causal single-head attention block whose OUTPUT channel-mixing is
    distance-delayed. delay_control in {none, distance, shuffled}.

    none     -> tau = 1 everywhere (a plain transformer block; single-step).
    distance -> tau_ij = round(dist_ij / v).
    shuffled -> same histogram as distance, geometry scrambled.
    """

    def __init__(self, D, space_dim, velocity, max_delay, delay_control, seed):
        super().__init__()
        self.D = D
        self.delay_control = delay_control
        self.max_delay = max_delay
        gen = torch.Generator().manual_seed(seed)

        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self.Wv = nn.Linear(D, D, bias=False)
        self.Wo = nn.Linear(D, D, bias=False)  # the DELAYED channel-mixing (analogue of W_rec)
        for lin in (self.Wq, self.Wk, self.Wv, self.Wo):
            with torch.no_grad():
                lin.weight.normal_(0.0, 1.0 / math.sqrt(D), generator=gen)

        pos = make_positions(D, space_dim, seed + 7)
        dist = distance_matrix(pos)
        self.register_buffer("dist", dist, persistent=False)
        if delay_control == "none":
            tau = torch.ones(D, D, dtype=torch.long)
        else:
            tau = integer_tau(dist, velocity, max_delay)
            if delay_control == "shuffled":
                tau = shuffle_offdiag(tau, seed)
        self.register_buffer("tau", tau, persistent=False)
        self.scale = 1.0 / math.sqrt(D)

    def forward(self, x):
        # x: (B, T, D). Causal single-head attention -> context c: (B, T, D).
        B, T, D = x.shape
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,T,T)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        c = torch.matmul(att, v)  # (B, T, D) context

        # Delayed channel-mixing output: y[t,i] = sum_j Wo_ij * c[t - tau_ij, j].
        # Build, per distinct lag d, the masked Wo and apply it to c shifted by d
        # query-steps (grouped-delay trick from sdrnn.delays.grouped_delay_weights).
        Wo = self.Wo.weight  # (out=i, in=j)
        tau = self.tau.to(x.device)
        y = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        zero = torch.zeros_like(Wo)
        for d in torch.unique(tau):
            d_int = int(d.item())
            masked = torch.where(tau == d, Wo, zero)  # (i, j)
            if d_int == 0:
                c_shift = c
            else:
                # c[t - d]: shift along time; pre-pad with zeros (no signal yet).
                c_shift = torch.zeros_like(c)
                if d_int < T:
                    c_shift[:, d_int:, :] = c[:, : T - d_int, :]
            # y[:, t, i] += sum_j masked_ij * c_shift[:, t, j]
            y = y + torch.matmul(c_shift, masked.t())
        return y

    @torch.no_grad()
    def conduction_cost(self):
        """sum_ij |Wo_ij| * tau_ij against the fixed geometric tau, plus the
        weighted-mean lag. All conditions score against the same geometric tau
        (from this block's distances), so only learned |Wo| differs."""
        Wo = self.Wo.weight.detach().abs().cpu()
        geo_tau = integer_tau(self.dist.cpu(), self._velocity_used(), self.max_delay).float()
        wsum = Wo.sum().item()
        raw = (Wo * geo_tau).sum().item()
        wmean = raw / max(wsum, 1e-12)
        return dict(raw_cost=raw, wmean_tau=wmean, wsum=wsum)

    def _velocity_used(self):
        return self._vel

    def set_velocity(self, v):
        self._vel = v


# Tiny model: embed -> 1 delayed-attention block (+ residual) -> readout.
class DelayedAttnNet(nn.Module):
    def __init__(self, in_size, out_size, D, space_dim, velocity, max_delay,
                 delay_control, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed + 3)
        self.embed = nn.Linear(in_size, D, bias=True)
        self.block = DelayedAttention(D, space_dim, velocity, max_delay, delay_control, seed)
        self.block.set_velocity(velocity)
        self.ln = nn.LayerNorm(D)
        self.readout = nn.Linear(D, out_size, bias=True)
        with torch.no_grad():
            self.embed.weight.normal_(0.0, 1.0 / math.sqrt(in_size), generator=gen)
            self.readout.weight.normal_(0.0, 1.0 / math.sqrt(D), generator=gen)

    def forward(self, x):
        h = self.embed(x)
        h = h + self.block(self.ln(h))  # residual delayed-attention
        return self.readout(h)

    def conduction_cost(self):
        return self.block.conduction_cost()


# Train / eval
def train_one(task, D, space_dim, velocity, max_delay, delay_control, seed,
              steps, lr, batch, device, reg_lambda):
    torch.manual_seed(seed)
    dgen = torch.Generator().manual_seed(seed + 1)
    model = DelayedAttnNet(task.input_size, task.output_size, D, space_dim,
                           velocity, max_delay, delay_control, seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        model.train()
        x, tgt, mask = task.generate(batch, generator=dgen)
        x, tgt, mask = x.to(device), tgt.to(device), mask.to(device)
        out = model(x)
        loss = task.loss(out, tgt, mask)
        if reg_lambda > 0:
            # Plain (non-spatial) L1 on the channel-mixing: pressure is on |W| only,
            # so the distance-vs-shuffled cost separation isn't a loss tautology.
            loss = loss + reg_lambda * model.block.Wo.weight.abs().mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    acc, eloss = evaluate(model, task, device)
    cost = model.conduction_cost()
    return dict(acc=acc, loss=eloss, **cost)


@torch.no_grad()
def evaluate(model, task, device):
    model.eval()
    g = torch.Generator().manual_seed(12345)
    x, tgt, mask = task.generate(512, generator=g)
    x, tgt, mask = x.to(device), tgt.to(device), mask.to(device)
    out = model(x)
    return float(task.accuracy(out, tgt, mask)), float(task.loss(out, tgt, mask).item())


def summarize(records):
    """records: list of dict with cond/seed/acc/raw_cost/wmean_tau."""
    by_cond = {}
    for r in records:
        by_cond.setdefault(r["cond"], []).append(r)
    out = {}
    for cond, rs in by_cond.items():
        out[cond] = dict(
            acc=float(np.mean([r["acc"] for r in rs])),
            acc_std=float(np.std([r["acc"] for r in rs])),
            raw_cost=float(np.mean([r["raw_cost"] for r in rs])),
            wmean_tau=float(np.mean([r["wmean_tau"] for r in rs])),
            n=len(rs),
        )
    return out


def paired_stats(records, metric):
    """Paired distance - shuffled on `metric`, per seed."""
    dist = {r["seed"]: r[metric] for r in records if r["cond"] == "distance"}
    shuf = {r["seed"]: r[metric] for r in records if r["cond"] == "shuffled"}
    seeds = sorted(set(dist) & set(shuf))
    diffs = [dist[s] - shuf[s] for s in seeds]
    if not diffs:
        return None
    diffs = np.array(diffs)
    mean = diffs.mean()
    if len(diffs) > 1 and diffs.std(ddof=1) > 0:
        t = mean / (diffs.std(ddof=1) / math.sqrt(len(diffs)))
    else:
        t = float("nan")
    return dict(mean_diff=float(mean), t=float(t), n=len(diffs),
                n_dist_lt_shuf=int((diffs < 0).sum()),
                per_seed={int(s): float(dist[s] - shuf[s]) for s in seeds})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=48, help="feature channels = spatial units")
    ap.add_argument("--space_dim", type=int, default=3)
    ap.add_argument("--velocity", type=float, default=0.08)
    ap.add_argument("--max_delay", type=int, default=14)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--reg_lambda", type=float, default=1e-3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tasks", default="copy,memory")
    a = ap.parse_args()
    device = torch.device(a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu")

    TASKS = {}
    if "copy" in a.tasks:
        # long-range delayed copy: lag is load-bearing (the delay-friendly task).
        TASKS["delayedcopy"] = dict(factory=lambda: DelayedCopyTask(n_symbols=4, lag=6, seq_len=24, noise=0.0),
                                    steps=a.steps)
    if "memory" in a.tasks:
        # easy memory hold: a WASH control (delays not load-bearing).
        TASKS["memorypro_easy"] = dict(factory=lambda: MemoryProTask(n_choices=4, delay_steps=6, noise=0.1),
                                       steps=max(1200, a.steps // 2))

    CONDS = ["none", "distance", "shuffled"]
    all_results = {}
    t0 = time.time()
    for tname, tcfg in TASKS.items():
        task = tcfg["factory"]()
        steps = tcfg["steps"]
        records = []
        for seed in range(a.seeds):
            for cond in CONDS:
                r = train_one(task, a.D, a.space_dim, a.velocity, a.max_delay,
                              cond, seed, steps, a.lr, a.batch, device, a.reg_lambda)
                r.update(cond=cond, seed=seed)
                records.append(r)
                print(f"[{tname}] seed={seed} cond={cond:8s} acc={r['acc']:.3f} "
                      f"raw_cost={r['raw_cost']:.1f} wmean_tau={r['wmean_tau']:.3f}", flush=True)
        summ = summarize(records)
        cost_stats = paired_stats(records, "raw_cost")
        wmean_stats = paired_stats(records, "wmean_tau")
        acc_stats = paired_stats(records, "acc")  # function: distance - shuffled acc
        all_results[tname] = dict(summary=summ, raw_cost_paired=cost_stats,
                                  wmean_tau_paired=wmean_stats, acc_paired=acc_stats,
                                  records=records, steps=steps)
        print(f"\n=== {tname} SUMMARY ===")
        for cond in CONDS:
            s = summ[cond]
            print(f"  {cond:8s} acc={s['acc']:.3f}+/-{s['acc_std']:.3f} "
                  f"raw_cost={s['raw_cost']:.1f} wmean_tau={s['wmean_tau']:.3f}")
        if cost_stats:
            print(f"  ECONOMY (dist-shuf) raw_cost: mean={cost_stats['mean_diff']:.1f} "
                  f"t={cost_stats['t']:.2f} dist<shuf {cost_stats['n_dist_lt_shuf']}/{cost_stats['n']}")
            print(f"  ECONOMY (dist-shuf) wmean_tau: mean={wmean_stats['mean_diff']:.3f} "
                  f"t={wmean_stats['t']:.2f} dist<shuf {wmean_stats['n_dist_lt_shuf']}/{wmean_stats['n']}")
        if acc_stats:
            print(f"  FUNCTION (dist-shuf) acc: mean={acc_stats['mean_diff']:.3f} "
                  f"t={acc_stats['t']:.2f} dist>shuf {acc_stats['n'] - acc_stats['n_dist_lt_shuf']}/{acc_stats['n']}\n")

    all_results["_meta"] = dict(D=a.D, space_dim=a.space_dim, velocity=a.velocity,
                                max_delay=a.max_delay, seeds=a.seeds, lr=a.lr,
                                batch=a.batch, reg_lambda=a.reg_lambda,
                                wall_seconds=time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"wrote {OUT} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
