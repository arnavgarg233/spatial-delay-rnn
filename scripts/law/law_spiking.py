"""Does the conduction-time economy reproduce in a spiking (LIF) network?

Recurrent leaky integrate-and-fire network: membrane V leaks and integrates synaptic current,
a hard threshold emits a binary spike (V reset after), and recurrent input is spike current
routed through per-synapse delays tau_ij = round(d_ij/v) clipped to [1, max_delay]:
    I_i[t] = sum_j W_ij * s_j[t - tau_ij]
Trained by surrogate gradient (fast-sigmoid) through the non-differentiable spike.

Task: delayed match-to-sample over spike-rate-coded inputs. A cue is shown, then a blank delay
that forces recurrent memory, then the class is read out over a response window.

Protocol: distance = tau from geometry; shuffled = same tau multiset, off-diagonal symmetric
permutation (matches rate-RNN _apply_delay_control('shuffled')). Both trained to the task; at
matched accuracy we measure C = sum |W|*tau and tau_bar = C/sum|W| on the learned weights
against the same geometric tau. Dose-response: saving grows as velocity drops.
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # repo root: scripts/<pillar>/file.py -> up 3
sys.path.insert(0, str(ROOT / "src"))

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from sdrnn.geometry import NeuronGeometry
from sdrnn.delays import integer_delays


class SurrogateSpike(torch.autograd.Function):
    """Heaviside(v - thr) forward; fast-sigmoid surrogate backward.
    grad = 1 / (1 + beta*|x|)^2 with x = v - thr.
    """

    beta = 10.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + SurrogateSpike.beta * x.abs()) ** 2
        return grad_out * sg


spike_fn = SurrogateSpike.apply


# Symmetric delay shuffle (matches the rate-RNN control): permute off-diagonal tau entries by a
# fixed permutation -> same delay multiset, geometry-to-delay correspondence scrambled.
def shuffle_delays(tau: torch.Tensor, seed: int) -> torch.Tensor:
    n = tau.shape[0]
    gen = torch.Generator().manual_seed(int(seed) + 1009)
    perm = torch.randperm(n * n - n, generator=gen)
    off = ~torch.eye(n, dtype=torch.bool, device=tau.device)
    out = tau.clone()
    vals = out[off]
    out[off] = vals[perm]
    return out


# Recurrent LIF spiking network with per-synapse conduction delays.
class DelayLIF(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, tau_delay, max_delay,
                 v_decay=0.9, syn_decay=0.8, thresh=1.0, seed=0):
        super().__init__()
        self.n_hidden = n_hidden
        self.max_delay = int(max_delay)
        self.v_decay = v_decay        # membrane leak per step
        self.syn_decay = syn_decay    # synaptic current decay per step
        self.thresh = thresh
        self.register_buffer("tau", tau_delay.long())   # (N,N) integer lags
        # group edges by delay once (constant over a forward pass)
        self._delay_groups = sorted(int(d.item()) for d in torch.unique(tau_delay))

        g = torch.Generator().manual_seed(seed)
        self.w_in = nn.Parameter(torch.randn(n_hidden, n_in, generator=g) / math.sqrt(n_in))
        # recurrent: scaled so spiking is sane at init
        self.w_rec = nn.Parameter(torch.randn(n_hidden, n_hidden, generator=g) / math.sqrt(n_hidden))
        self.w_out = nn.Parameter(torch.randn(n_out, n_hidden, generator=g) / math.sqrt(n_hidden))
        # zero recurrent self-connections (delay tau_ii is undefined / identity)
        with torch.no_grad():
            self.w_rec.fill_diagonal_(0.0)

    def forward(self, x):
        """x: (B, T, n_in) spike-rate-coded input. Returns class logits (B, n_out)."""
        B, T, _ = x.shape
        dev = self.w_rec.device
        N = self.n_hidden

        v = torch.zeros(B, N, device=dev)        # membrane potential
        isyn = torch.zeros(B, N, device=dev)     # synaptic current trace
        # spike history ring: hist[k] = spikes from k steps ago (k>=1)
        hist = torch.zeros(self.max_delay + 1, B, N, device=dev)
        out_acc = torch.zeros(B, self.w_out.shape[0], device=dev)
        spike_count = torch.zeros((), device=dev)

        # precompute masked recurrent matrices per distinct delay
        zero = torch.zeros_like(self.w_rec)
        groups = [(d, torch.where(self.tau == d, self.w_rec, zero)) for d in self._delay_groups]

        for t in range(T):
            # delayed recurrent current: sum_j W_ij s_j[t - tau_ij]
            rec = torch.zeros(B, N, device=dev)
            for d, wmask in groups:
                rec = rec + hist[d] @ wmask.t()
            inp = x[:, t] @ self.w_in.t()
            isyn = self.syn_decay * isyn + rec + inp
            v = self.v_decay * v + isyn
            spk = spike_fn(v - self.thresh)        # binary spike (surrogate grad)
            v = v - spk * self.thresh              # SOFT reset (subtract threshold)
            spike_count = spike_count + spk.sum()
            # push newest spikes, evict oldest
            hist = torch.roll(hist, shifts=1, dims=0)
            hist[1] = spk
            out_acc = out_acc + spk @ self.w_out.t()
        logits = out_acc / T
        self.last_spike_rate = (spike_count / (B * T * N)).item()
        return logits

    @torch.no_grad()
    def conduction_cost(self, geom_tau):
        """C = sum |W_rec| * tau (geometric), and weighted-mean tau_bar = C/sum|W|."""
        W = self.w_rec.detach().abs()
        off = ~torch.eye(self.n_hidden, dtype=torch.bool, device=W.device)
        Wm = W[off]
        tm = geom_tau.float().to(W.device)[off]
        wsum = Wm.sum().item() + 1e-9
        C = (Wm * tm).sum().item()
        return C, C / wsum


# Task: delayed match-to-sample with spike-rate-coded inputs.
#   cue window (class-specific rate pattern) -> blank delay -> response window
def make_batch(B, n_classes, n_in, T_cue, T_delay, T_resp, rate_hi, rate_lo, seed):
    g = torch.Generator().manual_seed(seed)
    T = T_cue + T_delay + T_resp
    labels = torch.randint(0, n_classes, (B,), generator=g)
    # each class drives a distinct subset of input channels at high rate
    chans_per = n_in // n_classes
    rates = torch.full((B, n_in), rate_lo)
    for c in range(n_classes):
        mask = labels == c
        lo, hi = c * chans_per, (c + 1) * chans_per
        rates[mask, lo:hi] = rate_hi
    x = torch.zeros(B, T, n_in)
    # spikes only during the cue window (Poisson via Bernoulli per step)
    cue = (torch.rand(B, T_cue, n_in, generator=g) < rates[:, None, :]).float()
    x[:, :T_cue] = cue
    return x, labels


# Train one condition (distance or shuffled) to the task; return cost + accuracy.
def train_condition(geom_tau_geometric, tau_used, max_delay, n_in, n_hidden, n_out,
                    task_cfg, steps, batch, lr, device, seed, log=False):
    torch.manual_seed(seed)
    net = DelayLIF(n_in, n_hidden, n_out, tau_used, max_delay, seed=seed).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    for it in range(steps):
        x, y = make_batch(batch, task_cfg["n_classes"], n_in,
                          task_cfg["T_cue"], task_cfg["T_delay"], task_cfg["T_resp"],
                          task_cfg["rate_hi"], task_cfg["rate_lo"], seed=seed * 100000 + it)
        x, y = x.to(device), y.to(device)
        logits = net(x)
        # light L1 on recurrent weights: a sparsity pressure that does not know geometry, so any
        # distance<shuffled gap comes from the travel-time principle, not the penalty
        l1 = task_cfg["l1"] * net.w_rec.abs().mean()
        loss = lossf(logits, y) + l1
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if log and (it % max(1, steps // 8) == 0 or it == steps - 1):
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"      it {it:4d}: loss={loss.item():.3f} acc={acc:.3f} "
                  f"spk_rate={net.last_spike_rate:.3f}", flush=True)

    # eval accuracy on fresh batches
    net.eval()
    accs = []
    with torch.no_grad():
        for e in range(8):
            x, y = make_batch(512, task_cfg["n_classes"], n_in,
                              task_cfg["T_cue"], task_cfg["T_delay"], task_cfg["T_resp"],
                              task_cfg["rate_hi"], task_cfg["rate_lo"], seed=987654 + e)
            x, y = x.to(device), y.to(device)
            accs.append((net(x).argmax(1) == y).float().mean().item())
    acc = float(np.mean(accs))
    C, tbar = net.conduction_cost(geom_tau_geometric)
    return dict(acc=acc, cost=C, tbar=tbar, spk_rate=net.last_spike_rate)


def sign_test_p(diffs):
    diffs = [x for x in diffs if x != 0.0]
    n = len(diffs)
    if n == 0:
        return 1.0
    n_neg = sum(1 for x in diffs if x < 0)
    k = min(n_neg, n - n_neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=60)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l1", type=float, default=6e-3)
    ap.add_argument("--velocity", type=float, default=0.08)
    ap.add_argument("--max_delay", type=int, default=30)
    ap.add_argument("--jitter", type=float, default=0.35)
    ap.add_argument("--velocities", type=float, nargs="+",
                    default=[0.05, 0.08, 0.16, 0.4])
    ap.add_argument("--dose", action="store_true", help="run velocity dose-response sweep")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.hidden, a.steps, a.seeds, a.batch = 30, 40, 2, 64

    dev = a.device
    if dev == "mps" and not torch.backends.mps.is_available():
        dev = "cpu"

    n_classes = 6
    n_in = 18
    n_out = n_classes
    # hard regime: many classes, low cue rate, long blank delay so the readout depends on
    # recurrent memory; stronger L1 makes weight scarce so it must be allocated where it helps
    task_cfg = dict(n_classes=n_classes, T_cue=6, T_delay=20, T_resp=6,
                    rate_hi=0.35, rate_lo=0.05, l1=a.l1)

    # geometry built per seed so distance/shuffled share the same geometry within a seed
    def build_taus(velocity, seed):
        # jitter widens the distance spread so delays span a real range (not a near-degenerate
        # grid), which is what makes the rearrangement measurable. Geometry is fixed, not learned.
        gen = torch.Generator().manual_seed(7000 + seed)
        geom = NeuronGeometry(a.hidden, dim=a.dim, learnable=False,
                              jitter=a.jitter, generator=gen)
        dist = geom.distance_matrix().detach()
        tau = integer_delays(dist, velocity, a.max_delay)
        tau_shuf = shuffle_delays(tau, seed)
        return tau, tau_shuf

    results = {"params": vars(a), "task_cfg": task_cfg,
               "n_in": n_in, "n_out": n_out}

    # main economy test at base velocity
    print("=" * 72)
    print(f"SPIKING LIF economy test  hidden={a.hidden} steps={a.steps} "
          f"seeds={a.seeds} v={a.velocity} max_delay={a.max_delay} dev={dev}")
    print("=" * 72)

    rows = {"distance": [], "shuffled": []}
    for s in range(a.seeds):
        tau_geo, tau_shuf = build_taus(a.velocity, s)
        # distance: train on true geometric tau; cost measured vs the same tau_geo
        rd = train_condition(tau_geo, tau_geo, a.max_delay, n_in, a.hidden, n_out,
                             task_cfg, a.steps, a.batch, a.lr, dev, seed=s, log=a.smoke or s == 0)
        # shuffled: train on shuffled tau; cost still measured vs tau_geo
        rs = train_condition(tau_geo, tau_shuf, a.max_delay, n_in, a.hidden, n_out,
                             task_cfg, a.steps, a.batch, a.lr, dev, seed=s, log=False)
        rows["distance"].append(rd)
        rows["shuffled"].append(rs)
        print(f"  seed {s}: distance acc={rd['acc']:.3f} tbar={rd['tbar']:.3f} "
              f"C={rd['cost']:.1f} spk={rd['spk_rate']:.3f} | "
              f"shuffled acc={rs['acc']:.3f} tbar={rs['tbar']:.3f} C={rs['cost']:.1f} "
              f"spk={rs['spk_rate']:.3f}", flush=True)

    d_tbar = np.array([r["tbar"] for r in rows["distance"]])
    s_tbar = np.array([r["tbar"] for r in rows["shuffled"]])
    d_C = np.array([r["cost"] for r in rows["distance"]])
    s_C = np.array([r["cost"] for r in rows["shuffled"]])
    d_acc = np.array([r["acc"] for r in rows["distance"]])
    s_acc = np.array([r["acc"] for r in rows["shuffled"]])

    diffs_tbar = (d_tbar - s_tbar).tolist()
    diffs_C = (d_C - s_C).tolist()
    wins = sum(1 for x in diffs_tbar if x < 0)
    acc_gap = abs(float(d_acc.mean() - s_acc.mean()))

    print("-" * 72)
    print(f"weighted-mean tau:  distance {d_tbar.mean():.3f} ± {d_tbar.std():.3f} | "
          f"shuffled {s_tbar.mean():.3f} ± {s_tbar.std():.3f}")
    print(f"raw cost C:         distance {d_C.mean():.1f} ± {d_C.std():.1f} | "
          f"shuffled {s_C.mean():.1f} ± {s_C.std():.1f}")
    print(f"accuracy:           distance {d_acc.mean():.3f} ± {d_acc.std():.3f} | "
          f"shuffled {s_acc.mean():.3f} ± {s_acc.std():.3f}  (gap {acc_gap:.3f})")
    print(f"paired tbar (distance-shuffled, neg=win): {['%+.3f'%x for x in diffs_tbar]}")
    print(f"distance wins {wins}/{a.seeds}  sign_p={sign_test_p(diffs_tbar):.4f}")
    saving_pct = 100.0 * (s_tbar.mean() - d_tbar.mean()) / (s_tbar.mean() + 1e-9)
    print(f"economy saving (tbar): {saving_pct:.1f}%   "
          f"ECONOMY {'REPRODUCES' if d_tbar.mean() < s_tbar.mean() else 'DOES NOT reproduce'}")
    print("=" * 72)

    results["economy"] = dict(
        velocity=a.velocity,
        distance=dict(tbar=d_tbar.tolist(), cost=d_C.tolist(), acc=d_acc.tolist(),
                      spk=[r["spk_rate"] for r in rows["distance"]]),
        shuffled=dict(tbar=s_tbar.tolist(), cost=s_C.tolist(), acc=s_acc.tolist(),
                      spk=[r["spk_rate"] for r in rows["shuffled"]]),
        diffs_tbar=diffs_tbar, diffs_cost=diffs_C,
        wins=wins, n=a.seeds, sign_p=sign_test_p(diffs_tbar),
        acc_gap=acc_gap, saving_pct=saving_pct,
        reproduces=bool(d_tbar.mean() < s_tbar.mean()),
        d_tbar_mean=float(d_tbar.mean()), s_tbar_mean=float(s_tbar.mean()),
        d_acc_mean=float(d_acc.mean()), s_acc_mean=float(s_acc.mean()),
    )

    # dose-response: saving vs velocity
    if a.dose and not a.smoke:
        print("\nDOSE-RESPONSE: does the saving grow as velocity drops (delays longer)?")
        dose = []
        for v in a.velocities:
            dd, ss = [], []
            dac, sac = [], []
            for s in range(a.seeds):
                tau_geo, tau_shuf = build_taus(v, s)
                rd = train_condition(tau_geo, tau_geo, a.max_delay, n_in, a.hidden, n_out,
                                     task_cfg, a.steps, a.batch, a.lr, dev, seed=s)
                rs = train_condition(tau_geo, tau_shuf, a.max_delay, n_in, a.hidden, n_out,
                                     task_cfg, a.steps, a.batch, a.lr, dev, seed=s)
                dd.append(rd["tbar"]); ss.append(rs["tbar"])
                dac.append(rd["acc"]); sac.append(rs["acc"])
            dd, ss = np.array(dd), np.array(ss)
            sav = float(ss.mean() - dd.mean())
            sav_pct = 100.0 * sav / (ss.mean() + 1e-9)
            dose.append(dict(velocity=v, d_tbar=float(dd.mean()), s_tbar=float(ss.mean()),
                             saving=sav, saving_pct=sav_pct,
                             d_acc=float(np.mean(dac)), s_acc=float(np.mean(sac))))
            print(f"  v={v:5.3f}: distance tbar={dd.mean():.3f} shuffled tbar={ss.mean():.3f} "
                  f"saving={sav:+.3f} ({sav_pct:+.1f}%) "
                  f"acc d={np.mean(dac):.3f} s={np.mean(sac):.3f}", flush=True)
        results["dose"] = dose
        savs = [d["saving"] for d in dose]
        # monotone? saving should INCREASE as velocity DECREASES (list is v ascending)
        mono = all(savs[i] >= savs[i + 1] - 1e-6 for i in range(len(savs) - 1))
        print(f"  saving (low->high v): {['%+.3f'%x for x in savs]}  "
              f"monotone-decreasing-in-v: {mono}")
        results["dose_monotone"] = bool(mono)

    outdir = ROOT / "results" / "law"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "law_spiking.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
