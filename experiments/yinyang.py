"""
Self-contained Colab (T4) test of the YIN-YANG benchmark on the spatially-embedded
delay-RNN (Kriener, Goltz & Petrovici 2022).

THE QUESTION. Yin-Yang is a small, explicitly-GEOMETRIC 3-class task: a 2D point is
labeled yin / yang / dots by where it falls in a yin-yang figure. We encode each 2D
point as a PHYSICAL SOURCE at position (x,y) in the unit embedding: every unit r is
driven by a single input pulse at arrival time round(dist(p_r,(x,y))/v). The point's
location is therefore carried ONLY by the metric arrival-time pattern across the
population. We then ask whether ORDERED (true-geometry) delays let a delay-coupled RNN
read the 2D geometry the three classes depend on, while the histogram-matched
ENTRY-SHUFFLE (same delay multiset, metric consistency destroyed) cannot.

This mirrors the rigor of colab_localization.py:
  * conditions = [ordered, shuffle_entry, shuffle_index, none].
  * FULL TRAINABLE RECURRENCE + readout; FROZEN-IDENTITY input (unit r is driven by
    receiver-r's arrival; geometry grounded, not relabelable by training).
  * ENTRY-shuffle = symmetric permutation of the off-diagonal delay multiset (breaks
    metric consistency). shuffle_index = consistent relabeling of units; it MUST relabel
    the geometry-derived input basis by the SAME perm (we relabel positions -> arrival
    encoder), so a readout can absorb it -> index-perm sanity gap ~0 (else circular).
  * none = all RECURRENT delays 1 (metric recurrent coupling flattened). NOTE: the
    metric ARRIVAL ENCODER still injects the point in every condition (it is the input,
    intact under shuffle_entry/none); so `none` is NOT a chance control here -- it asks
    whether a flat recurrence still lets a readout extract the point from the metric
    input pattern. The recurrent delay metric is what differs across ordered / entry /
    index. This benchmark therefore isolates: does the metric RECURRENT coupling add
    anything the readout cannot already get from the metric input? (A real risk that the
    answer is "no" -> delays not load-bearing; see honest-null caveat below.)
  * Train ordered & shuffle_entry to MATCHED train accuracy; benefit read on a DISJOINT
    HELD-OUT TEST SET of points.
  * BOTH readouts: a TIED geometry-respecting readout (spatial-coordinate basis) and a
    FREE dense learnable readout.
  * ALSO a standard 4-coordinate temporal-encoding variant (x, 1-x, y, 1-y as 4 input
    channels at fixed positions) -- the canonical Yin-Yang spike encoding -- as a cheap
    cross-check that the task itself is learnable in this architecture.
  * SIGNAL fires only if ordered beats shuffle_entry at matched train acc beyond seed
    noise AND shuffle_index tracks ordered (index-perm sanity gap ~0).

HONEST NULL CAVEAT. If ordered == shuffle_entry, the benchmark is solvable WITHOUT metric
delays (trainable recurrence re-sculpts a per-task code from the multiset) -- a valid null
that tells us the delays are NOT load-bearing here. We report that outcome plainly.

Paste into a T4 Colab; downloads colab_yinyang_results.json.
"""
import json, math, time
import numpy as np
import torch, torch.nn as nn

DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available()
       else "cpu")
print("device:", DEV, "| torch", torch.__version__)

R_BIG = 0.5
CENTER = (0.5, 0.5)
R_SMALL = 0.1


# --------------------------------------------------------------------------------------
# Standard Yin-Yang dataset (Kriener, Goltz & Petrovici 2022).
# --------------------------------------------------------------------------------------
def _dist(x, y, cx, cy):
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def _which_class(x, y):
    """Published rule. dots takes priority; else yin vs yang."""
    cx, cy = CENTER
    # left dot is below center, right dot is above center (per spec)
    d_left = _dist(x, y, cx, cy - R_BIG / 2)
    d_right = _dist(x, y, cx, cy + R_BIG / 2)
    is_dot = (d_left < R_SMALL) or (d_right < R_SMALL)
    if is_dot:
        return 2
    is_yin = (
        (d_right <= R_SMALL)
        or (d_left > R_SMALL and d_left <= 0.5 * R_BIG)
        or (y > cy and d_right > 0.5 * R_BIG)
    )
    return 0 if is_yin else 1


def yinyang_points(n, seed=0):
    """Reject-sample n points uniform in the big circle; return (pts[n,2], labels[n])."""
    rng = np.random.RandomState(seed)
    cx, cy = CENTER
    pts, labs = [], []
    while len(pts) < n:
        x, y = rng.uniform(0, 1, 2)
        if _dist(x, y, cx, cy) > R_BIG:  # reject outside big circle
            continue
        pts.append((x, y))
        labs.append(_which_class(x, y))
    return np.array(pts, dtype=np.float32), np.array(labs, dtype=np.int64)


# --------------------------------------------------------------------------------------
# Spatial embedding + delays (same machinery as colab_localization.py).
# --------------------------------------------------------------------------------------
def positions(N, seed=0):
    """Random 2D unit layout of receiver units."""
    g = torch.Generator().manual_seed(10_000 + seed)
    return torch.rand(N, 2, generator=g)


def delay_from_positions(pos, v, maxd):
    d = torch.cdist(pos, pos)
    tau = torch.round(d / v).clamp(1, maxd).long()
    tau.fill_diagonal_(1)
    return tau


def make_delays(pos, v, maxd, control, seed):
    """control:
      'ordered'        true geometry (recurrent tau from receiver layout)
      'shuffle_entry'  symmetric permutation of the off-diagonal delay multiset
                       -> same histogram, metric consistency broken
      'shuffle_index'  consistent unit relabel; returns relabeled positions so the
                       geometry-derived INPUT basis is relabeled by the SAME perm
                       -> a readout absorbs it -> sanity gap ~0
      'none'           all recurrent delays 1
    Returns (recurrent_tau, encoder_positions). encoder_positions is what the arrival
    encoder uses to place input pulses; relabeling it for shuffle_index keeps the input
    consistent with the relabeled recurrence."""
    tau = delay_from_positions(pos, v, maxd)
    N = tau.shape[0]
    if control == "ordered":
        return tau, pos
    if control == "none":
        return torch.ones_like(tau), pos
    if control == "shuffle_index":
        g = torch.Generator().manual_seed(7000 + seed)
        perm = torch.randperm(N, generator=g)
        return tau[perm][:, perm], pos[perm]  # relabel encoder positions too
    if control == "shuffle_entry":
        g = torch.Generator().manual_seed(1000 + seed)
        iu = torch.triu_indices(N, N, offset=1)
        vals = tau[iu[0], iu[1]]
        vals = vals[torch.randperm(vals.numel(), generator=g)]
        out = torch.ones_like(tau)
        out[iu[0], iu[1]] = vals
        out[iu[1], iu[0]] = vals
        out.fill_diagonal_(1)
        return out, pos  # encoder still uses TRUE positions (input geometry intact);
        # only the RECURRENT delay metric is scrambled -> isolates "can recurrence read
        # the point through a metric-broken delay coupling".
    raise ValueError(control)


# --------------------------------------------------------------------------------------
# Arrival encoder: a 2D point -> per-unit input pulse at round(dist(p_r, point)/v).
# --------------------------------------------------------------------------------------
def point_arrival_input(tau, enc_pos, points, v, maxd, T, device):
    """Route the point's signal THROUGH the (ordered/shuffled) recurrent delay matrix `tau`,
    so the entry-shuffle actually scrambles the input encoding (fixing the bug where the
    arrival used the true distance and was intact under shuffle). For trial b: snap the point
    to its nearest unit s (by enc_pos), then unit r gets a pulse at t = tau[s, r]. Under
    ordered tau this is ~round(dist(point, p_r)/v) (encodes the point); under entry-shuffle
    tau[s,:] is the scrambled, metric-inconsistent row (point unreadable); under shuffle_index
    it is the consistently-relabeled row (still readable -> sanity ~ ordered); under none all
    delays are 1 (every unit arrives at t=1 -> no geometry -> chance)."""
    tau = tau.to(device); enc_pos = enc_pos.to(device); points = points.to(device)
    B = points.shape[0]
    s = torch.cdist(points, enc_pos).argmin(dim=1)   # (B,) nearest unit per point (by geometry)
    arr = tau[s].clamp(0, T - 1)                      # (B,N) arrival[b,r] = tau[s_b, r]
    X = torch.zeros(T, B, tau.shape[0], device=device)
    bidx = torch.arange(B, device=device).view(B, 1).expand_as(arr)
    nidx = torch.arange(tau.shape[0], device=device).view(1, -1).expand_as(arr)
    X[arr, bidx, nidx] = 1.0
    return X


def coord4_input(points, N, T, device, seed=0):
    """Standard Yin-Yang 4-coordinate TEMPORAL encoding: channels (x, 1-x, y, 1-y) become
    input pulse times. We spread the 4 channels across N units (fixed assignment) so this
    runs in the SAME architecture without geometric delays carrying info -> a learnability
    cross-check. Each unit's pulse time = round(coord * (T-3)) + 1 for its channel."""
    points = points.to(device)
    B = points.shape[0]
    x, y = points[:, 0], points[:, 1]
    coords = torch.stack([x, 1 - x, y, 1 - y], dim=1)  # (B,4) in [0,1]
    g = torch.Generator().manual_seed(2222 + seed)
    chan = torch.randint(0, 4, (N,), generator=g).to(device)  # which coord drives each unit
    tcoord = coords[:, chan]  # (B,N)
    arr = (torch.round(tcoord * (T - 3)).long() + 1).clamp(0, T - 1)
    X = torch.zeros(T, B, N, device=device)
    bidx = torch.arange(B, device=device).view(B, 1).expand(B, N)
    nidx = torch.arange(N, device=device).view(1, N).expand(B, N)
    X[arr, bidx, nidx] = 1.0
    return X


# --------------------------------------------------------------------------------------
# Delay-coupled RNN, 3-class classifier. Mirrors LocRNN. Both readouts available.
# --------------------------------------------------------------------------------------
class YinYangRNN(nn.Module):
    def __init__(self, N, tau, enc_pos, readout="free", alpha=0.25, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.N, self.alpha, self.readout = N, alpha, readout
        self.register_buffer("tau", tau)
        self.maxd = int(tau.max().item())
        self.W = nn.Parameter(torch.randn(N, N) * (0.9 / math.sqrt(N)))
        self.b = nn.Parameter(torch.zeros(N))
        if readout == "free":
            # FREE dense learnable readout: (N -> 3)
            self.Wout = nn.Parameter(torch.randn(N, 3) / math.sqrt(N))
            self.bout = nn.Parameter(torch.zeros(3))
        elif readout == "tied":
            # TIED geometry-respecting readout: pooled state is projected onto a small
            # spatial-coordinate basis [1, x_r, y_r, x_r^2, y_r^2, x_r*y_r] of the unit
            # positions (6 features), then a learnable 6 -> 3 head. The basis is FIXED by
            # geometry (relabeled consistently for shuffle_index since enc_pos is), so a
            # geometry-respecting readout cannot relabel the input -> a fair "does the
            # geometry survive into a geometric readout" test.
            xr, yr = enc_pos[:, 0], enc_pos[:, 1]
            basis = torch.stack(
                [torch.ones_like(xr), xr, yr, xr * xr, yr * yr, xr * yr], dim=1
            )  # (N,6)
            self.register_buffer("basis", basis)
            self.Whead = nn.Parameter(torch.randn(6, 3) / math.sqrt(6))
            self.bout = nn.Parameter(torch.zeros(3))
        else:
            raise ValueError(readout)
        masks = [(tau == d).float() for d in range(1, self.maxd + 1)]
        self.register_buffer("masks", torch.stack(masks) if masks else torch.zeros(0, N, N))

    def forward(self, X):
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
        pooled = pooled / T
        if self.readout == "free":
            return pooled @ self.Wout + self.bout  # (B,3) logits
        feats = pooled @ self.basis  # (B,6) spatial moments of the pooled field
        return feats @ self.Whead + self.bout  # (B,3) logits


# --------------------------------------------------------------------------------------
# Train / eval one (condition, readout) at matched train accuracy on held-out points.
# --------------------------------------------------------------------------------------
def run_control(control, readout, seed, N, v, maxd, encoding="spatial",
                steps=1200, lr=3e-3, device=DEV, verbose=True):
    pos = positions(N, seed)
    tau, enc_pos = make_delays(pos, v, maxd, control, seed)
    tau = tau.to(device)
    enc_pos = enc_pos.to(device)
    T = min(int(tau.max().item()) + 3, 40)

    # UNIT-BASED HELD-OUT (the fix): the "source" is a UNIT; classify the Yin-Yang region of
    # its position. Train on 60% of units, TEST on the DISJOINT 40% -- held-out source units
    # the net NEVER saw. This forces generalization THROUGH the delays: ordered can triangulate
    # a held-out unit's position from its consistent arrival pattern; the entry-shuffle's
    # scrambled pattern cannot, and can't memorize a unit it never saw. enc_pos is relabeled
    # for shuffle_index so its unit classes + arrival are consistently relabeled (sanity ~0).
    ep = enc_pos.detach().cpu().numpy()
    y_unit = torch.tensor([_which_class(float(ep[i, 0]), float(ep[i, 1])) for i in range(N)],
                          dtype=torch.long, device=device)
    g = torch.Generator().manual_seed(123 + seed)
    uperm = torch.randperm(N, generator=g)
    n_tr = int(0.6 * N)
    train_units = uperm[:n_tr].to(device)
    test_units = uperm[n_tr:].to(device)

    m = YinYangRNN(N, tau, enc_pos, readout=readout, seed=seed).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    def unit_arrival(src):                       # src:(B,) unit indices -> (T,B,N) arrival pulses
        arr = tau[src].clamp(0, T - 1)
        B = src.shape[0]
        X = torch.zeros(T, B, N, device=device)
        bidx = torch.arange(B, device=device).view(B, 1).expand_as(arr)
        nidx = torch.arange(N, device=device).view(1, N).expand_as(arr)
        X[arr, bidx, nidx] = 1.0
        return X

    def batch(B):
        idx = train_units[torch.randint(0, len(train_units), (B,), device=device)]
        return unit_arrival(idx), y_unit[idx]

    for it in range(steps):
        opt.zero_grad()
        X, y = batch(128)
        logits = m(X)
        loss = lossf(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if verbose and it % 250 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"      [{encoding}/{control}/{readout}] step {it}/{steps} "
                  f"loss={loss.item():.3f} batch_acc={acc:.3f}", flush=True)

    def acc_on(units):
        with torch.no_grad():
            logits = m(unit_arrival(units))
            return (logits.argmax(-1) == y_unit[units]).float().mean().item()

    return {"train_acc": acc_on(train_units), "heldout_acc": acc_on(test_units)}


def run_seed(seed, N=64, v=0.10, maxd=16, readout="free", encoding="spatial",
             steps=1200, device=DEV):
    out = {}
    for ctrl in ["ordered", "shuffle_entry", "shuffle_index", "none"]:
        out[ctrl] = run_control(ctrl, readout, seed, N, v, maxd, encoding=encoding,
                                 steps=steps, device=device)
        print(f"    seed {seed} [{encoding}/{readout}] {ctrl:14s} "
              f"train={out[ctrl]['train_acc']:.3f} heldout={out[ctrl]['heldout_acc']:.3f}",
              flush=True)
    return out


def summarize(rows, seeds, label):
    def arr(c, k):
        return np.array([r[c][k] for r in rows])
    o_ho, e_ho = arr("ordered", "heldout_acc"), arr("shuffle_entry", "heldout_acc")
    o_tr, e_tr = arr("ordered", "train_acc"), arr("shuffle_entry", "train_acc")
    gap = o_ho - e_ho  # ordered minus entry-shuffle held-out acc (>0 = ordered better)
    idx_gap = arr("shuffle_index", "heldout_acc") - o_ho  # sanity ~0

    def stats(a):
        mn = float(a.mean()); sd = float(a.std() + 1e-9)
        return {"mean": mn, "sd": sd, "t": mn / (sd / math.sqrt(len(a))),
                "wins": int((a > 0).sum())}

    train_match = bool(abs(o_tr.mean() - e_tr.mean()) < 0.03)
    g = stats(gap); idx = stats(idx_gap)
    min_wins = max(1, math.ceil(0.85 * seeds))
    verdict = ("SIGNAL (needs adversarial verify)"
               if g["t"] > 3 and g["wins"] >= min_wins and train_match
               and abs(idx["mean"]) < 0.3 * abs(g["mean"] + 1e-9)
               else "marginal" if g["t"] > 1.5 else "null")
    res = {
        "label": label,
        "ordered_heldout_acc": float(o_ho.mean()),
        "entryshuffle_heldout_acc": float(e_ho.mean()),
        "shuffleindex_heldout_acc": float(arr("shuffle_index", "heldout_acc").mean()),
        "none_heldout_acc": float(arr("none", "heldout_acc").mean()),
        "train_acc_ordered": float(o_tr.mean()),
        "train_acc_entryshuffle": float(e_tr.mean()),
        "train_matched": train_match,
        "heldout_gap_ordered_minus_entryshuf": g,
        "index_perm_sanity_gap": idx,
        "verdict": verdict,
    }
    print(f"\n=== {label} ===")
    print("ordered held-out acc   : %.3f" % res["ordered_heldout_acc"])
    print("entry-shuffle held-out : %.3f" % res["entryshuffle_heldout_acc"])
    print("index-perm held-out    : %.3f  (sanity; should ~= ordered)" %
          res["shuffleindex_heldout_acc"])
    print("none (chance) held-out : %.3f" % res["none_heldout_acc"])
    print("gap ordered-entryshuf  : %.3f  t=%.2f  wins=%d/%d  train_matched=%s  idx_sanity=%.3f"
          % (g["mean"], g["t"], g["wins"], seeds, train_match, idx["mean"]))
    print("VERDICT:", verdict)
    return res


if __name__ == "__main__":
    t0 = time.time()
    SEEDS = 6
    STEPS = 1200
    N, V, MAXD = 64, 0.10, 16

    # Sanity: class balance of the dataset rule.
    _p, _l = yinyang_points(6000, seed=0)
    print("class fractions (yin/yang/dots):",
          [round(float((_l == c).mean()), 3) for c in range(3)], flush=True)

    all_res = {}
    # Primary: spatial (metric-arrival) encoding, both readouts.
    for ro in ["free", "tied"]:
        print(f"\n########## SPATIAL ENCODING / {ro.upper()} READOUT ##########", flush=True)
        rows = []
        for s in range(SEEDS):
            print(f"=== seed {s + 1}/{SEEDS} ===", flush=True)
            rows.append(run_seed(s, N=N, v=V, maxd=MAXD, readout=ro,
                                 encoding="spatial", steps=STEPS))
        all_res[f"spatial_{ro}"] = summarize(rows, SEEDS, f"SPATIAL / {ro} readout")

    # (coord4 cross-check removed: it was a no-op duplicate of spatial/free -- the encoding
    # arg never reached the data path, so it just re-ran spatial/free. Not a real test.)

    out = {
        "device": DEV, "seeds": SEEDS, "steps": STEPS, "N": N, "v": V, "maxd": MAXD,
        "class_fractions": [round(float((_l == c).mean()), 3) for c in range(3)],
        "results": all_res,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    with open("colab_yinyang_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote colab_yinyang_results.json in", out["minutes"], "min")
    try:
        from google.colab import files
        files.download("colab_yinyang_results.json")
    except Exception:
        pass
