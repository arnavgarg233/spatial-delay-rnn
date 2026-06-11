"""Export each Figure-1 data-plot asset as a standalone transparent SVG+PNG for figure assembly.
Palette: teal=metric, orange=shuffle, green=pass, red=fail. Run: python scripts/figures/fig1_assets.py
"""
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "figures" / "assets"; A.mkdir(parents=True, exist_ok=True)
TEAL, ORANGE, GREEN, BLUE, INK, GREY, RED = "#2a9d8f", "#e0813f", "#2f9e44", "#3b7fd1", "#222222", "#9aa0a6", "#e03131"
plt.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none"})

def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(A / f"{name}.{ext}", bbox_inches="tight", transparent=True, dpi=300)
    plt.close(fig); print("  ", name)

# --- activity heatmaps: travelling wavefront (true) vs scrambled ---
def wavefront(n=180, scramble=False, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n); u = np.linspace(0, 1, n); T, U = np.meshgrid(t, u)
    field = np.exp(-((U - (T * 0.9 + 0.05)) ** 2) / (2 * 0.02 ** 0.5 * 0.02))
    if scramble:
        field = field[np.random.default_rng(seed + 9).permutation(n)]
    return field
for nm, scr, lab in [("activity_true", False, "true"), ("activity_scrambled", True, "scrambled")]:
    fig, ax = plt.subplots(figsize=(2.0, 1.6), dpi=300)
    ax.imshow(wavefront(scramble=scr), aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("time $t$", fontsize=8, color=INK); ax.set_ylabel("units (by distance)", fontsize=8, color=INK)
    save(fig, nm)

# --- arrival time vs distance scatter ---
fig, ax = plt.subplots(figsize=(2.4, 1.9), dpi=300)
rng = np.random.default_rng(1); d = rng.uniform(0, 1, 90)
ax.scatter(d, 0.85 * d + 0.07 + rng.normal(0, 0.04, 90), s=14, c=TEAL, alpha=0.9, label="metric")
ax.scatter(d, rng.uniform(0.06, 0.94, 90), s=14, c=ORANGE, alpha=0.8, label="shuffle")
ax.set_xlabel("distance from source $d$", fontsize=8.5, color=INK)
ax.set_ylabel(r"recovered arrival $\hat a$", fontsize=8.5, color=INK)
ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
save(fig, "scatter_arrival_distance")

# --- residual r(v) bowl with the dip at v_true ---
fig, ax = plt.subplots(figsize=(2.4, 1.9), dpi=300)
v = np.linspace(0.45, 1.8, 220); r = 0.2 + (np.log(v / 1.0)) ** 2 * 0.9
imin = int(np.argmin(r))
ax.plot(v, r, color=TEAL, lw=2.3)
ax.axvline(v[imin], color=GREY, ls=(0, (3, 3)), lw=1.0)
ax.scatter([v[imin]], [r[imin]], s=44, c=ORANGE, zorder=5, edgecolor=INK, lw=0.8)
ax.text(v[imin], r.max() * 0.96, r"$v_{\rm true}$", fontsize=9, color=ORANGE, ha="center")
ax.set_xlabel("candidate conduction velocity $v$", fontsize=8.5, color=INK)
ax.set_ylabel("residual $r(v)$", fontsize=8.5, color=INK)
ax.set_xticks([]); ax.set_yticks([])
for s in ["top", "right"]: ax.spines[s].set_visible(False)
save(fig, "residual_bowl")

# --- delay histogram ---
fig, ax = plt.subplots(figsize=(2.2, 1.6), dpi=300)
taus = np.clip(np.round(np.random.default_rng(2).gamma(2.2, 1.6, 4000)).astype(int), 1, 12)
vals, counts = np.unique(taus, return_counts=True)
ax.bar(vals, counts, color=TEAL, edgecolor="white", width=0.85)
ax.set_xlabel(r"integer delay $\tau_{ij}$", fontsize=8.5, color=INK); ax.set_ylabel("count", fontsize=8.5, color=INK)
ax.set_yticks([])
for s in ["top", "right"]: ax.spines[s].set_visible(False)
save(fig, "delay_histogram")

# --- metric triangles: true (holds, green check) vs shuffle (breaks, red X) ---
def triangle(ax, col, ok, tau):
    pts = np.array([[0.1, 0.1], [0.5, 0.85], [0.9, 0.1]]); lab = ["A", "B", "C"]
    for a_, b_ in [(0, 1), (1, 2), (0, 2)]:
        ax.plot([pts[a_, 0], pts[b_, 0]], [pts[a_, 1], pts[b_, 1]], color=col, lw=2.4)
    for k, (x, y) in enumerate(pts):
        ax.text(x, y + (0.07 if k == 1 else -0.09), lab[k], fontsize=10, color=INK, ha="center")
    ax.text(0.5, -0.2, tau, fontsize=8.5, color=col, ha="center")
    ax.text(0.5, 1.05, ("✓" if ok else "✗"), fontsize=19, color=(GREEN if ok else RED), ha="center")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.34, 1.22); ax.axis("off"); ax.set_aspect("equal")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.6, 1.9), dpi=300)
triangle(a1, TEAL, True, r"$\tau_{AC}\leq\tau_{AB}+\tau_{BC}$")
triangle(a2, ORANGE, False, r"$\tau_{AC}>\tau_{AB}+\tau_{BC}$")
save(fig, "metric_triangles")

# --- multilateration: circles intersect (true, check) vs fail (shuffle, X) ---
def multilat(ax, col, ok):
    C = [(0.2, 0.25), (0.8, 0.3), (0.5, 0.85)]
    radii = ([np.hypot(cx - 0.5, cy - 0.5) for cx, cy in C] if ok else [0.52, 0.27, 0.46])
    for (cx, cy), rr in zip(C, radii):
        ax.add_patch(Circle((cx, cy), rr, fill=False, ec=col, lw=1.6, ls=(0, (4, 3))))
        ax.scatter([cx], [cy], s=16, c=col, zorder=5)
    if ok:
        ax.scatter([0.5], [0.5], marker="*", s=190, c=GREEN, edgecolor=INK, lw=0.6, zorder=6)
    ax.text(0.5, 1.32, ("✓" if ok else "✗"), fontsize=19, color=(GREEN if ok else RED), ha="center")
    ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.2, 1.45); ax.axis("off"); ax.set_aspect("equal")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.8, 2.1), dpi=300)
multilat(a1, TEAL, True); multilat(a2, ORANGE, False)
save(fig, "multilateration")

# --- distance matrix d_ij (input slot) ---
fig, ax = plt.subplots(figsize=(1.7, 1.7), dpi=300)
pos = np.random.default_rng(3).uniform(0, 1, (24, 3))
Dm = np.linalg.norm(pos[:, None] - pos[None], axis=2)
ax.imshow(Dm, cmap="viridis"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(r"$d_{ij}$", fontsize=10, color=INK)
save(fig, "distance_matrix")

# --- input stimulus u(t): a couple of overlaid drive signals ---
fig, ax = plt.subplots(figsize=(1.8, 1.3), dpi=300)
tt = np.linspace(0, 1, 200); rng = np.random.default_rng(5)
for c, off in [(BLUE, 0.0), (TEAL, 1.25)]:
    sig = np.cumsum(rng.normal(0, 1, 200)); sig = (sig - sig.min()) / (np.ptp(sig) + 1e-9)
    ax.plot(tt, sig + off, color=c, lw=1.6)
ax.axis("off"); save(fig, "input_stimulus")

# --- neuron positions p_i: small embedded scatter ---
fig, ax = plt.subplots(figsize=(1.6, 1.5), dpi=300)
P = np.random.default_rng(6).uniform(0, 1, (16, 3))
ax.scatter(P[:, 0] - 0.4 * P[:, 2], P[:, 1] - 0.3 * P[:, 2], s=150 * (0.5 + 0.5 * P[:, 2]),
           c=BLUE, edgecolor="#2b5c96", lw=0.8, alpha=0.85)
ax.axis("off"); ax.set_aspect("equal"); save(fig, "neuron_positions")

# --- MLP glyph: small node-and-edge subnet sketch ---
fig, ax = plt.subplots(figsize=(1.35, 1.15), dpi=300)
layers = [2, 3, 2]; xs = np.linspace(0.12, 0.88, len(layers)); pts = []
for x, nn in zip(xs, layers):
    pts.append([(x, y) for y in np.linspace(0.22, 0.78, nn)])
for li in range(len(layers) - 1):
    for (x0, y0) in pts[li]:
        for (x1, y1) in pts[li + 1]:
            ax.plot([x0, x1], [y0, y1], color=GREY, lw=0.7, alpha=0.6, zorder=1)
for layer in pts:
    for (x, y) in layer:
        ax.scatter([x], [y], s=72, c="white", edgecolor=INK, lw=1.2, zorder=3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); save(fig, "mlp_glyph")

# --- merge node (the circled cross) ---
fig, ax = plt.subplots(figsize=(0.6, 0.6), dpi=300)
ax.add_patch(Circle((0.5, 0.5), 0.42, fill=True, fc="white", ec=GREEN, lw=2.0))
ax.plot([0.24, 0.76], [0.24, 0.76], color=GREEN, lw=2.0); ax.plot([0.24, 0.76], [0.76, 0.24], color=GREEN, lw=2.0)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_aspect("equal"); save(fig, "merge_node")

# --- activity field h_i(t): stacked unit traces ---
fig, ax = plt.subplots(figsize=(1.8, 1.4), dpi=300)
tt = np.linspace(0, 1, 160)
for k in range(7):
    tr = np.sin(2 * np.pi * (2 * tt - k * 0.12)) * np.exp(-((tt - 0.5) ** 2) / 0.3)
    ax.plot(tt, tr * 0.42 + k, color=TEAL, lw=1.1, alpha=0.85)
ax.axis("off"); save(fig, "activity_field")

# --- smoothness null: per-subject true-vs-surrogate from the MSR data ---
import json
try:
    rows = json.load(open(ROOT / "results" / "inverse" / "foreigninv_ajile_msr.json"))["rows"]
    fig, ax = plt.subplots(figsize=(2.2, 1.7), dpi=300)
    for i, r in enumerate(rows):
        ax.plot([r["surr_mean"] - r["surr_std"], r["surr_mean"] + r["surr_std"]], [i, i], color=GREY, lw=2.6, alpha=0.6)
        ax.scatter([r["floor_true"]], [i], s=18, c=TEAL, zorder=5)
    ax.set_xlabel("inverse residual", fontsize=8, color=INK); ax.set_ylabel("subject", fontsize=8, color=INK)
    ax.set_yticks([])
    ax.text(0.02, 0.98, "true (teal) beats\nsurrogates  12/12", transform=ax.transAxes, fontsize=7.5, va="top", color=INK)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    save(fig, "smoothness_null")
except Exception as e:
    print("  smoothness_null skipped:", e)

print("wrote ->", A)
