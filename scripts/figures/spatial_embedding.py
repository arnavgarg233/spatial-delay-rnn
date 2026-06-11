"""Base spatial diagram for Fig 1 panel a: recurrent units at fixed 3D positions,
distance-decaying coupling, a highlighted source, and a dashed d_ij line.
Renders figures/spatial_embedding.{png,svg}.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
BLUE, SRC, INK, GREY = "#3b7fd1", "#e8853a", "#222222", "#9aa0a6"

rng = np.random.default_rng(7)
N = 14
P = rng.uniform(0, 1, (N, 3))                      # 3D positions in the unit cube
# oblique (cabinet) projection -> 2D, with depth = z for size/shading cues
def proj(p): return p[:, 0] - 0.42 * p[:, 2], p[:, 1] - 0.34 * p[:, 2]
X, Y = proj(P)
depth = P[:, 2]
order = np.argsort(depth)                          # back-to-front
size = 320 + 520 * (depth - depth.min()) / (np.ptp(depth) + 1e-9)   # nearer = bigger
src = int(np.argmax(P[:, 0] + 0.3))                # a fairly front/right node = source
far = int(np.argmin(P[:, 0] - 0.2 * P[:, 1]))      # a distant node for the d_ij line

fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=300)
ax.set_xlim(-0.55, 1.15); ax.set_ylim(-0.5, 1.15); ax.axis("off"); ax.set_aspect("equal")

# faint distance-decaying coupling edges (closer pairs -> stronger/darker)
D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
for i in range(N):
    for j in range(i + 1, N):
        if D[i, j] < 0.55:
            a = 0.30 * (1 - D[i, j] / 0.55)
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color=GREY, lw=0.8, alpha=a, zorder=1)

# nodes, painted back-to-front with depth shading
for k in order:
    is_src = (k == src)
    ax.scatter(X[k], Y[k], s=size[k], zorder=3 + depth[k],
               facecolor=(SRC if is_src else BLUE),
               edgecolor=INK if is_src else "#2b5c96", linewidths=1.3 if is_src else 0.9,
               alpha=0.55 + 0.45 * (depth[k] - depth.min()) / (np.ptp(depth) + 1e-9))
ax.scatter(X[src], Y[src], s=size[src] * 1.05, facecolor="none", edgecolor=SRC, lw=2.0, zorder=40)
ax.text(X[src], Y[src] - 0.115, "source", fontsize=8.5, color=SRC, ha="center", va="top", zorder=41)

# the d_ij distance line (dashed) between source and a far node
ax.plot([X[src], X[far]], [Y[src], Y[far]], color=INK, lw=1.3, ls=(0, (4, 3)), zorder=20)
mx, my = (X[src] + X[far]) / 2, (Y[src] + Y[far]) / 2
ax.text(mx + 0.02, my + 0.04, r"$d_{ij}$", fontsize=11, color=INK, ha="center", zorder=21)

# input / output ports (small arrows in/out of the cloud)
ax.add_patch(FancyArrowPatch((-0.48, 0.62), (X.min() + 0.02, Y[np.argmin(X)]),
             arrowstyle="-|>", mutation_scale=11, color=GREY, lw=1.4, zorder=5))
ax.text(-0.5, 0.66, "input", fontsize=8.5, color=GREY, ha="left", va="bottom")
ax.add_patch(FancyArrowPatch((X.max() - 0.02, Y[np.argmax(X)]), (1.08, 0.30),
             arrowstyle="-|>", mutation_scale=11, color=GREY, lw=1.4, zorder=5))
ax.text(1.06, 0.26, "output", fontsize=8.5, color=GREY, ha="right", va="top")

ax.text(-0.5, -0.42, r"$N$ recurrent units at fixed positions $p_i\in\mathbb{R}^3$",
        fontsize=8.0, color=GREY, ha="left")

fig.tight_layout(pad=0.2)
out = ROOT / "figures"
for ext in ("png", "svg"):
    fig.savefig(out / f"spatial_embedding.{ext}", bbox_inches="tight", facecolor="white", transparent=False)
print("wrote figures/spatial_embedding.{png,svg}")
