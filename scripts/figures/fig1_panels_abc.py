"""Figure 1 left-column panels a/b/c, each a self-contained transparent composite PNG+SVG.
  a : units in physical space -> delay-coupled RNN -> travelling-wave activity
  b : distance d_ij -> delay tau=round(d/v) (staircase + per-edge dumbbells)
  c : forward map and velocity inverse, as a travelling-wave isochrone field
Run: PYTHONPATH=.:src python scripts/figures/fig1_panels_abc.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Polygon, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, to_hex

# cohesive light -> teal -> dark ramp (ties the distance matrix to the teal palette)
TEAL_CMAP = LinearSegmentedColormap.from_list(
    "tealmap", ["#eef7f5", "#9bd7cf", "#2a9d8f", "#1c6b61", "#0d352f"])

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

TEAL, ORANGE, GREEN, BLUE, INK, GREY, RED = \
    "#2a9d8f", "#e0813f", "#2f9e44", "#3b7fd1", "#222222", "#9aa0a6", "#e03131"
EDGE = "#2b5c96"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.titleweight": "bold", "axes.labelweight": "bold"})


def save(fig, name):
    # FIXED canvas (no tight crop) so every panel exports at the SAME width (figsize*dpi);
    # a/b/c all use width 7.6in -> identical 2280px width, heights may differ.
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", transparent=True, dpi=300)
    plt.close(fig)
    print(f"  wrote assets/{name}.png  ({fig.get_size_inches()[0]:.1f} x "
          f"{fig.get_size_inches()[1]:.1f} in)")


# ---- shared geometry: well-spread units in 3D, no 2D overlap after projection ----
def proj(P):
    return P[:, 0] - 0.42 * P[:, 2], P[:, 1] - 0.34 * P[:, 2]


def spread3d(n, rng, min2d=0.20, lo=0.04, hi=0.96, tries=40000):
    """Reject-sample 3D points until n of them are >= min2d apart IN THE PROJECTION."""
    pts = []
    XY = []
    t = 0
    while len(pts) < n and t < tries:
        p = rng.uniform(lo, hi, 3)
        x, y = proj(p[None])
        x, y = float(x[0]), float(y[0])
        if all((x - qx) ** 2 + (y - qy) ** 2 > min2d ** 2 for qx, qy in XY):
            pts.append(p); XY.append((x, y))
        t += 1
    return np.array(pts)


# glossy-sphere node palette (matches figM0 panel-a 'Spatial embedding' style)
NODE_BASE, NODE_LIGHT, NODE_EDGE = "#6aa6e0", "#c4e2fb", "#3b78c2"
SRC_BASE, SRC_LIGHT, SRC_EDGE = "#e0564b", "#f6bcb4", "#bd3526"
MESH, DIST = "#c8ced6", "#d8392b"


def spread2d(n, rng, min_d=0.17, lo=0.06, hi=0.94, tries=60000):
    pts = []
    t = 0
    while len(pts) < n and t < tries:
        p = rng.uniform(lo, hi, 2)
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > min_d ** 2 for q in pts):
            pts.append(p)
        t += 1
    return np.array(pts)


def glossy_node(ax, x, y, r, base, light, edge, z=5, lw=1.1):
    """A 3D-looking sphere: body + lit hemisphere + specular highlight."""
    ax.add_patch(Circle((x, y), r, facecolor=base, edgecolor=edge, lw=lw, zorder=z))
    ax.add_patch(Circle((x - 0.26 * r, y + 0.26 * r), 0.60 * r, facecolor=light,
                         edgecolor="none", alpha=0.85, zorder=z + 0.1))
    ax.add_patch(Circle((x - 0.32 * r, y + 0.32 * r), 0.17 * r, facecolor="white",
                         edgecolor="none", alpha=0.95, zorder=z + 0.2))


def glossy_dot(ax, x, y, s, base, light, edge, z=6):
    """Scatter-based glossy node (always round regardless of axes aspect ratio)."""
    ax.scatter([x], [y], s=s, c=base, edgecolors=edge, linewidths=1.1, zorder=z)
    ax.scatter([x], [y], s=s * 0.34, c=light, edgecolors="none", alpha=0.55, zorder=z + 0.1)


def draw_network(ax, seed=4, n=14, P=None, legend=True, scalebar=True, box=False,
                 dist_fs=12.5, src_label=True):
    """Panel-a LEFT in the figM0 'Spatial embedding' style: glossy units on a 2D sheet with a
    light recurrent-coupling mesh, a red SOURCE, and red dashed distances d_ij / d_ik.
    Reusable: pass P to draw a GIVEN geometry; legend/scalebar/box toggle the chrome so the
    same clean figure drops into a small cell (panel c).  Returns (P, src, near, far)."""
    rng = np.random.default_rng(seed)
    if P is None:
        P = spread2d(n, rng, min_d=0.185)
    n = len(P)
    X, Y = P[:, 0], P[:, 1]
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    r = 0.042
    compact = not (legend or scalebar)

    if compact:                                            # FILL the (wide) cell, matching the fields
        pos = ax.get_position(); fw, fh = ax.figure.get_size_inches()
        cell_ratio = (pos.width * fw) / (pos.height * fh)
        cx, cy = X.mean(), Y.mean()
        yspan = (Y.max() - Y.min()) + 0.34
        xspan = max(yspan * cell_ratio, (X.max() - X.min()) + 0.30)
        ax.set_xlim(cx - xspan / 2, cx + xspan / 2); ax.set_ylim(cy - yspan / 2, cy + yspan / 2)
    else:
        cell_ratio = 1.0
        ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.30, 1.12)
    ax.axis("off"); ax.set_aspect("equal")

    if box:                                                # subtle frame so the cell reads as full
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                     boxstyle="round,pad=0,rounding_size=0.025", fc="#eef3fb", ec=EDGE,
                     lw=1.5, alpha=0.9, zorder=0, mutation_aspect=1.0 / cell_ratio))

    # recurrent-coupling mesh: each unit to its 3 nearest neighbours (light grey, thin)
    seen = set()
    for i in range(n):
        for j in np.argsort(D[i])[1:4]:
            key = (min(i, int(j)), max(i, int(j)))
            if key in seen:
                continue
            seen.add(key)
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color=MESH, lw=1.0, zorder=1)

    # source = a fairly central unit; pick a near + a far partner for d_ij / d_ik
    cen = np.array([X.mean(), Y.mean()])
    src = int(np.argmin(np.linalg.norm(P - cen, axis=1) + 0.3 * (X < cen[0])))
    d_to = np.argsort(D[src])
    near = int(d_to[1]); far = int(d_to[-3])

    # red dashed Euclidean-distance lines (under nodes), labels offset off the line
    for tgt, lab, sgn in [(near, r"$d_{ij}$", +1.0), (far, r"$d_{ik}$", -1.0)]:
        dx, dy = X[tgt] - X[src], Y[tgt] - Y[src]
        L = np.hypot(dx, dy) + 1e-9
        px, py = -dy / L, dx / L
        ax.plot([X[src], X[tgt]], [Y[src], Y[tgt]], color=DIST, lw=1.6,
                ls=(0, (4, 3)), zorder=3)
        mx = 0.5 * (X[src] + X[tgt]) + sgn * 0.05 * px
        my = 0.5 * (Y[src] + Y[tgt]) + sgn * 0.05 * py
        ax.text(mx, my, lab, fontsize=dist_fs, color=DIST, ha="center", va="center",
                style="italic", fontweight="bold", zorder=4)

    # nodes
    for k in range(n):
        if k == src:
            continue
        glossy_node(ax, X[k], Y[k], r, NODE_BASE, NODE_LIGHT, NODE_EDGE, z=5)
    glossy_node(ax, X[src], Y[src], r * 1.42, SRC_BASE, SRC_LIGHT, SRC_EDGE, z=8, lw=1.3)
    if src_label:                                          # source label to the RIGHT of the node
        ax.text(X[src] + r * 1.42 + 0.02, Y[src], "source", fontsize=10.5, color=SRC_EDGE,
                ha="left", va="center", fontweight="bold", zorder=9)

    if scalebar:
        x0, x1, yb = 0.06, 0.21, -0.10
        ax.plot([x0, x1], [yb, yb], color=INK, lw=1.5, zorder=6)
        for xc in (x0, x1):
            ax.plot([xc, xc], [yb - 0.013, yb + 0.013], color=INK, lw=1.5, zorder=6)
        ax.text((x0 + x1) / 2, yb - 0.035, r"$L$  (physical extent)", fontsize=8.5, color=INK,
                ha="center", va="top", fontweight="bold")

    if legend:                                             # auto-spaced so glyphs can't touch words
        handles = [
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=NODE_BASE,
                   markeredgecolor=NODE_EDGE, markersize=9, label="recurrent unit"),
            Line2D([0], [0], color=MESH, lw=2.4, label="coupling"),
            Line2D([0], [0], color=DIST, lw=2.4, ls=(0, (4, 3)), label=r"distance $d_{ij}$"),
        ]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, -0.02), ncol=3,
                  frameon=False, prop={"weight": "bold", "size": 8.0}, handlelength=1.4,
                  columnspacing=1.4, handletextpad=0.45, borderaxespad=0.0)
    return P, src, near, far


def draw_waterfall(ax, n=9):
    """panel-a RIGHT: travelling-wave activity as clean stacked traces -- far units fire later."""
    T = np.linspace(0, 1, 300)
    d = np.linspace(0, 1, n)                                 # unit distance rank (near -> far)
    cmap = plt.cm.viridis
    peaks = []
    for k in range(n):
        arrival = 0.13 + 0.66 * d[k]
        y = np.exp(-((T - arrival) ** 2) / (2 * 0.040 ** 2))
        base = float(k)
        col = cmap(0.12 + 0.78 * d[k])
        ax.fill_between(T, base, base + 0.95 * y, color=col, alpha=0.16, zorder=2 + k)
        ax.plot(T, base + 0.95 * y, color=col, lw=1.8, zorder=3 + k)
        peaks.append((arrival, base + 0.95))
    pa = [p[0] for p in peaks]
    ax.plot(pa, [p[1] for p in peaks], ls=(0, (4, 3)), color=INK, lw=1.3, alpha=0.75, zorder=30)
    ax.annotate("wavefront", xy=(0.78, peaks[-1][1] - 0.05), xytext=(0.30, n - 0.5),
                fontsize=9.5, color=INK, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.set_xlim(0, 1); ax.set_ylim(-0.3, n + 0.2)
    ax.set_xlabel("time", fontsize=10.5)
    ax.set_ylabel("unit  (near → far)", fontsize=10.5)
    ax.set_xticks([0, 1]); ax.set_yticks([])
    ax.tick_params(labelsize=8.5, length=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def operator_arrow(ax, top, bottom, col=INK, lw=2.0, fs=10.5):
    """Thin single-headed arrow with a label above (operator) and below (relation)."""
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyArrowPatch((0.04, 0.5), (0.96, 0.5), arrowstyle="-|>",
                 mutation_scale=18, color=col, lw=lw))
    if top:
        ax.text(0.5, 0.62, top, ha="center", va="bottom", fontsize=fs, color=col,
                fontweight="bold")
    if bottom:
        ax.text(0.5, 0.38, bottom, ha="center", va="top", fontsize=fs, color=col,
                fontweight="bold")


# ---- panel a ----
def panel_a():
    # same width+height as panel_b (7.6 x 3.0) so a/b drop in as a uniform pair
    fig = plt.figure(figsize=(7.6, 3.0))
    axL = fig.add_axes([0.00, 0.02, 0.40, 0.88])
    axM = fig.add_axes([0.355, 0.42, 0.25, 0.26])
    axR = fig.add_axes([0.66, 0.13, 0.33, 0.74])
    draw_network(axL)
    operator_arrow(axM, "delay-coupled RNN", None)
    draw_waterfall(axR)
    fig.text(0.005, 0.975, "a", fontsize=16, fontweight="bold", va="top")
    fig.text(0.20, 0.985, "spatial embedding", fontsize=11, color=INK, ha="center",
             va="top", fontweight="bold")
    fig.text(0.20, 0.928, "nodes have fixed 2D positions", fontsize=8.2, color=GREY,
             ha="center", va="top")
    fig.text(0.83, 0.985, "travelling-wave activity", fontsize=11, color=INK,
             ha="center", va="top", fontweight="bold")
    save(fig, "panel_a")


# ---- panel b ----
def panel_b():
    """Distance-to-delay map: rounding turns smooth distance into integer delays (staircase vs
    the d/v line); each edge inherits one integer delay."""
    v = 0.35
    kmax = 4
    cmap = plt.cm.viridis
    fig = plt.figure(figsize=(7.6, 3.0))

    fig.text(0.005, 0.975, "b", fontsize=16, fontweight="bold", va="top")
    fig.text(0.5, 0.985, "distance-to-delay map", fontsize=11, color=INK, ha="center",
             va="top", fontweight="bold")

    # equation in a soft box (top, centred)
    axEq = fig.add_axes([0.22, 0.80, 0.56, 0.13]); axEq.axis("off")
    axEq.text(0.5, 0.5, r"$\tau_{ij}=\mathrm{round}(d_{ij}\,/\,v)$", ha="center", va="center",
              fontsize=15, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.45", fc="#eef5ff", ec=BLUE, lw=1.6))

    # ---- staircase (left): smooth distance -> integer delay ----
    ax = fig.add_axes([0.09, 0.16, 0.40, 0.52])
    dd = np.linspace(0, (kmax + 0.45) * v, 400)
    ax.plot(dd, dd / v, ls=(0, (4, 3)), color=GREY, lw=1.8, zorder=2)
    ax.text((kmax + 0.30) * v, kmax + 0.35, r"$d/v$", color=GREY, fontsize=10.5,
            fontweight="bold", ha="right")
    for k in range(kmax + 1):
        lo = max(0.0, (k - 0.5) * v); hi = (k + 0.5) * v
        ax.plot([lo, hi], [k, k], color=cmap(k / kmax), lw=5.5, zorder=4,
                solid_capstyle="round")
        if k < kmax:
            ax.plot([hi, hi], [k, k + 1], color=MESH, lw=1.4, zorder=3)
    ax.set_xlim(0, (kmax + 0.4) * v); ax.set_ylim(-0.35, kmax + 0.55)
    ax.set_xlabel(r"distance  $d_{ij}$", fontsize=11)
    ax.set_ylabel(r"delay  $\tau_{ij}$", fontsize=11)
    ax.set_xticks([0, 0.5, 1.0, 1.5]); ax.set_yticks(range(kmax + 1))
    ax.tick_params(labelsize=8.5, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- "each edge gets one integer delay": four dumbbells, longer edge = bigger delay ----
    fig.text(0.755, 0.745, "each edge gets one integer delay", fontsize=9.5,
             fontweight="bold", color=INK, ha="center", va="center")
    axR = fig.add_axes([0.54, 0.10, 0.45, 0.60]); axR.axis("off")
    axR.set_xlim(0, 1); axR.set_ylim(0, 1)
    rows_y = [0.84, 0.61, 0.38, 0.15]                        # tau = 1 (top) .. 4 (bottom)
    for row, k in enumerate(range(1, kmax + 1)):
        y = rows_y[row]
        xl = 0.06
        xr = 0.06 + k * 0.165                                # longer distance -> bigger delay
        col = cmap(k / kmax)
        axR.plot([xl, xr], [y, y], color=col, lw=7.5, solid_capstyle="round", zorder=4)
        glossy_dot(axR, xl, y, 230, NODE_BASE, NODE_LIGHT, NODE_EDGE)
        glossy_dot(axR, xr, y, 230, NODE_BASE, NODE_LIGHT, NODE_EDGE)
        axR.text(0.985, y, rf"$\tau = {k}$", fontsize=12, fontweight="bold", color=INK,
                 ha="right", va="center")
    save(fig, "panel_b")


# ---- panel c ----
def _triangle(ax, ok, bg=True, fs=1.0):
    """Delay-triangle with delay-coloured edges, chips, a verdict line and a check/cross badge.
    ok=True -> metric holds; False -> AC edge violates. bg draws the tinted panel; fs scales text."""
    col = GREEN if ok else RED
    tint = "#eaf6ee" if ok else "#fdeceb"
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    if bg:
        ax.add_patch(FancyBboxPatch((0.05, 0.05), 0.90, 0.90,
                     boxstyle="round,pad=0,rounding_size=0.04", fc=tint, ec=col, lw=1.6,
                     alpha=0.95, zorder=0))

    pts = np.array([[0.22, 0.30], [0.78, 0.30], [0.50, 0.82]])
    lab = ["A", "B", "C"]
    if ok:
        e = {"AB": 3, "BC": 4, "AC": 6}                       # 6 <= 3+4 ✓
        verdict = r"$\tau_{AC}\leq\tau_{AB}+\tau_{BC}$"; extra = "(6 ≤ 3 + 4)"
    else:
        e = {"AB": 3, "BC": 4, "AC": 9}                       # 9 > 3+4 ✗  (after shuffle)
        verdict = r"$\tau_{AC}>\tau_{AB}+\tau_{BC}$"; extra = "(9 > 3 + 4)"

    ax.add_patch(Polygon(pts, closed=True, fc=col, ec="none", alpha=0.08, zorder=1))
    dcol = lambda d: plt.cm.viridis(0.05 + 0.9 * min(d, 9) / 9)   # tie edge colour to panel b
    cen = pts.mean(0)
    for i, j, key in [(0, 1, "AB"), (1, 2, "BC"), (0, 2, "AC")]:
        viol = (key == "AC" and not ok)
        ecol = RED if viol else dcol(e[key])
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=ecol,
                lw=5.0 if viol else 4.0, ls=(0, (4, 3)) if viol else "-",
                solid_capstyle="round", zorder=2)
        mx, my = pts[[i, j]].mean(0)
        ox, oy = mx - cen[0], my - cen[1]; L = np.hypot(ox, oy) + 1e-9
        cx, cy = mx - 0.06 * ox / L, my - 0.06 * oy / L              # nudge chip INWARD off the edge
        ax.scatter([cx], [cy], s=360 * fs ** 2, c="white", edgecolors=ecol, linewidths=1.8, zorder=4)
        ax.text(cx, cy, str(e[key]), ha="center", va="center", fontsize=11.5 * fs,
                fontweight="bold", color=ecol if viol else INK, zorder=5)
    for k in range(3):
        glossy_dot(ax, pts[k, 0], pts[k, 1], 430 * fs ** 2, NODE_BASE, NODE_LIGHT, NODE_EDGE, z=6)
        ax.text(pts[k, 0], pts[k, 1], lab[k], fontsize=10.5 * fs, color="white",
                ha="center", va="center", fontweight="bold", zorder=8)

    ax.text(0.5, 0.175, verdict, ha="center", va="center", fontsize=11.5 * fs, color=col,
            fontweight="bold", zorder=5)
    ax.text(0.5, 0.095, extra, ha="center", va="center", fontsize=9.5 * fs, color=col, zorder=5)
    ax.scatter([0.85], [0.85], s=620 * fs ** 2, c=col, edgecolors="white", linewidths=1.8, zorder=9)
    ax.text(0.85, 0.85, "✓" if ok else "✗", ha="center", va="center", fontsize=15 * fs,
            color="white", fontweight="bold", zorder=10)


def _geo_panel(ax, P, D, s, near, far):
    """Geometry domain: glossy units with a light coupling mesh and two highlighted distances
    d_ij / d_ik (the forward-map input)."""
    n = len(P); X, Y = P[:, 0], P[:, 1]
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.06, 1.06); ax.axis("off"); ax.set_aspect("equal")
    ax.add_patch(FancyBboxPatch((-0.04, -0.04), 1.08, 1.08,
                 boxstyle="round,pad=0,rounding_size=0.05", fc="#eef3fb", ec=EDGE, lw=1.5,
                 alpha=0.9, zorder=0))
    seen = set()
    for i in range(n):
        for j in np.argsort(D[i])[1:4]:
            k = (min(i, int(j)), max(i, int(j)))
            if k in seen:
                continue
            seen.add(k)
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color=MESH, lw=1.0, zorder=1)
    for tgt, lab in [(near, r"$d_{ij}$"), (far, r"$d_{ik}$")]:
        ax.plot([X[s], X[tgt]], [Y[s], Y[tgt]], color=DIST, lw=1.7, ls=(0, (4, 3)), zorder=3)
        mx, my = 0.5 * (X[s] + X[tgt]), 0.5 * (Y[s] + Y[tgt])
        ax.text(mx, my, lab, fontsize=11, color=DIST, style="italic", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.08", fc="#eef3fb", ec="none"), zorder=4)
    for k in range(n):
        r = 0.058 if k == s else 0.05
        base, light, edge = (SRC_BASE, SRC_LIGHT, SRC_EDGE) if k == s else (NODE_BASE, NODE_LIGHT, NODE_EDGE)
        glossy_node(ax, X[k], Y[k], r, base, light, edge, z=5)


def _delay_matrix(ax, M, ec, aspect="equal"):
    """Integer delay matrix tau_ij as a viridis heat-map."""
    n = M.shape[0]
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=M.max(), aspect=aspect)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ec); sp.set_linewidth(1.8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] < 0.55 * M.max() else "#1b1b1b", fontweight="bold")
    return im


def _tau_d_fit(ax, D, M, v):
    """Velocity inverse: every pair's delay vs distance falls on a line of slope 1/v, so the fit
    recovers the conduction speed."""
    iu = np.triu_indices_from(D, 1)
    d, t = D[iu], M[iu]
    xs = np.array([0.0, d.max() * 1.05])
    ax.plot(xs, xs / v, color=INK, lw=2.0, ls=(0, (5, 3)), zorder=2)
    ax.scatter(d, t, s=34, color=TEAL, edgecolor="white", lw=0.7, zorder=3)
    ax.set_xlim(0, d.max() * 1.08); ax.set_ylim(-0.4, t.max() + 0.6)
    ax.set_xlabel(r"distance  $d_{ij}$", fontsize=9)
    ax.set_ylabel(r"delay  $\tau_{ij}$", fontsize=9)
    ax.tick_params(labelsize=7.5, length=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.04, 0.90, r"slope $=1/\hat v$", transform=ax.transAxes, fontsize=9.5,
            color=INK, fontweight="bold", ha="left", va="top")
    ax.text(0.97, 0.08, r"$\hat v \approx v$  $\checkmark$", transform=ax.transAxes,
            fontsize=11, color=TEAL, fontweight="bold", ha="right", va="bottom")


def _delay_field(ax, P, D, src, v, tint, ec):
    """Concentric isochrone rings from the source + units coloured by delay tau = round(d_src/v)."""
    X, Y = P[:, 0], P[:, 1]; n = len(P)
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.06, 1.06); ax.axis("off"); ax.set_aspect("equal")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                 boxstyle="round,pad=0,rounding_size=0.03", fc=tint, ec=ec, lw=1.5,
                 alpha=0.9, zorder=0, mutation_aspect=0.6))
    sx, sy = X[src], Y[src]
    kmax = max(1, int(round(D[src].max() / v)))
    for k in range(1, kmax + 1):                              # isochrone rings (delay contours)
        col = to_hex(plt.cm.viridis(0.08 + 0.85 * k / kmax))
        ax.add_patch(Circle((sx, sy), k * v, fill=False, ec=col, lw=1.4, ls=(0, (3, 2)),
                     alpha=0.55, zorder=1))
    seen = set()
    for i in range(n):
        for j in np.argsort(D[i])[1:4]:
            key = (min(i, int(j)), max(i, int(j)))
            if key in seen:
                continue
            seen.add(key)
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color="#cdd3da", lw=0.8, zorder=2)
    for kk in range(n):
        if kk == src:
            glossy_dot(ax, X[kk], Y[kk], 250, SRC_BASE, SRC_LIGHT, SRC_EDGE, z=6)
            continue
        tk = min(round(D[src, kk] / v), kmax)
        base = to_hex(plt.cm.viridis(0.08 + 0.85 * tk / kmax))
        glossy_dot(ax, X[kk], Y[kk], 178, base, "#ffffff", "#33485e", z=5)


def _cmds(Dm, k=2):
    """Classical (Torgerson) MDS: recover k-D coordinates from a distance matrix."""
    n = Dm.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J.dot(Dm ** 2).dot(J)
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:k]
    return V[:, order] * np.sqrt(np.clip(w[order], 0, None))


def _procrustes(X, Y):
    """Best similarity transform (rotation/reflection/scale/translation) mapping X onto Y."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, S, Vt = np.linalg.svd(Xc.T.dot(Yc))
    R = U.dot(Vt)
    s = S.sum() / (Xc ** 2).sum()
    return s * Xc.dot(R) + Y.mean(0)


def _recover_mds(ax, P, M, v, tint, ec):
    """Reconstruct distances from delays (d_hat = tau*v), run classical MDS, Procrustes-align to
    truth, overlay. Grey connectors show the reconstruction residual."""
    Xr = _procrustes(_cmds(M.astype(float) * v, 2), P)
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.06, 1.06); ax.axis("off"); ax.set_aspect("equal")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                 boxstyle="round,pad=0,rounding_size=0.03", fc=tint, ec=ec, lw=1.5,
                 alpha=0.9, zorder=0, mutation_aspect=0.6))
    for k in range(len(P)):                                   # residual connectors (true -> recovered)
        ax.plot([P[k, 0], Xr[k, 0]], [P[k, 1], Xr[k, 1]], color="#9aa0a6", lw=1.0, zorder=2)
    ax.scatter(P[:, 0], P[:, 1], facecolors="none", edgecolors=GREY, s=130, lw=1.6, zorder=3)
    for k in range(len(P)):
        glossy_dot(ax, Xr[k, 0], Xr[k, 1], 150, NODE_BASE, "#ffffff", NODE_EDGE, z=4)
    ax.text(0.035, 0.95, r"$\circ$ true    $\bullet$ recovered", transform=ax.transAxes,
            fontsize=7.5, color=INK, ha="left", va="top", zorder=8)
    ax.text(0.5, 0.05, r"$\hat v \approx v$   $\checkmark$", transform=ax.transAxes, fontsize=10.5,
            color=ec, ha="center", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ec, lw=1.2), zorder=8)


def _recov_panel(ax, P, D):
    """Geometry recovered by the inverse: same units, teal-tinted, with a v_hat ~ v check."""
    n = len(P); X, Y = P[:, 0], P[:, 1]
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.06, 1.06); ax.axis("off"); ax.set_aspect("equal")
    ax.add_patch(FancyBboxPatch((-0.04, -0.04), 1.08, 1.08,
                 boxstyle="round,pad=0,rounding_size=0.05", fc="#e7f6f3", ec=TEAL, lw=1.5,
                 alpha=0.9, zorder=0))
    seen = set()
    for i in range(n):
        for j in np.argsort(D[i])[1:4]:
            k = (min(i, int(j)), max(i, int(j)))
            if k in seen:
                continue
            seen.add(k)
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color="#bfe0d9", lw=1.0, zorder=1)
    for k in range(n):
        glossy_node(ax, X[k], Y[k], 0.05, NODE_BASE, NODE_LIGHT, NODE_EDGE, z=5)
    ax.text(0.5, 0.05, r"$\hat v \approx v$   $\checkmark$", fontsize=10, color=TEAL,
            ha="center", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=TEAL, lw=1.2), zorder=8)


def _isochrone_field(ax, P, D, src, v, ec, annotate_spacing=False, recovered=False, ghost=None):
    """Delays as a travelling-wave isochrone field tau(x,y)=||(x,y)-src||/v: filled viridis bands
    from the source. Matches the data aspect to the axes box for circular rings. recovered=True
    draws the inferred layout in orange over true positions as white ghost rings."""
    X, Y = P[:, 0], P[:, 1]
    pos = ax.get_position(); fw, fh = ax.figure.get_size_inches()
    ratio = (pos.width * fw) / (pos.height * fh)              # axes box aspect (w/h)
    cx, cy = X.mean(), Y.mean()
    yspan = (Y.max() - Y.min()) + 0.40; xspan = yspan * ratio
    ax.set_xlim(cx - xspan / 2, cx + xspan / 2); ax.set_ylim(cy - yspan / 2, cy + yspan / 2)
    ax.set_aspect("equal"); ax.axis("off")
    sx, sy = X[src], Y[src]
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    GX, GY = np.meshgrid(np.linspace(x0, x1, 480), np.linspace(y0, y1, 320))
    TAU = np.hypot(GX - sx, GY - sy) / v
    kmax = max(2, int(np.ceil(TAU.max())))
    card = FancyBboxPatch((0.012, 0.018), 0.976, 0.964, transform=ax.transAxes,
                          boxstyle="round,pad=0,rounding_size=0.05", fc="none", ec=ec, lw=1.8,
                          zorder=6, mutation_aspect=1.0 / ratio)
    ax.add_patch(card)
    vmax = kmax + 1.4                                          # headroom keeps far corners calm
    cf = ax.contourf(GX, GY, TAU, levels=np.arange(0, kmax + 1.001, 1.0), cmap="viridis",
                     vmin=0, vmax=vmax, alpha=0.96, zorder=1); cf.set_clip_path(card)
    cl = ax.contour(GX, GY, TAU, levels=np.arange(1, kmax + 1), colors="white",
                    linewidths=1.0, alpha=0.5, zorder=2); cl.set_clip_path(card)
    seen = set()
    for i in range(len(P)):
        for j in np.argsort(D[i])[1:4]:
            key = (min(i, int(j)), max(i, int(j)))
            if key in seen:
                continue
            seen.add(key)
            ln, = ax.plot([X[i], X[j]], [Y[i], Y[j]], color="white", lw=0.9, alpha=0.45, zorder=3)
            ln.set_clip_path(card)
    if ghost is not None:                                      # true positions as white ghost rings
        sc = ax.scatter(ghost[:, 0], ghost[:, 1], facecolors="none", edgecolors="white",
                        s=150, lw=1.7, zorder=4); sc.set_clip_path(card)
    for k in range(len(P)):
        if k == src:
            continue
        base = ORANGE if recovered else to_hex(plt.cm.viridis(np.clip(round(D[src, k] / v) / vmax, 0, 1)))
        s1 = ax.scatter([X[k]], [Y[k]], s=185, c=base, edgecolors="white", linewidths=1.5, zorder=7)
        s2 = ax.scatter([X[k]], [Y[k]], s=55, c="white", edgecolors="none", alpha=0.5, zorder=7.1)
        s1.set_clip_path(card); s2.set_clip_path(card)
    glossy_dot(ax, sx, sy, 250, SRC_BASE, SRC_LIGHT, SRC_EDGE, z=8)
    ax.text(sx + 0.07, sy - 0.085, "source", fontsize=7.2, color="white", ha="left", va="top",
            fontweight="bold", zorder=9, bbox=dict(boxstyle="round,pad=0.16", fc=SRC_EDGE, ec="none", alpha=0.92))
    if annotate_spacing:                                      # radial caliper: ring spacing = v
        oth = np.array([k for k in range(len(P)) if k != src])
        angs = np.sort(np.arctan2(Y[oth] - sy, X[oth] - sx))  # place it in the EMPTIEST sector
        gaps = np.diff(np.concatenate([angs, [angs[0] + 2 * np.pi]]))
        ang = angs[int(np.argmax(gaps))] + gaps.max() / 2
        ux, uy = np.cos(ang), np.sin(ang)
        p1 = (sx + v * ux, sy + v * uy); p2 = (sx + 2 * v * ux, sy + 2 * v * uy)
        cap = FancyArrowPatch(p1, p2, arrowstyle="<|-|>", mutation_scale=9, color="white",
                              lw=2.0, zorder=8); ax.add_patch(cap); cap.set_clip_path(card)
        ax.text(0.5 * (p1[0] + p2[0]) - 0.08 * uy, 0.5 * (p1[1] + p2[1]) + 0.08 * ux, r"$\Delta r = v$",
                fontsize=8.2, color="white", ha="center", va="center", fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.16", fc=INK, ec="none", alpha=0.62))
    if recovered:                                             # legend: estimate (orange) vs true
        ax.text(0.5, 0.035, r"$\hat v \approx v$    $\bullet$ estimate   $\circ$ true",
                transform=ax.transAxes, fontsize=7.4, color="white", ha="center", va="bottom",
                fontweight="bold", zorder=12, bbox=dict(boxstyle="round,pad=0.2", fc=INK, ec="none", alpha=0.5))


def _recovered_layout(ax, P, M, v, ec):
    """Inverse output as a light recovered-geometry card: classical MDS on the delays ->
    Procrustes onto truth; orange recovered units vs grey true markers, with residual connectors."""
    Xr = _procrustes(_cmds(M.astype(float) * v, 2), P)
    pos = ax.get_position(); fw, fh = ax.figure.get_size_inches()
    ratio = (pos.width * fw) / (pos.height * fh)
    cx, cy = P[:, 0].mean(), P[:, 1].mean()
    yspan = (P[:, 1].max() - P[:, 1].min()) + 0.40; xspan = yspan * ratio
    ax.set_xlim(cx - xspan / 2, cx + xspan / 2); ax.set_ylim(cy - yspan / 2, cy + yspan / 2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.012, 0.018), 0.976, 0.964, transform=ax.transAxes,
                 boxstyle="round,pad=0,rounding_size=0.05", fc="#e9f6f2", ec=ec, lw=1.8,
                 zorder=0, mutation_aspect=1.0 / ratio))
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    seen = set()
    for i in range(len(P)):
        for j in np.argsort(D[i])[1:4]:
            key = (min(i, int(j)), max(i, int(j)))
            if key in seen:
                continue
            seen.add(key); ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color="#cdd6df", lw=1.0, zorder=1)
    # TRUE = open grey ring; ESTIMATE = orange glossy dot sitting INSIDE it (accurate -> centred)
    ax.scatter(P[:, 0], P[:, 1], facecolors="none", edgecolors=GREY, s=240, lw=1.7, zorder=3)
    for k in range(len(P)):
        glossy_dot(ax, Xr[k, 0], Xr[k, 1], 116, ORANGE, "#ffe3cf", "#b5611f", z=5)
    ax.text(0.5, 0.035, r"$\bullet$ recovered (est.)    $\circ$ true       $\hat v\approx v$",
            transform=ax.transAxes, fontsize=7.0, color=INK, ha="center", va="bottom",
            fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=ORANGE, lw=1.0, alpha=0.92))


def _slope_readout(ax, D, M, v):
    """Compact inverse engine: every pair's (distance, delay) lands on a line of slope 1/v."""
    iu = np.triu_indices_from(D, 1); d, t = D[iu], M[iu]
    ax.set_xlim(0, d.max() * 1.08); ax.set_ylim(-0.4, t.max() + 0.8)
    ax.plot([0, d.max() * 1.05], [0, d.max() * 1.05 / v], color=INK, lw=1.8, ls=(0, (5, 3)), zorder=2)
    ax.scatter(d, t, s=20, color=TEAL, edgecolor="white", lw=0.6, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GREY); ax.spines["bottom"].set_color(GREY)
    ax.set_xlabel(r"$d_{ij}$", fontsize=8, labelpad=1); ax.set_ylabel(r"$\tau_{ij}$", fontsize=8, labelpad=1)
    ax.text(0.05, 0.96, r"slope $=1/\hat v$", transform=ax.transAxes, fontsize=8.2, color=INK,
            fontweight="bold", ha="left", va="top")


def panel_c():
    """Forward map and velocity inverse as a travelling-wave isochrone field: two stacked domain
    objects -> arrows -> two stacked outputs, with a curved H^-1 inverse wrap."""
    fig = plt.figure(figsize=(7.6, 4.6))
    rng = np.random.default_rng(7)
    P = spread2d(8, rng, min_d=0.22)
    P[:, 0] = 0.5 + (P[:, 0] - 0.5) * 1.5         # widen layout so all 4 cells fill equally (symmetry)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    v = 0.14                                       # fine delays -> tight (~2%) MDS recovery
    M = np.round(D / v).astype(int); np.fill_diagonal(M, 0)
    src = int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))

    fig.text(0.005, 0.978, "c", fontsize=16, fontweight="bold", va="top")
    fig.text(0.5, 0.978, "forward map  &  velocity inverse", fontsize=11, color=INK,
             ha="center", va="top", fontweight="bold")

    # wide cells, SHORT central gutter (just enough for the operator formulas)
    xL0, xL1 = 0.095, 0.445
    xR0, xR1 = 0.600, 0.945
    wL, wR = xL1 - xL0, xR1 - xR0
    hcell, yT, yB = 0.355, 0.500, 0.095
    gut = (xL1, xR0)

    fig.text(0.042, yT + hcell / 2, "FORWARD", fontsize=9, color=EDGE, rotation=90,
             ha="center", va="center", fontweight="bold")
    fig.text(0.042, yB + hcell / 2, "INVERSE", fontsize=9, color=TEAL, rotation=90,
             ha="center", va="center", fontweight="bold")

    # ROW 1 (forward): geometry --tau=round(d/v)--> isochrone field
    gTL = fig.add_axes([xL0, yT, wL, hcell])
    draw_network(gTL, P=P, legend=False, scalebar=False, box=True, dist_fs=10.5, src_label=False)
    fTR = fig.add_axes([xR0, yT, wR, hcell]); _isochrone_field(fTR, P, D, src, v, TEAL, annotate_spacing=True)
    a1 = fig.add_axes([gut[0], yT + 0.105, gut[1] - gut[0], 0.15])
    operator_arrow(a1, r"$\tau_{ij}=\mathrm{round}(d_{ij}/v)$", None, fs=8.6)

    # ROW 2 (inverse): observed field --v<-spacing--> recovered geometry & v
    fBL = fig.add_axes([xL0, yB, wL, hcell]); _isochrone_field(fBL, P, D, src, v, TEAL)
    rBR = fig.add_axes([xR0, yB, wR, hcell]); _recovered_layout(rBR, P, M, v, TEAL)
    a2 = fig.add_axes([gut[0], yB + 0.10, gut[1] - gut[0], 0.16])
    operator_arrow(a2, r"$\hat v=\dfrac{\sum d_{ij}\,\tau_{ij}}{\sum \tau_{ij}^{2}}$", None,
                   col=TEAL, lw=2.4, fs=9.5)

    # curved H^-1 inverse wrap, curving out into the left margin
    axW = fig.add_axes([0.0, 0.0, 1.0, 1.0]); axW.axis("off"); axW.set_xlim(0, 1); axW.set_ylim(0, 1)
    yc_top, yc_bot = yT + hcell * 0.5, yB + hcell * 0.5
    wrap = FancyArrowPatch((0.078, yc_top), (0.078, yc_bot), connectionstyle="arc3,rad=0.22",
                           arrowstyle="-|>", mutation_scale=16, color=TEAL, lw=2.3, zorder=20)
    axW.add_patch(wrap)
    axW.text(0.052, (yc_top + yc_bot) / 2, r"$\mathcal{H}^{-1}$", fontsize=13, color=TEAL,
             ha="center", va="center", fontweight="bold", zorder=21)

    # sub-figure captions
    cyt, cyb = yT + hcell + 0.010, yB + hcell + 0.010
    vir55 = to_hex(plt.cm.viridis(0.55))
    fig.text((xL0 + xL1) / 2, cyt, r"geometry  $d_{ij}$", fontsize=8.6, color=EDGE,
             ha="center", va="bottom", fontweight="bold")
    fig.text((xR0 + xR1) / 2, cyt, r"isochrone field  $\tau(x,y)$", fontsize=8.6, color=vir55,
             ha="center", va="bottom", fontweight="bold")
    fig.text((xL0 + xL1) / 2, cyb, r"observed field  $\tau(x,y)$", fontsize=8.6, color=vir55,
             ha="center", va="bottom", fontweight="bold")
    fig.text((xR0 + xR1) / 2, cyb, r"recovered geometry  &  $\hat v$  (estimate)", fontsize=8.6,
             color=ORANGE, ha="center", va="bottom", fontweight="bold")

    # bottom takeaway strip
    axCap = fig.add_axes([0.03, 0.012, 0.94, 0.05]); axCap.axis("off")
    axCap.set_xlim(0, 1); axCap.set_ylim(0, 1)
    axCap.add_patch(Rectangle((0.0, 0.10), 1.0, 0.80, fc="#f4f6f8", ec=GREY, lw=1.0, zorder=0))
    axCap.text(0.5, 0.5, r"forward - distance sets the delay  ($\tau=\mathrm{round}(d/v)$);    "
               r"inverse - ring spacing $\Delta r$ recovers $v$ and the layout", ha="center",
               va="center", fontsize=7.6, color=INK, zorder=1)
    save(fig, "panel_c")


if __name__ == "__main__":
    print("building figure-1 a/b/c composites ->", OUT)
    panel_a(); panel_b(); panel_c()
    print("done.")
