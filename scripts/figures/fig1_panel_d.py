"""Figure 1 panel d/e -- architecture & training.

Top band: geometry d_ij + drive u(t) -> delays tau=round(d/v) -> one delay-coupled RNN ->
travelling-wave activity x(t), trained by ||x - x*||^2 + lambda*C.
Bottom band (e): metric delays -> coherent wave vs shuffled delays -> scrambled.

Reuses the fig1_panels_abc helpers (draw_network, glossy_node, palette). Near-square canvas so
it composes with panels a/b/c.
"""
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# reuse the fig1_panels_abc helpers
spec = importlib.util.spec_from_file_location(
    "HM", "/Users/akshgarg/spatial-delay-rnn/scripts/figures/fig1_panels_abc.py")
HM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(HM)

TEAL, ORANGE, GREEN, BLUE, INK, GREY, RED, EDGE = \
    HM.TEAL, HM.ORANGE, HM.GREEN, HM.BLUE, HM.INK, HM.GREY, HM.RED, HM.EDGE
MESH, DIST = HM.MESH, HM.DIST
NODE_BASE, NODE_LIGHT, NODE_EDGE = HM.NODE_BASE, HM.NODE_LIGHT, HM.NODE_EDGE
SRC_BASE, SRC_LIGHT, SRC_EDGE = HM.SRC_BASE, HM.SRC_LIGHT, HM.SRC_EDGE
VIR = plt.cm.viridis
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "mathtext.fontset": "dejavusans"})

FULL = "/Users/akshgarg/spatial-delay-rnn/figures/assets/panel_d.png"
SVG = "/Users/akshgarg/spatial-delay-rnn/figures/assets/panel_d.svg"
PREV = "/tmp/paneld_e_v1_prev.png"

# fixed near-square canvas in a 0..100 (x) by 0..96 (y) percentage grid  (AR ~ 1.10)
AR = 1.10
H_IN = 9.0
W_IN = H_IN * AR
XMAX, YMAX = 100.0, 100.0 / AR        # y range so the data coords are isotropic


def C(xp, yp):
    """percentage coord, y measured from the TOP."""
    return (xp / 100.0 * XMAX, YMAX - yp / 100.0 * YMAX)


def card(ax, x, y, w, h, ec=EDGE, fc="white", lw=1.4, r=1.4, z=2, dashed=False, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, ls=(0, (5, 3)) if dashed else "-", zorder=z,
                 joinstyle="round", alpha=alpha, mutation_aspect=1.0))


def rect_pct(l, t, r, b):
    """l,t,r,b in PERCENT (t,b from top) -> (x,y,w,h) in data coords."""
    x0, y0 = C(l, b)
    x1, y1 = C(r, t)
    return (x0, y0, x1 - x0, y1 - y0)


def arrow(ax, p0, p1, color=INK, lw=2.0, ms=15, dashed=False, z=6, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                 shrinkA=2, shrinkB=2, zorder=z, capstyle="round",
                 linestyle=(0, (5, 3)) if dashed else "-",
                 connectionstyle=f"arc3,rad={rad}"))


def label(ax, p, s, size=10, color=INK, weight="bold", ha="center", va="center", z=10, style="normal"):
    ax.text(p[0], p[1], s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            zorder=z, style=style)


# mini glyphs
def glyph_geometry(ax, x, y, w, h):
    """Small glossy spatial network with a red source + dashed distances."""
    rng = np.random.default_rng(11)
    P = HM.spread2d(7, rng, min_d=0.26)
    X = x + (0.12 + 0.76 * P[:, 0]) * w
    Y = y + (0.12 + 0.76 * P[:, 1]) * h
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    seen = set()
    for i in range(len(P)):
        for j in np.argsort(D[i])[1:3]:
            k = (min(i, int(j)), max(i, int(j)))
            if k in seen:
                continue
            seen.add(k)
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color=MESH, lw=1.0, zorder=3)
    src = int(np.argmin(np.hypot(P[:, 0] - 0.5, P[:, 1] - 0.5)))
    near = int(np.argsort(D[src])[1])
    ax.plot([X[src], X[near]], [Y[src], Y[near]], color=DIST, lw=1.5, ls=(0, (3, 2)), zorder=4)
    mx, my = 0.5 * (X[src] + X[near]), 0.5 * (Y[src] + Y[near])
    rr = 0.052 * min(w, h) / 0.5 * 5
    for k in range(len(P)):
        if k == src:
            continue
        HM.glossy_node(ax, X[k], Y[k], 0.05 * w, NODE_BASE, NODE_LIGHT, NODE_EDGE, z=5)
    HM.glossy_node(ax, X[src], Y[src], 0.066 * w, SRC_BASE, SRC_LIGHT, SRC_EDGE, z=6)
    ax.text(mx + 0.05 * w, my, r"$d_{ij}$", color=DIST, fontsize=8.5, style="italic",
            fontweight="bold", ha="left", va="center", zorder=7)


def glyph_drive(ax, x, y, w, h):
    """external drive u(t): a smooth orange pulse train."""
    t = np.linspace(0, 1, 240)
    s = (np.exp(-((t - 0.32) ** 2) / 0.006) + 0.75 * np.exp(-((t - 0.62) ** 2) / 0.012)
         + 0.5 * np.exp(-((t - 0.86) ** 2) / 0.008))
    ax.plot(x + (0.08 + 0.86 * t) * w, y + 0.22 * h + 0.62 * s * h, color=ORANGE, lw=2.0,
            zorder=4, solid_capstyle="round")


def glyph_delaymap(ax, x, y, w, h):
    """Delays as viridis staircase bars of growing length."""
    fr = [0.30, 0.52, 0.74, 0.95]
    for i, f in enumerate(fr):
        yy = y + (0.16 + i * 0.225) * h
        ax.plot([x + 0.10 * w, x + (0.10 + 0.80 * f) * w], [yy, yy], color=VIR(i / 3),
                lw=4.6, solid_capstyle="round", zorder=4)


def glossy_recurrent_net(ax, x, y, w, h, seed=4, scrambled=False, n=9, nr=0.052):
    """One delay-coupled RNN, glossy units in physical space.
    metric (scrambled=False): each unit couples to nearest neighbours, edges coloured by delay.
    shuffled (scrambled=True): same nodes, delay->edge assignment permuted -> long crossing edges."""
    rng = np.random.default_rng(seed)
    P = HM.spread2d(n, rng, min_d=0.24)
    X = x + (0.12 + 0.76 * P[:, 0]) * w
    Y = y + (0.12 + 0.76 * P[:, 1]) * h
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    dmax = D.max()
    rng2 = np.random.default_rng(seed + 5)
    edges = []
    if scrambled:
        # entry-shuffle: random long-range targets, random (mismatched) delay colours
        for i in range(n):
            for j in rng2.choice([k for k in range(n) if k != i], size=2, replace=False):
                edges.append((i, int(j), VIR(rng2.uniform(0.05, 0.95))))
    else:
        for i in range(n):
            for j in np.argsort(D[i])[1:3]:               # nearest neighbours
                edges.append((i, int(j), VIR(0.10 + 0.82 * D[i, int(j)] / dmax)))
    for i, j, col in edges:
        ax.add_patch(FancyArrowPatch((X[i], Y[i]), (X[j], Y[j]), arrowstyle="-|>",
                     mutation_scale=8, color=col, lw=1.8, alpha=0.92, zorder=4,
                     shrinkA=5, shrinkB=5, capstyle="round"))
    for k in range(n):
        HM.glossy_node(ax, X[k], Y[k], nr * w, NODE_BASE, NODE_LIGHT, NODE_EDGE, z=6)


def wave_surface(ax, x, y, w, h, coherent=True, z=3):
    """Travelling-wave activity x(t) on a (space x time) viridis surface.
    coherent: a clean diagonal plane wave. scrambled: phase noise with no coherent front."""
    g = 220
    GX, GY = np.meshgrid(np.linspace(0, 1, g), np.linspace(0, 1, g))
    if coherent:
        Z = np.sin(2 * np.pi * (2.3 * GX - 1.7 * GY))               # ordered diagonal travelling wave
    else:
        rng = np.random.default_rng(7)
        ph = rng.uniform(0, 2 * np.pi, g)                           # random per-column phase + speckle
        Z = np.sin(2 * np.pi * 2.3 * GX - 1.7 * 2 * np.pi * GY + ph[None, :])
        Z = Z * (0.5 + 0.5 * rng.uniform(size=(g, g)))
        Z += 0.9 * rng.standard_normal((g, g))
    extent = (x, x + w, y, y + h)
    im = ax.imshow(Z, extent=extent, origin="lower", cmap="viridis", aspect="auto",
                   zorder=z, interpolation="bilinear")
    return im


def add_net(fig, rect, seed=21, shuffled=False, dist_fs=8.0, pad=0.10, n=9):
    """Drop an HM.draw_network into data-rect `rect` via a sub-axes, so every net in panel d
    matches panel a. shuffled=True overlays red criss-crossing couplings."""
    x, y, w, h = rect
    sub = fig.add_axes([(x + pad * w) / XMAX, (y + pad * h) / YMAX,
                        (1 - 2 * pad) * w / XMAX, (1 - 2 * pad) * h / YMAX])
    sub.patch.set_visible(False)
    rng = np.random.default_rng(seed)
    P = HM.spread2d(n, rng, min_d=0.22)
    P, src, near, far = HM.draw_network(sub, P=P, legend=False, scalebar=False, box=False,
                                        src_label=False, dist_fs=dist_fs)
    if shuffled:
        X, Y = P[:, 0], P[:, 1]
        rs = np.random.default_rng(seed + 99)
        seen_sh, drawn = set(), 0
        while drawn < 3:
            i, j = int(rs.integers(0, len(P))), int(rs.integers(0, len(P)))
            key = (min(i, j), max(i, j))
            if i == j or key in seen_sh:
                continue
            seen_sh.add(key); drawn += 1
            sub.add_patch(FancyArrowPatch((X[i], Y[i]), (X[j], Y[j]),
                          connectionstyle="arc3,rad=0.32", arrowstyle="-", color=RED,
                          lw=1.8, alpha=0.8, zorder=4))
        sub.text(0.5, -0.03, "couplings rewired", transform=sub.transAxes, fontsize=7.0,
                 color=RED, ha="center", va="top", fontweight="bold", zorder=6)
    return sub


def line_wave(ax, x, y, w, h, n=8, z=4):
    """Travelling-wave activity x(t) as stacked viridis line traces: near units fire early, far
    units later -> a marching wavefront."""
    T = np.linspace(0, 1, 240)
    x0, x1 = x + 0.10 * w, x + 0.93 * w
    y0, y1 = y + 0.12 * h, y + 0.90 * h
    rows = np.linspace(y0, y1, n)
    amp = (y1 - y0) / n * 0.92
    peaks = []
    for k in range(n):
        d = k / (n - 1)
        arrival = 0.14 + 0.66 * d
        sig = np.exp(-((T - arrival) ** 2) / (2 * 0.045 ** 2))
        col = VIR(0.12 + 0.78 * d)
        xx = x0 + T * (x1 - x0)
        yy = rows[k] + amp * sig
        ax.plot(xx, yy, color=col, lw=1.6, zorder=z, solid_capstyle="round")
        peaks.append((x0 + arrival * (x1 - x0), rows[k] + amp))
    ax.plot([p[0] for p in peaks], [p[1] for p in peaks], ls=(0, (4, 3)), color=INK,
            lw=1.1, alpha=0.7, zorder=z + 1)


def build():
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_aspect("equal")
    ax.axis("off")

    # titles
    label(ax, C(1.5, 3.0), "d", size=17, ha="left", va="top")
    label(ax, C(50, 2.4), "architecture  &  training", size=13, va="top")

    def cap(p, s, color=INK, size=8.8):                   # title above a card
        label(ax, p, s, size=size, color=color, va="bottom")

    # top band: one network, one flow
    TY0, TY1 = 11.5, 30.0                                 # top-band card top/bottom (%)
    midy = (TY0 + TY1) / 2

    # (1) inputs: geometry (top) + drive (bottom), stacked in one soft card
    gx, gy, gw, gh = rect_pct(2.5, TY0, 19.5, TY1)
    card(ax, gx, gy, gw, gh, ec=EDGE, fc="#eef3fb", lw=1.3)
    add_net(fig, (gx, gy + 0.44 * gh, gw, 0.54 * gh), seed=11, dist_fs=0, n=7, pad=0.05)
    glyph_drive(ax, gx, gy + 0.02 * gh, gw, 0.34 * gh)
    ax.plot([gx + 0.10 * gw, gx + 0.90 * gw], [gy + 0.42 * gh] * 2, color="#cfd6df", lw=0.9, zorder=4)
    cap(C(11.0, TY0 - 1.0), r"geometry $d_{ij}$  +  drive $u(t)$", color=EDGE)

    # (2) delay map
    mx, my, mw, mh = rect_pct(26.5, TY0, 41.5, TY1)
    card(ax, mx, my, mw, mh, ec=TEAL, fc="#e9f6f2", lw=1.3)
    glyph_delaymap(ax, mx, my, mw, mh)
    cap(C(34.0, TY0 - 1.0), r"delays $\tau_{ij}=\mathrm{round}(d_{ij}/v)$", color=TEAL)

    # (3) ONE delay-coupled RNN
    rx, ry, rw, rh = rect_pct(48.5, TY0, 70.5, TY1)
    card(ax, rx, ry, rw, rh, ec=EDGE, fc="white", lw=1.9)
    add_net(fig, (rx, ry, rw, rh), seed=4, shuffled=False, dist_fs=9.0)
    cap(C(59.5, TY0 - 1.0), "one delay-coupled RNN", color=EDGE, size=9.2)

    # (4) activity x(t)
    ax_x, ay, aw, ah = rect_pct(77.5, TY0, 97.5, TY1)
    card(ax, ax_x, ay, aw, ah, ec=EDGE, fc="white", lw=1.3)
    line_wave(ax, ax_x + 0.04 * aw, ay + 0.04 * ah, 0.92 * aw, 0.92 * ah, n=8)
    cap(C(87.5, TY0 - 1.0), r"travelling-wave activity $x(t)$")

    # top-band flow arrows (horizontal, on the row midline)
    arrow(ax, C(19.5, midy), C(26.5, midy))
    arrow(ax, C(41.5, midy), C(48.5, midy))
    arrow(ax, C(70.5, midy), C(77.5, midy))

    # update equation, centred under the RNN
    label(ax, C(59.5, 34.8), r"$x_i(t)=f\!\left(\sum_j W_{ij}\,x_j(t-\tau_{ij})+b_i\,u(t)\right)$",
          size=10.5, weight="normal", va="center")

    # training objective: one feedback loop
    LY0, LY1 = 41.0, 53.0
    # target chip (right)
    tx, ty, tw, th = rect_pct(81.0, LY0 + 0.5, 90.5, LY1 - 0.5)
    card(ax, tx, ty, tw, th, ec=GREY, fc="white", lw=1.2)
    line_wave(ax, tx + 0.06 * tw, ty + 0.06 * th, 0.88 * tw, 0.88 * th, n=6)
    cap(C(85.75, LY0 - 0.4), r"target $x^*$", color=GREY, size=7.8)

    # loss objective box (centre)
    lx, ly, lw_, lh = rect_pct(28.0, LY0, 73.0, LY1)
    card(ax, lx, ly, lw_, lh, ec=BLUE, fc="#eef5ff", lw=1.7)
    cxl = lx + 0.5 * lw_
    cap(C(50.5, LY0 - 0.4), "trained by minimising", color=BLUE)
    label(ax, (cxl, ly + 0.60 * lh),
          r"$\mathcal{L}=\|x-x^*\|^2\;+\;\lambda\,\sum_{ij}|W_{ij}|\,\tau_{ij}$",
          size=11.5, weight="normal", color=INK, va="center")
    label(ax, (lx + 0.205 * lw_, ly + 0.20 * lh), "activity loss", size=7.4, color=GREY,
          weight="bold", va="center")
    label(ax, (lx + 0.745 * lw_, ly + 0.20 * lh), "conduction-time economy  $C$", size=7.4,
          color=TEAL, weight="bold", va="center")

    # feedback loop:
    #   activity x(t) -> target/loss column (down the right)
    arrow(ax, C(87.5, TY1 + 0.5), C(85.75, LY0 + 0.5), color=GREY, lw=1.3, ms=11)
    #   target -> loss box
    arrow(ax, C(81.0, LY0 + 6), C(73.0, LY0 + 6), color=GREY, lw=1.3, ms=11)
    #   backprop: loss -> W, up the left region back into the RNN bottom
    rpts = [C(28.0, LY0 + 6), C(13.0, LY0 + 6), C(13.0, TY1 + 2.5), C(49.0, TY1 + 2.5),
            C(54.0, TY1 + 0.4)]
    for a, b in zip(rpts[:-1], rpts[1:]):
        if b is rpts[-1]:
            arrow(ax, a, b, color=RED, lw=1.5, dashed=True, ms=13)
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=RED, lw=1.5, ls=(0, (5, 3)),
                    zorder=6, solid_capstyle="round")
    label(ax, C(6.5, (LY0 + 6 + TY1 + 2.5) / 2 - 0.5),
          r"backprop" + "\n" + r"$\nabla_W\mathcal{L}$",
          size=8.2, color=RED, ha="center", va="center")

    # part e: geometric effect (metric vs shuffled)
    label(ax, C(2.7, 57.3), "e", size=17, ha="left", va="center")
    label(ax, C(50, 57.3), "Geometric effect", size=13, va="center")
    label(ax, C(50, 59.8), "metric delays make the waves cohere", size=9.0,
          weight="normal", color=GREY, va="center")

    CT, CB = 62.0, 95.5

    def ecard(l, r, ec, tint, seed, coherent, shuffled, title, wave_lab, badge):
        cx, cy, cw, ch = rect_pct(l, CT, r, CB)
        card(ax, cx, cy, cw, ch, ec=ec, fc=tint, lw=1.6, r=1.8, z=2)
        label(ax, C((l + r) / 2, CT + 2.4), title, size=9.2, color=ec, va="center")
        # badge (check / cross) top-right
        bxy = C(r - 2.8, CT + 2.7)
        ax.scatter([bxy[0]], [bxy[1]], s=210, c=ec, edgecolors="white", lw=1.5, zorder=10)
        label(ax, bxy, badge, size=11, color="white", va="center", z=11)
        # network (left half)
        nl, nr = l + 1.5, l + 20.5
        nx, ny, nw, nh = rect_pct(nl, CT + 5.2, nr, CB - 2.0)
        add_net(fig, (nx, ny, nw, nh), seed=seed, shuffled=shuffled, dist_fs=0, n=9, pad=0.04)
        ay = (CT + 5.2 + CB - 2.0) / 2
        arrow(ax, C(nr + 0.4, ay), C(nr + 1.8, ay), color=ec, lw=1.6, ms=11)
        # wave raster (right half) with coloured border
        wl, wr = l + 22.5, r - 1.5
        wx, wy, ww, wh = rect_pct(wl, CT + 6.2, wr, CB - 4.0)
        wave_surface(ax, wx, wy, ww, wh, coherent=coherent)
        card(ax, wx, wy, ww, wh, ec=ec, fc="none", lw=1.5, r=0.7, z=8)
        label(ax, C((wl + wr) / 2, CB - 1.6), wave_lab, size=8.3, color=ec, va="center")

    ecard(3.5, 47.5, TEAL, "#e7f6f3", 4, True, False,
          r"metric delays  $\tau=\mathrm{round}(d/v)$", "coherent wave", "✓")
    ecard(52.5, 96.5, RED, "#fdeceb", 9, False, True,
          "shuffled delays  (entry-shuffle)", "scrambled activity", "✗")

    fig.savefig(FULL, transparent=True, dpi=300)
    fig.savefig(SVG, transparent=True)
    fig.savefig(PREV, facecolor="white", dpi=120)
    plt.close(fig)
    # preview as a <=1100px white-bg PNG
    try:
        from PIL import Image
        im = Image.open(PREV).convert("RGB")
        if max(im.size) > 1100:
            im.thumbnail((1100, 1100), Image.LANCZOS)
            im.save(PREV)
    except Exception as e:
        print("preview downscale skipped:", e)
    print(f"wrote {FULL}  ({round(W_IN*300)}x{round(H_IN*300)} px, AR={AR})")

    # full-figure composite: a/b/c column (left) + this panel d/e (right)
    from PIL import Image as _I
    def _flat(p):
        im = _I.open(p).convert("RGBA")
        bg = _I.new("RGBA", im.size, (255, 255, 255, 255))
        return _I.alpha_composite(bg, im).convert("RGB")
    _A = "/Users/akshgarg/spatial-delay-rnn/figures/assets"
    _wL, _gap = 720, 14
    _abc = [_flat(f"{_A}/panel_{k}.png") for k in "abc"]
    _abc = [im.resize((_wL, int(im.height * _wL / im.width))) for im in _abc]
    _H = sum(im.height for im in _abc) + _gap * (len(_abc) + 1)
    _col = _I.new("RGB", (_wL, _H), (255, 255, 255)); _yy = _gap
    for im in _abc:
        _col.paste(im, (0, _yy)); _yy += im.height + _gap
    _d = _flat(FULL); _dw = int(_d.width * _H / _d.height); _d = _d.resize((_dw, _H))
    _full = _I.new("RGB", (_wL + _gap + _dw, _H), (255, 255, 255))
    _full.paste(_col, (0, 0)); _full.paste(_d, (_wL + _gap, 0))
    _full.save("/tmp/full_e_v1.png")
    print("wrote /tmp/full_e_v1.png")


if __name__ == "__main__":
    build()
