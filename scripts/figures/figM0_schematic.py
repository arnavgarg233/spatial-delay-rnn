"""Fig M0: the a-d wired schematic (hand-authored SVG, rendered to PNG/PDF via resvg).

Left column a/b/c sets up the problem (spatial embedding, distance->delay, metric
consistency); right panel d is the forward-model + velocity-inverse dataflow.
Output: figures/figM0_schematic.{svg,png,pdf}.
"""
import os, io, base64, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Palette: white page, near-black line-work, pastel fills.
# GREEN = metric (ordered) delays; ORANGE = entry-shuffled (same multiset, metric broken).
BLUE   = "#3f70af"; BLUE_F  = "#cce4f6"; BLUE_FF = "#ecf4fb"
GREEN  = "#3f9c78"; GREEN_F = "#cfeecf"; GREEN_FF= "#eef8ef"
WARM   = "#d97a3c"; WARM_F  = "#fce4cc"; WARM_FF = "#fdf3e7"
RED    = "#c02400"; PURPLE = "#906c9c"
ORANGE  = "#e07b1f"; ORANGE_F = "#fde2c2"; ORANGE_FF = "#fdf2e3"
TRK_G   = GREEN
TRK_O   = ORANGE
INK    = "#1a1a1a"; GREY = "#9aa0a6"; HAIR = "#cdd2d8"
PAPER  = "#ffffff"
TEAL   = "#187878"
SPINE  = "#9aa1a8"                              # the dataflow rail colour
# Style lock: one cool neutral (BLUE = structure), one accent (GREEN = metric only),
# one hairline border weight.
NEUTRAL = BLUE
ACCENT  = GREEN
BORD_C  = HAIR
BORD_SW = 1.0
rng = np.random.default_rng(7)
# Delay ramp: one teal sequential scale encoding delay magnitude across panels b/d/e.
# pale aqua (short/fast) -> deep teal (long/slow).
from matplotlib.colors import LinearSegmentedColormap
VIR = LinearSegmentedColormap.from_list(
    "tealseq",
    ["#e4f5f1", "#a9e1d6", "#5cc0b0", "#2a9d8f", "#1c7d76", "#11534f"])
# Desaturated field ramp for the arrival heatmaps: white-ish (early) -> slate-teal (late).
FIELD = LinearSegmentedColormap.from_list(
    "fieldseq",
    ["#f4faf9", "#dcefe9", "#bfe0d8", "#9ecbc4", "#7fb1ab", "#5e918d"])

def vhex(t):
    rgb = VIR(float(np.clip(t, 0, 1)))
    return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

GINK = "#111417"

# Typography: one sans family, three sizes (title/label/annot); maths in serif-italic.
SANS = "Helvetica, Arial, sans-serif"
MATH = "Georgia, 'Times New Roman', 'DejaVu Serif', serif"
SZ_TITLE = 13.5
SZ_LABEL = 10.0
SZ_ANNOT = 8.5

def mtext(x, y, t, size=13, fill=INK, anchor="middle", weight="normal",
          opacity=1.0):
    add(f'<text x="{x:.2f}" y="{y:.2f}" font-family="{MATH}" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}" font-style="italic" opacity="{opacity}">{t}</text>')

def _sub(content, italic=True, size="0.62em", drop="0.34em"):
    sty = "italic" if italic else "normal"
    neg = "-" + drop
    return (f'<tspan font-size="{size}" font-style="{sty}" dy="{drop}">'
            f'{content}</tspan>'
            f'<tspan font-size="{size}" dy="{neg}">&#8203;</tspan>')

def _sup(content, italic=False, size="0.66em", lift="0.50em"):
    sty = "italic" if italic else "normal"
    return (f'<tspan font-size="{size}" font-style="{sty}" dy="-{lift}">'
            f'{content}</tspan>'
            f'<tspan font-size="{size}" dy="{lift}">&#8203;</tspan>')

def mvar(name, sub=None, sup=None, italic=True):
    sty = ' font-style="italic"' if italic else ' font-style="normal"'
    out = f'<tspan{sty}>{name}</tspan>'
    if sub is not None:
        out += _sub(sub)
    if sup is not None:
        out += _sup(sup)
    return out

def mathspan(t, size=13, fill=INK, anchor="middle", weight="normal",
             opacity=1.0, x=None, y=None):
    add(f'<text x="{x:.2f}" y="{y:.2f}" font-family="{MATH}" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}" font-style="italic" opacity="{opacity}">{t}</text>')

def axis_label(x, y, t, size=SZ_LABEL, fill=GREY, anchor="middle"):
    mtext(x, y, t, size=size, fill=fill, anchor=anchor)

def origin_mark(x, y, size=SZ_LABEL, fill=GREY):
    mtext(x, y, "O", size=size, fill=fill, anchor="middle")

# Field PNGs + delay-matrix thumbnails (matplotlib -> data URI).
def field_png(kind, src=(0.30, 0.52), n=600):
    """Two arrival fields.
    ordered -> smooth radial gradient (arrival time rises with distance from source).
    shuffle -> same value range re-laid by a low-frequency warp: non-radial (metric
               broken) but still a clean continuous gradient, no speckle.
    """
    gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    sx, sy = src
    dist = np.sqrt((gx - sx) ** 2 + (gy - sy) ** 2)
    if kind == "ordered":
        F = dist
        cmap = cm.get_cmap("viridis")
    else:
        # deterministic low-frequency warp (no noise): broad sinusoidal lobes break
        # the radial monotonicity.
        F = (0.50 + 0.30 * np.sin(3.1 * gx + 0.6) * np.cos(2.7 * gy - 0.4)
             + 0.22 * np.sin(2.3 * gy + 1.8)
             + 0.16 * np.cos(3.6 * gx - 1.0) * np.sin(2.0 * gy + 0.5)
             + 0.10 * np.sin(4.4 * (gx + gy)))
        cmap = cm.get_cmap("viridis")
    F = (F - F.min()) / (F.max() - F.min() + 1e-9)
    img = (cmap(F)[..., :3] * 255).astype("uint8")
    return Image.fromarray(img, "RGB")

def png_data_uri(pil_img):
    buf = io.BytesIO(); pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def cbar_png(n=512, h=22, horizontal=True):
    # arrival colorbar matching the viridis field ramp
    ramp = np.linspace(0, 1, n)[None, :].repeat(h, axis=0)
    img = (cm.get_cmap("viridis")(ramp)[..., :3] * 255).astype("uint8")
    if not horizontal:
        img = np.transpose(img[:, ::-1], (1, 0, 2))   # vertical, early(top)->late(bottom)
    return Image.fromarray(img, "RGB")

def scatter_png(w=420, h=300, dpi=140):
    """Recovered arrival time vs distance from source.
    metric  -> tight near-linear band; shuffle -> wide cloud (no readout).
    Returns a transparent-background PNG."""
    rng2 = np.random.default_rng(11)
    d = np.linspace(0.04, 0.96, 64)
    a_metric  = 0.10 + 0.82 * d + rng2.normal(0, 0.030, d.shape)
    a_shuffle = 0.50 + rng2.normal(0, 0.235, d.shape)
    a_shuffle = np.clip(a_shuffle, 0.02, 0.98)
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    axp = fig.add_axes([0.155, 0.165, 0.815, 0.80])
    axp.scatter(d, a_shuffle, s=20, c="#e07b1f", alpha=0.85, linewidths=0,
                zorder=3, label="shuffle")
    axp.scatter(d, a_metric, s=20, c="#2a9d8f", alpha=0.92, linewidths=0,
                zorder=4, label="metric")
    # metric trend line
    axp.plot([0.02, 0.98], [0.10 + 0.82 * 0.02, 0.10 + 0.82 * 0.98],
             color="#1c7d76", lw=1.4, zorder=5)
    axp.set_xlim(0, 1); axp.set_ylim(0, 1)
    axp.set_xticks([0, 0.5, 1.0]); axp.set_yticks([0, 0.5, 1.0])
    axp.set_xticklabels(["0", "", "max"], fontsize=8, color="#444")
    axp.set_yticklabels(["0", "", "late"], fontsize=8, color="#444")
    axp.set_xlabel("distance from source  $d$", fontsize=8.5, color="#333")
    axp.set_ylabel("recovered arrival  $\\hat{a}$", fontsize=8.5, color="#333")
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axp.spines[sp].set_color("#9aa0a6"); axp.spines[sp].set_linewidth(0.9)
    axp.tick_params(length=2.5, color="#9aa0a6")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")

def hist_png(w=440, h=156, dpi=150):
    """Panel-b delay histogram: integer conduction delays tau_ij = round(d_ij / v).
    Right-skewed (many short edges, few long)."""
    rng2 = np.random.default_rng(3)
    # sample distances on a lattice -> integer delays
    pts = rng2.random((400, 2))
    dd = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                  pts[:, None, 1] - pts[None, :, 1])
    dd = dd[np.triu_indices(len(pts), 1)]
    v = 0.14
    tau = np.round(dd / v).astype(int)
    tau = tau[(tau >= 1) & (tau <= 10)]
    counts = np.array([(tau == k).sum() for k in range(1, 11)], float)
    # bias toward a right-skewed profile (many short edges, few long)
    counts = counts * np.array([1.25, 1.18, 1.05, 0.92, 0.78, 0.62, 0.46,
                                0.32, 0.20, 0.12])
    counts /= counts.max()
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    axp = fig.add_axes([0.13, 0.16, 0.84, 0.80])
    cols = [vhex(k / 9.0) for k in range(10)]
    axp.bar(range(1, 11), counts, width=0.78, color=cols,
            edgecolor="white", linewidth=0.6, zorder=3)
    axp.set_xlim(0.3, 10.7); axp.set_ylim(0, 1.08)
    axp.set_xticks(range(1, 11, 2))
    axp.set_xticklabels([str(k) for k in range(1, 11, 2)], fontsize=8.5,
                        color="#444")
    axp.set_yticks([])
    # x-axis title is added in SVG below the image (avoids overflow)
    axp.set_ylabel("count", fontsize=9.0, color="#333")
    for sp in ("top", "right", "left"):
        axp.spines[sp].set_visible(False)
    axp.spines["bottom"].set_color("#9aa0a6")
    axp.spines["bottom"].set_linewidth(0.9)
    axp.tick_params(length=2.5, color="#9aa0a6")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")

def residual_png(w=460, h=300, dpi=150):
    """Velocity-inverse curve: reconstruction residual vs candidate velocity v.
    Sweeping v re-predicts delays tau=round(d/v) and re-simulates; the residual
    bottoms out at the true v_true (vertical guide + marker). Transparent bg."""
    v = np.linspace(0.4, 1.8, 240)
    v_true = 1.0
    # convex residual bowl, sharp min at v_true; slightly asymmetric (under-shooting
    # v is worse than over-shooting)
    resid = 0.06 + 1.05 * (np.log(v / v_true)) ** 2 + 0.18 * np.abs(v - v_true)
    resid += 0.10 * (1.0 / (v + 0.1))          # gentle low-v penalty
    resid = (resid - resid.min()) / (resid.max() - resid.min())
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    axp = fig.add_axes([0.165, 0.205, 0.805, 0.74])
    axp.plot(v, resid, color="#1c7d76", lw=2.0, zorder=4)
    vmin_i = int(np.argmin(resid))
    # vertical guide + marker at the recovered minimum (= true velocity)
    axp.axvline(v[vmin_i], color="#9aa0a6", lw=0.9, ls=(0, (4, 3)), zorder=2)
    axp.scatter([v[vmin_i]], [resid[vmin_i]], s=46, c="#e07b1f",
                edgecolors="#1c7d76", linewidths=1.2, zorder=6)
    axp.set_xlim(0.4, 1.8); axp.set_ylim(-0.04, 1.06)
    axp.set_xticks([v[vmin_i]]); axp.set_xticklabels([r"$v_{\mathrm{true}}$"],
                                                     fontsize=9.5, color="#1c7d76")
    axp.set_yticks([])
    axp.set_xlabel(r"candidate conduction velocity  $v$", fontsize=9.0,
                   color="#333")
    axp.set_ylabel(r"residual $\,r(v)$", fontsize=9.0, color="#333")
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axp.spines[sp].set_color("#9aa0a6"); axp.spines[sp].set_linewidth(0.9)
    axp.tick_params(length=2.5, color="#9aa0a6")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")

def raster_png(w=460, h=300, dpi=150):
    """Network-activity field (neuron x time) from the forward delayed-RNN. Units
    are sorted by distance from the source, so activity sweeps as a travelling
    front -- the signal the inverse step reads back. Smooth teal field, no speckle."""
    n_units, n_t = 56, 160
    tt = np.linspace(0, 1, n_t)[None, :]
    uu = np.linspace(0, 1, n_units)[:, None]          # 0 = near source, 1 = far
    # arrival time grows with distance; a bump sweeps past each unit, then echoes
    arr = 0.10 + 0.62 * uu                              # per-unit arrival time
    front = np.exp(-((tt - arr) ** 2) / (2 * 0.025 ** 2))
    echo = 0.45 * np.exp(-((tt - arr - 0.22) ** 2) / (2 * 0.04 ** 2))
    F = front + echo
    F = (F - F.min()) / (F.max() - F.min() + 1e-9)
    img = (cm.get_cmap("viridis")(F)[..., :3] * 255).astype("uint8")
    return Image.fromarray(img, "RGB")

def gate_png(w=150, h=100, dpi=140, kind="phi"):
    """Tiny activation/gating curve for panel d (nonlinearity phi or a gate)."""
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    axp = fig.add_axes([0.06, 0.10, 0.90, 0.84])
    xx = np.linspace(-3, 3, 200)
    if kind == "phi":
        yy = np.tanh(xx)
    else:
        yy = 1.0 / (1.0 + np.exp(-2.4 * xx))
        yy = yy * 2 - 1
    axp.plot(xx, yy, color="#2a9d8f", lw=1.8)
    axp.axhline(0, color="#cdd2d8", lw=0.7, zorder=0)
    axp.axvline(0, color="#cdd2d8", lw=0.7, zorder=0)
    axp.set_xlim(-3, 3); axp.set_ylim(-1.25, 1.25)
    axp.set_xticks([]); axp.set_yticks([])
    for sp in axp.spines.values():
        sp.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")

# Left-input-stack thumbnails for panel d: an embedding glyph (positions), a
# distance heatmap, a delay histogram. Each a transparent-bg thumbnail.
def lineplot_png(kind, w=300, h=120, dpi=150):
    """A small line-profile thumbnail on a tinted fill."""
    xx = np.linspace(0, 1, 260)
    if kind == "blue":
        yy = (0.55 + 0.40 * np.exp(-2.2 * xx) *
              np.cos(2 * np.pi * 9 * xx) + 0.05 * np.sin(2 * np.pi * 2 * xx))
        col, fill = "#3a56b0", "#d6e6fb"
    else:
        yy = (0.45 + 0.18 * np.sin(2 * np.pi * 1.7 * xx + 0.4)
              + 0.14 * np.sin(2 * np.pi * 3.3 * xx + 1.1)
              + 0.10 * np.cos(2 * np.pi * 2.2 * xx))
        col, fill = "#2f7d4f", "#d2f0d6"
    yy = (yy - yy.min()) / (yy.max() - yy.min() + 1e-9) * 0.86 + 0.06
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    axp = fig.add_axes([0, 0, 1, 1])
    axp.fill_between(xx, 0, yy, color=fill, zorder=1)
    axp.plot(xx, yy, color=col, lw=1.1, zorder=2)
    axp.set_xlim(0, 1); axp.set_ylim(0, 1.02)
    axp.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=dpi)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGBA")

def distmat_png(n=40):
    """A small distance-matrix heatmap d_ij (teal sequential)."""
    pts = np.linspace(0, 1, n)
    D = np.abs(pts[:, None] - pts[None, :])
    D = (D - D.min()) / (D.max() - D.min() + 1e-9)
    img = (VIR(D)[..., :3] * 255).astype("uint8")
    return Image.fromarray(img, "RGB")

def noisefield_png(n=200):
    """A dense red point cloud (the lattice edge population, no metric)."""
    fig = plt.figure(figsize=(2.0, 1.6), dpi=150)
    axp = fig.add_axes([0, 0, 1, 1])
    r = np.random.default_rng(5)
    axp.scatter(r.random(2600), r.random(2600), s=2.2, c="#d23b2e",
                alpha=0.8, linewidths=0)
    axp.set_xlim(0, 1); axp.set_ylim(0, 1); axp.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=150)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGBA")

# render the thumbnails once, to data URIs
URI_SCAT  = png_data_uri(scatter_png())
URI_HIST  = png_data_uri(hist_png())
URI_PHI   = png_data_uri(gate_png(kind="phi"))
URI_RESID = png_data_uri(residual_png())
URI_RASTER = png_data_uri(raster_png())
URI_LINEB = png_data_uri(lineplot_png("blue"))
URI_LINEG = png_data_uri(lineplot_png("green"))
URI_DMAT  = png_data_uri(distmat_png())
URI_NOISE = png_data_uri(noisefield_png())

# SVG authoring helpers
W, H = 1840, 1015     # narrow left column (a,b,c) + wide right panel d
S = []
def add(s): S.append(s)

def text(x, y, t, size=13, fill=INK, anchor="middle", weight="normal",
         style="normal", family="Helvetica, Arial, sans-serif", spacing=None,
         opacity=1.0):
    ls = f' letter-spacing="{spacing}"' if spacing is not None else ""
    add(f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}" font-style="{style}"{ls} '
        f'opacity="{opacity}">{t}</text>')

def rrect(x, y, w, h, r=10, fill="none", stroke=INK, sw=1.0, dash=None, op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke is not None else ' stroke="none"'
    add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'rx="{r}" ry="{r}" fill="{fill}"{st}{d} opacity="{op}"/>')

def line(x0, y0, x1, y1, stroke=INK, sw=1.0, dash=None, op=1.0, cap="round"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
        f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}"{d} '
        f'opacity="{op}"/>')

def circle(cx, cy, r, fill="white", stroke=INK, sw=1.0, op=1.0):
    add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>')

# Flat node fills: each former gradient name maps to a flat pastel fill.
NODE_FILL = {
    "gblue":   BLUE_F,
    "ggreen":  GREEN_F,
    "gwarm":   WARM_F,
    "gorange": ORANGE_F,
    "ggrey":   "#eceff2",
    "gred":    "#f7d8d0",
    "gpurple": "#ecdff0",
}

def neuron(cx, cy, r, grad="gblue", stroke=BLUE, sw=1.2):
    fill = NODE_FILL.get(grad, BLUE_F)
    add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>')

def cpath(d, stroke=INK, sw=1.4, fill="none", dash=None, marker=None, op=1.0,
          cap="round"):
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#{marker})"' if marker else ""
    add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linecap="{cap}" stroke-linejoin="round"{dd}{mk} opacity="{op}"/>')

def smooth_arrow(x0, y0, x1, y1, stroke=INK, sw=1.6, marker="arrow", bend=0.0,
                 op=1.0, inset=6.0):
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    ex, ey = x1 - ux * inset, y1 - uy * inset
    mx, my = (x0 + ex) / 2, (y0 + ey) / 2
    nx, ny = -uy, ux
    cx, cy = mx + nx * bend, my + ny * bend
    d = f"M {x0:.2f} {y0:.2f} Q {cx:.2f} {cy:.2f} {ex:.2f} {ey:.2f}"
    cpath(d, stroke=stroke, sw=sw, marker=marker, op=op)

def arc(x0, y0, x1, y1, rx, ry, stroke=INK, sw=1.2, marker=None, dash=None,
        op=1.0, sweep=0, large=0):
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#{marker})"' if marker else ""
    add(f'<path d="M {x0:.2f} {y0:.2f} A {rx:.2f} {ry:.2f} 0 {large} {sweep} '
        f'{x1:.2f} {y1:.2f}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linecap="round"{dd}{mk} opacity="{op}"/>')

def image(x, y, w, h, uri, r=6, stroke=INK, sw=1.0, smooth=False):
    cid = f"clip{len(S)}"
    add(f'<clipPath id="{cid}"><rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
        f'height="{h:.2f}" rx="{r}" ry="{r}"/></clipPath>')
    add(f'<image x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'preserveAspectRatio="none" href="{uri}" clip-path="url(#{cid})" '
        f'style="image-rendering:{"auto" if smooth else "pixelated"}"/>')
    if stroke is not None:
        rrect(x, y, w, h, r=r, fill="none", stroke=stroke, sw=sw)

def surface_strip(x, y, w, h, color, phase=0.0, freq=2.0, amp=0.62,
                  fill="#ffffff"):
    """A thin 1-D surface-profile strip (framed rect + one sinusoid) that caps an
    output heatmap. phase/freq/amp shape the wave: metric = near-monotone ramp,
    shuffle = busier non-monotone profile."""
    rrect(x, y, w, h, r=3, fill=fill, stroke=HAIR, sw=0.8)
    pts = []
    n = 64
    for i in range(n + 1):
        t = i / n
        xx = x + t * w
        # a smooth wave with a gentle overall trend; amp in units of strip height
        yv = 0.5 + 0.5 * amp * math.sin(2 * math.pi * freq * t + phase) \
             + 0.10 * amp * math.sin(2 * math.pi * (freq * 1.7) * t + phase * 1.3)  # amp in strip-height units
        yy = y + h - 4 - yv * (h - 8)
        pts.append(f"{xx:.1f},{yy:.1f}")
    add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>')

def tiny_net(cx, cy, w=46, h=34, layers=(2, 3, 2), node_r=3.4,
             grad="gblue", nstroke=BLUE, edge=HAIR):
    xs = [cx - w / 2 + w * i / (len(layers) - 1) for i in range(len(layers))]
    cols = []
    for li, nc in enumerate(layers):
        ys = [cy - h / 2 + h * (k + 0.5) / nc for k in range(nc)] if nc > 1 else [cy]
        cols.append([(xs[li], yy) for yy in ys])
    for a, b in zip(cols[:-1], cols[1:]):
        for ai, (xa, ya) in enumerate(a):
            for bi, (xb, yb) in enumerate(b):
                bold = ((ai + bi) % 3 == 0)
                line(xa, ya, xb, yb, stroke=GINK,
                     sw=0.9 if bold else 0.55, cap="butt")
    for col in cols:
        for (xx, yy) in col:
            add(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="{node_r:.2f}" '
                f'fill="#ffffff" stroke="{GINK}" stroke-width="1.0"/>')

def subnet_box(cx, cy, label=None, w=86, h=66, layers=(2, 3, 2), node_r=3.4,
               lab_size=9.5, lab_fill=INK):
    bx0, by0 = cx - w / 2, cy - h / 2
    if label is not None:
        text(cx, by0 - 7, label, size=lab_size, anchor="middle", fill=lab_fill)
    add(f'<rect x="{bx0:.2f}" y="{by0:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'rx="0" ry="0" fill="#ffffff" stroke="{GINK}" stroke-width="1.6" '
        f'stroke-dasharray="6 4"/>')
    tiny_net(cx, cy, w=w * 0.66, h=h * 0.62, layers=layers, node_r=node_r)
    return bx0, by0, w, h

def combine_node(cx, cy, r=11.5):
    # pale-green disc with a thin ink outline, plus an x glyph (circled-tensor look)
    add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" '
        f'fill="#dff0e1" stroke="{GINK}" stroke-width="1.6"/>')
    o = r * 0.707
    line(cx - o, cy - o, cx + o, cy + o, stroke=GINK, sw=1.6, cap="round")
    line(cx - o, cy + o, cx + o, cy - o, stroke=GINK, sw=1.6, cap="round")

# Module box: a thin dashed rounded rect grouping sub-parts, with a top-left
# label tab on a white knockout.
def module_box(x, y, w, h, label=None, ec=GREY, fill="#ffffff", sw=1.1,
               dash="5 4", r=12, lab_fill=None, lab_size=SZ_ANNOT + 0.5,
               op=1.0):
    rrect(x, y, w, h, r=r, fill=fill, stroke=ec, sw=sw, dash=dash, op=op)
    if label is not None:
        lf = lab_fill if lab_fill is not None else ec
        tw = 6.4 * len(label) + 12
        # knockout the dashed border under the tab
        add(f'<rect x="{x + 12:.2f}" y="{y - 7:.2f}" width="{tw:.2f}" '
            f'height="14" fill="#ffffff"/>')
        text(x + 12 + tw / 2, y + 3, label, size=lab_size, anchor="middle",
             weight="bold", fill=lf)

# Panel header: a bold letter + a regular title on one baseline, title 15px right.
def panel_header(x, y, letter, title):
    text(x, y, letter, size=15.5, anchor="start", weight="bold", fill=INK)
    text(x + 15, y, title, size=SZ_TITLE, anchor="start", weight="normal",
         fill=INK)

# Wiring spine: port dots, orthogonal connectors, and signal-name pills on the rail.
def port(x, y, r=3.0, col=SPINE):
    """No-op: standalone port dots are suppressed so unconnected ports never leave
    a dangling mark. Connected endpoints get their dot drawn inside `hwire`."""
    return

def port_dot(x, y, r=3.0):
    add(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="#ffffff" '
        f'stroke="{SPINE}" stroke-width="1.0"/>')

def rail_pill(xc, yc, label, math_label=None, col=INK, wide=False, maxw=None):
    """A compact single-line capsule naming the signal on the rail (word + math
    symbol on one baseline, white fill). Width is clamped to stay inside the
    gutter; `maxw` overrides the default GUT clamp for the wide panel-d rails."""
    s = SZ_ANNOT
    # width budget: word + symbol + tight padding
    wword = 5.6 * len(label)
    wsym = 12 if math_label is not None else 0
    gap = 6 if math_label is not None else 0
    wlab = wword + wsym + gap + 16
    wlab = min(wlab, (maxw if maxw is not None else GUT) - 18)   # clamp
    rrect(xc - wlab / 2, yc - 9, wlab, 18, r=9, fill="white",
          stroke=SPINE, sw=0.9)
    if math_label is not None:
        # centre the word+symbol pair around xc
        unit = wword + gap + wsym
        x_word = xc - unit / 2 + wword / 2
        x_sym = xc + unit / 2 - wsym / 2
        text(x_word, yc + 3.0, label, size=s, anchor="middle", fill=col)
        mtext(x_sym, yc + 3.4, math_label, size=s + 1.5, anchor="middle", fill=col)
    else:
        text(xc, yc + 3.0, label, size=s, anchor="middle", fill=col)

def hwire(x0, y0, x1, y1, label=None, math_label=None, mid=None, col=SPINE,
          marker="arrowSpine", pill_y=None):
    """Orthogonal H-V-H rail from (x0,y0) to (x1,y1), with a pill at the gutter mid-x."""
    mx = mid if mid is not None else (x0 + x1) / 2
    d = f"M {x0:.1f} {y0:.1f} H {mx:.1f} V {y1:.1f} H {x1:.1f}"
    cpath(d, stroke=col, sw=1.6, marker=marker, op=1.0, cap="round")
    port_dot(x0, y0)                              # terminal dot at the wire source
    if label is not None:
        py = pill_y if pill_y is not None else (y0 + y1) / 2
        rail_pill(mx, py, label, math_label=math_label)

# SVG header + reusable defs
add(f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'xmlns:xlink="http://www.w3.org/1999/xlink" '
    f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>')
# pastel spheres: light centre, gentle hue creep to the rim (defined but the flat
# NODE_FILL path is what actually renders nodes)
def radgrad(idn, light, dark):
    return (f'<radialGradient id="{idn}" cx="40%" cy="36%" r="78%">'
            f'<stop offset="0%" stop-color="#ffffff"/>'
            f'<stop offset="48%" stop-color="{light}"/>'
            f'<stop offset="100%" stop-color="{dark}" stop-opacity="0.55"/>'
            f'</radialGradient>')
add(radgrad("gblue",  BLUE_F,  BLUE))
add(radgrad("ggreen", GREEN_F, GREEN))
add(radgrad("gwarm",  WARM_F,  WARM))
add(radgrad("gorange",ORANGE_F, ORANGE))
add(radgrad("ggrey",  "#e7e9ec", "#6b7680"))
add(radgrad("gred",   "#fcd8d0", RED))
add(radgrad("gpurple","#e7d8ec", PURPLE))
add('<linearGradient id="gcombine" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0%" stop-color="#cdebcf"/>'
    '<stop offset="55%" stop-color="#aadcaf"/>'
    '<stop offset="100%" stop-color="#8ccb97"/></linearGradient>')
for mid, col in [("arrow", INK), ("arrowB", BLUE), ("arrowG", GREEN),
                 ("arrowW", WARM), ("arrowO", ORANGE), ("arrowR", RED),
                 ("arrowGrey", GREY), ("arrowP", PURPLE), ("arrowInk", INK),
                 ("arrowSpine", SPINE)]:
    add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="7.4" refY="5" '
        f'markerWidth="5.6" markerHeight="5.6" orient="auto-start-reverse">'
        f'<path d="M 0.6 1.7 L 8.6 5 L 0.6 8.3 Z" fill="{col}"/></marker>')
add('</defs>')
add(f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')

# LAYOUT
#   LEFT  : three small stacked panels a, b, c (problem setup).
#   RIGHT : one large panel d, left-to-right dataflow (input glyph -> subnet box ->
#           merge -> output heatmaps -> velocity-inverse -> constraint boxes).
DASH = "5 4"                      # the one dashed-border pattern
BORD = HAIR                       # the one neutral hairline colour for borders
SWB  = 1.1                        # the one thin border weight

def flow_arrow(x0, y0, x1, y1, col=INK, sw=1.5, label=None, lab_dy=-7,
               lab_size=SZ_ANNOT):
    """A straight dataflow arrow with one flush arrowhead."""
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    ex, ey = x1 - ux * 3.0, y1 - uy * 3.0
    cpath(f"M {x0:.2f} {y0:.2f} L {ex:.2f} {ey:.2f}", stroke=col, sw=sw,
          marker="arrowInk", cap="butt")
    if label is not None:
        text((x0 + x1) / 2, (y0 + y1) / 2 + lab_dy, label, size=lab_size,
             anchor="middle", fill=GREY)

def title_tab(x, y, w, label, ec=BORD, fill="#ffffff", lab_fill=INK,
              size=SZ_ANNOT + 0.5, weight="normal"):
    """A label on a white knockout, centred on the top dashed border of a module box."""
    tw = 6.6 * len(label) + 14
    add(f'<rect x="{x + w/2 - tw/2:.2f}" y="{y - 8:.2f}" width="{tw:.2f}" '
        f'height="15" fill="{fill}"/>')
    text(x + w / 2, y + 3, label, size=size, anchor="middle", weight=weight,
         fill=lab_fill)

def dmod(x, y, w, h, label=None, ec=BORD, fill="#ffffff", lab_fill=INK):
    """A dashed module box with an optional centred title-tab."""
    rrect(x, y, w, h, r=6, fill=fill, stroke=ec, sw=SWB, dash=DASH)
    if label is not None:
        title_tab(x, y, w, label, ec=ec, lab_fill=lab_fill)

# Panel-d box vocabulary:
#   solid_input_box : solid black-bordered input frame (left stack)
#   mlp_subnet      : dashed box w/ a title above + an MLP glyph inside
#   loss_box        : blue dashed box with a centred title
#   group_box       : the big green dashed group-enclosure with a top-edge title
INK_DASH = INK
GROUP_C  = "#5a8f5a"                 # group-enclosure green
LOSS_C   = "#2b3bd0"                 # loss-box blue

def solid_input_box(x, y, w, h):
    """Left-stack input frame: a solid thin black rounded rectangle."""
    rrect(x, y, w, h, r=4, fill="#ffffff", stroke=INK, sw=1.3)

def title_above(cx, top_y, label, fill=INK, size=SZ_LABEL):
    """A plain label sitting above its dashed box (no tab, no knockout)."""
    text(cx, top_y - 6, label, size=size, anchor="middle", fill=fill)

def mlp_subnet(cx, cy, w, h, label, ec=INK, layers=(3, 3), node_r=6.0,
               title_fill=INK):
    """A dashed rounded box, title above, MLP node-edge glyph inside."""
    x0, y0 = cx - w / 2, cy - h / 2
    title_above(cx, y0, label, fill=title_fill)
    add(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'rx="6" ry="6" fill="#ffffff" stroke="{ec}" stroke-width="1.4" '
        f'stroke-dasharray="6 4"/>')
    tiny_net(cx, cy, w=w * 0.60, h=h * 0.66, layers=layers, node_r=node_r)
    return x0, y0, w, h

def loss_box(x, y, w, h, label, ec=LOSS_C):
    """A blue dashed rounded box with a centred blue title."""
    rrect(x, y, w, h, r=6, fill="#ffffff", stroke=ec, sw=1.4, dash="6 4")
    text(x + w / 2, y + h / 2 + 4, label, size=SZ_ANNOT + 1.0, anchor="middle",
         fill=ec)

def group_box(x, y, w, h, label, ec=GROUP_C):
    """A big green dashed rounded rect, title centred on its top edge."""
    rrect(x, y, w, h, r=10, fill="none", stroke=ec, sw=1.3, dash="8 6")
    tw = 7.0 * len(label) + 16
    add(f'<rect x="{x + w/2 - tw/2:.2f}" y="{y - 9:.2f}" width="{tw:.2f}" '
        f'height="16" fill="#ffffff"/>')
    text(x + w / 2, y + 4, label, size=SZ_LABEL + 0.5, anchor="middle",
         fill=ec, style="italic")

# Column geometry
M_TOP   = 34                       # outer top margin
LX      = 40                       # left column x
LW      = 360                      # left column width (small panels)
GUT     = 34                       # gutter between left column and panel d
DX      = LX + LW + GUT            # panel-d left x
DW      = W - DX - 34              # panel-d width
PANEL_H = 270                      # height of each left panel
GAP_V   = (H - 2 * M_TOP - 3 * PANEL_H) / 2   # even vertical gap (a-b, b-c)

# left-panel y-tops
AY = M_TOP
BY = AY + PANEL_H + GAP_V
CY = BY + PANEL_H + GAP_V
ROW_H = PANEL_H                     # legacy alias used by panel bodies

# Panel a: spatial embedding -- units at 3D positions, a highlighted source, d_ij.
panel_header(LX, AY + 16, "a", "spatial embedding")
ax0, ay0, aw, ah = LX, AY + 28, LW, ROW_H - 28
# borderless node cloud inside the panel
np_rng = np.random.default_rng(4)
# place units on a tilted lattice to suggest 3D embedding
base = []
nx, ny = 4, 3
for i in range(nx):
    for j in range(ny):
        px = ax0 + 70 + i * 58 + j * 20
        py = ay0 + 34 + j * 52 - i * 6
        px += np_rng.normal(0, 3); py += np_rng.normal(0, 3)
        base.append((px, py))
# pick a source node (highlighted) and a target node for the d_ij measure
src_i = 5
tgt_i = 10
sx, sy = base[src_i]
tx, ty = base[tgt_i]
# faint connectivity edges to a few neighbours (structure, very light)
for k, (px, py) in enumerate(base):
    if k == src_i:
        continue
    dd = math.hypot(px - sx, py - sy)
    if dd < 110:
        line(sx, sy, px, py, stroke=HAIR, sw=0.7, op=0.85)
# the highlighted distance d_ij (source -> target) as a labelled measure line
line(sx, sy, tx, ty, stroke=BLUE, sw=1.6)
text((sx + tx) / 2 + 8, (sy + ty) / 2 - 7,
     '<tspan font-family="' + MATH + '" font-style="italic">d</tspan>'
     '<tspan font-family="' + MATH + '" font-size="8" baseline-shift="sub"'
     ' font-style="italic">ij</tspan>', size=SZ_LABEL + 1, fill=BLUE,
     anchor="middle")
# draw the units
for k, (px, py) in enumerate(base):
    if k == src_i:
        neuron(px, py, 9.5, grad="gorange", stroke=ORANGE, sw=1.8)
    elif k == tgt_i:
        neuron(px, py, 8.0, grad="gblue", stroke=BLUE, sw=1.4)
    else:
        neuron(px, py, 7.0, grad="gblue", stroke=BLUE, sw=1.2)
text(sx, sy - 16, "source", size=SZ_ANNOT, fill=ORANGE, anchor="middle")
# input / output ports
text(ax0 + 8, ay0 + 20, "input", size=SZ_ANNOT, fill=GREY, anchor="start")
flow_arrow(ax0 + 8, ay0 + 28, ax0 + 44, ay0 + 40, col=GREY, sw=1.2)
flow_arrow(ax0 + aw - 46, ay0 + ah - 30, ax0 + aw - 12, ay0 + ah - 18,
           col=GREY, sw=1.2)
text(ax0 + aw - 8, ay0 + ah - 8, "output", size=SZ_ANNOT, fill=GREY,
     anchor="end")
# axis hint (3D positions)
text(ax0 + 4, ay0 + ah - 6, "units at 3D positions",
     size=SZ_ANNOT, fill=GREY, anchor="start", style="italic")

# Panel b: distance sets the delay -- tau_ij = round(d_ij / v), plus a delay histogram.
panel_header(LX, BY + 16, "b", "distance sets the delay")
bx0, by0 = LX, BY + 30
bh = ROW_H - 30
# the equation, in serif-italic maths
eq_y = by0 + 26
mathspan('&#964;', x=bx0 + 18, y=eq_y, size=16, fill=INK, anchor="start")
mtext(bx0 + 28, eq_y + 5, "ij", size=9.5, fill=INK, anchor="start")
text(bx0 + 48, eq_y, "=", size=15, fill=INK, anchor="start")
text(bx0 + 64, eq_y, "round", size=13.5, fill=INK, anchor="start",
     style="italic", family=MATH)
text(bx0 + 108, eq_y, "(", size=16, fill=GREY, anchor="start")
# d_ij / v fraction
fr_x = bx0 + 120
mathspan("d", x=fr_x + 10, y=eq_y - 7, size=14, fill=INK, anchor="middle")
mtext(fr_x + 18, eq_y - 3, "ij", size=8.5, fill=INK, anchor="middle")
line(fr_x - 2, eq_y - 1, fr_x + 28, eq_y - 1, stroke=INK, sw=1.0)
mathspan("v", x=fr_x + 13, y=eq_y + 13, size=14, fill=INK, anchor="middle")
text(fr_x + 36, eq_y, ")", size=16, fill=GREY, anchor="start")
text(bx0 + 18, eq_y + 24, "geometry sets the integer conduction delay",
     size=SZ_ANNOT, fill=GREY, anchor="start", style="italic")
# delay histogram (right-skewed)
hist_w, hist_h = 250, 100
hist_y = by0 + 56
image(bx0 + 14, hist_y, hist_w, hist_h, URI_HIST, r=4, stroke=None)
text(bx0 + 14 + hist_w / 2, hist_y + hist_h + 16,
     "delay  &#964;  (lattice steps)", size=SZ_ANNOT, fill=GREY,
     anchor="middle")

# Panel c: metric consistency -- triangle inequality tau_AC <= tau_AB + tau_BC
# holds for the true geometry (green check), breaks under entry-shuffle (red X).
panel_header(LX, CY + 16, "c", "metric consistency")
cx0, cy0 = LX, CY + 30
ch = ROW_H - 30
# two mini triangles: true (holds) and shuffle (breaks)
def mini_triangle(ox, oy, holds, edge_col, tag, tag_col):
    # vertices A (top), B (lower-left), C (lower-right)
    A = (ox + 56, oy + 6)
    B = (ox + 14, oy + 78)
    C = (ox + 104, oy + 78)
    # edges
    line(*A, *B, stroke=edge_col, sw=1.8)
    line(*B, *C, stroke=edge_col, sw=1.8)
    line(*A, *C, stroke=edge_col, sw=1.8 if holds else 1.4,
         dash=None if holds else "4 3")
    for (vx, vy), lab in [(A, "A"), (B, "B"), (C, "C")]:
        neuron(vx, vy, 8.5, grad="gblue", stroke=BLUE, sw=1.3)
        text(vx, vy + 3.5, lab, size=SZ_ANNOT, fill=INK, anchor="middle",
             weight="bold")
    # verdict glyph (check or cross)
    gx, gy = ox + 59, oy + 50
    if holds:
        cpath(f"M {gx-7:.1f} {gy:.1f} L {gx-2:.1f} {gy+6:.1f} "
              f"L {gx+8:.1f} {gy-7:.1f}", stroke=GREEN, sw=2.4)
    else:
        line(gx - 7, gy - 7, gx + 7, gy + 7, stroke=RED, sw=2.4)
        line(gx - 7, gy + 7, gx + 7, gy - 7, stroke=RED, sw=2.4)
    text(ox + 59, oy + 96, tag, size=SZ_ANNOT, fill=tag_col, anchor="middle")

mini_triangle(cx0 + 6, cy0 + 4, True, GREEN, "true geometry", GREEN)
mini_triangle(cx0 + 196, cy0 + 4, False, ORANGE, "entry-shuffle", ORANGE)
# the inequality, below the triangles
ineq_y = cy0 + 130
mathspan("&#964;", x=cx0 + 14, y=ineq_y, size=12.5, fill=INK, anchor="start")
mtext(cx0 + 22, ineq_y + 4, "AC", size=8, fill=INK, anchor="start")
text(cx0 + 40, ineq_y, "&#8804;", size=12.5, fill=INK, anchor="start")
mathspan("&#964;", x=cx0 + 56, y=ineq_y, size=12.5, fill=INK, anchor="start")
mtext(cx0 + 64, ineq_y + 4, "AB", size=8, fill=INK, anchor="start")
text(cx0 + 82, ineq_y, "+", size=12.5, fill=INK, anchor="start")
mathspan("&#964;", x=cx0 + 96, y=ineq_y, size=12.5, fill=INK, anchor="start")
mtext(cx0 + 104, ineq_y + 4, "BC", size=8, fill=INK, anchor="start")
text(cx0 + 128, ineq_y, "(triangle inequality)", size=SZ_ANNOT, fill=GREY,
     anchor="start", style="italic")

# thin vertical rule separating the left setup column from panel d
line(DX - GUT / 2, M_TOP + 6, DX - GUT / 2, H - 24, stroke=HAIR, sw=0.8,
     dash="2 5")

# Panel d: forward model + velocity inverse, a left-to-right dataflow.
#   * left input stack (3 boxes): positions p_i / distances d_ij / delays tau_ij
#   * green group-enclosure over the subnet column + merge nodes
#   * subnet column: delay-coupled dynamics / recurrent W / leak+nonlinearity,
#     plus a 'velocity inverse' subnet
#   * two circled-x merge nodes; a central network-activity field
#   * output heatmaps (true wavefront / scrambled)
#   * blue loss/constraint boxes: recovered metric / residual r(v) / multilateration
panel_header(DX, M_TOP + 16, "d", "forward model and velocity inverse")
PY0 = M_TOP + 44                       # below the header baseline
PYB = H - 22                           # panel-d bottom
PH  = PYB - PY0

# column x-anchors across panel d
ix0   = DX + 10                        # left input-stack left edge
iw    = 150                            # input-box width
sub_x = DX + 248                       # subnet column centre-x
sub_w = 150                            # subnet box width
mg1_x = DX + 400                       # lower merge (x) node
diff_x= DX + 500                       # diffeomorphism-slot subnet centre-x
mg2_x = DX + 610                       # upper merge (x) node
hm_x  = DX + 700                       # output-heatmaps left edge
hm_w  = 200                            # heatmap width
loss_x= DX + 980                       # blue loss-box column left edge

# input/subnet columns start below PY0 so each subnet title (above its box) clears
# the group-title on the enclosure's top edge
COL_TOP = PY0 + 30
PHc = PYB - COL_TOP                     # column-stack height

# green group-enclosure
grp_x = sub_x - sub_w / 2 - 26
grp_y = PY0
grp_w = (hm_x - 24) - grp_x
grp_h = (PY0 + PH * 0.76) - grp_y
group_box(grp_x, grp_y, grp_w, grp_h, "forward model (delay-coupled RNN)")

# left input stack: 3 solid-bordered boxes, one per geometry signal
ig = 22                                # gap between stacked input boxes
ih = (PHc - 2 * ig) / 3                # each input box height (fills full panel)
in_ys = [COL_TOP + k * (ih + ig) for k in range(3)]

# (input-1) positions p_i  -- a small spatial-embedding glyph
iy = in_ys[0]
solid_input_box(ix0, iy, iw, ih)
exc, eyc = ix0 + iw / 2, iy + ih / 2 - 4
enodes = [(exc - 2, eyc - 30), (exc - 40, eyc + 4), (exc + 38, eyc - 6),
          (exc - 14, eyc + 34), (exc + 28, eyc + 30)]
hub = enodes[0]
for (px, py) in enodes[1:]:
    dd = math.hypot(px - hub[0], py - hub[1])
    line(hub[0], hub[1], px, py, stroke=vhex(min(1.0, dd / 80)),
         sw=1.4 + 1.0 * dd / 80)
for k, (px, py) in enumerate(enodes):
    g = "gorange" if k == 0 else "gblue"; sc = ORANGE if k == 0 else BLUE
    neuron(px, py, 8.0 if k == 0 else 6.4, grad=g, stroke=sc, sw=1.3)
text(ix0 + iw / 2, iy + ih - 10,
     '<tspan font-style="italic">positions  p</tspan>'
     '<tspan font-size="8" baseline-shift="sub" font-style="italic">i</tspan>',
     size=SZ_LABEL, fill=INK, anchor="middle", family=MATH)

# (input-2) distances d_ij  -- a small distance-matrix heatmap
iy = in_ys[1]
solid_input_box(ix0, iy, iw, ih)
dm_s = min(iw - 28, ih - 36)
image(ix0 + (iw - dm_s) / 2, iy + 12, dm_s, dm_s, URI_DMAT, r=3, stroke=BORD,
      sw=0.8, smooth=True)
text(ix0 + iw / 2, iy + ih - 8,
     '<tspan font-style="italic">distances  d</tspan>'
     '<tspan font-size="8" baseline-shift="sub" font-style="italic">ij</tspan>',
     size=SZ_LABEL, fill=INK, anchor="middle", family=MATH)

# (input-3) delays tau_ij  -- the delay histogram thumbnail
iy = in_ys[2]
solid_input_box(ix0, iy, iw, ih)
image(ix0 + 14, iy + 16, iw - 28, ih - 44, URI_HIST, r=3, stroke=None,
      smooth=True)
text(ix0 + iw / 2, iy + ih - 8,
     '<tspan font-style="italic">delays  &#964;</tspan>'
     '<tspan font-size="8" baseline-shift="sub" font-style="italic">ij</tspan>',
     size=SZ_LABEL, fill=INK, anchor="middle", family=MATH)

# subnet column: 3 dashed MLP boxes
sw_, sh_ = sub_w, (PHc - 2 * ig) / 3
s_ys = [COL_TOP + k * (sh_ + ig) + sh_ / 2 for k in range(3)]
mlp_subnet(sub_x, s_ys[0], sw_, sh_ - 6, "delay-coupled dynamics")
mlp_subnet(sub_x, s_ys[1], sw_, sh_ - 6, "recurrent weights  W", layers=(3, 2))
mlp_subnet(sub_x, s_ys[2], sw_, sh_ - 6, "leak + nonlinearity", ec=GROUP_C,
           title_fill=GROUP_C, layers=(2, 3))

# wires: each input box -> its subnet
for k in range(3):
    flow_arrow(ix0 + iw, s_ys[k], sub_x - sw_ / 2, s_ys[k], col=INK, sw=1.4)

# lower merge: subnets 2&3 -> central activity field
mg1_y = (s_ys[1] + s_ys[2]) / 2
combine_node(mg1_x, mg1_y, r=12)
flow_arrow(sub_x + sw_ / 2, s_ys[1], mg1_x - 13, mg1_y - 6, col=INK, sw=1.3)
flow_arrow(sub_x + sw_ / 2, s_ys[2], mg1_x - 13, mg1_y + 6, col=INK, sw=1.3)

# central network-activity field, below the velocity-inverse subnet
fld_w, fld_h = 150, 100
fld_cx = diff_x                        # centre the field under the inverse subnet
fld_x0 = fld_cx - fld_w / 2
fld_y = mg1_y - fld_h / 2
image(fld_x0, fld_y, fld_w, fld_h, URI_RASTER, r=4, stroke=INK, sw=1.1,
      smooth=True)
text(fld_cx, fld_y + fld_h + 14,
     '<tspan font-style="italic">activity  h</tspan>'
     '<tspan font-size="8" baseline-shift="sub" font-style="italic">i</tspan>'
     '<tspan font-style="italic">(t)</tspan>', size=SZ_ANNOT, fill=INK,
     anchor="middle", family=MATH)
flow_arrow(mg1_x + 12, mg1_y, fld_x0 - 6, fld_y + fld_h / 2, col=INK, sw=1.3)

# velocity-inverse subnet: the activity field feeds up into it; it feeds the top merge
diff_h = sh_ - 22
diff_y = (s_ys[0] + mg1_y) / 2 + 6     # between subnet-1 and the field
mlp_subnet(diff_x, diff_y, 138, diff_h, "velocity inverse", layers=(2, 3, 2),
           node_r=5.0)

# upper merge: subnet-1 (dynamics) & velocity-inverse -> output heatmaps
mg2_y = s_ys[0]
combine_node(mg2_x, mg2_y, r=12)
flow_arrow(sub_x + sw_ / 2, s_ys[0], mg2_x - 13, mg2_y, col=INK, sw=1.3)
# velocity-inverse subnet -> up into the top merge
cpath(f"M {diff_x:.1f} {diff_y - diff_h/2:.1f} V {mg2_y + 12:.1f} "
      f"H {mg2_x - 1:.1f}", stroke=INK, sw=1.3, marker="arrowInk", cap="round")
# central field -> up into the velocity-inverse subnet
flow_arrow(fld_cx, fld_y, diff_x, diff_y + diff_h / 2, col=INK, sw=1.3)

# output heatmaps (top-right): true wavefront / scrambled
hm_h = 116
hm_gap = 34
hm_y_top = PY0 + 6
hm_y_bot = hm_y_top + hm_h + hm_gap
image(hm_x, hm_y_top, hm_w, hm_h, png_data_uri(field_png("ordered")), r=4,
      stroke=INK, sw=1.1, smooth=True)
text(hm_x + hm_w / 2, hm_y_top + hm_h + 14, "network activity (true)",
     size=SZ_ANNOT, fill=TEAL, anchor="middle")
image(hm_x, hm_y_bot, hm_w, hm_h, png_data_uri(field_png("shuffle")), r=4,
      stroke=INK, sw=1.1, smooth=True)
text(hm_x + hm_w / 2, hm_y_bot + hm_h + 14, "scrambled (entry-shuffle)",
     size=SZ_ANNOT, fill=ORANGE, anchor="middle")
# merge -> the two heatmaps
flow_arrow(mg2_x + 12, mg2_y, hm_x - 6, hm_y_top + hm_h / 2, col=INK, sw=1.3)
flow_arrow(mg2_x + 12, mg2_y, hm_x - 6, hm_y_bot + hm_h / 2, col=INK, sw=1.3)

# blue loss/constraint boxes (right + bottom)
lw_ = DW - (loss_x - DX) - 8
lw_ = max(lw_, 200)
# recovered metric
rm_y = hm_y_top + 24
rm_h = 150
loss_box(loss_x, rm_y, lw_, rm_h, "")
text(loss_x + lw_ / 2, rm_y + 14, "recovered metric", size=SZ_ANNOT + 1.0,
     anchor="middle", fill=LOSS_C)
image(loss_x + 18, rm_y + 24, lw_ - 36, rm_h - 40, URI_SCAT, r=3, stroke=None,
      smooth=True)

# residual r(v): the velocity-inverse diagnostic
rv_y = rm_y + rm_h + 30
rv_h = 132
loss_box(loss_x, rv_y, lw_, rv_h, "")
text(loss_x + lw_ / 2, rv_y + 14, "residual  r(v)", size=SZ_ANNOT + 1.0,
     anchor="middle", fill=LOSS_C)
image(loss_x + 16, rv_y + 22, lw_ - 32, rv_h - 36, URI_RESID, r=3, stroke=None,
      smooth=True)

# multilateration (bottom, wide)
ml_y = rv_y + rv_h + 26
ml_h = PYB - ml_y
ml_w = lw_
loss_box(loss_x, ml_y, ml_w, ml_h, "")
text(loss_x + ml_w / 2, ml_y + 14, "multilateration", size=SZ_ANNOT + 1.0,
     anchor="middle", fill=LOSS_C)

def ml_sketch(ox, oy, ww, hh, ok):
    """Three range-circles around a common source, clipped to the sub-cell.
    ok -> circles meet (green check); shuffle -> radii mismatch, circles miss (red X)."""
    col = GREEN if ok else ORANGE
    cid = f"mlclip{len(S)}"
    add(f'<clipPath id="{cid}"><rect x="{ox:.1f}" y="{oy:.1f}" '
        f'width="{ww:.1f}" height="{hh:.1f}"/></clipPath>')
    add(f'<g clip-path="url(#{cid})">')
    spx, spy = ox + ww * 0.50, oy + hh * 0.50
    # anchors close to the source so the circles stay compact and inside
    a_r = min(ww, hh) * 0.30
    cxs = [(spx - a_r, spy - a_r * 0.55),
           (spx + a_r, spy - a_r * 0.55),
           (spx, spy + a_r)]
    base = [math.hypot(spx - a, spy - b) for a, b in cxs]
    if ok:
        radii = base
    else:
        radii = [r + (a_r * 0.30 if i != 1 else -a_r * 0.35)
                 for i, r in enumerate(base)]
    for (a, b), r in zip(cxs, radii):
        add(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{r:.1f}" fill="none" '
            f'stroke="{col}" stroke-width="1.1" opacity="0.9"/>')
        neuron(a, b, 3.0, grad="gblue", stroke=BLUE, sw=1.0)
    if ok:
        neuron(spx, spy, 4.2, grad="ggreen", stroke=GREEN, sw=1.4)
    add('</g>')
    vx, vy = ox + ww * 0.5, oy + hh - 2
    if ok:
        cpath(f"M {vx-7:.1f} {vy-3:.1f} L {vx-2:.1f} {vy+3:.1f} "
              f"L {vx+8:.1f} {vy-9:.1f}", stroke=GREEN, sw=2.2)
    else:
        line(vx - 6, vy - 9, vx + 6, vy + 3, stroke=RED, sw=2.2)
        line(vx - 6, vy + 3, vx + 6, vy - 9, stroke=RED, sw=2.2)

mlw = (ml_w - 18) / 2
ml_cell_h = ml_h - 52
ml_sketch(loss_x + 6, ml_y + 24, mlw, ml_cell_h, True)
ml_sketch(loss_x + 12 + mlw, ml_y + 24, mlw, ml_cell_h, False)
text(loss_x + 6 + mlw / 2, ml_y + ml_h - 8, "true metric", size=7.4,
     fill=GREEN, anchor="middle")
text(loss_x + 12 + mlw + mlw / 2, ml_y + ml_h - 8, "shuffle", size=7.4,
     fill=ORANGE, anchor="middle")

# wiring into the loss column
# top heatmap -> recovered metric
flow_arrow(hm_x + hm_w, hm_y_top + hm_h / 2, loss_x - 6, rm_y + rm_h / 2,
           col=INK, sw=1.3)
# velocity-inverse subnet -> residual r(v) via the bottom gutter
gutter_y = PYB - 6
cpath(f"M {diff_x:.1f} {diff_y + diff_h/2:.1f} V {gutter_y:.1f} "
      f"H {loss_x - 18:.1f} V {rv_y + rv_h/2:.1f} H {loss_x - 6:.1f}",
      stroke=INK, sw=1.3, marker="arrowInk", cap="round")
# scrambled heatmap -> multilateration
cpath(f"M {hm_x + hm_w:.1f} {hm_y_bot + hm_h/2:.1f} H {loss_x - 34:.1f} "
      f"V {ml_y + (ml_h-52)/2 + 24:.1f} H {loss_x - 6:.1f}", stroke=INK, sw=1.3,
      marker="arrowInk", cap="round")

# write SVG, render to PNG with resvg, then PDF
add('</svg>')
svg = "\n".join(S)
svg_path = os.path.join(FIGDIR, "figM0_schematic.svg")
with open(svg_path, "w") as f:
    f.write(svg)

import resvg_py
SCALE = 2.5
png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=int(W * SCALE),
                                  height=int(H * SCALE)))
png_path = os.path.join(FIGDIR, "figM0_schematic.png")
with open(png_path, "wb") as f:
    f.write(png)
print("wrote", svg_path)
print("wrote", png_path, "(", len(png), "bytes )")

pdf_path = os.path.join(FIGDIR, "figM0_schematic.pdf")
Image.open(png_path).convert("RGB").save(pdf_path, "PDF", resolution=200.0)
print("wrote", pdf_path)
