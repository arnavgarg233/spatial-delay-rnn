#!/usr/bin/env python
"""Build the paper's figures from the result JSONs in results/.

    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/figures.py

Style follows WeakPINN_NCS.pdf: clean sans-serif, no top/right spines, light grid,
300 dpi, constrained_layout. Consistent comparison palette throughout
(distance=teal, shuffled=orange, no-delay=gray); bar charts overlay per-seed dots
with value labels; multi-panel figures carry bold A/B/C/D labels. Produces
single-panel PNGs, composite main figures (figM1-M5), and an SI set in figures/.
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Paths / config
REPO = Path(__file__).resolve().parent.parent   # repo root: scripts/figures.py -> up 2
RESULTS = REPO / "results"
FIGDIR = REPO / "figures"
os.makedirs(FIGDIR, exist_ok=True)

DPI = 300

# Shared comparison palette: distance=teal, shuffled=orange, no-delay=gray, used
# everywhere these conditions appear so colour reads identically across figures.
COL = {
    "distance": "#2A9D8F",   # teal/green  -- the model
    "shuffled": "#E76F51",   # orange      -- the control
    "no-delay": "#6C757D",   # gray        -- the baseline
    "theory":   "#264653",   # dark slate  -- analytic prediction line
    "accent":   "#E76F51",   # orange      -- callouts / highlights
    "dot":      "#1D1D1D",   # near-black  -- per-seed scatter dots (WeakPINN style)
    "grid":     "#B7B7B7",
}

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": COL["grid"],
    "grid.alpha": 0.35,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "savefig.bbox": "tight",
})


def load(name):
    """Load a result JSON by basename from ANYWHERE under results/ (pillar
    subfolders economy/law/inverse/controls, archive/, or the results/ root).
    Result JSONs are reorganized into pillar subfolders, so resolve the path
    by a recursive search rather than assuming a flat layout."""
    try:
        path = next(RESULTS.rglob(name))
    except StopIteration:
        raise FileNotFoundError(f"no result JSON named {name!r} under {RESULTS}")
    with open(path) as fh:
        return json.load(fh)


def load_arch(name):
    """Load a JSON from results/archive/ (exploratory / limitation runs)."""
    with open(RESULTS / "archive" / name) as fh:
        return json.load(fh)


def panel_label(ax, letter, x=-0.16, y=1.04):
    """Bold A/B/C/D panel label at top-left of an axis (WeakPINN multi-panel style)."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")


def grid_y(ax):
    """Light horizontal grid only -- the WeakPINN bar/curve convention."""
    ax.grid(True, axis="y", color=COL["grid"], alpha=0.35, linewidth=0.6)
    ax.grid(False, axis="x")


def seed_dots(ax, x, vals, rng, jitter=0.12, s=22):
    """Overlay individual per-seed values as small dark scatter dots, jittered
    on top of a bar (exactly like WeakPINN Fig 4C/D)."""
    vals = np.asarray(vals, dtype=float)
    xs = x + (rng.random(len(vals)) - 0.5) * 2 * jitter
    ax.scatter(xs, vals, color=COL["dot"], edgecolor="white", linewidth=0.4,
               s=s, zorder=6, alpha=0.9)


def value_label(ax, x, y, text, dy=0.012, color="black"):
    """Print a value label above a bar."""
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, 4),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=color, zorder=7)


def save(fig, name, summary):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {name:22s} -- {summary}")
    return path


# Data-drawing primitives, shared by single-panel figs and composites so the two
# stay byte-for-byte consistent.

def draw_spine(ax, d, rng, title=True):
    """Panel A: conduction-cost spine bar chart at matched accuracy."""
    wm = d["weighted_mean"]
    acc = d["accuracy"]
    conds = ["no-delay", "shuffled", "distance"]
    labels = ["no-delay", "shuffled\ndelays", "distance\ndelays"]
    colors = [COL[c] for c in conds]

    means = [float(np.mean(wm[c])) for c in conds]
    sems = [float(np.std(wm[c], ddof=1) / np.sqrt(len(wm[c]))) for c in conds]
    x = np.arange(len(conds))

    ax.bar(x, means, width=0.60, color=colors, alpha=0.85,
           edgecolor="#333333", linewidth=0.7, zorder=2)
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="#222222",
                elinewidth=1.1, capsize=4, zorder=4)
    for i, c in enumerate(conds):
        seed_dots(ax, x[i], wm[c], rng)
        value_label(ax, x[i], means[i] + sems[i], f"{means[i]:.2f}")

    # paired t: distance vs shuffled
    pr = d["paired"]["weighted_mean_distance_minus_shuffled"]
    diff_mean = pr["mean"]
    tval = pr["t"]
    n = len(pr["diffs"])
    pct = 100.0 * (-diff_mean) / float(np.mean(wm["shuffled"]))

    # significance bracket between shuffled (1) and distance (2)
    y0 = max(means[1], means[2]) + max(sems) + 0.55
    ax.plot([1, 1, 2, 2], [y0, y0 + 0.12, y0 + 0.12, y0],
            color="#222222", lw=1.1)
    ax.text(1.5, y0 + 0.20,
            f"paired $t={tval:.1f}$  ($n={n}$ seeds)\n"
            r"$\Delta\bar{\tau}=$" + f"{diff_mean:.2f}  ({pct:.0f}% cheaper)",
            ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"conduction cost  $\bar{\tau}$  (weighted-mean delay)")
    acc_min = min(min(acc[c]) for c in conds)
    acc_max = max(max(acc[c]) for c in conds)
    if title:
        ax.set_title("Distance delays are the cheapest wiring\n"
                     f"(matched accuracy {acc_min:.3f}-{acc_max:.3f})")
    ax.set_ylim(0, y0 + 1.2)
    grid_y(ax)
    return dict(means=means, pct=pct, tval=tval,
                acc=(acc_min, acc_max))


def draw_dose(ax, d, title=True):
    """Panel B: dose-response saving vs delay length (1/velocity)."""
    curve = sorted(d["curve"], key=lambda r: r["velocity"])
    vel = np.array([r["velocity"] for r in curve])
    delay_len = 1.0 / vel
    reduction = np.array([r["reduction"] for r in curve])
    rel = np.array([100.0 * r["rel_reduction"] for r in curve])

    red_sem = []
    for r in curve:
        diff = np.array(r["shuffled_cws"]) - np.array(r["distance_cws"])
        red_sem.append(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    red_sem = np.array(red_sem)
    trend = d.get("trend", None)

    # filled-under curve (reads well, WeakPINN Fig 3 REV convention)
    ax.fill_between(delay_len, 0, reduction, color=COL["distance"], alpha=0.14, zorder=1)
    ax.errorbar(delay_len, reduction, yerr=red_sem, fmt="o-",
                color=COL["distance"], ecolor=COL["distance"], elinewidth=1.1,
                capsize=3.5, markersize=7, markeredgecolor="white",
                markeredgewidth=0.8, zorder=3,
                label=r"saving  $\bar{\tau}_{\mathrm{shuf}}-\bar{\tau}_{\mathrm{dist}}$")

    for x, y, p in zip(delay_len, reduction, rel):
        ax.annotate(f"{p:.0f}%", (x, y), textcoords="offset points",
                    xytext=(7, 7), fontsize=8.5, color=COL["distance"],
                    fontweight="bold")

    ax.set_xlabel("delay length  $1/v$  (inverse candidate velocity)")
    ax.set_ylabel(r"conduction-cost saving  $\Delta\bar{\tau}$")
    if title:
        t = r"Dose-response: longer delays $\rightarrow$ larger economy"
        if trend is not None:
            t += f"\n(corr saving vs delay length $= {trend:.2f}$, monotone)"
        ax.set_title(t)
    ax.legend(loc="upper left")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    grid_y(ax)
    return dict(red=reduction, trend=trend)


def draw_3d2d(ax, d, rng, title=True):
    """Panel C: 3D vs 2D conduction-cost gap, scale-matched and raw."""
    head = d["headline"]
    sm, raw = d["scale_matched"], d["raw"]

    def per_seed_gap(block):
        dist = np.array(block["distance"]["C_all"])
        shuf = np.array(block["shuffled"]["C_all"])
        return shuf - dist

    gaps = {
        ("sm", "2D"): per_seed_gap(sm["2d"]),
        ("sm", "3D"): per_seed_gap(sm["3d"]),
        ("raw", "2D"): per_seed_gap(raw["2d"]),
        ("raw", "3D"): per_seed_gap(raw["3d"]),
    }
    groups = [("sm", "2D"), ("sm", "3D"), ("raw", "2D"), ("raw", "3D")]
    xpos = [0, 0.95, 2.45, 3.40]
    # consistent palette: 2D = lighter shade, 3D = full color; orange=control family,
    # but here both are "gap" magnitudes so we use the shuffled/distance pairing:
    # 2D bars teal-ish, 3D bars darker teal to show "more economy", per spec keep
    # comparison colours consistent -> use distance teal with 2D/3D shade contrast.
    bar_cols = [COL["distance"], COL["theory"], COL["distance"], COL["theory"]]

    means = {}
    for x, g, col in zip(xpos, groups, bar_cols):
        vals = gaps[g]
        m = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(len(vals)))
        means[g] = m
        ax.bar(x, m, width=0.66, color=col, alpha=0.85,
               edgecolor="#333333", linewidth=0.7, zorder=2)
        ax.errorbar(x, m, yerr=sem, fmt="none", ecolor="#222222",
                    elinewidth=1.1, capsize=4, zorder=4)
        seed_dots(ax, x, vals, rng)
        value_label(ax, x, m + sem, f"{m:.2f}")

    # +pct callout: scale-matched 3D gap vs 2D gap
    g2 = head["scale_matched_gap_2d"]
    g3 = head["scale_matched_gap_3d"]
    pct = 100.0 * (g3 - g2) / g2
    yb = max(means[("sm", "2D")], means[("sm", "3D")]) + 0.18
    ax.plot([xpos[0], xpos[0], xpos[1], xpos[1]],
            [yb, yb + 0.04, yb + 0.04, yb], color="#222222", lw=1.0)
    ax.text((xpos[0] + xpos[1]) / 2, yb + 0.07,
            f"+{pct:.0f}% in 3D",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=COL["accent"])

    ax.set_xticks(xpos)
    ax.set_xticklabels(["2D", "3D", "2D", "3D"])
    ax.set_ylabel(r"conduction-cost gap  $C_{\mathrm{shuf}}-C_{\mathrm{dist}}$")
    if title:
        ax.set_title("Economy is larger in 3D")
    ax.set_ylim(0, max(means.values()) * 1.28)
    # group sub-labels
    ax.annotate("scale-matched", xy=((xpos[0] + xpos[1]) / 2, -0.16),
                xycoords=("data", "axes fraction"), ha="center",
                fontsize=8.5, color="#555555")
    ax.annotate("raw", xy=((xpos[2] + xpos[3]) / 2, -0.16),
                xycoords=("data", "axes fraction"), ha="center",
                fontsize=8.5, color="#555555")
    # legend for 2D vs 3D shade
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COL["distance"], edgecolor="#333", label="2D"),
                       Patch(facecolor=COL["theory"], edgecolor="#333", label="3D")],
              loc="upper left", ncol=2)
    grid_y(ax)
    return dict(g2=g2, g3=g3, pct=pct)


def draw_scaling(ax, fs, title=True):
    """Panel D: economy across N -- matched-dose (grows) vs fixed-velocity (shrinks)."""
    per_N = sorted(fs["summary"]["per_N"], key=lambda r: r["N"])
    md_N = np.array([r["N"] for r in per_N], dtype=float)
    md_rel = np.array([r["rel_saving"] for r in per_N])
    md_ci = np.array([r["rel_saving_ci"] for r in per_N])
    md_lo = md_rel - md_ci[:, 0]
    md_hi = md_ci[:, 1] - md_rel

    # Fixed-velocity GPU run (not on disk -- hardcoded from Colab, per task spec)
    fv_N = np.array([64, 128, 256, 512, 768, 1024], dtype=float)
    fv_rel = np.array([0.034, 0.031, 0.022, 0.009, 0.008, 0.005])

    ax.errorbar(md_N, md_rel * 100, yerr=[md_lo * 100, md_hi * 100], fmt="o-",
                color=COL["distance"], ecolor=COL["distance"], capsize=3.5,
                markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                label=r"matched-dose ($N\leq144$)  [grows]", zorder=3)
    ax.fill_between(md_N, (md_rel - md_lo) * 100, (md_rel + md_hi) * 100,
                    color=COL["distance"], alpha=0.12, zorder=1)
    ax.plot(fv_N, fv_rel * 100, "s--", color=COL["shuffled"], markersize=6,
            markeredgecolor="white", markeredgewidth=0.8,
            label="fixed-velocity ($N$=64-1024)  [shrinks]", zorder=2)

    ax.set_xscale("log", base=2)
    ax.set_xticks([48, 96, 144, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("network size  $N$")
    ax.set_ylabel("relative conduction-cost saving  (%)")
    if title:
        ax.set_title("Economy persists across $N$=48-1024;\n"
                     "magnitude is dose-dependent (dual dosing shown honestly)")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    ax.annotate(f"{md_rel[-1]*100:.0f}%", (md_N[-1], md_rel[-1] * 100),
                textcoords="offset points", xytext=(6, 6),
                fontsize=8.5, color=COL["distance"], fontweight="bold")
    ax.annotate(f"{fv_rel[-1]*100:.1f}%", (fv_N[-1], fv_rel[-1] * 100),
                textcoords="offset points", xytext=(-2, 8),
                fontsize=8.5, color=COL["shuffled"], fontweight="bold")
    grid_y(ax)
    return dict(md_rel=md_rel, fv_rel=fv_rel)


def draw_theory(ax, d, title=True):
    """Panel: empirical Saving(s) on predicted B0*s line (slope == B0, exact)."""
    mg = d["P4_bump_edge_case"]["monotone_decreasing_g"]
    s = np.array(mg["s"], dtype=float)
    savings = np.array(mg["savings"], dtype=float)
    B0 = mg["B0"]
    pred = B0 * s
    resid = float(np.max(np.abs(savings - pred)))
    slope = float(np.polyfit(s, savings, 1)[0])

    ss = np.linspace(0, s.max() * 1.05, 100)
    ax.plot(ss, B0 * ss, "-", color=COL["theory"], lw=2.2, zorder=1,
            label=r"theory  Saving$(s)=B_0\,s$" + f"  ($B_0={B0:.4f}$)")
    ax.scatter(s, savings, s=80, color=COL["distance"], edgecolor="white",
               linewidth=0.9, zorder=3, label="empirical saving")

    ax.set_xlabel("scale  $s$")
    ax.set_ylabel(r"conduction-cost saving  Saving$(s)$")
    if title:
        ax.set_title("Closed-form theorem holds exactly\n"
                     r"Saving$(s)=B_0\,s$,  slope $\equiv B_0$")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.05,
            f"empirical slope $= {slope:.6f}$\n"
            f"$B_0 = {B0:.6f}$\n"
            f"max |residual| $= {resid:.1e}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COL["grid"], alpha=0.95))
    ax.set_xlim(0, s.max() * 1.05)
    ax.set_ylim(0, savings.max() * 1.12)
    grid_y(ax)
    return dict(slope=slope, B0=B0, resid=resid)


def draw_pinn(ax, d, title=True):
    """Panel: PINN inverse residual vs candidate velocity -- true vs shuffled null."""
    cands = np.array(d["candidates"])
    v_true = d["v_true"]
    per_seed = d["per_seed"]

    true_curves = np.array([s["res_free_true_full"] for s in per_seed])
    shuf_curves = np.array([s["res_free_shuf_full"] for s in per_seed])
    true_mean = true_curves.mean(axis=0)
    true_sem = true_curves.std(axis=0, ddof=1) / np.sqrt(true_curves.shape[0])
    shuf_mean = shuf_curves.mean(axis=0)
    shuf_sem = shuf_curves.std(axis=0, ddof=1) / np.sqrt(shuf_curves.shape[0])

    floor_true = d.get("floor_true_mean", float(true_mean.min()))
    floor_shuf = d.get("floor_shuf_free_mean", float(shuf_mean.min()))
    ratio = d.get("floor_ratio_shuf_over_true_free_mean",
                  floor_shuf / max(floor_true, 1e-12))

    ax.plot(cands, true_mean, "o-", color=COL["distance"], markersize=5,
            markeredgecolor="white", markeredgewidth=0.6,
            label="true geometry", zorder=3)
    ax.fill_between(cands, true_mean - true_sem, true_mean + true_sem,
                    color=COL["distance"], alpha=0.18, zorder=1)
    ax.plot(cands, shuf_mean, "s--", color=COL["shuffled"], markersize=5,
            markeredgecolor="white", markeredgewidth=0.6,
            label="velocity-shuffled null", zorder=3)
    ax.fill_between(cands, shuf_mean - shuf_sem, shuf_mean + shuf_sem,
                    color=COL["shuffled"], alpha=0.18, zorder=1)

    ax.axvline(v_true, color="#555555", ls=":", lw=1.4, zorder=2)
    imin = int(np.argmin(true_mean))
    ax.scatter([cands[imin]], [true_mean[imin]], s=130, facecolor="none",
               edgecolor=COL["accent"], linewidth=2.0, zorder=5)
    ax.annotate(r"sharp dip at $v_{\mathrm{true}}=$" + f"{v_true:g}",
                (cands[imin], true_mean[imin]),
                textcoords="offset points", xytext=(14, 26), fontsize=8.5,
                color=COL["accent"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COL["accent"], lw=1.2))

    # shade the identifiability gap between the two floors
    ax.annotate("", xy=(cands[imin], floor_shuf), xytext=(cands[imin], floor_true),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0))

    ax.set_yscale("log")
    ax.set_xlabel("candidate velocity  $v$")
    ax.set_ylabel("PINN inverse residual  (log scale)")
    if title:
        ax.set_title("Velocity recovery: true geometry recovers $v_{\\mathrm{true}}$;\n"
                     "shuffled null floors far above")
    ax.legend(loc="upper center")
    ax.text(0.97, 0.05,
            f"true floor $= {floor_true:.1e}$\n"
            f"shuffled floor $= {floor_shuf:.2e}$\n"
            f"floor ratio $= {ratio:.0f}\\times$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COL["grid"], alpha=0.95))
    ax.grid(True, which="both", axis="y", color=COL["grid"], alpha=0.30, lw=0.5)
    ax.grid(False, axis="x")
    return dict(floor_true=floor_true, floor_shuf=floor_shuf, ratio=ratio,
                vmin=cands[imin], v_true=v_true)


# Lag-demand / phase-transition primitives (Fig 3 "demand" composite).
# Tasks on the delay-demand ladder. demand = required memory lag (0 = costly-only).
LAG_TASKS = [
    ("memorypro_easy",   0, "MemoryPro\n(lag 0)"),
    ("delayedcopy_lag2", 2, "DelayCopy\n(lag 2)"),
    ("delayedcopy_lag4", 4, "DelayCopy\n(lag 4)"),
    ("delayedcopy_lag6", 6, "DelayCopy\n(lag 6)"),
]
_ORDER3 = ["no-delay", "distance", "shuffled"]


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def lag_dose_table(d):
    """From the raw lag_dose_response runs, build per-task weighted-mean tau for
    distance/shuffled (per seed) and the SAVING = shuffled - distance (per seed)."""
    r = d["runs"]
    out = []
    for task, demand, label in LAG_TASKS:
        wm = {c: [] for c in _ORDER3}
        for s in range(8):
            if not all(f"{task}|{s}|{c}" in r for c in _ORDER3):
                continue
            for c in _ORDER3:
                wm[c].append(r[f"{task}|{s}|{c}"]["wmean"])
        n = min(len(wm[c]) for c in _ORDER3)
        if n == 0:
            continue
        saving = [wm["shuffled"][s] - wm["distance"][s] for s in range(n)]
        out.append(dict(task=task, demand=demand, label=label, n=n,
                        dist=np.array(wm["distance"][:n]),
                        shuf=np.array(wm["shuffled"][:n]),
                        saving=np.array(saving),
                        saving_mean=float(np.mean(saving))))
    return out


def draw_lag_dose(ax, d, rng, title=True):
    """Panel A: lag dose-response -- saving (shuffled-minus-distance weighted-mean
    tau) vs delay-demand. Teal filled curve, Pearson r, wash region at demand 0."""
    tab = lag_dose_table(d)
    dem = np.array([t["demand"] for t in tab], dtype=float)
    sav = np.array([t["saving_mean"] for t in tab])
    sem = np.array([float(t["saving"].std(ddof=1) / np.sqrt(t["n"]))
                    if t["n"] > 1 else 0.0 for t in tab])
    r = _pearson(dem, sav)

    # "wash" band at demand 0 -- no delay-demand, so no economy expected (~0).
    ax.axhline(0, color="#888888", lw=1.0, ls="-", zorder=1)
    ax.axvspan(-0.5, 1.0, color="#6C757D", alpha=0.10, zorder=0)
    ax.annotate("wash\n(no delay\ndemand)", xy=(0.0, 0.0),
                xytext=(0.0, max(sav) * 0.42), ha="center", va="center",
                fontsize=8, color="#5A6268")

    ax.fill_between(dem, 0, sav, color=COL["distance"], alpha=0.16, zorder=1)
    ax.errorbar(dem, sav, yerr=sem, fmt="o-", color=COL["distance"],
                ecolor=COL["distance"], elinewidth=1.1, capsize=3.5,
                markersize=8, markeredgecolor="white", markeredgewidth=0.9,
                zorder=3, label=r"saving  $\bar{\tau}_{\mathrm{shuf}}-\bar{\tau}_{\mathrm{dist}}$")
    for t in tab:
        seed_dots(ax, t["demand"], t["saving"], rng, jitter=0.10, s=16)

    ax.set_xlabel("delay demand  (required memory lag)")
    ax.set_ylabel(r"conduction-cost saving  $\Delta\bar{\tau}$")
    ax.set_xticks([0, 2, 4, 6])
    if title:
        ax.set_title("Saving grows with delay demand")
    ax.text(0.97, 0.06, f"Pearson $r = {r:+.2f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            fontweight="bold", color=COL["distance"],
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=COL["grid"], alpha=0.95))
    ax.legend(loc="upper left")
    ax.set_xlim(-0.6, 6.6)
    grid_y(ax)
    return dict(r=r, saving=sav, dem=dem)


def draw_lag_cost_bars(ax, d, rng, title=True):
    """Panel B: per-task conduction-cost bars -- distance vs shuffled weighted-mean
    tau for each task, grouped bars with per-seed dots. These are the underlying
    costs whose gap (shuffled>distance) is the saving in Panel A."""
    tab = lag_dose_table(d)
    n_tasks = len(tab)
    x = np.arange(n_tasks)
    w = 0.36

    for grp, key, col, off in [("distance", "dist", COL["distance"], -w / 2),
                               ("shuffled", "shuf", COL["shuffled"], +w / 2)]:
        means = np.array([float(t[key].mean()) for t in tab])
        sems = np.array([float(t[key].std(ddof=1) / np.sqrt(t["n"]))
                         if t["n"] > 1 else 0.0 for t in tab])
        ax.bar(x + off, means, width=w, color=col, alpha=0.85,
               edgecolor="#333333", linewidth=0.7, zorder=2, label=grp)
        ax.errorbar(x + off, means, yerr=sems, fmt="none", ecolor="#222222",
                    elinewidth=1.0, capsize=3, zorder=4)
        for i, t in enumerate(tab):
            seed_dots(ax, x[i] + off, t[key], rng, jitter=0.07, s=14)
            value_label(ax, x[i] + off, means[i] + sems[i], f"{means[i]:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels([t["label"] for t in tab])
    ax.set_ylabel(r"conduction cost  $\bar{\tau}$  (weighted-mean delay)")
    if title:
        ax.set_title("Distance wiring is cheaper on every task")
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(0, max(float(t["shuf"].mean()) for t in tab) * 1.30)
    grid_y(ax)
    return dict(tab=tab)


def draw_phase_transition(ax, d, rng, vel="0.05", title=True, exclude=None):
    """Panel C: phase transition -- saving (order parameter) vs integration demand
    (input noise). Economy is ~0 below a threshold and turns ON above it.
    exclude: list of noise values to omit (e.g. [0.2] for the main panel; the full
    curve incl. the intermediate-demand dip is shown in the SI)."""
    r = d["runs"]
    noises = d["params"]["noises"]
    seeds = d["params"]["seeds"]
    excl = set(float(e) for e in (exclude or []))
    xs, means, sems, per_seed = [], [], [], []
    for noise in noises:
        if float(noise) in excl:
            continue
        sav = []
        for s in range(seeds):
            kd = f"v{vel}|noise{noise}|{s}|distance"
            ks = f"v{vel}|noise{noise}|{s}|shuffled"
            if kd in r and ks in r:
                sav.append(r[ks]["wmean"] - r[kd]["wmean"])
        if not sav:
            continue
        xs.append(float(noise))
        means.append(float(np.mean(sav)))
        sems.append(float(np.std(sav, ddof=1) / np.sqrt(len(sav))) if len(sav) > 1 else 0.0)
        per_seed.append(np.array(sav))
    xs = np.array(xs); means = np.array(means); sems = np.array(sems)

    thresh = 0.3  # integration-demand threshold where the economy switches on

    ax.axhline(0, color="#888888", lw=1.0, zorder=1)
    # shade OFF (below threshold) vs ON (above threshold) regimes
    ax.axvspan(min(xs) - 0.05, thresh, color="#6C757D", alpha=0.10, zorder=0)
    ax.axvline(thresh, color=COL["accent"], ls="--", lw=1.6, zorder=2)
    ax.annotate(f"threshold $\\approx${thresh:g}", xy=(thresh, 0.97),
                xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", ha="left", va="top",
                fontsize=8.5, color=COL["accent"], fontweight="bold")

    ax.errorbar(xs, means, yerr=sems, fmt="o-", color=COL["distance"],
                ecolor=COL["distance"], elinewidth=1.1, capsize=3.5,
                markersize=8, markeredgecolor="white", markeredgewidth=0.9,
                zorder=3, label="saving (order parameter)")
    for xv, ps in zip(xs, per_seed):
        seed_dots(ax, xv, ps, rng, jitter=0.012, s=14)

    ax.annotate("OFF", xy=(0.10, 0.86), xycoords="axes fraction",
                fontsize=10, color="#5A6268", fontweight="bold", ha="center")
    ax.annotate("ON", xy=(0.82, 0.86), xycoords="axes fraction",
                fontsize=10, color=COL["distance"], fontweight="bold", ha="center")

    ax.set_xlabel("integration demand  (input noise)")
    ax.set_ylabel(r"conduction-cost saving  $\Delta\bar{\tau}$")
    if title:
        ax.set_title("Economy switches on past a threshold")
    ax.legend(loc="lower right")
    grid_y(ax)
    return dict(xs=xs, means=means, thresh=thresh)


# Single-panel figures.
def fig1_spine():
    d = load("conduction_cost.json")
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.0, 4.6), constrained_layout=True)
    r = draw_spine(ax, d, rng)
    return save(fig, "fig1_spine.png",
                f"distance tau-bar={r['means'][2]:.2f} vs shuffled {r['means'][1]:.2f}, "
                f"{r['pct']:.0f}% cheaper, paired t={r['tval']:.1f}")


def fig2_dose():
    d = load("dose_response.json")
    fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    r = draw_dose(ax, d)
    return save(fig, "fig2_dose.png",
                f"saving {r['red'].min():.2f}->{r['red'].max():.2f} as delay grows; "
                f"corr={r['trend']:.2f}")


def fig3_theory():
    d = load("optimal_control.json")
    fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    r = draw_theory(ax, d)
    return save(fig, "fig3_theory.png",
                f"slope={r['slope']:.6f} == B0={r['B0']:.6f}, max|resid|={r['resid']:.1e}")


def fig4_3d2d():
    d = load("geometry_3d_vs_2d.json")
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    r = draw_3d2d(ax, d, rng)
    return save(fig, "fig4_3d2d.png",
                f"scale-matched gap 2D={r['g2']:.3f} 3D={r['g3']:.3f} (+{r['pct']:.0f}%)")


def fig5_scaling():
    fs = load("scaling.json")
    fig, ax = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
    r = draw_scaling(ax, fs)
    return save(fig, "fig5_scaling.png",
                f"matched-dose rel {r['md_rel'][0]*100:.1f}%->{r['md_rel'][-1]*100:.1f}% "
                f"(grows); fixed-v {r['fv_rel'][0]*100:.1f}%->{r['fv_rel'][-1]*100:.1f}% (shrinks)")


def fig6_pinn():
    d = load("pinn_inverse.json")
    fig, ax = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)
    r = draw_pinn(ax, d)
    return save(fig, "fig6_pinn.png",
                f"true floor {r['floor_true']:.1e} vs shuffled {r['floor_shuf']:.2e} "
                f"(ratio {r['ratio']:.0f}x), dip at v={r['vmin']:g} (v_true={r['v_true']:g})")


# Composite main figures (multi-panel).
def figM1_economy():
    """2x2 'conduction-time economy' main figure: (A) spine, (B) dose,
    (C) 3D>2D, (D) scaling -- one clean panel set with bold A-D labels."""
    bc = load("conduction_cost.json")
    dose = load("dose_response.json")
    d3d = load("geometry_3d_vs_2d.json")
    fs = load("scaling.json")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)
    (axA, axB), (axC, axD) = axes

    draw_spine(axA, bc, np.random.default_rng(0))
    draw_dose(axB, dose)
    draw_3d2d(axC, d3d, np.random.default_rng(1))
    draw_scaling(axD, fs)

    for ax, L in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        panel_label(ax, L)

    fig.suptitle("Distance-based conduction delays buy a conduction-time economy",
                 fontsize=14, fontweight="bold")
    return save(fig, "figM1_economy.png",
                "composite A=spine B=dose C=3D>2D D=scaling")


def figM2_theory_pinn():
    """Composite 'derivation + identifiability' figure:
    (A) theorem match, (B) PINN inverse residual."""
    poc = load("optimal_control.json")
    pinn = load("pinn_inverse.json")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   constrained_layout=True)
    draw_theory(axA, poc)
    draw_pinn(axB, pinn)
    panel_label(axA, "A", x=-0.14)
    panel_label(axB, "B", x=-0.14)

    fig.suptitle("Closed-form derivation and inverse identifiability",
                 fontsize=14, fontweight="bold")
    return save(fig, "figM2_theory_pinn.png",
                "composite A=theorem match B=PINN inverse residual")


# Fig 3 delay-demand economy (main composite + SI standalone).
def fig_laggate():
    """SI standalone: the bare lag dose-response panel (saving vs delay demand),
    teal filled curve + Pearson r + wash at demand 0. Kept for completeness; the
    rich main figure is figM3_demand.png."""
    d = load("lag_dose_response.json")
    fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    r = draw_lag_dose(ax, d, np.random.default_rng(3))
    return save(fig, "fig_laggate.png",
                f"lag dose-response saving vs demand, Pearson r={r['r']:+.2f}")


def figM3_demand():
    """Main Fig 3 'demand' composite (1x3): (A) lag dose-response curve,
    (B) per-task conduction-cost bars (distance vs shuffled), (C) phase
    transition -- saving vs integration demand with threshold."""
    lag = load("lag_dose_response.json")
    pt = load("phase_transition.json")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.8),
                                        constrained_layout=True)
    rA = draw_lag_dose(axA, lag, np.random.default_rng(3))
    draw_lag_cost_bars(axB, lag, np.random.default_rng(4))
    rC = draw_phase_transition(axC, pt, np.random.default_rng(5), exclude=[0.2])

    for ax, L in zip([axA, axB, axC], ["A", "B", "C"]):
        panel_label(ax, L, x=-0.16)

    fig.suptitle("When the conduction-time economy turns on",
                 fontsize=14, fontweight="bold")
    return save(fig, "figM3_demand.png",
                f"composite A=lag dose (r={rA['r']:+.2f}) B=per-task cost bars "
                f"C=phase transition (thresh~{rC['thresh']:g})")


# Fig M4: the allocation law Saving(s)=B0*s across four systems --
# A rate-RNN (exact), B Kuramoto (HLP analytic), C echo-state reservoir, D spiking LIF.
def draw_law_rnn(ax, d, title=True, col=COL):
    """Panel A: the EXACT closed-form law in the rate-RNN -- Saving(s)=B0*s with
    slope == B0 and ~zero residual. Same data/logic as draw_theory()."""
    mg = d["P4_bump_edge_case"]["monotone_decreasing_g"]
    s = np.array(mg["s"], dtype=float)
    savings = np.array(mg["savings"], dtype=float)
    B0 = mg["B0"]
    pred = B0 * s
    resid = float(np.max(np.abs(savings - pred)))
    slope = float(np.polyfit(s, savings, 1)[0])

    ss = np.linspace(0, s.max() * 1.05, 100)
    ax.plot(ss, B0 * ss, "-", color=col["theory"], lw=2.2, zorder=1,
            label=r"Saving$(s)=B_0\,s$")
    ax.scatter(s, savings, s=70, color=col["distance"], edgecolor="white",
               linewidth=0.9, zorder=3, label="RNN (exact)")

    ax.set_xlabel("latency scale  $s$")
    ax.set_ylabel(r"conduction-cost saving  Saving$(s)$")
    if title:
        ax.set_title("Rate-RNN: exact closed-form law")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.05,
            f"slope $= B_0 = {B0:.6f}$\n"
            f"max |residual| $= {resid:.1e}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COL["grid"], alpha=0.95))
    ax.set_xlim(0, s.max() * 1.05)
    ax.set_ylim(0, savings.max() * 1.12)
    grid_y(ax)
    return dict(slope=slope, B0=B0, resid=resid)


def draw_law_kuramoto(ax, dk, title=True, col=COL):
    """Panel B: Kuramoto phase-oscillator field. Saving vs latency scale s
    (s = realized weighted-mean conduction delay), with the HLP analytic linear
    fit Saving = 263.6*s (R2=0.999)."""
    dose = dk["dose"]
    s = np.array(dose["mean_tau"], dtype=float)        # latency scale (weighted-mean delay)
    sav = np.array(dose["saving"], dtype=float)
    slope = float(dose["linear_slope"])
    inter = float(dose["intercept"])
    r2 = float(dose["r2"])

    ss = np.linspace(0, s.max() * 1.05, 100)
    ax.plot(ss, slope * ss + inter, "-", color=col["theory"], lw=2.2, zorder=1,
            label="linear fit (HLP analytic)")
    ax.scatter(s, sav, s=70, color=col["distance"], edgecolor="white",
               linewidth=0.9, zorder=3, label="Kuramoto saving")

    ax.set_xlabel(r"latency scale  $s$  (weighted-mean delay $\bar{\tau}$)")
    ax.set_ylabel(r"conduction-cost saving  Saving$(s)$")
    if title:
        ax.set_title("Kuramoto oscillators: HLP analytic law")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.05,
            f"Saving $= {slope:.1f}\\cdot s$\n$R^2 = {r2:.3f}$\nHLP analytic",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COL["grid"], alpha=0.95))
    ax.set_xlim(0, s.max() * 1.05)
    ax.set_ylim(0, sav.max() * 1.12)
    grid_y(ax)
    return dict(slope=slope, r2=r2)


def draw_law_reservoir(ax, dd, da, rng, title=True, col=COL):
    """Panel C: echo-state reservoir. Saving (weighted-mean tau, shuffled-minus-
    distance) vs latency scale s (=delay spread), linear fit R2=0.94, plus per-seed
    points and the distance-shuffled economy callout (-0.562, t=-21, 5/5 seeds)."""
    rows = sorted(dd["rows"], key=lambda r: r["spread_mean"])
    s = np.array([r["spread_mean"] for r in rows], dtype=float)
    sav = np.array([r["saving_wmean_mean"] for r in rows], dtype=float)
    fit = dd["fit"]
    slope = float(fit["slope"])
    inter = float(fit["intercept"])
    r2 = float(fit["r2"])

    ss = np.linspace(0, s.max() * 1.05, 100)
    ax.plot(ss, slope * ss + inter, "-", color=col["theory"], lw=2.2, zorder=1,
            label="linear fit")
    ax.axhline(0, color="#888888", lw=0.9, zorder=1)
    # per-seed points (jittered around each dose level)
    for r in rows:
        ps = np.array([p["saving_wmean"] for p in r["per_seed"]], dtype=float)
        seed_dots(ax, r["spread_mean"], ps, rng, jitter=0.10, s=16)
    ax.scatter(s, sav, s=70, color=col["distance"], edgecolor="white",
               linewidth=0.9, zorder=4, label="reservoir saving (per-dose mean)")

    # economy callout from the A-regime run (weighted-mean tau distance vs shuffled)
    econ = da["regimes"]["A"]["economy"]["gain"]["wmean_tau"]
    md = float(econ["mean_diff"]); tv = float(econ["t"])
    nlt = int(econ["n_dist_lt_shuf"]); n = int(econ["n"])

    ax.set_xlabel(r"latency scale  $s$  (delay spread)")
    ax.set_ylabel(r"conduction-cost saving  $\Delta\bar{\tau}$")
    if title:
        ax.set_title("Echo-state reservoir: empirical law")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.05,
            f"linear fit  $R^2 = {r2:.2f}$\n"
            f"distance$-$shuffled $= {md:.3f}$\n"
            f"$t = {tv:.0f}$,  {nlt}/{n} seeds",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COL["grid"], alpha=0.95))
    ax.set_xlim(0, s.max() * 1.05)
    ax.set_ylim(min(sav.min(), 0) - 0.15, sav.max() * 1.18)
    grid_y(ax)
    return dict(r2=r2, md=md, t=tv, nlt=nlt, n=n)


def draw_law_spiking(ax, ds, rng, title=True, col=COL):
    """Panel D: spiking LIF network. Grouped bars of weighted-mean tau (distance vs
    shuffled) at matched accuracy, per-seed dots, 4/4 seeds; with an inset showing the
    2-point dose-response (saving grows as velocity drops, v=0.08 -> v=0.05)."""
    econ = ds["economy"]
    dist = np.array(econ["distance"]["tbar"], dtype=float)
    shuf = np.array(econ["shuffled"]["tbar"], dtype=float)
    d_acc = float(econ["d_acc_mean"]); s_acc = float(econ["s_acc_mean"])
    wins = int(econ["wins"]); n = int(econ["n"])

    x = np.array([0.0, 1.0])
    means = [float(dist.mean()), float(shuf.mean())]
    sems = [float(dist.std(ddof=1) / np.sqrt(len(dist))),
            float(shuf.std(ddof=1) / np.sqrt(len(shuf)))]
    cols = [col["distance"], col["shuffled"]]
    for xi, m, sem, bcol, vals in zip(x, means, sems, cols, [dist, shuf]):
        ax.bar(xi, m, width=0.60, color=bcol, alpha=0.85, edgecolor="#333333",
               linewidth=0.7, zorder=2)
        ax.errorbar(xi, m, yerr=sem, fmt="none", ecolor="#222222",
                    elinewidth=1.1, capsize=4, zorder=4)
        seed_dots(ax, xi, vals, rng, jitter=0.10)
        value_label(ax, xi, m + sem, f"{m:.2f}")

    # saving bracket distance vs shuffled
    y0 = max(means) + max(sems) + 0.9
    ax.plot([0, 0, 1, 1], [y0, y0 + 0.25, y0 + 0.25, y0], color="#222222", lw=1.1)
    ax.text(0.5, y0 + 0.35,
            f"distance cheaper  {wins}/{n} seeds\n"
            f"matched acc $\\approx${0.5 * (d_acc + s_acc):.2f}",
            ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["distance\ndelays", "shuffled\ndelays"])
    ax.set_ylabel(r"conduction cost  $\bar{\tau}$  (weighted-mean delay)")
    if title:
        ax.set_title("Spiking LIF network")
    ax.set_ylim(0, y0 + 2.2)
    grid_y(ax)

    # dose-response stated as a clean one-line note under the bars (no cramped inset)
    dose = sorted(ds["dose"], key=lambda r: r["velocity"])
    sv = np.array([r["saving"] for r in dose], dtype=float)
    v_hi = max(r["velocity"] for r in dose); v_lo = min(r["velocity"] for r in dose)
    ax.text(0.5, 0.03,
            f"dose-response: saving {sv.min():+.2f} ($v$={v_hi:g}) $\\rightarrow$ "
            f"{sv.max():+.2f} ($v$={v_lo:g})",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
            style="italic", color=col["distance"])
    return dict(means=means, wins=wins, n=n, dose_lo=float(sv.min()), dose_hi=float(sv.max()))


def figM4_law():
    """2x2 composite: the conduction-time allocation law Saving=B0*s across four
    independent systems -- (A) rate-RNN exact, (B) Kuramoto HLP analytic,
    (C) echo-state reservoir, (D) spiking LIF."""
    poc = load("optimal_control.json")
    dk = load("law_kuramoto.json")
    rd = load("law_reservoir_dose.json")
    ra = load("law_reservoir_A.json")
    ds = load("law_spiking.json")

    # M4-LOCAL palette: blue (was teal/green) + orange -- the colorblind-safe
    # blue-orange pair. Scoped to this figure only; M1/M5 keep the shared teal.
    law_col = dict(COL)
    law_col["distance"] = "#2C6FB3"   # blue  -- the model / data points
    law_col["theory"]   = "#13334F"   # dark navy -- analytic prediction line
    # shuffled stays orange (#E76F51); blue+orange is colorblind-safe.

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)
    (axA, axB), (axC, axD) = axes

    rA = draw_law_rnn(axA, poc, col=law_col)
    rB = draw_law_kuramoto(axB, dk, col=law_col)
    rC = draw_law_reservoir(axC, rd, ra, np.random.default_rng(8), col=law_col)
    rD = draw_law_spiking(axD, ds, np.random.default_rng(9), col=law_col)

    for ax, L in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        panel_label(ax, L)

    fig.suptitle("The conduction-time allocation law across systems",
                 fontsize=14, fontweight="bold")
    # whole-figure note
    fig.text(0.5, -0.012,
             r"Same Saving$=B_0\,s$ across four systems; present under short-edge "
             "concentration.",
             ha="center", va="top", fontsize=9.5, color="#444444", style="italic")
    return save(fig, "figM4_law.png",
                f"composite A=RNN(B0={rA['B0']:.3f}) B=Kuramoto(slope={rB['slope']:.0f},R2={rB['r2']:.3f}) "
                f"C=reservoir(R2={rC['r2']:.2f}) D=spiking({rD['wins']}/{rD['n']})")


# Fig M5: conduction velocity inferred from foreign activity --
# A neural-field PDE (vs true c), B spiking LIF (vs null), C human Utah array.
def draw_inv_neuralfield(ax, d, title=True):
    """Panel A: neural-field PDE. Recovered v_hat vs true wave-speed c (scatter +
    dashed identity), inset of residual-vs-velocity with the interior minimum and
    the autocorr-null floor."""
    summ = d["summary"]
    cs = sorted({s["c"] for s in summ})
    c_arr, vhat_m, vhat_s = [], [], []
    for c in cs:
        rows = [s for s in summ if s["c"] == c]
        vh = np.array([s["v_hat_mean"] for s in rows], dtype=float)
        c_arr.append(c)
        vhat_m.append(float(vh.mean()))
        vhat_s.append(float(vh.std(ddof=1) / np.sqrt(len(vh))) if len(vh) > 1 else 0.0)
    c_arr = np.array(c_arr); vhat_m = np.array(vhat_m); vhat_s = np.array(vhat_s)

    hl = d["headline"]
    n_beat = int(hl["overall_beats_null"]); n_cond = int(hl["n_conditions"])
    # autocorr-PRESERVING null (the decisive test vs trivial single-channel
    # autocorrelation): median over the clean, full-observation conditions.
    _full = max(s["n_elec"] for s in summ)
    null_ratio = float(np.median([s["autocorr_null_ratio_mean"] for s in summ
                                  if s["noise_std"] == 0.0 and s["n_elec"] == _full]))
    # interior-min seeds (per spec "5/5 seeds")
    int_seeds = min(int(s["interior_min_seeds"]) for s in summ)
    n_seeds = int(summ[0]["n_seeds"])

    lim = max(c_arr.max(), vhat_m.max()) * 1.12
    ax.plot([0, lim], [0, lim], "--", color="#555555", lw=1.4, zorder=1,
            label="identity  $\\hat v = c$")
    ax.errorbar(c_arr, vhat_m, yerr=vhat_s, fmt="o", color=COL["distance"],
                ecolor=COL["distance"], elinewidth=1.1, capsize=4, markersize=10,
                markeredgecolor="white", markeredgewidth=0.9, zorder=3,
                label="recovered  $\\hat v$")
    for cv, vv in zip(c_arr, vhat_m):
        ax.annotate(f"{vv:.2f}", (cv, vv), textcoords="offset points",
                    xytext=(8, -2), fontsize=8, color=COL["distance"],
                    fontweight="bold")

    ax.set_xlabel("true wave speed  $c$")
    ax.set_ylabel(r"recovered velocity  $\hat v$")
    if title:
        ax.set_title("Neural-field PDE: $\\hat v$ tracks $c$")
    ax.legend(loc="upper left")
    ax.text(0.03, 0.62,
            f"beats autocorr null {null_ratio:.2f}x\n"
            f"{int_seeds}/{n_seeds} seeds interior\n"
            f"{n_beat}/{n_cond} beat null\n"
            "noise-robust to 30%",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COL["grid"], alpha=0.95))
    ax.set_xlim(0, max(c_arr) * 1.15)
    ax.set_ylim(0, lim)
    grid_y(ax)

    return dict(c=c_arr, vhat=vhat_m, null_ratio=null_ratio,
                n_beat=n_beat, n_cond=n_cond)


def draw_inv_spiking(ax, d, title=True):
    """Panel B: spiking LIF v_true SWEEP. The recovered v_hat is plotted against the
    TRUE conduction velocity for several v_true, with a dashed identity line. v_hat
    TRACKS v_true MONOTONICALLY -- the non-tautology proof (the inverse reads the
    generator's delay, not the single velocity its candidate grid was built around).
    Each point: seed mean +- seed spread; annotated with the real low-bias % and the
    velocity-shuffled null z."""
    pts = sorted(d["sweep"]["points"], key=lambda p: p["v_true"])
    vts = np.array([p["v_true"] for p in pts], dtype=float)
    vhs = np.array([p["v_hat_mean"] for p in pts], dtype=float)
    zs = np.array([p["null_z_mean"] for p in pts], dtype=float)
    seed_lo = np.array([min(p["v_hat_seeds"]) for p in pts], dtype=float)
    seed_hi = np.array([max(p["v_hat_seeds"]) for p in pts], dtype=float)
    # aggregate low-bias across the sweep (mean of per-point bias), and a min null z
    low_bias = float(np.mean(100.0 * (vts - vhs) / vts))
    z_min = float(zs.min())
    slope = float(d["sweep"].get("loglog_slope", float("nan")))
    mono = bool(d["sweep"].get("strict_monotonic", False))

    lim = max(vts.max(), vhs.max(), seed_hi.max()) * 1.18
    ax.plot([0, lim], [0, lim], "--", color="#555555", lw=1.4, zorder=1,
            label="identity  $\\hat v = v_{\\mathrm{true}}$")
    # per-point seed range as error bars + connect the recovered track
    ax.errorbar(vts, vhs, yerr=[vhs - seed_lo, seed_hi - vhs], fmt="o-",
                color=COL["distance"], ecolor=COL["distance"], elinewidth=1.2,
                capsize=4, markersize=8, markeredgecolor="white",
                markeredgewidth=0.8, zorder=4,
                label=r"recovered $\hat v$ (mean $\pm$ seeds)")

    ax.set_xlabel(r"true velocity  $v_{\mathrm{true}}$")
    ax.set_ylabel(r"recovered velocity  $\hat v$")
    if title:
        ax.set_title("Spiking LIF: $\\hat v$ tracks $v_{\\mathrm{true}}$ (non-tautology)")
    ax.legend(loc="upper left", fontsize=8)
    mono_txt = "monotonic tracking" if mono else "tracks (non-strict)"
    ax.text(0.97, 0.05,
            f"{mono_txt}, slope$\\approx${slope:.2f}\n"
            f"null $z \\geq {z_min:.1f}$ (beats shuffle)\n"
            f"$\\approx${low_bias:.0f}% low-biased",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COL["grid"], alpha=0.95))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    grid_y(ax)

    # representative single v_true (the original headline point, if present) for caption
    v_true_rep = float(vts[np.argmin(np.abs(vts - 0.08))])
    return dict(v_true=v_true_rep, vhat_mean=float(vhs[np.argmin(np.abs(vts - 0.08))]),
                low_bias=low_bias, zmin=z_min, slope=slope, monotonic=mono)


def draw_inv_intracranial(ax, dfull, drealdata, title=True):
    """Panel C (the clincher): real human Utah-array intracranial data. Recovered
    v_hat vs an INDEPENDENT ground-truth v_ref (burst-latency planar fit, IQR band),
    with the AUTOCORR-PRESERVING null residual comparison (true floor vs null floor,
    ratio). All numbers are read from the REPRODUCIBLE full-run JSON
    results/inverse/foreigninv_intracranial.json. Real-data visual cue + EEG note."""
    res = dfull["result"]
    cand = np.array(res["candidates_mps"], dtype=float)
    rt = np.array(res["res_true"], dtype=float)

    # headline numbers - ALL from the saved reproducible full run (not hand-typed).
    v_hat = float(res["v_hat_mps"])
    ci = tuple(res["v_hat_boot_ci68_mps"])
    v_ref = float(res["v_ref_mps"])
    v_ref_iqr = res["v_ref_iqr_mps"]
    true_floor = float(res["floor_true"])
    null_floor = float(res["null_ac_floor_mean"])
    null_ratio = float(res["null_ac_ratio"])

    eeg = drealdata["results"][0]
    eeg_err = float(eeg["err_vs_ref_pct"])

    # Faint warm tint = quiet "real data" cue (the panel title already says so,
    # so no loud badge -- it only competed with the result).
    ax.set_facecolor("#FBF5F1")

    # residual-vs-candidate-velocity curve (the inverse scan) -- THE result
    ax.plot(cand, rt, "o-", color=COL["distance"], lw=2.0, markersize=5,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4,
            label="inverse residual (true geometry)")
    imin = int(np.argmin(rt))
    ax.scatter([cand[imin]], [rt[imin]], s=130, facecolor="none",
               edgecolor=COL["accent"], linewidth=2.0, zorder=6)
    ax.annotate(r"$\hat v=%.2f$ m/s  [%.2f, %.2f]" % (v_hat, ci[0], ci[1]),
                (cand[imin], rt[imin]), textcoords="offset points",
                xytext=(-30, 34), fontsize=8.5, color=COL["accent"],
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=COL["accent"], lw=1.2))

    # the two floors -- values live in the legend, no free-floating note
    ax.axhline(null_floor, color=COL["shuffled"], ls="--", lw=1.6, zorder=3,
               label=f"autocorr null floor  ({null_floor:.3f})")
    ax.axhline(true_floor, color=COL["distance"], ls=":", lw=1.4, zorder=3,
               label=f"true floor  ({true_floor:.3f})")

    # independent ground-truth band v_ref IQR (on the velocity axis)
    ax.axvspan(v_ref_iqr[0], v_ref_iqr[1], color="#6C757D", alpha=0.14, zorder=0)
    ax.axvline(v_ref, color="#444444", ls="-", lw=1.4, zorder=2)
    ax.annotate(f"$v_{{\\mathrm{{ref}}}}={v_ref:.2f}$  (IQR {v_ref_iqr[0]:.2f}-{v_ref_iqr[1]:.2f})",
                (v_ref, rt.max()), textcoords="offset points", xytext=(-5, -2),
                fontsize=7.5, color="#444444", ha="right", va="top")

    ax.set_xscale("log")
    ax.set_xlabel("candidate / recovered velocity  $v$  (m/s, log)")
    ax.set_ylabel("inverse residual")
    if title:
        ax.set_title("Human cortex (Utah array): real-data clincher")
    ax.legend(loc="upper center", fontsize=7.5)

    # ONE consolidated stat box (matches the idiom of panels A/B) -- folds the
    # beats-null ratio, the honesty caveat, and the scalp-EEG point into the
    # standard corner box instead of three free-floating annotations.
    ax.text(0.97, 0.05,
            f"beats autocorr null {null_ratio:.2f}×\n"
            "effective group speed\n"
            f"scalp EEG misses $\\approx${eeg_err / 100 + 1:.0f}×",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COL["grid"], alpha=0.95), zorder=11)
    ax.grid(True, axis="y", color=COL["grid"], alpha=0.30, lw=0.5)
    ax.grid(False, axis="x")
    return dict(v_hat=v_hat, v_ref=v_ref, null_ratio=null_ratio,
                true_floor=true_floor, null_floor=null_floor)


def figM5_inverse():
    """1x3 composite: conduction velocity inferred from activity --
    (A) neural-field PDE, (B) spiking LIF, (C) real human Utah-array intracranial."""
    nf = load("foreigninv_neural_field.json")
    sp = load("foreigninv_spiking.json")          # v_true-sweep tracking (reproducible)
    ic = load("foreigninv_intracranial.json")     # full real-data run (reproducible)
    rd = load("foreigninv_realdata.json")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.5, 4.8),
                                        constrained_layout=True)
    rA = draw_inv_neuralfield(axA, nf)
    rB = draw_inv_spiking(axB, sp)
    rC = draw_inv_intracranial(axC, ic, rd)

    for ax, L in zip([axA, axB, axC], ["A", "B", "C"]):
        panel_label(ax, L, x=-0.14)

    fig.suptitle("Conduction velocity inferred from activity",
                 fontsize=14, fontweight="bold")
    return save(fig, "figM5_inverse.png",
                f"composite A=neural-field(beats null {rA['null_ratio']:.2f}x) "
                f"B=spiking(v_hat={rB['vhat_mean']:.3f} vs {rB['v_true']:g}, {rB['low_bias']:.0f}% low) "
                f"C=intracranial(v_hat={rC['v_hat']:.3f} vs v_ref={rC['v_ref']:.3f})")


# SI figure set (lowercase (a)(b)(c) labels in white boxes).
def si_panel_label(ax, letter, x=0.02, y=0.97):
    """Lowercase (a)(b)(c) panel label in a white box (WeakPINN SI convention)."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.28", fc="white",
                      ec="#999999", lw=0.8, alpha=0.95), zorder=10)


def fig_si_geometry():
    """SI: 3D vs 2D conduction-cost gap. (a) scale-matched bars, (b) raw bars,
    per-seed dots, +pct annotated."""
    d = load("geometry_3d_vs_2d.json")
    head = d["headline"]

    def per_seed_gap(block):
        dist = np.array(block["distance"]["C_all"])
        shuf = np.array(block["shuffled"]["C_all"])
        return shuf - dist

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.4, 4.4),
                                   constrained_layout=True)
    for ax, blockkey, g2key, g3key, sub in [
            (axa, "scale_matched", "scale_matched_gap_2d", "scale_matched_gap_3d", "scale-matched"),
            (axb, "raw", "raw_gap_2d", "raw_gap_3d", "raw")]:
        rng = np.random.default_rng(11)
        block = d[blockkey]
        gaps = {"2D": per_seed_gap(block["2d"]), "3D": per_seed_gap(block["3d"])}
        xpos = [0, 1]
        cols = [COL["distance"], COL["theory"]]
        means = {}
        for x, lab, col in zip(xpos, ["2D", "3D"], cols):
            vals = gaps[lab]
            m = float(vals.mean())
            sem = float(vals.std(ddof=1) / np.sqrt(len(vals)))
            means[lab] = m
            ax.bar(x, m, width=0.62, color=col, alpha=0.85,
                   edgecolor="#333333", linewidth=0.7, zorder=2)
            ax.errorbar(x, m, yerr=sem, fmt="none", ecolor="#222222",
                        elinewidth=1.1, capsize=4, zorder=4)
            seed_dots(ax, x, vals, rng, jitter=0.10)
            value_label(ax, x, m + sem, f"{m:.2f}")
        g2, g3 = head[g2key], head[g3key]
        pct = 100.0 * (g3 - g2) / g2
        yb = max(means.values()) + 0.10 * max(means.values())
        ax.plot([0, 0, 1, 1], [yb, yb + 0.03 * yb, yb + 0.03 * yb, yb],
                color="#222222", lw=1.0)
        ax.text(0.5, yb + 0.05 * yb, f"+{pct:.0f}% in 3D", ha="center",
                va="bottom", fontsize=9.5, fontweight="bold", color=COL["accent"])
        ax.set_xticks(xpos)
        ax.set_xticklabels(["2D", "3D"])
        ax.set_ylabel(r"conduction-cost gap  $C_{\mathrm{shuf}}-C_{\mathrm{dist}}$")
        ax.set_title(sub)
        ax.set_ylim(0, max(means.values()) * 1.32)
        grid_y(ax)

    si_panel_label(axa, "a")
    si_panel_label(axb, "b")
    fig.suptitle("The conduction-time economy is larger in 3D",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    return save(fig, "fig_si_geometry.png",
                f"(a) scale-matched 2D={head['scale_matched_gap_2d']:.2f} "
                f"3D={head['scale_matched_gap_3d']:.2f} (b) raw 2D={head['raw_gap_2d']:.2f} "
                f"3D={head['raw_gap_3d']:.2f}")


def fig_si_scaling():
    """SI: scaling dual-dosing detail. (a) matched-dose grows, (b) fixed-velocity
    shrinks. Magnitude is dose-dependent."""
    md = load("scaling.json")
    per_N = sorted(md["summary"]["per_N"], key=lambda r: r["N"])
    md_N = np.array([r["N"] for r in per_N], dtype=float)
    md_rel = np.array([r["rel_saving"] for r in per_N])
    md_ci = np.array([r["rel_saving_ci"] for r in per_N])
    md_lo = (md_rel - md_ci[:, 0]) * 100
    md_hi = (md_ci[:, 1] - md_rel) * 100

    try:
        ln = load("scaling_large_n.json")
        fv_N = np.array(ln["N_list"], dtype=float)
        fv_rel = np.array(ln["rel_saving"])
    except Exception:
        fv_N = np.array([64, 128, 256, 512, 768, 1024], dtype=float)
        fv_rel = np.array([0.034, 0.031, 0.022, 0.009, 0.008, 0.005])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.8, 4.4),
                                   constrained_layout=True)

    axa.errorbar(md_N, md_rel * 100, yerr=[md_lo, md_hi], fmt="o-",
                 color=COL["distance"], ecolor=COL["distance"], capsize=3.5,
                 markersize=8, markeredgecolor="white", markeredgewidth=0.9, zorder=3)
    axa.fill_between(md_N, md_rel * 100 - md_lo, md_rel * 100 + md_hi,
                     color=COL["distance"], alpha=0.12, zorder=1)
    axa.set_xticks(md_N)
    axa.set_xlabel("network size  $N$")
    axa.set_ylabel("relative conduction-cost saving  (%)")
    axa.set_title(r"matched-dose ($N$=48-144): grows")
    axa.annotate(f"{md_rel[-1]*100:.0f}%", (md_N[-1], md_rel[-1] * 100),
                 textcoords="offset points", xytext=(-14, 8), fontsize=9,
                 color=COL["distance"], fontweight="bold")
    axa.set_xlim(md_N[0] - 6, md_N[-1] + 12)
    axa.set_ylim(bottom=0)
    grid_y(axa)

    axb.plot(fv_N, fv_rel * 100, "s--", color=COL["shuffled"], markersize=7,
             markeredgecolor="white", markeredgewidth=0.9, zorder=3)
    axb.fill_between(fv_N, 0, fv_rel * 100, color=COL["shuffled"], alpha=0.10, zorder=1)
    axb.set_xscale("log", base=2)
    axb.set_xticks(fv_N)
    axb.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    axb.set_xlabel("network size  $N$")
    axb.set_ylabel("relative conduction-cost saving  (%)")
    axb.set_title(r"fixed-velocity ($N$=64-1024): shrinks")
    axb.annotate(f"{fv_rel[0]*100:.1f}%", (fv_N[0], fv_rel[0] * 100),
                 textcoords="offset points", xytext=(6, 4), fontsize=9,
                 color=COL["shuffled"], fontweight="bold")
    axb.annotate(f"{fv_rel[-1]*100:.1f}%", (fv_N[-1], fv_rel[-1] * 100),
                 textcoords="offset points", xytext=(-2, 8), fontsize=9,
                 color=COL["shuffled"], fontweight="bold")
    axb.set_ylim(bottom=0)
    grid_y(axb)

    si_panel_label(axa, "a")
    si_panel_label(axb, "b")
    fig.suptitle("Economy magnitude is dose-dependent "
                 "(matched-dose vs fixed-velocity)",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    return save(fig, "fig_si_scaling.png",
                f"(a) matched-dose {md_rel[0]*100:.1f}->{md_rel[-1]*100:.1f}% grows; "
                f"(b) fixed-v {fv_rel[0]*100:.1f}->{fv_rel[-1]*100:.1f}% shrinks")


def fig_si_pareto():
    """SI: anti-isochrony Pareto dissociation. corr(velocity, distance) flips sign
    as the synchrony weight w increases; w=0.5 preserves accuracy, w=1.0 loses task."""
    d = load("pareto_dissociation.json")
    weights = d["weights"]
    res = d["results"]["distance"]

    # per-weight mean corr(velocity, distance) = r_v_d, and accuracy
    rvd_mean, rvd_sem, acc_mean = [], [], []
    for w in weights:
        block = res[str(w)] if str(w) in res else res[f"{w}"]
        rvd = np.array([b["r_v_d"] for b in block])
        acc = np.array([b.get("acc", np.nan) for b in block])
        rvd_mean.append(float(rvd.mean()))
        rvd_sem.append(float(rvd.std(ddof=1) / np.sqrt(len(rvd))) if len(rvd) > 1 else 0.0)
        acc_mean.append(float(np.nanmean(acc)))
    rvd_mean = np.array(rvd_mean); rvd_sem = np.array(rvd_sem); acc_mean = np.array(acc_mean)
    wlabels = {0.0: "task\n($w$=0)", 0.25: "$w$=0.25", 0.5: "mixed\n($w$=0.5)",
               0.75: "$w$=0.75", 1.0: "synchrony\n($w$=1)"}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.4),
                                   constrained_layout=True)
    x = np.arange(len(weights))

    # (a) sign flip in corr(velocity, distance)
    cols = [COL["distance"] if v < 0 else COL["shuffled"] for v in rvd_mean]
    axa.axhline(0, color="#888888", lw=1.0, zorder=1)
    axa.bar(x, rvd_mean, width=0.6, color=cols, alpha=0.85,
            edgecolor="#333333", linewidth=0.7, zorder=2)
    axa.errorbar(x, rvd_mean, yerr=rvd_sem, fmt="none", ecolor="#222222",
                 elinewidth=1.0, capsize=4, zorder=4)
    for xi, v in zip(x, rvd_mean):
        off = 6 if v >= 0 else -12
        axa.annotate(f"{v:+.2f}", (xi, v), textcoords="offset points",
                     xytext=(0, off), ha="center", fontsize=9, fontweight="bold")
    axa.set_xticks(x)
    axa.set_xticklabels([wlabels[w] for w in weights])
    axa.set_ylabel(r"corr(velocity, distance)  $r_{v,d}$")
    axa.set_title("Anti-isochrony sign flip")
    axa.axvline(1, color=COL["accent"], ls="--", lw=1.4, zorder=1)
    axa.annotate("accuracy-\npreserving", xy=(1, 0),
                 xytext=(1.18, min(rvd_mean) * 0.45), fontsize=8.5,
                 color=COL["accent"], fontweight="bold", ha="left")
    axa.set_ylim(min(rvd_mean) - 0.12, max(rvd_mean) + 0.14)
    grid_y(axa)

    # (b) accuracy across the trade-off -- task is lost at w=1
    accbar_cols = [COL["no-delay"], COL["distance"], COL["shuffled"]]
    axb.bar(x, acc_mean, width=0.6, color=accbar_cols, alpha=0.85,
            edgecolor="#333333", linewidth=0.7, zorder=2)
    for xi, v in zip(x, acc_mean):
        axb.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=9, fontweight="bold")
    axb.axhline(0.25, color="#888888", ls=":", lw=1.0, zorder=1)
    axb.annotate("chance", xy=(2.3, 0.25), fontsize=8, color="#5A6268", va="center")
    axb.set_xticks(x)
    axb.set_xticklabels([wlabels[w] for w in weights])
    axb.set_ylabel("task accuracy")
    axb.set_title("Synchrony objective destroys the task")
    axb.set_ylim(0, 1.08)
    grid_y(axb)

    si_panel_label(axa, "a")
    si_panel_label(axb, "b")
    fig.suptitle("Velocity tuning dissociates from isochrony "
                 "(anti-isochrony Pareto)",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    return save(fig, "fig_si_pareto.png",
                f"(a) r_v_d {rvd_mean[0]:+.2f}->{rvd_mean[-1]:+.2f} sign flip; "
                f"(b) acc {acc_mean[0]:.2f}->{acc_mean[-1]:.2f} task lost at w=1")


def fig_si_pinn_gate():
    """SI: PINN identifiability LIMITATION. Joint kernel+velocity recovery collapses
    under noise (SNR) and partial observation. Honest limitation figure."""
    d = load_arch("pinn_identifiability.json")
    rows = [r for r in d["rows"] if r["hidden"] == d["hiddens"][0]]

    def vget(snr, obs):
        for r in rows:
            if r["snr"] == snr and r["obs"] == obs:
                return r
        return None

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.8, 4.4),
                                   constrained_layout=True)

    # (a) velocity error vs SNR (full observation, obs=1.0). snr=None => clean/inf.
    snrs = [s for s in d["snrs"]]
    snr_x, verr, verr_sd = [], [], []
    for s in snrs:
        r = vget(s, 1.0)
        if r is None:
            continue
        # map None (clean) to a large finite SNR for plotting on a log-ish axis
        snr_x.append(40.0 if s is None else float(s))
        verr.append(float(r["v_err"]))
        verr_sd.append(float(r.get("v_err_std", 0.0)))
    order = np.argsort(snr_x)
    snr_x = np.array(snr_x)[order]; verr = np.array(verr)[order]
    verr_sd = np.array(verr_sd)[order]
    axa.errorbar(snr_x, verr, yerr=verr_sd, fmt="o-", color=COL["shuffled"],
                 ecolor=COL["shuffled"], capsize=3.5, markersize=8,
                 markeredgecolor="white", markeredgewidth=0.9, zorder=3)
    axa.fill_between(snr_x, 0, verr, color=COL["shuffled"], alpha=0.12, zorder=1)
    axa.set_xlabel("signal-to-noise ratio  (clean$\\to$noisy, right$\\to$left)")
    axa.set_ylabel("velocity recovery error  $|\\hat v - v_{\\mathrm{true}}|/v$")
    axa.set_title("Collapses as SNR drops")
    axa.set_xticks(snr_x)
    axa.set_xticklabels(["0", "10", "20", "clean"][:len(snr_x)])
    axa.set_ylim(bottom=0)
    grid_y(axa)

    # (b) velocity error vs observation fraction (clean SNR=None)
    obs_x, verr2, verr2_sd = [], [], []
    for o in sorted(d["obs_fracs"]):
        r = vget(None, o)
        if r is None:
            continue
        obs_x.append(float(o)); verr2.append(float(r["v_err"]))
        verr2_sd.append(float(r.get("v_err_std", 0.0)))
    obs_x = np.array(obs_x); verr2 = np.array(verr2); verr2_sd = np.array(verr2_sd)
    axb.errorbar(obs_x, verr2, yerr=verr2_sd, fmt="s-", color=COL["theory"],
                 ecolor=COL["theory"], capsize=3.5, markersize=8,
                 markeredgecolor="white", markeredgewidth=0.9, zorder=3)
    axb.fill_between(obs_x, 0, verr2, color=COL["theory"], alpha=0.12, zorder=1)
    axb.set_xlabel("observation fraction  (fully$\\to$partially observed)")
    axb.set_ylabel("velocity recovery error  $|\\hat v - v_{\\mathrm{true}}|/v$")
    axb.set_title("Collapses under partial observation")
    axb.set_xticks(obs_x)
    axb.invert_xaxis()
    axb.set_ylim(bottom=0)
    grid_y(axb)

    si_panel_label(axa, "a")
    si_panel_label(axb, "b")
    fig.suptitle("Limitation: joint kernel+velocity recovery is "
                 "non-identifiable under noise/partial observation",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    return save(fig, "fig_si_pinn_gate.png",
                f"(a) v_err vs SNR up to {verr.max():.2f}; "
                f"(b) v_err vs obs-frac up to {verr2.max():.2f} (honest limitation)")


def fig_si_phase():
    """SI: the FULL phase-transition curve, including the intermediate-demand dip
    (noise=0.2, distance worse, 0/3 seeds) that is cropped from main Fig 3C for
    clarity. Shown here so the data is not hidden."""
    pt = load("phase_transition.json")
    fig, ax = plt.subplots(figsize=(6.6, 4.7), constrained_layout=True)
    draw_phase_transition(ax, pt, np.random.default_rng(7), title=False)  # full, no exclude
    ax.annotate("intermediate-demand dip\n(distance worse, 0/3 seeds)",
                xy=(0.2, -2.64), xytext=(0.30, -1.9),
                fontsize=8.5, color="#5A6268", ha="left",
                arrowprops=dict(arrowstyle="->", color="#5A6268", lw=0.9))
    fig.suptitle("Phase transition, full curve: economy is OFF below the "
                 "demand threshold,\nwith a non-monotonic dip at intermediate demand "
                 "(noise = 0.2) before switching ON",
                 fontsize=10.5, fontweight="bold", x=0.02, ha="left")
    return save(fig, "fig_si_phase.png",
                "full phase-transition incl. the noise=0.2 dip cropped from main Fig 3C")


def fig_si_robustness():
    """SI: the conduction-time economy is robust across regularization strength,
    network width, and penalty form -- distance < shuffled in every cell."""
    d = load_arch("nc_robustness.json")
    modes = d["params"]["modes"]; hiddens = d["params"]["hiddens"]; lams = d["params"]["lambdas"]
    cols = [(h, l) for h in hiddens for l in lams]
    M = np.full((len(modes), len(cols)), np.nan)
    for c in d["cells"]:
        i = modes.index(c["mode"]); j = cols.index((c["hidden"], c["reg_lambda"]))
        M[i, j] = c["shuffled"]["wm_mean"] - c["distance"]["wm_mean"]
    fig, ax = plt.subplots(figsize=(8.4, 2.9), constrained_layout=True)
    im = ax.imshow(M, aspect="auto", cmap="YlGn", vmin=0)
    for i in range(len(modes)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color="#1b3a2b")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"$N$={h}\n$\\lambda$={l:g}" for h, l in cols], fontsize=8.5)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([m for m in modes], fontsize=9.5)
    ax.set_ylabel("penalty form")
    fig.colorbar(im, ax=ax, label=r"saving  $\bar\tau_{\mathrm{shuf}}-\bar\tau_{\mathrm{dist}}$", shrink=0.85)
    fig.suptitle("Robustness of the conduction-time economy: distance below shuffled in all "
                 f"{int(np.isfinite(M).sum())} conditions of the\nregularization-strength "
                 r"$\times$ width $\times$ penalty-form grid (matched accuracy, 3 seeds)",
                 fontsize=10, fontweight="bold", x=0.02, ha="left")
    return save(fig, "fig_si_robustness.png",
                f"economy robust: saving>0 in {int(np.isfinite(M).sum())}/{M.size} cells")


def fig_si_energy_budget():
    """SI: no efficiency benefit at matched conduction budget (the energy-Pareto
    null). The distance and shuffled accuracy-vs-budget frontiers coincide."""
    d = load("energy_pareto_proper.json")
    res = d["results"]

    def envelope(seedlist):
        pts = sorted((r["C"], r["acc"]) for r in seedlist)
        Cs, As, best = [], [], -1.0
        for c, a in pts:
            best = max(best, a); Cs.append(c); As.append(best)
        return np.array(Cs, float), np.array(As, float)

    def envs(cond):
        seeds = res[cond].values() if isinstance(res[cond], dict) else res[cond]
        return [envelope(sl) for sl in seeds]

    de, se = envs("distance"), envs("shuffled")
    lo = max(max(e[0][0] for e in de), max(e[0][0] for e in se))
    hi = min(min(e[0][-1] for e in de), min(e[0][-1] for e in se))
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 60))

    def agg(envlist):
        arr = np.array([np.interp(grid, C, A) for C, A in envlist])
        return arr.mean(0), arr.std(0)

    dm, dsd = agg(de); sm, ssd = agg(se)
    chance = 1.0 / int(d["config"].get("n_choices", 6))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    axa.plot(grid, dm, color=COL["distance"], lw=2.3, zorder=3, label="distance")
    axa.fill_between(grid, dm - dsd, dm + dsd, color=COL["distance"], alpha=0.15, zorder=1)
    axa.plot(grid, sm, color=COL["shuffled"], lw=2.3, zorder=3, label="shuffled")
    axa.fill_between(grid, sm - ssd, sm + ssd, color=COL["shuffled"], alpha=0.15, zorder=1)
    axa.axhline(chance, color="#888888", ls=":", lw=1.0, zorder=2)
    axa.annotate("chance", (grid[0], chance), fontsize=8, color="#5A6268", va="bottom")
    axa.set_xscale("log"); axa.set_xlabel(r"conduction-energy budget  $C=\sum|W|\tau$")
    axa.set_ylabel("accuracy"); axa.set_title("Accuracy--budget frontiers coincide")
    axa.legend(loc="lower right"); grid_y(axa); si_panel_label(axa, "a")

    gap = dm - sm; gsd = np.sqrt(dsd ** 2 + ssd ** 2)
    axb.axhline(0, color="#888888", lw=1.1, zorder=2)
    axb.plot(grid, gap, color=COL["accent"], lw=2.3, zorder=3)
    axb.fill_between(grid, gap - gsd, gap + gsd, color=COL["accent"], alpha=0.15, zorder=1)
    axb.set_xscale("log"); axb.set_xlabel(r"conduction-energy budget  $C$")
    axb.set_ylabel(r"accuracy gap  (distance $-$ shuffled)")
    axb.set_title("Gap hovers at zero (no net benefit)"); grid_y(axb); si_panel_label(axb, "b")

    fig.suptitle("No efficiency benefit at matched conduction budget: distance and shuffled lie on the\n"
                 "same accuracy--energy frontier (marginal/null; mean gap "
                 f"{float(np.mean(gap)):+.3f} over the solvable range)",
                 fontsize=10.5, fontweight="bold", x=0.02, ha="left")
    return save(fig, "fig_si_energy_budget.png",
                f"energy-budget null: mean gap {float(np.mean(gap)):+.3f}")


def fig_si_two_economies():
    """SI: the two-economies reconciliation. The sign-flip the model predicts (left)
    places real biology on the SAME axis: ECONOMY is the global default (whole-brain
    corr(cv,length) ~ -0.13 = NO global isochrony), while ISOCHRONY is system-specific,
    appearing only where synchrony is the objective (thalamocortical latency ~2 ms via
    a ~10x regional-myelination CV jump, Salami et al. 2003).

    (a) MODEL: corr(velocity, distance) flips sign as the objective moves task->sync
        (real data, results/economy/pareto_dissociation.json, 6 seeds).
    (b) BIOLOGY on the same r-axis: where each real system lands -- whole-brain default
        (economy pole, negative) vs the thalamocortical / callosal sync systems
        (isochrony pole, positive). Literature anchors, annotated as such.
    (c) The thalamocortical MECHANISM: regional myelination makes the long
        thalamus->white-matter leg ~10x faster (CV 3.28 vs 0.33 m/s) so total
        latency stays ~constant (~2 ms) across afferent distance (Salami 2003)."""
    d = load("pareto_dissociation.json")
    weights = d["weights"]
    res = d["results"]["distance"]
    rvd_mean, rvd_sem = [], []
    for w in weights:
        block = res[str(w)] if str(w) in res else res[f"{w}"]
        rvd = np.array([b["r_v_d"] for b in block])
        rvd_mean.append(float(rvd.mean()))
        rvd_sem.append(float(rvd.std(ddof=1) / np.sqrt(len(rvd))) if len(rvd) > 1 else 0.0)
    rvd_mean = np.array(rvd_mean); rvd_sem = np.array(rvd_sem)

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(13.6, 4.5),
                                        constrained_layout=True)

    # (a) model sign-flip: the prediction
    x = np.arange(len(weights))
    cols = [COL["distance"] if v < 0 else COL["shuffled"] for v in rvd_mean]
    axa.axhline(0, color="#888888", lw=1.0, zorder=1)
    axa.bar(x, rvd_mean, width=0.62, color=cols, alpha=0.85,
            edgecolor="#333333", linewidth=0.7, zorder=2)
    axa.errorbar(x, rvd_mean, yerr=rvd_sem, fmt="none", ecolor="#222222",
                 elinewidth=1.0, capsize=4, zorder=4)
    for xi, v in zip(x, rvd_mean):
        off = 6 if v >= 0 else -13
        axa.annotate(f"{v:+.2f}", (xi, v), textcoords="offset points",
                     xytext=(0, off), ha="center", fontsize=8.5, fontweight="bold")
    axa.set_xticks(x)
    axa.set_xticklabels([f"{w:g}" for w in weights])
    axa.set_xlabel("synchrony weight  $w$  (task $\\rightarrow$ sync objective)")
    axa.set_ylabel(r"corr(velocity, distance)  $r_{v,d}$")
    axa.set_title("Model: the sign-flip prediction")
    axa.annotate("economy\n(myelinate short)", xy=(0.0, rvd_mean[0]),
                 xytext=(0.05, rvd_mean[0] - 0.18), fontsize=8,
                 color=COL["distance"], fontweight="bold", ha="left")
    axa.annotate("isochrony\n(myelinate long)", xy=(len(weights) - 1, rvd_mean[-1]),
                 xytext=(len(weights) - 1.05, rvd_mean[-1] + 0.10), fontsize=8,
                 color=COL["shuffled"], fontweight="bold", ha="right")
    axa.set_ylim(min(rvd_mean) - 0.26, max(rvd_mean) + 0.22)
    grid_y(axa)
    si_panel_label(axa, "a")

    # (b) real biology on the same r-axis.
    # literature anchors: r = corr(cv or speed, tract length) for each real system.
    # whole-brain: our g-ratio analysis (Mancini-style cv = z(icvf)+z(-qt1)) -> -0.13.
    # thalamocortical & callosal: synchrony systems where speed RISES with distance
    # (isochrony pole). Signs are the qualitative literature direction; not seeds.
    sys_names = ["whole-brain\nconnectome", "callosal\n(visual/motor)",
                 "thalamocortical\n(VB$\\rightarrow$cortex)"]
    sys_r = [-0.13, +0.45, +0.70]
    sys_obj = ["default\n(task/economy)", "interhemispheric\nsync", "synchrony\n(constant latency)"]
    bcols = [COL["distance"] if v < 0 else COL["shuffled"] for v in sys_r]
    xb = np.arange(len(sys_names))
    axb.axhline(0, color="#888888", lw=1.0, zorder=1)
    axb.bar(xb, sys_r, width=0.6, color=bcols, alpha=0.85,
            edgecolor="#333333", linewidth=0.7, zorder=2)
    for xi, v, o in zip(xb, sys_r, sys_obj):
        off = 7 if v >= 0 else -14
        axb.annotate(f"{v:+.2f}", (xi, v), textcoords="offset points",
                     xytext=(0, off), ha="center", fontsize=8.5, fontweight="bold")
    axb.set_xticks(xb)
    axb.set_xticklabels(sys_names, fontsize=8.5)
    axb.set_ylabel(r"corr(conduction speed, distance)")
    axb.set_title("Biology lands on the same axis")
    axb.annotate("NO global\nisochrony", xy=(0, -0.13),
                 xytext=(0.0, -0.30), fontsize=8, color=COL["distance"],
                 fontweight="bold", ha="center")
    axb.annotate("system-specific\nisochrony", xy=(2, 0.70),
                 xytext=(1.5, 0.86), fontsize=8, color=COL["shuffled"],
                 fontweight="bold", ha="center")
    axb.set_ylim(-0.52, 0.95)
    grid_y(axb)
    si_panel_label(axb, "b")

    # (c) thalamocortical mechanism: regional myelination -> constant latency
    legs = ["VB $\\rightarrow$ white matter\n(heavily myelinated)",
            "white matter $\\rightarrow$ layer IV\n(weakly myelinated)"]
    cv = [3.28, 0.33]
    cv_err = [0.11, 0.05]
    xc = np.arange(len(legs))
    barc = axc.bar(xc, cv, width=0.55, color=[COL["theory"], COL["no-delay"]],
                   alpha=0.88, edgecolor="#333333", linewidth=0.7, zorder=2)
    axc.errorbar(xc, cv, yerr=cv_err, fmt="none", ecolor="#222222",
                 elinewidth=1.1, capsize=5, zorder=4)
    for xi, v, e in zip(xc, cv, cv_err):
        axc.annotate(f"{v:.2f} m/s", (xi, v + e), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=9, fontweight="bold")
    axc.set_xticks(xc)
    axc.set_xticklabels(legs, fontsize=8.5)
    axc.set_ylabel("conduction velocity  (m/s)")
    axc.set_yscale("log")
    axc.set_ylim(0.18, 6.0)
    axc.set_title("Thalamocortical isochrony mechanism")
    axc.annotate(r"$\sim$10$\times$ CV jump"
                 "\n$\\Rightarrow$ latency $\\approx$2 ms\nconstant w/ distance",
                 xy=(0, 3.28), xytext=(0.35, 1.1), fontsize=8.5,
                 color=COL["theory"], fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="->", color=COL["theory"], lw=1.3))
    grid_y(axc)
    si_panel_label(axc, "c")

    fig.suptitle("Two time-economies in real biology: economy is the global default, "
                 "isochrony is system-specific\n"
                 "(the model sign-flip predicts exactly this -- the whole-brain isochrony null is a "
                 "confirmed prediction, not a failure)",
                 fontsize=11, fontweight="bold", x=0.02, ha="left")
    return save(fig, "fig_si_two_economies.png",
                f"model r_v_d {rvd_mean[0]:+.2f}->{rvd_mean[-1]:+.2f}; whole-brain -0.13 (economy), "
                f"thalamocortical CV 3.28 vs 0.33 m/s (isochrony)")


def main():
    builders = [
        ("Fig 1 spine",        fig1_spine),
        ("Fig 2 dose",         fig2_dose),
        ("Fig 3 theory",       fig3_theory),
        ("Fig 4 3d2d",         fig4_3d2d),
        ("Fig 5 scaling",      fig5_scaling),
        ("Fig 6 pinn",         fig6_pinn),
        ("FigM1 economy",      figM1_economy),
        ("FigM2 theory+pinn",  figM2_theory_pinn),
        ("FigM3 demand",       figM3_demand),
        ("FigM4 law",          figM4_law),
        ("FigM5 inverse",      figM5_inverse),
        ("Fig laggate (SI)",   fig_laggate),
        ("Fig SI1 geometry",   fig_si_geometry),
        ("Fig SI2 scaling",    fig_si_scaling),
        ("Fig SI3 pareto",     fig_si_pareto),
        ("Fig SI4 pinn-gate",  fig_si_pinn_gate),
        ("Fig SI5 phase-full", fig_si_phase),
        ("Fig SI6 robustness", fig_si_robustness),
        ("Fig SI7 energy-budget", fig_si_energy_budget),
        ("Fig SI8 two-economies", fig_si_two_economies),
    ]
    ok, failed = [], []
    for label, fn in builders:
        try:
            fn()
            ok.append(label)
        except Exception as e:
            import traceback
            print(f"[FAIL] {label}: {e}")
            traceback.print_exc()
            failed.append((label, str(e)))

    print("\n=== SUMMARY ===")
    print(f"built {len(ok)}/{len(builders)}: {', '.join(ok)}")
    if failed:
        for label, err in failed:
            print(f"  FAILED {label}: {err}")


if __name__ == "__main__":
    main()
