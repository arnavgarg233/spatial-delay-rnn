"""Functional-results dumbbell plot: ordered vs entry-shuffle vs none, wins + boundary nulls.

Per task, held-out performance is oriented higher=better (RMSE negated) and normalized so
none=0 (chance), ordered=1. The orange segment is the metric-consistency gap (loss from
scrambling the metric while keeping the same delay histogram).

TODO: some numbers (flashlag, the 1-seed nulls) are approximate; wire to exact
results/experiments/*.json before submission.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5, "axes.linewidth": 0.8,
})
BLUE, WARM, GREY, INK = "#2c6e9b", "#e0813f", "#9aa0a6", "#2b2b2b"

# task, metric ('acc' higher=better / 'rmse' lower=better), ordered, entry, none, is_win
TASKS = [
    ("Localization",  "rmse", 0.028, 0.638, 0.405, True),
    ("Relational",    "acc",  0.90,  0.64,  0.50,  True),
    ("Relay",         "acc",  0.882, 0.603, 0.656, True),
    ("Yin-Yang",      "acc",  0.827, 0.442, 0.481, True),
    ("Takens embed.", "acc",  0.608, 0.608, 0.559, False),
    ("SHD benchmark", "acc",  0.623, 0.625, 0.669, False),
    ("Wave transport","acc",  0.932, 0.935, 0.942, False),
    ("Temporal comp.","acc",  0.280, 0.285, 0.119, False),
]

def norm(metric, val, ordd, none):
    s  = (lambda x: -x) if metric == "rmse" else (lambda x: x)
    denom = s(ordd) - s(none)
    if abs(denom) < 1e-9: return 0.5
    return (s(val) - s(none)) / denom        # none->0, ordered->1

rows = []
for name, m, o, e, n, win in TASKS:
    rows.append((name, 0.0, norm(m, e, o, n), 1.0, win))   # (name, none, entry, ordered, win)

fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)

# y layout: wins on top, gap, nulls below
ys = []
y = len(rows) + 1.0
for i, r in enumerate(rows):
    if i == 4: y -= 1.0          # gap between wins and nulls
    ys.append(y); y -= 1.0
ys = np.array(ys)

# reference verticals
ax.axvline(0.0, color=GREY, lw=1.0, ls=(0, (4, 2)), zorder=1)
ax.axvline(1.0, color=BLUE, lw=1.0, ls=(0, (4, 2)), zorder=1)

for (name, none, entry, ordd, win), yy in zip(rows, ys):
    ax.plot([none, ordd], [yy, yy], color="#e6e6e6", lw=1.4, zorder=2)          # baseline none->ordered
    seg = WARM if win else "#d9d9d9"
    ax.plot([entry, ordd], [yy, yy], color=seg, lw=4.2, solid_capstyle="round",  # metric-consistency gap
            alpha=0.9, zorder=3)
    ax.scatter(none, yy, s=34, facecolor="white", edgecolor=GREY, lw=1.3, zorder=4)
    ax.scatter(entry, yy, s=46, facecolor=WARM if win else GREY, edgecolor="white", lw=0.8, zorder=5)
    ax.scatter(ordd, yy, s=52, facecolor=BLUE, edgecolor="white", lw=0.8, zorder=6)

ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
for tick, win in zip(ax.get_yticklabels(), [r[4] for r in rows]):
    tick.set_color(INK if win else GREY)
ax.set_xlim(-0.78, 1.2); ax.set_xticks([0, 0.5, 1.0])
ax.set_xticklabels(["none\n(chance)", "", "ordered"], fontsize=8.5)
ax.set_xlabel("held-out skill   (normalized: none = 0  →  ordered = 1)", fontsize=9)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.tick_params(left=False)

# group brackets / labels
ax.text(-0.74, ys[:4].mean(), "4 WINS", rotation=90, va="center", ha="center",
        fontsize=9.5, fontweight="bold", color=BLUE)
ax.text(-0.74, ys[4:].mean(), "4 BOUNDARY\nNULLS", rotation=90, va="center", ha="center",
        fontsize=8.6, fontweight="bold", color=GREY)
ax.axhline((ys[3] + ys[4]) / 2, color="#dddddd", lw=0.8)

# legend
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker='o', color='w', markerfacecolor=BLUE, markersize=8, label='ordered (metric)'),
       Line2D([0],[0], marker='o', color='w', markerfacecolor=WARM, markersize=8, label='entry-shuffle (same delays, scrambled)'),
       Line2D([0],[0], marker='o', color='w', markerfacecolor='white', markeredgecolor=GREY, markersize=8, label='none (no delays)')]
ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3,
          frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.2)
ax.set_title("Metric-consistency helps exactly where geometry must be read through the delay metric",
             fontsize=10.5, color=INK, pad=12)
# one short read-cue per group, in clear margin space
ax.text(-0.40, ys[2], "long bar →\nmetric needed", fontsize=7.4, color=WARM, ha="center", va="center")
ax.text(1.13, ys[6] + 0.5, "no bar →\nmetric irrelevant", fontsize=7.4, color=GREY, ha="center", va="center")

fig.savefig("figures/fig_functional.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("figures/fig_functional.pdf", bbox_inches="tight", facecolor="white")
print("wrote figures/fig_functional.png + .pdf")
