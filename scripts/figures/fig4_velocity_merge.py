"""Fig 4 velocity-allocation panels d-e + an SI triangle-violation dose panel.

  d) conduction time as a binding constraint (plastic vs matched-budget uniform)
  e) objective sets the allocation (corr(v,distance) sign-flip: economy <-> isochrony)
Numbers are exact run values (plastic_velocity_sweep_gpu, pareto_dissociation 6-seed,
dose_response_metric).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 11, "axes.linewidth": 1.0, "svg.fonttype": "none"})
TEAL, GREY, DARK, CORAL, INK = "#1b9e8f", "#9aa0a6", "#2b3a4a", "#e0813f", "#222222"

# data (exact run values)
deadline = np.array([4, 6, 8, 10, 14]);  plastic = np.array([0.526, 0.890, 1.000, 1.000, 1.000])
uniform = np.array([0.499, 0.500, 0.617, 0.807, 0.965]);  tvals = ["t=1.1", "t=20.3", "t=11.2", "t=8.3", "t=1.7"]
w = np.array([0.0, 0.25, 0.5, 0.75, 1.0]);  rvd = np.array([-0.708, 0.164, 0.262, 0.306, 0.327])
scr = np.array([0, 25, 50, 75, 100]) / 100;  rmse = np.array([0.035, 0.147, 0.217, 0.399, 0.674])


def style(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(length=4)


# Fig 4 d-e (velocity-allocation row)
fig, (axd, axe) = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=330)

# d) binding constraint
axd.fill_between(deadline, uniform, plastic, color=TEAL, alpha=0.13, zorder=1)
axd.plot(deadline, plastic, "-o", color=TEAL, lw=2.6, mfc="white", mec=TEAL, mew=2.0, ms=8,
         label="learnable velocity", zorder=3)
axd.plot(deadline, uniform, "-o", color=GREY, lw=2.4, mfc="white", mec=GREY, mew=2.0, ms=7,
         label="uniform (matched budget)", zorder=2)
for x, y, t in zip(deadline, plastic, tvals):
    axd.annotate(t, (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
                 fontsize=9, style="italic", color=TEAL if t not in ("t=1.1",) else GREY)
axd.annotate("allocation\nload-bearing", (9.2, 0.775), fontsize=10.5, fontweight="bold", color=TEAL, ha="center")
axd.set_xlabel("deadline (allowed arrival time)"); axd.set_ylabel("held-out accuracy")
axd.set_title("Conduction time as a binding constraint", fontsize=12.5, fontweight="bold", color=INK, pad=10)
axd.set_xticks(deadline); axd.set_ylim(0.45, 1.11); axd.legend(loc="lower right", frameon=False, fontsize=9.5)
style(axd)

# e) objective sets the allocation (sign-flip)
axe.axhline(0, color=GREY, ls=(0, (4, 3)), lw=1.0, zorder=1)
axe.plot(w, rvd, "-o", color=DARK, lw=2.6, mfc=DARK, mec=DARK, ms=7, zorder=3)
axe.plot(w[0], rvd[0], "o", color=TEAL, ms=12, zorder=4)       # economy pole
axe.plot(w[-1], rvd[-1], "o", color=CORAL, ms=12, zorder=4)    # isochrony pole
axe.annotate("short edges\n(economy)", (w[0], rvd[0]), textcoords="offset points", xytext=(18, 6),
             fontsize=10, color=TEAL, fontweight="bold")
axe.annotate("long edges\n(isochrony)", (w[-1], rvd[-1]), textcoords="offset points", xytext=(-12, 18),
             fontsize=10, color=CORAL, fontweight="bold", ha="right")
axe.set_xlabel("synchrony objective weight"); axe.set_ylabel("corr(velocity, distance)")
axe.set_title("The objective sets the allocation", fontsize=12.5, fontweight="bold", color=INK, pad=10)
axe.set_xticks(w); axe.set_ylim(-0.88, 0.62); style(axe)

fig.tight_layout(w_pad=3)
for ext in ["png", "pdf", "svg"]:
    fig.savefig(f"figures/fig4_de_velocity_alloc.{ext}", bbox_inches="tight", facecolor="white")
print("wrote figures/fig4_de_velocity_alloc.{png,pdf,svg}")

# SI: metric-consistency is graded
figs, axs = plt.subplots(figsize=(5.4, 4.3), dpi=200)
axs.plot(scr * 100, rmse, "-o", color=CORAL, lw=2.6, mfc="white", mec=CORAL, mew=2.0, ms=8)
axs.text(8, 0.60, "corr = 0.81", fontsize=13, fontweight="bold", color=CORAL)
axs.text(30, 0.135, "decoding fails as the\ntriangle inequality breaks", fontsize=9.5, style="italic", color=GREY)
axs.set_xlabel("delay entries scrambled (%)"); axs.set_ylabel("held-out localization RMSE")
axs.set_title("Metric consistency is graded", fontsize=12.5, fontweight="bold", color=INK, pad=10)
axs.set_xticks([0, 25, 50, 75, 100]); axs.set_ylim(0, 0.72); style(axs)
figs.tight_layout()
for ext in ["png", "pdf", "svg"]:
    figs.savefig(f"figures/fig_si_metric_graded.{ext}", bbox_inches="tight", facecolor="white")
print("wrote figures/fig_si_metric_graded.{png,pdf,svg}")
