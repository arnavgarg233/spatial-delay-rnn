"""Reconstruct colab_yinyang_results.json from the Colab CONSOLE output.

The Colab run timed out before its auto-save, so this transcribes the per-seed
held-out accuracies VERBATIM from the printed console (3-decimal precision) and
recomputes the summary. The means/gaps/wins this produces are checked against the
values the run itself printed (they match exactly). The t-statistics are stored as
PRINTED by the run -- they cannot be reproduced exactly from 3-decimal values, so a
recompute-from-rounded is also stored for transparency. The coord4 block is omitted
(it was wired incorrectly -> a no-op duplicate of spatial/free; not a real result).
"""
import json, math, statistics as st

# Per-seed HELD-OUT accuracy, transcribed from console. Order: ordered, entry, index, none
FREE = [
    (0.923, 0.385, 0.808, 0.423),
    (0.731, 0.538, 0.885, 0.423),
    (0.808, 0.346, 0.692, 0.577),
    (0.923, 0.346, 0.731, 0.577),
    (0.885, 0.462, 0.577, 0.500),
    (0.692, 0.577, 0.769, 0.385),
]
TIED = [
    (0.846, 0.423, 0.808, 0.423),
    (0.769, 0.538, 0.846, 0.423),
    (0.731, 0.346, 0.692, 0.577),
    (0.846, 0.308, 0.577, 0.577),
    (0.846, 0.385, 0.615, 0.500),
    (0.654, 0.577, 0.808, 0.385),
]
PRINTED = {  # exactly what the run printed, for cross-check
    "free": dict(ordered=0.827, entry=0.442, index=0.744, none=0.481, gap=0.385, t=5.48, wins=6, idx=-0.083),
    "tied": dict(ordered=0.782, entry=0.429, index=0.724, none=0.481, gap=0.353, t=5.59, wins=6, idx=-0.058),
}

def block(rows, name):
    ordd = [r[0] for r in rows]; ent = [r[1] for r in rows]
    idx = [r[2] for r in rows]; non = [r[3] for r in rows]
    gaps = [o - e for o, e in zip(ordd, ent)]
    mean = sum(gaps) / len(gaps)
    sd = st.stdev(gaps)
    t_round = mean / (sd / math.sqrt(len(gaps)))
    wins = sum(1 for g in gaps if g > 0)
    out = {
        "ordered_heldout_acc": round(sum(ordd)/len(ordd), 4),
        "entryshuffle_heldout_acc": round(sum(ent)/len(ent), 4),
        "index_perm_heldout_acc": round(sum(idx)/len(idx), 4),
        "none_heldout_acc": round(sum(non)/len(non), 4),
        "heldout_gap_ordered_minus_entryshuf": {
            "mean": round(mean, 4), "sd": round(sd, 4),
            "t_PRINTED_by_run": PRINTED[name]["t"],
            "t_recomputed_from_3dp": round(t_round, 3),
            "wins": wins,
        },
        "index_perm_sanity_gap_index_minus_ordered": round(sum(idx)/len(idx) - sum(ordd)/len(ordd), 4),
        "per_seed_heldout_ordered_entry_index_none": [list(r) for r in rows],
        "verdict": "SIGNAL",
    }
    # cross-check against printed
    p = PRINTED[name]
    chk = (abs(out["ordered_heldout_acc"]-p["ordered"])<1e-3 and
           abs(out["entryshuffle_heldout_acc"]-p["entry"])<1e-3 and
           abs(out["index_perm_heldout_acc"]-p["index"])<1e-3 and
           abs(out["none_heldout_acc"]-p["none"])<1e-3 and
           out["heldout_gap_ordered_minus_entryshuf"]["wins"]==p["wins"])
    print(f"[{name}] means match printed: {chk}  (recomputed t={t_round:.2f} vs printed {p['t']})")
    return out

res = {
    "task": "yinyang",
    "PROVENANCE": ("RECONSTRUCTED from Colab console output (run timed out before auto-save). "
                   "Per-seed held-out accuracies transcribed verbatim (3dp). Means/gaps/wins "
                   "computed here and verified == values printed by the run. t-stats are as "
                   "PRINTED (cannot be reproduced exactly from 3dp; recompute_from_3dp shown). "
                   "coord4 cross-check block omitted: it was a no-op duplicate of spatial/free."),
    "device": "cuda", "seeds": 6,
    "class_fractions": {"yin": 0.461, "yang": 0.447, "dots": 0.092},
    "config": {"N": 64, "v": 0.10, "maxd": 16, "steps": 1200},
    "spatial_free": block(FREE, "free"),
    "spatial_tied": block(TIED, "tied"),
    "verdict": ("SIGNAL on BOTH readouts (free t=5.48, tied t=5.59; 6/6 each; index-perm "
                "tracks ordered, none at chance). Metric-scramble (entry-shuffle) collapses "
                "held-out Yin-Yang classification; geometric delays required."),
}
with open("results/yinyang_colab.json", "w") as f:
    json.dump(res, f, indent=2)
print("wrote results/yinyang_colab.json")
