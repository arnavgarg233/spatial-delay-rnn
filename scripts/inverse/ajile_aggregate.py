"""Aggregate the per-subject AJILE12 results (foreigninv_ajile_sub*.json) into a POPULATION verdict:
does the true (metric) electrode geometry beat the shuffled-geometry null across subjects?"""
import json, glob, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
files = sorted(glob.glob(str(ROOT / "results" / "inverse" / "foreigninv_ajile_sub*.json")))
rows = []
for f in files:
    d = json.load(open(f))
    if d.get("rows"):
        rows.append(d["rows"][0])

n = len(rows)
if n == 0:
    print("no subject results yet"); sys.exit(0)

gaps = [r["null_gap"] for r in rows]
beats = [bool(r["null_beats_all"]) for r in rows]
relg = [r["relgap_max"] for r in rows]
mean = sum(gaps) / n
sd = (sum((g - mean) ** 2 for g in gaps) / max(n - 1, 1)) ** 0.5
t = mean / (sd / math.sqrt(n) + 1e-12)
n_pos = sum(1 for g in gaps if g > 0)
n_beats = sum(beats)
pop_sig = (t > 2.0 and mean > 0) or (n_pos >= n - 1 and n >= 6)
verdict = ("POPULATION SIGNAL: metric geometry beats the shuffle on real ECoG across subjects"
           if pop_sig else
           "POPULATION NULL: metric organization substitutable on real ECoG (strengthens structural thesis)")

print("=" * 64)
print(f"AJILE12 POPULATION ({n}/12 subjects)")
for r in rows:
    print(f"  {r['sub']}: gap={r['null_gap']:+.4f}  ratio={r['null_ratio']:.2f}x  "
          f"beats_all={r['null_beats_all']}  relgap_max={r['relgap_max']*100:.1f}%  vhat={r.get('v_hat_mps',0):.3f}")
print("-" * 64)
print(f"  mean gap = {mean:+.4f} ± {sd:.4f}   t = {t:.2f}")
print(f"  true-beats-all-shuffles: {n_beats}/{n}   |   gap>0: {n_pos}/{n}   |   relgap_max mean: {sum(relg)/n*100:.1f}%")
print(f"  VERDICT: {verdict}")
print("=" * 64)

out = dict(n_subjects=n, gap_mean=mean, gap_sd=sd, gap_t=t, n_beats_all=n_beats,
           n_pos_gap=n_pos, relgap_max_mean=sum(relg) / n, verdict=verdict, rows=rows)
json.dump(out, open(ROOT / "results" / "inverse" / "foreigninv_ajile_population.json", "w"), indent=2)
print("wrote results/inverse/foreigninv_ajile_population.json")
