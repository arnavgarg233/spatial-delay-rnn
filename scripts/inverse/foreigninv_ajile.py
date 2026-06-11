"""AJILE12 POPULATION metric-vs-shuffle test (the deep-research #1 ceiling-raiser).

Streams the public AJILE12 human ECoG dataset (DANDI 000055; 12 subjects, NWB, 500 Hz, electrode
x/y/z coordinates in the table) and, FOR EACH SUBJECT, runs the EXACT same delayed-field conduction-
velocity inverse + velocity-shuffled-geometry NULL used on the Utah-array data (reuses
foreigninv_realdata.run_dataset). The single-subject intracranial result barely beat the geometry
shuffle; the point here is POPULATION POWER: across N independent subjects, does the metric-consistent
(true) geometry fit the delayed field better than a metric-broken (shuffled-geometry) control,
consistently? A clean population sign/t-test either way is publishable:
  - true beats shuffle across subjects  -> metric-consistent delays predict REAL neural data (ceiling).
  - no population gap                    -> substitutability holds on real brains (strengthens thesis).

Streams only a window per subject via remfile (no multi-GB download). FS=500 matches the module default.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "inverse"))
import foreigninv_realdata as rd  # run_dataset, preprocess_field, shuffle_geometry, FS=500


def stream_subject(asset, win_s, skip_s, seg_s, max_chan):
    """Stream one NWB asset; return (ts=(C,T,S), xyz=(S,3) mm) for good channels."""
    import remfile, h5py, pynwb
    url = asset.get_content_url(follow_redirects=1, strip_query=True)
    f = remfile.File(url)
    with h5py.File(f, "r") as h5:
        with pynwb.NWBHDF5IO(file=h5, mode="r", load_namespaces=True) as io:
            nwb = io.read()
            et = nwb.electrodes
            cols = list(et.colnames)
            xyz = np.stack([et["x"][:], et["y"][:], et["z"][:]], axis=1).astype(np.float64)  # (S,3) mm
            good = et["good"][:].astype(bool) if "good" in cols else np.ones(len(et), bool)
            # finite coords only
            good &= np.isfinite(xyz).all(axis=1)
            es = nwb.acquisition["ElectricalSeries"]
            fs = float(es.rate) if es.rate else 500.0
            n_total = es.data.shape[0]
            start = int(skip_s * fs)
            nsamp = int(win_s * fs)
            stop = min(start + nsamp, n_total)
            data = es.data[start:stop, :].astype(np.float64)  # (T_win, S) - remfile fetches the slice
    # keep good channels
    gi = np.where(good)[0]
    if max_chan and len(gi) > max_chan:
        gi = gi[np.linspace(0, len(gi) - 1, max_chan).astype(int)]
    data = data[:, gi]
    xyz = xyz[gi]
    # drop dead channels (zero variance)
    v = data.std(0)
    keep = v > (np.median(v) * 1e-3)
    data, xyz = data[:, keep], xyz[keep]
    # segment into (C, T, S)
    T = int(seg_s * fs)
    C = data.shape[0] // T
    data = data[: C * T].reshape(C, T, data.shape[1])
    return data, xyz, fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=12, dest="n_subjects")
    ap.add_argument("--win-s", type=float, default=200.0, dest="win_s")
    ap.add_argument("--skip-s", type=float, default=120.0, dest="skip_s")
    ap.add_argument("--seg-s", type=float, default=1.0, dest="seg_s")
    ap.add_argument("--max-chan", type=int, default=84, dest="max_chan")
    ap.add_argument("--max-delay", type=int, default=60, dest="max_delay")
    ap.add_argument("--min-delay", type=int, default=1, dest="min_delay")
    ap.add_argument("--n-cands", type=int, default=17, dest="n_cands")
    ap.add_argument("--v-lo", type=float, default=0.05, dest="v_lo")
    ap.add_argument("--v-hi", type=float, default=2.0, dest="v_hi")
    ap.add_argument("--ridge", type=float, default=1e-1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=20, dest="n_boot")
    ap.add_argument("--n-shuffles", type=int, default=20, dest="n_shuffles")
    ap.add_argument("--sub-idx", type=int, default=-1, dest="sub_idx",
                    help="run ONLY this subject index (for parallel fan-out); -1 = all")
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "inverse" / "foreigninv_ajile.json"))
    args = ap.parse_args()

    from dandi.dandiapi import DandiAPIClient
    t0 = time.time()
    print(f"AJILE12 population metric-vs-shuffle | n_subjects={args.n_subjects} win={args.win_s}s "
          f"seg={args.seg_s}s n_shuffles={args.n_shuffles}", flush=True)
    with DandiAPIClient() as c:
        ds = c.get_dandiset("000055", "draft")
        assets = [a for a in ds.get_assets() if a.path.endswith(".nwb")]
    # one (smallest) session per subject
    by_sub = {}
    for a in assets:
        sub = a.path.split("/")[0]
        if sub not in by_sub or a.size < by_sub[sub].size:
            by_sub[sub] = a
    subs = sorted(by_sub)[: args.n_subjects]
    if args.sub_idx >= 0:                       # parallel fan-out: one subject per process
        subs = [subs[args.sub_idx]]
        args.out = str(ROOT / "results" / "inverse" / f"foreigninv_ajile_sub{args.sub_idx:02d}.json")
    print(f"subjects: {subs}  out={args.out}", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for sub in subs:
        try:
            ts, xyz, fs = stream_subject(by_sub[sub], args.win_s, args.skip_s, args.seg_s, args.max_chan)
            rd.FS = fs  # calibrate the module's sample->m/s conversion to this subject's rate
            print(f"\n### {sub}: ts{ts.shape} xyz{xyz.shape} fs={fs:.0f}", flush=True)
            r = rd.run_dataset(sub, ts, xyz, args, rng)
            rows.append({"sub": sub, "n_sensors": r["n_sensors"], "n_trials": r["n_trials"],
                         "v_hat_mps": r["v_hat_mps"], "v_ref_mps": r["v_ref_mps"],
                         "floor_true": r["floor_true"], "null_floor_mean": r["null_floor_mean"],
                         "null_gap": r["null_gap"], "null_ratio": r["null_ratio"],
                         "null_beats_all": r["null_beats_all"], "relgap_max": r["relgap_max"],
                         "interior_min": r["interior_min"], "dip": r["dip"]})
            print(f"   -> gap={r['null_gap']:+.4f} ratio={r['null_ratio']:.3f}x "
                  f"beats_all={r['null_beats_all']} relgap_max={r['relgap_max']*100:.1f}%", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"   !! {sub} failed: {e}", flush=True)

    # ---- POPULATION aggregation ----
    gaps = np.array([x["null_gap"] for x in rows], float)
    beats = np.array([x["null_beats_all"] for x in rows], bool)
    relgaps = np.array([x["relgap_max"] for x in rows], float)
    n = len(rows)
    from math import sqrt
    t_gap = float(gaps.mean() / (gaps.std(ddof=1) / sqrt(n) + 1e-12)) if n > 1 else float("nan")
    # sign test: P(gap>0) under null = binomial(n, 0.5)
    n_pos = int((gaps > 0).sum())
    out = {"config": vars(args), "minutes": round((time.time() - t0) / 60, 1),
           "n_subjects_run": n, "rows": rows,
           "gap_mean": float(gaps.mean()) if n else None, "gap_sd": float(gaps.std()) if n else None,
           "gap_t": t_gap, "n_true_beats_all": int(beats.sum()),
           "n_pos_gap": n_pos, "relgap_max_mean": float(relgaps.mean()) if n else None,
           "verdict": None}
    if n:
        pop_sig = (t_gap > 2.0 and gaps.mean() > 0) or (n_pos >= n - 1 and n >= 4)
        out["verdict"] = ("POPULATION SIGNAL: metric geometry beats the shuffle on real ECoG across subjects"
                          if pop_sig else
                          "POPULATION NULL: metric organization substitutable on real ECoG (strengthens thesis)")
    print("\n=== AJILE12 POPULATION ===")
    print(f"  subjects run: {n}  |  true-beats-all-shuffles: {int(beats.sum())}/{n}  |  gap>0: {n_pos}/{n}")
    print(f"  mean null_gap = {gaps.mean():+.4f} ± {gaps.std():.4f}  (t={t_gap:.2f})  relgap_max mean={relgaps.mean()*100:.1f}%")
    print("VERDICT:", out["verdict"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
