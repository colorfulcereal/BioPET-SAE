"""SAE latent selection with 1,000 random-half draws and a sensitivity table (v1.2).

Same procedure as `sae_select_latents.py` (max-pooled latents; Cliff's delta over all positive-negative pairs
in train+val; each draw = a random HALF of the proteins without replacement; top-10 per draw; hits = number of
draws a latent is in the top 10), with 1,000 draws. Stability is across random halves of PROTEINS, not across
families. The first 200 draws equal the 200-draw v1 run (same generator, seed 0), so cumulative hits at 200 are also
saved; on v1 data they must equal the stored `selection_full.npz` hits.

Outputs (in --out): selection_full.npz (delta, hits at 200 and at all draws), sensitivity_tables.json and CSVs:
  table1  latents selected at each stability level (default 50 / 62.5 / 75 / 87.5 % of draws) with delta >= --delta
  table2  number selected over a grid of delta cut-offs x stability levels
  table3  the v1 seven: delta, hits (%) in this run
  table4  top latents by hits: delta against all negatives, hard negatives only, bacterial hard negatives only,
          and bacterial positives vs bacterial hard negatives (taxonomy control)

Usage:
    uv run python -m biopet_sae.sae_sensitivity_v12 --release data/release/biopet_sae_dataset_v1_2.tsv \
        --features data/sae_650M_L33/sae_features.npz data/sae_650M_L33_new416/sae_features.npz --out results/v1_2/sae_selection
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from biopet_sae.sae_select_latents import cliff

V1_SEVEN = [9529, 5201, 7942, 5255, 2661, 5271, 7734]
LEVELS = [0.50, 0.625, 0.75, 0.875]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--features", type=Path, nargs="+", required=True)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.9)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.9, 0.85, 0.8, 0.7, 0.6])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rel = {r["seq_id"]: r for r in csv.DictReader(a.release.open(newline=""), delimiter="\t")}
    feat = {}
    for fp in a.features:
        z = np.load(fp, allow_pickle=True)
        A = z["feat_max"]                      # load once: indexing z[...] per row re-reads the whole array
        for i, s in enumerate(z["seq_ids"].astype(str)):
            if s not in feat: feat[s] = A[i]
    ids = [s for s, r in rel.items() if r["split"] in ("train", "val") and s in feat]
    missing = [s for s, r in rel.items() if r["split"] in ("train", "val") and s not in feat]
    if missing: raise SystemExit(f"{len(missing)} train/val proteins without SAE features: {missing[:5]}")
    F = np.stack([feat[s] for s in ids]).astype(np.float64)
    y = np.array([int(rel[s]["y"]) for s in ids]); grp = np.array([rel[s].get("negative_group", "") for s in ids])
    dom = np.array([rel[s].get("domain", "") for s in ids])
    n = len(y); d = cliff(F, y)
    rng = np.random.default_rng(a.seed); hits = np.zeros(F.shape[1], int); hits200 = None
    for k in range(a.draws):
        idx = rng.choice(n, n // 2, replace=False)
        hits[np.argsort(-cliff(F[idx], y[idx]))[:10]] += 1
        if k + 1 == 200: hits200 = hits.copy()
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "selection_full.npz", delta=d.astype(np.float32), hits=hits, hits_at_200=hits200 if hits200 is not None else hits, draws=a.draws)
    frac = hits / a.draws
    # delta against subsets (full data, no resampling)
    hard = grp == "ab_hydrolase_hard_negative"; bact = dom == "Bacteria"
    def dsub(mask_rows):
        yy = y[mask_rows]
        return cliff(F[mask_rows], yy) if 0 < yy.sum() < len(yy) else np.full(F.shape[1], np.nan)
    d_hard = dsub((y == 1) | hard); d_bh = dsub((y == 1) | (hard & bact)); d_match = dsub(((y == 1) & bact) | (hard & bact))
    out = {"n_proteins": n, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "n_hard_neg": int(hard.sum()),
           "n_bacterial_hard_neg": int((hard & bact).sum()), "pairs": int(y.sum() * (1 - y).sum()), "draws": a.draws, "half": n // 2, "seed": a.seed,
           "max_delta": float(d.max()), "n_delta_ge": {str(t): int((d >= t).sum()) for t in a.deltas}, "n_ever_top10": int((hits > 0).sum())}
    t1 = {}
    for L in LEVELS:
        sel = np.flatnonzero((d >= a.delta) & (frac >= L))
        t1[str(L)] = [{"latent": int(i), "delta": round(float(d[i]), 4), "hits": int(hits[i]), "pct": round(100 * float(frac[i]), 1)} for i in sel[np.argsort(-d[sel])]]
    t2 = {str(t): {str(L): int(((d >= t) & (frac >= L)).sum()) for L in LEVELS} for t in a.deltas}
    t3 = [{"latent": l, "delta": round(float(d[l]), 4), "delta_rank": int((d > d[l]).sum()) + 1, "hits": int(hits[l]), "pct": round(100 * float(frac[l]), 1),
           "v1_pct": None} for l in V1_SEVEN]
    top = np.argsort(-hits)[:20]
    t4 = [{"latent": int(i), "hits": int(hits[i]), "pct": round(100 * float(frac[i]), 1), "delta_all": round(float(d[i]), 3),
           "delta_hard": round(float(d_hard[i]), 3), "delta_bact_hard": round(float(d_bh[i]), 3), "delta_matched": round(float(d_match[i]), 3)} for i in top]
    out.update({"table1": t1, "table2": t2, "table3": t3, "table4": t4})
    (a.out / "sensitivity_tables.json").write_text(json.dumps(out, indent=2))
    for name, rows in (("table1", [{"level": k, **r} for k, v in t1.items() for r in v]), ("table3", t3), ("table4", t4)):
        if rows:
            with (a.out / f"{name}.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (a.out / "table2.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["delta_cutoff"] + [f"hits>={int(L*100*10)/10}%" for L in LEVELS])
        for t in a.deltas: w.writerow([t] + [t2[str(t)][str(L)] for L in LEVELS])
    print(f"{n} proteins ({out['n_pos']} pos x {out['n_neg']} neg = {out['pairs']:,} pairs; {out['n_hard_neg']} hard neg, {out['n_bacterial_hard_neg']} bacterial hard neg); {a.draws} half-draws of {n//2}")
    print(f"max delta {out['max_delta']:.4f}; latents with delta >= " + ", ".join(f">={t}: {out['n_delta_ge'][str(t)]}" for t in a.deltas) + f"; ever in a top-10: {out['n_ever_top10']}")
    print("\nTable 1: selected latents (delta >= %.2f) by stability level" % a.delta)
    for L in LEVELS:
        v = t1[str(L)]; print(f"  hits >= {L*100:g}% ({int(L*a.draws)} of {a.draws}): {len(v)} latents " + (", ".join(f"{r['latent']} (δ{r['delta']:.3f}, {r['pct']:g}%)" for r in v) if v else "none"))
    print("\nTable 2: number of latents selected, delta cut-off (rows) x stability level (columns)")
    print("  delta   " + "".join(f"{'>=%g%%' % (L*100):>9s}" for L in LEVELS))
    for t in a.deltas: print(f"  {t:<7g} " + "".join(f"{t2[str(t)][str(L)]:9d}" for L in LEVELS))
    print("\nTable 3: the v1 seven in this run")
    for r in t3: print(f"  latent {r['latent']}: delta {r['delta']:.4f} (rank {r['delta_rank']}), top-10 in {r['hits']}/{a.draws} draws ({r['pct']}%)")
    print("\nTable 4: top 20 latents by hits; delta against: all negatives | hard only | bacterial hard only | matched (bacterial pos vs bacterial hard)")
    for r in t4: print(f"  {r['latent']:5d} hits {r['pct']:5.1f}%  δ all {r['delta_all']:.3f} | hard {r['delta_hard']:.3f} | bact-hard {r['delta_bact_hard']:.3f} | matched {r['delta_matched']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
