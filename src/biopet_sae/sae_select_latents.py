"""SAE latent selection: Cliff's delta over all positive-negative pairs plus top-10 stability.

Committed form of the v1 selection (the original script was never committed). Verified against the
stored v1 result (`data/sae_650M_L33/selection_full.npz`): delta reproduces to 6e-8 and the top-10 hit
counts reproduce EXACTLY with `numpy.random.default_rng(0)` and half-draws without replacement.

Definitions
  features   per-protein max over residues of each of the 10,240 InterPLM latents (`feat_max`);
             NOT the mean-pooled activation (mean-pooled reproduces nothing).
  rows       train + validation proteins only (v1: 346 positives, 361 negatives = 707)
  delta_j    Cliff's delta of latent j: P(x_pos > x_neg) - P(x_pos < x_neg) over every positive-negative
             pair (ties contribute 0). Computed from ranks: delta = 2U / (n_pos * n_neg) - 1 (identical).
  stability  200 draws; each draws a random HALF of the 707 proteins WITHOUT replacement (353), recomputes
             delta for all 10,240 latents and records the 10 largest; hits_j = number of draws (of 200)
             in which latent j is in the top 10.
  selected   delta >= 0.9 AND hits >= 150 (75% of draws).

Usage:
    uv run python -m biopet_sae.sae_select_latents --release archive/v1/release/biopet_sae_dataset_v1.tsv \
        --out results/v1_repro/sae_selection
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def cliff(F: np.ndarray, y: np.ndarray) -> np.ndarray:
    n1, n0 = int(y.sum()), int((1 - y).sum())
    R = rankdata(F, axis=0)
    return 2 * (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0) - 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=Path("data/sae_650M_L33/sae_features.npz"))
    ap.add_argument("--pool", choices=["feat_max", "feat_mean"], default="feat_max")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.9)
    ap.add_argument("--min-hits", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rel = {r["seq_id"]: r for r in csv.DictReader(a.release.open(newline=""), delimiter="\t")}
    z = np.load(a.features, allow_pickle=True)
    ids = z["seq_ids"].astype(str)
    keep = np.array([i in rel and rel[i]["split"] in ("train", "val") for i in ids])
    if keep.sum() == 0: raise SystemExit("no train/val rows with SAE features; extract features for the release first")
    y = np.array([int(rel[i]["y"]) for i in ids[keep]]); F = z[a.pool][keep].astype(np.float64)
    d = cliff(F, y)
    rng = np.random.default_rng(a.seed); hits = np.zeros(F.shape[1], int); n = len(y)
    for _ in range(a.draws):
        idx = rng.choice(n, n // 2, replace=False)
        hits[np.argsort(-cliff(F[idx], y[idx]))[:10]] += 1
    sel = np.flatnonzero((d >= a.delta) & (hits >= a.min_hits))
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "selection_full.npz", delta=d.astype(np.float32), hits=hits, top10=hits / a.draws)
    rep = {"pool": a.pool, "n_rows": int(n), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "pairs": int(y.sum() * (1 - y).sum()),
           "draws": a.draws, "half_size": n // 2, "seed": a.seed, "criterion": {"delta": a.delta, "min_hits": a.min_hits},
           "n_delta_ge_threshold": int((d >= a.delta).sum()), "n_ever_in_top10": int((hits > 0).sum()),
           "selected": [{"latent": int(i), "delta": round(float(d[i]), 4), "hits": int(hits[i])} for i in sel[np.argsort(-d[sel])]]}
    (a.out / "latent_selection.json").write_text(json.dumps(rep, indent=2))
    print(f"{a.pool}: {n} proteins ({rep['n_pos']} pos x {rep['n_neg']} neg = {rep['pairs']:,} pairs); latents with delta>={a.delta}: {rep['n_delta_ge_threshold']}; ever in a top-10: {rep['n_ever_in_top10']}")
    for s in rep["selected"]: print(f"  latent {s['latent']:5d}  delta {s['delta']:.4f}  top-10 in {s['hits']}/{a.draws} draws")
    return 0


if __name__ == "__main__":
    sys.exit(main())
