"""Paired component bootstrap of (probe - BLASTp) AU-ROC and AU-PRC on v1.2, by cell.

For each cell the SAME resampled components are scored by both methods, so the interval is on the
difference. Positive and negative components are resampled separately (as everywhere in this project).
Probe score = mean over the five runs; BLASTp = -log10 best e-value. 5,000 draws, seed 0.
The interval excludes 0 only if the advantage survives resampling of homology families.

Usage:  uv run python -m biopet_sae.paired_probe_vs_blast_v12
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path("data/release/biopet_sae_dataset_v1_2.tsv"))
    ap.add_argument("--scores", type=Path, default=Path("results/v1_2/phase1/scores_5_runs.npz"))
    ap.add_argument("--blast", type=Path, default=Path("results/v1_2/cells/scores_seed_mean.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/v1_2/phase1/paired_probe_vs_blast.json"))
    ap.add_argument("--n", type=int, default=5000)
    a = ap.parse_args(argv)
    rel = {r["seq_id"]: r for r in csv.DictReader(a.release.open(newline=""), delimiter="\t")}
    z = np.load(a.scores, allow_pickle=True); zb = np.load(a.blast, allow_pickle=True)
    assert (z["seq_ids"] == zb["seq_ids"]).all()
    ids = z["seq_ids"].astype(str); P = z["scores"].mean(0); B = zb["blast"].astype(float)
    y = z["y"].astype(int); sp = z["split"].astype(str); grp = z["group"].astype(str); dom = z["domain"].astype(str)
    comp = np.array([rel[i]["component"] for i in ids])
    hard = grp == "ab_hydrolase_hard_negative"; bact = dom == "Bacteria"
    cells = {"all negatives": np.ones(len(y), bool), "hard negatives": (y == 1) | hard,
             "bacterial hard negatives": (y == 1) | (hard & bact),
             "matched bacterial cell": ((y == 1) & bact) | (hard & bact),
             "wrong-fold": (y == 1) | (grp == "ec311_wrong_fold"), "tRNA-processing": (y == 1) | (grp == "tRNA_processing_no_ser_nucleophile"),
             "naive controls": (y == 1) | (grp == "naive_control")}
    rng = np.random.default_rng(0); res = {}
    for split in ("val", "test"):
        res[split] = {}
        for cn, cm in cells.items():
            m = (sp == split) & cm; yy, p, b, c = y[m], P[m], B[m], comp[m]
            if yy.min() == yy.max(): continue
            pc, nc = np.unique(c[yy == 1]), np.unique(c[yy == 0])
            pi = {k: np.flatnonzero((c == k) & (yy == 1)) for k in pc}; ni = {k: np.flatnonzero((c == k) & (yy == 0)) for k in nc}
            dr, dp = [], []
            for _ in range(a.n):
                idx = np.concatenate([pi[k] for k in rng.choice(pc, len(pc))] + [ni[k] for k in rng.choice(nc, len(nc))])
                dr.append(roc_auc_score(yy[idx], p[idx]) - roc_auc_score(yy[idx], b[idx]))
                dp.append(average_precision_score(yy[idx], p[idx]) - average_precision_score(yy[idx], b[idx]))
            f = lambda v: [round(float(np.mean(v)), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            res[split][cn] = {"n_pos": int(yy.sum()), "n_neg": int((1 - yy).sum()),
                              "observed_diff_roc": round(float(roc_auc_score(yy, p) - roc_auc_score(yy, b)), 4),
                              "observed_diff_prc": round(float(average_precision_score(yy, p) - average_precision_score(yy, b)), 4),
                              "diff_roc_mean_lo_hi": f(dr), "diff_prc_mean_lo_hi": f(dp),
                              "roc_excludes_0": bool(np.percentile(dr, 2.5) > 0 or np.percentile(dr, 97.5) < 0),
                              "prc_excludes_0": bool(np.percentile(dp, 2.5) > 0 or np.percentile(dp, 97.5) < 0)}
    a.out.write_text(json.dumps(res, indent=2))
    for split in ("val", "test"):
        print(f"\n{split.upper()}: probe minus BLASTp (observed difference; 95% component-bootstrap interval)")
        print(f"{'cell':28s}{'n+':>4s}{'n-':>5s} | {'AU-ROC diff':>26s} | {'AU-PRC diff':>26s}")
        for cn, d in res[split].items():
            r, q = d["diff_roc_mean_lo_hi"], d["diff_prc_mean_lo_hi"]
            print(f"{cn:28s}{d['n_pos']:4d}{d['n_neg']:5d} | {d['observed_diff_roc']:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]{' *' if d['roc_excludes_0'] else '  '} | {d['observed_diff_prc']:+.3f} [{q[1]:+.3f}, {q[2]:+.3f}]{' *' if d['prc_excludes_0'] else '  '}")
    print("\n* = interval excludes 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
