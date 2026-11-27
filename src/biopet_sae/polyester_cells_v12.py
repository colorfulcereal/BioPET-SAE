"""Probe and BLASTp on v1.2, read by cell, with the taxonomy controls.

Cells (test; val is reported as a diagnostic): all negatives; all hard negatives; BACTERIAL hard
negatives; eukaryotic hard negatives; and the matched cell "bacterial positives vs bacterial hard
negatives" where domain carries no information. Each cell also reports two shortcut baselines
that never see function: domain (Bacteria=1, unknown=0.5, Eukaryota=0) and length.

Probe: ESM-2 650M max-pooled layer 33, SGD log-loss L2, C and eta0 selected by val log-loss
(`polyester_sweep`), threshold = val best-F1, 5 seeds (42-46); mean +/- SE across seeds (SE = sd/sqrt(5),
training randomness only). Interval: stratified component bootstrap on the seed-mean score.
BLASTp: database = train positives, score = -log10 best e-value (one deterministic run).

Usage:  uv run python -m biopet_sae.polyester_cells_v12
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from biopet_sae import polyester_sweep
from biopet_sae.polyester_blast import best_hits
from biopet_sae.polyester_probe import best_f1_threshold, comp_boot

SEEDS = [42, 43, 44, 45, 46]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path("data/release/biopet_sae_dataset_v1_2.tsv"))
    ap.add_argument("--embeddings", type=Path, default=Path("data/embeddings_650M_v1_2/embeddings.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/v1_2/cells"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    sw = a.out / "sweep"
    polyester_sweep.main(["--release", str(a.release), "--embeddings", str(a.embeddings), "--out", str(sw)])
    sel = json.loads((sw / "sweep.json").read_text())["selected"]        # e.g. "C=1e-01,eta=1e-03"
    C, eta = float(sel.split(",")[0][2:]), float(sel.split(",")[1][4:])
    rows = list(csv.DictReader(a.release.open(newline=""), delimiter="\t"))
    emb = np.load(a.embeddings, allow_pickle=True)
    pos = {s: i for i, s in enumerate(emb["seq_ids"].astype(str))}
    X = emb["max_L33"][[pos[r["seq_id"]] for r in rows]]
    ids = np.array([r["seq_id"] for r in rows]); y = np.array([int(r["y"]) for r in rows])
    sp = np.array([r["split"] for r in rows]); comp = np.array([r["component"] for r in rows])
    grp = np.array([r["negative_group"] for r in rows]); dom = np.array([r["domain"] for r in rows])
    ln = np.array([int(r["seq_len"]) for r in rows]); seq = {r["seq_id"]: r["sequence"] for r in rows}
    tr, va, te = sp == "train", sp == "val", sp == "test"
    sc = StandardScaler().fit(X[tr])
    scores, thrs = [], []
    for sd in SEEDS:
        clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1.0 / (C * tr.sum()), learning_rate="constant",
                            eta0=eta, max_iter=5000, tol=None, class_weight="balanced", random_state=sd).fit(sc.transform(X[tr]), y[tr])
        s = clf.predict_proba(sc.transform(X))[:, 1]; scores.append(s); thrs.append(best_f1_threshold(y[va], s[va])[0])
    S = np.array(scores); smean = S.mean(0)
    # BLASTp (db = train positives)
    work = Path(tempfile.mkdtemp(prefix="v12blast_"))
    try:
        db = {i: seq[i] for i in ids[tr & (y == 1)]}; hits = {}
        for w in ("val", "test"):
            hits.update(best_hits(db, {i: seq[i] for i in ids[sp == w]}, work / w))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ev = np.array([hits.get(i, (0.0, 1e9))[1] for i in ids]); blast = -np.log10(np.maximum(ev, 1e-300))
    domscore = np.array([{"Bacteria": 1.0, "unknown": 0.5}.get(d, 0.0) for d in dom]); lenscore = -ln.astype(float)

    hard = grp == "ab_hydrolase_hard_negative"
    bact = dom == "Bacteria"
    cells = {
        "all negatives": (y == 1) | (y == 0),
        "all hard negatives": (y == 1) | hard,
        "BACTERIAL hard negatives (all positives)": (y == 1) | (hard & bact),
        "EUKARYOTIC hard negatives (all positives)": (y == 1) | (hard & (dom == "Eukaryota")),
        "MATCHED: bacterial positives vs bacterial hard neg.": ((y == 1) & bact) | (hard & bact),
        "wrong-fold": (y == 1) | (grp == "ec311_wrong_fold"),
        "tRNA-processing": (y == 1) | (grp == "tRNA_processing_no_ser_nucleophile"),
        "naive controls": (y == 1) | (grp == "naive_control"),
    }
    def au(yy, ss): return (roc_auc_score(yy, ss), average_precision_score(yy, ss))
    res = {"selected": sel, "C": C, "eta0": eta, "seeds": SEEDS, "val_thresholds": [round(t, 4) for t in thrs], "cells": {}}
    for split_name, sm in (("test", te), ("val", va)):
        res["cells"][split_name] = {}
        for name, cm in cells.items():
            m = sm & cm; yy = y[m]
            if yy.min() == yy.max(): continue
            per = np.array([au(yy, S[k][m]) for k in range(len(SEEDS))])
            ci, nc = comp_boot(yy, smean[m], comp[m], roc_auc_score)
            res["cells"][split_name][name] = {
                "n_pos": int(yy.sum()), "n_neg": int((1 - yy).sum()), "prevalence": round(float(yy.mean()), 3),
                "probe_au_roc": [round(float(per[:, 0].mean()), 4), round(float(per[:, 0].std(ddof=1) / np.sqrt(len(SEEDS))), 4)],
                "probe_au_prc": [round(float(per[:, 1].mean()), 4), round(float(per[:, 1].std(ddof=1) / np.sqrt(len(SEEDS))), 4)],
                "probe_au_roc_ci95_component_on_seed_mean": ci, "n_components_pos_neg": nc,
                "blastp_au_roc": round(float(roc_auc_score(yy, blast[m])), 4), "blastp_au_prc": round(float(average_precision_score(yy, blast[m])), 4),
                "domain_only_au_roc": round(float(roc_auc_score(yy, domscore[m])), 4),
                "domain_only_au_prc": round(float(average_precision_score(yy, domscore[m])), 4),
                "length_only_au_roc": round(float(roc_auc_score(yy, lenscore[m])), 4),
                "length_only_au_prc": round(float(average_precision_score(yy, lenscore[m])), 4)}
    a_ = np.load  # keep linters quiet
    np.savez(a.out / "scores_seed_mean.npz", seq_ids=ids, probe=smean, blast=blast, y=y, split=sp, domain=dom, group=grp)
    (a.out / "cells.json").write_text(json.dumps(res, indent=2))
    print(f"selected {sel};  val-optimal thresholds {res['val_thresholds']}")
    for split_name in ("test", "val"):
        print(f"\n=== {split_name} ===")
        print(f"{'cell':52s}{'n+':>4s}{'n-':>5s} {'floor':>5s} | {'probe ROC':>14s} {'probe PRC':>14s} | {'BLAST ROC':>9s} {'PRC':>6s} | {'domain ROC':>10s} {'PRC':>6s} | {'length ROC':>10s} {'PRC':>6s} | 95% CI probe ROC")
        for n, v in res["cells"][split_name].items():
            print(f"{n:52s}{v['n_pos']:4d}{v['n_neg']:5d} {v['prevalence']:5.2f} | {v['probe_au_roc'][0]:7.4f}±{v['probe_au_roc'][1]:.4f} {v['probe_au_prc'][0]:7.4f}±{v['probe_au_prc'][1]:.4f} | {v['blastp_au_roc']:9.4f} {v['blastp_au_prc']:6.3f} | {v['domain_only_au_roc']:10.3f} {v['domain_only_au_prc']:6.3f} | {v['length_only_au_roc']:10.3f} {v['length_only_au_prc']:6.3f} | {v['probe_au_roc_ci95_component_on_seed_mean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
