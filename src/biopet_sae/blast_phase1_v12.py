"""BLASTp baseline tables for phase one (v1.2): threshold table, ROC and PR curves, recall at precision.

BLASTp is one deterministic run (no seeds, so no SE): database = train positives, score = -log10 of the
best-hit e-value (no hit = e-value 1e9). Reads the scores saved by `polyester_cells_v12`.
Tables use the same layout as the probe's: threshold (here an e-value cut-off), precision, recall (TPR), FPR, F1,
confusion counts; exact ROC and PR curves; recall at precision >= 95/90/85/80% chosen on val and carried to test,
and the best-case (chosen on the evaluated split itself).

Usage:  uv run python -m biopet_sae.blast_phase1_v12
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

PRECS = [0.95, 0.90, 0.85, 0.80]
EGRID = [10.0 ** (k / 2) for k in range(2, -61, -1)]   # half-decade e-value cut-offs 10 ... 1e-30 (same grid as polyester_blast)


def cm(y, s, t):
    pred = s >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "fpr": fp / max(fp + tn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, default=Path("results/v1_2/cells/scores_seed_mean.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/v1_2/phase1"))
    a = ap.parse_args(argv)
    z = np.load(a.scores, allow_pickle=True)
    y, sp, B = z["y"].astype(int), z["split"].astype(str), z["blast"].astype(float)
    va, te = sp == "val", sp == "test"
    res = {"val": {}, "test": {}}
    for name, m in (("val", va), ("test", te)):
        yy, ss = y[m], B[m]
        rows = []
        for e in EGRID:
            c = cm(yy, ss, -np.log10(e)); rows.append([e] + [c[k] for k in ("precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn")])
        with (a.out / f"blast_threshold_table_{name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["evalue_cutoff", "precision", "recall_tpr", "f1", "fpr", "tp", "fp", "fn", "tn"])
            for r in rows: w.writerow([f"{r[0]:.0e}"] + [("" if isinstance(v, float) and np.isnan(v) else round(v, 5) if isinstance(v, float) else v) for v in r[1:]])
        fpr, tpr, th = roc_curve(yy, ss)
        with (a.out / f"blast_roc_curve_{name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["score_threshold_neg_log10_e", "fpr", "tpr"])
            for x in zip(th, fpr, tpr): w.writerow([round(float(x[0]), 4) if np.isfinite(x[0]) else "inf", round(float(x[1]), 5), round(float(x[2]), 5)])
        pr, rc, th2 = precision_recall_curve(yy, ss)
        with (a.out / f"blast_pr_curve_{name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["score_threshold_neg_log10_e", "recall", "precision"])
            for i in range(len(th2)): w.writerow([round(float(th2[i]), 4), round(float(rc[i]), 5), round(float(pr[i]), 5)])
        res[name]["au_roc"], res[name]["au_prc"] = float(roc_auc_score(yy, ss)), float(average_precision_score(yy, ss))
        res[name]["floor"] = float(yy.mean())
        res[name]["best_case_recall"] = {}
        for p in PRECS:
            ok = pr[:-1] >= p; res[name]["best_case_recall"][str(int(p * 100))] = float(rc[:-1][ok].max()) if ok.any() else 0.0
    # carried: threshold = lowest val score reaching the precision on val
    pr, rc, th = precision_recall_curve(y[va], B[va])
    res["carried_to_test"] = {}
    for p in PRECS:
        ok = np.flatnonzero(pr[:-1] >= p)
        if not len(ok): res["carried_to_test"][str(int(p * 100))] = None; continue
        t = float(th[ok[0]]); cv, ct = cm(y[va], B[va], t), cm(y[te], B[te], t)
        res["carried_to_test"][str(int(p * 100))] = {"score_threshold": t, "evalue_cutoff": float(10 ** -t),
                                                      "val": {k: cv[k] for k in ("precision", "recall", "f1")},
                                                      "test": {k: ct[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")}}
    best = max(EGRID, key=lambda e: cm(y[va], B[va], -np.log10(e))["f1"])
    res["val_best_f1_evalue"] = best
    res["at_val_best_f1"] = {n: {k: v for k, v in cm(y[m], B[m], -np.log10(best)).items()} for n, m in (("val", va), ("test", te))}
    (a.out / "blast_phase1.json").write_text(json.dumps(res, indent=2))
    print(f"BLASTp, score = -log10 e-value; val-best-F1 e-value cut-off {best:.0e} (grid 1e1..1e-30)")
    for n in ("val", "test"):
        d = res[n]; c = res["at_val_best_f1"][n]
        print(f"  {n}: AU-ROC {d['au_roc']:.4f}  AU-PRC {d['au_prc']:.4f} (floor {d['floor']:.2f}) | at cut-off: P {c['precision']:.3f} R {c['recall']:.3f} F1 {c['f1']:.3f} (TP {c['tp']} FP {c['fp']} FN {c['fn']} TN {c['tn']})")
    print("  recall at precision targets (threshold chosen on val, carried to test) | best case on test:")
    for p in PRECS:
        c = res["carried_to_test"][str(int(p * 100))]
        bc = res["test"]["best_case_recall"][str(int(p * 100))]
        print(f"   >= {int(p*100)}%: " + (f"e <= {c['evalue_cutoff']:.0e}: val P {c['val']['precision']:.3f} R {c['val']['recall']:.3f} | TEST P {c['test']['precision']:.3f} R {c['test']['recall']:.3f}" if c else "not reached on val") + f" | best-case test recall {bc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
