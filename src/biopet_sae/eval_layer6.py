"""Focused PET-detection metrics for the layer-6 feature sets only.

Reports precision, recall, AU-PRC (average precision) and best achievable F1 for
`max_L6` and `mean_L6`, with CIs bootstrapped over components rather than sequences.

AU-PRC is reported against the prevalence floor: a random ranker scores AU-PRC equal to
the positive rate, which is 0.112 here, not 0.5. Best F1 is the maximum over all decision
thresholds, so it is an optimistic ceiling -- the threshold is chosen on the same data it
is scored on. The argmax row is what the probe would actually predict.

Usage:
    uv run python -m biopet_sae.eval_layer6 --embeddings data/embeddings_8M/embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

CLASSES = ["1_pet", "2_other_polyester", "3_fold_matched_esterase", "4_naive_control"]


def best_f1(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float, float]:
    """Return (best_f1, threshold, precision_at_best, recall_at_best)."""
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    i = int(np.argmax(f1))
    t = float(thr[i]) if i < len(thr) else 1.0
    return float(f1[i]), t, float(prec[i]), float(rec[i])


def boot_ci(
    y: np.ndarray, s: np.ndarray, comps: np.ndarray, fn, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(comps)
    by = {c: np.flatnonzero(comps == c) for c in uniq}
    vals = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[c] for c in picked])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path, default=Path("data/embeddings_8M/embeddings.npz"))
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    d = np.load(args.embeddings, allow_pickle=True)
    out_dir = args.out or Path(f"results/probe_{args.embeddings.parent.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = d["class_label"].astype(str)
    split = d["split"].astype(str)
    comps = d["component"].astype(str)
    keep = np.isin(labels, CLASSES)
    tr, va = keep & (split == "train"), keep & (split == "val")
    y_bin = (labels[va] == "1_pet").astype(int)
    prevalence = float(y_bin.mean())
    print(f"train {tr.sum()}  val {va.sum()}  val PET {int(y_bin.sum())}  "
          f"prevalence {prevalence:.3f}")
    print(f"val components: {len(np.unique(comps[va]))} "
          f"(PET-containing: {len(np.unique(comps[va][y_bin == 1]))})\n")

    binary_rows, perclass_rows = [], []
    for pooling in ("max", "mean"):
        key = f"{pooling}_L{args.layer}"
        if key not in d.files:
            print(f"skip {key}: not in embeddings")
            continue
        X = d[key]
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(
            C=1.0, max_iter=5000, class_weight="balanced", random_state=args.seed
        ).fit(scaler.transform(X[tr]), labels[tr])
        Xv = scaler.transform(X[va])
        proba = clf.predict_proba(Xv)
        pred = clf.predict(Xv)
        s = proba[:, list(clf.classes_).index("1_pet")]

        auprc = average_precision_score(y_bin, s)
        lo_ap, hi_ap = boot_ci(y_bin, s, comps[va], average_precision_score, seed=args.seed)
        f1b, thr, p_at, r_at = best_f1(y_bin, s)
        lo_f1, hi_f1 = boot_ci(
            y_bin, s, comps[va], lambda yy, ss: best_f1(yy, ss)[0], seed=args.seed
        )
        # what the probe actually predicts (4-way argmax)
        p_arg, r_arg, f1_arg, _ = precision_recall_fscore_support(
            y_bin, (pred == "1_pet").astype(int), average="binary", zero_division=0
        )
        binary_rows.append({
            "features": key,
            "AU-PRC": round(float(auprc), 4),
            "AU-PRC 95% CI": [round(lo_ap, 3), round(hi_ap, 3)],
            "AU-PRC floor (prevalence)": round(prevalence, 4),
            "best F1": round(f1b, 4),
            "best F1 95% CI": [round(lo_f1, 3), round(hi_f1, 3)],
            "thr at best F1": round(thr, 4),
            "precision at best F1": round(p_at, 4),
            "recall at best F1": round(r_at, 4),
            "precision at argmax": round(float(p_arg), 4),
            "recall at argmax": round(float(r_arg), 4),
            "F1 at argmax": round(float(f1_arg), 4),
        })
        p, r, f, sup = precision_recall_fscore_support(
            labels[va], pred, labels=CLASSES, zero_division=0
        )
        for c, pp, rr, ff, ss in zip(CLASSES, p, r, f, sup):
            perclass_rows.append({"features": key, "class": c, "n": int(ss),
                                  "precision": round(float(pp), 3),
                                  "recall": round(float(rr), 3),
                                  "F1": round(float(ff), 3)})

    bdf = pd.DataFrame(binary_rows)
    print("=== PET vs rest (binary detection) ===")
    print(bdf[["features", "AU-PRC", "AU-PRC 95% CI", "AU-PRC floor (prevalence)",
               "best F1", "best F1 95% CI", "thr at best F1"]].to_string(index=False))
    print()
    print(bdf[["features", "precision at best F1", "recall at best F1",
               "precision at argmax", "recall at argmax", "F1 at argmax"]].to_string(index=False))
    print("\n=== 4-way per class ===")
    print(pd.DataFrame(perclass_rows).to_string(index=False))

    payload = {"binary": binary_rows, "per_class": perclass_rows,
               "val_n": int(va.sum()), "val_pet": int(y_bin.sum()),
               "prevalence": prevalence,
               "embeddings": str(args.embeddings)}
    path = out_dir / f"layer{args.layer}_metrics.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
