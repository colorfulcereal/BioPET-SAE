"""One-shot held-out test evaluation.

The decision threshold is **passed in**, not fitted here. It comes from the validation
split; re-optimising a threshold on the test set would invalidate the evaluation. The
script prints what the test-optimal threshold would have been purely as a diagnostic,
clearly labelled as an oracle number rather than a result.

Reports overall metrics with component-bootstrap CIs, plus the per-negative-tier
diagnostic, plus a train-only vs train+val comparison.

Usage:
    uv run python -m biopet_sae.eval_test --features max_L6 --threshold 0.2223
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

CLASSES = ["1_pet", "2_other_polyester", "3_fold_matched_esterase", "4_naive_control"]
TIERS = ["2a_PBAT", "2b_aliphatic", "3_fold_matched_esterase", "4_naive_control"]


def best_f1(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    p, r, t = precision_recall_curve(y, s)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    i = int(np.argmax(f))
    return float(f[i]), (float(t[i]) if i < len(t) else 1.0)


def boot_ci(y, s, comps, fn, n_boot=2000, seed=0):
    """Resample whole homology components -- sequences inside one are near-duplicates."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(comps)
    by = {c: np.flatnonzero(comps == c) for c in uniq}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([by[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path,
                    default=Path("data/embeddings_8M/embeddings.npz"))
    ap.add_argument("--splits", type=Path,
                    default=Path("data/processed/dataset_splits_id40.tsv"))
    ap.add_argument("--merged", type=Path,
                    default=Path("data/processed/plastizymes_merged.tsv"))
    ap.add_argument("--features", default="max_L6")
    ap.add_argument("--threshold", type=float, required=True,
                    help="decision threshold selected on VAL, not fitted here")
    ap.add_argument("--C", type=float, default=1e-3)
    ap.add_argument("--eta0", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    d = np.load(args.embeddings, allow_pickle=True)
    ids = d["seq_ids"].astype(str)
    tbl = {r["seq_id"]: r for r in csv.DictReader(args.splits.open(newline=""), delimiter="\t")}
    grp = {r["seq_id"]: r["primary_group"]
           for r in csv.DictReader(args.merged.open(newline=""), delimiter="\t")}

    labels = np.array([tbl[i]["class_label"] if i in tbl else "" for i in ids])
    split = np.array([tbl[i]["split"] if i in tbl else "" for i in ids])
    comps = np.array([tbl[i]["component"] if i in tbl else "" for i in ids])
    sub = np.array([
        ("2a_PBAT" if grp.get(i) == "AROMATIC_COPOLYESTER" else "2b_aliphatic")
        if labels[j] == "2_other_polyester" else labels[j]
        for j, i in enumerate(ids)
    ])
    keep = np.isin(labels, CLASSES)
    tr, va, te = keep & (split == "train"), keep & (split == "val"), keep & (split == "test")
    y = (labels == "1_pet").astype(int)
    X = d[args.features]
    yt, ct, st = y[te], comps[te], sub[te]

    print(f"TEST EVALUATION — {args.features}, C={args.C:.0e}, eta0={args.eta0:.0e}, seed {args.seed}")
    print(f"threshold {args.threshold} carried over from VAL (not re-tuned on test)\n")
    print(f"test n={int(te.sum())}  PET={int(yt.sum())}  non-PET={int(te.sum() - yt.sum())}"
          f"  prevalence {yt.mean():.3f}")
    print("test negatives by tier: "
          + ", ".join(f"{t}={int((st == t).sum())}" for t in TIERS) + "\n")

    report = {"features": args.features, "threshold_from_val": args.threshold,
              "C": args.C, "eta0": args.eta0, "seed": args.seed,
              "test_n": int(te.sum()), "test_pet": int(yt.sum()),
              "test_prevalence": float(yt.mean()), "fits": {}}

    for name, mask in (("train_only", tr), ("train_plus_val", tr | va)):
        sc = StandardScaler().fit(X[mask])
        clf = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1.0 / (args.C * int(mask.sum())),
            learning_rate="constant", eta0=args.eta0, max_iter=args.epochs, tol=None,
            class_weight="balanced", random_state=args.seed,
        ).fit(sc.transform(X[mask]), y[mask])
        s = clf.predict_proba(sc.transform(X[te]))[:, 1]

        pred = (s >= args.threshold).astype(int)
        p, r, f, _ = precision_recall_fscore_support(yt, pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        ar, apr = roc_auc_score(yt, s), average_precision_score(yt, s)
        lo_r, hi_r = boot_ci(yt, s, ct, roc_auc_score, seed=args.seed)
        lo_p, hi_p = boot_ci(yt, s, ct, average_precision_score, seed=args.seed)
        f_or, t_or = best_f1(yt, s)

        print(f"--- trained on {name} (n={int(mask.sum())}, {int(y[mask].sum())} PET) ---")
        print(f"  AU-ROC     {ar:.4f}   95% CI [{lo_r:.3f}, {hi_r:.3f}]")
        print(f"  AU-PRC     {apr:.4f}   95% CI [{lo_p:.3f}, {hi_p:.3f}]   floor {yt.mean():.3f}")
        print(f"  at thr={args.threshold}:  Precision {p:.4f}   Recall {r:.4f}   F1 {f:.4f}")
        print(f"      TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  [oracle, NOT a result] best F1 on test itself {f_or:.4f} at thr {t_or:.4f}")
        tiers = {}
        for t in TIERS:
            m = (st == "1_pet") | (st == t)
            if m.sum() < 4 or not 0 < yt[m].sum() < m.sum():
                continue
            tiers[t] = {"n_neg": int((st == t).sum()),
                        "AU-ROC": float(roc_auc_score(yt[m], s[m]))}
            print(f"    vs {t:26s} n_neg={tiers[t]['n_neg']:3d}  AU-ROC {tiers[t]['AU-ROC']:.4f}")
        print()
        report["fits"][name] = {
            "n_train": int(mask.sum()), "n_train_pet": int(y[mask].sum()),
            "AU-ROC": float(ar), "AU-ROC_ci": [lo_r, hi_r],
            "AU-PRC": float(apr), "AU-PRC_ci": [lo_p, hi_p],
            "precision": float(p), "recall": float(r), "f1": float(f),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "oracle_best_f1": f_or, "oracle_threshold": t_or, "tiers": tiers,
        }

    out = args.out or Path("artifacts/ESM2_8M_Dense_Run_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
