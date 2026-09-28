"""Binary PET-vs-rest probe, evaluated separately against each negative tier.

Training is binary (PET = 1, classes 2/3/4 = 0). The earlier 4-way probe was failing at
the class-3-vs-class-4 boundary -- a distinction that carries no scientific weight here
and costs capacity that belongs to the PET boundary.

Evaluation keeps the tiers, which is what preserves the generic-hydrolase diagnostic the
multi-class design existed to provide:

  PET vs 2a_aromatic_copolyester  PBAT-type: already cleaves a terephthalate ester, so
                                  separating it rules out "recognises terephthalate"
  PET vs 2b_aliphatic_polyester   PCL/PLA/PBS: flexible backbone, easier substrate
  PET vs 2_other_polyester        the two above pooled
  PET vs 3_fold_matched_esterase  same fold, no plastic annotation
  PET vs 4_naive_control          not an esterase at all

Each contrast gets a permutation floor: labels held fixed, scores randomised 4000 times.
`average_precision_score` is upward-biased at small n, so raw prevalence understates the
floor (e.g. 0.550 rather than 0.500 for a 15-vs-15 contrast).

Usage:
    uv run python -m biopet_sae.train_binary_probe \
        --embeddings data/embeddings_8M/embeddings.npz \
        --splits data/processed/dataset_splits_id40.tsv
"""

from __future__ import annotations

import argparse
import csv
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
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

CLASSES = ["1_pet", "2_other_polyester", "3_fold_matched_esterase", "4_naive_control"]


def best_f1(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float, float]:
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    i = int(np.argmax(f1))
    return float(f1[i]), (float(thr[i]) if i < len(thr) else 1.0), float(prec[i]), float(rec[i])


def boot_ci(y, s, comps, fn, n_boot=2000, seed=0):
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


def perm_floor(n_pos: int, n_neg: int, observed: float, n_trials: int = 4000, seed: int = 0):
    """Null AU-PRC for this exact label count under random ranking."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
    null = np.array([average_precision_score(y, rng.random(n_pos + n_neg))
                     for _ in range(n_trials)])
    return float(null.mean()), float(np.percentile(null, 99)), float((null >= observed).mean())


def subtier_labels(splits_path: Path, merged_path: Path) -> dict[str, str]:
    """Class label, with class 2 split into 2a (PBAT-type) and 2b (aliphatic)."""
    group = {}
    with merged_path.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            group[r["seq_id"]] = r["primary_group"]
    out = {}
    with splits_path.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            c = r["class_label"]
            if c == "2_other_polyester":
                c = ("2a_aromatic_copolyester"
                     if group.get(r["seq_id"]) == "AROMATIC_COPOLYESTER"
                     else "2b_aliphatic_polyester")
            out[r["seq_id"]] = c
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path, default=Path("data/embeddings_8M/embeddings.npz"))
    ap.add_argument("--splits", type=Path, default=None,
                    help="override split/component assignment by seq_id")
    ap.add_argument("--merged", type=Path,
                    default=Path("data/processed/plastizymes_merged.tsv"))
    ap.add_argument("--eval-on", choices=["val", "test"], default="val")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    # C=1.0 leaves the probe in the separable regime: 320 features over 588 training
    # points fit perfectly (in-sample AU-ROC exactly 1.0000), weights grow until training
    # logits reach +-13, and every probability saturates to 0 or 1. Held-out positives
    # then land in the score band reserved for negatives -- the best-F1 threshold drops to
    # 0.0044 and recall at 0.5 collapses to 0.219. Ranking metrics are unaffected (they
    # are threshold-free) but the probability scale is meaningless. C=1e-3 was selected by
    # sweep on max_L6; see artifacts/ESM2_8M_Dense_Run.md Table 1.
    ap.add_argument("--C", type=float, default=1e-3,
                    help="inverse L2 regularization strength (default 1e-3, swept)")
    ap.add_argument("--eta0", type=float, default=1e-4,
                    help="SGD learning rate; ignored when --optimizer=lbfgs")
    ap.add_argument("--optimizer", choices=["sgd", "lbfgs"], default="sgd",
                    help="sgd matches the recorded run; lbfgs converges to the same "
                         "solution on this convex objective and needs no learning rate")
    ap.add_argument("--epochs", type=int, default=5000, help="SGD max_iter")
    args = ap.parse_args(argv)

    d = np.load(args.embeddings, allow_pickle=True)
    ids = d["seq_ids"].astype(str)
    labels = d["class_label"].astype(str)
    split = d["split"].astype(str)
    comps = d["component"].astype(str)
    tag = "orig"

    if args.splits:
        tbl = {r["seq_id"]: r for r in csv.DictReader(args.splits.open(newline=""), delimiter="\t")}
        missing = [i for i in ids if i not in tbl]
        if missing:
            print(f"warning: {len(missing)} embedded sequences absent from {args.splits}")
        split = np.array([tbl[i]["split"] if i in tbl else "" for i in ids])
        comps = np.array([tbl[i]["component"] if i in tbl else "" for i in ids])
        labels = np.array([tbl[i]["class_label"] if i in tbl else "" for i in ids])
        tag = args.splits.stem
        print(f"split assignment re-mapped from {args.splits}")

    sub = subtier_labels(args.splits, args.merged) if args.splits else {}
    out_dir = args.out or Path(f"results/binary_{args.embeddings.parent.name}_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = np.isin(labels, CLASSES)
    tr = keep & (split == "train")
    ev = keep & (split == args.eval_on)
    ytr = (labels[tr] == "1_pet").astype(int)
    yev = (labels[ev] == "1_pet").astype(int)
    print(f"\ntrain {tr.sum()} ({ytr.sum()} PET)   {args.eval_on} {ev.sum()} ({yev.sum()} PET)")
    tiers = ["2a_aromatic_copolyester", "2b_aliphatic_polyester",
             "2_other_polyester", "3_fold_matched_esterase", "4_naive_control"]
    ev_sub = np.array([sub.get(i, "") for i in ids])[ev] if sub else labels[ev]
    print(f"{args.eval_on} negatives: " + ", ".join(
        f"{t.split('_')[0]}={int((labels[ev] == t).sum() if t in CLASSES else (ev_sub == t).sum())}"
        for t in tiers) + "\n")

    keys = sorted([k for k in d.files if k.startswith(("mean_L", "max_L"))],
                  key=lambda k: (k.split("_")[0], int(k.split("L")[1])))
    overall, per_tier = [], []
    n_train = int(tr.sum())
    print(f"classifier: {args.optimizer}, C={args.C:.0e}"
          + (f", eta0={args.eta0:.0e}, {args.epochs} epochs" if args.optimizer == "sgd" else "")
          + f", seed {args.seed}")
    for key in keys:
        X = d[key]
        sc = StandardScaler().fit(X[tr])
        if args.optimizer == "sgd":
            # alpha is SGDClassifier's regularization term; 1/(C*n) puts it on the same
            # scale as LogisticRegression's C so the two optimizers are comparable
            clf = SGDClassifier(
                loss="log_loss", penalty="l2", alpha=1.0 / (args.C * n_train),
                learning_rate="constant", eta0=args.eta0, max_iter=args.epochs,
                tol=None, class_weight="balanced", random_state=args.seed,
            ).fit(sc.transform(X[tr]), ytr)
        else:
            clf = LogisticRegression(
                C=args.C, solver="lbfgs", max_iter=20000, class_weight="balanced",
                random_state=args.seed,
            ).fit(sc.transform(X[tr]), ytr)
        s = clf.predict_proba(sc.transform(X[ev]))[:, 1]

        aupr = float(average_precision_score(yev, s))
        lo, hi = boot_ci(yev, s, comps[ev], average_precision_score, seed=args.seed)
        floor, p99, pval = perm_floor(int(yev.sum()), int(len(yev) - yev.sum()), aupr)
        f1b, thr, p_at, r_at = best_f1(yev, s)
        p5, r5, f5, _ = precision_recall_fscore_support(
            yev, (s >= 0.5).astype(int), average="binary", zero_division=0)
        overall.append({
            "features": key,
            "AU-ROC": round(float(roc_auc_score(yev, s)), 4),
            "AU-PRC": round(aupr, 4), "AU-PRC CI": [round(lo, 3), round(hi, 3)],
            "AU-PRC floor": round(floor, 4), "p": round(pval, 4),
            "best F1": round(f1b, 4), "thr": round(thr, 4),
            "P@bestF1": round(p_at, 4), "R@bestF1": round(r_at, 4),
            "P@0.5": round(float(p5), 4), "R@0.5": round(float(r5), 4),
            "F1@0.5": round(float(f5), 4),
        })
        for t in tiers:
            mask = (ev_sub == "1_pet") | (ev_sub == t)
            yy, ss, cc = yev[mask], s[mask], comps[ev][mask]
            if yy.sum() == 0 or yy.sum() == len(yy) or len(yy) - yy.sum() < 2:
                continue
            ap_t = float(average_precision_score(yy, ss))
            lo_t, hi_t = boot_ci(yy, ss, cc, average_precision_score, seed=args.seed)
            fl_t, _, pv_t = perm_floor(int(yy.sum()), int(len(yy) - yy.sum()), ap_t)
            f1_t, thr_t, p_t, r_t = best_f1(yy, ss)
            per_tier.append({
                "features": key, "tier": t, "n_neg": int((ev_sub == t).sum()),
                "AU-ROC": round(float(roc_auc_score(yy, ss)), 4),
                "AU-PRC": round(ap_t, 4), "AU-PRC CI": [round(lo_t, 3), round(hi_t, 3)],
                "floor": round(fl_t, 4), "p": round(pv_t, 4),
                "best F1": round(f1_t, 4), "P@bestF1": round(p_t, 4),
                "R@bestF1": round(r_t, 4),
            })

    odf = pd.DataFrame(overall).sort_values("AU-PRC", ascending=False)
    print(f"=== binary PET vs all negatives ({args.eval_on}) ===")
    print(odf[["features", "AU-ROC", "AU-PRC", "AU-PRC CI", "AU-PRC floor", "p",
               "best F1", "thr", "P@bestF1", "R@bestF1"]].to_string(index=False))
    print("\n--- at the default 0.5 threshold ---")
    print(odf[["features", "P@0.5", "R@0.5", "F1@0.5"]].to_string(index=False))

    tdf = pd.DataFrame(per_tier)
    best_key = str(odf.iloc[0]["features"])
    print(f"\n=== diagnostic: per negative tier (features = {best_key}) ===")
    print(tdf[tdf["features"] == best_key].drop(columns=["features"]).to_string(index=False))
    print("\n=== AU-ROC per tier, all feature sets ===")
    print(tdf.pivot(index="features", columns="tier", values="AU-ROC").to_string())

    (out_dir / "binary_probe.json").write_text(json.dumps(
        {"overall": overall, "per_tier": per_tier, "eval_on": args.eval_on,
         "splits": str(args.splits), "embeddings": str(args.embeddings)}, indent=2))
    odf.to_csv(out_dir / "binary_overall.csv", index=False)
    tdf.to_csv(out_dir / "binary_per_tier.csv", index=False)
    print(f"\nwrote {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
