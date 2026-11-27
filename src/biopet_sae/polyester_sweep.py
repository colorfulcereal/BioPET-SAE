"""Hyperparameter sweep for the polyester probe: regularisation C x SGD learning rate eta0.

Selection uses VAL log-loss only (v1 rule, PROGRESS.md section 14: val AU-PRC saturates and
picks an unconverged, miscalibrated corner). Test is not touched here.

  C     inverse L2 strength (alpha = 1/(C n_train)): small C = strong shrinkage of the weights
  eta0  step size of the SGD optimiser: too large and training oscillates, too small and 5000
        epochs do not converge; it changes how well the optimum is reached, not the model class

Usage:
    uv run python -m biopet_sae.polyester_sweep --release data/release/biopet_sae_dataset_v1_1.tsv \
        --embeddings data/embeddings_650M_v1_1/embeddings.npz --out results/v1_1/sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
ETAS = [1e-4, 1e-3, 1e-2, 1e-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, default=Path("data/embeddings_650M/embeddings.npz"))
    ap.add_argument("--features", default="max_L33")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(a.release.open(newline=""), delimiter="\t"))
    emb = np.load(a.embeddings, allow_pickle=True)
    pos = {s: i for i, s in enumerate(emb["seq_ids"].astype(str))}
    X = emb[a.features][[pos[r["seq_id"]] for r in rows]]
    y = np.array([int(r["y"]) for r in rows]); sp = np.array([r["split"] for r in rows])
    tr, va = sp == "train", sp == "val"
    sc = StandardScaler().fit(X[tr]); Xt, Xv = sc.transform(X[tr]), sc.transform(X[va])
    grid = {}
    for C in CS:
        for eta in ETAS:
            clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1.0 / (C * tr.sum()),
                                learning_rate="constant", eta0=eta, max_iter=a.epochs, tol=None,
                                class_weight="balanced", random_state=a.seed).fit(Xt, y[tr])
            p = clf.predict_proba(Xv)[:, 1]
            grid[f"C={C:.0e},eta={eta:.0e}"] = {
                "val_logloss": round(float(log_loss(y[va], np.clip(p, 1e-15, 1 - 1e-15))), 4),
                "val_auprc": round(float(average_precision_score(y[va], p)), 4),
                "val_auroc": round(float(roc_auc_score(y[va], p)), 4),
                "train_logloss": round(float(log_loss(y[tr], np.clip(clf.predict_proba(Xt)[:, 1], 1e-15, 1 - 1e-15))), 4)}
    best = min(grid, key=lambda k: grid[k]["val_logloss"])
    res = {"features": a.features, "epochs": a.epochs, "seed": a.seed, "n_train": int(tr.sum()),
           "n_val": int(va.sum()), "selected": best, "selected_by": "min val log-loss", "grid": grid}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sweep.json").write_text(json.dumps(res, indent=2))
    print(f"{'cell':22s}{'val logloss':>12s}{'val AUPRC':>11s}{'val AUROC':>11s}{'train logloss':>15s}")
    for k, v in grid.items():
        print(f"{k:22s}{v['val_logloss']:12.4f}{v['val_auprc']:11.4f}{v['val_auroc']:11.4f}{v['train_logloss']:15.4f}{'   <- selected' if k == best else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
