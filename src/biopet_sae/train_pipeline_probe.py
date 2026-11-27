"""Train and SAVE the production probe: five runs on v1.2 train, probabilities averaged.

Same settings as phase one (max-pooled ESM-2 650M layer 33; StandardScaler fit on train; SGD log-loss L2,
C=0.01, eta0=1e-4, 5000 epochs, class_weight balanced; seeds 42-46). Each run is checked against the recorded
scores. Thresholds are re-derived on VALIDATION from the AVERAGED score (best F1; lowest threshold with
precision >= 0.95). Also copies the frozen aspartate classifier into the model file set.

Usage:  uv run python -m biopet_sae.train_pipeline_probe
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SEEDS, C, ETA = [42, 43, 44, 45, 46], 1e-2, 1e-4
GRID = np.unique(np.round(np.r_[0.001, 0.005, np.arange(0.01, 1.0, 0.01), 0.995, 0.999], 4))
OUT = Path("models/pipeline_v1_2")


def at(y, s, t):
    p = s >= t
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum()); fn = int((~p & (y == 1)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def main() -> int:
    rows = list(csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t"))
    emb = np.load("data/embeddings_650M_v1_2/embeddings.npz", allow_pickle=True)
    pos = {s: i for i, s in enumerate(emb["seq_ids"].astype(str))}
    X = emb["max_L33"][[pos[r["seq_id"]] for r in rows]]
    ids = np.array([r["seq_id"] for r in rows]); y = np.array([int(r["y"]) for r in rows]); sp = np.array([r["split"] for r in rows])
    tr, va, te = sp == "train", sp == "val", sp == "test"
    sc = StandardScaler().fit(X[tr])
    W, B, S = [], [], []
    for seed in SEEDS:
        clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1.0 / (C * tr.sum()), learning_rate="constant", eta0=ETA,
                            max_iter=5000, tol=None, class_weight="balanced", random_state=seed).fit(sc.transform(X[tr]), y[tr])
        W.append(clf.coef_[0]); B.append(float(clf.intercept_[0])); S.append(clf.predict_proba(sc.transform(X))[:, 1])
    S = np.array(S)
    ref = np.load("results/v1_2/phase1/scores_5_runs.npz", allow_pickle=True)
    order = {s: i for i, s in enumerate(ref["seq_ids"].astype(str))}
    diff = float(np.abs(S - ref["scores"][:, [order[i] for i in ids]]).max())
    print(f"reproduces the recorded five runs: max |score difference| = {diff:.2e}")
    mean = S.mean(0)
    f1 = [at(y[va], mean[va], t)["f1"] for t in GRID]; t_cand = float(GRID[int(np.argmax(f1))])
    high = [t for t in GRID if (mean[va] >= t).any() and at(y[va], mean[va], t)["precision"] >= 0.95]
    t_high = float(high[0])
    report = {"thresholds": {"candidate": t_cand, "high_confidence": t_high}, "reproduction_max_diff": diff}
    for name, m in (("val", va), ("test", te)):
        report[name] = {"au_roc": float(roc_auc_score(y[m], mean[m])), "au_prc": float(average_precision_score(y[m], mean[m])),
                        "floor": float(y[m].mean()), "candidate": at(y[m], mean[m], t_cand), "high_confidence": at(y[m], mean[m], t_high)}
    np.savez(OUT / "probe.npz", mean=sc.mean_, scale=sc.scale_, weights=np.array(W), intercepts=np.array(B), seeds=np.array(SEEDS),
             candidate=t_cand, high_confidence=t_high)
    asp = json.loads(Path("results/v1_2/asp_classifier/frozen_model.json").read_text())
    (OUT / "triad.json").write_text(json.dumps({"ser_latent": 2925, "his_latent": 3927, "fire": 0.3, "asp": asp}, indent=2))
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"thresholds from validation (averaged score): candidate {t_cand}, high confidence {t_high}")
    for name in ("val", "test"):
        r = report[name]
        print(f"{name:5s} AU-ROC {r['au_roc']:.4f}  AU-PRC {r['au_prc']:.4f} (floor {r['floor']:.2f})")
        for k in ("candidate", "high_confidence"):
            v = r[k]
            print(f"      {k:16s} precision {v['precision']:.3f}  recall {v['recall']:.3f}  F1 {v['f1']:.3f}  (TP {v['tp']} FP {v['fp']} FN {v['fn']})")
    return 0


if __name__ == "__main__":
    main()
