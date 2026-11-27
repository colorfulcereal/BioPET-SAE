"""Polyester-degrader probe: train on train, pick the threshold on val, read test by group.

This is the committed form of the procedure behind the v1 headline numbers (it was run from
uncommitted scripts before). Verified against the stored v1 scores
(`data/sae_650M_L33/polyester_scores_650M.npz`): seed 42 reproduces them with max |diff| 0.

Protocol (frozen from v1, PROGRESS.md section 14 / D10):
  * features: ESM-2 650M, max-pooled over residues, layer 33 (1280-d)
  * StandardScaler fit on train only
  * SGDClassifier(log_loss, L2, alpha = 1/(C*n_train), constant eta0, 5000 epochs, tol=None,
    class_weight="balanced", random_state=42), C=1e-2, eta0=1e-4
  * decision threshold = the val best-F1 threshold
  * test is read once; AU-ROC / AU-PRC reported blended and by negative group with the
    positives held fixed; intervals bootstrap over homology `component`, never proteins

Usage:
    uv run python -m biopet_sae.polyester_probe --release data/release/biopet_sae_dataset_v1.tsv \
        --out results/v1_repro/probe
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    i = int(np.argmax(f[:-1]))
    return float(t[i]), float(f[i])


def recall_at_precision(y, s, floor):
    p, r, _ = precision_recall_curve(y, s)
    ok = p[:-1] >= floor
    return float(r[:-1][ok].max()) if ok.any() else 0.0


def comp_boot(y, s, comp, fn, n=5000, seed=0):
    """Component bootstrap, stratified by class: positive components and negative components
    are resampled separately (as in the v1 intervals, 20 positive and 7 negative components for
    the hard-negative cell), so every draw keeps both classes and the class balance."""
    rng = np.random.default_rng(seed)
    pc = np.unique(comp[y == 1]); nc = np.unique(comp[y == 0])
    pi = {c: np.flatnonzero((comp == c) & (y == 1)) for c in pc}
    ni = {c: np.flatnonzero((comp == c) & (y == 0)) for c in nc}
    vals = []
    for _ in range(n):
        idx = np.concatenate([pi[c] for c in rng.choice(pc, len(pc))] +
                             [ni[c] for c in rng.choice(nc, len(nc))])
        vals.append(fn(y[idx], s[idx]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)], [len(pc), len(nc)]


def metrics(y, s, comp, boot=True):
    out = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
           "au_roc": round(float(roc_auc_score(y, s)), 4),
           "au_prc": round(float(average_precision_score(y, s)), 4)}
    if boot:
        out["au_roc_ci95_component"], out["n_components_pos_neg"] = comp_boot(y, s, comp, roc_auc_score)
        out["au_prc_ci95_component"], _ = comp_boot(y, s, comp, average_precision_score)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, default=Path("data/embeddings_650M/embeddings.npz"))
    ap.add_argument("--features", default="max_L33")
    ap.add_argument("--C", type=float, default=1e-2)
    ap.add_argument("--eta0", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = list(csv.DictReader(a.release.open(newline=""), delimiter="\t"))
    emb = np.load(a.embeddings, allow_pickle=True)
    pos = {s: i for i, s in enumerate(emb["seq_ids"].astype(str))}
    missing = [r["seq_id"] for r in rows if r["seq_id"] not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} release rows have no embedding in {a.embeddings}: {missing[:5]}")
    X = emb[a.features][[pos[r["seq_id"]] for r in rows]]
    y = np.array([int(r["y"]) for r in rows])
    sp = np.array([r["split"] for r in rows])
    comp = np.array([r["component"] for r in rows])
    grp = np.array([r["negative_group"] for r in rows])
    ids = np.array([r["seq_id"] for r in rows])
    batch = np.array([r.get("batch", "v1") for r in rows])
    tr, va, te = sp == "train", sp == "val", sp == "test"

    sc = StandardScaler().fit(X[tr])
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1.0 / (a.C * tr.sum()),
                        learning_rate="constant", eta0=a.eta0, max_iter=a.epochs, tol=None,
                        class_weight="balanced", random_state=a.seed).fit(sc.transform(X[tr]), y[tr])
    s = clf.predict_proba(sc.transform(X))[:, 1]
    thr, vf1 = best_f1_threshold(y[va], s[va])

    def at_thr(m):
        yy, ss = y[m], s[m]
        tp = int(((ss >= thr) & (yy == 1)).sum()); fp = int(((ss >= thr) & (yy == 0)).sum())
        fn = int(((ss < thr) & (yy == 1)).sum()); tn = int(((ss < thr) & (yy == 0)).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(p, 4),
                "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-12), 4)}

    res = {"config": {"features": a.features, "C": a.C, "eta0": a.eta0, "epochs": a.epochs,
                      "seed": a.seed, "release": str(a.release), "n_train": int(tr.sum())},
           "threshold": thr, "val_best_f1": round(vf1, 4)}
    for name, m in (("train", tr), ("val", va)):
        res[name] = metrics(y[m], s[m], comp[m], boot=False)
    res["test_blended"] = {**metrics(y[te], s[te], comp[te]), "at_threshold": at_thr(te),
                           "recall_at_precision": {f"{int(f*100)}": round(recall_at_precision(y[te], s[te], f), 4)
                                                   for f in (0.95, 0.90, 0.85, 0.80)}}
    by = {}
    for g in sorted({x for x in grp[te] if x}):
        m = te & ((y == 1) | (grp == g))
        by[g] = metrics(y[m], s[m], comp[m])
    res["test_by_negative_group"] = by
    if len(set(batch)) > 1:       # v1.1: report the original rows and the additions separately
        old = te & (batch == "v1")
        res["test_v1_rows_only"] = {**metrics(y[old], s[old], comp[old]), "at_threshold": at_thr(old)}
        new = te & (batch != "v1")
        pos_new = new & (y == 1)
        res["test_added_positives"] = {"n": int(pos_new.sum()),
                                       "scores": {i: round(float(v), 4) for i, v in zip(ids[pos_new], s[pos_new])},
                                       "recall_at_threshold": round(float((s[pos_new] >= thr).mean()), 4)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "probe_results.json").write_text(json.dumps(res, indent=2))
    np.savez(a.out / "probe_scores.npz", seq_ids=ids, score=s, y=y, split=sp, component=comp,
             negative_group=grp, threshold=thr)
    print(json.dumps({k: res[k] for k in ("threshold", "train", "val")}, indent=1))
    print("test blended:", json.dumps(res["test_blended"]))
    for g, v in by.items():
        print(f"  {g:28s} n_neg={v['n_neg']:3d}  AU-ROC {v['au_roc']}  CI {v['au_roc_ci95_component']}  AU-PRC {v['au_prc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
