"""Can a sequence's domain of life (Bacteria vs Eukaryota) be read from its ESM-2 embedding?

Tests the taxonomy shortcut in v1.2 (new hard negatives are 76% eukaryotic, positives mostly
bacterial). The classifier never sees function or the polyester label: it predicts domain only.
Trained on the v1.2 train split, evaluated on test (homology-disjoint, <= 40% identity), so a high
score means domain generalises across protein families, not within them.

Feature sets compared: ESM-2 650M max-pooled layer 33, mean-pooled layer 33, amino-acid
composition (20-d), and length alone. Also reported: domain prediction restricted to the hard
negatives, and a cross-group transfer (train on everything that is NOT a hard negative, test on
hard negatives), which asks whether the domain signal learned elsewhere appears in the hard cell.

Usage:  uv run python -m biopet_sae.domain_probe_v12
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

AA = "ACDEFGHIKLMNPQRSTVWY"


def comp(seq: str) -> np.ndarray:
    c = Counter(seq); n = max(len(seq), 1)
    return np.array([c[a] / n for a in AA], dtype=np.float32)


def fit_eval(Xtr, ytr, Xte, yte, C=0.1):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    return {"n_train": int(len(ytr)), "n_test": int(len(yte)), "test_pos(euk)": int(yte.sum()),
            "au_roc": round(float(roc_auc_score(yte, p)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(yte, p >= 0.5)), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path("data/release/biopet_sae_dataset_v1_2.tsv"))
    ap.add_argument("--embeddings", type=Path, default=Path("data/embeddings_650M_v1_2/embeddings.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/v1_2/domain_probe"))
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(a.release.open(newline=""), delimiter="\t"))
    emb = np.load(a.embeddings, allow_pickle=True)
    pos = {s: i for i, s in enumerate(emb["seq_ids"].astype(str))}
    miss = [r["seq_id"] for r in rows if r["seq_id"] not in pos]
    if miss: raise SystemExit(f"{len(miss)} rows without embeddings: {miss[:5]}")
    keep = [r for r in rows if r["domain"] in ("Bacteria", "Eukaryota")]
    idx = [pos[r["seq_id"]] for r in keep]
    y = np.array([r["domain"] == "Eukaryota" for r in keep], int)
    sp = np.array([r["split"] for r in keep])
    hard = np.array([r["negative_group"] == "ab_hydrolase_hard_negative" for r in keep])
    feats = {"ESM-2 650M max L33": emb["max_L33"][idx], "ESM-2 650M mean L33": emb["mean_L33"][idx],
             "amino-acid composition (20-d)": np.stack([comp(r["sequence"]) for r in keep]),
             "length only": np.array([[np.log(int(r["seq_len"]))] for r in keep], dtype=np.float32)}
    tr, te = sp == "train", sp == "test"
    res = {"rows_with_known_domain": len(keep), "euk": int(y.sum()), "bact": int((1 - y).sum()),
           "split_euk": {s: int(y[sp == s].sum()) for s in ("train", "val", "test")},
           "all_test": {}, "hard_negatives_only_test": {}, "transfer_non_hard_to_hard": {}}
    for name, X in feats.items():
        res["all_test"][name] = fit_eval(X[tr], y[tr], X[te], y[te])
        m_tr, m_te = tr & hard, te & hard
        res["hard_negatives_only_test"][name] = (
            fit_eval(X[m_tr], y[m_tr], X[m_te], y[m_te]) if y[m_te].min() != y[m_te].max() else "one class")
        nh = ~hard & (sp != "test")                # train on non-hard rows from train+val
        res["transfer_non_hard_to_hard"][name] = fit_eval(X[nh], y[nh], X[te & hard], y[te & hard])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "domain_probe.json").write_text(json.dumps(res, indent=2))
    print(f"known-domain rows {len(keep)}: {res['bact']} bacterial, {res['euk']} eukaryotic; eukaryotes per split {res['split_euk']}")
    for k in ("all_test", "hard_negatives_only_test", "transfer_non_hard_to_hard"):
        print(f"\n{k}")
        for n, v in res[k].items(): print(f"  {n:32s}", v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
