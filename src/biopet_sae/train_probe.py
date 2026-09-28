"""Linear probe on ESM-2 embeddings: 4-way classification and PET-vs-rest detection.

Trains on the `train` split and evaluates on `val`. The `test` split is never touched
here -- it is held for a single final evaluation once the layer and pooling are fixed.

Reports, per decision D5 and D6 in PROGRESS.md:
  * 4-way accuracy and macro-F1 across every SAE-compatible layer and both poolings
  * PET-vs-rest ROC-AUC with a bootstrap CI resampled over *components*, not sequences
    (sequences inside a component are near-duplicates and do not carry independent
    information)
  * an amino-acid composition baseline, because the real floor is not 0.5
  * the phylum control: performance restricted to Actinomycetota, where phylum carries
    no discriminative signal

The fitted StandardScaler is persisted alongside the model so later evaluation applies
the same transform rather than refitting.

Usage:
    uv run python -m biopet_sae.train_probe --embeddings data/embeddings_650M/embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

CLASSES = ["1_pet", "2_other_polyester", "3_fold_matched_esterase", "4_naive_control"]
AA = sorted("ACDEFGHIKLMNPQRSTVWY")


def composition(seqs: np.ndarray) -> np.ndarray:
    return np.array(
        [[str(s).count(a) / max(len(str(s)), 1) for a in AA] for s in seqs],
        dtype=np.float32,
    )


def bootstrap_auc_over_components(
    y: np.ndarray, scores: np.ndarray, components: np.ndarray,
    n_boot: int = 2000, seed: int = 0,
) -> tuple[float, float]:
    """Resample whole components with replacement; return the 2.5/97.5 percentiles."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(components)
    idx_by_comp = {c: np.flatnonzero(components == c) for c in uniq}
    out = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_comp[c] for c in picked])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        out.append(roc_auc_score(yy, scores[idx]))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def fit_eval(
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
    comps_va: np.ndarray, C: float = 1.0, seed: int = 42,
) -> dict[str, object]:
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(
        C=C, max_iter=5000, class_weight="balanced", random_state=seed,
    ).fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xva))
    proba = clf.predict_proba(scaler.transform(Xva))

    res: dict[str, object] = {
        "accuracy_4way": float((pred == yva).mean()),
        "macro_f1_4way": float(f1_score(yva, pred, average="macro", zero_division=0)),
    }
    # PET-vs-rest, taken from the multiclass head's PET probability
    pet_col = list(clf.classes_).index("1_pet") if "1_pet" in clf.classes_ else 0
    y_bin = (yva == "1_pet").astype(int)
    if 0 < y_bin.sum() < len(y_bin):
        scores = proba[:, pet_col]
        res["pet_auc"] = float(roc_auc_score(y_bin, scores))
        lo, hi = bootstrap_auc_over_components(y_bin, scores, comps_va)
        res["pet_auc_ci95"] = [round(lo, 4), round(hi, 4)]
        res["pet_recall"] = float(((pred == "1_pet") & (y_bin == 1)).sum() / y_bin.sum())
    per_class = {}
    for c in CLASSES:
        m = yva == c
        if m.sum():
            per_class[c] = {"n": int(m.sum()), "recall": float((pred[m] == c).mean())}
    res["per_class"] = per_class
    return res, scaler, clf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path,
                    default=Path("data/embeddings_650M/embeddings.npz"))
    ap.add_argument("--splits", type=Path, default=Path("data/processed/dataset_splits.tsv"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/probe_<model-dir-name>/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    d = np.load(args.embeddings, allow_pickle=True)
    out_dir = args.out or Path(f"results/probe_{args.embeddings.parent.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = d["class_label"].astype(str)
    split = d["split"].astype(str)
    comps = d["component"].astype(str)
    phylum = d["phylum"].astype(str)
    keep = np.isin(labels, CLASSES)
    tr = keep & (split == "train")
    va = keep & (split == "val")
    print(f"train {tr.sum()}  val {va.sum()}  (heldout/excluded not used)")
    for c in CLASSES:
        print(f"  {c:28s} train {int(((labels == c) & tr).sum()):4d}  "
              f"val {int(((labels == c) & va).sum()):4d}")

    feature_keys = sorted(
        [k for k in d.files if k.startswith(("mean_L", "max_L"))],
        key=lambda k: (k.split("_")[0], int(k.split("L")[1])),
    )
    rows = []
    best = None
    for key in feature_keys:
        X = d[key]
        res, scaler, clf = fit_eval(
            X[tr], labels[tr], X[va], labels[va], comps[va], seed=args.seed
        )
        pooling, layer = key.split("_L")
        row = {"features": key, "pooling": pooling, "layer": int(layer),
               "dim": X.shape[1], **{k: v for k, v in res.items() if k != "per_class"}}
        rows.append(row)
        if best is None or res["accuracy_4way"] > best[1]["accuracy_4way"]:
            best = (key, res, scaler, clf)

    # amino-acid composition baseline
    splits_df = pd.read_csv(args.splits, sep="\t", dtype=str, keep_default_na=False)
    seq_by_id = dict(zip(splits_df["seq_id"], splits_df["sequence"]))
    seqs = np.array([seq_by_id[s] for s in d["seq_ids"].astype(str)])
    Xc = composition(seqs)
    res_c, _, _ = fit_eval(Xc[tr], labels[tr], Xc[va], labels[va], comps[va], seed=args.seed)
    rows.append({"features": "aa_composition", "pooling": "-", "layer": -1, "dim": 20,
                 **{k: v for k, v in res_c.items() if k != "per_class"}})

    df = pd.DataFrame(rows).sort_values("accuracy_4way", ascending=False)
    print("\n=== 4-way classification, train -> val ===")
    cols = ["features", "dim", "accuracy_4way", "macro_f1_4way", "pet_auc",
            "pet_auc_ci95", "pet_recall"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))

    key, res, scaler, clf = best  # type: ignore[misc]
    print(f"\nbest features: {key}")
    print(json.dumps(res["per_class"], indent=2))

    # ---- phylum control (D6.2) -------------------------------------------
    print("\n=== phylum control: restricted to Actinomycetota ===")
    act = np.char.find(phylum, "Actinomycetota") >= 0
    X = d[key]
    ctrl_rows = []
    for name, mask in (("all phyla", np.ones_like(act)), ("Actinomycetota only", act)):
        t, v = tr & mask, va & mask
        n_pet_val = int(((labels == "1_pet") & v).sum())
        if v.sum() < 10 or n_pet_val < 3 or len(np.unique(labels[t])) < 2:
            ctrl_rows.append({"subset": name, "train": int(t.sum()), "val": int(v.sum()),
                              "val_pet": n_pet_val, "accuracy_4way": None, "pet_auc": None,
                              "note": "too few samples"})
            continue
        r, _, _ = fit_eval(X[t], labels[t], X[v], labels[v], comps[v], seed=args.seed)
        ctrl_rows.append({"subset": name, "train": int(t.sum()), "val": int(v.sum()),
                          "val_pet": n_pet_val,
                          "accuracy_4way": round(r["accuracy_4way"], 4),
                          "pet_auc": round(r.get("pet_auc", float("nan")), 4),
                          "note": ""})
    print(pd.DataFrame(ctrl_rows).to_string(index=False))

    # ---- persist ----------------------------------------------------------
    import pickle
    with (out_dir / "probe.pkl").open("wb") as fh:
        pickle.dump({"features": key, "scaler": scaler, "clf": clf,
                     "classes": list(clf.classes_)}, fh)
    df.to_csv(out_dir / "layer_sweep.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(
        {"best_features": key, "best": res, "sweep": rows,
         "phylum_control": ctrl_rows,
         "train_n": int(tr.sum()), "val_n": int(va.sum())},
        indent=2, default=str))
    print(f"\nwrote {out_dir}/results.json, layer_sweep.csv, probe.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
