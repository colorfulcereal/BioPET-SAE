"""Fit the catalytic-aspartate classifier on TRAIN only, look at VALIDATION. Test is not read.

One row per Asp residue; features = the SAE latent activations AT that residue; label = catalytic or not.
L1 logistic regression, class_weight balanced; columns kept = latents non-zero on >= 5 training rows.
C is compared on validation. Uses features.npz written by `asp_classifier`.

Usage:  uv run python -m biopet_sae.asp_fit_train
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

R = 6


def main() -> int:
    out = Path("results/v1_2/asp_classifier")
    z = np.load(out / "features.npz"); X = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    meta = z["meta"]; pi, y = meta[:, 0], meta[:, 2]
    w = np.load("results/v1_2/triad_latent_search/windows.npz"); ids = w["ids"].astype(str)
    rows = {r["seq_id"]: r for r in csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t")}
    P = [i for i in ids if rows[i]["split"] in ("train", "val")]
    split = np.array([rows[i]["split"] for i in P]); deg = np.array([rows[i]["y"] == "1" for i in P]); n = len(P)
    X0 = X[:, R * 10240:(R + 1) * 10240].tocsr()
    rtr, rva = split[pi] == "train", split[pi] == "val"
    cols = np.flatnonzero(np.asarray((X0[rtr] > 0).sum(0)).ravel() >= 5); Xc = X0[:, cols]
    print(f"train: {int((split=='train').sum())} proteins, {rtr.sum()} Asp rows, {int(y[rtr].sum())} catalytic | val: {int((split=='val').sum())} proteins, {rva.sum()} rows | latents usable: {len(cols)}")
    res = {}
    print(f"\n{'C':>5s} | {'latents used':>12s} | {'train correct':>16s} | {'val correct':>15s} | val right when answering at 0.5 / 0.9")
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=3000, class_weight="balanced", random_state=0).fit(Xc[rtr], y[rtr])
        p = clf.predict_proba(Xc)[:, 1]
        top = np.zeros(n); ok = np.zeros(n, bool)
        for k in range(n):
            m = np.flatnonzero(pi == k); j = m[np.argmax(p[m])]; top[k] = p[j]; ok[k] = y[j] == 1
        tr, va = split == "train", split == "val"
        extra = []
        for t in (0.5, 0.9):
            a = va & (top >= t); extra.append(f"{int((a & ok).sum())}/{int(a.sum())}")
        nz = int((clf.coef_ != 0).sum())
        res[str(C)] = {"latents": nz, "train": f"{int(ok[tr].sum())}/{int(tr.sum())}", "val": f"{int(ok[va].sum())}/{int(va.sum())}", "val_at_0.5": extra[0], "val_at_0.9": extra[1]}
        print(f"{C:5g} | {nz:12d} | {int(ok[tr].sum()):4d}/{int(tr.sum())} ({ok[tr].mean():6.1%}) | {int(ok[va].sum()):4d}/{int(va.sum())} ({ok[va].mean():5.1%}) | {extra[0]} , {extra[1]}")
        if C == 0.1:
            order = np.argsort(-np.abs(clf.coef_.ravel()))
            used = [(int(cols[i]), round(float(clf.coef_[0, i]), 2)) for i in order if clf.coef_[0, i] != 0]
            res["latents_at_0.1"] = used; res["intercept_at_0.1"] = round(float(clf.intercept_[0]), 3)
            print("        latents (weight):", used, " intercept", res["intercept_at_0.1"])
    ok157 = np.array([w["amax"][list(ids).index(i), 157] >= 0.3 for i in P])
    (out / "fit_train.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    main()
