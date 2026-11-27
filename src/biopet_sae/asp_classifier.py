"""Which aspartate is catalytic? Candidate ranking over every Asp, train+val only (test is not read).

One row per Asp residue in each annotated protein; label 1 = the annotated catalytic Asp. A protein is correct if
the top-scoring Asp is the catalytic one (every protein gets exactly one answer).
Methods, compared by 5-fold cross-validation with whole homology components held out:
  sae_l1     L1 logistic regression on SAE latent activations at offsets around the Asp (window and C swept)
  esm_l2     L2 logistic regression on the raw ESM-2 layer-33 vector at the Asp (no autoencoder)
  spacing    the Asp whose distance to the located His (latent 3927) is closest to the median training distance
  latent157  the rule so far: latent 157's peak, if it is >= 0.3 and on an Asp (chosen on these proteins: optimistic)
  random     expected accuracy of picking an Asp at random
Also: accuracy with abstention (answer only the most confident half / 75%).

Usage:  uv run python -m biopet_sae.asp_classifier
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from interplm.sae.dictionary import ReLUSAE
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, EsmModel

from biopet_sae.sae_localise_v12 import one_each

R = 6                                     # offsets -6..+6 stored
WINDOWS, CS = [0, 3, 6], [0.03, 0.1, 0.3]


def main() -> int:
    out = Path("results/v1_2/asp_classifier"); out.mkdir(parents=True, exist_ok=True)
    base = Path("results/v1_2/triad_latent_search")
    z = np.load(base / "windows.npz"); ids = z["ids"].astype(str); P, A = z["peak"], z["amax"]
    rows = {r["seq_id"]: r for r in csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t")}
    pos = json.load(open("results/v1_2/algorithm_localisation/positive_triad_sites.json"))
    new = {r["seq_id"]: [int(x[1:]) for x in r["active_sites"].split(";") if x] for r in
           csv.DictReader(open("data/raw/hard_negatives_v12/candidates_annotation_filtered.tsv"), delimiter="\t")}
    old = json.load(open("data/processed/uniprot_negative_annotation.json"))
    use = [k for k, i in enumerate(ids) if rows[i]["split"] in ("train", "val")]      # test never touched
    prot = []
    for k in use:
        i = ids[k]
        s = pos[i] if i in pos else one_each(rows[i]["sequence"], new.get(i) or old.get(i, {}).get("act") or [])
        prot.append((k, i, s["D"]))

    cache = out / "features.npz"
    if cache.exists():
        z2 = np.load(cache, allow_pickle=True)
        X = sparse.csr_matrix((z2["data"], z2["indices"], z2["indptr"]), shape=tuple(z2["shape"]))
        E, meta = z2["esm"], z2["meta"]
    else:
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        sae = ReLUSAE.from_pretrained(hf_hub_download("Elana/InterPLM-esm2-650m", "layer_33/ae_normalized.pt")).to(dev).eval()
        tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(dev).eval()
        blocks, E, meta = [], [], []
        with torch.no_grad():
            for n, (k, i, cat) in enumerate(prot):
                seq = rows[i]["sequence"]
                enc = {a: v.to(dev) for a, v in tok(seq, return_tensors="pt").items()}
                h = esm(**enc, output_hidden_states=True).hidden_states[33][0, 1:-1, :].float()
                f = sae.encode(h).cpu().numpy(); h = h.cpu().numpy()
                L = len(seq)
                for q in [j for j, c in enumerate(seq) if c == "D"]:
                    row = np.zeros((2 * R + 1, f.shape[1]), np.float32)
                    lo, hi = max(0, q - R), min(L, q + R + 1)
                    row[lo - (q - R): hi - (q - R)] = f[lo:hi]
                    blocks.append(sparse.csr_matrix(row.reshape(1, -1))); E.append(h[q]); meta.append((n, q, int(q == cat)))
                del f, h, enc
                if (n + 1) % 50 == 0:
                    if dev.type == "mps": torch.mps.empty_cache()
                    print(f"  {n+1}/{len(prot)}", flush=True)
        X = sparse.vstack(blocks).tocsr(); E = np.array(E, np.float32); meta = np.array(meta)
        np.savez_compressed(cache, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape), esm=E, meta=meta)
    pi, q, y = meta[:, 0], meta[:, 1], meta[:, 2]
    n = len(prot)
    comp = np.array([rows[i]["component"] for _, i, _ in prot]); deg = np.array([rows[i]["y"] == "1" for _, i, _ in prot])
    nD = np.bincount(pi, minlength=n)
    print(f"{n} train+val proteins, {len(y)} Asp residues (median {int(np.median(nD))} per protein), {int(y.sum())} catalytic; test not read", flush=True)
    folds = list(GroupKFold(5).split(np.arange(n), groups=comp))

    def pick(score):
        """per protein: correctness of the top Asp and the margin used as confidence"""
        ok = np.zeros(n, bool); conf = np.zeros(n)
        for p in range(n):
            m = np.flatnonzero(pi == p); o = m[np.argsort(-score[m])]
            ok[p] = y[o[0]] == 1; conf[p] = score[o[0]] - (score[o[1]] if len(o) > 1 else 0)
        return ok, conf

    def summary(ok, conf=None):
        d = {"correct": f"{int(ok.sum())}/{n}", "acc": round(float(ok.mean()), 3), "degraders": f"{int(ok[deg].sum())}/{int(deg.sum())}"}
        if conf is not None:
            order = np.argsort(-conf)
            for frac in (0.5, 0.75):
                d[f"acc_top{int(frac*100)}"] = round(float(ok[order[: int(frac * n)]].mean()), 3)
        return d

    res = {"n": n, "residues": int(len(y)), "methods": {}}
    res["methods"]["random"] = {"acc": round(float(np.mean(1 / nD)), 3)}
    ok157 = np.array([A[k, 157] >= 0.3 and int(P[k, 157]) == cat for k, _, cat in prot])
    res["methods"]["latent157"] = summary(ok157)

    his = []
    for k, i, cat in prot:
        s = rows[i]["sequence"]
        his.append(int(P[k, 3927]) if A[k, 3927] >= 0.3 and s[P[k, 3927]] == "H" else None)
    ok = np.zeros(n, bool)
    for tr, te in folds:
        gaps = [his[p] - prot[p][2] for p in tr if his[p] is not None]
        med = np.median(gaps)
        for p in te:
            if his[p] is None: continue
            m = np.flatnonzero(pi == p)
            ok[p] = q[m][np.argmin(np.abs((his[p] - q[m]) - med))] == prot[p][2]
    allg = np.array([his[p] - prot[p][2] for p in range(n) if his[p] is not None])
    res["methods"]["spacing"] = {**summary(ok), "median_His_minus_Asp": float(np.median(allg)), "iqr": [float(np.percentile(allg, 25)), float(np.percentile(allg, 75))]}
    for k_ in ("random", "latent157", "spacing"): print(f"{k_:10s} {res['methods'][k_]}", flush=True)

    def cv(Xm, scale, **kw):
        score = np.zeros(len(y)); sizes = []
        for tr, te in folds:
            a = np.isin(pi, tr); b = np.isin(pi, te)
            Xa, Xb = Xm[a], Xm[b]
            if scale:
                sc = StandardScaler().fit(Xa); Xa, Xb = sc.transform(Xa), sc.transform(Xb)
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0, **kw).fit(Xa, y[a])
            score[b] = clf.decision_function(Xb); sizes.append(int((clf.coef_ != 0).sum()))
        return score, sizes

    for C in (0.001, 0.01):
        s, _ = cv(E, True, C=C)
        ok, conf = pick(s); res["methods"][f"esm_l2 C={C}"] = summary(ok, conf); print(f"esm_l2 C={C}", res["methods"][f"esm_l2 C={C}"], flush=True)
    for w in WINDOWS:
        cols = np.concatenate([np.arange((R + o) * 10240, (R + o + 1) * 10240) for o in range(-w, w + 1)])
        Xw = X[:, cols].tocsr()
        Xw = Xw[:, np.flatnonzero(np.asarray((Xw > 0).sum(0)).ravel() >= 5)]
        for C in CS:
            s, sizes = cv(Xw, False, C=C, penalty="l1", solver="liblinear")
            ok, conf = pick(s)
            key = f"sae_l1 window=+/-{w} C={C}"
            res["methods"][key] = {**summary(ok, conf), "median_nonzero_features": int(np.median(sizes))}
            print(key, res["methods"][key], flush=True)
    (out / "asp_classifier.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    main()
