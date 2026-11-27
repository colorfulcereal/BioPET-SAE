"""Apply the FROZEN catalytic-aspartate classifier (fitted on train only, C=0.1) to the test proteins, once.

Refits on train from the cached features (deterministic), checks the 9 latents/weights equal the ones reported,
then computes activations at every Asp of every test protein. Per protein: the highest-scoring Asp; called if its
score >= cut-off (none / 0.5 / 0.9). Precision = right / called; recall = right / all test proteins.

Usage:  uv run python -m biopet_sae.asp_test
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
from transformers import AutoTokenizer, EsmModel

from biopet_sae.sae_localise_v12 import one_each

R = 6


def main() -> int:
    out = Path("results/v1_2/asp_classifier")
    z = np.load(out / "features.npz"); X = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    meta = z["meta"]; pi, y = meta[:, 0], meta[:, 2]
    w = np.load("results/v1_2/triad_latent_search/windows.npz"); ids = w["ids"].astype(str); P, A = w["peak"], w["amax"]
    rows = {r["seq_id"]: r for r in csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t")}
    tv = [i for i in ids if rows[i]["split"] in ("train", "val")]
    rtr = np.array([rows[i]["split"] == "train" for i in tv])[pi]
    X0 = X[:, R * 10240:(R + 1) * 10240].tocsr()
    cols = np.flatnonzero(np.asarray((X0[rtr] > 0).sum(0)).ravel() >= 5)
    clf = LogisticRegression(C=0.1, penalty="l1", solver="liblinear", max_iter=3000, class_weight="balanced", random_state=0).fit(X0[:, cols][rtr], y[rtr])
    nz = np.flatnonzero(clf.coef_[0]); lat = cols[nz]; wt = clf.coef_[0, nz]; b = float(clf.intercept_[0])
    frozen = json.loads((out / "fit_train.json").read_text())
    expect = {l: v for l, v in frozen["latents_at_0.1"]}
    assert set(int(l) for l in lat) == set(expect) and all(abs(expect[int(l)] - v) < 0.05 for l, v in zip(lat, wt)), "model differs from the frozen one"
    (out / "frozen_model.json").write_text(json.dumps({"latents": {int(l): float(v) for l, v in zip(lat, wt)}, "intercept": b, "cutoffs": [0.5, 0.9]}, indent=2))
    print(f"frozen model confirmed: {len(lat)} latents, intercept {b:.3f}")

    pos = json.load(open("results/v1_2/algorithm_localisation/positive_triad_sites.json"))
    new = {r["seq_id"]: [int(x[1:]) for x in r["active_sites"].split(";") if x] for r in
           csv.DictReader(open("data/raw/hard_negatives_v12/candidates_annotation_filtered.tsv"), delimiter="\t")}
    old = json.load(open("data/processed/uniprot_negative_annotation.json"))
    test = [(k, i) for k, i in enumerate(ids) if rows[i]["split"] == "test"]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sae = ReLUSAE.from_pretrained(hf_hub_download("Elana/InterPLM-esm2-650m", "layer_33/ae_normalized.pt")).to(dev).eval()
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(dev).eval()
    recs = []
    with torch.no_grad():
        for n, (k, i) in enumerate(test):
            seq = rows[i]["sequence"]
            t = pos[i] if i in pos else one_each(seq, new.get(i) or old.get(i, {}).get("act") or [])
            enc = {a: v.to(dev) for a, v in tok(seq, return_tensors="pt").items()}
            f = sae.encode(esm(**enc, output_hidden_states=True).hidden_states[33][0, 1:-1, :].float())[:, lat.tolist()].cpu().numpy()
            D = [j for j, c in enumerate(seq) if c == "D"]
            p = 1 / (1 + np.exp(-(f[D] @ wt + b))); j = int(np.argmax(p))
            recs.append({"seq_id": i, "deg": rows[i]["y"] == "1", "pred": D[j], "score": float(p[j]), "true": t["D"], "n_asp": len(D),
                         "ser_ok": bool(A[k, 2925] >= 0.3 and P[k, 2925] == t["S"]), "his_ok": bool(A[k, 3927] >= 0.3 and P[k, 3927] == t["H"]),
                         "asp157_ok": bool(A[k, 157] >= 0.3 and P[k, 157] == t["D"])})
    n = len(recs); res = {"n": n, "per_protein": recs, "table": {}}
    print(f"test proteins {n} ({sum(r['deg'] for r in recs)} degraders); median Asp per protein {int(np.median([r['n_asp'] for r in recs]))}")
    print(f"\n{'cut-off':>8s} | {'called':>6s} | {'right':>5s} | {'wrong':>5s} | {'precision':>9s} | {'recall':>7s} | degraders (right/called) | hard neg (right/called)")
    for cut in (0.0, 0.5, 0.9):
        called = [r for r in recs if r["score"] >= cut]; ok = sum(r["pred"] == r["true"] for r in called)
        grp = {g: (sum(r["pred"] == r["true"] for r in called if r["deg"] == v), sum(1 for r in called if r["deg"] == v)) for g, v in (("deg", True), ("hn", False))}
        res["table"][str(cut)] = {"called": len(called), "right": int(ok), "precision": ok / max(len(called), 1), "recall": ok / n, "degraders": grp["deg"], "hard_negatives": grp["hn"]}
        print(f"{'none' if cut == 0 else cut:>8} | {len(called):6d} | {ok:5d} | {len(called)-ok:5d} | {ok/max(len(called),1):9.1%} | {ok/n:7.1%} | {grp['deg'][0]}/{grp['deg'][1]} of {sum(r['deg'] for r in recs)} | {grp['hn'][0]}/{grp['hn'][1]} of {sum(not r['deg'] for r in recs)}")
    print(f"\nfor comparison, latent 157 alone on the same proteins: {sum(r['asp157_ok'] for r in recs)}/{n} ({np.mean([r['asp157_ok'] for r in recs]):.1%})")
    for cut in (0.0, 0.5, 0.9):
        a3 = [r["ser_ok"] and r["his_ok"] and r["score"] >= cut and r["pred"] == r["true"] for r in recs]
        res["table"][str(cut)]["all_three"] = int(sum(a3))
        print(f"whole triad correct (Ser 2925, His 3927, Asp classifier, cut-off {'none' if cut == 0 else cut}): {sum(a3)}/{n} ({np.mean(a3):.1%}) | degraders {sum(x for x, r in zip(a3, recs) if r['deg'])}/{sum(r['deg'] for r in recs)}")
    bad = [r for r in recs if r["pred"] != r["true"]]
    print("\nmisses:", [(r["seq_id"], "degrader" if r["deg"] else "hard neg", round(r["score"], 3)) for r in bad])
    (out / "test_result.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    main()
