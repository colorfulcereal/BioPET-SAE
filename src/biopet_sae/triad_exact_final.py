"""FINAL catalytic-triad latent selection and held-out test (rule frozen by the project lead, 2026-11-26).

Rule: a latent is correct on a protein if its max activation is >= 0.3 AND the residue where that maximum
occurs is exactly the annotated catalytic residue.
  1. on train+val (degraders and hard negatives together), count correct proteins for all 10,240 latents;
     choose the latent with the most (ties: higher mean peak activation), separately for Ser, Asp, His;
  2. 1,000 random halves of train+val without replacement (seed 0): how often each chosen latent is first;
  3. apply the three chosen latents, unchanged, to test; report degraders and hard negatives separately,
     all three together, the rank among all 10,240 latents, and which latent the same rule would pick on test.
The rule was refined after seeing train+val results; test was not read before this script.
Uses windows.npz from `triad_latent_search`.

Usage:  uv run python -m biopet_sae.triad_exact_final
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from biopet_sae.sae_localise_v12 import one_each

FIRE, DRAWS = 0.3, 1000
NAMES = {"S": "Ser", "D": "Asp", "H": "His"}


def main() -> int:
    base = Path("results/v1_2/triad_latent_search")
    z = np.load(base / "windows.npz"); ids = z["ids"].astype(str); P, A = z["peak"], z["amax"]
    rows = {r["seq_id"]: r for r in csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t")}
    pos = json.load(open("results/v1_2/algorithm_localisation/positive_triad_sites.json"))
    new = {r["seq_id"]: [int(x[1:]) for x in r["active_sites"].split(";") if x] for r in
           csv.DictReader(open("data/raw/hard_negatives_v12/candidates_annotation_filtered.tsv"), delimiter="\t")}
    old = json.load(open("data/processed/uniprot_negative_annotation.json"))
    site = [pos[i] if i in pos else one_each(rows[i]["sequence"], new.get(i) or old.get(i, {}).get("act") or []) for i in ids]
    split = np.array([rows[i]["split"] for i in ids]); deg = np.array([rows[i]["y"] == "1" for i in ids])
    comp = np.array([rows[i]["component"] for i in ids])
    fit, test = np.isin(split, ["train", "val"]), split == "test"
    H = {t: (A >= FIRE) & (P == np.array([s[t] for s in site])[:, None]) for t in "SDH"}
    mact = A[fit].mean(0)                                          # tie-break from train+val only
    fi = np.flatnonzero(fit); rng = np.random.default_rng(0)
    halves = [rng.choice(fi, len(fi) // 2, replace=False) for _ in range(DRAWS)]
    res = {"rule": "max activation >= 0.3 and argmax residue == annotated catalytic residue",
           "n": {"trainval": int(fit.sum()), "trainval_degraders": int((fit & deg).sum()), "test": int(test.sum()), "test_degraders": int((test & deg).sum()),
                 "trainval_families": int(len(set(comp[fit]))), "test_families": int(len(set(comp[test]))),
                 "test_degrader_families": int(len(set(comp[test & deg])))}, "residues": {}}

    def frac(h, m):
        return f"{int(h[m].sum())}/{int(m.sum())} ({h[m].mean():.1%})"

    best = {}
    print(f"STEP 1-2  TRAIN+VAL: {fit.sum()} proteins ({(fit & deg).sum()} degraders, {(fit & ~deg).sum()} hard negatives)")
    for t in "SDH":
        cnt = H[t][fit].sum(0); order = np.argsort(-(cnt + mact / 10)); b = int(order[0]); best[t] = b
        top = np.zeros(10240, int)
        for h in halves:
            top[int(np.argmax(H[t][h].sum(0) + mact / 10))] += 1
        res["residues"][t] = {"latent": b, "trainval": frac(H[t][:, b], fit), "trainval_degraders": frac(H[t][:, b], fit & deg),
                              "trainval_hard": frac(H[t][:, b], fit & ~deg), "draws_first": int(top[b]),
                              "runner_up": {"latent": int(order[1]), "trainval": frac(H[t][:, int(order[1])], fit), "draws_first": int(top[order[1]])},
                              "median_latent_trainval": float(np.median(H[t][fit].mean(0)))}
        r = res["residues"][t]
        print(f"  {NAMES[t]}: latent {b:5d} | all {r['trainval']} | degraders {r['trainval_degraders']} | hard neg {r['trainval_hard']} | first in {r['draws_first']}/{DRAWS}"
              f" | runner-up {r['runner_up']['latent']} {r['runner_up']['trainval']}")
    a3 = H["S"][:, best["S"]] & H["D"][:, best["D"]] & H["H"][:, best["H"]]
    res["all_three"] = {"trainval": frac(a3, fit), "trainval_degraders": frac(a3, fit & deg), "trainval_hard": frac(a3, fit & ~deg)}
    print(f"  all three: {res['all_three']['trainval']} | degraders {res['all_three']['trainval_degraders']} | hard neg {res['all_three']['trainval_hard']}")

    print(f"\nSTEP 3  TEST (latents frozen): {test.sum()} proteins ({(test & deg).sum()} degraders, {(test & ~deg).sum()} hard negatives)")
    for t in "SDH":
        b = best[t]; rate = H[t][test].mean(0)
        tt = int(np.argmax(H[t][test].sum(0) + A[test].mean(0) / 10))
        r = res["residues"][t]
        r.update({"test": frac(H[t][:, b], test), "test_degraders": frac(H[t][:, b], test & deg), "test_hard": frac(H[t][:, b], test & ~deg),
                  "latents_better_on_test": int((rate > rate[b]).sum()), "latents_equal_on_test": int((rate == rate[b]).sum()) - 1,
                  "median_latent_test": float(np.median(rate)), "same_rule_on_test_picks": tt, "that_latent_test": frac(H[t][:, tt], test)})
        print(f"  {NAMES[t]}: latent {b:5d} | all {r['test']} | degraders {r['test_degraders']} | hard neg {r['test_hard']} | latents better: {r['latents_better_on_test']}"
              f" | typical latent {r['median_latent_test']:.1%} | same rule run on test picks {tt} ({r['that_latent_test']})")
    res["all_three"].update({"test": frac(a3, test), "test_degraders": frac(a3, test & deg), "test_hard": frac(a3, test & ~deg)})
    print(f"  all three: {res['all_three']['test']} | degraders {res['all_three']['test_degraders']} | hard neg {res['all_three']['test_hard']}")
    print(f"\nfamilies: train+val {res['n']['trainval_families']} homology components, test {res['n']['test_families']} (degraders in {res['n']['test_degrader_families']})")
    (base / "final_exact.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
