"""Annotate a triad from fixed SAE latents, and score it (train+val only unless --split test).

Per residue: Ser = latent 2925; His = latent 3927; Asp = latents 2661 and 157.
  a latent PROPOSES a position if its max activation >= 0.3: the nearest residue of that type within +/-3 of the
  residue where it is strongest. With two latents: same proposal or only one -> that; different -> the proposal of the
  latent with the higher activation; none -> "not located".
Correct = proposed position equals the annotated one.  Uses windows.npz (peak and max per protein and latent).

Usage:  uv run python -m biopet_sae.triad_annotate [--split trainval|test]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from biopet_sae.sae_localise_v12 import one_each

W, FIRE = 3, 0.3
LAT = {"S": [2925], "D": [2661, 157], "H": [3927]}


def nearest(seq: str, q: int, t: str):
    cand = [j for j in range(max(0, q - W), min(len(seq), q + W + 1)) if seq[j] == t]
    return min(cand, key=lambda j: (abs(j - q), j)) if cand else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["trainval", "test"], default="trainval")
    a = ap.parse_args(argv)
    base = Path("results/v1_2/triad_latent_search")
    z = np.load(base / "windows.npz"); ids = z["ids"].astype(str); P, A = z["peak"], z["amax"]
    rows = {r["seq_id"]: r for r in csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t")}
    pos = json.load(open("results/v1_2/algorithm_localisation/positive_triad_sites.json"))
    new = {r["seq_id"]: [int(x[1:]) for x in r["active_sites"].split(";") if x] for r in
           csv.DictReader(open("data/raw/hard_negatives_v12/candidates_annotation_filtered.tsv"), delimiter="\t")}
    old = json.load(open("data/processed/uniprot_negative_annotation.json"))
    want = ("train", "val") if a.split == "trainval" else ("test",)
    ks = [k for k, i in enumerate(ids) if rows[i]["split"] in want]
    recs = []
    for k in ks:
        i = ids[k]; seq = rows[i]["sequence"]
        truth = pos[i] if i in pos else one_each(seq, new.get(i) or old.get(i, {}).get("act") or [])
        r = {"deg": rows[i]["y"] == "1"}
        for t, lats in LAT.items():
            props = {l: (nearest(seq, int(P[k, l]), t), float(A[k, l])) for l in lats if A[k, l] >= FIRE}
            props = {l: v for l, v in props.items() if v[0] is not None}
            sites = {v[0] for v in props.values()}
            chosen = None if not props else max(props.values(), key=lambda v: v[1])[0]
            r[t] = {"called": chosen is not None, "correct": chosen == truth[t], "n_props": len(props), "disagree": len(sites) > 1,
                    "single": {l: (l in props and props[l][0] == truth[t]) for l in lats}}
        recs.append(r)
    groups = {"degraders": [r for r in recs if r["deg"]], "hard negatives": [r for r in recs if not r["deg"]], "all": recs}
    print(f"{a.split}: {len(recs)} proteins" + (" (test not read)" if a.split == "trainval" else ""))
    out = {}
    for g, R in groups.items():
        n = len(R); out[g] = {"n": n}
        print(f"\n  {g} (n={n})")
        for t, name in (("S", "Ser"), ("D", "Asp"), ("H", "His")):
            c = sum(r[t]["called"] for r in R); ok = sum(r[t]["correct"] for r in R)
            out[g][name] = {"called": c, "correct": ok}
            extra = ""
            if t == "D":
                both = [r for r in R if r[t]["n_props"] == 2]; dis = [r for r in both if r[t]["disagree"]]; agr = [r for r in both if not r[t]["disagree"]]
                out[g][name].update({"2661_alone": sum(r[t]["single"][2661] for r in R), "157_alone": sum(r[t]["single"][157] for r in R),
                                     "agree": len(agr), "agree_correct": sum(r[t]["correct"] for r in agr), "disagree": len(dis), "disagree_correct": sum(r[t]["correct"] for r in dis)})
                extra = (f"\n        2661 alone correct {out[g][name]['2661_alone']}/{n} | 157 alone {out[g][name]['157_alone']}/{n}"
                         f"\n        both propose the same site: {len(agr)} ({out[g][name]['agree_correct']} correct) | propose different sites: {len(dis)} ({out[g][name]['disagree_correct']} correct)")
            print(f"    {name}: called {c}/{n} ({c/n:.1%}) | correct {ok}/{n} ({ok/n:.1%}) | correct when called {ok/max(c,1):.1%}{extra}")
        allok = sum(all(r[t]["correct"] for t in "SDH") for r in R); allc = sum(all(r[t]["called"] for t in "SDH") for r in R)
        out[g]["all_three"] = {"called": allc, "correct": allok}
        print(f"    all three: located {allc}/{n} ({allc/n:.1%}) | all correct {allok}/{n} ({allok/n:.1%}) | correct when located {allok/max(allc,1):.1%}")
    (base / f"annotate_{a.split}.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
