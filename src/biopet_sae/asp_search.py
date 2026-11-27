"""Search for a better aspartate annotation: fixed-offset latents and combinations. Choose on TRAIN, check on VAL.

A candidate = (latent, offset k): it PROPOSES position peak+k when the latent's max activation >= 0.3 and the
residue at peak+k is an Asp. Correct = proposed position is the annotated catalytic Asp.
  singles       best candidates over all 10,240 latents and offsets -40..+40
  combinations  greedy priority list: take the best candidate; each next one is used only on proteins where the
                earlier ones propose nothing; up to 4. Candidates must be >= 80% correct when they propose.
Test proteins are not read.

Usage:  uv run python -m biopet_sae.asp_search
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from biopet_sae.sae_localise_v12 import one_each

FIRE, K = 0.3, 40


def main() -> int:
    base = Path("results/v1_2/triad_latent_search")
    z = np.load(base / "windows.npz"); ids = z["ids"].astype(str); P, A = z["peak"], z["amax"]
    rows = {r["seq_id"]: r for r in csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t")}
    pos = json.load(open("results/v1_2/algorithm_localisation/positive_triad_sites.json"))
    new = {r["seq_id"]: [int(x[1:]) for x in r["active_sites"].split(";") if x] for r in
           csv.DictReader(open("data/raw/hard_negatives_v12/candidates_annotation_filtered.tsv"), delimiter="\t")}
    old = json.load(open("data/processed/uniprot_negative_annotation.json"))
    split = np.array([rows[i]["split"] for i in ids])
    use = np.flatnonzero(np.isin(split, ["train", "val"]))
    P, A, ids, split = P[use], A[use], ids[use], split[use]
    n = len(ids); tr = split == "train"; va = split == "val"
    site = np.array([(pos[i] if i in pos else one_each(rows[i]["sequence"], new.get(i) or old.get(i, {}).get("act") or []))["D"] for i in ids])
    isD = [np.array([c == "D" for c in rows[i]["sequence"]]) for i in ids]
    offs = np.arange(-K, K + 1)
    PROP = np.zeros((len(offs), n, 10240), bool); OK = np.zeros_like(PROP)
    for i in range(n):
        q = P[i][None, :] + offs[:, None]
        inside = (q >= 0) & (q < len(isD[i]))
        d = np.zeros_like(inside); d[inside] = isD[i][q[inside]]
        PROP[:, i, :] = d & (A[i] >= FIRE)[None, :]
        OK[:, i, :] = PROP[:, i, :] & (q == site[i])
    ok_tr = OK[:, tr].sum(1); prop_tr = PROP[:, tr].sum(1)
    prec = ok_tr / np.maximum(prop_tr, 1)

    def stats(order, mask):
        done = np.zeros(n, bool); right = np.zeros(n, bool)
        for o, l in order:
            take = PROP[o, :, l] & ~done
            right |= take & OK[o, :, l]; done |= take
        c, p, m = int(right[mask].sum()), int(done[mask].sum()), int(mask.sum())
        return {"correct": f"{c}/{m} ({c/m:.1%})", "reports": f"{p/m:.1%}", "right_when_reports": f"{c/max(p,1):.1%}"}

    flat = np.argsort(-ok_tr, axis=None)[:6]
    singles = [np.unravel_index(f, ok_tr.shape) for f in flat]
    print(f"train {tr.sum()}  val {va.sum()}  (test not read)")
    print("\nBest single (latent, offset), chosen on train:")
    for o, l in singles:
        a, b = stats([(o, l)], tr), stats([(o, l)], va)
        print(f"  latent {l:5d} offset {offs[o]:+3d} | train {a['correct']}, right when reports {a['right_when_reports']} | val {b['correct']}, right when reports {b['right_when_reports']}")
    a, b = stats([(int(np.where(offs == 0)[0][0]), 157)], tr), stats([(int(np.where(offs == 0)[0][0]), 157)], va)
    print(f"  current: 157 offset +0     | train {a['correct']}, right when reports {a['right_when_reports']} | val {b['correct']}, right when reports {b['right_when_reports']}")

    chosen, done = [], np.zeros(n, bool)
    eligible = prec >= 0.80
    print("\nGreedy combination (each added only where earlier ones are silent), chosen on train:")
    for step in range(4):
        gain = (OK[:, tr & ~done, :].sum(1) * eligible)
        o, l = np.unravel_index(int(np.argmax(gain)), gain.shape)
        if gain[o, l] == 0: break
        chosen.append((int(o), int(l))); done |= PROP[o, :, l]
        a, b = stats(chosen, tr), stats(chosen, va)
        print(f"  +latent {l:5d} offset {offs[o]:+3d} | train {a['correct']}, reports {a['reports']}, right {a['right_when_reports']} | val {b['correct']}, reports {b['reports']}, right {b['right_when_reports']}")
    (base / "asp_search.json").write_text(json.dumps({"combination": [{"latent": l, "offset": int(offs[o])} for o, l in chosen],
                                                       "singles": [{"latent": int(l), "offset": int(offs[o])} for o, l in singles]}, indent=2))
    return 0


if __name__ == "__main__":
    main()
