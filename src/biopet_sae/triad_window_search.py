"""Find latents that fire within +/-3 residues of each catalytic residue on train+val positives; test them.

Rule, identical everywhere: a latent CONFORMS on a protein if its max activation is >= 0.3 and the residue where
it peaks is within 3 positions of the catalytic residue. No fitted offset, no sliding to a residue type.
  1. on train+val positives, conformance of every latent for Ser, Asp, His;
  2. the pattern = signed offsets (peak - residue) of the best latents;
  3. the SAME latents scored on test positives, and, as a larger held-out set that is not positives, on test hard
     negatives with curated sites;
  4. chance level = conformance of all 10,240 latents.
Selection among ties: smallest median |offset|, then highest mean activation, all on train+val only.

Usage:  uv run python -m biopet_sae.triad_window_search
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from interplm.sae.dictionary import ReLUSAE
from transformers import AutoTokenizer, EsmModel

from biopet_sae.sae_localise_v12 import one_each

W, FIRE = 3, 0.3
NAMES = {"S": "Ser", "D": "Asp", "H": "His"}


def main() -> int:
    out = Path("results/v1_2/algorithm_localisation")
    rows = {r["seq_id"]: r for r in csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t")}
    sites = json.loads((out / "positive_triad_sites.json").read_text())
    ids = sorted(sites)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sae = ReLUSAE.from_pretrained(hf_hub_download("Elana/InterPLM-esm2-650m", "layer_33/ae_normalized.pt")).to(dev).eval()
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(dev).eval()
    peaks, amax = [], []
    with torch.no_grad():
        for i in ids:
            enc = {k: v.to(dev) for k, v in tok(rows[i]["sequence"][:1022], return_tensors="pt").items()}
            f = sae.encode(esm(**enc, output_hidden_states=True).hidden_states[33][0, 1:-1, :].float()).cpu().numpy()
            peaks.append(f.argmax(0)); amax.append(f.max(0))
    P, M = np.array(peaks), np.array(amax)
    sp = np.array([sites[i]["split"] for i in ids]); comp = np.array([sites[i]["component"] for i in ids])
    fit, test = np.isin(sp, ["train", "val"]), sp == "test"

    # test hard negatives with curated triads (peaks cached by the localisation run)
    z = np.load("results/v1_2/localisation/peaks.npz"); hid = z["ids"].astype(str)
    new = {r["seq_id"]: [int(x[1:]) for x in r["active_sites"].split(";") if x] for r in
           csv.DictReader(open("data/raw/hard_negatives_v12/candidates_annotation_filtered.tsv"), delimiter="\t")}
    old = json.load(open("data/processed/uniprot_negative_annotation.json"))
    hm = [k for k, i in enumerate(hid) if rows[i]["split"] == "test" and rows[i]["y"] == "0"]
    hs = [one_each(rows[hid[k]]["sequence"], new.get(hid[k]) or old.get(hid[k], {}).get("act") or []) for k in hm]
    HP, HM = z["peaks"][hm], z["amax"][hm]

    res = {"rule": f"activation >= {FIRE} and |peak - residue| <= {W}", "n": {"fit_positives": int(fit.sum()), "test_positives": int(test.sum()), "test_hard_negatives": len(hm)},
           "fit_families": int(len(set(comp[fit]))), "test_families": int(len(set(comp[test]))), "residues": {}}
    print(f"train+val positives {fit.sum()} ({res['fit_families']} families) | test positives {test.sum()} ({res['test_families']} families) | test hard negatives {len(hm)}")
    for t in "SDH":
        cur = np.array([sites[i][t] for i in ids]); off = P - cur[:, None]
        ok = (M >= FIRE) & (np.abs(off) <= W)
        rf, rt = ok[fit].mean(0), ok[test].mean(0)
        hcur = np.array([s[t] for s in hs]); hok = (HM >= FIRE) & (np.abs(HP - hcur[:, None]) <= W); rh = hok.mean(0)
        best = rf.max(); tied = np.flatnonzero(rf == best)
        med = np.array([np.median(np.abs(off[fit, l][ok[fit, l]])) for l in tied]); act = M[fit][:, tied].mean(0)
        order = tied[np.lexsort((-act, med))]
        show = list(order[:5])
        if len(show) < 5:                                   # fill with the next-best on fit
            rest = np.array([l for l in np.argsort(-rf, kind="stable") if l not in set(show)])
            show += list(rest[: 5 - len(show)])
        lat = []
        for l in show:
            c = Counter(int(o) for o in off[fit, l][ok[fit, l]])
            lat.append({"latent": int(l), "fit": f"{ok[fit, l].sum()}/{fit.sum()}", "offsets": dict(c.most_common(4)),
                        "test_pos": f"{ok[test, l].sum()}/{test.sum()}", "test_hard": f"{hok[:, l].sum()}/{len(hm)}", "test_hard_rate": round(float(rh[l]), 3)})
        res["residues"][t] = {"best_fit_rate": round(float(best), 3), "n_latents_at_best": int(len(tied)),
                              "chance": {"fit_median": float(np.median(rf)), "test_pos_median": float(np.median(rt)), "test_hard_median": float(np.median(rh)),
                                         "latents_with_all_test_positives": int((rt == 1).sum()), "latents_over_half_on_test_hard": int((rh > 0.5).sum())},
                              "latents": lat}
        r = res["residues"][t]
        print(f"\n{NAMES[t]}: best train+val conformance {best:.0%}, reached by {len(tied)} latent(s). Chance (median latent): fit {r['chance']['fit_median']:.0%}, test hard neg {r['chance']['test_hard_median']:.0%}. "
              f"Latents hitting all test positives by this rule: {r['chance']['latents_with_all_test_positives']}")
        print(f"  {'latent':>6s}  {'train+val':>9s}  {'test pos':>8s}  {'test hard neg':>16s}  offsets on train+val (offset: count)")
        for x in lat:
            print(f"  {x['latent']:6d}  {x['fit']:>9s}  {x['test_pos']:>8s}  {x['test_hard']:>8s} ({x['test_hard_rate']:.0%})  {x['offsets']}")
    (out / "window_search.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
