"""Choose one SAE latent per catalytic residue by window activation, test stability and held-out prediction.

Proteins: alpha/beta hydrolases with a curated Ser/Asp/His (degraders and hard negatives together).
  window score  score(latent, protein) = max activation within +/-3 residues of the catalytic residue
  choice        latent with the highest MEAN window score over train+val proteins
                (also recorded: how often each latent is the single best on one protein)
  stability     1,000 random halves (without replacement, seed 0) of train+val; count how often each latent
                has the highest mean window score
  test (a)      the same choice rule applied to the test proteins
  test (b)      PREDICTION, annotation not used: the latent's strongest residue over the whole protein
                (activation >= 0.3) lies within +/-3 of the catalytic residue
  chance        test (b) for all 10,240 latents

Usage:  uv run python -m biopet_sae.triad_latent_search
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

from biopet_sae.sae_localise_v12 import MAXLEN, one_each

W, FIRE, DRAWS = 3, 0.3, 1000
NAMES = {"S": "Ser", "D": "Asp", "H": "His"}


def main() -> int:
    out = Path("results/v1_2/triad_latent_search"); out.mkdir(parents=True, exist_ok=True)
    rows = {r["seq_id"]: r for r in csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t")}
    posites = json.load(open("results/v1_2/algorithm_localisation/positive_triad_sites.json"))
    new = {r["seq_id"]: [int(x[1:]) for x in r["active_sites"].split(";") if x] for r in
           csv.DictReader(open("data/raw/hard_negatives_v12/candidates_annotation_filtered.tsv"), delimiter="\t")}
    old = json.load(open("data/processed/uniprot_negative_annotation.json"))
    prot = []
    for i, r in rows.items():
        if len(r["sequence"]) > MAXLEN:
            continue
        if r["y"] == "1" and i in posites:
            prot.append((i, "degrader", {t: posites[i][t] for t in "SDH"}))
        elif r["negative_group"] == "ab_hydrolase_hard_negative":
            s = one_each(r["sequence"], new.get(i) or old.get(i, {}).get("act") or [])
            if s:
                prot.append((i, "hard_negative", s))
    N = len(prot)
    sp = np.array([rows[i]["split"] for i, _, _ in prot]); kind = np.array([k for _, k, _ in prot])
    fit, test = np.isin(sp, ["train", "val"]), sp == "test"
    print(f"choose on {fit.sum()} ({(fit & (kind=='degrader')).sum()} degraders, {(fit & (kind=='hard_negative')).sum()} hard neg) | "
          f"test {test.sum()} ({(test & (kind=='degrader')).sum()} degraders, {(test & (kind=='hard_negative')).sum()} hard neg)", flush=True)

    cache = out / "windows.npz"
    if cache.exists():
        z = np.load(cache); WIN, PEAK, AMAX = z["win"], z["peak"], z["amax"]
        assert list(z["ids"]) == [p[0] for p in prot]
    else:
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        sae = ReLUSAE.from_pretrained(hf_hub_download("Elana/InterPLM-esm2-650m", "layer_33/ae_normalized.pt")).to(dev).eval()
        tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(dev).eval()
        WIN = np.zeros((N, 3, 10240), np.float32); PEAK = np.zeros((N, 10240), np.int32); AMAX = np.zeros((N, 10240), np.float32)
        with torch.no_grad():
            for k, (i, _, s) in enumerate(prot):
                enc = {a: v.to(dev) for a, v in tok(rows[i]["sequence"], return_tensors="pt").items()}
                f = sae.encode(esm(**enc, output_hidden_states=True).hidden_states[33][0, 1:-1, :].float()).cpu().numpy()
                PEAK[k] = f.argmax(0); AMAX[k] = f.max(0)
                for j, t in enumerate("SDH"):
                    WIN[k, j] = f[max(0, s[t] - W): s[t] + W + 1].max(0)
                del f, enc
                if (k + 1) % 50 == 0:
                    if dev.type == "mps": torch.mps.empty_cache()
                    print(f"  {k+1}/{N}", flush=True)
        np.savez_compressed(cache, ids=np.array([p[0] for p in prot]), win=WIN, peak=PEAK, amax=AMAX)

    res = {"n": {"fit": int(fit.sum()), "test": int(test.sum())}, "draws": DRAWS, "residues": {}}
    fit_idx = np.flatnonzero(fit); rng = np.random.default_rng(0)
    halves = [rng.choice(fit_idx, len(fit_idx) // 2, replace=False) for _ in range(DRAWS)]
    chosen_hit = {}
    for j, t in enumerate("SDH"):
        S = WIN[:, j, :]; cur = np.array([p[2][t] for p in prot])
        hit = (AMAX >= FIRE) & (np.abs(PEAK - cur[:, None]) <= W)          # prediction rule
        exact = (AMAX >= FIRE) & (PEAK == cur[:, None])
        mean_fit = S[fit].mean(0); best = int(mean_fit.argmax())
        wins = Counter(S[fit].argmax(1).tolist())
        stab = Counter(int(S[h].mean(0).argmax()) for h in halves)
        mean_test = S[test].mean(0)
        test_rate_all = hit[test].mean(0)
        def card(l):
            d = {"latent": int(l), "mean_window_fit": round(float(mean_fit[l]), 3), "per_protein_wins_fit": f"{wins.get(int(l),0)}/{fit.sum()}",
                 "top_in_draws": f"{stab.get(int(l),0)}/{DRAWS}", "mean_window_test": round(float(mean_test[l]), 3),
                 "rank_on_test_by_same_rule": int((mean_test > mean_test[l]).sum()) + 1,
                 "pred_fit": round(float(hit[fit, l].mean()), 3), "pred_test": round(float(test_rate_all[l]), 3),
                 "pred_test_exact": round(float(exact[test, l].mean()), 3), "latents_better_on_test": int((test_rate_all > test_rate_all[l]).sum())}
            for k_ in ("degrader", "hard_negative"):
                m = test & (kind == k_); d[f"pred_test_{k_}"] = f"{int(hit[m, l].sum())}/{int(m.sum())}"
            return d
        top5 = [int(x) for x in np.argsort(-mean_fit)[:5]]
        res["residues"][t] = {"chosen": card(best), "next_by_mean_window": [card(l) for l in top5[1:]],
                              "stability_top": [{"latent": int(l), "draws": int(c)} for l, c in stab.most_common(5)],
                              "most_frequent_per_protein_winner": [{"latent": int(l), "wins": int(c), "pred_test": round(float(test_rate_all[l]), 3)} for l, c in wins.most_common(5)],
                              "test_top_latent_by_same_rule": int(mean_test.argmax()),
                              "chance_median_pred_test": float(np.median(test_rate_all))}
        chosen_hit[t] = hit[:, best]
        r = res["residues"][t]; c = r["chosen"]
        print(f"\n{NAMES[t]}: chosen latent {best} | mean window {c['mean_window_fit']} | winner on {c['per_protein_wins_fit']} proteins | top in {c['top_in_draws']} draws")
        print(f"   test, same rule: top latent is {r['test_top_latent_by_same_rule']} (chosen one ranks {c['rank_on_test_by_same_rule']})")
        print(f"   test PREDICTION within +/-3: {c['pred_test']:.1%} (degraders {c['pred_test_degrader']}, hard neg {c['pred_test_hard_negative']}); exact {c['pred_test_exact']:.1%}; "
              f"typical latent {r['chance_median_pred_test']:.1%}; latents better: {c['latents_better_on_test']}")
        print("   next by mean window: " + "; ".join(f"{x['latent']} ({x['mean_window_fit']}, test pred {x['pred_test']:.0%})" for x in r["next_by_mean_window"]))
        print("   stability: " + ", ".join(f"{x['latent']} {x['draws']}" for x in r["stability_top"]))
        print("   most frequent single-protein winners: " + ", ".join(f"{x['latent']} x{x['wins']} (test pred {x['pred_test']:.0%})" for x in r["most_frequent_per_protein_winner"][:4]))
    allthree = chosen_hit["S"] & chosen_hit["D"] & chosen_hit["H"]
    res["all_three_test"] = {k: f"{int(allthree[test & (kind == k)].sum())}/{int((test & (kind == k)).sum())}" for k in ("degrader", "hard_negative")}
    res["all_three_test"]["rate"] = round(float(allthree[test].mean()), 3)
    print("\nall three residues correct on test:", res["all_three_test"])

    # evidence codes behind the annotations
    raw = {e["primaryAccession"]: e for e in json.load(open("data/raw/hard_negatives_v12/uniprot_raw.json"))}
    codes = Counter(); n_cov = 0
    for i, k, _ in prot:
        if k != "hard_negative" or i not in raw: continue
        n_cov += 1
        for f in raw[i].get("features", []):
            if f["type"] == "Active site":
                codes.update({ev.get("evidenceCode", "none") for ev in f.get("evidences", [])} or {"no evidence tag"})
    res["evidence_codes_hard_negatives"] = {"proteins_covered": n_cov, "annotations": dict(codes)}
    print(f"\nevidence codes on active-site annotations ({n_cov} hard negatives from the cached query):", dict(codes))
    (out / "search.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
