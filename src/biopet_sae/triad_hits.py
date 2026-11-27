"""Choose the catalytic latent per residue by HIT COUNT, on train+val only, at several firing thresholds.

hit(latent, protein) = latent's max activation >= threshold AND the residue where it peaks (over the whole
protein) is within +/-3 of the annotated catalytic residue. Chosen latent = most hits over train+val proteins
(degraders + hard negatives together); ties broken by higher mean peak activation. Stability = share of 1,000
random halves (no replacement, seed 0) in which the latent again has the most hits (same tie rule).
TEST PROTEINS ARE NOT READ. Uses windows.npz written by `triad_latent_search`.

Usage:  uv run python -m biopet_sae.triad_hits
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

W, THRESHOLDS, DRAWS = 3, [0.15, 0.3, 0.5], 1000
NAMES = {"S": "Ser", "D": "Asp", "H": "His"}


def main() -> int:
    base = Path("results/v1_2/triad_latent_search")
    z = np.load(base / "windows.npz"); ids = z["ids"].astype(str); PEAK, AMAX = z["peak"], z["amax"]
    rows = {r["seq_id"]: r for r in csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t")}
    pos = json.load(open("results/v1_2/algorithm_localisation/positive_triad_sites.json"))
    new = {r["seq_id"]: [int(x[1:]) for x in r["active_sites"].split(";") if x] for r in
           csv.DictReader(open("data/raw/hard_negatives_v12/candidates_annotation_filtered.tsv"), delimiter="\t")}
    old = json.load(open("data/processed/uniprot_negative_annotation.json"))
    from biopet_sae.sae_localise_v12 import one_each
    cur = {}
    for i in ids:
        cur[i] = pos[i] if i in pos else one_each(rows[i]["sequence"], new.get(i) or old.get(i, {}).get("act") or [])
    split = np.array([rows[i]["split"] for i in ids]); deg = np.array([rows[i]["y"] == "1" for i in ids])
    keep = np.isin(split, ["train", "val"])                 # test rows are dropped here and never used
    PEAK, AMAX, ids, deg = PEAK[keep], AMAX[keep], ids[keep], deg[keep]
    n = len(ids); rng = np.random.default_rng(0)
    halves = [rng.choice(n, n // 2, replace=False) for _ in range(DRAWS)]
    print(f"train+val: {n} proteins = {deg.sum()} degraders + {(~deg).sum()} hard negatives (test not read)")
    res = {"n": int(n), "degraders": int(deg.sum()), "hard_negatives": int((~deg).sum()), "thresholds": {}}
    for thr in THRESHOLDS:
        fire = AMAX >= thr
        mean_act = AMAX.mean(0)
        allhit = np.ones(n, bool); out = {}
        print(f"\n===== firing threshold {thr} =====")
        print(f"{'residue':7s} {'latent':>6s} | {'degraders':>14s} | {'hard negatives':>16s} | {'all':>14s} | {'on top in draws':>15s} | typical latent | runner-up")
        for t in "SDH":
            site = np.array([cur[i][t] for i in ids])
            H = fire & (np.abs(PEAK - site[:, None]) <= W)
            count = H.sum(0)
            score = count + mean_act / 10                   # tie-break: higher mean activation (< 1 hit)
            order = np.argsort(-score); best, second = int(order[0]), int(order[1])
            top = np.zeros(10240, int)
            for h in halves:
                top[int(np.argmax(H[h].sum(0) + mean_act / 10))] += 1
            d, nd = int(H[deg, best].sum()), int(H[~deg, best].sum())
            allhit &= H[:, best]
            out[t] = {"latent": best, "degraders": f"{d}/{deg.sum()}", "hard_negatives": f"{nd}/{(~deg).sum()}", "all": f"{int(count[best])}/{n}",
                      "draws_on_top": int(top[best]), "typical_latent_rate": float(np.median(H.mean(0))), "runner_up": second, "runner_up_hits": int(count[second]),
                      "other_draw_winners": [(int(l), int(top[l])) for l in np.argsort(-top)[:3] if top[l]]}
            print(f"{NAMES[t]:7s} {best:6d} | {d:3d}/{deg.sum()} ({d/deg.sum():6.1%}) | {nd:4d}/{(~deg).sum()} ({nd/(~deg).sum():6.1%}) | {int(count[best]):4d}/{n} ({count[best]/n:5.1%}) | "
                  f"{top[best]:9d}/{DRAWS} | {np.median(H.mean(0)):13.1%} | {second} ({int(count[second])})")
        out["all_three"] = {"degraders": f"{int(allhit[deg].sum())}/{deg.sum()}", "hard_negatives": f"{int(allhit[~deg].sum())}/{(~deg).sum()}"}
        print(f"all three residues correct: degraders {allhit[deg].sum()}/{deg.sum()} ({allhit[deg].mean():.1%}) | hard negatives {allhit[~deg].sum()}/{(~deg).sum()} ({allhit[~deg].mean():.1%})")
        res["thresholds"][str(thr)] = out
    (base / "hits_trainval.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
