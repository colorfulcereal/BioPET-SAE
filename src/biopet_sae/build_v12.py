"""Build release v1.2: v1.1 + new hard negatives, re-split by homology component.

* Positives (404) and the other negatives are unchanged (sequences, labels, sources).
* New hard negatives come from `select_hard_negatives_v12.py`.
* Components: union-find over every pair with global identity >= 0.40 (cached pair identities
  from `select_hard_negatives_v12.py`, which realigns every mmseqs pair with local identity
  >= 0.30). Component id = smallest member id; the v1.1 id is kept in `component_v1_1`.
* Split: whole components to train / val / test by seeded random-restart greedy assignment,
  minimising the relative deviation of every (split, group) count from its target.
  Positives cannot hit an even 60/20/20 because one component holds 64% of them; targets are set
  to what is feasible and the ACHIEVED counts are reported.
* The bound is measured afterwards from the pair table, not assumed.

Usage:  uv run python -m biopet_sae.build_v12 --hard-fracs 0.5,0.25,0.25
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from biopet_sae.make_splits import UnionFind

SPLITS = ("train", "val", "test")
ELBOW = re.compile(r"G.[ST].G")
GROUPS = ["pos_pet", "pos_other", "hard", "wrongfold", "trna", "naive"]
GMAP = {"ab_hydrolase_hard_negative": "hard", "ec311_wrong_fold": "wrongfold",
        "tRNA_processing_no_ser_nucleophile": "trna", "naive_control": "naive"}


BACT = {"Actinomycetota", "Pseudomonadota", "Bacillota", "Bacteroidota", "Cyanobacteriota", "Thermotogota",
        "Deinococcota", "Myxococcota", "Chloroflexota", "Planctomycetota", "Acidobacteriota", "Campylobacterota",
        "Verrucomicrobiota", "Spirochaetota", "Fusobacteriota", "Thermodesulfobacteriota", "Bdellovibrionota"}


def domain_of(r: dict) -> str:
    """Bacteria / Eukaryota / Archaea from the stored lineage field when present, else from the
    phylum name; blank phylum -> unknown (74 positives have no phylum in the source databases)."""
    if r.get("domain"): return r["domain"]
    ph = r["phylum"]
    if not ph: return "unknown"
    if ph in BACT: return "Bacteria"
    if ph.endswith("mycota") or ph in ("Chordata", "Arthropoda", "Streptophyta", "Nematoda", "Apicomplexa"): return "Eukaryota"
    return "other"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path("data/release/biopet_sae_dataset_v1_1.tsv"))
    ap.add_argument("--new", type=Path, default=Path("data/raw/hard_negatives_v12/hard_negatives_new.tsv"))
    ap.add_argument("--pairs", type=Path, default=Path("data/raw/hard_negatives_v12/pair_identities.tsv"))
    ap.add_argument("--out", type=Path, default=Path("data/release"))
    ap.add_argument("--pet-fracs", default="0.79,0.105,0.105")
    ap.add_argument("--other-fracs", default="0.6,0.2,0.2")
    ap.add_argument("--hard-fracs", default="0.5,0.25,0.25")
    ap.add_argument("--easy-fracs", default="0.7,0.15,0.15")
    ap.add_argument("--restarts", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    old = list(csv.DictReader(a.release.open(newline=""), delimiter="\t"))
    new = list(csv.DictReader(a.new.open(newline=""), delimiter="\t"))
    cols = list(old[0])
    rows = []
    for r in old:
        r = dict(r); r["split_v1_1"] = r["split"]; r["component_v1_1"] = r["component"]
        rows.append(r)
    for r in new:
        rows.append({"seq_id": r["seq_id"], "split": "", "y": "0",
                     "class_label": "3_fold_matched_esterase",
                     "negative_group": "ab_hydrolase_hard_negative", "ab_hydrolase_fold": "1",
                     "gxsxg_elbow": str(int(bool(ELBOW.search(r["sequence"])))),
                     "n_curated_active_sites": str(len([x for x in r["active_sites"].split(";") if x])),
                     "ec": r["ec"], "pfam": r["pfam"], "activity_verified": "", "extrapolated_only": "",
                     "substrates": "", "source": "UniProt", "organism": r["organism"],
                     "phylum": r["phylum"], "component": "", "cluster": "", "seq_len": r["seq_len"],
                     "sequence": r["sequence"], "batch": "v1.2", "split_v1_1": "", "component_v1_1": "", "domain": r["domain"]})
    ids = [r["seq_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate ids"
    grp = {r["seq_id"]: (("pos_pet" if r["class_label"] == "1_pet" else "pos_other") if r["y"] == "1" else GMAP[r["negative_group"]]) for r in rows}
    pairs = []
    for r in csv.DictReader(a.pairs.open(newline=""), delimiter="\t"):
        if r["a"] in grp and r["b"] in grp:
            pairs.append((r["a"], r["b"], float(r["identity"])))

    uf = UnionFind()
    for i in ids: uf.find(i)
    for x, y, v in pairs:
        if v >= 0.40: uf.union(x, y)
    members = defaultdict(list)
    for i in ids: members[uf.find(i)].append(i)
    cid = {c: min(m) for c, m in members.items()}
    comp = {i: cid[uf.find(i)] for i in ids}
    comps = {cid[c]: m for c, m in members.items()}
    # consistency with v1.1: old rows that were together must still be together
    oldmap = {r["seq_id"]: r["component_v1_1"] for r in rows if r["component_v1_1"]}
    merged = sum(1 for c, m in comps.items() if len({oldmap[i] for i in m if i in oldmap}) > 1)

    fr = {"pos_pet": [float(x) for x in a.pet_fracs.split(",")],
          "pos_other": [float(x) for x in a.other_fracs.split(",")],
          "hard": [float(x) for x in a.hard_fracs.split(",")]}
    for g in ("wrongfold", "trna", "naive"): fr[g] = [float(x) for x in a.easy_fracs.split(",")]
    tot = Counter(grp[i] for i in ids)
    target = {(s, g): fr[g][k] * tot[g] for g in GROUPS for k, s in enumerate(SPLITS)}
    cc = {c: Counter(grp[i] for i in m) for c, m in comps.items()}
    w = {g: 1.0 / max(tot[g], 1) for g in GROUPS}
    rng = random.Random(a.seed)
    best, best_score = None, 1e18
    keys = sorted(comps, key=lambda c: (-len(comps[c]), c))
    for it in range(a.restarts):
        order = sorted(keys, key=lambda c: -len(comps[c]) * (0.6 + 0.8 * rng.random()))
        cur = Counter(); asg = {}
        for c in order:
            bs, bv = None, -1e18
            for s in SPLITS:
                # gain = reduction in squared relative deficit if c goes to s
                v = sum(((target[(s, g)] - cur[(s, g)]) ** 2 - (target[(s, g)] - cur[(s, g)] - n) ** 2) * w[g] ** 2
                        for g, n in cc[c].items())
                v += rng.random() * 1e-9
                if v > bv: bs, bv = s, v
            asg[c] = bs
            for g, n in cc[c].items(): cur[(bs, g)] += n
        score = sum(abs(cur[(s, g)] - target[(s, g)]) * w[g] for s in SPLITS for g in GROUPS)
        if score < best_score: best, best_score = asg, score
    for r in rows:
        r["split"] = best[comp[r["seq_id"]]]
        r["component"] = r["cluster"] = comp[r["seq_id"]]
    # ---- bound, measured from the pair table
    sp = {r["seq_id"]: r["split"] for r in rows}
    worst = defaultdict(float)
    for x, y, v in pairs:
        if sp[x] != sp[y]:
            k = "|".join(sorted((sp[x], sp[y]))); worst[k] = max(worst[k], v)
    assert all(v < 0.40 for v in worst.values()), dict(worst)
    # ---- write
    for r in rows:
        r["domain"] = domain_of(r)
    out_cols = cols + ["split_v1_1", "component_v1_1", "domain"]
    order_ = {"train": 0, "val": 1, "test": 2}
    rows.sort(key=lambda r: (order_[r["split"]], r["seq_id"]))
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "biopet_sae_dataset_v1_2.tsv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=out_cols, delimiter="\t"); wr.writeheader(); wr.writerows(rows)
    with (a.out / "biopet_sae_dataset_v1_2.fasta").open("w") as fh:
        for r in rows:
            fh.write(f">{r['seq_id']} split={r['split']} class={r['class_label']} group={r['negative_group'] or 'positive'}\n")
            for k in range(0, len(r["sequence"]), 60): fh.write(r["sequence"][k:k + 60] + "\n")
    ach = {s: {g: sum(1 for r in rows if r["split"] == s and grp[r["seq_id"]] == g) for g in GROUPS} for s in SPLITS}
    rep = {"seed": a.seed, "restarts": a.restarts, "fractions": fr, "totals": dict(tot),
           "targets": {f"{s}|{g}": round(target[(s, g)], 1) for s, g in target},
           "achieved": ach, "score": best_score,
           "n_components": len(comps), "components_that_bridge_v1_1_components": merged,
           "max_cross_split_identity": {k: round(v, 4) for k, v in worst.items()},
           "bound_below_0.40": True, "rows": len(rows)}
    (a.out / "biopet_sae_dataset_v1_2_report.json").write_text(json.dumps(rep, indent=2))
    print(f"rows {len(rows)}  components {len(comps)}  bridging old components: {merged}")
    print(f"{'split':6s}" + "".join(f"{g:>10s}" for g in GROUPS) + f"{'total':>8s}")
    for s in SPLITS:
        print(f"{s:6s}" + "".join(f"{ach[s][g]:10d}" for g in GROUPS) + f"{sum(ach[s].values()):8d}")
    print("max cross-split identity:", rep["max_cross_split_identity"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
