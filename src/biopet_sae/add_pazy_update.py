"""Add newly published PAZy PET enzymes to the release dataset, as v1.1.

v1 stays frozen (every reported result is tied to its 104-row test split). v1.1 is v1
plus the PAZy entries that appeared after the 2026-09-27 export, placed WITHOUT touching
any existing assignment:

* each new sequence is aligned (same global-identity routine as the original splits) to
  every v1 sequence; any v1 sequence at >= threshold identity puts the new sequence in
  that sequence's component and split;
* a new sequence linked to components in two different splits cannot be placed without
  breaking the bound, so the script stops (none occurred);
* an unlinked new sequence goes to train (val/test are not enlarged without cause);
* negatives >= 0.90 identical to a new positive are dropped (the original D2 rule: such a
  negative is a label conflict, not a negative).

Usage:  uv run python -m biopet_sae.add_pazy_update
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from biopet_sae.make_splits import UnionFind, global_identity, make_aligner

ELBOW = re.compile(r"G.[ST].G")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path("data/release/biopet_sae_dataset_v1.tsv"))
    ap.add_argument("--new", type=Path,
                    default=Path("data/raw/pazy_update/pazy_new_pet_2026-11-23.json"))
    ap.add_argument("--out", type=Path, default=Path("data/release"))
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--near-duplicate", type=float, default=0.90)
    ap.add_argument("--first-id", type=int, default=543, help="first free BPS number")
    args = ap.parse_args(argv)

    with args.release.open(newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        cols = list(rd.fieldnames)
        rows = list(rd)
    new = json.loads(args.new.read_text())
    al = make_aligner()

    # ---- identity of every new sequence to every v1 sequence and to each other ----
    ids = {}
    for k, o in enumerate(sorted(new, key=lambda o: o["id"])):
        ids[o["id"]] = f"BPS{args.first_id + k:04d}"
    seqs = {ids[o["id"]]: o["amino_acid_sequence"] for o in new}
    ident = {n: {r["seq_id"]: global_identity(s, r["sequence"], al) for r in rows}
             for n, s in seqs.items()}
    names = list(seqs)
    inter = {(a, b): global_identity(seqs[a], seqs[b], al)
             for i, a in enumerate(names) for b in names[i + 1:]}

    # ---- label conflicts: negatives that are really the same protein ----
    by_id = {r["seq_id"]: r for r in rows}
    dropped = {}
    for n in names:
        for sid, v in ident[n].items():
            if by_id[sid]["y"] == "0" and v >= args.near_duplicate:
                dropped[sid] = {"identical_to_new": n, "identity": round(v, 4),
                                "negative_group": by_id[sid]["negative_group"],
                                "split": by_id[sid]["split"]}

    # ---- placement: union over new + linked v1 positives AND negatives ----
    uf = UnionFind()
    for n in names:
        uf.find(n)
    for (a, b), v in inter.items():
        if v >= args.threshold:
            uf.union(a, b)
    linked = defaultdict(set)          # new seq -> {(v1 component, split)}
    for n in names:
        for sid, v in ident[n].items():
            if sid in dropped:
                continue
            if v >= args.threshold:
                linked[n].add((by_id[sid]["component"], by_id[sid]["split"]))
    groups = defaultdict(list)
    for n in names:
        groups[uf.find(n)].append(n)
    place = {}
    for g, members in groups.items():
        comps = {c for n in members for c, _ in linked[n]}
        splits = {s for n in members for _, s in linked[n]}
        if len(splits) > 1:
            raise SystemExit(f"CONFLICT: new group {members} links splits {splits}")
        split = splits.pop() if splits else "train"
        comp = sorted(comps)[0] if comps else sorted(members)[0]
        for n in members:
            place[n] = (split, comp)
        if len(comps) > 1:
            print(f"NOTE: {members} bridge v1 components {sorted(comps)}; relabelling")
    relabel = {}
    for g, members in groups.items():
        comps = {c for n in members for c, _ in linked[n]}
        for c in comps:
            relabel[c] = place[members[0]][1]

    # ---- bound check: new vs everything in a different split ----
    worst = defaultdict(float)
    for n in names:
        sp = place[n][0]
        for sid, v in ident[n].items():
            if sid in dropped or by_id[sid]["split"] == sp:
                continue
            worst[f"{sp}|{by_id[sid]['split']}"] = max(worst[f"{sp}|{by_id[sid]['split']}"], v)
    bad = {k: round(v, 4) for k, v in worst.items() if v >= args.threshold}
    assert not bad, f"bound violated: {bad}"

    # ---- write v1.1 ----
    out_cols = cols + ["batch"]
    out_rows = []
    for r in rows:
        if r["seq_id"] in dropped:
            continue
        r = dict(r)
        r["component"] = r["cluster"] = relabel.get(r["component"], r["component"])
        r["batch"] = "v1"
        out_rows.append(r)
    for o in sorted(new, key=lambda o: o["id"]):
        n = ids[o["id"]]
        subs = [s["abbreviation"] or s["name"] for s in o["substrates"]]
        assert "PET" in subs and o["verified"], o["name"]
        org = o["organism"]
        out_rows.append({
            "seq_id": n, "split": place[n][0], "y": "1", "class_label": "1_pet",
            "negative_group": "", "ab_hydrolase_fold": "",
            "gxsxg_elbow": str(int(bool(ELBOW.search(o["amino_acid_sequence"])))),
            "n_curated_active_sites": "", "ec": "", "pfam": "",
            "activity_verified": "1", "extrapolated_only": "0",
            "substrates": "|".join(sorted(subs)), "source": "PAZy",
            "organism": org["scientific_name"], "phylum": org["phylum"] or "",
            "component": place[n][1], "cluster": place[n][1],
            "seq_len": str(len(o["amino_acid_sequence"])),
            "sequence": o["amino_acid_sequence"], "batch": "v1.1"})
    tsv = args.out / "biopet_sae_dataset_v1_1.tsv"
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    with (args.out / "biopet_sae_dataset_v1_1.fasta").open("w") as fh:
        for r in out_rows:
            g = r["negative_group"] or r["class_label"]
            fh.write(f">{r['seq_id']} split={r['split']} class={r['class_label']} group={g}\n")
            for k in range(0, len(r["sequence"]), 60):
                fh.write(r["sequence"][k:k + 60] + "\n")

    rep = {
        "threshold": args.threshold,
        "added": {ids[o["id"]]: {"pazy_id": o["id"], "name": o["name"],
                                 "split": place[ids[o["id"]]][0],
                                 "component": place[ids[o["id"]]][1],
                                 "max_identity_v1": round(max(v for s, v in ident[ids[o["id"]]].items() if s not in dropped), 4),
                                 "dois": [l["doi"] for l in o["literature"]]}
                  for o in new},
        "dropped_negatives": dropped,
        "relabelled_components": relabel,
        "max_cross_split_identity_new_vs_v1": {k: round(v, 4) for k, v in worst.items()},
        "bound_satisfied": True,
        "rows": {"v1": len(rows), "v1_1": len(out_rows)},
    }
    (args.out / "biopet_sae_dataset_v1_1_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("dropped_negatives", "relabelled_components",
                                          "max_cross_split_identity_new_vs_v1", "rows")}, indent=1))
    for n in names:
        print(n, [o["name"] for o in new if ids[o["id"]] == n][0], *place[n])
    return 0


if __name__ == "__main__":
    sys.exit(main())
