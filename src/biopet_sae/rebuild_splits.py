"""Rebuild train/val/test splits at an arbitrary identity threshold from the pair cache.

Reuses `data/processed/pair_identities.tsv` (70,230 cached global Needleman-Wunsch
alignments), so no mmseqs run or realignment is needed and any threshold is instant.

Union-find runs directly over *sequences* joined by identity >= threshold, which gives the
same connected components as clustering-then-merging but with one less step. The bound
holds by construction; it is still measured afterwards rather than assumed.

Also reports what the strict 0.30 split made impossible: the phylum composition per split,
and whether the Actinomycetota-only control (D6.2) has enough PET in val to run.

Usage:
    uv run python -m biopet_sae.rebuild_splits --threshold 0.40
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from biopet_sae.make_splits import (
    HELDOUT_CLASS,
    SPLITS,
    TRAINED_CLASSES,
    UnionFind,
    assign_components,
)


def load_pool(pos: Path, neg: Path) -> tuple[dict, dict, dict]:
    records, labels, meta = {}, {}, {}
    with pos.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["class_label"] == "excluded":
                continue
            records[r["seq_id"]] = r["sequence"]
            labels[r["seq_id"]] = r["class_label"]
            meta[r["seq_id"]] = {"phylum": r["phylum"], "organism": r["organism"],
                                 "protein_names": r["protein_names"],
                                 "substrates": r["substrates"], "source": r["sources"]}
    with neg.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            records[r["accession"]] = r["sequence"]
            labels[r["accession"]] = r["class_label"]
            meta[r["accession"]] = {"phylum": r["phylum"], "organism": r["organism"],
                                    "protein_names": r["protein_names"],
                                    "substrates": "", "source": "UniProt"}
    return records, labels, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--positives", type=Path,
                    default=Path("data/processed/plastizymes_merged.tsv"))
    ap.add_argument("--negatives", type=Path,
                    default=Path("data/processed/uniprot_negatives.tsv"))
    ap.add_argument("--cache", type=Path, default=Path("data/processed/pair_identities.tsv"))
    ap.add_argument("--near-duplicate", type=float, default=0.90)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: data/processed/dataset_splits_id<threshold>.tsv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    records, labels, meta = load_pool(args.positives, args.negatives)
    pairs = [
        (r["a"], r["b"], float(r["ident_min"]))
        for r in csv.DictReader(args.cache.open(newline=""), delimiter="\t")
    ]
    pairs = [(a, b, i) for a, b, i in pairs if a in records and b in records]
    print(f"pool {len(records)} sequences, {len(pairs)} cached pair alignments")

    # D2: drop negatives that are near-duplicates of a positive
    POS = {"1_pet", "2_other_polyester"}
    NEG = {"3_fold_matched_esterase", "4_naive_control"}
    dropped = set()
    for a, b, i in pairs:
        if i < args.near_duplicate:
            continue
        for x, y in ((a, b), (b, a)):
            if labels.get(x) in NEG and labels.get(y) in POS:
                dropped.add(x)
    for x in dropped:
        records.pop(x, None)
        labels.pop(x, None)
    print(f"dropped {len(dropped)} negatives >={args.near_duplicate:.0%} identical to a positive")

    # union-find over sequences joined at or above the threshold
    uf = UnionFind()
    for s in records:
        uf.find(s)
    for a, b, i in pairs:
        if i >= args.threshold and a in records and b in records:
            uf.union(a, b)
    comps: dict[str, list[str]] = defaultdict(list)
    for s in records:
        comps[uf.find(s)].append(s)
    sizes = sorted((len(v) for v in comps.values()), reverse=True)
    pet_ids = {s for s, c in labels.items() if c == "1_pet"}
    largest_pet = max((sum(1 for s in m if s in pet_ids) for m in comps.values()), default=0)
    print(f"\nthreshold {args.threshold:.2f}: {len(comps)} components "
          f"(largest {sizes[0]}, largest PET-count {largest_pet})")
    print(f"component sizes: {sizes[:10]}")

    trainable = {c: [s for s in m if labels[s] in TRAINED_CLASSES] for c, m in comps.items()}
    trainable = {c: m for c, m in trainable.items() if m}
    assign = assign_components(trainable, labels, args.seed)
    for c, m in comps.items():
        for s in m:
            if labels[s] == HELDOUT_CLASS:
                assign[s] = "heldout"

    # ---- composition ------------------------------------------------------
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for s, sp in assign.items():
        counts[(sp, labels[s])] += 1
    print(f"\n{'split':10s}" + "".join(f"{c.split('_')[0]:>10s}" for c in TRAINED_CLASSES) + f"{'total':>9s}")
    for sp in (*SPLITS, "heldout"):
        tot = sum(counts[(sp, c)] for c in (*TRAINED_CLASSES, HELDOUT_CLASS))
        print(f"{sp:10s}" + "".join(f"{counts[(sp, c)]:10d}" for c in TRAINED_CLASSES) + f"{tot:9d}")

    # ---- verify the bound -------------------------------------------------
    worst: dict[str, float] = {}
    for a, b, i in pairs:
        sa, sb = assign.get(a), assign.get(b)
        if sa is None or sb is None or sa == sb:
            continue
        k = "|".join(sorted((sa, sb)))
        worst[k] = max(worst.get(k, 0.0), i)
    print(f"\nmax cross-split global identity (bound {args.threshold:.2f}):")
    ok = True
    for k in sorted(worst):
        flag = ""
        if worst[k] >= args.threshold:
            flag = "  <-- VIOLATION"
            if "heldout" not in k:
                ok = False
        print(f"  {k:22s} {worst[k]:.4f}{flag}")
    print(f"  bound satisfied for trained splits: {ok}")

    # ---- phylum distribution, the reason the strict split was unusable ----
    print("\nphylum composition of the PET class per split:")
    for sp in SPLITS:
        ph = Counter()
        for s, a in assign.items():
            if a == sp and labels[s] == "1_pet":
                p = meta[s]["phylum"].split("|")[0] or "(none)"
                ph[p] += 1
        tot = sum(ph.values())
        top = ", ".join(f"{k} {v}" for k, v in ph.most_common(4))
        act = ph.get("Actinomycetota", 0)
        print(f"  {sp:6s} n={tot:4d}  Actinomycetota {act:3d} ({100*act/max(tot,1):4.1f}%)   {top}")

    # ---- is the Actinomycetota control runnable now? ----------------------
    print("\nActinomycetota-only subset (D6.2 control):")
    for sp in SPLITS:
        n = sum(1 for s, a in assign.items()
                if a == sp and "Actinomycetota" in meta[s]["phylum"]
                and labels[s] in TRAINED_CLASSES)
        npet = sum(1 for s, a in assign.items()
                   if a == sp and "Actinomycetota" in meta[s]["phylum"]
                   and labels[s] == "1_pet")
        print(f"  {sp:6s} n={n:4d}  PET={npet:3d}")

    # ---- write ------------------------------------------------------------
    out = args.out or Path(
        f"data/processed/dataset_splits_id{int(round(args.threshold*100)):02d}.tsv"
    )
    rows = []
    for s in sorted(records):
        rows.append({"seq_id": s, "class_label": labels[s], "split": assign[s],
                     "cluster": uf.find(s), "component": uf.find(s),
                     **meta[s], "seq_len": len(records[s]), "sequence": records[s]})
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    rep = {"threshold": args.threshold, "n_components": len(comps),
           "largest_component": sizes[0], "largest_pet_component": largest_pet,
           "dropped_near_duplicate_negatives": sorted(dropped),
           "composition": {f"{sp}|{c}": counts[(sp, c)]
                           for sp in (*SPLITS, "heldout")
                           for c in (*TRAINED_CLASSES, HELDOUT_CLASS) if counts[(sp, c)]},
           "max_cross_split_identity": {k: round(v, 4) for k, v in worst.items()},
           "bound_satisfied": ok}
    rp = out.with_name(out.stem + "_report.json")
    rp.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {out}\nwrote {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
