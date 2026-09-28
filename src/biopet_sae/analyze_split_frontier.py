"""Measure the achievable homology-split frontier for this dataset.

The first split run showed that enforcing "no cross-split pair >=30% global identity"
via single-linkage merging collapses the PET class into one 289-member component, leaving
only ~15 test positives. That is a property of the data, not a bug: known PET hydrolases
are largely one cutinase-like superfamily, so they form a connected homology network at
30%.

Rather than quietly pick a threshold that happens to look good, this script measures the
whole trade-off. For each candidate threshold and each of two standard identity
definitions it reports the largest connected component and the resulting test-set size.

Two identity definitions, because the choice materially changes the answer:
  ident_min : matches / len(shorter sequence)      -- CD-HIT style, conservative/larger
  ident_aln : matches / alignment length           -- stricter, penalises partial overlap

Pair alignments are cached to data/processed/pair_identities.tsv so the sweep is cheap to
re-run.

Usage:
    uv run python -m biopet_sae.analyze_split_frontier [--recompute]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from Bio import Align

from biopet_sae.make_splits import (
    HELDOUT_CLASS,
    SPLIT_FRACTIONS,
    TRAINED_CLASSES,
    UnionFind,
    assign_components,
    make_aligner,
    mmseqs_all_vs_all,
    mmseqs_cluster,
    require_mmseqs,
)

THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def identities(a: str, b: str, aligner: Align.PairwiseAligner) -> tuple[float, float]:
    """Return (matches/min_len, matches/alignment_len)."""
    try:
        aln = aligner.align(a, b)[0]
    except (ValueError, IndexError):
        return 0.0, 0.0
    matches = 0
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        matches += sum(1 for x, y in zip(a[s1:e1], b[s2:e2]) if x == y)
    return matches / min(len(a), len(b)), matches / max(aln.length, 1)


def load_pool(pos_path: Path, neg_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    records: dict[str, str] = {}
    labels: dict[str, str] = {}
    with pos_path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["class_label"] == "excluded":
                continue
            records[row["seq_id"]] = row["sequence"]
            labels[row["seq_id"]] = row["class_label"]
    with neg_path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            records[row["accession"]] = row["sequence"]
            labels[row["accession"]] = row["class_label"]
    return records, labels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positives", type=Path,
                    default=Path("data/processed/plastizymes_merged.tsv"))
    ap.add_argument("--negatives", type=Path,
                    default=Path("data/processed/uniprot_negatives.tsv"))
    ap.add_argument("--cache", type=Path,
                    default=Path("data/processed/pair_identities.tsv"))
    ap.add_argument("--out", type=Path, default=Path("results/split_frontier.json"))
    ap.add_argument("--base-cluster", type=float, default=0.25,
                    help="mmseqs pre-clustering level; only affects speed, not the result")
    ap.add_argument("--candidate-floor", type=float, default=0.12)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)

    records, labels = load_pool(args.positives, args.negatives)
    print(f"pooled set: {len(records)} sequences")

    # ---- cache of pairwise identities -------------------------------------
    if args.cache.exists() and not args.recompute:
        print(f"loading cached alignments from {args.cache}")
        pair_rows = list(csv.DictReader(args.cache.open(newline=""), delimiter="\t"))
        pairs = [(r["a"], r["b"], float(r["ident_min"]), float(r["ident_aln"]))
                 for r in pair_rows if r["a"] in records and r["b"] in records]
    else:
        exe = require_mmseqs()
        workdir = Path(tempfile.mkdtemp(prefix="biopet_frontier_"))
        print("running mmseqs all-vs-all to find candidate pairs")
        raw = mmseqs_all_vs_all(exe, records, workdir)
        seen: set[tuple[str, str]] = set()
        cands = []
        for a, b, local_id in raw:
            if a == b or local_id < args.candidate_floor:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            cands.append(key)
        print(f"{len(raw)} hits -> {len(cands)} unique candidate pairs to realign")
        aligner = make_aligner()
        pairs = []
        for i, (a, b) in enumerate(cands):
            if i and i % 5000 == 0:
                print(f"  {i}/{len(cands)}")
            im, ia = identities(records[a], records[b], aligner)
            pairs.append((a, b, im, ia))
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with args.cache.open("w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["a", "b", "ident_min", "ident_aln"])
            for a, b, im, ia in pairs:
                w.writerow([a, b, f"{im:.4f}", f"{ia:.4f}"])
        print(f"cached {len(pairs)} pair identities to {args.cache}")
        shutil.rmtree(workdir, ignore_errors=True)

    # ---- the sweep --------------------------------------------------------
    pet_ids = {s for s, c in labels.items() if c == "1_pet"}
    trained = {s for s, c in labels.items() if c in TRAINED_CLASSES}
    results = []
    print(f"\n{'defn':10s}{'thresh':>8s}{'components':>12s}{'largest':>9s}"
          f"{'largest(PET)':>14s}{'PET test':>10s}{'PET val':>9s}")
    for defn, idx in (("ident_min", 2), ("ident_aln", 3)):
        for t in THRESHOLDS:
            uf = UnionFind()
            for s in records:
                uf.find(s)
            for p in pairs:
                if p[idx] >= t:
                    uf.union(str(p[0]), str(p[1]))
            comps: dict[str, list[str]] = defaultdict(list)
            for s in records:
                comps[uf.find(s)].append(s)
            trainable = {c: [s for s in m if s in trained] for c, m in comps.items()}
            trainable = {c: m for c, m in trainable.items() if m}
            assignment = assign_components(
                trainable, {s: labels[s] for s in trained}, seed=42
            )
            pet_per_split: dict[str, int] = defaultdict(int)
            for s in pet_ids:
                if s in assignment:
                    pet_per_split[assignment[s]] += 1
            largest = max(len(m) for m in comps.values())
            largest_pet = max(
                (sum(1 for s in m if s in pet_ids) for m in comps.values()), default=0
            )
            row = {
                "definition": defn, "threshold": t, "n_components": len(comps),
                "largest_component": largest, "largest_component_pet": largest_pet,
                "pet_train": pet_per_split["train"], "pet_val": pet_per_split["val"],
                "pet_test": pet_per_split["test"],
            }
            results.append(row)
            print(f"{defn:10s}{t:>8.2f}{len(comps):>12d}{largest:>9d}"
                  f"{largest_pet:>14d}{pet_per_split['test']:>10d}{pet_per_split['val']:>9d}")

    # ---- held-out PHA contamination --------------------------------------
    print("\nheld-out PHA probe set: identity to the trained pool")
    pha = {s for s, c in labels.items() if c == HELDOUT_CLASS}
    worst: dict[str, float] = {s: 0.0 for s in pha}
    for a, b, im, ia in pairs:
        if a in pha and b in trained:
            worst[a] = max(worst[a], im)
        elif b in pha and a in trained:
            worst[b] = max(worst[b], im)
    for bound in (0.30, 0.40, 0.50, 0.70):
        keep = sum(1 for v in worst.values() if v < bound)
        print(f"  PHA sequences with <{bound:.0%} identity to any trained sequence: "
              f"{keep}/{len(pha)}")
    contaminated = sorted(((v, s) for s, v in worst.items()), reverse=True)[:8]
    print(f"  most contaminated: {[(s, round(v, 3)) for v, s in contaminated]}")

    payload = {
        "frontier": results,
        "pha_max_identity_to_trained": {s: round(v, 4) for s, v in worst.items()},
        "pha_survivors": {
            f"<{b:.0%}": sum(1 for v in worst.values() if v < b)
            for b in (0.30, 0.40, 0.50, 0.70)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
