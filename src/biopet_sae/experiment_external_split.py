"""Measure real cross-split sequence identity on the Balci et al. dataset.

Experiment: adopt the external benchmark (Balci, Akguller, Erdogan 2026, MIT licensed,
1701 sequences) as the main dataset, form train/val/test, and measure what identity
actually exists between the splits.

Two split strategies are compared on identical data:

  A. THEIR `cluster30` column, grouped into 70/15/15. This is the published
     cluster-aware approach -- mmseqs clustering at 30%, then whole clusters assigned to
     splits.
  B. Our verified pipeline: global Needleman-Wunsch realignment of every candidate pair,
     union-find merge of any clusters joined above threshold, then assignment. The bound
     holds by construction.

The comparison matters because clustering at 30% bounds identity between a member and its
cluster *representative*, not between two different clusters. Strategy A should therefore
leak; the question is by how much.

Usage:
    uv run python -m biopet_sae.experiment_external_split
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from biopet_sae.make_splits import (
    SPLITS,
    UnionFind,
    assign_components,
    make_aligner,
    mmseqs_all_vs_all,
    require_mmseqs,
)

DEFAULT_MASTER = Path(
    "/Users/ramsindhu/Downloads/plastic-degrading-enzyme-notebooks/dataset_v1/master.csv"
)


def norm_seq(s: str) -> str:
    return re.sub(r"[^A-Z]", "", (s or "").upper())


def strat_label(row: dict[str, str]) -> str:
    """Coarse class used only to keep the splits balanced."""
    if row["source_class"] == "positive":
        return "pet" if "PET" in row["polymer_list_str"] else "other_positive"
    return row["neg_type"] or "negative"


def identities(a: str, b: str, aligner) -> tuple[float, float]:
    try:
        aln = aligner.align(a, b)[0]
    except (ValueError, IndexError):
        return 0.0, 0.0
    matches = 0
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        matches += sum(1 for x, y in zip(a[s1:e1], b[s2:e2]) if x == y)
    return matches / min(len(a), len(b)), matches / max(aln.length, 1)


def measure(
    assignment: dict[str, str],
    pairs: list[tuple[str, str, float, float]],
    labels: dict[str, str],
    threshold: float,
) -> dict[str, object]:
    """Summarise identity across every split boundary."""
    per_pair: dict[str, list[float]] = defaultdict(list)
    worst: dict[str, tuple[float, str, str]] = {}
    violations: list[dict[str, object]] = []
    for a, b, im, _ia in pairs:
        sa, sb = assignment.get(a), assignment.get(b)
        if sa is None or sb is None or sa == sb:
            continue
        key = "|".join(sorted((sa, sb)))
        per_pair[key].append(im)
        if key not in worst or im > worst[key][0]:
            worst[key] = (im, a, b)
        if im >= threshold:
            violations.append({
                "a": a, "b": b, "split_a": sa, "split_b": sb,
                "class_a": labels[a], "class_b": labels[b], "identity": round(im, 4),
            })
    summary = {}
    for key, vals in sorted(per_pair.items()):
        above = [v for v in vals if v >= threshold]
        summary[key] = {
            "n_candidate_pairs": len(vals),
            "max_identity": round(max(vals), 4),
            "median_identity": round(statistics.median(vals), 4),
            "pairs_at_or_above_threshold": len(above),
            "worst_pair": [worst[key][1], worst[key][2]],
        }
    return {"per_split_pair": summary, "violations": violations}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    ap.add_argument("--cache", type=Path,
                    default=Path("data/external/balci_pair_identities.tsv"))
    ap.add_argument("--out", type=Path, default=Path("results/external_split_experiment.json"))
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--candidate-floor", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)

    with args.master.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    records: dict[str, str] = {}
    labels: dict[str, str] = {}
    their_cluster: dict[str, str] = {}
    for r in rows:
        sid = r["sequence_md5"][:12]
        seq = norm_seq(r["sequence"])
        if not (20 <= len(seq) <= 2000):
            continue
        records[sid] = seq
        labels[sid] = strat_label(r)
        their_cluster[sid] = r["cluster30"]
    print(f"external dataset: {len(records)} sequences")
    print(f"  classes: {dict(sorted(((k, sum(1 for v in labels.values() if v == k)) for k in set(labels.values())), key=lambda kv: -kv[1]))}")
    print(f"  their cluster30: {len(set(their_cluster.values()))} clusters")

    # ---- cached pairwise identities --------------------------------------
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if args.cache.exists() and not args.recompute:
        print(f"loading cached alignments from {args.cache}")
        pairs = [
            (r["a"], r["b"], float(r["ident_min"]), float(r["ident_aln"]))
            for r in csv.DictReader(args.cache.open(newline=""), delimiter="\t")
            if r["a"] in records and r["b"] in records
        ]
    else:
        exe = require_mmseqs()
        workdir = Path(tempfile.mkdtemp(prefix="balci_"))
        print("mmseqs all-vs-all search ...")
        raw = mmseqs_all_vs_all(exe, records, workdir)
        seen: set[tuple[str, str]] = set()
        cands = []
        for a, b, local_id in raw:
            if a == b or local_id < args.candidate_floor:
                continue
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                cands.append(key)
        print(f"{len(raw)} hits -> {len(cands)} unique candidate pairs to realign globally")
        aligner = make_aligner()
        pairs = []
        for i, (a, b) in enumerate(cands):
            if i and i % 10000 == 0:
                print(f"  aligned {i}/{len(cands)}")
            im, ia = identities(records[a], records[b], aligner)
            pairs.append((a, b, im, ia))
        with args.cache.open("w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["a", "b", "ident_min", "ident_aln"])
            for a, b, im, ia in pairs:
                w.writerow([a, b, f"{im:.4f}", f"{ia:.4f}"])
        print(f"cached {len(pairs)} pair identities")
        shutil.rmtree(workdir, ignore_errors=True)

    report: dict[str, object] = {
        "n_sequences": len(records),
        "threshold": args.threshold,
        "n_candidate_pairs": len(pairs),
    }

    # ---- strategy A: their cluster30, grouped into 70/15/15 --------------
    print(f"\n{'='*72}\nSTRATEGY A -- their published cluster30, grouped 70/15/15\n{'='*72}")
    groups_a: dict[str, list[str]] = defaultdict(list)
    for sid, cl in their_cluster.items():
        groups_a[cl].append(sid)
    assign_a = assign_components(dict(groups_a), labels, args.seed)
    res_a = measure(assign_a, pairs, labels, args.threshold)
    report["strategy_a"] = res_a | {"n_groups": len(groups_a)}
    print(f"{'split pair':16s}{'cand pairs':>12s}{'max ident':>11s}{'median':>9s}{'>=30%':>8s}")
    for key, s in res_a["per_split_pair"].items():  # type: ignore[union-attr]
        print(f"{key:16s}{s['n_candidate_pairs']:>12d}{s['max_identity']:>11.4f}"
              f"{s['median_identity']:>9.4f}{s['pairs_at_or_above_threshold']:>8d}")
    print(f"\ntotal cross-split pairs >= {args.threshold:.0%} identity: "
          f"{len(res_a['violations'])}")  # type: ignore[arg-type]
    top = sorted(res_a["violations"], key=lambda v: -float(v["identity"]))[:8]  # type: ignore[arg-type]
    for v in top:
        print(f"  {v['identity']:.3f}  {v['split_a']:>5s}/{v['split_b']:<5s}  "
              f"{v['class_a']} vs {v['class_b']}")

    # ---- strategy B: our verified pipeline -------------------------------
    print(f"\n{'='*72}\nSTRATEGY B -- verified: union-find over global identity\n{'='*72}")
    uf = UnionFind()
    for sid in records:
        uf.find(their_cluster[sid])
    merged = 0
    for a, b, im, _ia in pairs:
        if im >= args.threshold and their_cluster[a] != their_cluster[b]:
            if uf.find(their_cluster[a]) != uf.find(their_cluster[b]):
                merged += 1
            uf.union(their_cluster[a], their_cluster[b])
    groups_b: dict[str, list[str]] = defaultdict(list)
    for sid in records:
        groups_b[uf.find(their_cluster[sid])].append(sid)
    sizes = sorted((len(v) for v in groups_b.values()), reverse=True)
    print(f"{len(groups_a)} clusters -> {len(groups_b)} components "
          f"({merged} merging joins)")
    print(f"largest components: {sizes[:8]}")
    assign_b = assign_components(dict(groups_b), labels, args.seed)
    res_b = measure(assign_b, pairs, labels, args.threshold)
    report["strategy_b"] = res_b | {
        "n_components": len(groups_b), "largest_component_sizes": sizes[:10],
    }
    print(f"\n{'split pair':16s}{'cand pairs':>12s}{'max ident':>11s}{'median':>9s}{'>=30%':>8s}")
    for key, s in res_b["per_split_pair"].items():  # type: ignore[union-attr]
        print(f"{key:16s}{s['n_candidate_pairs']:>12d}{s['max_identity']:>11.4f}"
              f"{s['median_identity']:>9.4f}{s['pairs_at_or_above_threshold']:>8d}")
    print(f"\ntotal cross-split pairs >= {args.threshold:.0%} identity: "
          f"{len(res_b['violations'])}")  # type: ignore[arg-type]

    # ---- what each strategy costs in test-set size -----------------------
    print(f"\n{'='*72}\nCOST: class composition per split\n{'='*72}")
    for name, assign in (("A (their clusters)", assign_a), ("B (verified)", assign_b)):
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for sid, sp in assign.items():
            counts[(sp, labels[sid])] += 1
        classes = sorted({labels[s] for s in records})
        print(f"\n{name}")
        print(f"  {'split':8s}" + "".join(f"{c[:14]:>16s}" for c in classes))
        for sp in SPLITS:
            print(f"  {sp:8s}" + "".join(f"{counts[(sp, c)]:>16d}" for c in classes))
        report[f"composition_{name[0]}"] = {
            f"{sp}|{c}": counts[(sp, c)] for sp in SPLITS for c in classes
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
