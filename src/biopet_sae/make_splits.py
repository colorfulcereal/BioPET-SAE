"""Cluster the pooled dataset and build train/val/test splits with a verified 30% bound.

Implements decisions D2, D3 and D4 in PROGRESS.md.

The original research plan ran `mmseqs easy-cluster --min-seq-id 0.3`, split the clusters,
and asserted that no test sequence shares >=30% identity with train. That does not follow:
clustering at 30% bounds identity between a member and its cluster *representative*, and
says nothing about two different clusters. This script gets the guarantee properly:

  1. mmseqs cluster at 90% -> drop class-3/4 negatives that are near-duplicates of a
     positive (D2). A negative >=90% identical to a known positive is probably a
     mislabelled positive.
  2. mmseqs cluster at 30% -> initial split units.
  3. mmseqs all-vs-all search (sensitive) -> candidate related pairs. mmseqs reports
     *local* identity, so it is used only as a recall-oriented filter.
  4. Global Needleman-Wunsch realignment of every candidate cross-cluster pair. Global,
     not local: the kinase project hit a real false alarm from using local alignment when
     re-deriving identity, and "30% identity" in a homology-split context means global.
  5. Union-find over clusters joined by any pair above the threshold. The resulting
     connected components are the final split units, so the bound holds *by
     construction* rather than by hope.
  6. Greedy stratified assignment of components to train/val/test, balancing each class
     separately.
  7. Independent verification pass that measures the achieved maximum cross-split
     identity and reports it as a number.

Usage:
    uv run python -m biopet_sae.make_splits [--threshold 0.30] [--seed 42]
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
from Bio.Align import substitution_matrices

SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
TRAINED_CLASSES = ("1_pet", "2_other_polyester", "3_fold_matched_esterase", "4_naive_control")
HELDOUT_CLASS = "heldout_pha"


# ---------------------------------------------------------------------------
# sequence identity
# ---------------------------------------------------------------------------

def make_aligner() -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    # do not penalise terminal gaps: one source stores precursors, the other mature
    # forms, and a signal peptide should not count against identity
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    return aligner


def global_identity(a: str, b: str, aligner: Align.PairwiseAligner) -> float:
    """Fraction of identical residues over the shorter sequence.

    Dividing by the shorter length rather than the alignment length is the
    conservative choice -- it yields the larger number, so the 30% bound is enforced
    more strictly.
    """
    try:
        aln = aligner.align(a, b)[0]
    except (ValueError, IndexError):
        return 0.0
    matches = 0
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        matches += sum(1 for x, y in zip(a[s1:e1], b[s2:e2]) if x == y)
    return matches / min(len(a), len(b))


# ---------------------------------------------------------------------------
# mmseqs wrappers
# ---------------------------------------------------------------------------

def require_mmseqs() -> str:
    exe = shutil.which("mmseqs")
    if not exe:
        sys.exit(
            "mmseqs not found. Install with `brew install mmseqs2` (arm64 build "
            "available), then re-run."
        )
    return exe


def write_fasta(path: Path, records: dict[str, str]) -> None:
    with path.open("w") as fh:
        for sid, seq in records.items():
            fh.write(f">{sid}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")


def mmseqs_cluster(
    exe: str, records: dict[str, str], min_seq_id: float, coverage: float, workdir: Path
) -> dict[str, str]:
    """Return {sequence_id: cluster_representative_id}."""
    fasta = workdir / f"in_{min_seq_id}.fasta"
    write_fasta(fasta, records)
    prefix = workdir / f"clu_{min_seq_id}"
    subprocess.run(
        [exe, "easy-cluster", str(fasta), str(prefix), str(workdir / f"tmp_{min_seq_id}"),
         "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0",
         "-v", "1"],
        check=True, capture_output=True, text=True,
    )
    assignment: dict[str, str] = {}
    with (prefix.parent / f"{prefix.name}_cluster.tsv").open() as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            assignment[member] = rep
    return assignment


def mmseqs_all_vs_all(
    exe: str, records: dict[str, str], workdir: Path, sensitivity: float = 7.5
) -> list[tuple[str, str, float]]:
    """Return candidate related pairs as (id_a, id_b, local_identity)."""
    fasta = workdir / "search.fasta"
    write_fasta(fasta, records)
    db = workdir / "db"
    res = workdir / "res"
    out = workdir / "hits.tsv"
    run = lambda cmd: subprocess.run(cmd, check=True, capture_output=True, text=True)
    run([exe, "createdb", str(fasta), str(db), "-v", "1"])
    run([exe, "search", str(db), str(db), str(res), str(workdir / "tmp_search"),
         "-s", str(sensitivity), "-e", "1000", "--max-seqs", "4000", "-v", "1"])
    run([exe, "convertalis", str(db), str(db), str(res), str(out),
         "--format-output", "query,target,fident,alnlen", "-v", "1"])
    pairs = []
    with out.open() as fh:
        for line in fh:
            q, t, fident, _ = line.rstrip("\n").split("\t")
            if q != t:
                pairs.append((q, t, float(fident)))
    return pairs


# ---------------------------------------------------------------------------
# union-find over clusters
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ---------------------------------------------------------------------------
# split assignment
# ---------------------------------------------------------------------------

def assign_components(
    components: dict[str, list[str]], labels: dict[str, str], seed: int
) -> dict[str, str]:
    """Greedily assign whole components to splits, balancing every class at once.

    Components are placed largest-first; each goes to whichever split is currently
    furthest below its target, summed as a deficit across the classes the component
    contains. Largest-first matters because one 40-member component can otherwise
    single-handedly skew a 15% test split.
    """
    class_totals: dict[str, int] = defaultdict(int)
    for sid in labels:
        class_totals[labels[sid]] += 1
    targets = {
        (split, cls): SPLIT_FRACTIONS[split] * total
        for split in SPLITS
        for cls, total in class_totals.items()
    }
    current: dict[tuple[str, str], int] = defaultdict(int)

    ordered = sorted(components.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    assignment: dict[str, str] = {}
    for comp_id, members in ordered:
        comp_counts: dict[str, int] = defaultdict(int)
        for sid in members:
            comp_counts[labels[sid]] += 1
        best_split, best_score = None, None
        for split in SPLITS:
            # deficit = how far below target this split still is for the classes in
            # this component; higher deficit means it needs the component more
            score = sum(
                (targets[(split, cls)] - current[(split, cls)]) / max(targets[(split, cls)], 1)
                * n
                for cls, n in comp_counts.items()
            )
            if best_score is None or score > best_score:
                best_split, best_score = split, score
        for sid in members:
            assignment[sid] = best_split  # type: ignore[assignment]
        for cls, n in comp_counts.items():
            current[(best_split, cls)] += n  # type: ignore[index]
    return assignment


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positives", type=Path,
                    default=Path("data/processed/plastizymes_merged.tsv"))
    ap.add_argument("--negatives", type=Path,
                    default=Path("data/processed/uniprot_negatives.tsv"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="max allowed global identity between splits")
    ap.add_argument("--near-duplicate", type=float, default=0.90,
                    help="negatives this similar to a positive are dropped (D2)")
    ap.add_argument("--candidate-floor", type=float, default=0.15,
                    help="mmseqs local identity below which a pair is not realigned")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workdir", type=Path, default=None)
    args = ap.parse_args(argv)
    exe = require_mmseqs()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- assemble the pooled set ------------------------------------------
    records: dict[str, str] = {}
    labels: dict[str, str] = {}
    meta: dict[str, dict[str, str]] = {}

    with args.positives.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["class_label"] == "excluded":
                continue
            sid = row["seq_id"]
            records[sid] = row["sequence"]
            labels[sid] = row["class_label"]
            meta[sid] = {"phylum": row["phylum"], "organism": row["organism"],
                         "name": row["protein_names"], "substrates": row["substrates"],
                         "source": row["sources"]}
    with args.negatives.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            sid = row["accession"]
            records[sid] = row["sequence"]
            labels[sid] = row["class_label"]
            meta[sid] = {"phylum": row["phylum"], "organism": row["organism"],
                         "name": row["protein_names"], "substrates": "",
                         "source": "UniProt"}

    print(f"pooled set: {len(records)} sequences")
    for cls in (*TRAINED_CLASSES, HELDOUT_CLASS):
        print(f"  {cls:28s} {sum(1 for v in labels.values() if v == cls)}")

    workdir = Path(tempfile.mkdtemp(prefix="biopet_splits_")) if args.workdir is None else args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"threshold": args.threshold, "seed": args.seed}
    aligner = make_aligner()

    # ---- step 1: drop negatives that are near-duplicates of a positive ----
    print(f"\n[1/6] clustering at {args.near_duplicate:.0%} to find mislabelled negatives")
    clu90 = mmseqs_cluster(exe, records, args.near_duplicate, 0.8, workdir)
    by_rep90: dict[str, list[str]] = defaultdict(list)
    for sid, rep in clu90.items():
        by_rep90[rep].append(sid)
    positive_classes = {"1_pet", "2_other_polyester"}
    dropped: list[dict[str, str]] = []
    for rep, members in by_rep90.items():
        kinds = {labels[m] for m in members}
        if kinds & positive_classes and kinds - positive_classes - {HELDOUT_CLASS}:
            for m in members:
                if labels[m] in ("3_fold_matched_esterase", "4_naive_control"):
                    pos = [x for x in members if labels[x] in positive_classes]
                    dropped.append({"id": m, "class": labels[m],
                                    "near_duplicate_of": ",".join(pos)})
    for d in dropped:
        records.pop(d["id"], None)
        labels.pop(d["id"], None)
    print(f"      dropped {len(dropped)} negatives >={args.near_duplicate:.0%} identical to a positive")
    for d in dropped[:10]:
        print(f"        {d['id']} ({d['class']}) ~ {d['near_duplicate_of']}")
    report["dropped_near_duplicate_negatives"] = dropped

    # ---- step 2: initial clustering at the split threshold ----------------
    print(f"\n[2/6] clustering at {args.threshold:.0%} for initial split units")
    clu = mmseqs_cluster(exe, records, args.threshold, 0.8, workdir)
    for sid in records:
        clu.setdefault(sid, sid)  # singletons mmseqs may omit
    n_clusters = len(set(clu.values()))
    print(f"      {len(records)} sequences -> {n_clusters} clusters")

    # ---- step 3: all-vs-all candidate pairs ------------------------------
    print("\n[3/6] mmseqs all-vs-all search for candidate related pairs")
    pairs = mmseqs_all_vs_all(exe, records, workdir)
    cross = [
        (a, b, ident) for a, b, ident in pairs
        if a in clu and b in clu and clu[a] != clu[b] and ident >= args.candidate_floor
    ]
    # deduplicate symmetric pairs
    seen: set[tuple[str, str]] = set()
    uniq_cross = []
    for a, b, ident in cross:
        key = (a, b) if a < b else (b, a)
        if key not in seen:
            seen.add(key)
            uniq_cross.append((a, b, ident))
    print(f"      {len(pairs)} hits -> {len(uniq_cross)} cross-cluster candidate pairs "
          f"above {args.candidate_floor:.0%} local identity")

    # ---- step 4: global realignment of candidates ------------------------
    print(f"\n[4/6] global (Needleman-Wunsch) realignment of {len(uniq_cross)} candidates")
    violations: list[dict[str, object]] = []
    for i, (a, b, local_id) in enumerate(uniq_cross):
        if i and i % 2000 == 0:
            print(f"      {i}/{len(uniq_cross)}")
        gid = global_identity(records[a], records[b], aligner)
        if gid >= args.threshold:
            violations.append({"a": a, "b": b, "class_a": labels[a], "class_b": labels[b],
                               "local_identity": round(local_id, 3),
                               "global_identity": round(gid, 3)})
    print(f"      {len(violations)} cross-cluster pairs exceed {args.threshold:.0%} global identity")
    report["cross_cluster_violations_before_merge"] = len(violations)

    # ---- step 5: union-find merge into final split units -----------------
    print("\n[5/6] merging violating clusters into connected components")
    uf = UnionFind()
    for sid in records:
        uf.find(clu[sid])
    for v in violations:
        uf.union(clu[str(v["a"])], clu[str(v["b"])])
    components: dict[str, list[str]] = defaultdict(list)
    for sid in records:
        components[uf.find(clu[sid])].append(sid)
    print(f"      {n_clusters} clusters -> {len(components)} components")
    sizes = sorted((len(v) for v in components.values()), reverse=True)
    print(f"      largest components: {sizes[:8]}")
    report["n_clusters_at_threshold"] = n_clusters
    report["n_components"] = len(components)
    report["largest_component_sizes"] = sizes[:10]

    # mixed-class components are the most informative test cases
    mixed = {
        cid: sorted({labels[s] for s in members})
        for cid, members in components.items()
        if len({labels[s] for s in members}) > 1
    }
    print(f"      {len(mixed)} components contain more than one class")
    report["n_mixed_class_components"] = len(mixed)

    # ---- step 6: assign, then verify -------------------------------------
    print("\n[6/6] assigning components to splits, then verifying the bound")
    trainable = {
        cid: [s for s in members if labels[s] in TRAINED_CLASSES]
        for cid, members in components.items()
    }
    trainable = {cid: m for cid, m in trainable.items() if m}
    assignment = assign_components(trainable, labels, args.seed)
    # held-out PHA never enters a split; its component may still touch trained data,
    # which is reported rather than silently allowed
    for cid, members in components.items():
        for s in members:
            if labels[s] == HELDOUT_CLASS:
                assignment[s] = "heldout"

    rows = []
    for sid in sorted(records):
        rows.append({
            "seq_id": sid, "class_label": labels[sid], "split": assignment[sid],
            "cluster": clu[sid], "component": uf.find(clu[sid]),
            "phylum": meta[sid]["phylum"], "organism": meta[sid]["organism"],
            "protein_names": meta[sid]["name"], "substrates": meta[sid]["substrates"],
            "source": meta[sid]["source"], "seq_len": len(records[sid]),
            "sequence": records[sid],
        })
    split_path = args.out / "dataset_splits.tsv"
    with split_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    # composition table
    comp_counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        comp_counts[(r["split"], r["class_label"])] += 1
    print("\nsplit composition:")
    header = f"  {'split':10s}" + "".join(f"{c.split('_')[0]:>10s}" for c in TRAINED_CLASSES) + f"{'total':>10s}"
    print(header)
    for split in (*SPLITS, "heldout"):
        cells = "".join(f"{comp_counts[(split, c)]:10d}" for c in TRAINED_CLASSES)
        total = sum(comp_counts[(split, c)] for c in (*TRAINED_CLASSES, HELDOUT_CLASS))
        print(f"  {split:10s}{cells}{total:10d}")
    report["split_composition"] = {
        f"{s}|{c}": comp_counts[(s, c)]
        for s in (*SPLITS, "heldout") for c in (*TRAINED_CLASSES, HELDOUT_CLASS)
        if comp_counts[(s, c)]
    }

    # independent verification: measure achieved max cross-split identity
    print("\nverifying: realigning every candidate pair that crosses a split boundary")
    worst: dict[str, dict[str, object]] = {}
    checked = 0
    for a, b, local_id in uniq_cross:
        sa, sb = assignment.get(a), assignment.get(b)
        if not sa or not sb or sa == sb:
            continue
        checked += 1
        gid = global_identity(records[a], records[b], aligner)
        key = "|".join(sorted((str(sa), str(sb))))
        if key not in worst or gid > float(worst[key]["global_identity"]):  # type: ignore[index]
            worst[key] = {"a": a, "b": b, "class_a": labels[a], "class_b": labels[b],
                          "global_identity": round(gid, 4)}
    print(f"  checked {checked} cross-split candidate pairs")
    print(f"  {'split pair':22s} {'max global identity':>20s}")
    ok = True
    for key in sorted(worst):
        gid = float(worst[key]["global_identity"])  # type: ignore[index]
        flag = "" if gid < args.threshold else "  <-- VIOLATION"
        if gid >= args.threshold and "heldout" not in key:
            ok = False
        print(f"  {key:22s} {gid:>20.4f}{flag}")
    report["max_cross_split_identity"] = worst
    report["bound_satisfied"] = ok
    print(f"\n  bound <{args.threshold:.0%} satisfied for trained splits: {ok}")

    (args.out / "splits_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {split_path}")
    print(f"wrote {args.out / 'splits_report.json'}")
    if args.workdir is None:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
