"""Build a 30,000-family screening set from the cached MGnify plastisphere assemblies.

Stratified equally across the three studies so no single assembly dominates. One sequence
per 90%-identity family, nothing resembling a known plastizyme.

Usage:
    uv run python -m biopet_sae.build_screen30k --per-study 10000
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ELBOW = re.compile(r"G.[ST].G")
PLASTIZYME_ELBOW = re.compile(r"G[HWYFDENQ]S[MQA]G")
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
ANALYSIS_TO_STUDY = {
    "MGYA00572563": "MGYS00005625", "MGYA00572564": "MGYS00005625",
    "MGYA00693733": "MGYS00006544", "MGYA00693734": "MGYS00006544",
    "MGYA00594112": "MGYS00005970", "MGYA00594113": "MGYS00005970",
}
STUDY_LABEL = {
    "MGYS00005625": "Engineered — plastic metagenome (PRJEB15404)",
    "MGYS00006544": "Marine — genomic signatures (PRJNA777294)",
    "MGYS00005970": "Lab enrichment — PE film, in situ marine",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-study", type=int, default=10000)
    ap.add_argument("--raw", type=Path, default=Path("data/mgnify/raw"))
    ap.add_argument("--merged", type=Path,
                    default=Path("data/processed/plastizymes_merged.tsv"))
    ap.add_argument("--out", type=Path, default=Path("data/mgnify/screen30k_set.tsv"))
    ap.add_argument("--max-identity", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    funnel: dict[str, int] = {}

    # ---- 1-4: parse, filter, dedup, elbow ------------------------------------------
    total = 0
    kept: dict[str, tuple[str, str, str]] = {}
    seen: set[str] = set()
    n_len = n_dup = 0
    for f in sorted(args.raw.glob("*.faa.gz")):
        acc = f.name.split("_")[0]
        study = ANALYSIS_TO_STUDY[acc]
        txt = gzip.decompress(f.read_bytes()).decode()
        name, buf = None, []
        def flush():
            nonlocal n_len, n_dup
            if not name:
                return
            s = "".join(buf)
            if not (50 <= len(s) <= 1000) or set(s) - STANDARD:
                return
            n_len_local = True
            h = hashlib.md5(s.encode()).hexdigest()
            if h in seen:
                return
            seen.add(h)
            if not ELBOW.search(s):
                return
            kept[f"{acc}|{name}"] = (s, study, acc)
        for ln in txt.split("\n"):
            if ln.startswith(">"):
                flush(); total += 1
                name = ln[1:].split(" #")[0]; buf = []
            elif name:
                buf.append(ln.strip())
        flush()
        print(f"  {acc} [{study}] parsed; running total kept {len(kept):,}")
    funnel["1_predicted_cds"] = total
    funnel["4_elbow_unique_in_range"] = len(kept)

    # ---- 5: drop anything resembling a known plastizyme; 6: family-dedup ------------
    known = {r["seq_id"]: r["sequence"] for r in
             csv.DictReader(args.merged.open(newline=""), delimiter="\t")
             if r["sequence"]}
    print(f"\nclustering {len(kept):,} candidates with {len(known)} known plastizymes "
          f"at {args.max_identity:.0%}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with (td / "all.fa").open("w") as fh:
            for k, v in known.items():
                fh.write(f">KNOWN::{k}\n{v}\n")
            for k, v in kept.items():
                fh.write(f">CAND::{k}\n{v[0]}\n")
        subprocess.run(["mmseqs", "easy-cluster", str(td / "all.fa"), str(td / "clu"),
                        str(td / "tmp"), "--min-seq-id", str(args.max_identity),
                        "-c", "0.8", "--cov-mode", "1", "-v", "1"],
                       check=True, capture_output=True)
        members: dict[str, list[str]] = {}
        for ln in (td / "clu_cluster.tsv").open():
            rep, mem = ln.rstrip("\n").split("\t")
            members.setdefault(rep, []).append(mem)

    dropped_known = 0
    reps: dict[str, list[str]] = {}          # study -> [candidate ids]
    for rep, mem in members.items():
        cands = [m[6:] for m in mem if m.startswith("CAND::")]
        if not cands:
            continue
        if any(m.startswith("KNOWN::") for m in mem):
            dropped_known += len(cands)
            continue
        pick = cands[0]
        reps.setdefault(kept[pick][1], []).append(pick)
    funnel["5_not_near_a_known_plastizyme"] = len(kept) - dropped_known
    funnel["6_families"] = sum(len(v) for v in reps.values())
    print(f"   dropped as near a known plastizyme : {dropped_known}")
    for s, v in sorted(reps.items()):
        print(f"   {s} families: {len(v):,}")

    # ---- 7: stratified sample ------------------------------------------------------
    sample: list[str] = []
    per_study_taken = {}
    for s, v in sorted(reps.items()):
        take = min(args.per_study, len(v))
        idx = rng.choice(len(v), size=take, replace=False)
        sample += [v[i] for i in idx]
        per_study_taken[s] = take
    funnel["7_sampled"] = len(sample)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["cand_id", "study", "analysis", "length", "elbow",
                    "plastizyme_style_elbow", "sequence"])
        for k in sample:
            seq, study, acc = kept[k]
            m = ELBOW.search(seq)
            e5 = seq[m.start():m.start() + 5]
            w.writerow([k, study, acc, len(seq), e5,
                        int(bool(PLASTIZYME_ELBOW.match(e5))), seq])
    json.dump({"funnel": funnel, "studies": STUDY_LABEL,
               "per_study_families": {s: len(v) for s, v in reps.items()},
               "per_study_taken": per_study_taken,
               "seed": args.seed, "max_identity": args.max_identity},
              (args.out.parent / "screen30k_funnel.json").open("w"), indent=1)

    print("\nFUNNEL")
    for k, v in funnel.items():
        print(f"   {k:34s}{v:>12,}")
    print(f"\nwrote {args.out} ({len(sample):,} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
