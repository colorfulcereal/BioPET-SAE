"""Identity filtering and spread-sampling of the v1.2 hard-negative candidates.

Pairs: mmseqs all-vs-all (sensitive) over the v1.1 dataset plus all annotation-filtered
candidates, then every pair with local identity >= 0.30 is re-aligned globally with the
project's routine (BLOSUM62, gap -11/-1, free end gaps, identity over the shorter sequence).

Identity filters (recorded as a funnel):
  * >= 0.90 to any positive                    -> drop (label conflict; removed P37957 before)
  * 0.40-0.90 to any positive                  -> drop by default and list (--keep-near-positive to keep)
  * >= 0.90 to an existing dataset sequence    -> drop (near-duplicate)
  * >= 0.90 to another candidate               -> keep one per group
Then, if more than --max remain, a spread sample: candidates are grouped into homology
components (0.40), components are drawn round-robin across phyla with a fixed seed. Nothing is
selected or rejected by probe score.

Usage:  uv run python -m biopet_sae.select_hard_negatives_v12
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
import shutil
from collections import defaultdict
from pathlib import Path

from biopet_sae.make_splits import (UnionFind, global_identity, make_aligner,
                                    mmseqs_all_vs_all, require_mmseqs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path("data/release/biopet_sae_dataset_v1_1.tsv"))
    ap.add_argument("--cand", type=Path, default=Path("data/raw/hard_negatives_v12/candidates_annotation_filtered.tsv"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/hard_negatives_v12"))
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-near-positive", action="store_true")
    a = ap.parse_args(argv)
    rel = list(csv.DictReader(a.release.open(newline=""), delimiter="\t"))
    cand = list(csv.DictReader(a.cand.open(newline=""), delimiter="\t"))
    seq = {r["seq_id"]: r["sequence"] for r in rel}
    seq.update({r["seq_id"]: r["sequence"] for r in cand})
    pos = {r["seq_id"] for r in rel if r["y"] == "1"}
    existing = {r["seq_id"] for r in rel}
    cands = {r["seq_id"] for r in cand}

    cache = a.out / "pair_identities.tsv"
    if cache.exists():
        ident = {}
        for r in csv.DictReader(cache.open(newline=""), delimiter="\t"):
            ident[(r["a"], r["b"])] = float(r["identity"])
        print(f"loaded {len(ident)} cached pair identities")
    else:
        exe = require_mmseqs(); work = Path(tempfile.mkdtemp(prefix="v12_"))
        try:
            pairs = mmseqs_all_vs_all(exe, seq, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        seen = {}
        for q, t, f in pairs:
            k = tuple(sorted((q, t))); seen[k] = max(seen.get(k, 0), f)
        todo = [k for k, f in seen.items() if f >= 0.30]
        print(f"{len(seen)} candidate pairs, realigning {len(todo)} with local identity >= 0.30", flush=True)
        al = make_aligner(); ident = {}
        for i, (x, y) in enumerate(todo):
            ident[(x, y)] = global_identity(seq[x], seq[y], al)
            if (i + 1) % 5000 == 0: print(f"  {i+1}/{len(todo)}", flush=True)
        with cache.open("w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t"); w.writerow(["a", "b", "identity"])
            for (x, y), v in ident.items(): w.writerow([x, y, round(v, 5)])
    nbr = defaultdict(dict)
    for (x, y), v in ident.items():
        nbr[x][y] = v; nbr[y][x] = v

    funnel = [("annotation-filtered candidates", len(cands))]
    keep = set(cands); dropped = {}
    def drop(ids, why, detail=None):
        for i in ids:
            dropped[i] = {"reason": why, **(detail(i) if detail else {})}
        keep.difference_update(ids)
    mx = lambda i, S: max(((v, j) for j, v in nbr[i].items() if j in S), default=(0.0, ""))
    d90 = [i for i in keep if mx(i, pos)[0] >= 0.90]
    drop(d90, ">=90% identical to a positive", lambda i: {"identity": round(mx(i, pos)[0], 3), "to": mx(i, pos)[1]})
    funnel.append((">= 90% identical to a positive removed", len(keep)))
    near = [i for i in keep if 0.40 <= mx(i, pos)[0] < 0.90]
    if not a.keep_near_positive:
        drop(near, "40-90% identical to a positive", lambda i: {"identity": round(mx(i, pos)[0], 3), "to": mx(i, pos)[1]})
    funnel.append(("40-90% identical to a positive removed" if not a.keep_near_positive else "40-90% kept", len(keep)))
    dup_ex = [i for i in keep if mx(i, existing)[0] >= 0.90]
    drop(dup_ex, ">=90% identical to an existing dataset sequence", lambda i: {"identity": round(mx(i, existing)[0], 3), "to": mx(i, existing)[1]})
    funnel.append((">= 90% identical to an existing sequence removed", len(keep)))
    uf = UnionFind()
    for i in keep: uf.find(i)
    for i in keep:
        for j, v in nbr[i].items():
            if j in keep and v >= 0.90: uf.union(i, j)
    groups = defaultdict(list)
    for i in keep: groups[uf.find(i)].append(i)
    byid = {r["seq_id"]: r for r in cand}
    rep = {g: sorted(m, key=lambda i: (-len(byid[i]["active_sites"].split(";")), i))[0] for g, m in groups.items()}
    drop([i for m in groups.values() for i in m if i != rep[uf.find(i)]], "near-duplicate of another candidate (>= 90%)")
    funnel.append(("one per 90%-identity group", len(keep)))

    # homology components among the survivors (0.40) for the spread sample
    uf2 = UnionFind()
    for i in keep: uf2.find(i)
    for i in keep:
        for j, v in nbr[i].items():
            if j in keep and v >= 0.40: uf2.union(i, j)
    comps = defaultdict(list)
    for i in keep: comps[uf2.find(i)].append(i)
    funnel.append((f"homology components among survivors (0.40): {len(comps)}", len(keep)))
    chosen = set(keep)
    if len(keep) > a.max:
        rng = random.Random(a.seed)
        by_ph = defaultdict(list)
        for c, m in comps.items():
            by_ph[byid[m[0]]["phylum"]].append(c)
        for v in by_ph.values(): rng.shuffle(v)
        order = sorted(by_ph, key=lambda p: (-len(by_ph[p]), p))
        chosen = set()
        while len(chosen) < a.max and any(by_ph.values()):
            for p in order:
                if by_ph[p] and len(chosen) < a.max:
                    chosen.update(comps[by_ph[p].pop()])
        # trim overshoot deterministically
        chosen = set(sorted(chosen)[: a.max]) if len(chosen) > a.max else chosen
        drop(keep - chosen, "not drawn in the spread sample (cap)")
    funnel.append((f"spread sample (cap {a.max})", len(chosen)))
    out = [byid[i] for i in sorted(chosen)]
    with (a.out / "hard_negatives_new.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cand[0]), delimiter="\t"); w.writeheader(); w.writerows(out)
    (a.out / "dropped.json").write_text(json.dumps(dropped, indent=1))
    (a.out / "funnel_identity.json").write_text(json.dumps({"funnel": funnel, "seed": a.seed, "max": a.max}, indent=2))
    for l, n in funnel: print(f"{n:6d}  {l}")
    print("dropped by reason:", {r: sum(1 for d in dropped.values() if d['reason'] == r) for r in {d['reason'] for d in dropped.values()}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
