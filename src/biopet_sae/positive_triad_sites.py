"""Curated catalytic Ser/Asp/His positions for the positives, from UniProt "Active site" features.

Every accession listed for a positive is tried. Positions are transferred onto our sequence through a global
alignment and kept only if the two sequences are >= 95% identical and the residue letter is unchanged.
Duplicated annotations of one position are counted once. Usable = exactly one S, one D and one H.
Only public accessions are sent.

Usage:  uv run python -m biopet_sae.positive_triad_sites
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

from biopet_sae.make_splits import global_identity, make_aligner


def main() -> int:
    rows = {r["seq_id"]: r for r in csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t")}
    merged = {r["seq_id"]: r for r in csv.DictReader(open("data/processed/plastizymes_merged.tsv"), delimiter="\t")}
    al = make_aligner()
    out, log = {}, Counter()
    for sid, r in rows.items():
        if r["y"] != "1" or sid not in merged:
            continue
        accs = [x for x in re.split(r"[|;,\s]+", merged[sid]["uniprot"]) if x]
        if not accs:
            log["no accession"] += 1
            continue
        best = None
        for acc in accs:
            try:
                with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{acc}.json", timeout=60) as f:
                    e = json.load(f)
            except Exception:                                    # noqa: BLE001
                continue
            time.sleep(0.05)
            useq = e.get("sequence", {}).get("value", "")
            pos = sorted({x["location"]["start"]["value"] for x in e.get("features", []) if x["type"] == "Active site"})
            if not useq or not pos:
                continue
            s = r["sequence"]
            if global_identity(s, useq, al) < 0.95:
                continue
            m = {}
            aln = al.align(useq, s)[0]
            for (a0, a1), (b0, b1) in zip(*aln.aligned):
                for k in range(a1 - a0):
                    m[a0 + k] = b0 + k
            got = {}
            for p in pos:
                q = m.get(p - 1)
                if q is not None and s[q] == useq[p - 1] and s[q] in "SDH":
                    got.setdefault(s[q], set()).add(q)
            if all(len(got.get(t, ())) == 1 for t in "SDH"):
                best = {"acc": acc, **{t: int(next(iter(got[t]))) for t in "SDH"}}
                break
            log["entry without exactly one S, D, H"] += 1
        if best:
            out[sid] = {**best, "split": r["split"], "component": r["component"]}
    p = Path("results/v1_2/algorithm_localisation")
    p.mkdir(parents=True, exist_ok=True)
    (p / "positive_triad_sites.json").write_text(json.dumps(out, indent=1))
    print("positives with a usable curated triad:", len(out), dict(Counter(v["split"] for v in out.values())))
    print("homology families represented:", {s: len({v["component"] for v in out.values() if v["split"] == s}) for s in ("train", "val", "test")})
    print(dict(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
