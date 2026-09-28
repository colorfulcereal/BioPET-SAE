"""Pull UniProt negatives for classes 3 and 4, phylum-matched to the PET positives.

Implements decisions D2 and D6.1 in PROGRESS.md.

Class 3 (fold-matched hard negatives): reviewed `ec:3.1.1.-` carboxylic-ester hydrolases
with no plastic-degradation annotation. Same fold family as the PETases, PET status
unknown -- the point is that they are *homologous*, so they are deliberately NOT filtered
at 30% identity.

Class 4 (naive controls): reviewed proteins outside EC 3.1 entirely.

Both are sampled to match the phylum distribution of the PET positives, because the PET
class is 72.6% Actinomycetota while the assay-confirmed contrast class is 2.5% -- a skew
big enough that a probe could score well by detecting phylum instead of PET activity.
Matching the mix removes phylum as a discriminative signal.

`ec:3.1.1.-` is an exact subclass match. `ec:3.1.1.*` is a loose string prefix and
returns 6,174 reviewed entries against 4,294 for the exact form -- 1,880 spurious.

Accession-level exclusion of known positives happens here. Identity-level exclusion
(dropping negatives >=90% identical to a positive) happens in the clustering step, where
mmseqs2 can do it properly.

Usage:
    uv run python -m biopet_sae.fetch_negatives [--target-class3 240] [--target-class4 240]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,id,protein_name,organism_name,organism_id,ec,length,sequence,keyword"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# NCBI taxonomy ids for the phyla present in the PET class. Names are the current
# (post-2021) phylum names; UniProt still resolves the ids.
PHYLUM_TAXA: dict[str, int] = {
    "Actinomycetota": 201174,   # formerly Actinobacteria -- high-GC, the dominant phylum
    "Pseudomonadota": 1224,     # formerly Proteobacteria
    "Bacillota": 1239,          # formerly Firmicutes
    "Ascomycota": 4890,         # fungi
    "Bacteroidota": 976,
    "Basidiomycota": 5204,
    "Chloroflexota": 200795,
    "Deinococcota": 1297,
}

# any of these in a UniProt keyword or protein name means the entry is not safely a
# negative, whatever its EC annotation says
PLASTIC_HINTS = re.compile(
    r"polyethylene terephthalate|PETase|MHETase|terephthalate|cutinase|"
    r"plastic|polyester hydrolase|polyurethan",
    re.IGNORECASE,
)


def fetch_query(query: str, cap: int, session: requests.Session) -> list[dict[str, str]]:
    """Page through a UniProt search, stopping once `cap` rows are collected."""
    rows: list[dict[str, str]] = []
    url = f"{UNIPROT_SEARCH}?query={quote(query)}&fields={FIELDS}&format=tsv&size=500"
    while url and len(rows) < cap:
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
        lines = resp.text.splitlines()
        if len(lines) > 1:
            reader = csv.DictReader(lines, delimiter="\t")
            rows.extend(reader)
        # UniProt paginates with a cursor in the Link header
        link = resp.headers.get("Link", "")
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
        if url:
            time.sleep(0.2)
    return rows[:cap]


def passes_qc(row: dict[str, str]) -> bool:
    seq = (row.get("Sequence") or "").upper()
    if not (50 <= len(seq) <= 1000):
        return False
    if set(seq) - STANDARD_AA:
        return False
    blob = " ".join(
        (row.get(k) or "") for k in ("Protein names", "Keywords", "EC number")
    )
    return not PLASTIC_HINTS.search(blob)


def phylum_quota(counts: dict[str, int], target: int) -> dict[str, int]:
    """Scale an observed phylum distribution to a target total."""
    total = sum(counts.values())
    return {ph: max(1, round(target * n / total)) for ph, n in counts.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positives", type=Path,
                    default=Path("data/processed/plastizymes_merged.tsv"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--target-class3", type=int, default=240)
    ap.add_argument("--target-class4", type=int, default=240)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- what to exclude, and what mix to match --------------------------------
    with args.positives.open(newline="") as fh:
        pos = list(csv.DictReader(fh, delimiter="\t"))
    pet = [p for p in pos if p["class_label"] == "1_pet"]

    banned_accessions = set()
    for p in pos:
        for acc in (p["uniprot"] or "").split("|"):
            if acc.strip():
                banned_accessions.add(acc.strip())
    print(f"excluding {len(banned_accessions)} accessions already in the positive set")

    observed: dict[str, int] = {}
    for p in pet:
        for ph in (p["phylum"] or "").split("|"):
            ph = ph.strip()
            if ph in PHYLUM_TAXA:
                observed[ph] = observed.get(ph, 0) + 1
    print(f"PET-class phylum mix to match: {observed}")

    session = requests.Session()
    session.headers["User-Agent"] = "BioPET-SAE/0.1 (research; contact via repo)"

    report: dict[str, object] = {"target_mix": observed, "classes": {}}
    all_rows: list[dict[str, object]] = []

    specs = [
        ("3_fold_matched_esterase", args.target_class3,
         "(ec:3.1.1.-) AND (reviewed:true) AND (length:[50 TO 1000])"),
        ("4_naive_control", args.target_class4,
         "(reviewed:true) AND (length:[50 TO 1000]) NOT (ec:3.1.*)"),
    ]

    for class_label, target, base in specs:
        quota = phylum_quota(observed, target)
        print(f"\n=== {class_label} (target {target}) ===")
        print(f"quota: {quota}")
        got: dict[str, int] = {}
        available: dict[str, int] = {}
        for ph, want in sorted(quota.items(), key=lambda kv: -kv[1]):
            q = f"{base} AND (taxonomy_id:{PHYLUM_TAXA[ph]})"
            # over-fetch, because QC and accession exclusion will remove some
            raw = fetch_query(q, cap=max(want * 4, 100), session=session)
            available[ph] = len(raw)
            pool = [
                r for r in raw
                if r.get("Entry") not in banned_accessions and passes_qc(r)
            ]
            rng.shuffle(pool)
            picked = pool[:want]
            got[ph] = len(picked)
            short = " (SHORT)" if len(picked) < want else ""
            print(f"  {ph:16s} want {want:4d}  fetched {len(raw):5d}  "
                  f"usable {len(pool):5d}  took {len(picked):4d}{short}")
            for r in picked:
                all_rows.append({
                    "accession": r.get("Entry", ""),
                    "entry_name": r.get("Entry Name", ""),
                    "class_label": class_label,
                    "protein_names": r.get("Protein names", ""),
                    "organism": r.get("Organism", ""),
                    "organism_id": r.get("Organism (ID)", ""),
                    "phylum": ph,
                    "ec": r.get("EC number", ""),
                    "keywords": r.get("Keywords", ""),
                    "seq_len": len(r.get("Sequence", "")),
                    "sequence": (r.get("Sequence") or "").upper(),
                })
        report["classes"][class_label] = {
            "target": target, "quota": quota, "obtained": got,
            "total": sum(got.values()), "available_before_filters": available,
        }

    # dedupe across the two classes and against itself by sequence
    seen: set[str] = set()
    unique_rows = []
    for r in all_rows:
        if r["sequence"] in seen:
            continue
        seen.add(str(r["sequence"]))
        unique_rows.append(r)
    report["dropped_duplicate_sequences"] = len(all_rows) - len(unique_rows)

    cols = list(unique_rows[0].keys())
    tsv = args.out / "uniprot_negatives.tsv"
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(unique_rows)

    fasta = args.out / "uniprot_negatives.fasta"
    with fasta.open("w") as fh:
        for r in unique_rows:
            fh.write(f">{r['accession']} {r['class_label']} {r['phylum']}\n")
            seq = str(r["sequence"])
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")

    report["total_written"] = len(unique_rows)
    (args.out / "negatives_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n{json.dumps(report['classes'], indent=2)}")
    print(f"\nwrote {tsv} ({len(unique_rows)} rows) and {fasta}")
    print("NOTE: identity-based exclusion (>=90% to a positive) happens in the "
          "clustering step, not here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
