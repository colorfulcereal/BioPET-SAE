"""Build the hand-off release of the dataset: one annotated TSV plus a FASTA.

Adds the corrected, fold-based negative classification that §21-§22c established, so a
downstream user does not repeat the EC-class mistake. Fold/active-site annotation is
fetched live from UniProt and cached.

Usage:
    uv run python -m biopet_sae.build_release
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

POS = {"1_pet", "2_other_polyester"}
NEG = {"3_fold_matched_esterase", "4_naive_control"}
ELBOW = re.compile(r"G.[ST].G")
TRNA_PFAM = {"PF02580", "PF01195"}       # D-aminoacyl-tRNA deacylase, peptidyl-tRNA hydrolase
AB_FOLD = "IPR029058"                    # alpha/beta hydrolase fold superfamily
FIELDS = "accession,ec,xref_pfam,xref_interpro,ft_act_site"


def fetch_uniprot(accs: list[str], cache: Path, refresh: bool = False) -> dict[str, dict]:
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    out: dict[str, dict] = {}
    for i in range(0, len(accs), 100):
        q = " OR ".join(f"accession:{a}" for a in accs[i : i + 100])
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {"query": q, "fields": FIELDS, "format": "json", "size": 500})
        with urllib.request.urlopen(url, timeout=180) as fh:
            d = json.load(fh)
        for e in d["results"]:
            desc = e.get("proteinDescription", {})
            ec = [p["value"] for p in desc.get("recommendedName", {}).get("ecNumbers", [])]
            for inc in desc.get("includes", []):
                ec += [p["value"] for p in inc.get("recommendedName", {}).get("ecNumbers", [])]
            # EC can also sit only on an alternative or submission name -- two negatives here
            # are EC 3.1.1 that way, and would otherwise be miscounted as naive controls
            for alt in desc.get("alternativeNames", []) + desc.get("submissionNames", []):
                ec += [p["value"] for p in alt.get("ecNumbers", [])]
            out[e["primaryAccession"]] = {
                "ec": sorted(set(ec)),
                "pfam": [x["id"] for x in e.get("uniProtKBCrossReferences", [])
                         if x["database"] == "Pfam"],
                "ipr": [x["id"] for x in e.get("uniProtKBCrossReferences", [])
                        if x["database"] == "InterPro"],
                "act": [f["location"]["start"]["value"] for f in e.get("features", [])
                        if f["type"] == "Active site"],
            }
        time.sleep(0.2)
        print(f"   fetched {len(out):,}/{len(accs):,}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def negative_group(a: dict) -> str:
    """The fold-based classification. EC 3.1.1 is a reaction class, not a fold."""
    if TRNA_PFAM & set(a["pfam"]):
        return "tRNA_processing_no_ser_nucleophile"
    if AB_FOLD in a["ipr"]:
        return "ab_hydrolase_hard_negative"
    if any(e.startswith("3.1.1") for e in a["ec"]):
        return "ec311_wrong_fold"
    return "naive_control"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path,
                    default=Path("data/processed/dataset_splits_id40.tsv"))
    ap.add_argument("--merged", type=Path,
                    default=Path("data/processed/plastizymes_merged.tsv"))
    ap.add_argument("--cache", type=Path,
                    default=Path("data/processed/uniprot_negative_annotation.json"))
    ap.add_argument("--out", type=Path, default=Path("data/release"))
    ap.add_argument("--refresh", action="store_true",
                    help="re-query UniProt and overwrite the annotation cache")
    args = ap.parse_args(argv)

    rows = list(csv.DictReader(args.splits.open(newline=""), delimiter="\t"))
    mg = {r["seq_id"]: r for r in csv.DictReader(args.merged.open(newline=""), delimiter="\t")}
    negs = [r["seq_id"] for r in rows if r["class_label"] in NEG]
    print(f"{len(rows)} rows; annotating {len(negs)} negatives from UniProt")
    ann = fetch_uniprot(negs, args.cache, refresh=args.refresh)
    missing = [a for a in negs if a not in ann]
    if missing:
        print(f"WARNING: {len(missing)} negatives unresolved: {missing[:5]}")

    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["seq_id", "split", "y", "class_label", "negative_group", "ab_hydrolase_fold",
            "gxsxg_elbow", "n_curated_active_sites", "ec", "pfam", "activity_verified",
            "extrapolated_only", "substrates", "source", "organism", "phylum",
            "component", "cluster", "seq_len", "sequence"]
    with (args.out / "biopet_sae_dataset_v1.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            i, lab = r["seq_id"], r["class_label"]
            a = ann.get(i)
            m = mg.get(i, {})
            w.writerow([
                i, r["split"], 1 if lab in POS or lab == "heldout_pha" else 0, lab,
                negative_group(a) if a else "",
                int(AB_FOLD in a["ipr"]) if a else "",
                int(bool(ELBOW.search(r["sequence"]))),
                len(a["act"]) if a else "",
                ";".join(a["ec"]) if a else "",
                ";".join(a["pfam"]) if a else "",
                m.get("verified_activity", ""), m.get("extrapolated_only", ""),
                r["substrates"], r["source"], r["organism"], r["phylum"],
                r["component"], r["cluster"], r["seq_len"], r["sequence"],
            ])
    with (args.out / "biopet_sae_dataset_v1.fasta").open("w") as fh:
        for r in rows:
            a = ann.get(r["seq_id"])
            g = negative_group(a) if a else r["class_label"]
            fh.write(f">{r['seq_id']} split={r['split']} class={r['class_label']} group={g}\n")
            for k in range(0, len(r["sequence"]), 60):
                fh.write(r["sequence"][k : k + 60] + "\n")
    print(f"wrote {args.out}/biopet_sae_dataset_v1.tsv  ({len(rows)} rows, {len(cols)} columns)")
    print(f"wrote {args.out}/biopet_sae_dataset_v1.fasta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
