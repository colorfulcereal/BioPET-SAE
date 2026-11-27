"""Fetch and filter candidate hard negatives for dataset v1.2 from UniProt (Swiss-Prot).

Hard negative = alpha/beta-hydrolase-fold (InterPro IPR029058) carboxylic ester hydrolase
(EC 3.1.1.-, exact subclass) with a curated Ser-His-Asp active site and no report of degrading
a plastic. This script produces the candidate pool and the funnel of counts; it applies only
annotation filters. Identity filters (>= 90% to a positive, duplicates) are in
`select_hard_negatives_v12.py`, because they need the full sequence set.

Filters, in order (each count is recorded):
  1 reviewed, IPR029058, EC 3.1.1.- (the UniProt query)
  2 not already in the dataset (positive accession or existing negative)
  3 standard 20 amino acids only, length 100-1022
  4 curated "Active site" features whose residues at those positions include S, H and D
  5 not plastic-linked: EC 3.1.1.101 (PET hydrolase), 3.1.1.74 (cutinase), 3.1.1.75/3.1.1.76
    (PHB depolymerases: they degrade a polyester, so they are positives by definition), or a
    name/keyword mentioning PET / plastic / polyester / cutinase / depolymerase / PLA / PCL ...

Usage:  uv run python -m biopet_sae.fetch_hard_negatives_v12
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

QUERY = "reviewed:true AND xref:interpro-IPR029058 AND ec:3.1.1.-"
PLASTIC_EC = {"3.1.1.101", "3.1.1.74", "3.1.1.75", "3.1.1.76"}
PLASTIC_RE = re.compile(
    r"(?i:\bcutinase|\bPETase|\bMHETase|\bplastics?\b|polyester|\bpoly\(|polyethylene|polylactic|"
    r"polycaprolactone|depolymerase|polyurethane|\bnylon|polyhydroxy)|"
    r"\b(?:PET|PHA|PHB|PLA|PCL|PBAT|PBS|PBSA|PEF|PU)\b")      # acronyms: upper-case whole words only
STD = set("ACDEFGHIKLMNPQRSTVWY")


def fetch_all(url: str) -> list[dict]:
    out, nxt = [], url
    while nxt:
        req = urllib.request.Request(nxt, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            out += json.load(r)["results"]
            link = r.headers.get("Link", "")
            m = re.search(r'<([^>]+)>;\s*rel="next"', link)
            nxt = m.group(1) if m else None
        time.sleep(0.3)
        print(f"  fetched {len(out)}", flush=True)
    return out



def phylum_of(lin: list[str]) -> str:
    """Phylum from a UniProt lineage list. Bacteria/Archaea: the second name. Fungi: the name
    ending in -mycota. Other eukaryotes: the third name (e.g. Metazoa -> Chordata)."""
    if not lin: return ""
    if lin[0] in ("Bacteria", "Archaea"): return lin[1] if len(lin) > 1 else ""
    for n in lin:
        if n.endswith("mycota"): return n
    return lin[2] if len(lin) > 2 else lin[-1]


def names(e: dict) -> str:
    d = e.get("proteinDescription", {})
    parts = []
    for k in ("recommendedName",):
        if k in d: parts.append(d[k]["fullName"]["value"])
    for k in ("alternativeNames", "submissionNames", "includes", "contains"):
        for x in d.get(k, []):
            n = x.get("fullName") or x.get("recommendedName", {}).get("fullName")
            if n: parts.append(n["value"])
    return " | ".join(parts)


def ec_numbers(e: dict) -> list[str]:
    d = e.get("proteinDescription", {})
    ec = []
    for k in ("recommendedName",):
        ec += [p["value"] for p in d.get(k, {}).get("ecNumbers", [])]
    for k in ("alternativeNames", "submissionNames"):
        for x in d.get(k, []):
            ec += [p["value"] for p in x.get("ecNumbers", [])]
    for inc in d.get("includes", []):
        ec += [p["value"] for p in inc.get("recommendedName", {}).get("ecNumbers", [])]
    return sorted(set(ec))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path("data/release/biopet_sae_dataset_v1_1.tsv"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw/hard_negatives_v12/uniprot_raw.json"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/hard_negatives_v12"))
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.raw.exists() and not a.refresh:
        entries = json.loads(a.raw.read_text())
    else:
        url = ("https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {"query": QUERY, "format": "json", "size": 500}))
        entries = fetch_all(url)
        a.raw.write_text(json.dumps(entries))
    meta = {"query": QUERY, "n_fetched": len(entries),
            "uniprot_release": None, "fetched": time.strftime("%Y-%m-%d")}
    rel = list(csv.DictReader(a.release.open(newline=""), delimiter="\t"))
    have = {r["seq_id"] for r in rel}
    pos_seq = {r["sequence"] for r in rel if r["y"] == "1"}
    funnel = [("UniProt query (reviewed, IPR029058, EC 3.1.1.-)", len(entries))]
    cur = []
    for e in entries:
        cur.append(e)
    def step(label, keep):
        nonlocal_cur[:] = [e for e in nonlocal_cur if keep(e)]
        funnel.append((label, len(nonlocal_cur)))
    nonlocal_cur = list(cur)
    step("not already in the dataset", lambda e: e["primaryAccession"] not in have
         and e["sequence"]["value"] not in pos_seq)
    step("standard residues only, length 100-1022",
         lambda e: set(e["sequence"]["value"]) <= STD and 100 <= e["sequence"]["length"] <= 1022)

    def triad(e):
        seq = e["sequence"]["value"]
        res = set()
        for f in e.get("features", []):
            if f["type"] == "Active site":
                p = f["location"]["start"]["value"]
                if 1 <= p <= len(seq): res.add(seq[p - 1])
        return {"S", "H", "D"} <= res
    step("curated active site with Ser, His and Asp", triad)

    def plastic_linked(e):
        if PLASTIC_EC & set(ec_numbers(e)): return True
        kw = " ".join(k["name"] for k in e.get("keywords", []))
        return bool(PLASTIC_RE.search(names(e) + " " + kw))
    step("not plastic-linked (EC / name / keyword)", lambda e: not plastic_linked(e))

    rows = []
    for e in nonlocal_cur:
        xr = e.get("uniProtKBCrossReferences", [])
        acts = sorted({(f["location"]["start"]["value"], e["sequence"]["value"][f["location"]["start"]["value"] - 1])
                       for f in e.get("features", []) if f["type"] == "Active site"})
        lin = e.get("organism", {}).get("lineage", [])
        rows.append({"seq_id": e["primaryAccession"], "name": names(e)[:120],
                     "organism": e["organism"]["scientificName"],
                     "phylum": phylum_of(lin),
                     "domain": lin[0] if lin else "",
                     "kingdom": lin[1] if len(lin) > 1 else "",
                     "ec": ";".join(ec_numbers(e)),
                     "pfam": ";".join(sorted({x["id"] for x in xr if x["database"] == "Pfam"})),
                     "interpro": ";".join(sorted({x["id"] for x in xr if x["database"] == "InterPro"})),
                     "active_sites": ";".join(f"{r}{p}" for p, r in acts),
                     "seq_len": e["sequence"]["length"], "sequence": e["sequence"]["value"]})
    with (a.out / "candidates_annotation_filtered.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
    meta["funnel"] = funnel
    (a.out / "funnel_annotation.json").write_text(json.dumps(meta, indent=2))
    for l, n in funnel: print(f"{n:6d}  {l}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
