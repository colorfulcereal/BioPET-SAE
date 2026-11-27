"""Cross-check the 16 PAZy additions against UniProt/UniParc/InterPro.

For each sequence: UniParc lookup by MD5 (UniParc's checksum is the MD5 of the sequence),
then, for every UniProtKB accession on that UniParc entry, fetch the entry and record the
protein name, organism, EC, InterPro/Pfam cross-references, curated active sites and
whether the stored sequence is identical. Only public checksums/accessions are sent.

Usage:  uv run python -m biopet_sae.check_pazy_update_uniprot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


def get(url: str):
    for k in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2)
    raise err


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", type=Path, default=Path("data/raw/pazy_update/pazy_new_pet_2026-11-23.json"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/pazy_update/uniprot_check.json"))
    a = ap.parse_args(argv)
    res = {}
    for o in json.loads(a.new.read_text()):
        seq = o["amino_acid_sequence"]
        q = urllib.parse.quote("checksum:" + o["sequence_md5"])
        up = get(f"https://rest.uniprot.org/uniparc/search?query={q}&size=5")["results"]
        accs = sorted({x.split(".")[0] for r in up for x in r.get("uniProtKBAccessions", [])})
        entries = []
        for acc in accs:
            try:
                e = get(f"https://rest.uniprot.org/uniprotkb/{acc}.json")
            except Exception:  # noqa: BLE001
                entries.append({"accession": acc, "error": "not retrievable (obsolete?)"})
                continue
            if e.get("entryType") == "Inactive":
                entries.append({"accession": acc, "error": "inactive: " + str(e.get("inactiveReason", {}).get("inactiveReasonType"))})
                continue
            pd = e.get("proteinDescription", {})
            nm = (pd.get("recommendedName") or (pd.get("submissionNames") or [{}])[0]).get("fullName", {}).get("value", "")
            xr = e.get("uniProtKBCrossReferences", [])
            entries.append({
                "accession": acc, "reviewed": e.get("entryType", "").startswith("UniProtKB reviewed"),
                "name": nm, "organism": e.get("organism", {}).get("scientificName", ""),
                "ec": [x["value"] for x in xr if x["database"] == "EC"] or
                      [c["value"] for k in ("recommendedName",) for c in pd.get(k, {}).get("ecNumbers", [])],
                "interpro": sorted({f'{x["id"]}:{(x.get("properties") or [{}])[0].get("value","")}' for x in xr if x["database"] == "InterPro"}),
                "pfam": sorted({x["id"] for x in xr if x["database"] == "Pfam"}),
                "act_sites": [f["location"]["start"]["value"] for f in e.get("features", []) if f["type"] == "Active site"],
                "same_sequence": e.get("sequence", {}).get("value") == seq,
                "annotation_score": e.get("annotationScore"),
            })
        res[o["name"]] = {"pazy_id": o["id"], "pazy_organism": o["organism"]["scientific_name"],
                          "uniparc": [r["uniParcId"] for r in up], "n_uniprot": len(entries), "entries": entries}
        print(o["name"], [e["accession"] for e in entries], flush=True)
    a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
