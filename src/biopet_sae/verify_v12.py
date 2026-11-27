"""Independent checks of release v1.2 (project rule: do not trust the builder's own statistics).

1. Live UniProt re-query of a random sample of the new hard negatives: reviewed, IPR029058,
   EC 3.1.1.-, curated active site with Ser/His/Asp, sequence identical to the file.
2. Structural checks: unique ids, no sequence in two splits, positives and old labels unchanged
   against v1.1, group totals.
3. Cross-split bound with a DIFFERENT algorithm: mmseqs clustering at 0.40 identity (local
   alignment, so stricter than the global-identity bound); clusters spanning splits are listed.

Usage:  uv run python -m biopet_sae.verify_v12
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from biopet_sae.make_splits import write_fasta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v12", type=Path, default=Path("data/release/biopet_sae_dataset_v1_2.tsv"))
    ap.add_argument("--v11", type=Path, default=Path("data/release/biopet_sae_dataset_v1_1.tsv"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/release/biopet_sae_dataset_v1_2_verification.json"))
    a = ap.parse_args(argv)
    v12 = list(csv.DictReader(a.v12.open(newline=""), delimiter="\t"))
    v11 = {r["seq_id"]: r for r in csv.DictReader(a.v11.open(newline=""), delimiter="\t")}
    rep: dict = {}

    # 2. structure
    ids = [r["seq_id"] for r in v12]
    seqs = defaultdict(set)
    for r in v12: seqs[r["sequence"]].add(r["split"])
    rep["unique_ids"] = len(ids) == len(set(ids))
    rep["sequences_in_more_than_one_split"] = sum(1 for s in seqs.values() if len(s) > 1)
    changed = [i for i in v11 if i in {r["seq_id"] for r in v12} and any(
        {x["seq_id"]: x for x in v12}[i][k] != v11[i][k] for k in ("y", "sequence", "negative_group", "class_label", "substrates", "source"))]
    rep["v1_1_rows_with_changed_label_or_sequence"] = len(changed)
    rep["v1_1_rows_missing_from_v1_2"] = sorted(set(v11) - set(ids))
    rep["positives_v1_1_vs_v1_2"] = [sum(r["y"] == "1" for r in v11.values()), sum(r["y"] == "1" for r in v12)]
    rep["group_counts"] = dict(Counter(r["negative_group"] or "positive" for r in v12))
    rep["split_counts"] = dict(Counter(r["split"] for r in v12))

    # 1. live UniProt sample
    new = [r for r in v12 if r["batch"] == "v1.2"]
    random.Random(a.seed).shuffle(new)
    sample, bad, detail = new[: a.n], [], []
    for r in sample:
        try:
            with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{r['seq_id']}.json", timeout=60) as f:
                e = json.load(f)
        except Exception as ex:  # noqa: BLE001
            bad.append((r["seq_id"], f"fetch failed: {ex}")); continue
        xr = e.get("uniProtKBCrossReferences", [])
        d = e.get("proteinDescription", {})
        ecs = [p["value"] for p in d.get("recommendedName", {}).get("ecNumbers", [])]
        for k in ("alternativeNames", "submissionNames"):
            for x in d.get(k, []): ecs += [p["value"] for p in x.get("ecNumbers", [])]
        for inc in d.get("includes", []): ecs += [p["value"] for p in inc.get("recommendedName", {}).get("ecNumbers", [])]
        seq = e["sequence"]["value"]
        res = {seq[f["location"]["start"]["value"] - 1] for f in e.get("features", []) if f["type"] == "Active site"}
        chk = {"reviewed": e["entryType"].startswith("UniProtKB reviewed"),
               "IPR029058": any(x["database"] == "InterPro" and x["id"] == "IPR029058" for x in xr),
               "EC_3.1.1": any(c.startswith("3.1.1.") for c in ecs),
               "triad_SHD": {"S", "H", "D"} <= res, "same_sequence": seq == r["sequence"]}
        detail.append({"seq_id": r["seq_id"], **chk})
        if not all(chk.values()): bad.append((r["seq_id"], chk))
        time.sleep(0.2)
    rep["uniprot_resample"] = {"n": len(sample), "failures": bad, "all_pass": not bad}

    # 3. independent bound check (mmseqs clustering, local-alignment identity 0.40)
    exe = shutil.which("mmseqs"); work = Path(tempfile.mkdtemp(prefix="v12v_"))
    try:
        write_fasta(work / "all.fa", {r["seq_id"]: r["sequence"] for r in v12})
        subprocess.run([exe, "easy-cluster", str(work / "all.fa"), str(work / "c"), str(work / "tmp"),
                        "--min-seq-id", "0.4", "-c", "0.8", "--cov-mode", "0", "-v", "1"],
                       check=True, capture_output=True)
        sp = {r["seq_id"]: r["split"] for r in v12}
        cl = defaultdict(set)
        for line in (work / "c_cluster.tsv").read_text().splitlines():
            rep_, m = line.split("\t"); cl[rep_].add(m)
        span = [sorted(m)[:6] for m in cl.values() if len({sp[x] for x in m}) > 1]
        rep["mmseqs_cluster_0.40_spanning_splits"] = {"n_clusters": len(cl), "n_spanning": len(span), "examples": span[:5]}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    a.out.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "uniprot_resample"}, indent=1))
    print("UniProt resample:", rep["uniprot_resample"]["n"], "checked; all pass:", rep["uniprot_resample"]["all_pass"], rep["uniprot_resample"]["failures"][:3])
    return 0


if __name__ == "__main__":
    sys.exit(main())
