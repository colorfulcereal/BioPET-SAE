"""Assemble a prospective screening set from MGnify plastisphere assemblies.

Pulls predicted CDS from plastic-associated metagenome assemblies, applies a fold filter
(the GxSxG nucleophile elbow), removes anything resembling a known plastizyme or another
candidate, and samples a fixed-size set for scoring.

The elbow filter is deliberately the *generic* motif rather than the variants seen in our
positives: filtering on our own training motifs would pre-select for what the model already
knows and make any hit partly circular.

Usage:
    uv run python -m biopet_sae.fetch_metagenome --n 500
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
import requests

API = "https://www.ebi.ac.uk/metagenomics/api/v1"
# studies with assemblies, chosen for biome breadth
STUDIES = {
    "MGYS00005625": "Engineered — plastic metagenome (PRJEB15404)",
    "MGYS00006544": "Marine — tracking genomic signatures (PRJNA777294)",
    "MGYS00005970": "Lab enrichment — polyethylene film, in situ marine",
}
ELBOW = re.compile(r"G.[ST].G")
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "BioPET-SAE/0.1 (research; contact via repo)"})
    return s


def assemblies(s: requests.Session, study: str, cap: int) -> list[tuple[str, str]]:
    """[(analysis_accession, cds_filename), ...]"""
    s.headers["Accept"] = "application/json"
    r = s.get(f"{API}/studies/{study}/analyses", params={"page_size": 100}, timeout=120)
    r.raise_for_status()
    out = []
    for a in r.json()["data"]:
        if a["attributes"].get("experiment-type") != "assembly":
            continue
        acc = a["attributes"]["accession"]
        d = s.get(f"{API}/analyses/{acc}/downloads", timeout=120)
        if d.status_code != 200:
            continue
        for f in d.json()["data"]:
            if f["id"].endswith("_FASTA_predicted_cds.faa.gz"):
                out.append((acc, f["id"]))
                break
        if len(out) >= cap:
            break
    return out


def fetch_cds(s: requests.Session, acc: str, fid: str, cache: Path) -> dict[str, str]:
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{acc}_{fid}"
    if not local.exists():
        s.headers.pop("Accept", None)
        r = s.get(f"{API}/analyses/{acc}/file/{fid}", timeout=600)
        r.raise_for_status()
        local.write_bytes(r.content)
    txt = gzip.decompress(local.read_bytes()).decode()
    seqs, name = {}, None
    for ln in txt.split("\n"):
        if ln.startswith(">"):
            name = ln[1:].split(" #")[0]
            seqs[name] = ""
        elif name:
            seqs[name] += ln.strip()
    return seqs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500, help="candidates to sample")
    ap.add_argument("--per-study", type=int, default=2, help="assemblies per study")
    ap.add_argument("--positives", type=Path,
                    default=Path("data/processed/dataset_splits_id40.tsv"))
    ap.add_argument("--cache", type=Path, default=Path("data/mgnify/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/mgnify/screen_set.tsv"))
    ap.add_argument("--max-identity", type=float, default=0.90,
                    help="drop candidates at or above this identity to a known positive")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    s = session()

    funnel: dict[str, int] = {}
    raw: dict[str, tuple[str, str, str]] = {}          # id -> (seq, study, analysis)
    for study, label in STUDIES.items():
        got = assemblies(s, study, args.per_study)
        print(f"{study}: {len(got)} assemblies  [{label}]")
        for acc, fid in got:
            seqs = fetch_cds(s, acc, fid, args.cache)
            print(f"   {acc}: {len(seqs):>7} predicted CDS")
            for k, v in seqs.items():
                raw[f"{acc}|{k}"] = (v, study, acc)
    funnel["1_predicted_cds"] = len(raw)

    ok = {k: v for k, v in raw.items()
          if 50 <= len(v[0]) <= 1000 and not set(v[0]) - STANDARD}
    funnel["2_length_and_alphabet"] = len(ok)

    seen: set[str] = set()
    uniq = {}
    for k, v in ok.items():
        h = hashlib.md5(v[0].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq[k] = v
    funnel["3_exact_dedup"] = len(uniq)

    elbow = {k: v for k, v in uniq.items() if ELBOW.search(v[0])}
    funnel["4_gxsxg_elbow"] = len(elbow)

    # ---- drop anything resembling a known plastizyme, or a duplicate family -----------
    POS = {"1_pet", "2_other_polyester"}
    rows = list(csv.DictReader(args.positives.open(newline=""), delimiter="\t"))
    pos = {r["seq_id"]: r["sequence"] for r in rows if r["class_label"] in POS}
    print(f"\nclustering {len(elbow)} candidates against {len(pos)} known positives "
          f"at {args.max_identity:.0%} identity")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with (td / "all.fa").open("w") as fh:
            for k, v in pos.items():
                fh.write(f">POS::{k}\n{v}\n")
            for k, v in elbow.items():
                fh.write(f">CAND::{k}\n{v[0]}\n")
        subprocess.run(["mmseqs", "easy-cluster", str(td / "all.fa"), str(td / "clu"),
                        str(td / "tmp"), "--min-seq-id", str(args.max_identity),
                        "-c", "0.8", "--cov-mode", "1", "-v", "1"],
                       check=True, capture_output=True)
        members: dict[str, list[str]] = {}
        for ln in (td / "clu_cluster.tsv").open():
            rep, mem = ln.rstrip("\n").split("\t")
            members.setdefault(rep, []).append(mem)

    drop_known, keep = set(), []
    for rep, mem in members.items():
        has_pos = any(m.startswith("POS::") for m in mem)
        cands = [m[6:] for m in mem if m.startswith("CAND::")]
        if has_pos:
            drop_known.update(cands)
        elif cands:
            keep.append(cands[0])              # one representative per family
    funnel["5_not_near_a_known_positive"] = len(elbow) - len(drop_known)
    funnel["6_family_deduplicated"] = len(keep)
    print(f"   dropped as near-identical to a known positive: {len(drop_known)}")
    print(f"   distinct candidate families kept              : {len(keep)}")

    take = sorted(rng.choice(len(keep), size=min(args.n, len(keep)), replace=False))
    sample = [keep[i] for i in take]
    funnel["7_sampled"] = len(sample)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["cand_id", "study", "analysis", "length", "elbow", "sequence"])
        for k in sample:
            seq, study, acc = elbow[k]
            m = ELBOW.search(seq)
            w.writerow([k, study, acc, len(seq), seq[m.start():m.start() + 5], seq])
    json.dump({"funnel": funnel, "studies": STUDIES, "seed": args.seed,
               "max_identity": args.max_identity},
              (args.out.parent / "screen_funnel.json").open("w"), indent=1)

    print("\nFUNNEL")
    for k, v in funnel.items():
        print(f"   {k:34s}{v:>9}")
    print(f"\nwrote {args.out} ({len(sample)} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
