"""BLASTp baseline for the polyester task, read the same way as the probe.

Database = the TRAIN positives only (no leakage); queries = val and test. Score = -log10 of the best-hit
e-value (no hit = e-value 1e9; ranks identically to e-value; reproduces v1 exactly, whereas
bitscore does not). Decision rule = best-hit e-value <= e*, with e* chosen on val by F1
from a half-decade grid and carried to test (mirrors the probe's val-selected threshold).
Reported blended and by negative group with component bootstraps (positive and negative
components resampled separately), plus recall at fixed precision.

Regression target on frozen v1 (PROGRESS.md section 14): test AU-ROC 0.9040, AU-PRC 0.7552,
e* = 3.16e-2, TP/FP/FN/TN = 40/8/2/54.

Usage:
    uv run python -m biopet_sae.polyester_blast --release data/release/biopet_sae_dataset_v1.tsv \
        --out results/v1_repro/blast
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from biopet_sae.polyester_probe import comp_boot, metrics, recall_at_precision

GRID = [10.0 ** (k / 2) for k in range(2, -13, -1)]     # 10 ... 1e-6 in half decades


def write_fa(path: Path, seqs: dict[str, str]) -> None:
    with path.open("w") as fh:
        for k, v in seqs.items():
            fh.write(f">{k}\n{v}\n")


def best_hits(db: dict[str, str], q: dict[str, str], work: Path) -> dict[str, tuple[float, float]]:
    work.mkdir(parents=True, exist_ok=True)
    write_fa(work / "db.fa", db); write_fa(work / "q.fa", q)
    subprocess.run(["makeblastdb", "-in", str(work / "db.fa"), "-dbtype", "prot", "-out", str(work / "db")],
                   check=True, capture_output=True)
    subprocess.run(["blastp", "-query", str(work / "q.fa"), "-db", str(work / "db"), "-evalue", "1000",
                    "-outfmt", "6 qseqid sseqid evalue bitscore", "-max_target_seqs", "500",
                    "-num_threads", "4", "-out", str(work / "hits.tsv")], check=True, capture_output=True)
    best: dict[str, tuple[float, float]] = {}
    for line in (work / "hits.tsv").read_text().splitlines():
        qid, _s, ev, bits = line.split("\t")
        ev, bits = float(ev), float(bits)
        if qid not in best or bits > best[qid][0]:
            best[qid] = (bits, ev)
    return best


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-12), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(a.release.open(newline=""), delimiter="\t"))
    ids = np.array([r["seq_id"] for r in rows]); y = np.array([int(r["y"]) for r in rows])
    sp = np.array([r["split"] for r in rows]); comp = np.array([r["component"] for r in rows])
    grp = np.array([r["negative_group"] for r in rows]); seq = {r["seq_id"]: r["sequence"] for r in rows}
    batch = np.array([r.get("batch", "v1") for r in rows])
    db = {i: seq[i] for i in ids[(sp == "train") & (y == 1)]}
    work = Path(tempfile.mkdtemp(prefix="pblast_"))
    try:
        hits = {}
        for which in ("val", "test"):
            q = {i: seq[i] for i in ids[sp == which]}
            hits.update(best_hits(db, q, work / which))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ev = np.array([hits.get(i, (0.0, 1e9))[1] for i in ids])
    bits = -np.log10(np.maximum(ev, 1e-300))          # score = -log10(best e-value); no hit -> -9
    va, te = sp == "val", sp == "test"
    # e* on val: best F1 over the grid (ties -> stricter e-value)
    f1s = [prf(y[va], (ev[va] <= e).astype(int))["f1"] for e in GRID]
    e_star = GRID[int(np.argmax(f1s))]
    res = {"db_size": len(db), "e_star": e_star, "val_f1_at_e_star": max(f1s),
           "val": {**metrics(y[va], bits[va], comp[va], boot=False), "at_e_star": prf(y[va], (ev[va] <= e_star).astype(int))},
           "test_blended": {**metrics(y[te], bits[te], comp[te]),
                            "at_e_star": prf(y[te], (ev[te] <= e_star).astype(int)),
                            "recall_at_precision": {f"{int(f*100)}": round(recall_at_precision(y[te], bits[te], f), 4)
                                                    for f in (0.95, 0.90, 0.85, 0.80)}}}
    by = {}
    for g in sorted({x for x in grp[te] if x}):
        m = te & ((y == 1) | (grp == g)); by[g] = metrics(y[m], bits[m], comp[m])
    res["test_by_negative_group"] = by
    if len(set(batch)) > 1:
        old = te & (batch == "v1")
        res["test_v1_rows_only"] = {**metrics(y[old], bits[old], comp[old]), "at_e_star": prf(y[old], (ev[old] <= e_star).astype(int))}
        new = te & (batch != "v1") & (y == 1)
        res["test_added_positives"] = {"n": int(new.sum()), "score_neg_log10_e": {i: round(float(b), 1) for i, b in zip(ids[new], bits[new])},
                                       "recalled_at_e_star": int((ev[new] <= e_star).sum())}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "blast_results.json").write_text(json.dumps(res, indent=2))
    np.savez(a.out / "blast_scores.npz", seq_ids=ids, bitscore=bits, evalue=ev, y=y, split=sp, component=comp, negative_group=grp)
    print("e* =", e_star, "| db", len(db))
    print("test blended:", json.dumps(res["test_blended"]))
    for g, v in by.items():
        print(f"  {g:36s} n_neg={v['n_neg']:3d} AU-ROC {v['au_roc']} AU-PRC {v['au_prc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
