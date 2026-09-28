"""BLASTp baseline, scored continuously so it is comparable to the ESM-2 probe.

BLAST is used as a *ranker*, not as a hit/no-hit classifier: the database contains only
the training PET sequences, every val/test sequence is queried against it, and the score
is the best bit score (and -log10 e-value) of any hit. A sequence with no hit scores 0.
AU-ROC and AU-PRC then mean the same thing they do for the probe.

The database is train-PET only. Including val/test sequences, or including negatives,
would leak.

Reported per negative tier, because the overall number is dominated by the naive controls
for both methods and hides the comparison that matters. The informative question is
whether the probe beats BLAST specifically on the fold-matched tiers -- if both score
~0.99 against naive controls and ~0.64 against other polyester degraders, then the
language model is doing homology detection with extra steps.

Usage:
    uv run python -m biopet_sae.blast_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

CLASSES = ["1_pet", "2_other_polyester", "3_fold_matched_esterase", "4_naive_control"]
TIERS = ["2a_PBAT", "2b_aliphatic", "3_fold_matched_esterase", "4_naive_control"]


def require(*tools: str) -> None:
    missing = [t for t in tools if not shutil.which(t)]
    if missing:
        sys.exit(f"missing tool(s): {', '.join(missing)}. Install with `brew install blast`.")


def write_fasta(path: Path, records: dict[str, str]) -> None:
    with path.open("w") as fh:
        for sid, seq in records.items():
            fh.write(f">{sid}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")


def best_f1(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float, float]:
    p, r, t = precision_recall_curve(y, s)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    i = int(np.argmax(f))
    return float(f[i]), (float(t[i]) if i < len(t) else 1.0), float(p[i]), float(r[i])


def boot_ci(y, s, comps, fn, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(comps)
    by = {c: np.flatnonzero(comps == c) for c in uniq}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([by[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_blast(db_records, query_records, workdir: Path, evalue: float):
    """Return {query_id: (best_bitscore, best_evalue, best_pident, n_hits)}."""
    workdir.mkdir(parents=True, exist_ok=True)
    db_fa, q_fa = workdir / "db.fasta", workdir / "query.fasta"
    write_fasta(db_fa, db_records)
    write_fasta(q_fa, query_records)
    db = workdir / "petdb"
    subprocess.run(["makeblastdb", "-in", str(db_fa), "-dbtype", "prot", "-out", str(db)],
                   check=True, capture_output=True, text=True)
    out = workdir / "hits.tsv"
    subprocess.run(
        ["blastp", "-query", str(q_fa), "-db", str(db), "-out", str(out),
         "-outfmt", "6 qseqid sseqid evalue bitscore pident length",
         "-evalue", str(evalue), "-max_target_seqs", "5", "-num_threads", "4"],
        check=True, capture_output=True, text=True,
    )
    best: dict[str, tuple[float, float, float, int]] = {}
    with out.open() as fh:
        for line in fh:
            q, _s, ev, bits, pid, _ln = line.rstrip("\n").split("\t")
            ev, bits, pid = float(ev), float(bits), float(pid)
            cur = best.get(q)
            if cur is None or bits > cur[0]:
                best[q] = (bits, ev, pid, (cur[3] + 1) if cur else 1)
            else:
                best[q] = (cur[0], cur[1], cur[2], cur[3] + 1)
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path,
                    default=Path("data/processed/dataset_splits_id40.tsv"))
    ap.add_argument("--merged", type=Path,
                    default=Path("data/processed/plastizymes_merged.tsv"))
    ap.add_argument("--evalue", type=float, default=1000.0,
                    help="permissive BLAST run; the reported cutoff is tuned on val offline")
    ap.add_argument("--tune-on-val", action="store_true", default=True,
                    help="select the e-value cutoff by F1 on val, then apply it to test")
    ap.add_argument("--out", type=Path, default=Path("artifacts/BLAST_Baseline.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    require("makeblastdb", "blastp")

    rows = list(csv.DictReader(args.splits.open(newline=""), delimiter="\t"))
    grp = {r["seq_id"]: r["primary_group"]
           for r in csv.DictReader(args.merged.open(newline=""), delimiter="\t")}
    keep = [r for r in rows if r["class_label"] in CLASSES]
    seq = {r["seq_id"]: r["sequence"] for r in keep}
    lab = {r["seq_id"]: r["class_label"] for r in keep}
    spl = {r["seq_id"]: r["split"] for r in keep}
    cmp_ = {r["seq_id"]: r["component"] for r in keep}
    sub = {s: (("2a_PBAT" if grp.get(s) == "AROMATIC_COPOLYESTER" else "2b_aliphatic")
               if lab[s] == "2_other_polyester" else lab[s]) for s in seq}

    db = {s: q for s, q in seq.items() if spl[s] == "train" and lab[s] == "1_pet"}
    print(f"BLAST database: {len(db)} training PET sequences (train-PET only, no leakage)")

    workdir = Path(tempfile.mkdtemp(prefix="blast_"))
    val_threshold = None
    report: dict[str, object] = {"db_size": len(db), "evalue_cutoff": args.evalue, "splits": {}}
    try:
        for which in ("val", "test"):
            q = {s: v for s, v in seq.items() if spl[s] == which}
            hits = run_blast(db, q, workdir / which, args.evalue)
            ids = sorted(q)
            y = np.array([1 if lab[i] == "1_pet" else 0 for i in ids])
            comps = np.array([cmp_[i] for i in ids])
            st = np.array([sub[i] for i in ids])
            bits = np.array([hits.get(i, (0.0, 1e9, 0.0, 0))[0] for i in ids])
            pid = np.array([hits.get(i, (0.0, 1e9, 0.0, 0))[2] for i in ids])
            nohit = int((bits == 0).sum())
            print(f"\n=== {which}: n={len(ids)}, PET={int(y.sum())}, "
                  f"prevalence {y.mean():.3f}, no BLAST hit at all: {nohit} ===")
            print(f"  mean best bitscore: PET {bits[y==1].mean():.1f}  "
                  f"non-PET {bits[y==0].mean():.1f}")
            print(f"  mean best %identity: PET {pid[y==1].mean():.1f}  "
                  f"non-PET {pid[y==0].mean():.1f}")
            ar, apr = roc_auc_score(y, bits), average_precision_score(y, bits)
            lo_r, hi_r = boot_ci(y, bits, comps, roc_auc_score, seed=args.seed)
            lo_p, hi_p = boot_ci(y, bits, comps, average_precision_score, seed=args.seed)
            f1, thr, p_at, r_at = best_f1(y, bits)
            if which == "val":
                val_threshold = thr          # carried to test, mirroring the probe
            # on test, apply the VAL-selected threshold; the test-optimal one is an
            # oracle number and is reported separately, never as the headline
            applied = thr if which == "val" else val_threshold
            pred = (bits >= applied).astype(int)
            p_ap, r_ap, f_ap, _ = precision_recall_fscore_support(
                y, pred, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            print(f"  AU-ROC  {ar:.4f}  95% CI [{lo_r:.3f}, {hi_r:.3f}]")
            print(f"  AU-PRC  {apr:.4f}  95% CI [{lo_p:.3f}, {hi_p:.3f}]  floor {y.mean():.3f}")
            print(f"  at bitscore >= {applied:.1f} ({'val-selected' if which=='val' else 'carried from val'}):"
                  f"  Precision {p_ap:.4f}  Recall {r_ap:.4f}  F1 {f_ap:.4f}"
                  f"   TP={tp} FP={fp} FN={fn} TN={tn}")
            if which != "val":
                print(f"  [oracle, NOT a result] best F1 on test itself {f1:.4f} at bitscore >= {thr:.1f}")
            tiers = {}
            print(f"  {'tier':28s}{'n_neg':>7}{'AU-ROC':>9}{'AU-PRC':>9}")
            for t in TIERS:
                m = (st == "1_pet") | (st == t)
                if m.sum() < 4 or not 0 < y[m].sum() < m.sum():
                    continue
                tiers[t] = {"n_neg": int((st == t).sum()),
                            "AU-ROC": float(roc_auc_score(y[m], bits[m])),
                            "AU-PRC": float(average_precision_score(y[m], bits[m]))}
                print(f"  {t:28s}{tiers[t]['n_neg']:>7}{tiers[t]['AU-ROC']:>9.4f}"
                      f"{tiers[t]['AU-PRC']:>9.4f}")
            report["splits"][which] = {
                "n": len(ids), "pet": int(y.sum()), "prevalence": float(y.mean()),
                "no_hit": nohit,
                "mean_bitscore_pet": float(bits[y == 1].mean()),
                "mean_bitscore_nonpet": float(bits[y == 0].mean()),
                "mean_pident_pet": float(pid[y == 1].mean()),
                "mean_pident_nonpet": float(pid[y == 0].mean()),
                "AU-ROC": float(ar), "AU-ROC_ci": [lo_r, hi_r],
                "AU-PRC": float(apr), "AU-PRC_ci": [lo_p, hi_p],
                "threshold_bitscore_applied": float(applied),
                "threshold_source": "val-selected" if which == "val" else "carried from val",
                "precision": float(p_ap), "recall": float(r_ap), "f1": float(f_ap),
                "oracle_best_f1": f1, "oracle_threshold_bitscore": thr,
                "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                "tiers": tiers,
            }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
