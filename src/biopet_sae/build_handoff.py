"""Build the hand-off package: clean labelled data plus an agent-readable spec.

Produces a self-contained directory a second team can drop into a fresh repository and
reproduce the experiment from, without access to this one.

Usage:
    uv run python -m biopet_sae.build_handoff
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

POS = {"1_pet", "2_other_polyester"}

# split values are rewritten so nothing implies a claim the data does not support
SPLIT_OUT = {"train": "train", "val": "val", "test": "test", "heldout": "unused"}

SCHEMA = {
    "seq_id": "Stable identifier. BPS* for positives (internal), UniProt accession for negatives.",
    "split": "train | val | test | unused. 'unused' rows are PHA depolymerases that carry no claim; they are not a held-out test and were never used for training, tuning or evaluation.",
    "label": "1 = polyester degrader, 0 = negative.",
    "negative_group": "For negatives only. ab_hydrolase_hard_negative | ec311_wrong_fold | trna_processing | naive_control. USE THIS, not class_label.",
    "class_label": "Legacy label kept only for cross-referencing published results. '3_fold_matched_esterase' is a misnomer: only 50 of its 185 members carry the fold.",
    "component": "Homology component id. Sequences in one component share >=40% identity. BOOTSTRAP OVER THIS COLUMN, never over individual proteins.",
    "cluster": "Finer identity cluster inside a component.",
    "substrates": "Pipe-separated plastics the enzyme is reported to degrade (positives only).",
    "activity_verified": "1 = degradation measured in the literature; 0 = reported but not independently confirmed.",
    "ab_hydrolase_fold": "1 if UniProt/InterPro IPR029058 (alpha/beta hydrolase fold). Blank where no UniProt accession exists.",
    "gxsxg_elbow": "1 if the sequence contains the G-x-[ST]-x-G nucleophile elbow motif.",
    "n_curated_active_sites": "Count of curated UniProt ACT_SITE residues (negatives only).",
    "ec": "UniProt EC numbers (negatives only).",
    "pfam": "Pfam accessions (negatives only).",
    "organism": "Source organism.",
    "phylum": "Source phylum. Positives are 72.6% Actinomycetota; negatives were drawn to match.",
    "seq_len": "Sequence length in residues.",
    "sequence": "Amino-acid sequence, standard 20 letters.",
}

COLS = list(SCHEMA)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path,
                    default=Path("data/release/biopet_sae_dataset_v1.tsv"))
    ap.add_argument("--out", type=Path, default=Path("handoff"))
    args = ap.parse_args(argv)

    rows = list(csv.DictReader(args.release.open(newline=""), delimiter="\t"))
    args.out.mkdir(parents=True, exist_ok=True)

    GRP = {"ab_hydrolase_hard_negative": "ab_hydrolase_hard_negative",
           "ec311_wrong_fold": "ec311_wrong_fold",
           "tRNA_processing_no_ser_nucleophile": "trna_processing",
           "naive_control": "naive_control", "": ""}

    out_rows = []
    for r in rows:
        out_rows.append({
            "seq_id": r["seq_id"],
            "split": SPLIT_OUT[r["split"]],
            "label": r["y"],
            "negative_group": GRP[r["negative_group"]],
            "class_label": r["class_label"],
            "component": r["component"],
            "cluster": r["cluster"],
            "substrates": r["substrates"],
            "activity_verified": r["activity_verified"],
            "ab_hydrolase_fold": r["ab_hydrolase_fold"],
            "gxsxg_elbow": r["gxsxg_elbow"],
            "n_curated_active_sites": r["n_curated_active_sites"],
            "ec": r["ec"], "pfam": r["pfam"],
            "organism": r["organism"], "phylum": r["phylum"],
            "seq_len": r["seq_len"], "sequence": r["sequence"],
        })

    tsv = args.out / "dataset.tsv"
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    fa = args.out / "dataset.fasta"
    with fa.open("w") as fh:
        for r in out_rows:
            fh.write(f">{r['seq_id']} split={r['split']} label={r['label']}"
                     f" group={r['negative_group'] or 'positive'} component={r['component']}\n")
            s = r["sequence"]
            for i in range(0, len(s), 60):
                fh.write(s[i:i + 60] + "\n")

    def count(**kw):
        return sum(1 for r in out_rows if all(r[k] == v for k, v in kw.items()))

    manifest = {
        "name": "polyester-degrader-benchmark",
        "version": "1.0",
        "built": "2026-10-11",
        "task": "binary classification: does this protein degrade a polyester?",
        "counts": {
            "rows_total": len(out_rows),
            "modelled": sum(1 for r in out_rows if r["split"] != "unused"),
            "unused": count(split="unused"),
            "positives": sum(1 for r in out_rows if r["label"] == "1" and r["split"] != "unused"),
            "negatives": sum(1 for r in out_rows if r["label"] == "0"),
            "by_split": {s: count(split=s) for s in ("train", "val", "test", "unused")},
            "negatives_by_group": {
                g: sum(1 for r in out_rows if r["negative_group"] == g)
                for g in ("ab_hydrolase_hard_negative", "ec311_wrong_fold",
                          "trna_processing", "naive_control")},
            "hard_negatives_by_split": {
                s: sum(1 for r in out_rows
                       if r["negative_group"] == "ab_hydrolase_hard_negative" and r["split"] == s)
                for s in ("train", "val", "test")},
        },
        "constraints": {
            "max_cross_split_identity": {"train_val": 0.3967, "train_test": 0.3800,
                                         "val_test": 0.3983},
            "hard_negatives_total": 51,
            "hard_negatives_in_val": 1,
            "largest_positive_component": {"id": "BPS0190", "n": 256, "fraction": 0.66},
            "positive_components": 74,
        },
        "baselines_to_reproduce": {
            "probe": {
                "model": "facebook/esm2_t33_650M_UR50D",
                "representation": "max-pool over residues, hidden_states[33]",
                "classifier": "sklearn SGDClassifier(loss='log_loss', penalty='l2', "
                              "alpha=1/(1e-2*n_train), learning_rate='constant', eta0=1e-4, "
                              "max_iter=5000, tol=None, random_state=0)",
                "scaler": "StandardScaler fit on train only",
                "threshold": 0.1455,
                "threshold_selected_on": "val log-loss",
                "test_au_roc": 0.9681, "test_au_prc": 0.9526,
                "val_au_roc": 0.9976, "val_au_prc": 0.9954,
                "test_by_negative_group": {
                    "naive_control": {"n": 36, "au_roc": 1.0000},
                    "trna_processing": {"n": 3, "au_roc": 1.0000},
                    "ec311_wrong_fold": {"n": 23, "au_roc": 0.9141},
                    "ab_hydrolase_hard_negative": {"n": 10, "au_roc": 0.8167,
                                                   "ci95": [0.676, 0.926]},
                },
            },
            "blastp": {"test_au_roc": 0.9040, "test_au_prc": 0.7552},
            "aa_composition": {"test_au_roc": 0.7020,
                               "random_cv_au_roc": 0.8926,
                               "note": "the 0.19 gap is what a non-homology-aware split buys"},
        },
        "files": {},
    }
    for f in (tsv, fa):
        manifest["files"][f.name] = {"sha256": sha256(f), "bytes": f.stat().st_size}

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "schema.json").write_text(json.dumps(
        {"columns": SCHEMA, "order": COLS}, indent=2) + "\n")

    print(f"wrote {args.out}/ -> dataset.tsv, dataset.fasta, manifest.json, schema.json")
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
