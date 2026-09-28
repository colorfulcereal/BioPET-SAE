"""Differential scoring of SAE latents, per negative tier.

Implements the research plan's Stage-1 statistic

    S_j = (mu_pet,j - mu_control,j) / (sigma_control,j + eps)     eps = 1e-8

and a corrected variant, because the plan's form breaks on sparse features: SAE latents are
frequently exactly 0 across every control, so sigma_control = 0 and S_j ~ mu_pet * 1e8. The
raw ranking is then dominated by latents firing in one or two sequences. The corrected form
uses a variance floor and a minimum-prevalence filter:

    S'_j = (mu_pet,j - mu_control,j) / (sigma_control,j + floor)
    floor = median(sigma over all sequences),  latent must fire in >= min_prev of PETases

Both rankings are reported. A latent present only in the raw ranking is an artifact of the
epsilon, not a finding.

Scores are computed separately against each negative tier, because pooling them answers the
wrong question (see D8 in PROGRESS.md):

    vs 4_naive_control          -> "is this an esterase at all"
    vs 3_fold_matched_esterase  -> "is this a polyester hydrolase"
    vs 2b_aliphatic             -> PET specificity, the unsolved contrast

Usage:
    uv run python -m biopet_sae.sae_differential --top 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CLASSES = ["1_pet", "2_other_polyester", "3_fold_matched_esterase", "4_naive_control"]
TIERS = ["2a_PBAT", "2b_aliphatic", "3_fold_matched_esterase", "4_naive_control"]


def subtiers(splits: Path, merged: Path) -> dict[str, str]:
    grp = {r["seq_id"]: r["primary_group"]
           for r in csv.DictReader(merged.open(newline=""), delimiter="\t")}
    out = {}
    for r in csv.DictReader(splits.open(newline=""), delimiter="\t"):
        c = r["class_label"]
        if c == "2_other_polyester":
            c = "2a_PBAT" if grp.get(r["seq_id"]) == "AROMATIC_COPOLYESTER" else "2b_aliphatic"
        out[r["seq_id"]] = c
    return out


def differential(pos: np.ndarray, ctl: np.ndarray, eps: float, floor: float | None):
    """Return (S_raw, S_corrected, prevalence_in_pos, prevalence_in_ctl)."""
    mu_p, mu_c = pos.mean(0), ctl.mean(0)
    sd_c = ctl.std(0)
    s_raw = (mu_p - mu_c) / (sd_c + eps)
    s_cor = (mu_p - mu_c) / (sd_c + (floor if floor is not None else eps))
    return s_raw, s_cor, (pos > 0).mean(0), (ctl > 0).mean(0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path,
                    default=Path("data/sae_650M_L33/sae_features.npz"))
    ap.add_argument("--splits", type=Path,
                    default=Path("data/processed/dataset_splits_id40.tsv"))
    ap.add_argument("--merged", type=Path,
                    default=Path("data/processed/plastizymes_merged.tsv"))
    ap.add_argument("--probe-scores", type=Path,
                    default=Path("data/sae_650M_L33/probe_scores.npz"),
                    help="per-sequence probe P(PET) and prediction, for the correct-only filter")
    ap.add_argument("--pooling", choices=["max", "mean"], default="max")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--min-prevalence", type=float, default=0.10)
    ap.add_argument("--correct-only", action="store_true", default=True)
    ap.add_argument("--all-sequences", dest="correct_only", action="store_false")
    # Selection must not see val/test. Computing the statistic over all splits and then
    # evaluating the chosen latents on test is circular: it inflated the reported AU-ROC
    # from 0.5108 to 0.6129 and changed 10 of 13 candidates.
    ap.add_argument("--select-on", choices=["train", "all"], default="train",
                    help="which splits the differential statistic may use (default train)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/SAE_650M_L33_differential.json"))
    args = ap.parse_args(argv)

    d = np.load(args.features, allow_pickle=True)
    ids = d["seq_ids"].astype(str)
    lab = d["class_label"].astype(str)
    F = d[f"feat_{args.pooling}"]
    sub = np.array([subtiers(args.splits, args.merged).get(i, "") for i in ids])
    print(f"{F.shape[0]} sequences x {F.shape[1]} latents ({args.pooling}-pooled)")
    print(f"mean latents active per sequence: {d['n_active'].mean():.0f} "
          f"({100*d['n_active'].mean()/F.shape[1]:.1f}% of {F.shape[1]})")
    print(f"mean per-residue reconstruction cosine: {d['recon_cos'].mean():.4f}")

    split = d["split"].astype(str)
    keep = np.ones(len(ids), bool)
    if args.select_on == "train":
        keep &= (split == "train")
        print("selection restricted to the TRAIN split (val/test never seen)")
    if args.correct_only and args.probe_scores.exists():
        ps = np.load(args.probe_scores, allow_pickle=True)
        pid = {s: i for i, s in enumerate(ps["seq_ids"].astype(str))}
        correct = np.array([bool(ps["correct"][pid[s]]) if s in pid else False for s in ids])
        keep = keep & correct
        print(f"restricting to sequences the probe classifies correctly: "
              f"{keep.sum()}/{len(keep)}")
    elif args.correct_only:
        print(f"WARNING: {args.probe_scores} missing -- using all sequences")

    pos_mask = keep & (lab == "1_pet")
    pos = F[pos_mask]
    floor = float(np.median(F.std(0)))
    print(f"variance floor (median sd across all sequences): {floor:.6g}")
    print(f"PET sequences used: {pos.shape[0]}")

    report = {"pooling": args.pooling, "eps": args.eps, "variance_floor": floor,
              "min_prevalence": args.min_prevalence, "correct_only": bool(args.correct_only),
              "n_pet_used": int(pos.shape[0]), "tiers": {}}

    for tier in TIERS:
        ctl_mask = keep & (sub == tier)
        if ctl_mask.sum() < 5:
            print(f"\n{tier}: only {ctl_mask.sum()} sequences -- skipped")
            continue
        ctl = F[ctl_mask]
        s_raw, s_cor, prev_p, prev_c = differential(pos, ctl, args.eps, floor)
        eligible = prev_p >= args.min_prevalence
        s_cor_f = np.where(eligible, s_cor, -np.inf)
        top_raw = np.argsort(-s_raw)[: args.top]
        top_cor = np.argsort(-s_cor_f)[: args.top]
        overlap = len(set(top_raw.tolist()) & set(top_cor.tolist()))
        zero_var = int((ctl.std(0) == 0).sum())
        print(f"\n=== PET (n={pos.shape[0]}) vs {tier} (n={ctl_mask.sum()}) ===")
        print(f"  latents with zero variance in controls: {zero_var} "
              f"({100*zero_var/F.shape[1]:.1f}%)  <- these blow up the raw statistic")
        print(f"  latents passing the >={args.min_prevalence:.0%} PET-prevalence filter: "
              f"{int(eligible.sum())}")
        print(f"  overlap between raw and corrected top-{args.top}: {overlap}/{args.top}")
        print(f"  {'rank':>4}{'latent':>8}{'S_corrected':>13}{'S_raw':>12}"
              f"{'prev_PET':>10}{'prev_ctl':>10}{'mu_PET':>9}{'mu_ctl':>9}")
        rows = []
        for r, j in enumerate(top_cor, 1):
            rows.append(dict(rank=r, latent=int(j), S_corrected=float(s_cor[j]),
                             S_raw=float(s_raw[j]), prev_pet=float(prev_p[j]),
                             prev_ctl=float(prev_c[j]), mu_pet=float(pos[:, j].mean()),
                             mu_ctl=float(ctl[:, j].mean())))
            if r <= 15:
                print(f"  {r:>4}{j:>8}{s_cor[j]:>13.3f}{s_raw[j]:>12.3g}"
                      f"{prev_p[j]:>10.2f}{prev_c[j]:>10.2f}"
                      f"{pos[:, j].mean():>9.3f}{ctl[:, j].mean():>9.3f}")
        report["tiers"][tier] = {
            "n_control": int(ctl_mask.sum()), "zero_variance_latents": zero_var,
            "n_eligible": int(eligible.sum()), "raw_corrected_overlap": overlap,
            "top_corrected": rows,
            "top_raw": [int(j) for j in top_raw],
        }

    # which latents are shared across tiers -- shared = generic, unique to 2b = PET-specific
    if len(report["tiers"]) >= 2:
        sets = {t: {r["latent"] for r in v["top_corrected"]}
                for t, v in report["tiers"].items()}
        print(f"\n=== overlap of top-{args.top} corrected latents between tiers ===")
        ts = list(sets)
        print(f"  {'':28s}" + "".join(f"{t.split('_')[0]:>10s}" for t in ts))
        for a in ts:
            print(f"  {a:28s}" + "".join(f"{len(sets[a] & sets[b]):>10d}" for b in ts))
        if "2b_aliphatic" in sets:
            others = set().union(*[v for t, v in sets.items() if t != "2b_aliphatic"])
            uniq = sorted(sets["2b_aliphatic"] - others)
            print(f"\n  latents in the 2b top-{args.top} and NO other tier: {len(uniq)}")
            print(f"    {uniq}")
            report["pet_specific_candidates"] = uniq

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
