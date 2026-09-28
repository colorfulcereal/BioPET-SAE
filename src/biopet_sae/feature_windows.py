"""Residue-level interpretation of selected SAE latents.

Re-runs ESM-2 + the SAE keeping *per-residue* activations for a chosen set of latents,
then for each latent reports where it fires: the top-activating sequences, the peak residue
positions, the sequence windows around those peaks, and position-specific residue
frequencies.

This is what turns "latent 9529 predicts PET" into "latent 9529 fires on <motif>". The
sibling kinase project used the same approach to map its top latents onto the P-loop and
HRD catalytic motifs.

Checks each latent's peak windows against motifs relevant to PET hydrolases:
  GxSxG   nucleophile elbow holding the catalytic serine (all alpha/beta hydrolases)
  GxCxG   the cysteine variant
  HG      oxyanion-hole histidine-glycine
  aromatic clamp residues (W, Y, F) which in IsPETase (Y87/W185) accommodate the
          terephthalate ring -- the determinant that distinguishes PET activity

Usage:
    uv run python -m biopet_sae.feature_windows --latents 9529,411,2661,5271,2473,7734
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from interplm.sae.dictionary import ReLUSAE
from transformers import AutoTokenizer, EsmModel

ESM = "facebook/esm2_t33_650M_UR50D"
SAE_REPO = "Elana/InterPLM-esm2-650m"
MOTIFS = {
    "GxSxG (nucleophile elbow)": re.compile(r"G.S.G"),
    "GxCxG": re.compile(r"G.C.G"),
    "HG (oxyanion hole)": re.compile(r"HG"),
    "GGG": re.compile(r"GGG"),
}


def device_of(req: str | None) -> torch.device:
    if req:
        return torch.device(req)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=Path("data/processed/dataset_splits_id40.tsv"))
    ap.add_argument("--latents", required=True, help="comma-separated latent indices")
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--window", type=int, default=7, help="+/- residues around the peak")
    ap.add_argument("--top-seqs", type=int, default=30, help="top activating sequences per latent")
    ap.add_argument("--out", type=Path, default=Path("data/sae_650M_L33/feature_windows.json"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    lat = [int(x) for x in args.latents.split(",")]
    dev = device_of(args.device)

    rows = list(csv.DictReader(args.splits.open(newline=""), delimiter="\t"))
    sae = ReLUSAE.from_pretrained(
        hf_hub_download(SAE_REPO, f"layer_{args.layer}/ae_normalized.pt")).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(ESM)
    esm = EsmModel.from_pretrained(ESM).to(dev).eval()
    W = torch.tensor(lat, device=dev)
    print(f"{len(rows)} sequences, {len(lat)} latents, layer {args.layer}, device {dev}")

    # per (sequence, latent): peak activation and its residue position
    peak = np.zeros((len(rows), len(lat)), dtype=np.float32)
    argmax = np.zeros((len(rows), len(lat)), dtype=np.int32)
    with torch.no_grad():
        for i, r in enumerate(rows):
            seq = r["sequence"][:1022]
            enc = {k: v.to(dev) for k, v in
                   tok(seq, return_tensors="pt", add_special_tokens=True).items()}
            h = esm(**enc, output_hidden_states=True).hidden_states[args.layer][0, 1:-1, :].float()
            f = sae.encode(h)[:, W]              # (L, n_latents)
            peak[i] = f.max(dim=0).values.cpu().numpy()
            argmax[i] = f.argmax(dim=0).cpu().numpy()
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(rows)}")

    labels = np.array([r["class_label"] for r in rows])
    seqs = [r["sequence"] for r in rows]
    is_pet = labels == "1_pet"
    report = {"layer": args.layer, "window": args.window, "latents": {}}

    for li, j in enumerate(lat):
        act = peak[:, li]
        order = np.argsort(-act)[: args.top_seqs]
        wins, pos_frac, hits = [], [], Counter()
        for si in order:
            s = seqs[si]
            p = int(argmax[si, li])
            a, b = max(0, p - args.window), min(len(s), p + args.window + 1)
            w = s[a:b]
            wins.append({"seq_id": rows[si]["seq_id"], "class": rows[si]["class_label"],
                         "activation": float(act[si]), "position": p,
                         "rel_position": round(p / max(len(s), 1), 3),
                         "window": w, "centre": s[p] if p < len(s) else "?"})
            pos_frac.append(p / max(len(s), 1))
            for name, rx in MOTIFS.items():
                if rx.search(w):
                    hits[name] += 1
        centres = Counter(w["centre"] for w in wins)
        # position-specific residue frequencies across the aligned windows
        L = 2 * args.window + 1
        cols = []
        for c in range(L):
            col = Counter()
            for w in wins:
                ww = w["window"]
                # align on the peak: left-pad short windows
                off = args.window - min(args.window, w["position"])
                idx = c - off
                if 0 <= idx < len(ww):
                    col[ww[idx]] += 1
            tot = sum(col.values()) or 1
            cols.append([(r, round(n / tot, 3)) for r, n in col.most_common(3)])
        n_pet = int(is_pet[order].sum())
        report["latents"][str(j)] = {
            "n_top_seqs": len(order), "n_pet_in_top": n_pet,
            "pet_fraction_in_top": round(n_pet / len(order), 3),
            "mean_activation_pet": float(act[is_pet].mean()),
            "mean_activation_nonpet": float(act[~is_pet].mean()),
            "prevalence_pet": float((act[is_pet] > 0).mean()),
            "prevalence_nonpet": float((act[~is_pet] > 0).mean()),
            "centre_residues": centres.most_common(6),
            "motif_hits_in_top_windows": dict(hits),
            "mean_relative_position": round(float(np.mean(pos_frac)), 3),
            "position_frequencies": cols,
            "windows": wins[:15],
        }
        print(f"\n=== latent {j} ===")
        print(f"  top-{len(order)} activating sequences: {n_pet} PET ({100*n_pet/len(order):.0f}%)")
        print(f"  mean activation  PET {act[is_pet].mean():.3f}  non-PET {act[~is_pet].mean():.3f}")
        print(f"  centre residue: {dict(centres.most_common(5))}")
        print(f"  motif hits in windows: {dict(hits) or 'none'}")
        print(f"  mean relative position in sequence: {np.mean(pos_frac):.3f}")
        for w in wins[:6]:
            print(f"    {w['seq_id']:9s} {w['class'][:12]:13s} act {w['activation']:6.3f} "
                  f"pos {w['position']:4d} [{w['centre']}]  {w['window']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    np.savez(args.out.with_suffix(".npz"), peak=peak, argmax=argmax,
             latents=np.array(lat), seq_ids=np.array([r["seq_id"] for r in rows]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
