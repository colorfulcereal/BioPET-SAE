"""Extract InterPLM SAE features from ESM-2 650M layer 33.

Pipeline per sequence:

    seq -> ESM-2 650M -> layer 33 per-residue (L x 1280)
        -> SAE.encode -> (L x 10240) sparse features -> pool over residues

The SAE operates per residue, so the pooled embeddings cached by `embed_esm2.py` cannot be
reused -- this needs a fresh forward pass.

Also saves the SAE *reconstruction* of the dense activations, pooled the same way, so the
reconstruction-fidelity gate can be run before any feature claim is trusted: if the probe
does not survive SAE round-tripping, downstream feature analysis is meaningless. This is
the kinase project's precedent.

SAEs are loaded with `ReLUSAE.from_pretrained` after `hf_hub_download` -- `interplm`'s
packaged `load_sae_from_hf` cannot work because the upstream package omits its `train`
subpackage.

Usage:
    uv run python -m biopet_sae.extract_sae_features --layer 33
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from interplm.sae.dictionary import ReLUSAE
from transformers import AutoTokenizer, EsmModel

SAE_REPO = {"650M": "Elana/InterPLM-esm2-650m", "8M": "Elana/InterPLM-esm2-8m"}
ESM_MODEL = {"650M": "facebook/esm2_t33_650M_UR50D", "8M": "facebook/esm2_t6_8M_UR50D"}


def pick_device(req: str | None) -> torch.device:
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
    ap.add_argument("--model", choices=sorted(SAE_REPO), default="650M")
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--weights", default="ae_normalized.pt",
                    help="ae_normalized.pt or ae_unnormalized.pt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-len", type=int, default=1022)
    args = ap.parse_args(argv)

    out_dir = args.out or Path(f"data/sae_{args.model}_L{args.layer}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    rows = list(csv.DictReader(args.splits.open(newline=""), delimiter="\t"))
    print(f"{len(rows)} sequences from {args.splits}")

    sae_path = hf_hub_download(SAE_REPO[args.model], f"layer_{args.layer}/ae_normalized.pt"
                               if args.weights == "ae_normalized.pt"
                               else f"layer_{args.layer}/{args.weights}")
    sae = ReLUSAE.from_pretrained(sae_path).to(device)
    sae.eval()
    d_in, d_sae = sae.encoder.weight.shape[1], sae.encoder.weight.shape[0]
    print(f"SAE layer {args.layer}: {d_in} -> {d_sae} features ({args.weights})")

    tok = AutoTokenizer.from_pretrained(ESM_MODEL[args.model])
    esm = EsmModel.from_pretrained(ESM_MODEL[args.model]).to(device)
    esm.eval()

    n = len(rows)
    feat_max = np.zeros((n, d_sae), dtype=np.float32)
    feat_mean = np.zeros((n, d_sae), dtype=np.float32)
    recon_max = np.zeros((n, d_in), dtype=np.float32)   # SAE round-trip, for the gate
    dense_max = np.zeros((n, d_in), dtype=np.float32)   # raw, same pass, for comparison
    n_active = np.zeros(n, dtype=np.int32)              # distinct latents ever active
    frac_active = np.zeros(n, dtype=np.float32)         # per-residue sparsity
    recon_cos = np.zeros(n, dtype=np.float32)           # per-residue reconstruction cosine
    truncated = 0
    t0 = time.time()

    with torch.no_grad():
        for i, row in enumerate(rows):
            seq = row["sequence"]
            if len(seq) > args.max_len:
                seq = seq[: args.max_len]
                truncated += 1
            enc = {k: v.to(device) for k, v in
                   tok(seq, return_tensors="pt", add_special_tokens=True).items()}
            h = esm(**enc, output_hidden_states=True).hidden_states[args.layer][0, 1:-1, :].float()
            f = sae.encode(h)                 # (L, d_sae)
            xh = sae.decode(f)                # (L, d_in)

            feat_max[i] = f.max(dim=0).values.cpu().numpy()
            feat_mean[i] = f.mean(dim=0).cpu().numpy()
            recon_max[i] = xh.max(dim=0).values.cpu().numpy()
            dense_max[i] = h.max(dim=0).values.cpu().numpy()
            n_active[i] = int((f > 0).any(dim=0).sum().item())
            frac_active[i] = float((f > 0).float().mean().item())
            recon_cos[i] = float(
                torch.nn.functional.cosine_similarity(h, xh, dim=-1).mean().item()
            )
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{n}  {rate:.1f} seq/s  eta {(n-i-1)/rate:.0f}s")

    path = out_dir / "sae_features.npz"
    np.savez_compressed(
        path,
        seq_ids=np.array([r["seq_id"] for r in rows]),
        class_label=np.array([r["class_label"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        phylum=np.array([r["phylum"] for r in rows]),
        component=np.array([r["component"] for r in rows]),
        feat_max=feat_max, feat_mean=feat_mean,
        recon_max=recon_max, dense_max=dense_max,
        n_active=n_active, frac_active=frac_active, recon_cos=recon_cos,
    )
    meta = {
        "esm_model": ESM_MODEL[args.model], "sae_repo": SAE_REPO[args.model],
        "layer": args.layer, "weights": args.weights,
        "d_in": int(d_in), "d_sae": int(d_sae), "n_sequences": n,
        "truncated": truncated, "device": str(device),
        "mean_frac_active_per_residue": float(frac_active.mean()),
        "mean_latents_active_per_sequence": float(n_active.mean()),
        "mean_reconstruction_cosine": float(recon_cos.mean()),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "sae_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
