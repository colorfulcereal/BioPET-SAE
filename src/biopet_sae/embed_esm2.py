"""Extract ESM-2 embeddings for the split dataset.

Uses HuggingFace `transformers` (`EsmModel`/`AutoTokenizer`), not `fair-esm`: InterPLM's
own embedder uses HuggingFace, and the pretrained SAEs are only guaranteed compatible
with activations produced the same way.

Extracts at the layers where InterPLM ships SAEs so the same tensors serve the later
SAE work without a second forward pass:
  esm2_t33_650M_UR50D -> layers 1, 9, 18, 24, 30, 33   (1280 dim, 10240 SAE features)
  esm2_t6_8M_UR50D    -> layers 1-6                    (320 dim, 10240 SAE features)

Both mean- and max-pooled vectors are saved. Max-pooling matters here: the catalytic
signal is a handful of residues out of ~300, and mean-pooling divides it by the length.

CLS and EOS tokens are stripped before pooling.

Usage:
    uv run python -m biopet_sae.embed_esm2 --model 650M
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
from transformers import AutoTokenizer, EsmModel

MODELS = {
    "650M": {
        "hf": "facebook/esm2_t33_650M_UR50D",
        "sae_layers": [1, 9, 18, 24, 30, 33],
        "dim": 1280,
    },
    "8M": {
        "hf": "facebook/esm2_t6_8M_UR50D",
        "sae_layers": [1, 2, 3, 4, 5, 6],
        "dim": 320,
    },
}


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=Path("data/processed/dataset_splits.tsv"))
    ap.add_argument("--model", choices=sorted(MODELS), default="650M")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: data/embeddings_<model>/")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-len", type=int, default=1022)
    args = ap.parse_args(argv)

    spec = MODELS[args.model]
    out_dir = args.out or Path(f"data/embeddings_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    with args.splits.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    print(f"{len(rows)} sequences from {args.splits}")
    print(f"model {spec['hf']} on {device}, layers {spec['sae_layers']}")

    tokenizer = AutoTokenizer.from_pretrained(spec["hf"])
    model = EsmModel.from_pretrained(spec["hf"]).to(device)
    model.eval()

    layers = spec["sae_layers"]
    n, dim = len(rows), spec["dim"]
    mean_pooled = {ell: np.zeros((n, dim), dtype=np.float32) for ell in layers}
    max_pooled = {ell: np.zeros((n, dim), dtype=np.float32) for ell in layers}
    truncated = 0
    t0 = time.time()

    with torch.no_grad():
        for i, row in enumerate(rows):
            seq = row["sequence"]
            if len(seq) > args.max_len:
                seq = seq[: args.max_len]
                truncated += 1
            enc = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            for ell in layers:
                # hidden_states[0] is the embedding layer, so index ell is transformer
                # layer ell. Drop CLS (0) and EOS (-1).
                h = out.hidden_states[ell][0, 1:-1, :].float()
                mean_pooled[ell][i] = h.mean(dim=0).cpu().numpy()
                max_pooled[ell][i] = h.max(dim=0).values.cpu().numpy()
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{n}  {rate:.1f} seq/s  eta {(n-i-1)/rate:.0f}s")

    payload = {
        "seq_ids": np.array([r["seq_id"] for r in rows]),
        "class_label": np.array([r["class_label"] for r in rows]),
        "split": np.array([r["split"] for r in rows]),
        "phylum": np.array([r["phylum"] for r in rows]),
        "component": np.array([r["component"] for r in rows]),
        "seq_len": np.array([int(r["seq_len"]) for r in rows]),
    }
    for ell in layers:
        payload[f"mean_L{ell}"] = mean_pooled[ell]
        payload[f"max_L{ell}"] = max_pooled[ell]

    path = out_dir / "embeddings.npz"
    np.savez_compressed(path, **payload)
    meta = {
        "model": spec["hf"], "layers": layers, "dim": dim, "n_sequences": n,
        "pooling": ["mean", "max"], "truncated_over_max_len": truncated,
        "max_len": args.max_len, "device": str(device),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "embedding_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{json.dumps(meta, indent=2)}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
