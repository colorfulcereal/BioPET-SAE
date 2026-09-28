"""Assemble data for the 3D structure viewer.

For each PET entry that has a solved structure, computes:
  * per-residue SAE activations for the selected latents (full profile, not just the peak)
  * the ESM-2 650M dense probe's confidence
  * the catalytic triad positions, located from the latents themselves and cross-checked
    against the GxSxG motif
  * a mapping from our sequence index to PDB residue numbers, by aligning our sequence to
    the structure's observed residues

Output feeds `artifacts/petase_structures.html`.

Usage:
    uv run python -m biopet_sae.build_structure_viewer_data
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from interplm.sae.dictionary import ReLUSAE
from transformers import AutoTokenizer, EsmModel

ESM = "facebook/esm2_t33_650M_UR50D"
SAE_REPO = "Elana/InterPLM-esm2-650m"
# the five latents that validated on held-out data, with their assignments
LATENTS = {
    9529: {"role": "catalytic Ser", "motif": "GxSxG nucleophile elbow", "offset": 1},
    7734: {"role": "catalytic His", "motif": "A-G-A-D-H-G", "offset": 1},
    2661: {"role": "catalytic Asp", "motif": "S-G-Q-A-D", "offset": 3},
    411: {"role": "Asp region (upstream)", "motif": "V-P-T-hydrophobic", "offset": None},
    2473: {"role": "His region (upstream)", "motif": "Y-L-E-V-A-G-A-D-H", "offset": None},
}
THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def fetch_pdb(pdb_id: str, cache: Path) -> str | None:
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{pdb_id.upper()}.pdb"
    if f.exists():
        return f.read_text()
    try:
        with urllib.request.urlopen(
            f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=60
        ) as r:
            txt = r.read().decode()
        f.write_text(txt)
        return txt
    except Exception as e:
        print(f"    could not fetch {pdb_id}: {type(e).__name__}")
        return None


def pdb_chain_residues(pdb_text: str) -> dict[str, list[tuple[int, str]]]:
    """{chain: [(resseq, one_letter), ...]} from CA atoms, in order."""
    out: dict[str, list[tuple[int, str]]] = {}
    seen: set[tuple[str, int]] = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        ch = line[21]
        try:
            rs = int(line[22:26])
        except ValueError:
            continue
        if (ch, rs) in seen:
            continue
        seen.add((ch, rs))
        aa = THREE2ONE.get(line[17:20].strip().upper())
        if aa:
            out.setdefault(ch, []).append((rs, aa))
    return out


def align_offset(our: str, pdb_seq: str) -> tuple[int, int]:
    """Find the shift that best matches pdb_seq onto our sequence.

    Structures usually omit the signal peptide, so the PDB sequence is a sub-segment of
    ours. Returns (best_shift, n_matches): pdb index i corresponds to our index i+shift.
    """
    best = (0, -1)
    for shift in range(-len(pdb_seq), len(our)):
        n = 0
        for i, c in enumerate(pdb_seq):
            j = i + shift
            if 0 <= j < len(our) and our[j] == c:
                n += 1
        if n > best[1]:
            best = (shift, n)
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged", type=Path, default=Path("data/processed/plastizymes_merged.tsv"))
    ap.add_argument("--splits", type=Path, default=Path("data/processed/dataset_splits_id40.tsv"))
    ap.add_argument("--probe", type=Path, default=Path("data/sae_650M_L33/probe_scores.npz"))
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--cache", type=Path, default=Path("data/pdb_cache"))
    ap.add_argument("--out", type=Path, default=Path("data/sae_650M_L33/structure_viewer.json"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    merged = {r["seq_id"]: r for r in csv.DictReader(args.merged.open(newline=""), delimiter="\t")}
    splits = {r["seq_id"]: r for r in csv.DictReader(args.splits.open(newline=""), delimiter="\t")}
    ps = np.load(args.probe, allow_pickle=True)
    score = {s: float(v) for s, v in zip(ps["seq_ids"].astype(str), ps["score"])}
    thr = float(ps["threshold"])

    targets = [r for r in merged.values() if r["class_label"] == "1_pet" and r["pdb"]]
    print(f"{len(targets)} PET entries with structures")

    dev = (torch.device(args.device) if args.device
           else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    sae = ReLUSAE.from_pretrained(
        hf_hub_download(SAE_REPO, f"layer_{args.layer}/ae_normalized.pt")).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(ESM)
    esm = EsmModel.from_pretrained(ESM).to(dev).eval()
    lat_ids = list(LATENTS)
    W = torch.tensor(lat_ids, device=dev)

    entries = []
    for n, r in enumerate(targets, 1):
        sid = r["seq_id"]
        seq = r["sequence"]
        pdbs = [p for p in (r["pdb"] or "").replace("|", ";").split(";") if p.strip()]
        if not pdbs:
            continue
        print(f"[{n}/{len(targets)}] {sid} {pdbs[0]} len={len(seq)}")

        with torch.no_grad():
            enc = {k: v.to(dev) for k, v in
                   tok(seq[:1022], return_tensors="pt", add_special_tokens=True).items()}
            h = esm(**enc, output_hidden_states=True).hidden_states[args.layer][0, 1:-1, :].float()
            acts = sae.encode(h)[:, W].cpu().numpy()      # (L, n_latents)

        # locate the triad from the latents, cross-checked against GxSxG
        peaks = {int(j): int(acts[:, i].argmax()) for i, j in enumerate(lat_ids)}
        ser = peaks[9529] + 1
        his = peaks[7734] + 1
        asp = peaks[2661] + 3  # +3 validated against LCC/IsPETase/Cut190 literature
        gx = [m.start() + 2 for m in re.finditer(r"(?=G.S.G)", seq)]
        ser_ok = bool(gx) and min(abs(ser - g) for g in gx) <= 1
        if gx and not ser_ok:
            ser = min(gx, key=lambda g: abs(g - ser))

        # structure + alignment
        pdb_id = pdbs[0].strip().upper()[:4]
        txt = fetch_pdb(pdb_id, args.cache)
        chains, chain, shift, nmatch = {}, None, 0, 0
        if txt:
            chains = pdb_chain_residues(txt)
            best = None
            for ch, res in chains.items():
                s2 = "".join(a for _, a in res)
                sh, nm = align_offset(seq, s2)
                if best is None or nm > best[2]:
                    best = (ch, sh, nm)
            if best:
                chain, shift, nmatch = best
        resmap = {}
        if chain:
            for i, (rs, aa) in enumerate(chains[chain]):
                j = i + shift
                if 0 <= j < len(seq):
                    resmap[str(j)] = rs

        entries.append({
            "seq_id": sid,
            "name": (r["protein_names"] or "").split("|")[0] or sid,
            "all_names": r["protein_names"],
            "organism": (r["organism"] or "").split("|")[0],
            "phylum": r["phylum"],
            "substrates": r["substrates"],
            "length": len(seq),
            "sequence": seq,
            "pdb_id": pdb_id,
            "pdb_chain": chain,
            "pdb_n_structures": int(r["n_pdb"]),
            "align_matches": nmatch,
            "align_coverage": round(nmatch / max(len(chains.get(chain, [])), 1), 3) if chain else 0,
            "resmap": resmap,
            "split": splits.get(sid, {}).get("split", "?"),
            "probe_score": round(score.get(sid, float("nan")), 4),
            "activations": {str(j): [round(float(x), 4) for x in acts[:, i]]
                            for i, j in enumerate(lat_ids)},
            "peaks": {str(j): int(p) for j, p in peaks.items()},
            "triad": {"ser": ser, "his": his, "asp": asp, "ser_confirmed_by_motif": ser_ok},
            "gxsxg_positions": gx,
        })

    payload = {
        "latents": {str(k): v for k, v in LATENTS.items()},
        "probe_threshold": thr,
        "display_threshold": 0.75,
        "layer": args.layer,
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    ok = sum(1 for e in entries if e["pdb_chain"])
    print(f"\n{len(entries)} entries, {ok} with a usable structure alignment")
    print(f"mean alignment coverage: "
          f"{np.mean([e['align_coverage'] for e in entries if e['pdb_chain']]):.3f}")
    print(f"catalytic Ser confirmed by GxSxG motif: "
          f"{sum(1 for e in entries if e['triad']['ser_confirmed_by_motif'])}/{len(entries)}")
    print(f"wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
