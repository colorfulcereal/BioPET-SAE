"""Where do the family-associated latents peak? Sequence position, local motif and 3D location.

For each latent and each positive (and each hard negative on which it fires, activation >= 0.3):
  * peak residue, activation;
  * offset from the catalytic serine, estimated as (latent 9529 peak + 1), which hit the curated Ser in
    99% of held-out hard negatives; only proteins where 9529 fires are used for offsets;
  * the residues at peak -7 .. +7 (most common residue per position and its share).
Structures: IsPETase (5XFY) and LCC (4EB0): distance from the peak residue's CA to the CA of the catalytic
Ser, and the fraction of all residues closer to the Ser than that.

Usage:  uv run python -m biopet_sae.sae_peak_context_v12
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from interplm.sae.dictionary import ReLUSAE
from transformers import AutoTokenizer, EsmModel

from biopet_sae.make_splits import make_aligner

L = [3688, 7820, 2919, 7018, 1817, 5325]
STRUCT = {"BPS0282": ("IsPETase", "5XFY"), "BPS0269": ("LCC", "4EB0")}
THREE = dict(zip("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL MSE".split(), "ARNDCQEGHILKMFPSTWYVM"))


def ca_chain(path: Path):
    seq, xyz, seen, chain = "", [], set(), None
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"): break
        if not line.startswith(("ATOM", "HETATM")) or line[12:16].strip() != "CA" or line[17:20] not in THREE: continue
        if chain is None: chain = line[21]
        if line[21] != chain or line[22:27] in seen: continue
        seen.add(line[22:27]); seq += THREE[line[17:20]]; xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return seq, np.array(xyz)


def main() -> int:
    out = Path("results/v1_2/peak_context"); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t"))
    todo = [r for r in rows if (r["y"] == "1" or r["negative_group"] == "ab_hydrolase_hard_negative") and len(r["sequence"]) <= 1022]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sae = ReLUSAE.from_pretrained(hf_hub_download("Elana/InterPLM-esm2-650m", "layer_33/ae_normalized.pt")).to(dev).eval()
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(dev).eval()
    cols = L + [9529]
    rec = {}
    with torch.no_grad():
        for k, r in enumerate(todo):
            enc = {x: v.to(dev) for x, v in tok(r["sequence"], return_tensors="pt").items()}
            f = sae.encode(esm(**enc, output_hidden_states=True).hidden_states[33][0, 1:-1, :].float())[:, cols].cpu().numpy()
            rec[r["seq_id"]] = {l: (int(f[:, j].argmax()), float(f[:, j].max())) for j, l in enumerate(cols)}
            if (k + 1) % 200 == 0: print(f"  {k+1}/{len(todo)}", flush=True)
    seq = {r["seq_id"]: r["sequence"] for r in rows}
    kind = {r["seq_id"]: ("positive" if r["y"] == "1" else "hard negative") for r in todo}
    res = {}
    for l in L:
        res[l] = {}
        for g in ("positive", "hard negative"):
            ids = [i for i in rec if kind[i] == g and rec[i][l][1] >= 0.3]
            off = [rec[i][l][0] - (rec[i][9529][0] + 1) for i in ids if rec[i][9529][1] >= 0.3]
            win = [[] for _ in range(15)]
            for i in ids:
                p = rec[i][l][0]
                for d in range(-7, 8):
                    q = p + d; win[d + 7].append(seq[i][q] if 0 <= q < len(seq[i]) else "-")
            cons = [Counter(w).most_common(1)[0] for w in win] if ids else []
            res[l][g] = {"n_firing": len(ids), "n_group": sum(kind[i] == g for i in rec),
                         "offset_from_ser_median": float(np.median(off)) if off else None,
                         "offset_iqr": [float(np.percentile(off, 25)), float(np.percentile(off, 75))] if off else None,
                         "offset_top": Counter(off).most_common(3),
                         "consensus": "".join(a if n / len(ids) >= 0.5 else "." for a, n in cons) if ids else "",
                         "centre_residue": Counter(win[7]).most_common(3) if ids else [],
                         "window_conservation": [[a, round(n / len(ids), 2)] for a, n in cons]}
    al = make_aligner(); st = {}
    for sid, (name, pdb) in STRUCT.items():
        s = seq[sid]; pseq, xyz = ca_chain(Path(f"data/pdb_cache/{pdb}.pdb"))
        aln = al.align(s, pseq)[0]; m = {}
        for (a0, a1), (b0, b1) in zip(*aln.aligned):
            for i in range(a1 - a0): m[a0 + i] = b0 + i
        ser = rec[sid][9529][0] + 1
        if ser not in m: continue
        dist = np.linalg.norm(xyz - xyz[m[ser]], axis=1)
        st[name] = {"pdb": pdb, "serine_residue": f"{s[ser]}{ser+1}", "latents": {}}
        for l in L:
            p, act = rec[sid][l]
            st[name]["latents"][l] = {"residue": f"{s[p]}{p+1}", "activation": round(act, 2), "seq_offset_from_ser": p - ser,
                                      "CA_distance_A": round(float(dist[m[p]]), 1) if p in m else None,
                                      "frac_residues_closer": round(float((dist < dist[m[p]]).mean()), 2) if p in m else None}
    (out / "peak_context.json").write_text(json.dumps({"by_latent": res, "structures": st}, indent=2))
    for l in L:
        for g in ("positive", "hard negative"):
            v = res[l][g]
            print(f"{l:5d} {g:13s} fires {v['n_firing']:3d}/{v['n_group']}  offset from Ser median {v['offset_from_ser_median']} IQR {v['offset_iqr']} top {v['offset_top'][:2]}  centre {v['centre_residue']}\n        consensus(-7..+7) {v['consensus']}")
    for n, d in st.items():
        print(f"\n{n} ({d['pdb']}), catalytic Ser {d['serine_residue']}")
        for l, v in d["latents"].items(): print(f"  {l:5d} peak {v['residue']:6s} act {v['activation']:.2f}  seq offset {v['seq_offset_from_ser']:+4d}  CA distance {v['CA_distance_A']} A  (closer residues: {v['frac_residues_closer']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
