"""Score the 30k metagenomic screening set: probe, catalytic triad, substrate pocket.

No BLAST. Writes incrementally so progress survives interruption.

Usage:
    uv run python -m biopet_sae.score_screen30k
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
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, EsmModel

POS = {"1_pet", "2_other_polyester"}
NEG = ["3_fold_matched_esterase", "4_naive_control"]
TRIAD = {"9529": ("Ser", 1, "S"), "2661": ("Asp", 3, "DE"), "7734": ("His", 1, "H")}
POCKET = [7942, 5201, 5255]
LAT = [9529, 2661, 7734, 7942, 5201, 5255]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", type=Path, default=Path("data/mgnify/screen30k_set.tsv"))
    ap.add_argument("--splits", type=Path,
                    default=Path("data/processed/dataset_splits_id40.tsv"))
    ap.add_argument("--emb", type=Path, default=Path("data/embeddings_650M/embeddings.npz"))
    ap.add_argument("--out", type=Path, default=Path("data/mgnify/screen30k_scored.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    cand = list(csv.DictReader(args.set.open(newline=""), delimiter="\t"))
    if args.limit:
        cand = cand[: args.limit]
    sp = {r["seq_id"]: r for r in csv.DictReader(args.splits.open(newline=""), delimiter="\t")}

    # ---- refit the shipped probe on train only -------------------------------------
    d = np.load(args.emb, allow_pickle=True)
    eids = d["seq_ids"].astype(str)
    X = d["max_L33"]
    lab = np.array([sp[i]["class_label"] if i in sp else "" for i in eids])
    spl = np.array([sp[i]["split"] if i in sp else "" for i in eids])
    m = np.isin(lab, list(POS) + NEG)
    tr = m & (spl == "train")
    y = np.isin(lab, list(POS))
    scaler = StandardScaler().fit(X[tr])
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1 / (1e-2 * tr.sum()),
                        learning_rate="constant", eta0=1e-4, max_iter=5000,
                        random_state=0, tol=None).fit(scaler.transform(X[tr]), y[tr])
    print(f"probe refit on {int(tr.sum())} train sequences", flush=True)

    dev = (torch.device("mps") if torch.backends.mps.is_available()
           else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    sae = ReLUSAE.from_pretrained(hf_hub_download(
        "Elana/InterPLM-esm2-650m", "layer_33/ae_normalized.pt")).to(dev).eval()
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(dev).eval()
    W = torch.tensor(LAT, device=dev)
    print(f"device {dev}; scoring {len(cand):,} candidates", flush=True)

    # resume support
    done = set()
    if args.out.exists():
        for ln in args.out.open():
            try:
                done.add(json.loads(ln)["cand_id"])
            except Exception:
                pass
        print(f"resuming — {len(done):,} already scored", flush=True)

    t0 = time.time()
    with args.out.open("a") as fh:
        for i, r in enumerate(cand, 1):
            if r["cand_id"] in done:
                continue
            s = r["sequence"][:1022]
            with torch.no_grad():
                enc = {k: v.to(dev) for k, v in tok(s, return_tensors="pt").items()}
                h = esm(**enc, output_hidden_states=True).hidden_states[33][0, 1:-1, :].float()
                emb = h.max(dim=0).values.cpu().numpy()
                A = sae.encode(h)[:, W].cpu().numpy()
            p = float(clf.predict_proba(scaler.transform(emb[None]))[0, 1])
            rec = {"cand_id": r["cand_id"], "study": r["study"], "length": int(r["length"]),
                   "elbow": r["elbow"],
                   "plastizyme_elbow": int(r["plastizyme_style_elbow"]),
                   "probe": round(p, 4)}
            hits = 0
            for j, (role, off, want) in TRIAD.items():
                col = LAT.index(int(j))
                pk = int(A[:, col].argmax())
                pred = pk + off
                ok = 0 <= pred < len(s) and s[pred] in want
                rec[f"{j}_peak"] = pk
                rec[f"{j}_act"] = round(float(A[pk, col]), 3)
                rec[f"{j}_res"] = s[pred] if 0 <= pred < len(s) else "?"
                rec[f"{j}_ok"] = int(ok)
                hits += ok
            rec["triad_hits"] = hits
            rec["pocket_act"] = round(float(np.mean(
                [A[:, LAT.index(q)].max() for q in POCKET])), 3)
            if p >= 0.3 or hits == 3:
                rec["sequence"] = s
            fh.write(json.dumps(rec) + "\n")
            if i % 500 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"   {i:,}/{len(cand):,}  {el/60:.1f} min  "
                      f"eta {(len(cand)-i)*el/max(i,1)/60:.0f} min", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
