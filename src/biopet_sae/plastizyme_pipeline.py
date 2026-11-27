"""Polyester-degrader nomination + catalytic-triad annotation, from sequence.

  Part 1  ESM-2 650M layer 33, max over residues -> five linear probes, probabilities averaged.
          score >= 0.44 high confidence | >= 0.15 candidate | below: not nominated
  Part 2  (candidates only, unless --annotate-all) same layer-33 residue vectors -> InterPLM SAE:
          Ser = strongest residue of latent 2925, if activation >= 0.3 and that residue is S
          His = strongest residue of latent 3927, same rule, H
          Asp = every D scored by the 9-latent classifier; best one: >= 0.9 confident, >= 0.5 tentative
Scope: alpha/beta hydrolases, sequences up to 1022 residues. A located triad is hydrolase machinery, not
evidence of plastic degradation; nominations are candidates for lab testing. Positions are 1-based.

Usage:
    uv run python -m biopet_sae.plastizyme_pipeline --fasta proteins.fa --out predictions.tsv
    uv run python -m biopet_sae.plastizyme_pipeline --check          # end-to-end test-set verification
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from interplm.sae.dictionary import ReLUSAE
from transformers import AutoTokenizer, EsmModel

MODEL_DIR = Path("models/pipeline_v1_2")
MAXLEN = 1022


class Pipeline:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str | None = None):
        p = np.load(model_dir / "probe.npz")
        self.mean, self.scale, self.W, self.b = p["mean"], p["scale"], p["weights"], p["intercepts"]
        self.t_cand, self.t_high = float(p["candidate"]), float(p["high_confidence"])
        t = json.loads((model_dir / "triad.json").read_text())
        self.ser, self.his, self.fire = t["ser_latent"], t["his_latent"], t["fire"]
        self.asp_lat = [int(k) for k in t["asp"]["latents"]]; self.asp_w = np.array(list(t["asp"]["latents"].values()))
        self.asp_b = t["asp"]["intercept"]; self.asp_tent, self.asp_conf = t["asp"]["cutoffs"]
        self.dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
        self.sae = ReLUSAE.from_pretrained(hf_hub_download("Elana/InterPLM-esm2-650m", "layer_33/ae_normalized.pt")).to(self.dev).eval()
        self.tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(self.dev).eval()
        self._n = 0

    @torch.no_grad()
    def predict(self, seq: str, annotate_all: bool = False) -> dict:
        seq = seq.strip().upper().rstrip("*")
        out = {"length": len(seq), "truncated": len(seq) > MAXLEN}
        seq = seq[:MAXLEN]
        enc = {k: v.to(self.dev) for k, v in self.tok(seq, return_tensors="pt").items()}
        h = self.esm(**enc, output_hidden_states=True).hidden_states[33][0, 1:-1, :].float()
        x = (h.max(0).values.cpu().numpy() - self.mean) / self.scale
        score = float(np.mean(1 / (1 + np.exp(-(self.W @ x + self.b)))))
        out["score"] = score
        out["call"] = "high_confidence" if score >= self.t_high else "candidate" if score >= self.t_cand else "not_nominated"
        if out["call"] != "not_nominated" or annotate_all:
            f = self.sae.encode(h)[:, [self.ser, self.his] + self.asp_lat].cpu().numpy()
            for name, col, aa in (("ser", 0, "S"), ("his", 1, "H")):
                q = int(f[:, col].argmax())
                ok = f[q, col] >= self.fire and seq[q] == aa
                out[name] = q + 1 if ok else None
                out[f"{name}_activation"] = float(f[q, col])
            D = [i for i, c in enumerate(seq) if c == "D"]
            out["asp"], out["asp_score"], out["asp_level"] = None, None, "not_located"
            if D:
                p = 1 / (1 + np.exp(-(f[D, 2:] @ self.asp_w + self.asp_b))); j = int(np.argmax(p))
                out["asp_score"] = float(p[j])
                if p[j] >= self.asp_tent:
                    out["asp"] = D[j] + 1
                    out["asp_level"] = "confident" if p[j] >= self.asp_conf else "tentative"
        self._n += 1
        if self.dev.type == "mps" and self._n % 50 == 0:
            torch.mps.empty_cache()
        return out


def read_fasta(path: Path):
    name, seq = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name: yield name, "".join(seq)
            name, seq = line[1:].split()[0], []
        else:
            seq.append(line.strip())
    if name: yield name, "".join(seq)


def first_existing(*paths: str) -> Path:
    """First path that exists. `results/` is a local cache and is gitignored, so the committed
    copy under `paper/` is used when it is absent (e.g. a fresh clone)."""
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise SystemExit("none of these files exist: " + ", ".join(paths))


def check(pl: Pipeline) -> None:
    """Run every v1.2 test sequence end to end and compare with the recorded results."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from biopet_sae.sae_localise_v12 import one_each
    rows = [r for r in csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t") if r["split"] == "test"]
    pos = json.loads(first_existing("results/v1_2/algorithm_localisation/positive_triad_sites.json",
                                    "paper/v1_2_triad/positive_triad_sites.json").read_text())
    new = {r["seq_id"]: [int(x[1:]) for x in r["active_sites"].split(";") if x] for r in
           csv.DictReader(open("data/raw/hard_negatives_v12/candidates_annotation_filtered.tsv"), delimiter="\t")}
    old = json.load(open("data/processed/uniprot_negative_annotation.json"))
    P = {r["seq_id"]: pl.predict(r["sequence"], annotate_all=True) for r in rows}
    y = np.array([int(r["y"]) for r in rows]); s = np.array([P[r["seq_id"]]["score"] for r in rows])
    print(f"\nPART 1 on {len(rows)} test proteins: AU-ROC {roc_auc_score(y, s):.4f}  AU-PRC {average_precision_score(y, s):.4f} (floor {y.mean():.2f})")
    for name, t in (("candidate", pl.t_cand), ("high_confidence", pl.t_high)):
        c = s >= t; tp = int((c & (y == 1)).sum())
        print(f"  {name:16s} >= {t}: precision {tp/max(c.sum(),1):.3f}  recall {tp/y.sum():.3f}  ({tp} of {int(y.sum())} degraders, {int(c.sum())-tp} false)")
    ann = []
    for r in rows:
        i = r["seq_id"]
        if i in pos: t = pos[i]
        elif r["negative_group"] == "ab_hydrolase_hard_negative":
            t = one_each(r["sequence"], new.get(i) or old.get(i, {}).get("act") or [])
        else: t = None
        if t and len(r["sequence"]) <= MAXLEN: ann.append((P[i], t, r["y"] == "1"))
    print(f"\nPART 2 on {len(ann)} annotated test proteins (annotation run on all of them):")
    for name, key, t in (("Ser", "ser", "S"), ("His", "his", "H"), ("Asp", "asp", "D")):
        called = [a for a in ann if a[0][key] is not None]; ok = sum(a[0][key] == a[1][t] + 1 for a in called)
        print(f"  {name}: precision {ok/max(len(called),1):.1%}  recall {ok/len(ann):.1%}  ({ok} right / {len(called)} called)")
    full = sum(all(a[0][k] == a[1][t] + 1 for k, t in (("ser", "S"), ("his", "H"), ("asp", "D"))) for a in ann)
    print(f"  whole triad: {full}/{len(ann)} ({full/len(ann):.1%})")
    deg = [a for a in ann if a[2]]
    real = sum(a[0]["call"] != "not_nominated" for a in deg)
    both = sum(a[0]["call"] != "not_nominated" and all(a[0][k] == a[1][t] + 1 for k, t in (("ser", "S"), ("his", "H"), ("asp", "D"))) for a in deg)
    print(f"  end to end on annotated test degraders ({len(deg)}): nominated {real}, nominated with correct whole triad {both}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", type=Path); ap.add_argument("--out", type=Path)
    ap.add_argument("--annotate-all", action="store_true", help="annotate the triad even when not nominated")
    ap.add_argument("--check", action="store_true", help="verify against the recorded test results")
    a = ap.parse_args(argv)
    if not a.check and not a.fasta:
        ap.error("give --fasta or --check")
    pl = Pipeline()
    if a.check:
        check(pl); return 0
    cols = ["id", "length", "truncated", "score", "call", "ser", "his", "asp", "asp_level", "asp_score", "ser_activation", "his_activation"]
    fh = open(a.out, "w", newline="") if a.out else sys.stdout
    w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore"); w.writeheader()
    for name, seq in read_fasta(a.fasta):
        r = pl.predict(seq, a.annotate_all)
        w.writerow({"id": name, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
