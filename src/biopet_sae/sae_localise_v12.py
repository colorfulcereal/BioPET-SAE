"""Catalytic-triad localisation of SAE latents on v1.2, with the all-latent null (v1 procedure).

Ground truth: curated UniProt "Active site" positions, kept only when there is exactly one Ser, one Asp
and one His among them. Hard negatives come from the candidate table / cached annotation; positives are
looked up by their UniProt accession and kept only if the sequences are identical or one contains the other
(positions are shifted accordingly).

For every latent (all 10,240) and each residue type:
  peak     residue index (0-based) where the latent's activation is largest in that protein
  offset   most common (curated index - peak) over TRAIN+VAL proteins
  hit      peak + offset is the curated residue, or the nearest residue of that type within +/-3 of
           peak + offset is the curated residue
  rate     hits / TEST proteins (held out for the offset; NOT held out for latent selection of positives
           or hard negatives that were in train+val)
The null is simply the distribution of that rate over all 10,240 latents (identical treatment, no
activation requirement). `firing` = fraction of test proteins where the latent's peak activation >= 0.3.
Regression: peaks of 9529 / 2661 / 7734 are compared with the stored v1 peaks (split_localization.json).

WARNING (2026-11-26): a latent that is silent on a protein gets argmax 0 here, so results for latents that are often
silent (and the all-latent null) are invalid. Use the firing-only step 1 of `sae_biology_v12.py`. Results for 9529,
2661 and 7734 are unaffected (never silent on these proteins).

Usage:  uv run python -m biopet_sae.sae_localise_v12
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from interplm.sae.dictionary import ReLUSAE
from transformers import AutoTokenizer, EsmModel

LATENTS = [1817, 5325, 3688, 7820, 2919, 9529, 7018]       # v1.2 top 7
REF = [2661, 7734]                                           # v1 Asp / His latents, for reference
TYPES = ["S", "D", "H"]
MAXLEN = 1022


def one_each(seq: str, positions: list[int]) -> dict[str, int] | None:
    got: dict[str, list[int]] = {t: [] for t in TYPES}
    for p in positions:                                     # 1-based
        if 1 <= p <= len(seq) and seq[p - 1] in got:
            got[seq[p - 1]].append(p - 1)
    return {t: v[0] for t, v in got.items()} if all(len(v) == 1 for v in got.values()) else None


def uniprot_positive_sites(rows, merged: Path, cache: Path) -> dict[str, list[int]]:
    if cache.exists():
        return json.loads(cache.read_text())
    acc = {r["seq_id"]: r["uniprot"].split("|")[0].split(";")[0].strip()
           for r in csv.DictReader(merged.open(newline=""), delimiter="\t") if r["uniprot"].strip()}
    out = {}
    for r in rows:
        if r["y"] != "1" or r["seq_id"] not in acc:
            continue
        try:
            with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{acc[r['seq_id']]}.json", timeout=60) as f:
                e = json.load(f)
            useq = e.get("sequence", {}).get("value", "")
            pos = [x["location"]["start"]["value"] for x in e.get("features", []) if x["type"] == "Active site"]
        except Exception:                                   # noqa: BLE001
            continue
        s = r["sequence"]
        if not pos or not useq:
            continue
        if s in useq:
            shift = useq.index(s)
        elif useq in s:
            shift = -s.index(useq)
        else:
            continue
        out[r["seq_id"]] = [p - shift for p in pos]
        time.sleep(0.1)
    cache.write_text(json.dumps(out))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path("data/release/biopet_sae_dataset_v1_2.tsv"))
    ap.add_argument("--out", type=Path, default=Path("results/v1_2/localisation"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(a.release.open(newline=""), delimiter="\t"))

    new = {r["seq_id"]: [int(x[1:]) for x in r["active_sites"].split(";") if x] for r in
           csv.DictReader(open("data/raw/hard_negatives_v12/candidates_annotation_filtered.tsv"), delimiter="\t")}
    old = json.loads(Path("data/processed/uniprot_negative_annotation.json").read_text())
    psites = uniprot_positive_sites(rows, Path("data/processed/plastizymes_merged.tsv"), a.out / "positive_sites_uniprot.json")
    prot = []
    for r in rows:
        i, s = r["seq_id"], r["sequence"]
        if len(s) > MAXLEN:
            continue
        if r["y"] == "1":
            pos = psites.get(i); kind = "positive"
        elif r["negative_group"] == "ab_hydrolase_hard_negative":
            pos = new.get(i) or old.get(i, {}).get("act"); kind = "hard_negative"
        else:
            continue
        sites = one_each(s, pos or [])
        if sites:
            prot.append({"id": i, "seq": s, "split": r["split"], "kind": kind, "domain": r["domain"], "sites": sites})
    n = Counter((p["kind"], p["split"]) for p in prot)
    print("proteins with a curated Ser/Asp/His:", dict(sorted((f"{k}|{s}", v) for (k, s), v in n.items())), flush=True)

    # v1 positives used only for the regression check of the peak computation
    stored = {d["seq_id"]: d for d in json.loads(Path("data/sae_650M_L33/split_localization.json").read_text())}
    have = {p["id"] for p in prot}
    reg_ids = [r for r in rows if r["seq_id"] in stored and r["seq_id"] not in have and len(r["sequence"]) <= MAXLEN][:20]
    todo = prot + [{"id": r["seq_id"], "seq": r["sequence"], "split": "", "kind": "regression"} for r in reg_ids]

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sae = ReLUSAE.from_pretrained(hf_hub_download("Elana/InterPLM-esm2-650m", "layer_33/ae_normalized.pt")).to(dev).eval()
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(dev).eval()
    peaks = np.zeros((len(todo), 10240), np.int32); amax = np.zeros((len(todo), 10240), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for k, p in enumerate(todo):
            enc = {x: v.to(dev) for x, v in tok(p["seq"], return_tensors="pt").items()}
            h = esm(**enc, output_hidden_states=True).hidden_states[33][0, 1:-1, :].float()
            f = sae.encode(h).cpu().numpy()
            peaks[k] = f.argmax(0); amax[k] = f.max(0)
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)

    # regression against stored v1 peaks
    checked = agree = 0
    for k, p in enumerate(todo):
        d = stored.get(p["id"])
        if not d:
            continue
        for l in (9529, 2661, 7734):
            if d[str(l)]["act"] > 0.05:
                checked += 1; agree += int(peaks[k, l] == d[str(l)]["peak"])
    print(f"REGRESSION vs stored v1 peaks: {agree}/{checked} identical", flush=True)

    N = len(prot); peaks, amax = peaks[:N], amax[:N]
    np.savez_compressed(a.out / "peaks.npz", ids=np.array([p["id"] for p in prot]), peaks=peaks, amax=amax)
    sp = np.array([p["split"] for p in prot]); kind = np.array([p["kind"] for p in prot])
    fit = np.isin(sp, ["train", "val"])
    evals = {"test hard negatives": (sp == "test") & (kind == "hard_negative"),
             "test positives": (sp == "test") & (kind == "positive"),
             "train+val positives (NOT held out)": fit & (kind == "positive")}
    res = {"n": {k: int(v.sum()) for k, v in evals.items()}, "n_offset_fit": int(fit.sum()), "regression": [agree, checked], "residues": {}}
    for t in TYPES:
        cur = np.array([p["sites"][t] for p in prot])
        # good[i][q] : predicted position q counts as a hit for protein i
        good = []
        for p in prot:
            s, c = p["seq"], p["sites"][t]; g = np.zeros(len(s), bool)
            for q in range(len(s)):
                cand = [j for j in range(max(0, q - 3), min(len(s), q + 4)) if s[j] == t]
                if cand and min(cand, key=lambda j: (abs(j - q), j)) == c:
                    g[q] = True
            g[c] = True; good.append(g)
        gap = cur[:, None] - peaks
        off = np.zeros(10240, np.int32); cons = np.zeros(10240, np.int32)
        for l in range(10240):
            v, c = np.unique(gap[fit, l], return_counts=True); j = int(np.argmax(c)); off[l] = v[j]; cons[l] = c[j]
        H = np.zeros((N, 10240), bool)
        for i in range(N):
            q = peaks[i] + off; ok = (q >= 0) & (q < len(good[i])); H[i, ok] = good[i][q[ok]]
        exact = (peaks + off) == cur[:, None]
        tr = {}
        for name, m in evals.items():
            if not m.sum():
                continue
            rate = H[m].mean(0); ex = exact[m].mean(0)
            tr[name] = {"n": int(m.sum()), "null_median": float(np.median(rate)), "null_p95": float(np.quantile(rate, .95)), "null_max": float(rate.max()),
                        "latents": {str(l): {"offset": int(off[l]), "offset_consistency": f"{cons[l]}/{int(fit.sum())}", "rate": round(float(rate[l]), 3), "exact": round(float(ex[l]), 3),
                                              "percentile": round(100 * float((rate < rate[l]).mean()), 2), "n_higher": int((rate > rate[l]).sum()),
                                              "firing": round(float((amax[m, l] >= 0.3).mean()), 3)} for l in LATENTS + REF}}
        res["residues"][t] = tr
        np.save(a.out / f"rates_all_latents_{t}.npy", np.stack([H[m].mean(0) for m in evals.values() if m.sum()]))
    (a.out / "localisation.json").write_text(json.dumps(res, indent=2))
    for name in evals:
        if name not in res["residues"]["S"]:
            continue
        print(f"\n===== {name} (n={res['residues']['S'][name]['n']}) =====")
        print("null over all 10,240 latents (median / 95th / max): " + "; ".join(f"{t}: {res['residues'][t][name]['null_median']:.1%} / {res['residues'][t][name]['null_p95']:.1%} / {res['residues'][t][name]['null_max']:.1%}" for t in TYPES))
        print(f"{'latent':>7s} {'fires':>6s} | " + " | ".join(f"{t:>2s}: rate  exact  off  pct " for t in TYPES))
        for l in LATENTS + REF:
            c = [res["residues"][t][name]["latents"][str(l)] for t in TYPES]
            print(f"{l:7d} {c[0]['firing']:6.0%} | " + " | ".join(f"{x['rate']:6.1%} {x['exact']:6.1%} {x['offset']:+4d} {x['percentile']:5.1f}" for x in c) + ("   (v1 ref)" if l in REF else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
