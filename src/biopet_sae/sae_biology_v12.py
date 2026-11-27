"""Biological reading of the ten highest-delta v1.2 latents (descriptive; selection was on train+val).

Step 1  triad localisation done correctly: proteins where a latent is silent are EXCLUDED, offsets are the
        modal (curated index - peak) over train+val proteins where it fires (activation >= 0.3), rate on test
        firing proteins, compared with all 10,240 latents treated the same way (needs >= 10 fit and >= 10 test).
        For positives (few curated sites) a proxy triad from 9529 (+1), 2661 (+2), 7734 (+1) is used, whose
        held-out accuracy on curated hard negatives was 99% / 72% / 98%.
Step 4  residue frequencies at peak -5..+5, positives and hard negatives separately.
Step 5  share of positives firing, by plastic, inside and outside the largest family.
Step 2  IsPETase with HEMT (PDB 5XH3): binding-site residues = any atom within 5 A of the ligand; for every
        positive with >= 30% identity to IsPETase, does the latent's peak align to a binding-site residue?
        Compared with the share of all residues that are binding-site residues.

Usage:  uv run python -m biopet_sae.sae_biology_v12
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

from biopet_sae.make_splits import global_identity, make_aligner
from biopet_sae.sae_localise_v12 import MAXLEN, TYPES, one_each
from biopet_sae.sae_peak_context_v12 import THREE

TOP = [1817, 5325, 3688, 7820, 2919, 9529, 7018, 9458, 5160, 5272]
PROXY = {"S": (9529, 1), "D": (2661, 2), "H": (7734, 1)}
FIRE = 0.3


def pdb_atoms(path):
    res, lig = {}, []
    for ln in path.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")) or ln[21] != "A" or ln[76:78].strip() == "H":
            continue
        xyz = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
        if ln[17:20] == "856":
            lig.append(xyz)
        elif ln.startswith("ATOM") and ln[17:20] in THREE:
            res.setdefault(ln[22:27], [THREE[ln[17:20]], []])[1].append(xyz)
    keys = list(res)
    return "".join(res[k][0] for k in keys), [np.array(res[k][1]) for k in keys], np.array(lig)


def main() -> int:
    out = Path("results/v1_2/biology"); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t"))
    rel = {r["seq_id"]: r for r in rows}
    big = Counter(r["component"] for r in rows if r["y"] == "1").most_common(1)[0][0]
    res = {"latents": TOP}

    # ---------------- step 1: curated sites, all latents ----------------
    z = np.load("results/v1_2/localisation/peaks.npz"); ids = z["ids"].astype(str); P = z["peaks"]; M = z["amax"]
    new = {r["seq_id"]: [int(x[1:]) for x in r["active_sites"].split(";") if x] for r in
           csv.DictReader(open("data/raw/hard_negatives_v12/candidates_annotation_filtered.tsv"), delimiter="\t")}
    old = json.load(open("data/processed/uniprot_negative_annotation.json"))
    ps = json.load(open("results/v1_2/localisation/positive_sites_uniprot.json"))
    sites = [one_each(rel[i]["sequence"], ps.get(i) or new.get(i) or old.get(i, {}).get("act") or []) for i in ids]
    sp = np.array([rel[i]["split"] for i in ids]); fit = np.isin(sp, ["train", "val"]); test = sp == "test"
    step1 = {}
    for t in TYPES:
        cur = np.array([s[t] for s in sites]); good = []
        for i, s_ in zip(ids, sites):
            s, c = rel[i]["sequence"], s_[t]; g = np.zeros(len(s), bool)
            for q in range(len(s)):
                cand = [j for j in range(max(0, q - 3), min(len(s), q + 4)) if s[j] == t]
                if cand and min(cand, key=lambda j: (abs(j - q), j)) == c: g[q] = True
            g[c] = True; good.append(g)
        rate = np.full(10240, np.nan); off = np.zeros(10240, int); nt = np.zeros(10240, int); nf = np.zeros(10240, int); cons = np.zeros(10240)
        for l in range(10240):
            f = np.flatnonzero(fit & (M[:, l] >= FIRE)); e = np.flatnonzero(test & (M[:, l] >= FIRE))
            nf[l], nt[l] = len(f), len(e)
            if len(f) < 10 or len(e) < 10: continue
            v, c = np.unique(cur[f] - P[f, l], return_counts=True); j = int(np.argmax(c)); off[l] = v[j]; cons[l] = c[j] / len(f)
            rate[l] = np.mean([0 <= P[i, l] + off[l] < len(good[i]) and good[i][P[i, l] + off[l]] for i in e])
        ok = ~np.isnan(rate)
        step1[t] = {"n_eligible_latents": int(ok.sum()), "null_median": float(np.median(rate[ok])), "null_p95": float(np.quantile(rate[ok], .95)),
                    "latents": {l: ({"eligible": True, "fit_n": int(nf[l]), "test_n": int(nt[l]), "offset": int(off[l]), "consistency": round(float(cons[l]), 2),
                                     "rate": round(float(rate[l]), 3), "n_better": int((rate[ok] > rate[l]).sum())} if ok[l]
                                    else {"eligible": False, "fit_n": int(nf[l]), "test_n": int(nt[l])}) for l in TOP}}
    res["step1_curated"] = step1
    print("step 1 done", flush=True)

    # ---------------- forward pass on positives and hard negatives ----------------
    todo = [r for r in rows if (r["y"] == "1" or r["negative_group"] == "ab_hydrolase_hard_negative") and len(r["sequence"]) <= MAXLEN]
    cols = TOP + [2661, 7734]
    cache = out / "forward_pass.json"
    rec = {i: {int(k): tuple(v) for k, v in d.items()} for i, d in json.loads(cache.read_text()).items()} if cache.exists() else {}
    left = [r for r in todo if r["seq_id"] not in rec]
    if left:
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        sae = ReLUSAE.from_pretrained(hf_hub_download("Elana/InterPLM-esm2-650m", "layer_33/ae_normalized.pt")).to(dev).eval()
        tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(dev).eval()
        with torch.no_grad():
            for k, r in enumerate(left):
                enc = {x: v.to(dev) for x, v in tok(r["sequence"], return_tensors="pt").items()}
                h = esm(**enc, output_hidden_states=True).hidden_states[33][0, 1:-1, :].float()
                f = sae.encode(h)[:, cols].cpu().numpy()
                rec[r["seq_id"]] = {l: (int(f[:, j].argmax()), float(f[:, j].max())) for j, l in enumerate(cols)}
                del enc, h
                if (k + 1) % 25 == 0:
                    if dev.type == "mps": torch.mps.empty_cache()
                    cache.write_text(json.dumps(rec))
                    print(f"  {k+1}/{len(left)}", flush=True)
        cache.write_text(json.dumps(rec))
        del sae, esm
    print("forward pass done", len(rec), flush=True)
    pos = [i for i in rec if rel[i]["y"] == "1"]; hard = [i for i in rec if rel[i]["y"] == "0"]

    # proxy triad on positives
    prox = {}
    for l in TOP:
        prox[l] = {}
        for t, (pl, po) in PROXY.items():
            d = [rec[i][l][0] - (rec[i][pl][0] + po) for i in pos if rec[i][l][1] >= FIRE and rec[i][pl][1] >= FIRE]
            if not d: prox[l][t] = None; continue
            m, c = Counter(d).most_common(1)[0]
            prox[l][t] = {"n": len(d), "modal_offset": int(m), "share_at_mode": round(c / len(d), 2), "share_within_3_of_residue": round(float(np.mean(np.abs(d) <= 3)), 2)}
    res["step1_proxy_positives"] = prox

    # ---------------- step 4: motifs ----------------
    motif = {}
    for l in TOP:
        motif[l] = {}
        for g, G in (("positives", pos), ("hard_negatives", hard)):
            I = [i for i in G if rec[i][l][1] >= FIRE]
            cons = []
            for d in range(-5, 6):
                c = Counter(rel[i]["sequence"][rec[i][l][0] + d] if 0 <= rec[i][l][0] + d < len(rel[i]["sequence"]) else "-" for i in I)
                cons.append(c.most_common(3) if I else [])
            motif[l][g] = {"n_firing": len(I), "n": len(G),
                           "logo": [[(a, round(n / len(I), 2)) for a, n in c] for c in cons] if I else [],
                           "consensus40": "".join(c[0][0] if c and c[0][1] / len(I) >= 0.4 else "." for c in cons) if I else ""}
    res["step4_motifs"] = motif
    print("step 4 done", flush=True)

    # ---------------- step 5: by plastic ----------------
    plastics = ["PET", "PLA", "PCL", "PBAT", "PHA", "PBS"]
    feat = {}
    for fp in ["data/sae_650M_L33/sae_features.npz", "data/sae_650M_L33_new416/sae_features.npz"]:
        q = np.load(fp, allow_pickle=True)
        A = q["feat_max"]                     # load once: indexing q[...] per row re-reads the whole array
        for i, s in enumerate(q["seq_ids"].astype(str)): feat.setdefault(s, A[i])
    allpos = [r["seq_id"] for r in rows if r["y"] == "1" and r["seq_id"] in feat]
    st5 = {}
    for fam, F in (("largest family", [i for i in allpos if rel[i]["component"] == big]), ("other families", [i for i in allpos if rel[i]["component"] != big])):
        st5[fam] = {}
        groups = {p: [i for i in F if p in rel[i]["substrates"].split("|")] for p in plastics}
        groups["no PET"] = [i for i in F if "PET" not in rel[i]["substrates"].split("|")]
        for g, I in groups.items():
            st5[fam][g] = {"n": len(I), "n_components": len({rel[i]["component"] for i in I}),
                           "firing": {l: (round(float(np.mean([feat[i][l] >= FIRE for i in I])), 2) if I else None) for l in TOP}}
    res["step5_by_plastic"] = st5
    print("step 5 done", flush=True)

    # ---------------- step 2: ligand-bound IsPETase ----------------
    pseq, atoms, lig = pdb_atoms(Path("data/pdb_cache/5XH3.pdb"))
    dmin = np.array([np.sqrt(((a[:, None, :] - lig[None]) ** 2).sum(-1)).min() for a in atoms])
    site = dmin <= 5.0
    al = make_aligner(); ref = rel["BPS0282"]["sequence"]

    def mapping(s):
        aln = al.align(s, pseq)[0]; m = {}
        for (a0, a1), (b0, b1) in zip(*aln.aligned):
            for k in range(a1 - a0): m[a0 + k] = b0 + k
        return m

    homologs = [i for i in pos if global_identity(rel[i]["sequence"], ref, al) >= 0.30]
    maps = {i: mapping(rel[i]["sequence"]) for i in homologs}
    ser_ref = mapping(ref)[159]
    st2 = {"pdb": "5XH3", "ligand": "856 (HEMT)", "n_binding_site_residues": int(site.sum()), "chain_length": len(pseq),
           "base_rate": round(float(site.mean()), 3), "n_homologs_positives_ge30": len(homologs),
           "catalytic_ser_position_min_distance_A": round(float(dmin[ser_ref]), 1), "latents": {}}
    for l in TOP + [2661, 7734]:
        I = [i for i in homologs if rec[i][l][1] >= FIRE and rec[i][l][0] in maps[i]]
        q = [maps[i][rec[i][l][0]] for i in I]
        st2["latents"][l] = {"n_firing_mapped": len(I), "share_peaks_in_binding_site": round(float(site[q].mean()), 2) if I else None,
                             "median_min_distance_A": round(float(np.median(dmin[q])), 1) if I else None,
                             "most_common_position": [(f"{pseq[a]}(pdb index {a})", n) for a, n in Counter(q).most_common(2)]}
    res["step2_binding_site"] = st2
    (out / "biology.json").write_text(json.dumps(res, indent=2, default=str))

    print("\nSTEP 1 (curated sites; silent proteins excluded) test firing proteins: rate [n] (latents better of eligible)")
    for t in TYPES:
        s = step1[t]; print(f"  {t}: eligible latents {s['n_eligible_latents']}, null median {s['null_median']:.1%}, 95th {s['null_p95']:.1%}")
    for l in TOP:
        out_ = []
        for t in TYPES:
            v = step1[t]["latents"][l]
            out_.append(f"{t}: {v['rate']:.0%} [n={v['test_n']}] off {v['offset']:+d} better {v['n_better']}" if v["eligible"] else f"{t}: not testable (fit {v['fit_n']}, test {v['test_n']})")
        print(f"  {l:5d} " + " | ".join(out_))
    print("\nSTEP 1b positives, proxy triad: modal offset (share at mode) / share within 3 residues")
    for l in TOP:
        print(f"  {l:5d} " + " | ".join(f"{t}: {v['modal_offset']:+4d} ({v['share_at_mode']:.0%}) / {v['share_within_3_of_residue']:.0%}" if v else f"{t}: -" for t, v in prox[l].items()))
    print("\nSTEP 4 motif at peak-5..+5 (residue shown if >=40%)")
    for l in TOP:
        a, b = motif[l]["positives"], motif[l]["hard_negatives"]
        print(f"  {l:5d} pos {a['n_firing']:3d}/{a['n']} {a['consensus40']:11s} | hard {b['n_firing']:3d}/{b['n']} {b['consensus40']:11s}")
    print("\nSTEP 5 share of positives firing, by plastic")
    for fam, G in st5.items():
        print(f"  [{fam}] " + "  ".join(f"{g} n={v['n']}({v['n_components']}fam)" for g, v in G.items()))
        for l in TOP:
            print(f"    {l:5d} " + "  ".join(f"{g}:{v['firing'][l]:.0%}" if v['n'] else f"{g}:-" for g, v in G.items()))
    print(f"\nSTEP 2 binding site (5XH3): {st2['n_binding_site_residues']} of {st2['chain_length']} residues within 5 A of ligand (base {st2['base_rate']:.1%}); catalytic Ser position {st2['catalytic_ser_position_min_distance_A']} A; homolog positives {len(homologs)}")
    for l, v in st2["latents"].items():
        print(f"  {l:5d} n={v['n_firing_mapped']:3d} peaks in site {('-' if v['share_peaks_in_binding_site'] is None else format(v['share_peaks_in_binding_site'], '.0%')):>4s}  median distance {v['median_min_distance_A']} A  common {v['most_common_position']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
