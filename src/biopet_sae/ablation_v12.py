"""Causal test: remove chosen SAE latents from the layer-33 residue vectors and re-score the probe on test.

Ablation (error-preserving): h' = h - sum over latents s in the set of f_s(residue) * decoder_s, i.e. exactly
decode(f) - decode(f with the set zeroed) is subtracted, so the autoencoder's reconstruction error is untouched.
Then max over residues -> saved five-run probe (averaged), same thresholds (0.15, 0.44).

Sets: triad (2925, 3927 + the nine aspartate latents), its parts, and as comparisons
  * 100 random sets of 11 latents matched one-to-one on how often they fire (>= 0.3) in TRAIN proteins;
  * the 11 latents with the largest |Cliff's delta| on train+val (a set the probe might plausibly use);
  * all latents active anywhere (= remove the whole SAE reconstruction) as an upper bound.
Removed size is reported (mean norm of the subtracted vector / norm of h), since bigger removals move more.

Usage:  uv run python -m biopet_sae.ablation_v12
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from interplm.sae.dictionary import ReLUSAE
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoTokenizer, EsmModel

SER, HIS = 2925, 3927
N_RANDOM = 100


def main() -> int:
    out = Path("results/v1_2/ablation"); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open("data/release/biopet_sae_dataset_v1_2.tsv"), delimiter="\t"))
    test = [r for r in rows if r["split"] == "test"]
    y = np.array([int(r["y"]) for r in test])
    hard = np.array([r["negative_group"] == "ab_hydrolase_hard_negative" for r in test])
    P = np.load("models/pipeline_v1_2/probe.npz")
    asp = [int(k) for k in json.load(open("models/pipeline_v1_2/triad.json"))["asp"]["latents"]]
    triad = [SER, HIS] + asp
    delta = np.load("results/v1_2/sae_selection/selection_full.npz")["delta"]
    top_delta = [int(i) for i in np.argsort(-np.abs(delta))[: len(triad)]]

    # firing frequency of every latent on TRAIN proteins (for matched random sets)
    feat = {}
    for fp in ["data/sae_650M_L33/sae_features.npz", "data/sae_650M_L33_new416/sae_features.npz"]:
        z = np.load(fp, allow_pickle=True); A = z["feat_max"]
        for i, s in enumerate(z["seq_ids"].astype(str)): feat.setdefault(s, A[i])
    F = np.stack([feat[r["seq_id"]] for r in rows if r["split"] == "train"])
    freq = (F >= 0.3).mean(0)
    rng = np.random.default_rng(0)
    pool = {s: np.array([j for j in np.argsort(np.abs(freq - freq[s]))[:300] if j not in triad]) for s in triad}
    randoms = []
    for _ in range(N_RANDOM):
        chosen = []
        for s in triad:
            chosen.append(int(rng.choice([j for j in pool[s] if j not in chosen])))
        randoms.append(chosen)

    sets = {"nothing (baseline)": [], "triad: Ser + His + 9 Asp (11)": triad, "Ser 2925 only": [SER], "His 3927 only": [HIS],
            "Ser + His": [SER, HIS], "9 Asp latents": asp, "top 11 by |delta|": top_delta}
    sets.update({f"random{i}": s for i, s in enumerate(randoms)})

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sae = ReLUSAE.from_pretrained(hf_hub_download("Elana/InterPLM-esm2-650m", "layer_33/ae_normalized.pt")).to(dev).eval()
    assert not getattr(sae, "normalize_to_sqrt_d", False), "SAE expects normalised inputs; ablation would be wrong"
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(dev).eval()
    D = sae.decoder.weight.detach()                      # (1280, 10240)
    mean = torch.tensor(P["mean"], dtype=torch.float32, device=dev); scale = torch.tensor(P["scale"], dtype=torch.float32, device=dev)
    W = torch.tensor(P["weights"], dtype=torch.float32, device=dev); b = torch.tensor(P["intercepts"], dtype=torch.float32, device=dev)
    names = list(sets) + ["all latents (whole reconstruction)"]
    scores = np.zeros((len(names), len(test))); size = np.zeros((len(names), len(test)))
    with torch.no_grad():
        for i, r in enumerate(test):
            enc = {k: v.to(dev) for k, v in tok(r["sequence"][:1022], return_tensors="pt").items()}
            h = esm(**enc, output_hidden_states=True).hidden_states[33][0, 1:-1, :].float()
            f = sae.encode(h)
            hn = h.norm(dim=1)
            for k, name in enumerate(names):
                if name.startswith("all latents"):
                    d = f @ D.T
                else:
                    s = sets[name]
                    d = f[:, s] @ D[:, s].T if s else torch.zeros_like(h)
                x = ((h - d).max(0).values - mean) / scale
                scores[k, i] = torch.sigmoid(W @ x + b).mean().item()
                size[k, i] = (d.norm(dim=1) / hn).mean().item()
            del h, f, enc
            if (i + 1) % 25 == 0:
                if dev.type == "mps": torch.mps.empty_cache()
                print(f"  {i+1}/{len(test)}", flush=True)

    base = scores[0]
    def metrics(s):
        m = {"au_roc": roc_auc_score(y, s), "au_prc": average_precision_score(y, s)}
        for tag, t in (("cand", float(P["candidate"])), ("high", float(P["high_confidence"]))):
            c = s >= t; tp = int((c & (y == 1)).sum()); fp = int((c & (y == 0)).sum()); fn = int(y.sum()) - tp
            m.update({f"{tag}_precision": tp / max(tp + fp, 1), f"{tag}_recall": tp / y.sum(), f"{tag}_f1": 2 * tp / max(2 * tp + fp + fn, 1)})
        m.update({"mean_score_degraders": float(s[y == 1].mean()), "mean_score_hard_neg": float(s[hard].mean()),
                  "mean_change_degraders": float((s - base)[y == 1].mean()), "mean_change_hard_neg": float((s - base)[hard].mean()),
                  "mean_abs_change": float(np.abs(s - base).mean())})
        return {k: float(v) for k, v in m.items()}
    res = {n: {**metrics(scores[k]), "removed_fraction": float(size[k].mean()), "latents": sets.get(n, "all")} for k, n in enumerate(names)}
    np.savez_compressed(out / "scores.npz", names=np.array(names), scores=scores, size=size, ids=np.array([r["seq_id"] for r in test]), y=y)
    rnd = [res[f"random{i}"] for i in range(N_RANDOM)]
    summary = {k: [float(np.mean([r[k] for r in rnd])), float(np.percentile([r[k] for r in rnd], 2.5)), float(np.percentile([r[k] for r in rnd], 97.5))]
               for k in rnd[0] if k != "latents"}
    res["random_matched_summary"] = summary
    (out / "ablation.json").write_text(json.dumps(res, indent=2))

    def line(n, m):
        print(f"{n:36s} {m['au_roc']:.4f} {m['au_prc']:.4f} | {m['cand_precision']:.3f} {m['cand_recall']:.3f} {m['cand_f1']:.3f} | {m['high_precision']:.3f} {m['high_recall']:.3f} {m['high_f1']:.3f} | {m['mean_change_degraders']:+.4f} {m['mean_change_hard_neg']:+.4f} | {m['removed_fraction']:.3f}")
    print(f"\ntest {len(test)} ({int(y.sum())} degraders)   floor {y.mean():.2f}")
    print(f"{'ablated':36s} AUROC  AUPRC | at 0.15: P R F1      | at 0.44: P R F1      | score change deg / hard | removed")
    for n in list(sets)[:7] + ["all latents (whole reconstruction)"]:
        line(n, res[n])
    s = summary
    line("random matched 11 (mean of 100)", {k: v[0] for k, v in s.items()})
    print("   random 95%% range: AU-ROC [%.4f, %.4f]  AU-PRC [%.4f, %.4f]  F1@0.15 [%.3f, %.3f]  change on degraders [%+.4f, %+.4f]  removed [%.3f, %.3f]" % (
        s['au_roc'][1], s['au_roc'][2], s['au_prc'][1], s['au_prc'][2], s['cand_f1'][1], s['cand_f1'][2], s['mean_change_degraders'][1], s['mean_change_degraders'][2], s['removed_fraction'][1], s['removed_fraction'][2]))
    t = res["triad: Ser + His + 9 Asp (11)"]
    for k in ("au_roc", "au_prc", "cand_f1", "mean_change_degraders", "mean_abs_change"):
        v = np.array([r[k] for r in rnd])
        print(f"   triad {k}: {t[k]:+.4f} | random sets lower than triad: {int((v < t[k]).sum())}/100")
    return 0


if __name__ == "__main__":
    main()
