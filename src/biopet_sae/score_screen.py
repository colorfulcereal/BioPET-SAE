"""Score the metagenomic screening set with the probe, BLASTp and the SAE latents."""
import csv, json, pickle, re, subprocess, tempfile
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download
from interplm.sae.dictionary import ReLUSAE
from transformers import AutoTokenizer, EsmModel
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

POS = {"1_pet", "2_other_polyester"}
NEG = ["3_fold_matched_esterase", "4_naive_control"]
LAT = [9529, 2661, 7734, 7942, 5201, 5255]
OFF = {"9529": ("Ser", 1, "S"), "2661": ("Asp", 3, "DE"), "7734": ("His", 1, "H")}

cand = list(csv.DictReader(open("data/mgnify/screen_set.tsv"), delimiter="\t"))
sp = {r["seq_id"]: r for r in csv.DictReader(
    open("data/processed/dataset_splits_id40.tsv"), delimiter="\t")}
print(f"{len(cand)} candidates")

# ---- refit the probe exactly as shipped (train split only) ---------------------------
d = np.load("data/embeddings_650M/embeddings.npz", allow_pickle=True)
eids = d["seq_ids"].astype(str); X = d["max_L33"]
lab = np.array([sp[i]["class_label"] if i in sp else "" for i in eids])
spl = np.array([sp[i]["split"] if i in sp else "" for i in eids])
m = np.isin(lab, list(POS) + NEG); tr = m & (spl == "train")
y = np.isin(lab, list(POS))
sc = StandardScaler().fit(X[tr])
clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1 / (1e-2 * tr.sum()),
                    learning_rate="constant", eta0=1e-4, max_iter=5000,
                    random_state=0, tol=None).fit(sc.transform(X[tr]), y[tr])
print(f"probe refit on {tr.sum()} train sequences (C=1e-2, eta0=1e-4)")

# ---- embed the candidates ------------------------------------------------------------
dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
sae = ReLUSAE.from_pretrained(hf_hub_download(
    "Elana/InterPLM-esm2-650m", "layer_33/ae_normalized.pt")).to(dev).eval()
tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(dev).eval()
W = torch.tensor(LAT, device=dev)

emb, acts = [], []
for i, r in enumerate(cand, 1):
    s = r["sequence"][:1022]
    with torch.no_grad():
        enc = {k: v.to(dev) for k, v in tok(s, return_tensors="pt").items()}
        h = esm(**enc, output_hidden_states=True).hidden_states[33][0, 1:-1, :].float()
        emb.append(h.max(dim=0).values.cpu().numpy())
        acts.append(sae.encode(h)[:, W].cpu().numpy())
    if i % 100 == 0:
        print(f"   embedded {i}/{len(cand)}")
E = np.stack(emb)
score = clf.predict_proba(sc.transform(E))[:, 1]

# ---- BLASTp against the 303 training positives ---------------------------------------
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    with (td / "db.fa").open("w") as fh:
        for i in np.flatnonzero(tr & y):
            fh.write(f">{eids[i]}\n{sp[eids[i]]['sequence']}\n")
    with (td / "q.fa").open("w") as fh:
        for r in cand:
            fh.write(f">{r['cand_id'].replace('|','_')}\n{r['sequence']}\n")
    subprocess.run(["makeblastdb", "-in", str(td / "db.fa"), "-dbtype", "prot",
                    "-out", str(td / "db")], check=True, capture_output=True)
    subprocess.run(["blastp", "-query", str(td / "q.fa"), "-db", str(td / "db"),
                    "-out", str(td / "h.tsv"), "-outfmt", "6 qseqid sseqid evalue pident",
                    "-evalue", "1000", "-max_target_seqs", "1"], check=True, capture_output=True)
    best = {}
    for ln in (td / "h.tsv").open():
        p = ln.rstrip().split("\t")
        if p[0] not in best or float(p[2]) < best[p[0]][0]:
            best[p[0]] = (float(p[2]), p[1], float(p[3]))

# ---- assemble --------------------------------------------------------------------
out = []
for k, r in enumerate(cand):
    s = r["sequence"][:1022]
    A = acts[k]
    ev, hit, pid = best.get(r["cand_id"].replace("|", "_"), (float("inf"), "-", 0.0))
    rec = {"cand_id": r["cand_id"], "study": r["study"], "length": int(r["length"]),
           "elbow": r["elbow"], "probe": round(float(score[k]), 4),
           "blast_evalue": ev, "blast_hit": hit, "blast_pident": pid,
           "sequence": s}
    hits = 0
    for j, (role, off, want) in OFF.items():
        col = LAT.index(int(j))
        pk = int(A[:, col].argmax())
        pred = pk + off
        okres = 0 <= pred < len(s) and s[pred] in want
        rec[f"lat{j}_peak"] = pk
        rec[f"lat{j}_act"] = round(float(A[pk, col]), 4)
        rec[f"lat{j}_res"] = s[pred] if 0 <= pred < len(s) else "?"
        rec[f"lat{j}_ok"] = bool(okres)
        hits += okres
    rec["triad_hits"] = hits
    rec["pocket_act"] = round(float(np.mean([A[:, LAT.index(j)].max()
                                             for j in (7942, 5201, 5255)])), 4)
    out.append(rec)

json.dump(out, open("data/mgnify/screen_scored.json", "w"), indent=1)
arr = np.array([o["probe"] for o in out])
print(f"\nprobe score: median {np.median(arr):.4f}  max {arr.max():.4f}")
for t in (0.9, 0.7, 0.5, 0.3, 0.1455):
    print(f"   >= {t:<7.4f} {int((arr >= t).sum()):>4} candidates")
print("wrote data/mgnify/screen_scored.json")
