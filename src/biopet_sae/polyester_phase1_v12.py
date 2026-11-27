"""Phase one, v1.2 probe: five training runs, threshold table, ROC and PR tables with standard errors.

Procedure (as specified by the project lead):
  1. Train the probe 5 times (seeds 42-46) on the TRAIN split only; hyperparameters from the val sweep.
  2. For every run and every probability threshold, compute precision, recall (= TPR), FPR and F1 on val
     and on test. Average each over the 5 runs; SE = sd / sqrt(5) (training randomness only).
  3. Pick the ONE threshold that maximises the mean val F1.
  4. Report val and test at that frozen threshold (mean +/- SE). Thresholds for precision targets
     (95/90/85/80%) are likewise the lowest grid threshold whose mean val precision reaches the target.
  5. ROC curve = mean TPR against mean FPR per threshold; PR curve = mean precision against mean recall.
     Tables hold both means and SEs. AU-ROC and AU-PRC are exact per-run values, reported mean +/- SE.
  Precision is undefined when nothing is predicted positive: those runs are left out of that threshold's
  mean (count stored in `n_runs_precision`) and F1 is 0.

Usage:  uv run python -m biopet_sae.polyester_phase1_v12
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from biopet_sae.polyester_blast import GRID, best_hits

SEEDS = [42, 43, 44, 45, 46]
THR = np.unique(np.round(np.r_[0.001, 0.005, np.arange(0.01, 1.0, 0.01), 0.995, 0.999], 4))
PRECS = [0.95, 0.90, 0.85, 0.80]


def counts(y, s, thr):
    pred = s[None, :] >= np.asarray(thr)[:, None]
    tp = (pred & (y == 1)).sum(1); fp = (pred & (y == 0)).sum(1)
    fn = (~pred & (y == 1)).sum(1); tn = (~pred & (y == 0)).sum(1)
    return tp, fp, fn, tn


def per_run_tables(y, S, thr):
    """arrays (runs, thresholds): precision (nan if undefined), recall, fpr, f1, tp, fp, fn, tn"""
    out = {k: [] for k in ("precision", "recall", "fpr", "f1", "tp", "fp", "fn", "tn")}
    for s in S:
        tp, fp, fn, tn = counts(y, s, thr)
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
        rec = tp / max(int((y == 1).sum()), 1); fpr = fp / max(int((y == 0).sum()), 1)
        f1 = np.where((tp + fp > 0) & (tp > 0), 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
        for k, v in zip(out, (prec, rec, fpr, f1, tp, fp, fn, tn)): out[k].append(v)
    return {k: np.array(v, float) for k, v in out.items()}


def mean_se(a, axis=0):
    n = np.sum(~np.isnan(a), axis=axis)
    m = np.nanmean(a, axis=axis)
    se = np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1)) if a.shape[axis] > 1 else np.zeros_like(m)
    return m, se, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path("data/release/biopet_sae_dataset_v1_2.tsv"))
    ap.add_argument("--embeddings", type=Path, default=Path("data/embeddings_650M_v1_2/embeddings.npz"))
    ap.add_argument("--sweep", type=Path, default=Path("results/v1_2/cells/sweep/sweep.json"))
    ap.add_argument("--out", type=Path, default=Path("results/v1_2/phase1"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    sel = json.loads(a.sweep.read_text())["selected"]
    C, eta = float(sel.split(",")[0][2:]), float(sel.split(",")[1][4:])
    rows = list(csv.DictReader(a.release.open(newline=""), delimiter="\t"))
    emb = np.load(a.embeddings, allow_pickle=True)
    pos = {s: i for i, s in enumerate(emb["seq_ids"].astype(str))}
    X = emb["max_L33"][[pos[r["seq_id"]] for r in rows]]
    ids = np.array([r["seq_id"] for r in rows]); y = np.array([int(r["y"]) for r in rows])
    sp = np.array([r["split"] for r in rows]); grp = np.array([r["negative_group"] for r in rows])
    dom = np.array([r["domain"] for r in rows]); seq = {r["seq_id"]: r["sequence"] for r in rows}
    tr, va, te = sp == "train", sp == "val", sp == "test"
    sc = StandardScaler().fit(X[tr]); S = []
    for sd in SEEDS:
        clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1.0 / (C * tr.sum()), learning_rate="constant", eta0=eta,
                            max_iter=5000, tol=None, class_weight="balanced", random_state=sd).fit(sc.transform(X[tr]), y[tr])
        S.append(clf.predict_proba(sc.transform(X))[:, 1])
    S = np.array(S)
    np.savez(a.out / "scores_5_runs.npz", seq_ids=ids, scores=S, y=y, split=sp, group=grp, domain=dom, seeds=np.array(SEEDS))

    hard = grp == "ab_hydrolase_hard_negative"; bact = dom == "Bacteria"
    cells = {"ALL negatives (whole split)": np.ones(len(y), bool), "hard negatives only": (y == 1) | hard,
             "bacterial hard negatives": (y == 1) | (hard & bact),
             "matched: bacterial pos. vs bacterial hard neg.": ((y == 1) & bact) | (hard & bact)}
    res = {"selected": sel, "seeds": SEEDS, "n_train": int(tr.sum())}
    tabs = {}
    for name, sm in (("val", va), ("test", te)):
        T = per_run_tables(y[sm], S[:, sm], THR); tabs[name] = T
        cols = {}
        for k in ("precision", "recall", "fpr", "f1", "tp", "fp", "fn", "tn"):
            m, se, n = mean_se(T[k]); cols[k + "_mean"] = m; cols[k + "_se"] = se
            if k == "precision": cols["n_runs_precision"] = n
        with (a.out / f"threshold_table_{name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["threshold"] + list(cols))
            for i, t in enumerate(THR): w.writerow([t] + [round(float(cols[c][i]), 5) for c in cols])
        with (a.out / f"roc_curve_{name}.csv").open("w", newline="") as fh:       # mean TPR vs mean FPR
            w = csv.writer(fh); w.writerow(["threshold", "fpr_mean", "fpr_se", "tpr_mean", "tpr_se"])
            for i, t in enumerate(THR): w.writerow([t, round(float(cols["fpr_mean"][i]), 5), round(float(cols["fpr_se"][i]), 5), round(float(cols["recall_mean"][i]), 5), round(float(cols["recall_se"][i]), 5)])
        with (a.out / f"pr_curve_{name}.csv").open("w", newline="") as fh:        # mean precision vs mean recall
            w = csv.writer(fh); w.writerow(["threshold", "recall_mean", "recall_se", "precision_mean", "precision_se", "n_runs_precision"])
            for i, t in enumerate(THR): w.writerow([t, round(float(cols["recall_mean"][i]), 5), round(float(cols["recall_se"][i]), 5), round(float(cols["precision_mean"][i]), 5) if not np.isnan(cols["precision_mean"][i]) else "", round(float(cols["precision_se"][i]), 5) if not np.isnan(cols["precision_se"][i]) else "", int(cols["n_runs_precision"][i])])
        tabs[name]["cols"] = cols
    # ---- the single threshold, chosen on validation
    mf1 = tabs["val"]["cols"]["f1_mean"]; ib = int(np.argmax(mf1)); tstar = float(THR[ib])
    res["threshold"] = tstar
    def at(name, i):
        c = tabs[name]["cols"]
        return {k: [float(c[k + "_mean"][i]), float(c[k + "_se"][i])] for k in ("precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn")}
    res["at_threshold"] = {"val": at("val", ib), "test": at("test", ib)}
    res["precision_targets"] = {}
    vp = tabs["val"]["cols"]["precision_mean"]
    for p in PRECS:
        ok = np.flatnonzero(~np.isnan(vp) & (vp >= p))
        if len(ok):
            i = int(ok[0]); res["precision_targets"][str(int(p * 100))] = {"threshold": float(THR[i]), "val": at("val", i), "test": at("test", i)}
    # ---- threshold-free metrics, per run, per cell, val and test
    def au(yy, k, m): return (roc_auc_score(yy, S[k][m]), average_precision_score(yy, S[k][m]))
    work = Path(tempfile.mkdtemp(prefix="p1blast_"))
    try:
        db = {i: seq[i] for i in ids[tr & (y == 1)]}; hits = {}
        for w in ("val", "test"): hits.update(best_hits(db, {i: seq[i] for i in ids[sp == w]}, work / w))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ev = np.array([hits.get(i, (0.0, 1e9))[1] for i in ids]); B = -np.log10(np.maximum(ev, 1e-300))
    def f1b(e): 
        pr = B[va] >= -np.log10(e); tp = (pr & (y[va] == 1)).sum(); fp = (pr & (y[va] == 0)).sum(); fn = (~pr & (y[va] == 1)).sum()
        return 2 * tp / max(2 * tp + fp + fn, 1)
    e_star = GRID[int(np.argmax([f1b(e) for e in GRID]))]; res["blast_e_star"] = e_star
    res["cells"] = {}
    for name, sm in (("val", va), ("test", te)):
        res["cells"][name] = {}
        for cn, cm in cells.items():
            m = sm & cm; yy = y[m]
            if yy.min() == yy.max(): continue
            runs = np.array([au(yy, k, m) for k in range(len(SEEDS))])
            T = per_run_tables(yy, S[:, m], np.array([tstar]))
            bp = B[m] >= -np.log10(e_star); btp = int((bp & (yy == 1)).sum()); bfp = int((bp & (yy == 0)).sum()); bfn = int((~bp & (yy == 1)).sum())
            res["cells"][name][cn] = {
                "n_pos": int(yy.sum()), "n_neg": int((1 - yy).sum()), "floor": float(yy.mean()),
                "au_roc": [float(runs[:, 0].mean()), float(runs[:, 0].std(ddof=1) / np.sqrt(len(SEEDS)))],
                "au_prc": [float(runs[:, 1].mean()), float(runs[:, 1].std(ddof=1) / np.sqrt(len(SEEDS)))],
                "at_threshold": {k: [float(np.nanmean(T[k][:, 0])), float(np.nanstd(T[k][:, 0], ddof=1) / np.sqrt(len(SEEDS)))] for k in ("precision", "recall", "f1")},
                "blast": {"au_roc": float(roc_auc_score(yy, B[m])), "au_prc": float(average_precision_score(yy, B[m])),
                          "precision": btp / max(btp + bfp, 1), "recall": btp / max(btp + bfn, 1)}}
    (a.out / "phase1.json").write_text(json.dumps(res, indent=2))
    f = lambda t, d=3: f"{t[0]:.{d}f} ± {t[1]:.{d}f}"
    L = [f"v1.2 probe, phase one. {sel}; 5 runs (seeds {SEEDS}); chosen threshold (best mean val F1) = {tstar}; BLASTp e* = {e_star:.3g}\n"]
    L.append("Threshold table, mean ± SE over 5 runs (thresholds 0.1 to 0.9; full 0.001-0.999 grid in the CSVs)")
    for name in ("val", "test"):
        L.append(f"\n{name.upper()}")
        L.append(f"{'thr':>5s} | {'precision':>15s} {'recall (TPR)':>15s} {'F1':>15s} {'FPR':>15s}")
        c = tabs[name]["cols"]
        for t in np.round(np.arange(0.1, 0.91, 0.1), 2):
            i = int(np.argmin(np.abs(THR - t)))
            L.append(f"{t:5.1f} | " + " ".join(f"{c[k+'_mean'][i]:7.3f} ± {c[k+'_se'][i]:.3f}" for k in ("precision", "recall", "f1", "fpr")))
    L.append(f"\nCHOSEN threshold {tstar}: ")
    for name in ("val", "test"):
        d = res["at_threshold"][name]; L.append(f"  {name:5s} precision {f(d['precision'])}  recall {f(d['recall'])}  F1 {f(d['f1'])}  FPR {f(d['fpr'])}  TP {d['tp'][0]:.1f} FP {d['fp'][0]:.1f} FN {d['fn'][0]:.1f} TN {d['tn'][0]:.1f}")
    L.append("\nPrecision targets (threshold = lowest grid value reaching the target on mean val precision):")
    for p, d in res["precision_targets"].items():
        L.append(f"  >= {p}%: thr {d['threshold']:.3f} | val precision {f(d['val']['precision'])} recall {f(d['val']['recall'])} | TEST precision {f(d['test']['precision'])} recall {f(d['test']['recall'])}")
    for name in ("val", "test"):
        L.append(f"\nAU-ROC / AU-PRC, {name.upper()} (probe: mean ± SE over 5 runs; floor = prevalence)")
        L.append(f"{'cell':48s}{'n+':>4s}{'n-':>5s}{'floor':>6s} | {'AU-ROC':>15s} {'AU-PRC':>15s} | {'BLAST ROC':>9s} {'PRC':>6s} | at threshold: P / R / F1")
        for cn, d in res["cells"][name].items():
            o = d["at_threshold"]
            L.append(f"{cn:48s}{d['n_pos']:4d}{d['n_neg']:5d}{d['floor']:6.2f} | {f(d['au_roc'],4):>15s} {f(d['au_prc'],4):>15s} | {d['blast']['au_roc']:9.4f} {d['blast']['au_prc']:6.3f} | {o['precision'][0]:.3f} / {o['recall'][0]:.3f} / {o['f1'][0]:.3f}")
    (a.out / "report.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
