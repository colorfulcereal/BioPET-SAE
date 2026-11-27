"""Phase-one report for the v1.2 polyester probe: validation and test, 5 seeds, BLASTp alongside.

For every split (val, test) and cell it reports AU-ROC and AU-PRC (with the prevalence floor),
the operating point at the VAL best-F1 threshold, and recall at precision >= 95/90/85/80 in two ways:
  best-case   threshold chosen on the evaluated split itself (an oracle; upper bound);
  carried     threshold chosen on VAL as the lowest score reaching that precision on val, applied
              unchanged to test (what a user could actually do; precision on test may fall short).
Probe numbers are mean +/- SE over seeds 42-46 (SE = sd/sqrt(5), training randomness only);
intervals are stratified component bootstraps on the seed-mean score.
Hyperparameters come from `results/v1_2/cells/sweep/sweep.json` (val log-loss).

Usage:  uv run python -m biopet_sae.polyester_report_v12
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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

from biopet_sae.polyester_blast import GRID, best_hits
from biopet_sae.polyester_probe import best_f1_threshold, comp_boot

SEEDS = [42, 43, 44, 45, 46]
PRECS = [0.95, 0.90, 0.85, 0.80]


def best_case_recall(y, s, p):
    pr, rc, _ = precision_recall_curve(y, s)
    ok = pr[:-1] >= p
    return float(rc[:-1][ok].max()) if ok.any() else 0.0


def carried(yv, sv, yt, st, p):
    """Lowest val score whose val precision >= p, applied to test: (threshold, test precision, test recall)."""
    pr, rc, th = precision_recall_curve(yv, sv)
    ok = np.flatnonzero(pr[:-1] >= p)
    if not len(ok): return None
    t = th[ok[0]]                      # thresholds ascend, so the first index is the lowest score
    pred = st >= t
    tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum())
    return float(t), tp / max(tp + fp, 1), tp / max(int((yt == 1).sum()), 1)


def at_thr(y, s, t):
    pred = s >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12)}


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path("data/release/biopet_sae_dataset_v1_2.tsv"))
    ap.add_argument("--embeddings", type=Path, default=Path("data/embeddings_650M_v1_2/embeddings.npz"))
    ap.add_argument("--sweep", type=Path, default=Path("results/v1_2/cells/sweep/sweep.json"))
    ap.add_argument("--out", type=Path, default=Path("results/v1_2/report"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    sel = json.loads(a.sweep.read_text())["selected"]
    C, eta = float(sel.split(",")[0][2:]), float(sel.split(",")[1][4:])
    rows = list(csv.DictReader(a.release.open(newline=""), delimiter="\t"))
    emb = np.load(a.embeddings, allow_pickle=True)
    pos = {s: i for i, s in enumerate(emb["seq_ids"].astype(str))}
    X = emb["max_L33"][[pos[r["seq_id"]] for r in rows]]
    ids = np.array([r["seq_id"] for r in rows]); y = np.array([int(r["y"]) for r in rows])
    sp = np.array([r["split"] for r in rows]); comp = np.array([r["component"] for r in rows])
    grp = np.array([r["negative_group"] for r in rows]); dom = np.array([r["domain"] for r in rows])
    seq = {r["seq_id"]: r["sequence"] for r in rows}
    tr, va, te = sp == "train", sp == "val", sp == "test"
    sc = StandardScaler().fit(X[tr])
    S = []
    for sd in SEEDS:
        clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1.0 / (C * tr.sum()), learning_rate="constant", eta0=eta,
                            max_iter=5000, tol=None, class_weight="balanced", random_state=sd).fit(sc.transform(X[tr]), y[tr])
        S.append(clf.predict_proba(sc.transform(X))[:, 1])
    S = np.array(S); smean = S.mean(0)
    thr = [best_f1_threshold(y[va], S[k][va])[0] for k in range(len(SEEDS))]
    # BLASTp, e-value cut-off chosen on val by F1 (same rule as before)
    work = Path(tempfile.mkdtemp(prefix="v12rep_"))
    try:
        db = {i: seq[i] for i in ids[tr & (y == 1)]}; hits = {}
        for w in ("val", "test"): hits.update(best_hits(db, {i: seq[i] for i in ids[sp == w]}, work / w))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ev = np.array([hits.get(i, (0.0, 1e9))[1] for i in ids]); B = -np.log10(np.maximum(ev, 1e-300))
    f1s = [at_thr(y[va], -np.log10(np.maximum(ev[va], 1e-300)), -np.log10(e))["f1"] for e in GRID]
    e_star = GRID[int(np.argmax(f1s))]; bthr = -np.log10(e_star)

    hard = grp == "ab_hydrolase_hard_negative"; bact = dom == "Bacteria"
    cells = {"ALL negatives (the whole split)": np.ones(len(y), bool), "hard negatives only": (y == 1) | hard,
             "bacterial hard negatives": (y == 1) | (hard & bact),
             "matched: bacterial pos. vs bacterial hard neg.": ((y == 1) & bact) | (hard & bact)}
    out = {"selected": sel, "seeds": SEEDS, "val_thresholds": thr, "blast_e_star": e_star, "splits": {}}
    for name_s, sm, other in (("val", va, None), ("test", te, va)):
        out["splits"][name_s] = {}
        for cn, cm in cells.items():
            m = sm & cm; yy = y[m]
            if yy.min() == yy.max(): continue
            d = {"n_pos": int(yy.sum()), "n_neg": int((1 - yy).sum()), "floor": float(yy.mean())}
            roc = [roc_auc_score(yy, S[k][m]) for k in range(len(SEEDS))]; prc = [average_precision_score(yy, S[k][m]) for k in range(len(SEEDS))]
            d["probe_roc"], d["probe_prc"] = ms(roc), ms(prc)
            d["probe_roc_ci"], _ = comp_boot(yy, smean[m], comp[m], roc_auc_score)
            d["probe_prc_ci"], _ = comp_boot(yy, smean[m], comp[m], average_precision_score)
            d["blast_roc"], d["blast_prc"] = float(roc_auc_score(yy, B[m])), float(average_precision_score(yy, B[m]))
            op = [at_thr(yy, S[k][m], thr[k]) for k in range(len(SEEDS))]
            d["probe_at_val_threshold"] = {k2: ms([o[k2] for o in op]) for k2 in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")}
            d["blast_at_val_threshold"] = at_thr(yy, B[m], bthr)
            d["best_case_recall"] = {str(int(p * 100)): {"probe": ms([best_case_recall(yy, S[k][m], p) for k in range(len(SEEDS))]),
                                                         "blast": best_case_recall(yy, B[m], p)} for p in PRECS}
            if other is not None and cn.startswith("ALL"):
                cr = {}
                for p in PRECS:
                    r_ = [carried(y[va], S[k][va], yy, S[k][m], p) for k in range(len(SEEDS))]
                    rb = carried(y[va], B[va], yy, B[m], p)
                    cr[str(int(p * 100))] = {"probe_test_precision": ms([x[1] for x in r_]) if all(r_) else None,
                                             "probe_test_recall": ms([x[2] for x in r_]) if all(r_) else None,
                                             "blast_test_precision": rb[1] if rb else None, "blast_test_recall": rb[2] if rb else None}
                d["carried_from_val"] = cr
            out["splits"][name_s][cn] = d
    (a.out / "report.json").write_text(json.dumps(out, indent=2))
    np.savez(a.out / "scores_5_seeds.npz", seq_ids=ids, scores=S, blast=B, y=y, split=sp, domain=dom, group=grp, component=comp)
    f = lambda t: f"{t[0]:.4f}±{t[1]:.4f}"
    L = [f"Probe: ESM-2 650M max-pool L33, {sel}, seeds {SEEDS}; val thresholds {[round(t,3) for t in thr]}; BLASTp e* {e_star:.3g}\n"]
    for sn in ("val", "test"):
        L.append(f"\n=== {sn.upper()} ===")
        L.append(f"{'cell':50s}{'n+':>4s}{'n-':>5s}{'floor':>6s} | {'probe AU-ROC':>15s} {'AU-PRC':>15s} | {'BLAST ROC':>9s} {'PRC':>6s} | 95% CI ROC / PRC (component)")
        for cn, d in out["splits"][sn].items():
            L.append(f"{cn:50s}{d['n_pos']:4d}{d['n_neg']:5d}{d['floor']:6.2f} | {f(d['probe_roc']):>15s} {f(d['probe_prc']):>15s} | {d['blast_roc']:9.4f} {d['blast_prc']:6.3f} | {d['probe_roc_ci']} / {d['probe_prc_ci']}")
        L.append(f"\n  operating point at the val best-F1 threshold ({sn}):")
        for cn, d in out["splits"][sn].items():
            o = d["probe_at_val_threshold"]; b = d["blast_at_val_threshold"]
            L.append(f"   {cn:48s} probe P {o['precision'][0]:.3f} R {o['recall'][0]:.3f} F1 {o['f1'][0]:.3f} (TP {o['tp'][0]:.1f} FP {o['fp'][0]:.1f} FN {o['fn'][0]:.1f} TN {o['tn'][0]:.1f}) | BLAST P {b['precision']:.3f} R {b['recall']:.3f} F1 {b['f1']:.3f}")
        L.append(f"\n  recall at precision >= p, best case (threshold chosen on {sn} itself):")
        L.append(f"   {'cell':48s}" + "".join(f"{'p>='+str(int(p*100))+'% probe / BLAST':>26s}" for p in PRECS))
        for cn, d in out["splits"][sn].items():
            L.append(f"   {cn:48s}" + "".join(f"{f(d['best_case_recall'][str(int(p*100))]['probe']):>16s} / {d['best_case_recall'][str(int(p*100))]['blast']:.3f} " for p in PRECS))
    d = out["splits"]["test"]["ALL negatives (the whole split)"]["carried_from_val"]
    L.append("\n  TEST, threshold carried from val (lowest val score reaching precision p):")
    for p in PRECS:
        c = d[str(int(p * 100))]
        L.append(f"   target precision >= {int(p*100)}%: probe test precision {f(c['probe_test_precision']) if c['probe_test_precision'] else '-'}  recall {f(c['probe_test_recall']) if c['probe_test_recall'] else '-'} | BLASTp precision {c['blast_test_precision'] if c['blast_test_precision'] is None else round(c['blast_test_precision'],3)} recall {c['blast_test_recall'] if c['blast_test_recall'] is None else round(c['blast_test_recall'],3)}")
    (a.out / "report.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
