"""EDA of release v1.2: per split and combined. Writes CSV tables and a markdown summary.

Every number is computed from `data/release/biopet_sae_dataset_v1_2.tsv` (and the pair table for
cross-split identity). Tables go to `data/release/eda_v1_2/`.

Usage:  uv run python -m biopet_sae.eda_v12
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SPL = ["train", "val", "test"]
GORD = ["positive", "ab_hydrolase_hard_negative", "ec311_wrong_fold",
        "tRNA_processing_no_ser_nucleophile", "naive_control"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path("data/release/biopet_sae_dataset_v1_2.tsv"))
    ap.add_argument("--pairs", type=Path, default=Path("data/raw/hard_negatives_v12/pair_identities.tsv"))
    ap.add_argument("--out", type=Path, default=Path("data/release/eda_v1_2"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.release, sep="\t", dtype=str, keep_default_na=False)
    d["group"] = np.where(d["y"] == "1", "positive", d["negative_group"])
    d["seq_len"] = d["seq_len"].astype(int)
    d["phylum"] = d["phylum"].replace("", "(unknown)")
    S = lambda df: df.copy()

    def tab(df, row, name, top=None):
        t = pd.crosstab(df[row], df["split"]).reindex(columns=SPL, fill_value=0)
        t["all"] = t.sum(axis=1)
        t = t.sort_values("all", ascending=False)
        if top: t = t.head(top)
        t.to_csv(a.out / name); return t

    comp = pd.crosstab(d["group"], d["split"]).reindex(index=GORD, columns=SPL, fill_value=0)
    comp["all"] = comp.sum(axis=1); comp.loc["total"] = comp.sum()
    comp.to_csv(a.out / "composition_by_group_and_split.csv")
    prev = pd.DataFrame({s: [int(comp.loc["positive", s]), int(comp.loc["total", s] - comp.loc["positive", s]),
                             round(comp.loc["positive", s] / comp.loc["total", s], 3),
                             int(comp.loc["ab_hydrolase_hard_negative", s])] for s in SPL + ["all"]},
                        index=["positives", "negatives", "prevalence", "hard_negatives"]).T
    prev.to_csv(a.out / "prevalence_by_split.csv")

    P = d[d["y"] == "1"]
    cl = tab(P, "class_label", "positives_by_class.csv")
    src = tab(P, "source", "positives_by_source.csv")
    ver = tab(P, "activity_verified", "positives_by_verified.csv")
    sub = Counter(); subs = defaultdict(Counter)
    for _, r in P.iterrows():
        for s in r["substrates"].split("|"):
            if s: subs[s][r["split"]] += 1
    st = pd.DataFrame({k: {**{s: v[s] for s in SPL}, "all": sum(v.values())} for k, v in subs.items()}).T
    st = st.sort_values("all", ascending=False); st.to_csv(a.out / "positives_by_substrate.csv")
    tab(P, "phylum", "positives_by_phylum.csv", 15)

    H = d[d["group"] == "ab_hydrolase_hard_negative"]
    hp = tab(H, "phylum", "hard_negatives_by_phylum.csv")
    tab(H, "organism", "hard_negatives_by_organism_top30.csv", 30)
    ecs = defaultdict(Counter)
    for _, r in H.iterrows():
        for e in r["ec"].split(";"):
            if e: ecs[e][r["split"]] += 1
    ect = pd.DataFrame({k: {**{s: v[s] for s in SPL}, "all": sum(v.values())} for k, v in ecs.items()}).T
    ect = ect.sort_values("all", ascending=False); ect.to_csv(a.out / "hard_negatives_by_EC.csv")
    pf = defaultdict(Counter)
    for _, r in H.iterrows():
        for e in r["pfam"].split(";"):
            if e: pf[e][r["split"]] += 1
    pft = pd.DataFrame({k: {**{s: v[s] for s in SPL}, "all": sum(v.values())} for k, v in pf.items()}).T
    pft = pft.sort_values("all", ascending=False); pft.head(30).to_csv(a.out / "hard_negatives_by_Pfam_top30.csv")
    H2 = H.assign(batch_=H["batch"].replace({"v1": "existing", "v1.1": "existing", "v1.2": "new"}))
    tab(H2, "batch_", "hard_negatives_new_vs_existing.csv")

    dg = pd.crosstab([d["domain"], d["group"]], d["split"]).reindex(columns=SPL, fill_value=0)
    dg["all"] = dg.sum(axis=1); dg.to_csv(a.out / "domain_by_group_and_split.csv")
    dgs = pd.crosstab(d["group"], d["domain"]).reindex(index=GORD, fill_value=0)
    dgs["all"] = dgs.sum(axis=1); dgs.to_csv(a.out / "domain_by_group.csv")
    # lengths
    rows = []
    for g in GORD:
        for s in SPL + ["all"]:
            x = d[(d["group"] == g) & ((d["split"] == s) | (s == "all"))]["seq_len"]
            if len(x): rows.append([g, s, len(x), x.min(), int(x.quantile(.25)), int(x.median()), int(x.quantile(.75)), x.max(), round(x.mean(), 1)])
    pd.DataFrame(rows, columns=["group", "split", "n", "min", "q25", "median", "q75", "max", "mean"]).to_csv(a.out / "length_summary.csv", index=False)
    bins = list(range(0, 1100, 100))
    d["len_bin"] = pd.cut(d["seq_len"], bins=bins, right=False).astype(str)
    pd.crosstab(d["len_bin"], d["group"]).reindex(columns=GORD, fill_value=0).to_csv(a.out / "length_histogram_by_group.csv")

    # components
    cr = []
    for s in SPL + ["all"]:
        x = d[(d["split"] == s) | (s == "all")]
        cc = x.groupby("component").agg(n=("seq_id", "size"), pos=("y", lambda v: (v == "1").sum()),
                                         hard=("group", lambda v: (v == "ab_hydrolase_hard_negative").sum()))
        p = cc[cc["pos"] > 0]; h = cc[cc["hard"] > 0]
        cr.append([s, len(cc), len(p), len(h), int(cc["n"].max()), int(p["pos"].max()) if len(p) else 0,
                   round(p["pos"].max() / p["pos"].sum(), 3) if len(p) else 0,
                   int(h["hard"].max()) if len(h) else 0, round(h["hard"].max() / h["hard"].sum(), 3) if len(h) else 0])
    pd.DataFrame(cr, columns=["split", "components", "components_with_positives", "components_with_hard_negatives",
                              "largest_component_size", "largest_positive_component", "its_share_of_positives",
                              "largest_hard_negative_component", "its_share_of_hard_negatives"]).to_csv(a.out / "components_by_split.csv", index=False)

    # nearest sequence in another split (from the realigned pair table; none = below the 0.30 local-identity prefilter)
    sp = dict(zip(d["seq_id"], d["split"])); best = defaultdict(float)
    for r in csv.DictReader(a.pairs.open(newline=""), delimiter="\t"):
        x, y, v = r["a"], r["b"], float(r["identity"])
        if x in sp and y in sp and sp[x] != sp[y]:
            best[x] = max(best[x], v); best[y] = max(best[y], v)
    d["nearest_other_split"] = d["seq_id"].map(best).fillna(0.0)
    ni = d.groupby(["split", "group"])["nearest_other_split"].describe(percentiles=[.5, .9, .99])[["count", "50%", "90%", "99%", "max"]].round(3)
    ni.to_csv(a.out / "nearest_identity_to_other_splits.csv")
    maxcross = d["nearest_other_split"].max()

    # phylum x group
    pd.crosstab(d["phylum"], d["group"]).reindex(columns=GORD, fill_value=0).assign(all=lambda t: t.sum(axis=1)).sort_values("all", ascending=False).head(20).to_csv(a.out / "phylum_by_group_all.csv")

    # ---- markdown summary
    L = []
    w = L.append
    w("# EDA of dataset v1.2 (generated by `src/biopet_sae/eda_v12.py`)\n")
    w(f"Source file: `{a.release}`; tables in `{a.out}/`.\n")
    w("## Composition\n"); w(comp.to_markdown() + "\n")
    w("## Prevalence\n"); w(prev.to_markdown() + "\n")
    w("## Positives by class\n"); w(cl.to_markdown() + "\n")
    w("## Positives by source and verification\n"); w(src.to_markdown() + "\n"); w(ver.to_markdown() + "\n")
    w("## Positives by plastic (an enzyme can appear under several plastics)\n"); w(st.to_markdown() + "\n")
    w("## Domain of life by group (a possible shortcut: positives are mostly bacterial)\n"); w(dgs.to_markdown() + "\n")
    w("Domain by group and split:\n"); w(dg.to_markdown() + "\n")
    w("## Hard negatives\n"); w(pd.read_csv(a.out / "hard_negatives_new_vs_existing.csv", index_col=0).to_markdown() + "\n")
    w("By phylum:\n"); w(hp.head(12).to_markdown() + "\n")
    w("By EC number (top 10):\n"); w(ect.head(10).to_markdown() + "\n")
    w("By Pfam (top 10):\n"); w(pft.head(10).to_markdown() + "\n")
    w("## Homology components\n"); w(pd.read_csv(a.out / "components_by_split.csv").to_markdown(index=False) + "\n")
    w("## Sequence length (residues)\n"); w(pd.read_csv(a.out / "length_summary.csv").query("split=='all'").to_markdown(index=False) + "\n")
    w("## Nearest sequence in another split (global identity)\n")
    w(f"Maximum over the dataset: {maxcross:.4f} (bound 0.40). Sequences with no entry in the pair table have local identity < 0.30 to every other-split sequence.\n")
    w(ni.to_markdown() + "\n")
    (a.out / "EDA_v1_2.md").write_text("\n".join(L))
    print("\n".join(L[:14]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
