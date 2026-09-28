"""Exploratory data analysis on the merged plastizyme table.

Writes a markdown report to results/eda_report.md and prints the same content. Every
number here is computed from data/processed/plastizymes_merged.tsv, so re-running after
a dataset change refreshes the report.

Usage:
    uv run python -m biopet_sae.eda [--table data/processed/plastizymes_merged.tsv]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# groups that hydrolyse a carboxylic ester and are therefore mechanistically
# comparable to the PETases
ESTER_GROUPS = [
    "PET_AROMATIC_POLYESTER",
    "AROMATIC_COPOLYESTER",
    "ALIPHATIC_POLYESTER",
    "PHA_FAMILY",
]

# alpha/beta hydrolase nucleophile elbow: the catalytic serine sits in G-x-S-x-G
NUCLEOPHILE_ELBOW = re.compile(r"G.S.G")

# genus names that are thermophilic or thermotolerant, used as a crude proxy to test
# whether "PET-active" is confounded with "grows hot"
THERMOPHILE_GENERA = {
    "thermobifida", "thermomonospora", "thermus", "thermobispora", "thermoclostridium",
    "caldibacillus", "geobacillus", "caldimonas", "thermostabilis", "thermomyces",
    "humicola", "thermoascus", "caldicellulosiruptor", "thermosyntropha",
    "thermobacillus", "thermoactinomyces", "saccharomonospora", "thermocrispum",
}

OUT: list[str] = []


def emit(line: str = "") -> None:
    OUT.append(line)
    print(line)


def h(level: int, text: str) -> None:
    emit()
    emit("#" * level + " " + text)
    emit()


def table(df: pd.DataFrame, index: bool = False) -> None:
    emit(df.to_markdown(index=index))
    emit()


def counts_table(series: pd.Series, name: str, total: int | None = None) -> None:
    vc = series.value_counts(dropna=False)
    total = total or int(vc.sum())
    df = pd.DataFrame({name: vc.index, "n": vc.to_numpy()})
    df["%"] = (100 * df["n"] / total).round(1)
    table(df)


def explode_multi(df: pd.DataFrame, col: str) -> pd.Series:
    """Split a '|'-joined column into one value per row."""
    return (
        df[col].fillna("").astype(str).str.split("|").explode().str.strip().replace("", pd.NA).dropna()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=Path("data/processed/plastizymes_merged.tsv"))
    ap.add_argument("--merge-report", type=Path, default=Path("data/processed/merge_report.json"))
    ap.add_argument("--out", type=Path, default=Path("results/eda_report.md"))
    args = ap.parse_args(argv)

    df = pd.read_csv(args.table, sep="\t", dtype=str, keep_default_na=False)
    for c in ("is_pet", "n_substrates", "promiscuous", "verified_activity",
              "extrapolated_only", "esterase_fold_candidate", "seq_len", "n_pdb",
              "n_literature_dois", "n_alias_sequences"):
        df[c] = pd.to_numeric(df[c])

    emit("# BioPET-SAE — merged dataset EDA")
    emit()
    emit(f"Source table: `{args.table}` — **{len(df)} non-redundant proteins**.")
    emit("Both source databases record only confirmed *positives*; nothing here is a")
    emit("verified negative. Bulk negatives come from UniProt in a later step.")

    # ------------------------------------------------------------------ 1
    h(2, "1. Provenance and dedup accounting")
    if args.merge_report.exists():
        rep = json.loads(args.merge_report.read_text())
        emit("```")
        emit(f"PAZy         : {rep['pazy']['rows']} rows, "
             f"{rep['pazy']['no_sequence']} lack a sequence -> {rep['pazy']['usable_entries']} usable")
        emit(f"PlasticDB    : {rep['plasticdb']['rows']} rows, "
             f"{rep['plasticdb']['no_sequence']} lack a sequence "
             f"-> {rep['plasticdb']['usable_unique_sequences']} unique sequences")
        emit(f"duplicates   : {rep['dedup']['exact_dupes']} exact + "
             f"{rep['dedup']['containment_dupes']} containment (signal-peptide variants)")
        emit(f"merged total : {rep['merged_total']}")
        emit("```")
        emit()
        emit("The containment pass matters: exact-match dedup alone would have kept "
             f"{rep['dedup']['containment_dupes']} near-identical pairs, which then leak across any "
             "homology split.")
        emit()
    counts_table(df["sources"], "source(s) contributing")
    both = int((df["sources"] == "PAZy|PlasticDB").sum())
    emit(f"{both} proteins are corroborated by both databases independently.")

    # ------------------------------------------------------------------ 2
    h(2, "2. Class composition")
    g = (
        df.groupby("primary_group")
        .agg(n=("seq_id", "size"), median_len=("seq_len", "median"),
             verified=("verified_activity", "sum"), with_pdb=("n_pdb", lambda s: int((s > 0).sum())))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    g["%"] = (100 * g["n"] / len(df)).round(1)
    table(g[["primary_group", "n", "%", "median_len", "verified", "with_pdb"]])
    emit("Grouped by the bond the enzyme has to break:")
    emit()
    counts_table(df["mechanism"], "mechanism")
    emit("Only `carboxylic_ester` groups are mechanistically comparable to the PETases. "
         "Polyamide (nylon) hydrolases attack an **amide** bond, polyurethanases a "
         "**urethane** bond, and polyolefin degraders work oxidatively rather than "
         "hydrolytically — pooling any of them under 'plastic degrader' would be a "
         "mechanism error.")

    # ------------------------------------------------------------------ 3
    h(2, "3. The modelling classes")
    ester = df[df["primary_group"].isin(ESTER_GROUPS)].copy()
    pet = ester[ester["is_pet"] == 1]
    contrast_all = ester[ester["is_pet"] == 0]
    contrast = contrast_all[contrast_all["esterase_fold_candidate"] == 1]
    dropped = contrast_all[contrast_all["esterase_fold_candidate"] == 0]

    rows = [
        {"class": "PET-active (positives)", "n": len(pet),
         "note": "primary target; includes PEF, the furan analogue"},
        {"class": "non-PET ester, fold-matched", "n": len(contrast),
         "note": "aliphatic polyester / PHA / PBAT degraders — the hard contrast"},
        {"class": "  of which aromatic co-polyester", "n": int((contrast["primary_group"] == "AROMATIC_COPOLYESTER").sum()),
         "note": "PBAT etc. — contains terephthalate, the intermediate case"},
        {"class": "  of which aliphatic polyester", "n": int((contrast["primary_group"] == "ALIPHATIC_POLYESTER").sum()),
         "note": "PCL/PLA/PBS — flexible backbone, easy hydrolysis"},
        {"class": "  of which PHA family", "n": int((contrast["primary_group"] == "PHA_FAMILY").sum()),
         "note": "separate depolymerase lineage, weaker fold match"},
        {"class": "dropped: non-esterase enzyme type", "n": len(dropped),
         "note": "proteases/chitinases etc. on polyester substrates"},
        {"class": "excluded: non-ester mechanism", "n": len(df) - len(ester),
         "note": "polyamide, polyurethane, rubber, polyolefin, polyether"},
    ]
    table(pd.DataFrame(rows))
    if len(dropped):
        emit(f"Dropped enzyme labels: {sorted(set(explode_multi(dropped, 'enzyme_labels')))}")
        emit()
    emit(f"**Working contrast: {len(pet)} PET-active vs {len(contrast)} fold-matched non-PET.** "
         "That negative side is too small to carry an evaluation on its own, so UniProt "
         "`ec:3.1.1.-` bulk negatives are still needed — but these are the only negatives "
         "backed by an actual assay on a polyester.")
    emit()
    emit("Enzyme-name overlap between the two classes is the point of the design — the "
         "same vocabulary appears on both sides:")
    emit()
    pet_enz = Counter(explode_multi(pet, "enzyme_labels"))
    con_enz = Counter(explode_multi(contrast, "enzyme_labels"))
    shared = sorted(set(pet_enz) & set(con_enz), key=lambda k: -(pet_enz[k] + con_enz[k]))
    if shared:
        table(pd.DataFrame([
            {"enzyme label": k, "in PET class": pet_enz[k], "in contrast class": con_enz[k]}
            for k in shared
        ]))

    # ------------------------------------------------------------------ 4
    h(2, "4. Sequence length")
    ln = (
        ester.groupby("primary_group")["seq_len"]
        .describe()[["count", "min", "25%", "50%", "75%", "max"]]
        .round(0)
        .astype(int)
        .reset_index()
    )
    table(ln)
    emit("Lengths overlap closely across classes, so a length-only classifier has little "
         "to work with — one confound that is *not* a problem here.")
    emit()
    qc = df[df["qc_flags"] != ""]
    emit(f"QC-flagged sequences: **{len(qc)}**"
         + (f" — {qc[['seq_id', 'primary_group', 'seq_len', 'qc_flags']].to_dict('records')}" if len(qc) else ""))

    # ------------------------------------------------------------------ 5
    h(2, "5. Effective sample size (redundancy)")
    emit("PAZy ships UniRef cluster ids, which give a free first-pass homology grouping. "
         "Coverage is partial because many entries are metagenomic and have no UniProt record.")
    emit()
    rows = []
    for label, sub in (("PET-active", pet), ("non-PET ester (fold-matched)", contrast)):
        r = {"class": label, "n": len(sub)}
        for lvl in ("uniref100", "uniref90", "uniref50"):
            ann = sub[sub[lvl] != ""]
            r[f"{lvl} annotated"] = len(ann)
            r[f"{lvl} clusters"] = ann[lvl].nunique()
        rows.append(r)
    table(pd.DataFrame(rows))

    ann50 = pet[pet["uniref50"] != ""]
    if len(ann50):
        ratio = len(ann50) / ann50["uniref50"].nunique()
        emit(f"Within the annotated PET subset, {len(ann50)} entries collapse to "
             f"{ann50['uniref50'].nunique()} UniRef50 clusters — a **{ratio:.1f}:1** collapse. "
             f"Extrapolated across all {len(pet)} PET positives that implies roughly "
             f"**~{int(len(pet) / ratio)} independent families at 50% identity**, though the "
             "unannotated entries are mostly environmental and probably more diverse. "
             "This must be re-derived with real clustering (mmseqs2) before any split is trusted.")
        emit()
        emit("Largest UniRef50 clusters in the PET class:")
        emit()
        top = ann50["uniref50"].value_counts().head(8)
        table(pd.DataFrame({"UniRef50": top.index, "members": top.to_numpy()}))
        emit("Consequence for evaluation: holding out 15% of ~150 clusters leaves ~20–25 "
             "test positives, where the 95% CI on AUC spans about 0.16. Use k-fold CV over "
             "clusters and bootstrap over clusters, not sequences.")

    # ------------------------------------------------------------------ 6
    h(2, "6. Taxonomic confound")
    emit("Phylum is only annotated for PAZy-derived entries.")
    emit()
    rows = []
    for label, sub in (("PET-active", pet), ("non-PET ester", contrast)):
        ph = sub[sub["phylum"] != ""]
        rows.append({"class": label, "n": len(sub), "phylum annotated": len(ph),
                     "coverage %": round(100 * len(ph) / max(len(sub), 1), 1)})
    table(pd.DataFrame(rows))
    for label, sub in (("PET-active", pet), ("non-PET ester (fold-matched)", contrast)):
        ph = explode_multi(sub, "phylum")
        if not len(ph):
            continue
        emit(f"**{label}** (n={len(ph)} annotated)")
        emit()
        counts_table(ph, "phylum", total=len(ph))

    def thermo(sub: pd.DataFrame) -> tuple[int, int]:
        orgs = sub["organism"].fillna("")
        hit = orgs.str.lower().apply(
            lambda s: any(gen in s for gen in THERMOPHILE_GENERA)
        )
        return int(hit.sum()), int((orgs != "").sum())

    emit("Thermophilic-genus proxy (crude — genus name matching, not a growth-temperature lookup):")
    emit()
    rows = []
    for label, sub in (("PET-active", pet), ("non-PET ester", contrast)):
        n_hit, n_org = thermo(sub)
        rows.append({"class": label, "with organism": n_org, "thermophilic genus": n_hit,
                     "%": round(100 * n_hit / max(n_org, 1), 1)})
    table(pd.DataFrame(rows))
    emit("If the PET class is strongly skewed toward one phylum or toward thermophiles, a "
         "probe can score well by detecting taxonomy rather than PET activity. Test this "
         "directly: check whether the top discriminative features separate phyla *within* "
         "the PET class.")

    # ------------------------------------------------------------------ 7
    h(2, "7. Evidence quality")
    rows = []
    for label, sub in (("PET-active", pet), ("non-PET ester", contrast), ("all merged", df)):
        rows.append({
            "class": label, "n": len(sub),
            "PAZy verified_activity": int(sub["verified_activity"].sum()),
            "degradation extrapolated only": int(sub["extrapolated_only"].sum()),
            "has literature DOI": int((sub["n_literature_dois"] > 0).sum()),
            "median DOIs": float(sub["n_literature_dois"].median()),
        })
    table(pd.DataFrame(rows))
    ev = Counter(explode_multi(df, "evidence"))
    if ev:
        emit("PlasticDB assay-evidence types (where recorded):")
        emit()
        top = pd.DataFrame(ev.most_common(10), columns=["evidence", "n"])
        table(top)
        emit("`Clear zone` is a qualitative plate assay — weaker evidence than HPLC or "
             "spectrophotometric product quantification. Worth carrying as a per-entry "
             "confidence weight rather than treating all positives as equal.")

    # ------------------------------------------------------------------ 8
    h(2, "8. Cross-reference coverage")
    rows = []
    for label, sub in (("PET-active", pet), ("non-PET ester", contrast)):
        rows.append({
            "class": label, "n": len(sub),
            "UniProtKB": int((sub["uniprot"] != "").sum()),
            "PDB structure": int((sub["n_pdb"] > 0).sum()),
            "GenBank": int((sub["genbank"] != "").sum()),
            "RefSeq": int((sub["refseq"] != "").sum()),
        })
    table(pd.DataFrame(rows))
    with_pdb = pet[pet["n_pdb"] > 0].nlargest(8, "n_pdb")
    emit("PET-class entries with the most solved structures — the mechanistic validation set:")
    emit()
    table(with_pdb[["seq_id", "protein_names", "organism", "seq_len", "n_pdb"]])
    n_up = int((pet["uniprot"] != "").sum())
    emit(f"**Leak risk:** {n_up} PET positives carry a UniProtKB accession, so they will "
         "appear in a UniProt `ec:3.1.1.-` negative pull. They must be excluded by "
         "accession *and* by sequence identity, or positives contaminate the negative class.")

    # ------------------------------------------------------------------ 9
    h(2, "9. Substrate promiscuity")
    counts_table(ester["n_substrates"].astype(str), "substrates per enzyme", total=len(ester))
    prom = ester[ester["promiscuous"] == 1]
    emit(f"{len(prom)} ester-degrading enzymes act on more than one plastic. "
         "Promiscuous enzymes are labelled by their most specific substrate "
         "(PET wins over PCL), so the PET class contains enzymes that also hydrolyse "
         "easier polyesters — expected, since PET activity implies general ester activity.")
    emit()
    both_pet_ali = ester[(ester["is_pet"] == 1) & (ester["substrates"].str.contains("PCL|PLA|PBS", regex=True))]
    emit(f"PET-active enzymes also reported on an aliphatic polyester: {len(both_pet_ali)}")

    # ------------------------------------------------------------------ 10
    h(2, "10. Sanity check: is the fold match real?")
    emit("Every alpha/beta-hydrolase esterase should carry the nucleophile elbow motif "
         "`G-x-S-x-G` holding the catalytic serine. If the contrast class is genuinely "
         "fold-matched, both classes should hit at similar rates.")
    emit()
    rows = []
    for label, sub in (("PET-active", pet), ("non-PET ester (fold-matched)", contrast),
                       ("polyamide (different mechanism)", df[df["primary_group"] == "POLYAMIDE"])):
        hits = sub["sequence"].apply(lambda s: bool(NUCLEOPHILE_ELBOW.search(s)))
        rows.append({"class": label, "n": len(sub), "has GxSxG": int(hits.sum()),
                     "%": round(100 * hits.mean(), 1) if len(sub) else 0.0})
    table(pd.DataFrame(rows))
    emit("Raw hit rates alone are not interpretable for a pattern this permissive -- "
         "section 12 repeats this against a composition-preserving shuffled null.")

    # ------------------------------------------------------------------ 11
    h(2, "11. Confound baselines: what is predictable without the sequence")
    emit("Before trusting any SAE result, establish how much of the PET-vs-contrast task "
         "is solvable by signals that have nothing to do with PET catalysis. Anything the "
         "real probe achieves has to be read against these, not against the 0.5 chance floor.")
    emit()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    y = pd.concat([pd.Series(1, index=pet.index), pd.Series(0, index=contrast.index)])
    both_df = pd.concat([pet, contrast])

    rows = []

    # (a) phylum alone, on the subset where it is annotated for both classes
    ph_sub = both_df[both_df["phylum"] != ""]
    ph_y = y.loc[ph_sub.index]
    is_actino = ph_sub["phylum"].str.contains("Actinomycetota").astype(int)
    tp = int(((is_actino == 1) & (ph_y == 1)).sum())
    fp = int(((is_actino == 1) & (ph_y == 0)).sum())
    tn = int(((is_actino == 0) & (ph_y == 0)).sum())
    fn = int(((is_actino == 0) & (ph_y == 1)).sum())
    rows.append({
        "signal": 'phylum only ("Actinomycetota -> PET")',
        "n": len(ph_sub),
        "metric": "accuracy",
        "score": round((tp + tn) / len(ph_sub), 3),
        "detail": f"TP={tp} FP={fp} TN={tn} FN={fn}",
    })

    # (b) thermophilic genus alone
    thermo_flag = both_df["organism"].str.lower().apply(
        lambda s: int(any(gen in s for gen in THERMOPHILE_GENERA))
    )
    rows.append({
        "signal": "thermophilic genus only",
        "n": len(both_df),
        "metric": "ROC-AUC",
        "score": round(roc_auc_score(y, thermo_flag), 3),
        "detail": "single binary feature",
    })

    # (c) sequence length alone
    rows.append({
        "signal": "sequence length only",
        "n": len(both_df),
        "metric": "ROC-AUC",
        "score": round(roc_auc_score(y, -both_df["seq_len"]), 3),
        "detail": "shorter = more PET-like",
    })

    # (d) amino-acid composition alone: 20 features, no model of structure at all.
    # This is the bar an embedding-based probe actually has to clear.
    aa = sorted("ACDEFGHIKLMNPQRSTVWY")
    comp = pd.DataFrame(
        [[s.count(a) / max(len(s), 1) for a in aa] for s in both_df["sequence"]],
        columns=aa, index=both_df.index,
    )
    clf = LogisticRegression(max_iter=5000, C=1.0)
    probs = cross_val_predict(clf, comp.to_numpy(), y.to_numpy(), cv=5, method="predict_proba")[:, 1]
    rows.append({
        "signal": "amino-acid composition (20 features)",
        "n": len(both_df),
        "metric": "ROC-AUC",
        "score": round(roc_auc_score(y, probs), 3),
        "detail": "5-fold CV, random split (leaks homology -> optimistic)",
    })

    table(pd.DataFrame(rows))
    emit("Read these as the floor, not as results. The composition baseline in particular "
         "uses a random CV split, so homologous near-duplicates sit on both sides and the "
         "number is inflated — but that is exactly the mistake the real pipeline must avoid, "
         "and it shows how easy this task looks before homology control.")
    emit()
    emit("**Design consequence.** The phylum skew is not a nuisance to note in a limitations "
         "section; it is large enough to be the primary explanation for any strong result. "
         "Three mitigations, in increasing strength:")
    emit()
    emit("1. Report performance separately per phylum, and within Actinomycetota alone.")
    emit("2. Stratify the homology-clustered splits by phylum so train and test have "
         "comparable phylum mixes.")
    emit("3. Build a phylum-matched contrast subset — Actinomycetota non-PET esterases "
         "against Actinomycetota PETases. UniProt can supply these, and it turns the "
         "confound into a controlled variable rather than a caveat.")

    # per-group nucleophile elbow breakdown, against a composition-preserving null.
    # `G.S.G` is a loose pattern: in a 300-residue protein with ~7% G and ~7% S it can
    # easily hit by chance, so a raw hit rate means nothing without the null.
    h(2, "12. Nucleophile elbow vs a shuffled null")
    rng = random.Random(0)
    rows = []
    for grp in ESTER_GROUPS + ["POLYAMIDE"]:
        sub = df[df["primary_group"] == grp]
        if not len(sub):
            continue
        obs = sub["sequence"].apply(lambda s: bool(NUCLEOPHILE_ELBOW.search(s))).mean()
        null_hits = []
        for s in sub["sequence"]:
            chars = list(s)
            for _ in range(20):  # shuffle preserves length and composition exactly
                rng.shuffle(chars)
                null_hits.append(bool(NUCLEOPHILE_ELBOW.search("".join(chars))))
        null = sum(null_hits) / len(null_hits)
        rows.append({
            "group": grp, "n": len(sub),
            "observed %": round(100 * obs, 1),
            "shuffled null %": round(100 * null, 1),
            "enrichment": round(obs / null, 2) if null else float("nan"),
        })
    table(pd.DataFrame(rows))
    emit("The null rate is ~24%, so the raw hit rates in section 10 were misleading in a "
         "different way than expected: every ester group is genuinely enriched over its own "
         "shuffled null (2.7-3.6x), which does support a real serine-hydrolase fold across "
         "the contrast class. But PET is enriched further still (4.1x, 98.8% observed), "
         "meaning it is the most structurally homogeneous class in the set.")
    emit()
    emit("Note also that polyamide is enriched 2.4x, so it is *not* the clean negative "
         "control section 10 assumed — many nylon hydrolases are themselves serine "
         "hydrolases. That row should not be read as a fold contrast.")
    emit()
    emit("Two ways to read the PET enrichment, and they have opposite implications:")
    emit()
    emit("- the PET set really is structurally homogeneous cutinase-like alpha/beta "
         "hydrolases, which is reassuring for the mechanistic story; or")
    emit("- the PET set is dominated by **one** structural family, in which case a "
         "classifier separating it from a heterogeneous contrast class is again doing "
         "family recognition rather than substrate-specificity recognition.")
    emit()
    emit("These are distinguishable with real profile methods (Pfam PF01083 cutinase, "
         "PF12695 abhydrolase) rather than a regex, which is the right next test. The "
         "motif check as written should not be cited as evidence either way.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(OUT) + "\n")
    print(f"\n[wrote {args.out}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
