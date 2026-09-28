# Polyester-Degrader Dataset

Generated 2026-09-28. The dataset under its revised labelling: **does this enzyme degrade
a polyester, or not?** Supersedes the PET-only framing in `dataset_splits_0.4.md`, which
remains the reference for provenance, deduplication and split construction.

## Why the task was redefined

The PET-only classifier plateaued at AU-ROC ~0.68 against other polyester degraders at
both model scales and for BLAST alike. The SAE interpretation explains why
(`SAE_Feature_Interpretation.md`): the model's strongest features locate the
**Ser-Asp-His catalytic triad**, which PETases and cutinases share almost exactly. A
PET-vs-cutinase boundary is not available in this representation.

Most "PETases" are in fact cutinases assayed on PET — LCC is *Leaf-branch Compost
**Cutinase***, Cut190 and TfCut are cutinases, and 20 of the 320 PET-class enzymes are
named "cutinase" outright. The PET/non-PET line runs through one protein family, and it
is drawn by which substrate somebody happened to test.

So the question was changed to one the biology supports and the data can answer.

---

## Labels

| tier | train | val | test | total | role |
|---|---|---|---|---|---|
| `1_pet` — PET degraders | 257 | 32 | 31 | 320 | **positive** |
| `2a_PBAT` — PBAT-type co-polyester | 13 | 2 | 5 | 20 | **positive** |
| `2b_aliphatic` — aliphatic polyester (PCL, PLA, PBS) | 33 | 9 | 6 | 48 | **positive** |
| `3_fold_matched_esterase` — UniProt esterases, no plastic annotation | 119 | 40 | 26 | 185 | negative |
| `4_naive_control` — non-esterase controls | 166 | 36 | 36 | 238 | negative |
| **POSITIVES** | **303** | **43** | **42** | **388** | |
| **NEGATIVES** | **285** | **76** | **62** | **423** | |
| `heldout_pha` — PHA depolymerases | — | — | — | 36 | **held out, never trained** |

**The split is unchanged.** Train/val/test were built from homology components under a
verified 40% identity bound; only the labels moved. Max cross-split identity remains
0.3967 (train/val) and 0.3800 (train/test) — see `dataset_splits_0.4.md` §6.

### Why PHA is held out

PHA depolymerases hydrolyse polyhydroxyalkanoates — a polyester, and a plastic — but they
are a **separate enzyme lineage** from the cutinase-like hydrolases in the positive set.
Excluding them from training turns them into a generalisation probe: a model that has
learned "polyester hydrolase" as a concept should recognise them; one that has memorised
the cutinase family should not. Because the negative set contains 238 non-esterase
controls, a model cannot pass this probe simply by flagging everything.

---

## Confound baselines

What the task yields to signals that have nothing to do with catalysis. Measured on the
**homology-bounded train → test split**, the same one the models are scored on.

| signal | test AU-ROC | test AU-PRC |
|---|---|---|
| amino-acid composition (20 features) | **0.7020** | 0.6737 |
| sequence length alone | 0.5823 | — |
| Actinomycetota indicator alone | 0.5842 | — |
| prevalence (AU-PRC floor) | — | 0.4784 |

**0.7020 is the floor any real result must clear**, not 0.5.

One number to treat with suspicion: the same composition baseline scores **0.8926** under
a random 5-fold CV, because homologous near-duplicates land on both sides of the split.
That figure is meaningless here and is reported only to show the size of the inflation —
0.19 AU-ROC — that homology leakage produces on this dataset.

## The phylum confound shrank, and reversed

| task | positives | negatives | gap |
|---|---|---|---|
| OLD — PET only | 72.6% Actinomycetota | 66.0% | **+6.6 pts** |
| NEW — any polyester | 66.0% | 70.4% | **−4.4 pts** |

Adding PBAT and aliphatic degraders — largely *Pseudomonadota* and *Bacillota* — dilutes
the actinomycete dominance of the PET class. The gap narrows and changes sign, so a model
can no longer gain by detecting "is this a high-GC actinomycete protein". An
Actinomycetota-only classifier now scores AU-ROC 0.5842.

This matters because the PET task's single largest error mode was taxonomic: the 8M probe
made 9 of its 11 Actinomycetota class-3 errors, including two *Thermobifida fusca*
carboxylesterases from the very organism whose PETases it trained on
(`BLAST_vs_ESM2_8M.md` Table 5).

## Sequence length

| | median | IQR |
|---|---|---|
| positives | 302 aa | 275–411 |
| negatives | 281 aa | 190–428 |

Heavily overlapping, which is why length alone reaches only 0.5823.

---

## Validation and test are not equally hard

A property of this split worth stating, because it affects every comparison drawn from it.

| split | negatives | median BLASTp e-value to the training positives | clearing e ≤ 3.16e-2 |
|---|---|---|---|
| val | 76 | 1.9 | **2/76** |
| test | 62 | 1.0 | **8/62** |

**Test drew esterases markedly more homologous to the training polyester degraders.** Four
times the rate of val, on a smaller negative set.

The positives run the other way — test positives are *closer* to the training set (median
best e-value 3.7e-15 against val's 1e-09), so this is not a remoteness effect. It is which
negatives happened to land where.

Consequence: any method comparison on this split is partly a measurement of how each method
handles hard negatives, and validation understates the difficulty. Both are reported in
`BLAST_vs_ESM2_650M_Polyester.md` Table 1b rather than test alone.

---

## What this dataset can and cannot support

**Can:** *does this enzyme degrade a polyester?* 388 assay-confirmed positives spanning
three substrate classes, against 423 negatives of which 185 are fold-matched esterases.

**Cannot:** *does this enzyme degrade PET specifically?* That contrast has 48 aliphatic
and 20 PBAT degraders as its negative set — 6 and 5 in test respectively — and every
method tried lands near 0.68 on it. See `SAE_Feature_Interpretation.md` §5.

**One caveat on the negatives.** PAZy and PlasticDB record only confirmed *positives*, so
`3_fold_matched_esterase` means "not annotated as plastic-degrading", never "assayed and
shown inactive". A false positive in that tier may be a correct prediction about an
untested enzyme. Metrics are therefore reported per tier rather than pooled.

## Provenance

| file | contents |
|---|---|
| `artifacts/dataset_splits_0.4.md` | sources, dedup, class rationale, split verification |
| `data/processed/dataset_splits_id40.tsv` | 847 rows: class, split, component |
| `data/sae_650M_L33/polyester_eda.json` | every number in this document |
