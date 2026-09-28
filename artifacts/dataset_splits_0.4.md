# BioPET-SAE — Dataset & Splits at a 40% Identity Bound

Generated 2026-09-27. Every number recomputed from
`data/processed/dataset_splits_id40.tsv`, `plastizymes_merged.tsv`,
`uniprot_negatives.tsv` and the cached alignments in `pair_identities.tsv`.

Reproduce with:

```
uv run python -m biopet_sae.build_dataset
uv run python -m biopet_sae.fetch_negatives
uv run python -m biopet_sae.rebuild_splits --threshold 0.40
```

---

## 1. Why PET is the hard case (the rationale behind every class decision)

Everything in the class design follows from one physical fact: **PET is far harder to
degrade enzymatically than the other polyesters in the dataset, and the reason is
physical accessibility rather than chemistry of the bond.**

Ester hydrolysis requires a chain segment to locally unwind and thread into the active
site so the carbonyl can reach the catalytic serine. Whether that is possible depends on
chain mobility, which is governed by glass transition temperature:

| polymer | backbone | Tg (approx.) | state at 30 °C |
|---|---|---|---|
| PCL | aliphatic | −60 °C | rubbery, chains mobile |
| PBAT | aliphatic-**aromatic** co-polyester | −30 °C | rubbery, chains mobile |
| PLA | aliphatic | 55–60 °C | glassy, borderline |
| **PET** | **aromatic** | **67–81 °C** | **glassy, chains frozen** |

Crystallinity compounds it: bottle-grade PET is ~30–40% crystalline and crystalline
domains are effectively impenetrable. So PET resists not because its ester bond is
chemically unusual but because it is inaccessible below Tg. This is why the enzymes that
actually work on it — LCC, LCC-ICCG, FAST-PETase — are thermostable and are run at
65–72 °C on amorphised, micronised PET, and why the industrial process operates hot.

**Difficulty ordering: aliphatic polyester < PBAT < PET.**

### 1.1 Consequence: PBAT-degraders are the sharpest negatives

PBAT is a co-polyester of butylene *adipate* (aliphatic, flexible) and butylene
*terephthalate* — the same aromatic ester unit found in PET. Its aliphatic segments
supply enough chain mobility for enzymes to work at ambient temperature despite the
terephthalate being present. That is why PBAT is sold as compostable and PET is not.

This makes PBAT-degraders uniquely diagnostic: **they can already cleave a terephthalate
ester.** So if a probe separates PET-degraders from PBAT-degraders, the discriminating
feature cannot be "recognises terephthalate" — it must concern handling a rigid,
crystalline, high-Tg substrate: the wide shallow substrate cleft, the aromatic clamp
(Y87/W185 in IsPETase numbering), the DS1 disulfide, thermostability. That is precisely
the mechanistic claim this project wants to make, and PBAT is what isolates it.

### 1.2 Consequence: aliphatic-polyester activity is a weak label

Because aliphatic esters are easy to hydrolyse, almost any α/β-hydrolase esterase, lipase
or cutinase will show some activity on PCL. So "aliphatic polyester degrader" is a
*non-distinctive* label — those entries are largely enzymes somebody happened to assay,
not a coherent functional group. PET activity, by contrast, is genuinely restrictive.

### 1.3 Consequence: the contrast that carries scientific content

> **can attack a stiff aromatic crystalline polyester** vs. **can only hydrolyse flexible
> aliphatic esters** — within one fold family, sharing the same Ser-His-Asp catalytic
> triad.

One asymmetry caveat: 36 of the 320 PET enzymes are also reported active on an aliphatic
polyester, consistent with PET activity being a superset capability. This asymmetry
**cannot be demonstrated from these labels**, because the labelling rule makes PET win
whenever both are recorded, so every dual-active enzyme sits in class 1 by construction.
The claim rests on the chemistry, not on this dataset.

---

## 2. How the dataset was created

### 2.1 Positives: two curated databases

Both record only experimentally confirmed degraders. **Neither records confirmed
non-degraders** — this is the central data limitation of the project and the reason a
measured false-positive rate is not defensible.

- **PAZy** (`pazy.eu`) — 462 entries, 442 with sequences, all `verified_activity = 1`.
  Exported through the web UI. The *metadata* export is the essential one: it carries
  `substrates`, `organism_name`, `phylum`, `UniProtKB`, `UniRef50/90/100`, `PDB` and
  `literature_dois`. The *sequences* export alone is unusable for labelling — it has no
  substrate column and ~19% of its `protein_name` values are bare numeric ids.
- **PlasticDB** (`plasticdb.org`) — 2,535 rows, but 2,008 are organism-level degradation
  reports with no protein sequence → 313 unique sequences.

### 2.2 Two-pass deduplication: 755 source records → 542 proteins

| pass | removed |
|---|---|
| exact sequence match | 106 |
| **substring containment** (≤60 residue length difference) | **107** |

The second pass is not optional. Those 107 are the *same protein* stored as
precursor-with-signal-peptide in one database and mature form in the other. Exact-match
dedup alone leaves all 107 near-identical pairs in the data, where they then leak across
any homology split. Balci et al. (2026) deduped by MD5 only and this is exactly the trap
that produced the label contradictions in their release (see §7).

210 of the 542 proteins are corroborated by both databases independently.

### 2.3 Substrates grouped by the bond hydrolysed, not by polymer name

Grouping by chemistry rather than by product name is what keeps the comparison
mechanistically valid.

| group | n | bond attacked | role |
|---|---|---|---|
| PET_AROMATIC_POLYESTER (PET, PEF) | 320 | carboxylic ester | **class 1** |
| AROMATIC_COPOLYESTER (PBAT, PBSeT) | 20 | carboxylic ester | **class 2a** |
| ALIPHATIC_POLYESTER (PCL, PLA, PBS, PBSA, PES) | 57 | carboxylic ester | **class 2b** (48 kept) |
| PHA_FAMILY (PHA, PHB, PHBV, PHO) | 36 | carboxylic ester, distinct lineage | **held out** |
| POLYAMIDE (PA, nylon) | 65 | **amide** | excluded |
| POLYURETHANE | 15 | **urethane** | excluded |
| POLYOLEFIN_VINYL (PE, PS, PP, PVC) | 13 | **C–C oxidative** | excluded |
| RUBBER / POLYCARBONATE / POLYETHER / UNKNOWN | 16 | various | excluded |

Polyamidases hydrolyse an **amide** bond, polyurethanases a **urethane** bond, and
polyolefin degraders work oxidatively (laccases, oxidases) rather than hydrolytically.
Pooling any of them under "plastic degrader" would be a mechanism error, so 109 sequences
are excluded on mechanism grounds.

### 2.4 Final classes

| class | n | definition |
|---|---|---|
| `1_pet` | 320 | PET or PEF substrate recorded, assay-confirmed |
| `2_other_polyester` | 68 | fold-matched non-PET polyester degraders (see split below) |
| ├ `2a_aromatic_copolyester` | 20 | PBAT-type — **contains terephthalate**, hardest negative |
| └ `2b_aliphatic_polyester` | 48 | PCL/PLA/PBS — flexible backbone, easier contrast |
| `3_fold_matched_esterase` | 185 | UniProt `ec:3.1.1.-` reviewed, no plastic annotation |
| `4_naive_control` | 238 | UniProt reviewed, outside EC 3.1 entirely |
| `heldout_pha` | 36 | never trained — cross-activation probe |

**Classes 3 and 4 are kept separate rather than merged into one "negative" class.** If
merged, a strong score cannot distinguish "the model learned polyester-activity" from
"the model learned what an esterase looks like." Separating them costs nothing — it is a
label, not extra data — and it is the direct fix for the label-collapsing flaw in the
original research plan.

**PHA is held out rather than trained as a class.** n=36 is too small to train on and it
is a distinct depolymerase lineage, so including it would make class 2 partly
family-defined. Held out, it answers a sharper question: *does the PET detector fire on a
different depolymerase lineage?* This mirrors the peptidase cross-check in the sibling
kinase project, which is what exposed two candidate latents there as generic scaffolding.

### 2.5 What left the aliphatic/PBAT pool, and why

The aliphatic group starts at 57 and contributes 48. The 9 removed:

```
BPS0010  not an esterase fold: Chitinase
BPS0018  not an esterase fold: Protease
BPS0019  not an esterase fold: Protease
BPS0023  named PETase by homology, only ever assayed on PCL
BPS0042  not an esterase fold: Protease
BPS0043  not an esterase fold: Chitinase
BPS0044  not an esterase fold: Protease
BPS0045  not an esterase fold: Protease
BPS0057  not an esterase fold: Protease
```

Six proteases and two chitinases. PLA is frequently attacked by proteases, which are a
different fold entirely, so these are **not fold-matched** to the PETases and would make
the contrast artificially easy.

`BPS0023` is a curated judgement call worth recording: PlasticDB lists gene `g16887.t1`
from *Clonostachys rosea* with the enzyme name "PETase", but that is a homology-based gene
annotation — the source paper assayed **PCL** (clear zone + weight loss, Sigma PCL
pellets) and never assayed PET. Placing it in class 2 would assert PET-inactivity;
placing it in class 1 would assert PET activity. Neither is supported, so it is excluded
from both. The exclusion is encoded by source id in `build_dataset.py`, not hand-edited.

A systematic scan for name/substrate mismatches found only 3 such cases, and only this
one crosses the PET boundary.

### 2.6 Negatives: phylum-matched to the positives

The PET class is **72.6% Actinomycetota** while the assay-confirmed contrast class is
**2.5%** — near-perfectly anti-correlated. A rule as crude as "predict PET if
Actinomycetota" scores **75.7% accuracy** (TP=207, FP=1, TN=39, FN=78). Worse, amino-acid
composition alone predicts phylum *within* the PET class at **AUC 0.873**, and predicts
PET-vs-contrast at **AUC 0.809** — so the second number is plausibly the first in
disguise. The direction matches genome GC bias exactly (Actinomycetota PETases run
+2.97pp Thr, +2.35pp Ala, +1.71pp Pro, −1.94pp Lys, −1.54pp Ile; GC-codon-biased residues
total 36.7% vs 31.1%).

The skew is **partly real biology**, which is why it cannot simply be filtered: PET needs
thermostability, and thermophilic actinomycetes are where thermostable secreted cutinases
live. A sampling artifact sits on top (post-*Thermobifida* screening bias).

Negatives were therefore quota-sampled to match the PET phylum distribution:

| class | target | obtained | Actinomycetota share |
|---|---|---|---|
| 3 fold-matched esterase | 240 | **185** | 66% (150 of only 174 available) |
| 4 naive control | 240 | **238** | 74% |
| 1 PET, for reference | — | 320 | 73% |

Class 4 matches almost exactly. Class 3 falls short because **only 174 reviewed
`ec:3.1.1.-` Actinomycetota entries exist in all of UniProt.** TrEMBL was deliberately not
used as a fallback (36,478 available) because class 3's premise is "well-annotated, and
*not* annotated as plastic-degrading", which unreviewed entries cannot support.

Query notes: `ec:3.1.1.-` is an exact subclass match; `ec:3.1.1.*` is a loose string
prefix returning 6,174 reviewed entries against 4,294 — 1,880 spurious.
`keyword:"Plastic degradation"` is **not a real UniProt term** and returns zero.

Exclusions applied: known positive accessions, and anything ≥90% identical to a positive
(a near-duplicate of a known positive is probably a mislabelled positive). Deliberately
**not** filtered at 30% — that would delete the very hard negatives class 3 exists to
provide.

Validation: zero class-4 rows carry an EC 3.1.x annotation, zero duplicate accessions,
zero plastic-name leaks.

### 2.7 Splitting with a bound that actually holds

`mmseqs easy-cluster --min-seq-id X` bounds identity between a member and its cluster
*representative*. It places **no bound between two different clusters**, so
cluster-then-split does not deliver the guarantee it appears to. Measured on the external
Balci dataset, their `cluster30` has 28,823 across-cluster pairs ≥30% identical, 70 ≥90%,
7 ≥95%.

The pipeline used here instead:

1. mmseqs cluster at 90% → drop negatives that are near-duplicates of a positive.
2. mmseqs all-vs-all search (sensitivity 7.5) → candidate related pairs. mmseqs reports
   *local* identity, so it is used only as a recall-oriented filter.
3. **Global Needleman-Wunsch realignment** of every candidate pair (70,230 alignments,
   cached to `pair_identities.tsv`).
4. Union-find over sequences joined at ≥ threshold → connected components.
5. Whole components assigned to splits, largest-first, stratified across all classes.
6. Independent verification pass measuring the achieved maximum.

The bound then holds **by construction**, and is still measured afterwards rather than
assumed.

Identity is defined as `matches / length of the shorter sequence`, with terminal gaps
unpenalised (one source stores precursors, the other mature forms, and a signal peptide
should not count against identity). **Global, not local** — using local alignment when
re-deriving identity produced a real false alarm in the sibling kinase project.

Clustering is on **sequence identity, not ESM-2 embeddings**: splitting on the very
representation under test would be circular.

---

## 3. Overall train/val/test distribution

```
split         n    share      (target)
train       588    72.5%         70%
val         119    14.7%         15%
test        104    12.8%         15%
TOTAL       811   100.0%
heldout_pha  36    (never trained, not part of any split)
```

## 4. Per-class split shares

```
class                       total   train    val   test    train%   val%  test%
1_pet                         320     257     32     31     80.3%  10.0%   9.7%
2_other_polyester              68      46     11     11     67.6%  16.2%  16.2%
3_fold_matched_esterase       185     119     40     26     64.3%  21.6%  14.1%
4_naive_control               238     166     36     36     69.7%  15.1%  15.1%
```

With class 2 split into its diagnostic sub-tiers:

```
sub-tier                    train    val   test   total
1_pet                         257     32     31     320
2a_aromatic_copolyester        13      2      5      20
2b_aliphatic_polyester         33      9      6      48
3_fold_matched_esterase       119     40     26     185
4_naive_control               166     36     36     238
```

The 2a tier has only 2 val and 5 test members. It is the most scientifically informative
contrast and the least statistically powered; it should be reported as an observation, not
a measurement.

## 5. PET vs non-PET per split

```
split        PET   nonPET   total   PET %
train        257      331     588    43.7%
val           32       87     119    26.9%
test          31       73     104    29.8%
TOTAL        320      491     811    39.5%
```

PET prevalence is **not constant across splits** — 43.7% in train against 26.9% in val,
because the indivisible 256-member PET component sits in train. Any threshold tuned on val
is therefore calibrated to a different base rate than training saw.

## 6. Max global identity across split boundaries (bound 0.40)

```
boundary      pair type                n      max   median     p95   >=0.40  >=0.35
train|val     overall               7355   0.3967   0.1838  0.3168        0      47
train|val     PET vs PET            3358   0.3916   0.2290  0.3176        0      21
train|val     PET vs nonPET         2274   0.3967   0.0713  0.3257        0      26
train|val     nonPET vs nonPET      1723   0.3455   0.0356  0.2461        0       0

train|test    overall               9941   0.3800   0.1699  0.3014        0      41
train|test    PET vs PET            4607   0.3716   0.2565  0.3105        0      28
train|test    PET vs nonPET         3262   0.3763   0.0264  0.2714        0       7
train|test    nonPET vs nonPET      2072   0.3800   0.0364  0.2697        0       6
```

PET against each class-2 sub-tier:

```
boundary      contrast                              n      max   median
train|val     PET vs 2a_aromatic_copolyester      167   0.3311   0.0367
train|val     PET vs 2b_aliphatic_polyester       728   0.3967   0.2986
train|test    PET vs 2a_aromatic_copolyester      427   0.3763   0.0105
train|test    PET vs 2b_aliphatic_polyester      1086   0.3722   0.0297
```

Worst individual pairs:

```
train|val:   0.3967   BPS0003 (2_other_polyester, train) <-> BPS0219 (1_pet, val)
train|test:  0.3800   A1SMR4 (3_ester, test) <-> Q6D6I7 (3_ester, train)
val|test:    0.3983   BPS0100 (1_pet, val) <-> Q47M62 (3_ester, test)
```

**The bound holds: zero pairs at or above 0.40 across any boundary.** Maxima press right
against the ceiling (0.3967, 0.3800), which is what single-linkage merging produces by
construction.

## 7. Integrity checks

```
components:                       357   (largest 256, entirely PET)
components spanning >1 split:       0   (must be 0)
components containing >1 class:    15   (the most informative test cases)
pairs >=0.40 across any boundary:   0   bound satisfied
negatives dropped as near-duplicates of a positive: 0
```

---

## 8. Known limitations

1. **PET is unevenly allocated** — 80.3% train vs ~10% each in val/test, because one
   indivisible 256-member component holds 80% of the class. Threshold choice moves this
   but does not remove it: at 0.30 the blob is 283 and val/test get 15 each; at 0.50 it is
   252 and they get 34 each; at 0.60 the blob is 122 but lands wholly in val, giving val
   101 PET against train's 180 — unusable.
2. **Held-out PETases are taxonomic outliers, at every threshold.** Train PET is 76.7%
   Actinomycetota; val PET is 12.5% (6 fungal Ascomycota, 4 Bacteroidota, 4
   Actinomycetota); test PET is 19.4% (10 Pseudomonadota, 10 Bacillota, 6 Actinomycetota).
   Sequences that escape the giant component are by definition the phylogenetically
   distant ones, so evaluation carries a kingdom-level domain shift. Relaxing the bound
   buys sample size but **not** composition.
3. **The Actinomycetota-only control (D6.2) cannot run** — val has 4 PET, test 6.
   The only remaining route is a phylum-matched contrast set, and UniProt cannot supply
   it (174 reviewed Actinomycetota esterases exist; 150 are already used).
4. **PET–PET pairs sit systematically closer than other pairs** (median 0.229 train/val,
   0.257 train/test, against 0.036 for nonPET–nonPET). The PET side of the task is easier
   than the bound alone implies.
5. **47 train/val pairs lie in the 0.35–0.40 band** (21 of them PET–PET). Legal at 0.40,
   violations at 0.30. That band is exactly what the looser bound buys.
6. **0.40 is looser than the original 30% plan.** It is still stricter than any published
   plastizyme work — PlasticEnz used CD-HIT 95% (dedup only), PEZy-miner used a random
   7:3 split with no homology control, and Balci et al. nominally used mmseqs at 30% but
   measured **1.000** maximum cross-fold identity. Unlike all three, this bound is
   measured rather than asserted.
7. **No verified negatives exist anywhere in this dataset.** PAZy and PlasticDB record
   only positives, so "PET-inactive" is an inference from absence of annotation. A
   measured false-positive rate is therefore not defensible, and metrics should be
   reported per negative tier rather than pooled. PEZy-miner's supplement contains 36
   assayed non-degraders — the one available source of real negatives, not yet extracted.

## 9. Provenance

| file | contents |
|---|---|
| `data/raw/pazy_proteins_metadata.csv` | PAZy metadata export (462 rows) |
| `data/raw/pazy_proteins_sequences.csv` | PAZy sequence export (442 rows) |
| `data/raw/degraders_list.tsv` | PlasticDB (2,535 rows) |
| `data/processed/plastizymes_merged.tsv` | 542 positives, 35 columns, class labels + reasons |
| `data/processed/uniprot_negatives.tsv` | 423 negatives with phylum |
| `data/processed/pair_identities.tsv` | 70,230 cached global alignments |
| `data/processed/dataset_splits_id40.tsv` | **this split** |
| `data/processed/dataset_splits_id40_report.json` | machine-readable summary |
