# BioPET-SAE — Progress Log

Running log of the BioPET-SAE project: detecting PET-degrading enzymes (plastizymes)
from protein language model embeddings, and explaining the decision with sparse
autoencoder features. Sibling project: `colorfulcereal/PLMCircuitInterp` (SAE circuits
→ EC 2.7 kinase function), whose ESM-2 / InterPLM plumbing and methodology carry over.

Updated as work progresses. This is the source of truth for project state — read it
before re-deriving anything.

---

## Background

Source documents (read into the project 2026-09-27):

- `~/Downloads/biopet_sae_research_plan.md` — the original research specification.
- `~/Downloads/kinase_circuit_and_biopet_summary.md` — summary of the completed kinase
  circuit project plus a principal-scientist review of the BioPET-SAE plan.

**Original plan, in brief:** detect remote (<30% identity) plastic-degrading hydrolases
via a two-stage chained sparse probe (univariate differential selection → L1 Lasso) on
InterPLM SAE features over ESM-2 layer-4 activations, benchmarked against BLASTp and
HMMER. Targets: ROC-AUC ≥ 0.92, recall ≥ 85% at FPR ≤ 1%, ≤ 20 active features.

**Issues found in that plan before any code was run** (from the review, all verified
live rather than assumed):

1. Positive-class N assumed ≈ 1,200; the plan's own UniProt query returns **65**.
2. `keyword:"Plastic degradation"` is not a real UniProt term — returns zero.
3. The provided `SimpleSAE` is a randomly-initialised `nn.Linear`, never loads InterPLM
   weights, and does not match the real `ReLUSAE` architecture.
4. Hard negatives and naive controls are collapsed into one label at training time,
   undermining the plan's own stated rationale.
5. `ec:3.1.1.*` wildcard is a loose prefix match (6,174 hits vs 4,294 for `ec:3.1.1.-`).
6. Mean-pooling dilutes the localised catalytic signal the plan targets.

---

## Work completed

### 1. Data sourcing decided and datasets acquired (2026-09-27)

Rejected the plan's UniProt-EC-query approach for positives (65 entries) in favour of
two curated databases:

- **PlasticDB** (`plasticdb.org`) — `static/PlasticDB.fasta` and
  `static/degraders_list.tsv`, direct scriptable static-file downloads.
  2,535 rows, but **2,008 have no protein sequence** (whole-organism degradation
  reports) → 313 unique sequences.
- **PAZy** (`pazy.eu`) — exported manually through the web UI as
  `pazy_proteins_metadata.csv`, `pazy_proteins_sequences.csv`, `pazy_proteins.fasta`.
  462 metadata rows, 442 with sequences. **All 462 are `verified_activity = 1`.**

The PAZy *sequences* export carries no substrate column (`protein_id, protein_name,
amino_acid_sequence` only) and ~19% of its `protein_name` values are bare numeric ids,
so it is unusable for labelling on its own. The separate **metadata** export is the one
that matters — it has `substrates`, `organism_name`, `phylum`, `UniProtKB`,
`UniRef50/90/100`, `PDB`, `literature_dois`, `sequence_length`.

All five raw files are now committed under `data/raw/` with a `DOWNLOADED_AT.txt`
timestamp. Previously they existed only in `/tmp` and would have been lost on reboot.

### 2. Project skeleton (2026-09-27)

`uv` project, Python 3.13, `src/biopet_sae/` layout with `uv_build`. Dependencies so far
are data-only (pandas, numpy, scikit-learn, biopython, requests, tabulate). **torch,
transformers and interplm are deliberately not installed yet** — they arrive with the
embedding step.

### 3. Merge pipeline (2026-09-27) — `src/biopet_sae/build_dataset.py`

Produces `data/processed/plastizymes_merged.tsv` (33 columns, one row per protein),
`plastizymes_merged.fasta`, and `merge_report.json`.

**Two-pass dedup, and the second pass is not optional.** Exact sequence match removes
106 duplicates; a substring-containment pass with a 60-residue length tolerance removes
a further **107**. Those are the same protein stored as precursor-with-signal-peptide in
one database and mature form in the other. Exact-match dedup alone would have carried
107 near-identical pairs into training, where they leak across any homology split.

755 source records → **542 non-redundant proteins**. 210 are corroborated by both
databases independently.

**Substrates are grouped by the bond being hydrolysed**, not by polymer name, because
that determines whether two enzymes are mechanistically comparable at all:

| group | n | mechanism |
|---|---|---|
| PET_AROMATIC_POLYESTER (PET, PEF) | 320 | carboxylic ester |
| POLYAMIDE (PA, nylon) | 65 | **amide** |
| ALIPHATIC_POLYESTER (PCL, PLA, PBS, PBSA, PES) | 57 | carboxylic ester |
| PHA_FAMILY (PHA, PHB, PHBV, PHO) | 36 | carboxylic ester, distinct lineage |
| AROMATIC_COPOLYESTER (PBAT, PBSeT) | 20 | carboxylic ester, contains terephthalate |
| POLYURETHANE | 15 | **urethane** |
| POLYOLEFIN_VINYL (PE, PS, PP, PVC) | 13 | **C–C oxidative** |
| RUBBER, POLYCARBONATE, POLYETHER, UNKNOWN | 16 | various |

Pooling polyamidases or polyurethanases under "plastic degrader" would have been a
mechanism error — they do not hydrolyse esters.

### 4. EDA (2026-09-27) — `src/biopet_sae/eda.py` → `results/eda_report.md`

12 sections, every number recomputed from the merged table. Findings that change the
design, in order of severity:

**(a) The phylum confound is the biggest threat to the project.**

```
PET class      : 72.6% Actinomycetota, 16.5% Pseudomonadota   (n=285 annotated)
contrast class :  2.5% Actinomycetota, 42.5% Pseudomonadota   (n=40 annotated)
```

Near-perfectly anti-correlated. A "predict PET if Actinomycetota" rule alone gets
**75.7% accuracy** (TP=207, FP=1, TN=39, FN=78). This is large enough to be the primary
explanation for any strong result, so it is not a limitations-section caveat. Planned
mitigations, increasing in strength: report per-phylum performance; stratify homology
splits by phylum; build a **phylum-matched contrast subset** (Actinomycetota non-PET
esterases vs Actinomycetota PETases) from UniProt.

**(b) Confound baselines — the real floor is not 0.5.**

| signal | metric | score |
|---|---|---|
| phylum only | accuracy | 0.757 |
| amino-acid composition, 20 features | ROC-AUC | 0.809 |
| sequence length only | ROC-AUC | 0.595 |
| thermophilic genus only | ROC-AUC | 0.548 |

The composition baseline uses a random CV split, so it leaks homology and is optimistic
— but any SAE result has to be read against these, not against chance.

**(c) Effective sample size is far below 320.** PAZy's UniRef ids give a free first-pass
homology grouping: within the annotated PET subset, 117 entries collapse to **59
UniRef50 clusters** (2.0:1). Extrapolated, ~161 independent families at 50% identity.
Holding out 15% leaves ~20–25 test positives, where the 95% CI on AUC spans ~0.16 and a
model whose *true* AUC is 0.92 measures below 0.92 on **45%** of draws. Decision:
**k-fold CV over clusters, bootstrap over clusters not sequences, and state success
criteria as paired comparisons** (beats dense-ESM probe / BLASTp on the same folds)
rather than absolute thresholds.

**(d) Working classes: 320 PET-active vs 105 fold-matched non-PET esterases.** The
negative side is too small to carry an evaluation alone, so UniProt bulk negatives are
still needed — but these 105 are the only negatives backed by an actual polyester assay.
Enzyme-name vocabulary overlaps across the two classes (cutinase 20 vs 4, esterase 8 vs
10, lipase 4 vs 7), which is the point of the design.

**(e) Leak risk, not flagged in the original plan:** 124 of the 320 PET positives carry
a UniProtKB accession, so they will appear in a UniProt `ec:3.1.1.-` negative pull. They
must be excluded by accession *and* by sequence identity.

**(f) Structural validation set exists:** 32 PET entries have solved PDB structures —
IsPETase (55), FsC cutinase (46), CalB (27), LCC (20), Cut190 (18), TfH (6). Enough to
map discriminative SAE features onto real residue coordinates, the direct analogue of
the kinase project's HRD/P-loop mapping.

**(g) Evidence quality varies and should be carried as a weight.** PlasticDB assay
types: Spectrophotometry 137, HPLC 89, Clear zone 77 (qualitative plate assay — weakest),
Weight loss 34. 50 merged entries have degradation *extrapolated* from the enzyme rather
than measured.

**(h) A methodology correction made mid-EDA, worth keeping.** The first version of the
fold-match check reported raw `G-x-S-x-G` hit rates (PET 98.8%, contrast 75.2%, polyamide
50.8%) and concluded the contrast class was poorly fold-matched. That reading was wrong:
against a composition-preserving shuffled null (~24%), *every* ester group is enriched
2.7–3.6x, so the fold match is real; PET is simply the most homogeneous (4.1x). Polyamide
is also enriched 2.4x, so it is not the clean negative control the first version assumed.
A permissive regex needs its null computed before any hit rate means anything.

---

### 5. UniProt negatives pulled (2026-09-27) — `src/biopet_sae/fetch_negatives.py`

Output: `data/processed/uniprot_negatives.tsv` / `.fasta`, `negatives_report.json`.

**Binding constraint discovered:** only **174** reviewed `ec:3.1.1.-` entries exist in all
of UniProt for Actinomycetota — fewer than the ~254 a 72.6% phylum quota on 350 would
need. Rather than fall back to TrEMBL (36,478 available, but unreviewed annotation cannot
support "*not* annotated as plastic-degrading"), class 3 was capped. 240 vs 320 is an
acceptable ratio anyway, since the positive count is what binds the error bar.

| class | target | obtained | Actinomycetota share |
|---|---|---|---|
| 3 fold-matched esterase | 240 | **185** | 66% (150 of 174 available) |
| 4 naive control | 240 | **238** | 74% |
| PET class, for comparison | — | 320 | 73% |

Class 4's phylum mix matches the PET class almost exactly. Class 3 falls short at 66%
purely because UniProt does not contain more reviewed Actinomycetota esterases.
24 Actinomycetota candidates were lost to the plastic-name filter and QC.

Validation: zero class-4 rows carry an EC 3.1.x annotation (no label contradiction), zero
duplicate accessions, zero plastic-name leaks. 30 exact-duplicate sequences dropped.

Re-confirmed live: `ec:3.1.1.-` = 4,294 reviewed vs `ec:3.1.1.*` = 6,174 — the wildcard
bug is real.

### 6. mmseqs2 installed (2026-09-27)

`brew install mmseqs2` → version 18-8cc5c, native arm64. This avoids the
SpanSeq/CCPhylo route that forced the kinase project onto Lightning AI.

### 7. Splits built and the 30% bound verified (2026-09-27) — `src/biopet_sae/make_splits.py`

Output: `data/processed/dataset_splits.tsv`, `splits_report.json`.

Pipeline per D3/D4: cluster at 90% to drop mislabelled negatives → cluster at 30% →
mmseqs all-vs-all (sensitivity 7.5) for candidate pairs → **global Needleman-Wunsch**
realignment of all 53,702 cross-cluster candidates → union-find merge of any clusters
joined by a pair above threshold → greedy stratified assignment → independent
verification.

**The bound holds, and is measured rather than assumed:**

| split pair | max global identity |
|---|---|
| test / train | 0.2917 |
| train / val | 0.2996 |
| test / val | 0.2829 |

20,085 of 53,702 cross-cluster candidate pairs exceeded 30% global identity and were
merged. 358 clusters → 278 components; 20 components contain more than one class.

---

## FINDING: the "<30% remote homolog" premise is not testable with this data

This is the most consequential result so far and it was found before any compute was
spent on embeddings.

Enforcing the 30% bound collapses the PET class into **one component of 283 of 320
sequences**, leaving 15 test positives (split 290/15/15 instead of ~224/48/48). A
threshold sweep (`src/biopet_sae/analyze_split_frontier.py` →
`results/split_frontier.json`) shows this is a property of the data, not of the method:

| threshold | components | largest PET component | PET test |
|---|---|---|---|
| 0.25 | 127 | 320 (all of it) | 0 |
| **0.30** | 278 | **283** | **15** |
| 0.40 | 370 | 256 | 31 |
| 0.50 | 411 | 252 | 34 |
| 0.60 | 463 | 122 | 39 |
| 0.70 | 554 | 75 | 47 |
| 0.90 | 734 | 24 | 48 |

Both standard identity definitions agree (`matches/min_len` and
`matches/alignment_len`), so it is not a definitional artifact. Verified it is also not a
single-linkage hub artifact:

- **80.6% of all PET–PET pairs are ≥30% identical**; median pairwise identity **0.447**
- median degree at ≥50% identity is **91.5** — a typical PET sequence has ~92 relatives
  at ≥50%
- removing the 20 highest-degree sequences moves the largest component only 252 → 232

**Known PET hydrolases are one densely connected cutinase-like family, not a set of
remote homologs.** No splitting strategy fixes this, including k-fold CV: a 283-member
component is indivisible, so whichever fold holds it has 283 PET in test and 37 in train.

### Consequences

1. The original plan's headline claim — "detect remote (<30% identity) plastic-degrading
   hydrolases where BLASTp fails" — **cannot be evaluated** on the available data. Not
   because of a pipeline flaw, but because the independent PET families do not exist in
   the curated databases.
2. This resolves open decision #1 in favour of **mechanism-first framing**. Feature
   interpretation does not need a held-out split the way a performance claim does: all
   320 PET sequences can be used for the mechanistic analysis, with the split reserved
   for a supporting detection number at an explicitly stated bound.
3. The supporting detection benchmark should be reported at a **relaxed, stated bound**
   (0.50 gives 34 test positives; 0.60 gives 39) and described accurately as
   *within-family discrimination*, not remote-homolog detection.

### Secondary problem: the PHA probe is contaminated

The held-out PHA set reaches **0.7168** identity to the val split, so as constituted it
cannot support the cross-activation test — "the PET detector fires on PHA" would be
uninterpretable if the PHA sequence is a near-relative of a training sequence.

| bound | PHA sequences below it |
|---|---|
| <30% to any trained sequence | **12 / 36** |
| <40% | 22 / 36 |
| <50% | 24 / 36 |
| <70% | 31 / 36 |

Worst offenders: BPS0405 (0.988), BPS0428 (0.908), BPS0411 (0.836), BPS0429 (0.834).
The probe set must be filtered to the clean subset, and 12 sequences is small enough that
the test becomes suggestive rather than conclusive.

### 8. External dataset evaluated (2026-09-27) — `src/biopet_sae/experiment_external_split.py`

**Balci, Akgüller & Erdoğan 2026** (submitted JCIM), *"Cluster-Aware Benchmarking of
Protein Language Models for Plastic-Degrading Enzyme Prediction: Ensemble Embeddings
Improve Accuracy while Out-of-Family Generalization Remains Unsolved"*. Zenodo
`20573955`, MIT licensed. 1701 sequences, 642 positives (403 PET) / 1059 negatives
(559 hard UniProt hydrolases + 500 easy PlasticEnz), positives from PAZy + PlasticEnz.

Their title states the same conclusion this project reached independently today.

**It effectively contains our positive set:** 233 exact sequence matches with our 320
PET, and 80 of our 87 remainder are precursor/mature variants of theirs. Merging would
add ~7 genuinely new PET sequences.

**They have things we do not:** `confidence` tiers (verified 498 / high 17 / medium 84 /
low 43), a `temporal_flag` time-based split (only 13 test sequences), precomputed
cluster-aware folds, hard/easy negative typing, and notebooks running LOCO with
Bonferroni/BH-corrected paired bootstrap tests. They benchmark ESM-2 at 8M/35M/150M/650M/**3B**,
ProtBERT, SaProt, and LoRA across 14 configs.

**They do no SAE or mechanistic interpretability work at all.** Their result is negative:
out-of-family transfer fails and they do not explain why. That is the opening for this
project, and it independently confirms the mechanism-first framing.

#### Measured cross-split identity on their data

Cached: `data/external/balci_pair_identities.tsv` (221,777 global alignments).

| strategy | max cross-split identity | pairs ≥30% |
|---|---|---|
| their `cluster30` grouped 70/15/15 | **0.9674** | 23,801 |
| their own published 5-fold folds | **1.0000** | 27,808 |
| our verified union-find pipeline | **0.2996** | **0** |

**The leakage is inherent to their clusters, not to how the groups were assigned:** 28,823
*across-cluster* pairs are ≥30% identical, 70 are ≥90%, 7 are ≥95%. A violation between
two different clusters cannot be fixed by any assignment of clusters to splits — only by
merging them. Same D4 problem, confirmed on independent data.

In fairness: their headline protocol includes Leave-One-Cluster-Out, and their own
conclusion already concedes generalization is unsolved, so they do not claim a clean 30%
bound for the 5-fold. This refines rather than refutes.

#### Real data bug in their release

**7 positive/negative pairs are ≥90% identical, 2 at exactly 1.000** — the same protein
labelled both a `verified` PET degrader and a negative:

```
1.000  POS a219d0ace89c (PET, verified)  <->  NEG eba0373101b5 (easy_PlasticEnz)
1.000  POS aac25498e57d (PET, verified)  <->  NEG 7f597bfa435a (easy_PlasticEnz)
0.975  POS a156bb1e609a (PBS, verified)  <->  NEG 17ce7a364fdb (hard_natural_hydrolase)
```

11 sequences involved, 4 of them negatives. They deduped by MD5, so precursor/mature and
truncated variants survived as separate rows — the same trap our containment pass caught
107 times. Our D2 rule removes these automatically.

#### Switching datasets does not move the wall

| dataset | PET train / val / test at a verified 30% bound | largest component |
|---|---|---|
| ours (847 seq) | 290 / 15 / 15 | 283 |
| theirs (1701 seq) | 346 / 29 / 28 | **851** |

28 test positives instead of 15. Better, not transformative. The homology ceiling is
external to any dataset — it is a property of the ~400 known PET-active enzymes.

### 9. Literature check: how the published work handles this (2026-09-27)

| work | split method | identity actually bounded at | claims remote-homology? |
|---|---|---|---|
| PEZy-miner (2024) | random 7:3 | nothing | no |
| PlasticEnz (2026) | CD-HIT 95%, cluster-level | ~95% (dedup only) | no |
| Balci et al. (2026) | mmseqs `cluster30` + 5-fold + LOCO | measured 1.000 max | no ("unsolved") |
| this project's original plan | mmseqs 30%, cluster split | asserted <30% | **yes** |

PEZy-miner's "30% identity" is a BLAST *inclusion* filter for recruiting candidate
homologs to screen, not a train/test criterion. Its 236 units are enzyme/plastic *pairs*,
not proteins — which is also why it has 36 genuinely assayed non-degraders, the one thing
our dataset lacks (PAZy and PlasticDB record positives only). Worth extracting from
`~/Downloads/1-s2.0-S2214030124000178-mmc1.docx`.

Note: only Balci et al. was measured directly (we have their data). PlasticEnz and
PEZy-miner are characterised from their methods sections, not from measurement.

**Conclusion: no published work has an honest remote-homology benchmark for plastizymes**,
because the data cannot support one. Two openings follow: publish the verified bound with
its 28 test positives stated as underpowered, and explain *why* out-of-family transfer
fails using SAE features.

---

## Decisions locked (2026-09-27)

### D1. Class design: 4-way, with PHA held out

| # | Class | n | Source |
|---|---|---|---|
| 1 | PET-active | 320 | PAZy + PlasticDB, assay-confirmed |
| 2 | Other polyester, fold-matched (aliphatic 49 + PBAT 20) | 69 | PAZy + PlasticDB, assay-confirmed |
| 3 | Fold-matched esterase, no plastic annotation | cap ~350 | UniProt `ec:3.1.1.-` |
| 4 | Non-esterase control | cap ~350 | UniProt, outside EC 3.1 |
| — | PHA family — **held out, never trained** | 36 | cross-activation probe |

**Why classes 3 and 4 are separate rather than one "negative" class.** If merged, a high
score cannot distinguish "the model learned polyester-activity" from "the model learned
what an esterase looks like." Separating them costs nothing — it is a label, not extra
data — and it is the direct fix for the label-collapsing bug in the original plan.

**Why PHA is held out instead of trained as a class.** n=36 is too small to train on, and
it is a distinct depolymerase lineage, so including it would make class 2 partly
family-defined. Held out, it answers a sharper question: *does the PET detector fire on a
different depolymerase lineage?* This is the direct analogue of the kinase project's
peptidase cross-check, which is what exposed its two layer-3 candidate latents as generic
classifier scaffolding rather than kinase circuitry.

**Why PBAT stays inside class 2 rather than being dropped or isolated.** PBAT contains
terephthalate, so PBAT-degraders are the genuine intermediate case between PET and purely
aliphatic polyesters. n=20 is too small for its own class, so it sits in class 2 and is
reported as a subgroup at test time: "PET separates even from terephthalate-containing
co-polyester degraders" is the strongest available version of the finding.

### D2. Negative sourcing: exclude by accession and ≥90% identity, NOT at 30%

Filtering out everything ≥30% identical to a positive would delete exactly the hard
negatives class 3 exists to provide. Keep the 30–90% identity band; cluster-based
splitting handles the leakage by keeping homologs in the same split. Exclusions are:
the 124 class-1 and 28 class-2 UniProt accessions, plus anything ≥90% identical to a
positive (a near-duplicate of a known positive is probably a mislabelled positive).

### D3. Splitting: pooled clustering, stratified assignment

Cluster the **pooled** set across all four classes, not per class. In a multiclass task a
class-1 protein in test that is homologous to a class-2 protein in train is still
leakage. Then assign **whole clusters** to train/val/test, stratified by class so
proportions hold. Mixed-class clusters (homologous enzymes with different substrates) go
entirely to one split and are the most informative test cases, so they get counted and
reported.

Clustering is on **sequence identity**, not on ESM-2 embeddings — splitting on the
representation under test would be circular.

### D4. The plan's "<30% identity between splits" does not follow from its method

`mmseqs easy-cluster --min-seq-id 0.3` bounds identity between a member and its cluster
*representative*. It places **no bound between two different clusters**, so the original
spec's guarantee is unsupported. Chosen fix: cluster with mmseqs2, split, then
**independently verify** with all-vs-all **global** (Needleman-Wunsch) alignment between
splits and move violators until the bound holds. Report the achieved maximum cross-split
identity as a measured number.

Global, not local, alignment specifically: the kinase project hit a real false-alarm bug
from using local alignment when re-deriving identity.

Rejected alternative: SpanSeq makespan partitioning, which is purpose-built for this
bound and is what the kinase project used, but depends on CCPhylo which has no arm64
build — it would force this step onto Lightning AI.

### D5. Evaluation: fixed split for the pipeline, 10-fold cluster CV for the headline

Both come from the same cluster assignment. The fixed train/val/test drives the pipeline
and the Lasso `C` sweep; **10-fold cluster CV** (`GroupKFold` with cluster id as the
group) produces the reported metric. Rationale: a single 70/15/15 split tests only ~24
of ~161 PET families and yields a 0.16-wide CI on AUC, where a model whose *true* AUC is
0.92 measures below 0.92 on 45% of draws. CV tests every family exactly once and gives a
measured spread instead of a formula-derived error bar. Hyperparameters are tuned on a
validation slice carved from each fold's own training portion (nested CV), never on the
test fold.

Success criteria are stated as **paired comparisons** on identical folds (beats the dense
ESM-2 probe; beats BLASTp) rather than absolute thresholds like "AUC ≥ 0.92", because
paired comparison cancels the fold-to-fold variance that dominates at this N.

### D6. Phylum confound: three controls, all mandatory

The confound was quantified, not assumed:

- PET class is 72.6% Actinomycetota; contrast class is 2.5%. A "predict PET if
  Actinomycetota" rule alone scores **75.7% accuracy** (TP=207, FP=1, TN=39, FN=78).
- Amino-acid composition predicts **phylum within the PET class** at **AUC 0.873**, and
  predicts **PET vs contrast** at **AUC 0.809** — so the second number is plausibly the
  first one in disguise.
- The direction matches genome GC bias exactly: Actinomycetota PETases run +2.97pp Thr,
  +2.35pp Ala, +1.71pp Pro, +0.97pp Arg and −1.94pp Lys, −1.86pp Glu, −1.54pp Ile.
  GC-codon-biased residues total 36.7% vs 31.1%.

The skew is **partly real biology**, which is why it cannot simply be filtered out: PET
is only attackable near its glass transition (~70 °C), so PET-active enzymes must be
thermostable, and thermophilic actinomycetes are where thermostable secreted cutinases
live. A sampling artifact sits on top (post-*Thermobifida* screening bias toward
relatives and compost metagenomes).

Controls:

1. **Phylum quota on class 3** — deliberately over-sample Actinomycetota esterases from
   UniProt so class 3's phylum mix resembles class 1's. Phylum then carries no
   discriminative information.
2. **Within-phylum evaluation** — repeat training and evaluation restricted to
   Actinomycetota. Performance holding up means the signal is functional; collapsing
   means it was taxonomic.
3. **Feature-level audit** — for every top discriminative SAE feature, test whether it
   separates Actinomycetota from Pseudomonadota *within* the PET class. A feature that
   does is a taxonomy detector, not a function detector.

---

## Current status (checkpoint, 2026-09-27)

Dataset is built, split, and verified. No embeddings, no SAE, no probe yet — deliberately,
because the split analysis changed what the project should claim.

**Assembled set: 847 sequences across 5 labels.**

| label | n | source |
|---|---|---|
| 1_pet | 320 | PAZy + PlasticDB, assay-confirmed |
| 2_other_polyester | 68 | PAZy + PlasticDB, assay-confirmed |
| 3_fold_matched_esterase | 185 | UniProt `ec:3.1.1.-`, phylum-quota matched |
| 4_naive_control | 238 | UniProt, outside EC 3.1, phylum-quota matched |
| heldout_pha | 36 | cross-activation probe (only 12 usable, see above) |
| excluded | 118 | non-ester mechanism + 1 curated exclusion |

**Split as currently built (30% bound, verified):** class 1 is 290/15/15, which is not
usable for a detection claim — see the finding above.

**Artefacts:**

| file | contents |
|---|---|
| `data/processed/plastizymes_merged.tsv` | 542 positives, 35 columns, class labels + reasons |
| `data/processed/uniprot_negatives.tsv` | 423 negatives with phylum |
| `data/processed/dataset_splits.tsv` | 847 rows: class, split, cluster, component |
| `data/processed/pair_identities.tsv` | 70,230 cached global alignments (both definitions) |
| `results/eda_report.md` | 12-section EDA |
| `results/split_frontier.json` | threshold sweep + PHA contamination |

## Still open

1. **Detection bound to report at** — 0.50 (34 test positives) or 0.60 (39), described
   accurately as within-family discrimination. The 0.30 bound is verified but yields only
   15 test positives.
2. **Backbone** — ESM-2 8M (reuse kinase infrastructure, layers 1–6 SAEs) vs 650M
   (verified available: SAEs for layers 1, 9, 18, 24, 30, 33, all 10,240 features over
   1,280 dims). With 847 sequences there is no compute argument for 8M.
3. **Whether to expand the positive set** to make remote-homology testable at all. The
   ceiling is external: PAZy and PlasticDB together are the curated universe. Metagenomic
   mining (the PlasticEnz route) would add candidates but not *verified* ones.

## Next steps

1. Rebuild splits at the chosen relaxed bound, keeping the verified 0.30 split alongside
   as the strict-but-underpowered variant.
2. Filter the PHA probe set to the 12 sequences below 30% identity to trained data.
3. Replace the regex fold check with real profile methods (Pfam PF01083 cutinase,
   PF12695 abhydrolase) to settle whether the PET class is one structural family or
   several — the 80.6%/≥30% density finding suggests one, which would mean any classifier
   result is family recognition unless the contrast class is inside the same family.
4. Then embeddings: ESM-2 via HuggingFace `transformers` (not `fair-esm`), max-pooled,
   retaining per-residue activations so residue mapping stays possible.
5. Run the three phylum controls (D6) as part of the first probe evaluation, not after.

## Environment notes

- `mmseqs`, `blastp`, `makeblastdb`, `cd-hit`, `hmmscan`, `diamond` are all **not
  installed** on this machine as of 2026-09-27.
- InterPLM SAE availability confirmed live on HuggingFace:
  `Elana/InterPLM-esm2-8m` layers 1–6 (320 dim → 10,240 features);
  `Elana/InterPLM-esm2-650m` layers 1, 9, 18, 24, 30, 33 (1,280 dim → 10,240 features).
  Each layer ships `ae_normalized.pt` and `ae_unnormalized.pt`.
