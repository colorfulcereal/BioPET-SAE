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

**Published basis for this rule** — [R1] Danso, Chow & Streit (2019) organise microbial
plastic degradation around exactly this split: polymers with a hydrolysable backbone (ester,
amide, urethane) versus a pure C–C backbone (PE, PP, PS, PVC), which admits no hydrolase and
requires oxidative attack. [R3] Wei et al. (2020) state the consequence directly —
biotechnological progress is still limited to polyesters — which is the one-line justification
for restricting the task to polyester degraders rather than "plastic degraders". Cite [R1] for
the exclusion of the 109 non-ester-mechanism candidates and [R3] for the task definition.

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

### 10. Modelling runs: ESM-2 8M, 650M, and the BLAST baseline (2026-09-27/28)

Split used throughout: `dataset_splits_id40.tsv` (40% identity bound, verified; the 30%
bound leaves only 15 val PET, see the FINDING section above). Train 588 (257 PET), val 119
(32 PET), test 104 (31 PET). **Test was untouched until hyperparameters were fixed on val.**

Protocol, applied identically to every method: tune on val, apply to test unchanged.
Artifacts: `artifacts/ESM2_8M_Dense_Run.md`, `ESM2_650M_Dense_Run.md`,
`BLAST_vs_ESM2_8M.md`, `BLAST_vs_ESM2_650M.md`.

#### Test-set results

| method | Precision | Recall | F1 | AU-ROC | AU-PRC | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| BLASTp @ e-value 3.16e-2 (val-tuned) | **0.828** | 0.774 | **0.800** | 0.8778 | **0.8500** | 24 | 5 | 7 | 68 |
| ESM-2 650M `max_L33` (C=0.1, eta=1e-3) | 0.742 | 0.742 | 0.742 | **0.8856** | 0.8150 | 23 | 8 | 8 | 65 |
| ESM-2 8M `max_L6` (C=1e-3, eta=1e-4) | 0.600 | **0.968** | 0.741 | 0.9015 | 0.7899 | 30 | 20 | 1 | 53 |
| BLASTp @ e-value 1e-5 (the plan's spec) | 0.947 | 0.581 | 0.720 | — | — | 18 | 1 | 13 | 72 |

**No pairwise difference among the three methods is statistically significant.** Paired
bootstrap over the 66 test homology components: 650M − BLAST gives dAU-ROC +0.0074
[−0.096, +0.123] and dAU-PRC −0.0412 [−0.190, +0.076]; 650M − 8M gives dAU-ROC −0.0152
[−0.097, +0.048] and dAU-PRC +0.0285 [−0.083, +0.131]. Every interval crosses zero at
n=104 with 31 positives.

The bootstrap resamples **whole homology components, not sequences** — members of a
component are ≥40% identical, i.e. one family observed several times, so resampling
sequences would report a dishonestly narrow interval. Both methods are scored on the same
resample and the difference recorded; their errors correlate at r = 0.61, which cuts
comparison noise ~36% versus comparing two separate intervals.

#### What survives as a finding: error profiles, not scores

| | 8M `max_L6` | 650M `max_L33` | BLASTp |
|---|---|---|---|
| total false positives | 20 | 8 | 5 |
| class-3 FP | 10/26 | 2/26 | 0/26 |
| class-3 FP, Actinomycetota | **9/11** | **1/11** | 0/11 |
| class-3 FP, non-Actinomycetota | 1/15 | 1/15 | 0/15 |
| failure mode | confidently wrong (scores 0.31–0.51 vs thr 0.22) | marginally wrong (7 of 8 score 0.004–0.008 vs thr 0.003) | — |

**The taxonomic shortcut predicted by D6 was confirmed at 8M and is resolved at 650M.** The
8M probe made 9 of its 11 Actinomycetota class-3 errors — an 82% false-positive rate on that
group against 7% elsewhere — and two of those errors were *Thermobifida fusca*
carboxylesterases, i.e. non-PET enzymes from the very organism whose PETases (TfCut, TfH)
are in the training set. That is organism recognition, not function recognition. At 650M the
rate falls to 1/11, equal to the non-Actinomycetota rate.

**Rebalancing does not fix it; scale does.** Three variants were tested at 8M (baseline;
phylum reweighting keeping all data; subsampling Actinomycetota positives to match the
negative rate, costing 111 of 257 positives). **All three produced exactly 10/26 class-3
errors with a 9/11 Actinomycetota skew.** Reweighting removed only 2 class-4 errors;
subsampling lowered recall to 0.903 and fixed nothing. The probe never sees a phylum label —
it sees embedding dimensions, and the GC-driven compositional signature is inside the
features. Reweighting the loss moves a boundary; it cannot delete a feature.

The competing explanation — that those esterases are genuine PETase relatives and the
negative label is merely unverified — was also rejected. BLASTed against the training PET
set, the 10 false positives have median best e-value 0.47 (min 0.067) against 1.7e-07 for
true positives, six orders of magnitude worse. One correctly-rejected sequence (`P37446`,
e-value 0.051) has a *better* alignment than 9 of the 10 errors.

**Conclusion: the feature was absent from the 8M representation, not mis-weighted in the
data.** Max-pooled ESM-2 8M layer 6 carries compositional and taxonomic information, which
is what a linear readout extracts. Layer 33 of 650M carries more of the functional signal.

#### The tier that does not move

| method | `2b_aliphatic` AU-ROC |
|---|---|
| ESM-2 8M `max_L6` | 0.6398 |
| ESM-2 650M `max_L33` | 0.6828 |
| BLASTp | 0.6989 |

Two model scales 80× apart and a sequence-alignment method all land near 0.68 on separating
PET degraders from other polyester degraders. Unchanged by scale, and the same tier where
BLAST also performs worst. This is the strongest evidence that **PET specificity, not fold
recognition, is the unsolved problem**, and it is the case for the SAE work: dense
dimensions are polysemantic, so "not linearly decodable at either scale" does not establish
that the feature is absent.

#### Other results worth carrying forward

- **Max-pooling beats mean-pooling at both scales** (8M: AU-PRC 0.848 vs 0.710 at L3; 650M:
  0.818 vs 0.734 at L33), consistent with the catalytic signal occupying a few residues.
- **Learning rate is not a meaningful hyperparameter for a linear probe.** Across four orders
  of magnitude at fixed C, AU-PRC moves <0.01 and SGD matches lbfgs to three decimals — the
  objective is convex. Only eta0=1e-1 destabilises, and it produced two spurious val optima
  (`mean_L3` at 8M, `mean_L33` at 650M) that did not transfer.
- **Both scales sit in the separable regime.** Best-F1 thresholds land at 0.222 (8M) and
  0.0030 (650M), far below 0.5, because the training data is linearly separable and
  probabilities saturate. Ranking metrics are unaffected; the probability scale is not
  calibrated and the default 0.5 threshold is unusable.
- **Seed standard errors are fourth-decimal** (5 seeds) and measure optimizer determinism,
  not performance uncertainty. The honest interval is the component bootstrap, ~0.25–0.31
  wide on AU-PRC — three orders of magnitude larger.
- **The plan's "≥40% higher recall than BLASTp" target** is met against its own specified
  baseline (0.968 vs 0.581 at e-value 1e-5, +67% relative) but 1e-5 is a weak operating
  point; against a val-tuned BLAST the comparison reverses on F1.
- **`mean_L4` scored highest of all 12 8M feature sets** (AU-PRC 0.8745) but was never in the
  comparison set, which was chosen as L3/L6 before the full sweep existed. Not pursued.

### 11. SAE feature extraction and differential scoring (2026-09-28)

**Status: complete. Result is negative — the plan's Stage-1 score found nothing on the
target contrast.** Details in section 12 below.

#### Loading (verified)

`interplm` installed from git. `ReLUSAE.from_pretrained(hf_hub_download(...))` works;
`load_sae_from_hf` cannot, because the upstream package omits its `train` subpackage (the
gotcha already recorded in CLAUDE.md). `einops` is an undeclared runtime dependency and had
to be added separately.

`Elana/InterPLM-esm2-650m`, `layer_33/ae_normalized.pt`:

```
encoder.weight  (10240, 1280)     decoder.weight  (1280, 10240)
bias            (1280,)           <- subtracted BEFORE encoding
encode: ReLU(encoder(x - bias))   decode: decoder(f) + bias
```

#### Pipeline

```
seq -> ESM-2 650M -> layer 33 per-residue (L x 1280) -> SAE.encode -> (L x 10240) -> pool
```

The SAE operates **per residue**, so the pooled tensors in `data/embeddings_650M/` cannot be
reused — this needs a fresh forward pass (~4 min on MPS). Max- and mean-pooled variants are
both saved; max-pooling won at both dense scales and is the natural choice for sparse
non-negative activations.

Layer 33 chosen because it is where the dense probe performed best (AU-PRC 0.818) and is one
of the six layers where InterPLM ships SAEs for this backbone.

#### Three method decisions, ahead of results

**D7. The plan's differential score needs a variance floor.** The spec is

```
S_j = (mu_pet - mu_control) / (sigma_control + 1e-8)          eps = 1e-8
```

which was written for dense activations. SAE latents are sparse, so many will be exactly 0
across all controls, giving sigma_control = 0 and S_j ~ mu_pet x 1e8. The top-50 list would
be dominated by ultra-rare latents firing in one or two sequences. Mitigation: report the
plan's raw statistic *and* a corrected version using a variance floor
(`sigma_control + median(sigma_all)`) plus a minimum-prevalence filter (latent active in
>=10% of PETases). Any latent appearing only in the raw ranking is an artifact of the
epsilon, not a finding.

**D8. "Control" must be computed per negative tier, not pooled.** The plan leaves
`mu_control` undefined. Given the measured tier structure, the choice determines the answer:

| control set | what the top latents encode |
|---|---|
| `4_naive_control` | "is this an esterase at all" |
| `3_fold_matched_esterase` | "is this a polyester hydrolase" |
| **`2b_aliphatic`** | **PET specificity — the unsolved contrast** |

All three are computed separately. The 2b contrast is the scientifically meaningful one and
also the weakest powered (48 training sequences).

**D9. Restricting to sequences the probe classifies correctly conditions on the model.**
Per the user's request, the top-50 latents are computed over sequences where the ESM-2 650M
probe predicts correctly. This follows the kinase project's precedent (it analysed the
27/200 held-out proteins the classifier confidently called kinase). It must be stated that
the resulting latents explain **the probe's decision**, not PET biology directly — which is
the correct target for an interpretability claim, but not the same claim.

#### Mandatory gate before any feature claim

The extraction also saves the SAE **reconstruction** of the dense activations, pooled
identically, plus per-residue reconstruction cosine. The probe must be re-run on
SAE-reconstructed activations and survive; if reconstruction destroys the signal, every
downstream feature result is meaningless. Kinase-project precedent, and it is also why
comparisons must use a **reconstruction baseline rather than a raw baseline** — otherwise
generic round-trip noise contaminates the causal signal.

### 12. SAE differential scoring — results (2026-09-28)

Artifact: `artifacts/SAE_650M_L33_differential.json`.
Features: `data/sae_650M_L33/sae_features.npz` (847 x 10240, max- and mean-pooled).

#### Extraction succeeded and the reconstruction gate passed

224s on MPS. Per-residue sparsity **1.42%** (145 of 10,240 latents active per residue),
mean reconstruction cosine **0.898**. Sanity check: the dense activations from this pass
match the cached `max_L33` embeddings exactly (max abs diff 0.00e+00).

**Gate result — the probe survives SAE round-tripping:**

| representation | dim | test AU-ROC | test AU-PRC | test F1 |
|---|---|---|---|---|
| raw dense layer 33 | 1280 | 0.8856 | 0.8150 | 0.7419 |
| SAE reconstruction | 1280 | 0.9068 | **0.8386** | 0.6829 |
| SAE features | 10240 | 0.8900 | 0.8115 | 0.7222 |

Reconstruction retains **102.9%** of raw test AU-PRC — slightly better, within noise. The
sparse features match raw dense. So the SAE does not destroy the signal, and any downstream
failure is attributable to the analysis rather than the representation.

#### D7 confirmed: the plan's epsilon makes the statistic unusable on sparse features

Overlap between the plan's raw `S_j` top-50 and the variance-floored version:

| control tier | zero-variance latents | raw-vs-corrected overlap |
|---|---|---|
| 2a_PBAT | 1302 (12.7%) | **0/50** |
| 2b_aliphatic | 1242 (12.1%) | **0/50** |
| 3_fold_matched_esterase | 1151 (11.2%) | 12/50 |
| 4_naive_control | 1109 (10.8%) | 30/50 |

**For both hard tiers the two rankings share no latents at all.** ~12% of latents have
exactly zero control variance, so `sigma + 1e-8` inflates them without bound and the raw
top-50 consists entirely of latents firing in one or two sequences. Running the plan as
written would have produced 50 artifacts and called them PET features.

#### Cross-tier structure (corrected statistic, top-50 each)

```
                          2a    2b     3     4
2a_PBAT                   50    27    27    14
2b_aliphatic              27    50    28    12
3_fold_matched_esterase   27    28    50    20
4_naive_control           14    12    20    50
```

The three fold-matched tiers share ~27-28 of 50 latents with each other but only 12-20 with
naive controls — a coherent "polyester hydrolase" feature set distinct from a "generic
protein" one, consistent with the dense-probe tier results. 13 latents were unique to the
2b top-50: `[813, 2133, 2359, 2846, 3026, 3350, 3361, 3390, 3628, 4336, 7150, 7370, 10081]`.

Note the top 2b latents are **magnitude differences, not on/off detectors**: prevalence in
PET 0.90-1.00 but prevalence in controls 0.23-1.00. Latent 2359 fires in 100% of both
classes and differs only in strength (mu 0.608 vs 0.182).

#### The 13 candidates are indistinguishable from random — NEGATIVE RESULT

PET vs `2b_aliphatic`, trained on train, evaluated on test (train 290, test 37, 31 PET):

| feature set | n_feat | test AU-ROC | pairs correct |
|---|---|---|---|
| 9 candidates, **selection on train only** | 9 | **0.5108** | 95/186 |
| 13 candidates, selection saw val+test (leaky) | 13 | 0.6129 | 114/186 |
| top-50 latents (2b tier, leaky) | 50 | 0.5699 | — |
| all SAE latents | 10240 | 0.5860 | — |
| **9 RANDOM latents** (200 draws) | 9 | **0.5770**, 90% range [0.333, 0.806] | — |
| dense `max_L33` (reference) | 1280 | **0.7312** | — |

**A leak was found and corrected.** The differential statistic was originally computed over
`keep = correct` across *all* splits, so selection saw 27 val and 23 test PET sequences plus
the test 2b controls; the chosen latents were then evaluated on test. That is circular. It
inflated the result from **0.5108 to 0.6129** and changed **10 of the 13** candidates —
leak-free selection yields `[3281, 3350, 3628, 3793, 4336, 6926, 8088, 10184, 10213]`, only
3 of which appear in the original list.

Leak-free, the selected latents score **0.5108 — a coin flip** — and sit at the **29th
percentile** of the random-latent distribution, i.e. worse than a median random draw. The
top-50 set also performs worse than random.

AU-ROC here is over 31 x 6 = 186 positive-negative pairs, so 0.5108 means 95 of 186 pairs
correctly ordered. `sae_differential.py` now defaults to `--select-on train`.
All 10,240 sparse features underperform the 1,280 dense dimensions on the target contrast.
(AU-PRC is uninformative here — the test 2b contrast is 31 PET vs 6 negatives, prevalence
0.84, so everything including random scores ~0.89.)

#### The ranking is also unstable to removing in-sample data

| tier | top-50 overlap, all-correct vs held-out-only |
|---|---|
| 2b_aliphatic | **not computable — only 4 held-out controls** |
| 3_fold_matched_esterase | 22/50 |
| 4_naive_control | 41/50 |

D9's concern was warranted: the "correct-only" set is 307 PET of which **257 are training
sequences the probe memorised** (in-sample accuracy 257/257 at the saturated threshold).
Dropping them changes more than half the tier-3 ranking.

#### What this establishes, and what it does not

**Establishes:** the research plan's Stage-1 univariate differential score, applied to these
SAE features, does not identify PET-specific latents. Three independent signs — no better
than random, ranking unstable to in-sample removal, sparse features underperforming dense
ones on the target contrast.

**Does not establish:** that the feature is absent from the SAE. The test 2b contrast has
**6 negatives**; nothing concluded from it is strong. The honest statement is "this
selection method at this sample size found nothing."

The reconstruction gate passing at 102.9% matters here: the failure is in the *selection
method and sample size*, not in the SAE destroying the signal.

#### Implication for next steps

Univariate selection was always the weak part of the plan, and the kinase project reached
the same conclusion by a different route — its single-latent tests looked null until
**greedy, conditioned** selection revealed a real distributed effect. Supervised selection
against the probe (L1 over SAE features, or gradient x activation attribution on the
PET logit) is the indicated replacement, with greedy sequential ablation for validation.

### 13. SAE feature interpretation — the main scientific result (2026-09-28)

Artifact: **`artifacts/SAE_Feature_Interpretation.md`**. Task changed to **PET (320) vs
everything else (491)** per user direction; tiers retained only for interpretation.

#### Supervised selection works where the differential score did not

Three selection methods on train only (L1 at C=0.1 giving exactly 50 non-zero; univariate
AU-ROC; variance-floored differential). They agree only partially — L1 ∩ univariate = 26,
all three = 14 — so validation decided. All three top-50 sets land at test AU-ROC ~0.85,
AU-PRC ~0.75, at the **93rd percentile** of 50 random latents: better than random but not
past the 95th.

#### The signal is concentrated, not distributed

| k (top by train univariate AU-ROC) | test AU-ROC | test AU-PRC | % of full |
|---|---|---|---|
| **1** | 0.8878 | 0.7655 | **94.3%** |
| 10 | 0.8515 | 0.7852 | 96.8% |
| 50 | 0.8529 | 0.7522 | 92.7% |
| 10240 | 0.8900 | 0.8115 | 100% |

**One latent recovers 94% of the full model.** Drop-one-out from the top-50 moves AU-PRC by
at most 0.022 — massive redundancy. This is the **opposite** of the sibling kinase project,
where recognition was distributed and no single latent mattered.

#### Six latents validate on held-out data (raw activation, no classifier)

| latent | train | val | test | held-out |
|---|---|---|---|---|
| **9529** | 0.9938 | 0.9418 | **0.9160** | **0.9282** |
| 411 | 0.9918 | 0.9397 | 0.9103 | 0.9199 |
| 2661 | 0.9874 | 0.9174 | 0.9129 | 0.9142 |
| 7734 | 0.9917 | 0.9165 | 0.8975 | 0.9053 |
| 5271 | 0.9938 | 0.9124 | 0.8904 | 0.8991 |
| 2473 | 0.9982 | 0.9095 | 0.8878 | 0.8991 |

**Latent 9529 alone reaches test AU-ROC 0.9160** — above dense `max_L33` (0.8856), the full
10,240-latent probe (0.8900) and BLASTp (0.8778). The best single number in the project.

Multiple-comparison check clears: best achievable by *any* of the 10,240 latents is 0.9222,
99.9th percentile 0.8958, so 9529 sits above the 99.9th percentile and 6 of 12
train-selected latents beat the 99th. Validation was necessary though — roughly half the
top-12 by train AU-ROC collapse on held-out data (3255: 0.991 → 0.576; 2359: 0.988 → 0.467).

#### Five of six latents map onto the Ser-Asp-His catalytic triad

Per-residue activations recomputed to locate each peak:

```
9529   NRLAVAGHSMGGGGA   next residue S 40/40   -> catalytic SERINE (nucleophile elbow GxSxG)
411    LANDRVPTMVISGQA ┐ peaks ~6 residues apart
2661   PTMVISGQADTVVTP ┘ -> catalytic ASPARTATE region
2473   ATTESVYLEVAGADH ┐ peaks ~7 residues apart
7734   YLEVAGADHGFMVGR ┘ -> catalytic HISTIDINE (A-G-A-D-H-G), next residue H 40/40
5271   PNIPNKIIGKYSVAW   C-terminal, no consistent motif — unassigned
```

The pairwise overlap explains the redundancy: these are **three views of one catalytic
apparatus**, not six independent detectors.

Latent 9529 is unambiguous — the residue after its peak is **S in 100% of PET**, 98% of 2b,
41% of class-3 esterases; the peak lands on a `GxSxG` in **316/320 (99%)** of PET versus
**127/527 (24%)** of non-PET.

#### Validated against UniProt curated active sites: 32/32

The triad positions derived from latent peaks were checked against UniProt `ACT_SITE`
annotations. 9 of 31 accessions carry them, covering 32 residues; **all 32 are recovered
within +/-1**. Exact matches include IsPETase (S160 D206 H237), LCC (S165 D210 H242) and
Est119 (S169 D215 H247).

PDB `SITE` records are **not** usable for this — they are software-generated
ligand-binding sites present in only 9 of 29 files. UniProt `ACT_SITE` covers only 9 of 31
entries, so the latent-derived assignment fills in the remaining 23 rather than duplicating
existing annotation.

Numbering must be reconciled by alignment: UniProt numbers IsPETase full-length (S160),
PDB 5XFY numbers from the mature protein (S131), and three Thermobifida entries are
truncated constructs offset 38-39 residues from their UniProt record. An initial
validation run scored 23/32 purely because substring matching failed on those three
variants; proper alignment gives 3/3 for each.

#### THE FINDING: it detects the fold, not the substrate

| latent | PET | 2b aliph. | 2a PBAT | 3 esterase | 4 naive | PET/2b |
|---|---|---|---|---|---|---|
| 9529 | 0.950 | 0.715 | 0.560 | 0.237 | 0.030 | 1.33 |
| 2473 | 0.953 | 0.407 | 0.217 | 0.156 | 0.031 | 2.34 |
| 7734 | 0.872 | 0.482 | 0.264 | 0.181 | 0.022 | 1.81 |

Every latent shows the same monotone gradient — **PET > 2b > 2a > esterase > naive** — which
is an ordering in *how canonically alpha/beta-hydrolase-like* a protein is, not a PET axis.

The strongest non-PET activators settle it:

```
BPS0038  2b_aliphatic  1.004   GRVGTSGHSQGGGGS
BPS0021  2b_aliphatic  0.952   DKFAVSGWSMGGGGA
PET consensus          ~1.02   NRLAVAGHSMGGGGA
```

`GWSMGGGGA` vs `GHSMGGGGA` — same motif, one position different, fires just as hard. It
**cannot** distinguish PETases from aliphatic-polyester cutinases, and never could: they
share the nucleophile elbow. It scores 0.92 only because 76% of the negative set lacks the
motif entirely.

#### This single feature explains every prior result

| observation | explanation |
|---|---|
| dense probe ~0.89 AU-ROC overall | the fold signal is strong and easy |
| near-perfect vs `4_naive_control` | non-esterases have no nucleophile elbow |
| ~0.64-0.70 vs `2b_aliphatic` at **every** scale, and for BLAST | those enzymes have the same motif |
| 8M → 650M fixed the phylum shortcut but not F1 | scale sharpened the fold detector, never the bottleneck |
| L1 sparsity on dense embeddings hurt | the fold signal is spread across correlated dimensions |
| the differential score found nothing on `2b` | no PET-specific latent exists at this layer to find |

**The model is an excellent serine-hydrolase detector and not a PET detector.** Same shape
of result as the kinase project, reached independently: its top latents also mapped onto
textbook catalytic motifs (P-loop, HRD) rather than anything substrate-specific.

#### Limits

One layer, one SAE (middle layers 9/18/24 untested and may carry structural rather than
sequence-level features); max-pooling discards position; **no causal test** — these are
correlational and the kinase precedent is greedy sequential ablation against a
reconstruction baseline; `5271` unassigned; and `3_fold_matched_esterase` is more
heterogeneous than intended (only 41% have a catalytic serine at the peak), which inflates
how well a fold detector separates the classes.

### 14. Task redefined: polyester-degrader classifier — the headline result (2026-09-28)

Artifacts: **`EDA_polyester_task.md`**, **`ESM2_650M_Polyester_Run.md`**,
**`BLAST_vs_ESM2_650M_Polyester.md`**.

#### Why

The PET-only classifier plateaued at ~0.68 AU-ROC against other polyester degraders at both
model scales and for BLAST. Section 13 explains why: the strongest features locate the
Ser-Asp-His catalytic triad, which PETases and cutinases share almost exactly. Most
"PETases" *are* cutinases assayed on PET — LCC is Leaf-branch Compost **Cutinase**, Cut190
and TfCut are cutinases, and 20 of the 320 PET-class enzymes are named cutinase outright.
The PET/non-PET line runs through one protein family and is drawn by which substrate
someone happened to test.

**New labels: polyester degrader (388 = PET 320 + PBAT 20 + aliphatic 48) vs non-degrader
(423 = esterase 185 + naive 238).** PHA (36) held out as a generalisation probe. The split
is unchanged — only labels moved, so the verified 40% identity bound still holds.

#### Clean run, not a relabelling of old results

Sweep, hyperparameter selection and threshold were all redone on this task. **Selection was
switched from val AU-PRC to val log-loss**, because AU-PRC saturates here: several cells tie
at 0.9995 and C=1/eta=1e-1 reaches a perfect 1.0000 — the corner where log-loss is 2.72,
i.e. unconverged and miscalibrated. AU-PRC selection would have chosen exactly that cell.
Selected: 650M `max_L33`, **C=1e-2, eta0=1e-4**.

#### Results

| | test AU-ROC | test AU-PRC | P | R | F1 |
|---|---|---|---|---|---|
| **ESM-2 650M `max_L33`** | **0.9681** [0.921, 0.997] | **0.9526** [0.857, 0.996] | 0.830 | 0.929 | 0.876 |
| BLASTp @ e≤3.16e-2 | 0.9040 | 0.7552 | 0.833 | 0.952 | 0.889 |
| amino-acid composition (floor) | 0.7020 | 0.6737 | | | |
| *PET-only task, for contrast* | *0.8856* | *0.8150* | *0.742* | *0.742* | *0.742* |

`TP=39 FP=8 FN=3 TN=54`. Recall by tier: PET 30/31, PBAT 3/5, aliphatic 6/6 — all three
substrate classes, not carried by the PET majority. **All 8 false positives are
fold-matched esterases; zero are naive controls.**

#### FIRST SIGNIFICANT WIN OVER BLAST

Paired bootstrap over the 66 test homology components:

| metric | delta (probe − BLAST) | 95% CI | verdict |
|---|---|---|---|
| AU-ROC | **+0.0603** | [+0.0058, +0.1237] | **SIGNIFICANT** |
| AU-PRC | **+0.1708** | [+0.0136, +0.2722] | **SIGNIFICANT** |

Against the PET task, where AU-ROC was +0.0074 [−0.0959, +0.1232] and AU-PRC −0.0412
[−0.1899, +0.0756] — both non-significant, BLAST ahead on AU-PRC.

At the *operating point* the two are equivalent (identical 8 false positives, BLAST one TP
ahead). The win is in ranking quality.

#### Validation vs test, and what the comparison actually shows

| | VAL probe | VAL BLASTp | TEST probe | TEST BLASTp |
|---|---|---|---|---|
| AU-ROC | 0.9976 | 0.9910 | **0.9681** | 0.9040 |
| AU-PRC | 0.9954 | 0.9895 | **0.9526** | 0.7552 |
| Precision | 0.9773 | 0.9545 | 0.8298 | 0.8333 |
| Recall | 1.0000 | 0.9767 | 0.9286 | 0.9524 |
| TP/FP/FN | 43/1/0 | 42/2/1 | 39/8/3 | 40/8/2 |

**On validation the methods are nearly tied (+0.006 AU-PRC); on test the probe leads by
+0.197.** The gap *widens* on held-out data because BLASTp degrades far more
(−0.234 AU-PRC val→test, against the probe's −0.043).

**Diagnosed cause — test drew harder negatives, not more remote positives.** Test positives
are actually closer to the BLAST database (median e-value 3.7e-15 vs val 1e-09). But 8 of 62
test negatives clear BLASTp's cutoff against only 2 of 76 in val. Both methods make 8 false
positives; BLASTp scatters them among its top hits while the probe keeps them near its
boundary.

**This tempers the claim.** The paired bootstrap resamples within test and cannot account
for split composition. The defensible statement is *"the probe degrades more gracefully as
negatives get harder"*, not *"uniformly better"*.

#### The precision question

At the operating point BLASTp shows precision 0.8333 vs the probe's 0.8298. That 0.0035 is
**one sequence** — both make exactly 8 false positives; BLASTp catches one more true
positive so its denominator is 48 not 47. Posed properly as a trade-off:

| precision required | probe recall | BLASTp recall | enzymes returned (of 42) |
|---|---|---|---|
| ≥ 95% | 59.5% | 2.4% | 25 vs 1 |
| ≥ 90% | **78.6%** | 2.4% | **33 vs 1** |
| ≥ 85% | 92.9% | 2.4% | 39 vs 1 |
| ≥ 80% | 97.6% | 95.2% | 41 vs 40 |

**BLASTp plateaus near 84% precision and cannot be pushed higher** — its top-ranked hits
already contain homologous esterases, so tightening the cutoff removes true positives first.

#### Train/val/test for the probe

| | TRAIN | VAL | TEST |
|---|---|---|---|
| AU-ROC | 1.0000 | 0.9976 | 0.9681 |
| AU-PRC | 1.0000 | 0.9954 | 0.9526 |
| Precision | 0.9619 | 0.9773 | 0.8298 |
| Recall | 1.0000 | 1.0000 | 0.9286 |

Train AU-ROC is exactly 1.0000 (separable regime) but the train→test gap is only −0.032.
**Ranking transfers, the threshold does not**: AU-ROC/AU-PRC fall −0.029/−0.043 val→test
while precision falls −0.148, because 0.1455 was fitted where negatives were easier and val
prevalence (0.361) differs from test (0.404). The val column is not unbiased — it selected
C, eta0 and the threshold.

**Figure:** `https://claude.ai/artifact/S1rzDp1cJYxA39xSTPZteE` — PR curves, both splits,
operating points marked.

#### Generalisation to an unseen lineage

| | PHA depolymerases flagged |
|---|---|
| ESM-2 650M | **30/36 (83%)** |
| BLASTp | 28/36 (78%) |

PHA depolymerases were excluded from training and from the BLAST database and are a
separate enzyme lineage. Recognising them is evidence the model learned "polyester
hydrolase" as a property rather than memorising the cutinase family — and it cannot be
passed by flagging everything, since the same model returns 0/36 on non-esterase controls.

> **SUPERSEDED — see §25 (2026-10-11).** The premise in the paragraph above, that these are
> an *unseen* separate lineage, was asserted when the holdout was created and never tested.
> It is false for 14 of the 36: they share a homology component with a trained positive, and
> 25 trained positives already act on a PHA. Corrected probe figure on the 22 genuinely clean
> proteins: **16/22 (73%)**, not 30/36 (83%). The BLASTp 28/36 cannot be corrected without
> re-running it and must not be compared against the corrected probe number.

#### The confound also improved

| task | positives Actinomycetota | negatives | gap |
|---|---|---|---|
| PET only | 72.6% | 66.0% | +6.6 pts |
| any polyester | 66.0% | 70.4% | **−4.4 pts** |

Adding PBAT and aliphatic degraders (mostly Pseudomonadota and Bacillota) dilutes the
actinomycete dominance. An Actinomycetota-only classifier now scores AU-ROC 0.5842. This
matters because taxonomy was the 8M probe's single largest error mode (section 10).

#### Honest framing

The model did not improve — **the question changed to one the representation can answer.**
The new task is also genuinely easier: the composition floor is 0.7020. The defensible
claim is the lift over floor plus the significant margin over BLAST, not the raw 0.968.

**Caveat carried in all three documents:** the test sequences are the PET task's test set
relabelled. Split, identity bound and all tuning were redone, but these sequences are not
virgin — their behaviour under a different labelling has been observed.

#### The two claims, now cleanly separated

```
"detects polyester-degrading enzymes"   AU-ROC 0.968, beats BLAST on both metrics (significant)
"detects PET specifically"              AU-ROC ~0.68, no method clears it
```

Both measured, and the SAE explains the gap: the latents find the catalytic triad every
polyester hydrolase shares. This also resolves the cutinase objection — under the new
labels cutinases are *positives*, so validating the triad-finder on them is coherent rather
than contradictory.

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

> **SUPERSEDED by D10 (2026-09-28).** The 10-fold cluster CV described here was tested and
> is **not viable on this data**: one homology component holds 84% of the training
> positives, so it cannot be divided across folds. The fixed train/val/test split is the
> headline, and nested CV is abandoned. The rationale below for *why* CV would have been
> preferable still stands — it is the reason the CI is as wide as it is.

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

### D10. Hyperparameter selection stays on val; cluster CV is abandoned (2026-09-28)

**Supersedes D5.** Decision: **keep the three-way protocol — fit on train, select C/eta0 and
the decision threshold on val, read test once.** Do not move selection into a
cross-validation loop.

Reopened while looking for more statistical power. The test set is small: 42 positives
spanning only 20 homology components, Kish effective n **10.6**, 95% CI half-width on recall
**+/-18 pp**. Pooling val+test looks attractive on paper — 85 positives, 41 components,
n_eff 18.4, +/-13.7 pp — and val/test components are disjoint, so pooling is structurally
clean.

**Two protocols considered, one tested:**

1. *Tune on val, report on pooled val+test.* **Rejected as invalid.** Val would be in-sample
   for selection. Same class of error as the BLAST threshold asymmetry caught earlier
   (F1 0.820 tuned in place vs 0.598 carried honestly).
2. *Nested CV — selection in an inner component-aware CV inside train, then one pooled read.*
   **Rejected: not viable on this data.** Tested directly (`scratchpad/nested_demo.py`,
   5-fold outer x 4-fold inner over train, `max_L33`):

| outer fold | n | positives | chosen C | chosen eta | outer AU-PRC |
|---|---|---|---|---|---|
| 1 | 256 | **256** | 1e-01 | 1e-03 | **undefined — no negatives** |
| 2 | 83 | **0** | 1e-02 | 1e-03 | **undefined — no positives** |
| 3 | 83 | 16 | 1e-02 | 1e-03 | 0.7228 |
| 4 | 83 | 18 | 1e-02 | 1e-03 | 0.8095 |
| 5 | 83 | 13 | 1e-02 | 1e-03 | 0.5590 |

**Cause: one component holds 256 of 303 train positives (84%);** the other 32 share 47. CV
assigns whole components to folds, so that component lands entirely in one fold, which then
holds all the positives while the rest average 12. Two of five folds yield no metric at all,
and the three that do swing 0.56-0.81 on 13-18 positives each. This is the indivisible-
component problem already recorded for the PET task, now confirmed for polyester.

**Consequence — and the framing to use.** The current train/val/test split is not a
compromise; it is the only arrangement this homology structure permits. `assign_components`
works *because* it does not build equal folds: it places the giant component in train and
balances the remainder across val and test. State that positively rather than
apologetically.

**What remains available:** a pooled val+test read as a *supporting* analysis only,
restricted to threshold-free metrics (AU-ROC, AU-PRC). Hyperparameters were selected on val,
so the pooled figure is not a clean held-out estimate and must never be the headline. Real
power needs more independent families (PEZy-miner negatives; positives outside the cutinase
clade), not more passes over the same data.

**Selection stability, recorded:** 4 of 5 outer folds chose C=1e-2, one chose C=1e-1; all
chose eta=1e-3. Shipped config is C=1e-2, eta0=1e-4 — same regularization, different
learning rate, selected on val log-loss.

---

### 14b. Why BLASTp's test PR curve is a flat shelf (2026-09-28)

Follow-up to the PR-curve figure. BLASTp's test curve sits at **0.844 precision from recall
0.05 to 0.95** with no high-precision region. Diagnosed by walking down its ranking: ranks 2,
3 and 5 are tier-3 fold-matched esterases (P86325 e=1e-74, Q47M62 e=3e-74, Q01470 e=4e-55),
interleaved with genuine PET hydrolases at ranks 1 and 4. Six of 62 test negatives beat
BLASTp's own median positive (e ≤ 3.7e-15).

Because the contamination is at the *top* of the ranking, no threshold can remove it —
tightening the cutoff strips true positives off the bottom first. That is the shelf.

First-error rank by split/method: val BLAST 41 (93.0% recall already reached), val probe 36
(81.4%), **test BLAST 2 (2.4%)**, test probe 25 (57.1%). This is the mechanism behind the
0.9895 → 0.7552 val→test AU-PRC drop for BLAST.

The probe scores 4 of the 6 correctly (0.187–0.313) and is caught by the same two hardest
cases (0.525, 0.522) — evidence it reads signal beyond raw homology, and an honest statement
of where that signal runs out. Recorded in `artifacts/BLAST_vs_ESM2_650M_Polyester.md`
("Reading the shape of the BLASTp test curve"). Figure itself is correct as drawn; no change.

*Caveat:* six sequences in one split. Explains the gap's mechanism, not its magnitude.

---

### 14c. Artifact audit (2026-09-28)

Full consistency pass over `artifacts/`. Verified: every headline number in the polyester
docs matches `data/sae_650M_L33/{polyester_results,pr_curves}.json`; every cross-document
file reference resolves; both trade-off tables in `BLAST_vs_ESM2_650M_Polyester.md`
(recall-at-precision and precision-at-recall) match source and are not transposed.

Three gaps found and fixed:

1. **No supersession markers.** `BLAST_vs_ESM2_650M.md`, `ESM2_650M_Dense_Run.md`,
   `ESM2_8M_Dense_Run.md`, `BLAST_vs_ESM2_8M.md` report the **PET-only** task with nothing
   saying the task was later redefined. A reader landing on one would take a superseded
   headline as current. Banner added to each pointing at the polyester equivalents;
   `SAE_Feature_Interpretation.md` got a scope note instead (it is not superseded — its
   fold-not-substrate finding is what motivated the redefinition).
2. **The SAE leak correction lived only in PROGRESS.md.** The artifact that actually reports
   the SAE work carried the conclusion but never the number or the correction. Added §6b
   with the leak-free table (0.5108, 29th percentile of random) and the correction block;
   `--select-on train` added to its Reproduce commands.
3. **Broken markdown tables.** Substrate cells containing raw `|` (`PBS|PHA`,
   `PBAT|PHA|PLA|PUR`, `PCL|PES|PHBV|PHO`) split their rows into extra columns, so 4 rows
   across two docs rendered misaligned. Pipes escaped. All tables in all 9 docs now validate.

*Self-inflicted, recorded for honesty:* the first repair attempt used a cell-merge heuristic
that damaged tables in 6 files that were already correct. Restored from git (HEAD for the
tracked four, index for the two staged ones), then re-applied only the intended edits and
fixed the pipes by escaping instead. Verified no content loss: +8 lines per banner file,
diffs contain banners and escapes only.

---

### 14d. Label provenance audited; the "fold-matched" tier is only half fold-matched (2026-09-28)

Prompted by the question "did the labels come from PAZy and PlasticDB?" Re-derived from the
source files rather than from memory.

**Positives only.** PAZy + PlasticDB supply the 542-row positive set (PAZy only 232,
PAZy+PlasticDB 210, PlasticDB only 100), each row carrying a traceable `source_ids`. All 423
negatives come from UniProt queries in `fetch_negatives.py` and were **never assayed** —
their label is an assumption, not a measurement. (PEZy-miner is a *different* resource and
is still unused; its 36 assayed non-degraders remain unextracted.)

**12% of modelled positives are not experimentally verified:**

| class | n | verified | extrapolated |
|---|---|---|---|
| `1_pet` | 320 | 301 | 19 |
| `2_other_polyester` | 68 | **39** | **29** |
| `heldout_pha` | 36 | 10 | 26 |

`2_other_polyester` is the weak half — 29 of 68 rest on homology annotation. That class is
exactly what the task redefinition added to the positive side, so the polyester headline
partly rests on unmeasured labels. Worth stating before a judge finds it.

**Why EC 3.1.1 (recorded, was only in the script docstring):** it is the carboxylic-ester
hydrolase subclass PETases themselves belong to (PET hydrolase 3.1.1.101, MHETase 3.1.1.102,
cutinase 3.1.1.74). Same chemistry, same fold, same catalytic triad, different substrate —
which is what makes the tier hard, and why class 3 is deliberately *not* filtered at 30%
identity. Verified clean: 0 of 185 class-3 entries are EC 3.1.1.101, so the accession and
>=90%-identity exclusions worked.

**NEW LIMITATION — EC is a leaky proxy for fold.** Composition of `3_fold_matched_esterase`:

| EC | n | enzyme | alpha/beta fold? |
|---|---|---|---|
| 3.1.1.96 | 56 | D-aminoacyl-tRNA deacylase | **no** |
| 3.1.1.29 | 43 | peptidyl-tRNA hydrolase | **no** |
| 3.1.1.- / .1 / .3 | 39 | esterases, lipases | yes |
| others | 47 | phospholipases, lactonases | mixed |

**99 of 185 (54%) are tRNA-processing hydrolases** — in EC 3.1.1 by formal reaction
chemistry (they cleave an aminoacyl-tRNA ester bond) but structurally unrelated to
cutinases. Only ~30 of the tier are genuine esterases.

This is the mechanism behind the caveat already in `SAE_Feature_Interpretation.md` §7 (only
41% of tier-3 sequences have a catalytic serine at the peak). **Tier 3 is easier than its
name claims**, so tier-3 numbers understate the task's difficulty and overstate performance.
It does not affect the *overall* polyester metrics' validity, but it does weaken the claim
that the negative set is fold-matched — which is the project's central design argument.
Recorded in `artifacts/EDA_polyester_task.md` ("Label provenance").

---

### 21. The negative set, clarified on fold rather than EC — val has no hard negatives (2026-10-11)

Prompted by "can we clarify our negative set", asked while planning the hard-negative triad
test. Every number below was re-derived by querying UniProt live for InterPro, Pfam and
`ACT_SITE` on all 423 negatives, then recomputing the probe metrics from
`polyester_scores_650M.npz`. The top line of the decomposition reproduces the published
headline exactly (test AU-ROC 0.9681 / AU-PRC 0.9526), which is what licenses the rest.

**This supersedes the 14d estimate.** 14d put tier 3 at "only ~46% fold-matched" using
*non-tRNA* as a proxy for fold. On direct fold annotation it is **27.0%** (50 of 185). The proxy was too
generous: it counted GDSL/SGNH lipases, patatin phospholipases and β-lactamase-fold
hydrolases as fold-matched when none of them are α/β hydrolases.

**What the 423 negatives actually are.** All from UniProt, all never assayed. Classified by
InterPro `IPR029058` (α/β hydrolase fold), cross-checked against Pfam:

| group | n | evidence |
|---|---|---|
| tRNA-processing hydrolases | 99 | PF02580 D-aminoacyl-tRNA deacylase ×56 (EC 3.1.1.96), PF01195 peptidyl-tRNA hydrolase ×43 (EC 3.1.1.29) |
| other EC 3.1.1, non-α/β fold | 36 | PF13472/PF00657 GDSL-SGNH, PF01734 patatin, PF00144 β-lactamase, PF01661 macro domain, PF01339 CheB |
| genuine α/β hydrolase fold | 51 | `IPR029058`; 50 in tier 3, 1 mislabelled into tier 4 |
| naive controls (no EC 3.1.1) | 237 | kinases, gyrases, monooxygenases, 61 with no EC at all |

So **51 of 423 negatives (12%)** are fold-matched. The tRNA entries are in the set because
they are EC 3.1.1 by reaction chemistry — the `ec:3.1.1.-` trap already in the environment
notes. The query was correct; an EC class is simply not a fold. Pfam agrees with InterPro
independently on every one of the 135 tier-3 non-α/β entries, so this is not an artifact of
leaning on a single accession.

**The split problem.** The 99 tRNA entries collapse into exactly **3 homology components**,
and the identity-bounded split sent each component entirely to one split — correct behaviour
for a homology-aware split, with a bad consequence:

| component | n | goes to |
|---|---|---|
| `P0A025` | 56 | train |
| `A1A141` | 40 | **val** |
| `B3ESC6` | 3 | test |

| negatives by hardness | train | val | test |
|---|---|---|---|
| all | 285 | 76 | 62 |
| tRNA-processing | 56 | **40** | 3 |
| EC 3.1.1 non-tRNA | 63 | **0** | 23 |
| α/β hydrolase fold | 40 | **1** | 10 |
| α/β fold + GxSxG + 3 curated `ACT_SITE` | 32 | **1** | 9 |

**Val's 76 negatives are 40 tRNA hydrolases plus 36 random proteins. It contains zero
genuine fold-matched esterases.**

**What that does to the headline.** Test AU-ROC decomposed by negative subgroup, positives
held fixed at 42; CIs are 5,000-draw bootstrap over both classes:

| negatives | n | AU-ROC | AU-PRC |
|---|---|---|---|
| all — published headline | 62 | 0.9681 | 0.9526 |
| naive controls | 36 | 1.0000 | 1.0000 |
| tRNA-processing | 3 | 1.0000 | 1.0000 |
| EC 3.1.1 non-tRNA (= 13 wrong-fold + 10 α/β; see §28g) | 23 | 0.9141 [0.840, 0.971] | 0.9526 |
| α/β hydrolase fold | 10 | 0.8167 [0.676, 0.926] | 0.9550 |
| α/β fold + GxSxG + triad | 9 | 0.7963 [0.659, 0.910] | 0.9550 |

39 of the 62 test negatives (63%) are separated perfectly, so the headline is a weighted
blend in which most of the weight is free. The fold-matched task is **0.82, with the CI
reaching down to 0.68**. On val the same decomposition gives 1.0000 against the tRNA family
and 0.9948 against naive controls, and the fold-matched cell is *unmeasurable*.

**Three consequences, all load-bearing.**

1. The operating threshold **0.1455 was tuned on a split containing no hard negatives.**
   That is the mechanism behind test precision falling to 0.830 — the threshold was set
   where nothing hard was pushing back.
2. The seven latents were selected on **train+val** by Cliff's δ against this negative set,
   and val contributes 76 easy negatives to that δ. Recomputing δ against only the 51
   fold-matched negatives is therefore **required**, not an optional robustness check.
3. **No claim about the hard task can rest on val.** Only test can carry one, and there
   n = 10.

**Fix, and why relabelling is not enough.** Rebuilding tier 3 on fold (already on the
Still-open list) exposes the real problem: the dataset holds only 51 fold-matched negatives
against 388 positives, so no relabelling of existing rows can give val a usable hard
negative set. New negatives have to be fetched — UniProt reviewed, `IPR029058` plus three
curated `ACT_SITE` residues, never assayed on a plastic, excluding the cutinase/PETase
families — and there are thousands available. Target 300–400 so that every split, val
included, gets genuine hard negatives, then re-split and re-run detection and latent
selection.

Scratch: `neg_annot.json`, `neg_actsites.json`, `pos_annot.json`, `negclar.py`, `negclar2.py`.

---

### 22. Test 3 — the δ≥0.9 selection was powered by the easy negatives (2026-10-11)

The check §21 said was required. The seven latents were selected on train+val by Cliff's
δ ≥ 0.9 plus bootstrap top-10 ≥ 75%, against a negative set that §21 showed is 88% not
fold-matched. This re-runs the identical procedure against each negative subgroup separately.

**Implementation validated first.** Recomputing δ for all 10,240 latents against the original
all-negatives basis reproduces `selection_full.npz` with **max |difference| = 0.00e+00**, and
the stored per-latent values to ten decimal places. The procedure below is therefore the same
one, differing only in which negatives it sees.

**δ by negative subgroup** (train+val, n_pos = 346, 2,000-draw bootstrap CI):

| latent | all neg (361) | naive (201) | tRNA (96) | other EC, wrong fold (23) | **α/β FOLD-MATCHED (41)** |
|---|---|---|---|---|---|
| 9529 Ser | 0.959 | 0.989 | 0.997 | 0.967 | **0.716** [0.64, 0.79] |
| 5201 pocket | 0.949 | 0.986 | 0.998 | 0.968 | **0.646** [0.54, 0.74] |
| 7942 pocket | 0.937 | 0.980 | 0.998 | 0.965 | **0.569** [0.40, 0.71] |
| 5255 pocket | 0.936 | 0.989 | 0.998 | 0.977 | **0.506** [0.34, 0.65] |
| 2661 Asp | 0.936 | 0.974 | 0.997 | 0.956 | **0.595** [0.48, 0.70] |
| 7734 His | 0.934 | 0.975 | 0.996 | 0.954 | **0.577** [0.46, 0.68] |
| 5271 shell | 0.934 | 0.971 | 0.992 | 0.949 | **0.608** [0.49, 0.71] |

**Not one of the seven clears δ ≥ 0.9 against fold-matched negatives.** The best, 9529, falls
to 0.716 with a CI topping out at 0.79 — the criterion is excluded, not merely missed. 5255
reaches 0.506, i.e. AUC 0.75.

The tRNA-processing family gives the **highest** δ of any subgroup, 0.992–0.998. The latents
were, in effect, selected for separating plastizymes from D-aminoacyl-tRNA deacylases.

**Re-running the full selection on fold-matched negatives only** (all 10,240 latents, same
200-draw random-half bootstrap, same δ ≥ 0.9 and top-10 ≥ 75% criterion):

| basis | n_neg | latents passing | latents with δ≥0.9 | max δ |
|---|---|---|---|---|
| all negatives (original) | 361 | **5** — 9529, 5201, 2661, 7942, 5255 | 26 | 0.9589 (9529) |
| fold-matched only | 41 | **0** | **0** | 0.8469 (latent 9638) |

Against fold-matched negatives **no latent in the dictionary** reaches δ ≥ 0.9; the ceiling is
0.847. And the seven fall out of contention entirely — top-10 frequency **0% for all seven**,
ranks **133, 326, 435, 482, 540, 570, 810** of 10,240:

| latent | δ fold-matched | top-10 | rank |
|---|---|---|---|
| 9529 | 0.7162 | 0% | 133 |
| 5201 | 0.6463 | 0% | 326 |
| 5271 | 0.6079 | 0% | 435 |
| 2661 | 0.5951 | 0% | 482 |
| 7734 | 0.5773 | 0% | 540 |
| 7942 | 0.5693 | 0% | 570 |
| 5255 | 0.5057 | 0% | 810 |

**Control — it is not a sample-size artifact.** n_neg = 41 is small, so each easy subgroup was
subsampled to exactly 41 and δ recomputed, 500 draws (median [5th, 95th]):

| latent | naive n=41 | tRNA n=41 | all-neg n=41 | fold n=41 |
|---|---|---|---|---|
| 9529 | 0.990 [0.99, 0.99] | 0.997 [1.00, 1.00] | 0.960 [0.93, 0.98] | **0.716** |
| 5201 | 0.986 [0.98, 0.99] | 0.998 [1.00, 1.00] | 0.951 [0.91, 0.98] | **0.646** |
| 7942 | 0.980 [0.97, 0.98] | 0.998 [1.00, 1.00] | 0.946 [0.89, 0.98] | **0.569** |
| 5255 | 0.989 [0.99, 0.99] | 0.998 [1.00, 1.00] | 0.939 [0.88, 0.98] | **0.506** |
| 2661 | 0.974 [0.97, 0.98] | 0.997 [1.00, 1.00] | 0.939 [0.90, 0.97] | **0.595** |
| 7734 | 0.975 [0.97, 0.98] | 0.996 [0.99, 1.00] | 0.935 [0.89, 0.96] | **0.577** |
| 5271 | 0.971 [0.97, 0.98] | 0.992 [0.99, 0.99] | 0.937 [0.89, 0.96] | **0.608** |

At *identical* n, the easy subgroups still return 0.93–0.998 and the 5th percentile of the
all-negatives draw never drops below 0.88 — far above every fold-matched point estimate. The
collapse is driven by negative-set **composition**, not by n.

**Minor note on the bootstrap frequency.** δ reproduces exactly, but top-10 frequency is
draw-dependent: this run gives 7734 = 70% and 5271 = 73% where the stored run gave 79.5% and
78.5%. Rank order is unchanged and the same five pass strictly. Consistent with §16's existing
caveat that 7734 and 5271 straddle the stability cutoff. Irrelevant to the conclusion here,
where all seven sit at 0%.

**What this does and does not overturn.**

- It does **not** touch the localisation result. 57/59 proteins at 96.6%, the 0-of-10,240
  random-latent null, and the offset consistency all stand — those ask *where* a latent fires,
  and are unaffected by which negatives were used to pick it.
- It **does** overturn the claim implicit in "δ ≥ 0.9 selection": these are **not** latents that
  distinguish plastizymes from fold-matched esterases. They distinguish hydrolases-with-a-
  canonical-active-site from tRNA deacylases and random proteins. §15 and §14d already read
  this direction from the 1,326-esterase run; test 3 makes it quantitative on our own data and
  shows the selection criterion itself carried the confound.
- The honest statement of the explainability claim is therefore: the SAE latents are a
  **validated readout of serine-hydrolase catalytic machinery** — which is worth having, and is
  independently verified — but they are **not** the signal the probe uses to call something a
  polyester degrader. Whatever drives the 0.82 fold-matched AU-ROC is not in these seven.
- The ceiling of 0.847 over the whole dictionary says a single-latent explanation of the hard
  task does not exist in this SAE at layer 33. If the probe's fold-matched margin is real, it is
  distributed, and finding it needs a multi-latent or attribution-based method, not δ ranking.

Caveat carried forward: n_neg = 41 fold-matched negatives in train+val (40 train, 1 val). The
collapse is robust to the size control, but the *ceiling* of 0.847 and the identity of the new
top latents (9638, 7370, 6875) are estimated on 41 proteins and should not be treated as a
replacement selection until the negative set is rebuilt per §21.

Scratch: `test3.py`, `test3b.py`, `test3c.py`, `test3_delta.json`.

#### 22b. Revision — "collapsed" was the wrong word; the honest verdict is undetermined

Prompted by the objection that δ of 0.5–0.72 is a *large* effect, not a collapse, and that the
latents may still be isolating plastizyme machinery from fold-matched esterases. The objection
is correct on its first half, and the follow-up analysis overturns it on the second. Both
corrections are recorded because §22 as first written was wrong in one direction and the naive
reading is wrong in the other.

**Where the objection is right — these are not noise-level values.** δ over all 10,240 latents
on the plastizyme-vs-fold-matched comparison (train+val, 346 vs 41):

| percentile | δ |
|---|---|
| 50th (median latent) | +0.0000 |
| 75th | +0.1850 |
| 95th | +0.5866 |
| 99th | +0.7302 |
| max | +0.8469 |

Only **8.1%** of latents reach δ ≥ 0.5 and 11.1% reach the conventional "large" threshold of
0.43. 9529 at 0.716 is the **98.7th percentile**. Describing that as a collapse was wrong; the
correct statement is that it fails the 0.9 *selection criterion*, which is a different claim.

**Length is not the confound.** Sequence length alone gives δ = +0.052 on the same comparison
(positive median 303 aa, fold-matched negative median 298 aa). Ruled out.

**Homology is the confound.** The 346 train+val positives collapse into 54 homology components
and **one component, `BPS0190`, holds 256 of them — 74%**; 38 components are singletons. The 41
fold-matched negatives spread across 32 components, largest 3. Collapsing both sides to one
value per component (median over members):

| latent | per-protein δ | per-component δ | component-level percentile |
|---|---|---|---|
| 9529 Ser | +0.716 | **+0.116** | 79.4 |
| 5201 pocket | +0.646 | **+0.108** | 78.2 |
| 7942 pocket | +0.569 | +0.068 | 70.8 |
| 5255 pocket | +0.506 | −0.010 | 41.3 |
| 2661 Asp | +0.595 | **−0.104** | 21.3 |
| 7734 His | +0.577 | **−0.160** | 12.8 |
| 5271 shell | +0.608 | **−0.178** | 10.7 |

Component-level median across the dictionary is 0.000 and its max is +0.560, so 9529's +0.116
is "small" by convention and the Asp, His and shell latents turn **negative**.

**Cluster bootstrap** (resample components with replacement, keep all members, 2,000 draws) —
protein-level δ with a CI that respects homology structure:

| latent | δ | 95% CI |
|---|---|---|
| 9529 | +0.716 | [−0.254, +0.898] |
| 5201 | +0.646 | [−0.242, +0.829] |
| 5271 | +0.608 | [−0.488, +0.855] |
| 2661 | +0.595 | [−0.439, +0.839] |
| 7734 | +0.577 | [−0.508, +0.837] |
| 7942 | +0.569 | [−0.319, +0.827] |
| 5255 | +0.506 | [−0.370, +0.738] |

**Every CI includes zero.**

**Verdict.** Whether the seven latents separate polyester degraders from fold-matched esterases
is **undetermined**, not refuted. The per-protein δ of 0.72 is one cutinase family counted 256
times against 41 assorted esterases. With homology respected the point estimate falls to ≈0.12
and the interval spans zero — and it spans zero because the effective sample is ~54 positive
families with one at 74% against 32 negative families, i.e. the test has almost no power. This
is the same wall as D5/D10, where one component holding 84% of train positives made cluster CV
non-viable.

Both of §22's directional claims therefore need qualifying:

- "The latents do not distinguish plastizymes from fold-matched esterases" — **overstated**.
  They fail the δ ≥ 0.9 criterion and are not demonstrably better than chance once homology is
  respected, but the data cannot rule the effect out either.
- "δ ≈ 0.5–0.72 shows the machinery still isolates plastizymes" — **also unsupported**, because
  that number is driven by positive-set homology density, not by 346 independent observations.

What survives unchanged: the localisation result (57/59 proteins, the 0-of-10,240 random-latent
null, offset consistency), which asks *where* a latent fires and is independent of the negative
basis; and the reading that these latents are a validated readout of serine-hydrolase catalytic
machinery.

**Consequence for the §21 fix.** The rebuild cannot be negatives-only. 300–400 fold-matched
negatives would fix one side, but with 74% of positives in a single component the comparison
stays underpowered. The positive side needs either homology-aware weighting (one vote per
family) or genuine positive diversity beyond the dominant cutinase clade — and any future δ or
AU-ROC on the hard task must be reported with a cluster bootstrap, never a per-protein one.

Scratch: `test3d.py`, `test3e.py`, `test3d.json`.

#### 22c. Is the Ser–His–Asp machinery shared across the groups? Yes for esterases, no for tRNA

Prompted by "is the serine hydrolase catalytic machinery common across plastizymes, hydrolases
and the tRNA thing also?" Answered two independent ways — from UniProt's own catalytic
annotations, and from latent activation — which agree.

**What UniProt annotates as catalytic:**

| group | n | sites per protein | residues | UniProt description |
|---|---|---|---|---|
| fold-matched esterase | 51 | 43 of 47 have **3** | S 46, H 44, D 35, E 8 | "Nucleophile" (Ser); "Charge relay system" (His, Asp) |
| tRNA-processing | 99 | 43 have **exactly 1**, 56 have none | **H 43, zero Ser** | "Proton acceptor" |
| other EC 3.1.1, wrong fold | 36 | mixed | S 19, D 17, H 14 | "Nucleophile" |
| naive control | 237 | 147 have none | D 34, H 25, E 23, C 18, S 10 | scattered |

**The tRNA group are not serine hydrolases at all.** Peptidyl-tRNA hydrolase (43, PF01195) uses
a single His general base to activate water — one annotated site, "Proton acceptor", no
nucleophile. D-aminoacyl-tRNA deacylase (56, PF02580) carries **no active-site annotation
whatsoever**. They are EC 3.1.1 because they cleave an ester bond by an unrelated mechanism.
So 99 of the 185 entries in the tier labelled "fold-matched esterase" lack both the fold and
the catalytic machinery.

**Latent activation agrees exactly** (median `feat_max` / % above the 0.3 firing threshold):

| group | n | 9529 Ser | 2661 Asp | 7734 His | 5201 pocket |
|---|---|---|---|---|---|
| plastizymes | 388 | 0.970 / 96.1% | 0.917 / 84.0% | 0.919 / 89.4% | 0.942 / 96.1% |
| fold-matched esterase | 51 | 0.783 / **100.0%** | 0.757 / 80.4% | 0.709 / 84.3% | 0.793 / **100.0%** |
| other EC 3.1.1, wrong fold | 36 | 0.043 / 13.9% | 0.030 / 0.0% | 0.029 / 0.0% | 0.043 / 0.0% |
| tRNA-processing | 99 | **0.000 / 0.0%** | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |
| naive control | 237 | 0.018 / 0.0% | 0.019 / 0.0% | 0.014 / 0.0% | 0.030 / 0.8% |
| held-out PHA | 36 | 0.837 / 97.2% | 0.845 / 97.2% | 0.729 / 97.2% | 0.759 / 100.0% |

Three readings, each load-bearing:

1. **Shared with genuine esterases.** 100% of fold-matched esterases fire 9529 and the pocket
   latent, against 96.1% of plastizymes. Same triad, same fold, same charge relay, and the SAE
   reads it in both. The gap is one of degree — median 0.783 vs 0.970 — which is precisely the
   residual δ ≈ 0.6–0.72 of §22, and §22b shows that residue is confounded by positive-set
   homology. **This is the mechanistic reason δ cannot be large against this group.**
2. **Absent in the tRNA group.** Median exactly 0.000, not one of 99 above 0.3. The δ of 0.997
   against them was the latent separating plastizymes from proteins with no Ser nucleophile at
   all — a fold test with the answer written on it, not a hard negative.
3. **The wrong-fold EC 3.1.1 group is the discriminating control.** They *do* have Ser
   nucleophiles (19 annotated), and 13.9% fire 9529 — but **0% fire the Asp, His or pocket
   latents.** So the latents do not detect "a serine"; they detect the α/β charge-relay
   geometry. A Ser nucleophile in a GDSL/SGNH or patatin fold earns partial credit on the Ser
   latent and nothing else.

Incidental confirmation: held-out PHA scores 97.2% on all three triad latents, so the machinery
reading generalises to a substrate class absent from training.

Scratch: `mech.py`, `mech2.py`.



---

### 23. Dataset release v1 — packaged for hand-off (2026-10-11)

Asked for so the dataset can be recorded and given to someone else for experimentation. The
point of the release is not the sequences — those were always in
`dataset_splits_id40.tsv` — but the **fold-based negative classification** from §21–§22c,
shipped as a column so a downstream user cannot repeat the EC-class mistake.

**Files.**

| path | contents |
|---|---|
| `data/release/biopet_sae_dataset_v1.tsv` | 847 rows × 20 columns, sequences included (399 KB) |
| `data/release/biopet_sae_dataset_v1.fasta` | same sequences, header carries split/class/group (330 KB) |
| `DATASET_CARD.md` | the card, repo root, 215 lines |
| `src/biopet_sae/build_release.py` | builder; `--refresh` re-queries UniProt |
| `data/processed/uniprot_negative_annotation.json` | cached annotation, 55 KB, fetched 2026-10-11 |

Both release files are small enough to commit, unlike `data/mgnify/` (398 MB, still needs a
`.gitignore` entry).

**The columns that carry the new information:**

| column | why it exists |
|---|---|
| `negative_group` | the fold-based classification — `ab_hydrolase_hard_negative` (51), `ec311_wrong_fold` (36), `tRNA_processing_no_ser_nucleophile` (99), `naive_control` (237) |
| `component` | homology component, so a downstream user can cluster-bootstrap instead of resampling proteins |
| `ab_hydrolase_fold` | InterPro `IPR029058` flag |
| `gxsxg_elbow` | sequence-level nucleophile-elbow flag |
| `n_curated_active_sites` | curated UniProt `ACT_SITE` count |
| `ec`, `pfam` | the evidence behind `negative_group` |

`class_label` is retained for continuity with every published result, with the card stating
plainly that `3_fold_matched_esterase` is a misnomer (50 of 185 carry the fold).

**One bug found and fixed while building it.** The first pass of `build_release.py` extracted
EC numbers only from UniProt's `recommendedName` and `includes`, where the earlier scratch
annotation also read `alternativeNames` and `submissionNames`. Two negatives carry their
EC 3.1.1 only on an alternative name, so they were classified `naive_control` instead of
`ec311_wrong_fold` — group totals came out 34/239 against the correct 36/237. Caught by
diffing the written file against the independent scratch annotation, which is the only reason
it surfaced; the α/β count of 51 was unaffected either way. Fixed, re-fetched, now **0
disagreements**.

**Verification.** 21 assertions re-derived from the written TSV, not from the builder's
internals: row and split counts, all four group counts and their per-split breakdown, the
verified/unverified/extrapolated positive counts, the 74 components with 256 in the largest,
the tRNA active-site pattern (43 with one site, 56 with none, all carrying PF02580/PF01195),
and tier 3 at 185 with 50 carrying the fold. All pass. FASTA checked record-by-record against
the TSV: 847 records, 0 sequence mismatches.

**What the card tells a downstream user, in priority order.** Two numbers bound every claim —
**51** fold-matched negatives (1 in val) and **66%** of positives in component `BPS0190`.
Then: report test decomposed by `negative_group`; bootstrap over `component`, never over
proteins; say whether you tuned on val; state n for the fold-matched cell (10); leave
`heldout_pha` unused. The reference-results table gives the headline 0.9681 beside the
per-group decomposition down to 0.8167, so the blend is visible rather than asserted.

**Artifacts.**

| artifact | URL | state |
|---|---|---|
| **BioPET-SAE Dataset v1** | `https://claude.ai/artifact/TrmNy5H5KtFTNRgkKcLS8y` | **v1, new** |
| Plastizyme Detection Methods | `https://claude.ai/artifact/US29Zu7ScaoAKSoepRQeMg` | **v2** |

The methods page was revised rather than duplicated: the "fold-matched" tier corrected from
"86 genuine non-tRNA esterases" to 27% fold-matched with the four-group composition table; a
new "What the negatives actually are" subsection; the val-has-one-hard-negative flag with the
three-component mechanism; the positive-homology-density flag with the cluster-bootstrap
instruction; the label-asymmetry paragraph corrected to separate 48 unverified from 29
extrapolated-only; and the "what this does not show" bullet rewritten from the single 0.79
figure to the full per-group decomposition with its CI. Both pages are private — they need
sharing from the page's Share menu before anyone else can open them.

---

### 24. Positive substrate classes enumerated exhaustively; four errors corrected (2026-10-11)

Prompted by "is this exhaustive?" against the methods page's three-constraint table. It was
not. Re-derived from `plastizymes_merged.tsv` joined to the splits.

**What defines the positive classes** — not the substrate group, as the artifact implied, but
one question. Verified: all 320 `1_pet` carry PET in `substrates` and `is_pet = 1`; zero of the
68 do.

| class | n | definition |
|---|---|---|
| `1_pet` | 320 | acts on PET **at all**, whatever else it also does — 44 have >1 substrate |
| `2_other_polyester` | 68 | degrades a polyester but **never** PET |
| `heldout_pha` | 36 | **primary** substrate group is PHA — withheld entirely |

This is why `1_pet` contains 30 enzymes whose primary group is aliphatic polyester: they are
PET-active enzymes that also act on PCL or PLA.

**All 25 positive substrates.** Counts are enzymes acting on that substrate and sum past 388
because enzymes are promiscuous.

| family | substrates (n enzymes) |
|---|---|
| aromatic polyester — 284 primary | PET 320, PEF 7 |
| aromatic–aliphatic copolyester — 21 primary | PBAT 31, PBSeT 2, Ecovio-FT 2 |
| aliphatic polyester — 83 primary | PLA 52, PCL 43, PBS 22, PES 16, PBSA 15, PEA 3, PPL 1 |
| PHA family — held out as a class | PHA 23, PHB 11, PHBV 5, P3HP 3, P4HB 3, PHO 2, PHBH 1, PHBVH 1, PHPV 1 |
| non-polyester, on kept enzymes | PUR 16, PU 10, PC 5, PA 1 |

`PES` here is polyethylene **succinate**, not polyethersulfone — worth disambiguating wherever
it is written.

**Four errors in the methods artifact, now fixed (v3).**

1. **Removed-substrate list enumerated 105 of 109.** It named polyamide 65, polyurethane 15,
   polyolefin 13, polycarbonate 4, rubber 8 = 105, omitting polyether 2 and unknown 2. The
   total was right; the list was not.
2. **Kept-substrate list named 6 of 25.** PEF, PBSA, PBSeT, PEA, P3HP, P4HB, PPL and Ecovio-FT
   were all absent.
3. **"lineage: cutinase-like" was wrong twice over.** The kept positives span **74 homology
   components**, not one lineage; and the constraint applied is a holdout on *primary*
   substrate class, not a lineage filter. Relabelled "primary substrate class".
4. **`2_other_polyester` described as "20 PBAT, 48 aliphatic".** The 20 is correct as an
   enzyme count on PBAT, but the two figures were presented as a partition and are not one:
   53 act on an aliphatic polyester and 20 on PBAT, overlapping by 5. By primary class it is
   53 aliphatic / 15 aromatic copolyester. Corrected to state both framings.

**A substantive consequence, not just a wording fix.** Holding out PHA removed 36 enzymes whose
*primary* class is PHA. It did **not** remove PHA activity from the training set — **25 kept
positives still act on a PHA**. The PHA generalisation result (30/36, 83%) therefore measures
"unseen as a primary substrate class", not "never seen". That qualification now sits on the
artifact and should accompany the 83% wherever it is quoted.

Artifact: `https://claude.ai/artifact/US29Zu7ScaoAKSoepRQeMg` v3.

#### 24b. The positives do not hydrolyse esters exclusively, and `mechanism` is derived not assayed

Prompted by "do 1_pet and 2_other_polyester degrade only ester bonds?" They do not.

| mechanism as recorded | n |
|---|---|
| carboxylic ester **only** | **363 (93.6%)** |
| carboxylic ester + urethane | 17 |
| carbonate ester + carboxylic ester | 3 |
| carboxylic ester + unknown | 2 |
| carbonate ester + carboxylic ester + urethane | 2 |
| amide + carboxylic ester | 1 |

`1_pet` 304 of 320 pure, 16 mixed; `2_other_polyester` 59 of 68 pure, 9 mixed.

**`mechanism` is a deterministic function of `substrates`, so it is a lookup and not an assay.**
Checked: **0 of 79** distinct substrate strings map to more than one mechanism, and the
PUR/PU ↔ `urethane` relation is a perfect **39/39 biconditional in both directions**. The
column therefore records which bond types the *material* contains, never which bond the enzyme
was shown to cleave. Nothing in this dataset resolves that for any entry.

Two consequences:

- **The 19 urethane entries are probably still ester chemistry.** Polyester-polyurethanes are
  built from polyester polyols, so the backbone alternates ester and urethane linkages; an
  esterase degrading one most likely cleaves the ester segments. Treating these as evidence of
  urethane-bond hydrolysis would repeat the material-vs-bond error that §14d's exclusion rule
  exists to prevent — here inside our own positive set.
- **The 5 polycarbonate entries are esters regardless**, a carbonate ester (O–CO–O) rather
  than a carboxylic one.

**The one genuine exception: `BPS0383`**, mechanism `amide|carboxylic_ester`, substrates
`PA|PET`, `verified_activity = 1`. A single experimentally verified entry with polyamide
activity inside a positive set defined by ester hydrolysis. Not an extrapolation artifact.
Worth a decision: keep it and say so, or exclude it for consistency with the 65 polyamide
entries already removed. It is currently kept and undocumented, which is the one option that
should not stand.

**Also noted:** `BPS0190`, the representative of the homology component holding 66% of the
positives, is itself promiscuous — `PBS|PCL|PEF|PET|PHA|PHB|PLA|PU|PUR`, verified. The dominant
family is a broad-spectrum polyesterase rather than a PET specialist, which bears on how the
66% concentration should be described.

**Correction to the first version of this note.** It said the claim "hydrolyses only ester
bonds" is *false for 25 enzymes*, while also saying it is *unsupported either way* for the 19
urethane ones. Both cannot hold. 25 is the count of positives carrying any
non-carboxylic-ester annotation, which is not the count for which the exclusivity claim is
false — that conflates "has a non-ester annotation" with "demonstrably cleaves a non-ester
bond". The correct accounting:

| extra bond recorded | n | is "only ester bonds" false? |
|---|---|---|
| carbonate ester | 3 | **no** — a carbonate ester is an ester; not a counterexample |
| urethane | 19 | **undetermined** — polyester-PU contains both bonds |
| unknown substrate | 2 | **undetermined** by definition |
| amide | 1 | **yes** — nylon's backbone is amide, with no ester to cleave |

False for **1**, undetermined for **21**, not a counterexample for **3**. (Stated as "only
*carboxylic* ester bonds", the 3 carbonate entries become false too, giving 4.)

**The real reason to avoid the exclusivity wording** is stronger and simpler than a count: the
dataset cannot support an exclusivity claim for **any** of the 388, the 363 pure-ester ones
included, because none of them was assayed against an amide or a urethane. Absence of recorded
activity is evidence that nobody looked. The inclusion criterion is "hydrolyses **at least
one** polyester"; "hydrolyses only ester bonds" is unsupported across the whole positive set.

**Symmetric issue on the excluded side.** All **15** entries excluded as "non-ester mechanism
(POLYURETHANE)" have PU/PUR as their *only* substrate — 0 of 15 carry any other — and 13 of 15
are verified. So the boundary was applied consistently: kept enzymes have PUR alongside a real
polyester, excluded ones have nothing else. But if esterases degrade polyester-polyurethane by
attacking its ester segments, some of those 15 may themselves be esterases, making that
exclusion a decision about the substrate's name rather than about mechanism. Open question,
and it mirrors the one inside the positives.


---

### 25. The PHA holdout premise was never tested, and it is partly false (2026-10-11)

Prompted by "have you proved the PHA anywhere? we haven't proved that yet, right?" Correct —
the **number** was recorded (§14, and `methods_detection.html`), but the **premise** under it
was not. Both places call PHA depolymerases "a separate enzyme lineage" and title the result
"Generalisation to an unseen lineage". That claim was asserted when `build_dataset.py` assigned
`CLASS_HELDOUT_PHA` with the reason string *"distinct depolymerase lineage, held out as probe"*
and was never checked against the homology structure. This is the first test of it.

**What PHA is, for the record.** Polyhydroxyalkanoates are polyesters bacteria make as
carbon-storage granules — natural polymers, unlike every other substrate in the positive set.
PET is synthetic (c. 1941) and nothing evolved to degrade it, so PETases are repurposed
cutinases; PHA depolymerases are dedicated enzymes with an ancient natural substrate. Our 36
are named accordingly: 17 PHB depolymerase, 12 PHA depolymerase, plus variants, mostly from
*Cupriavidus* and *Paucimonas*. 10 of 36 have verified activity.

**The premise fails on two counts.**

| check | result |
|---|---|
| PHA homology components | 19, of which **6 are shared with trained positives** |
| PHA proteins with a ≥40%-identity relative in training | **14 of 36** |
| trained positives that already act on a PHA | **25** (17 of them in the train split) |

So "unseen lineage" is false for 14 of 36, and "the model never saw PHA activity" is false
outright.

**The result split on that contamination** (probe scores from
`polyester_scores_650M.npz`, threshold 0.1455):

| subset | n | detected | median score |
|---|---|---|---|
| all held-out PHA — **the published 83%** | 36 | 30 (83%) | 0.645 |
| shares a component with a trained positive | 14 | **14 (100%)** | 0.768 |
| **clean — no component overlap** | 22 | **16 (73%)** | **0.388** |

**All six misses fall in the clean subset**, and the contaminated proteins score roughly double.
The corrected generalisation figure is **16/22 = 73%**.

**BLASTp cannot be corrected.** `polyester_blast.json` stores only the aggregate
`pha: {probe: [30,36], blast: [28,36]}` — no per-protein results — so the 28/36 cannot be
decomposed without re-running BLAST. Until then the corrected probe figure must not be put
beside it; comparing 73% against an uncorrected 78% would be worse than the original error.

**Should the 36 be moved into the positives instead?** Quantified, the trade is:

| | positives | components | largest component |
|---|---|---|---|
| now | 388 | 74 | 66% |
| + all 36 | 424 | **87 (+13 new families)** | 60% |
| + only the 14 overlapping | 402 | 74 (**+0 new families**) | 62% |

Adding all 36 buys 13 new positive families and +18% effective sample size, and changes no
conclusion: §22b showed the fold-matched comparison is underpowered at 74 positive components
against 32 negative ones, and 87 against 32 is still underpowered. The cost is the project's
only evidence against its most likely criticism — that the probe detects cutinase-shaped
sequences rather than polyester-degrading activity, which **test performance cannot refute**
because test positives come from the same families as train positives. Recommendation: keep
the holdout.

**But redefine it as the 22 clean proteins.** The 14 overlapping ones add zero new families if
moved to training and inflate the figure if left in the holdout — the one group with no good
home. Either move them to the positives or footnote them, but stop counting them in the
headline.

**The ceiling is the field's, not the dataset's.** [R4] records fewer than 50 experimentally
verified plastics-active enzymes at publication. There is no reservoir of additional positive
diversity to draw on, which is a limitation to state rather than engineer around.

Corrected in `methods_detection.html`; §14's subsection now carries a supersession note.

---

### 26. PHA generalisation retired; the exclusion rationale stated properly (2026-10-11)

**Decision (user's, 2026-10-11): the PHA generalisation claim is withdrawn entirely.** §25
showed the premise was false for 14 of 36 and that 25 trained positives already act on a PHA,
so there was never an unseen lineage to generalise to. Rather than report a corrected 73%, the
claim is dropped. The 36 PHA depolymerase rows stay in `dataset_splits_id40.tsv` with
`split=heldout`, unused and carrying no claim. **No metric changes** — they were never in
train, val or test, so every published number stands exactly as it was.

The task is now stated in three parts and nothing else:

| | definition |
|---|---|
| **positive** | an enzyme experimentally reported to degrade a plastic whose backbone is joined by carboxylic ester bonds — 388 |
| **hard negative** | an α/β-hydrolase-fold protein with the same Ser–His–Asp triad, never reported to degrade a plastic — 51 |
| **naive control** | unrelated proteins, plus the two intermediate tiers — 372 |

PHA remains a *substrate* in the positive table (PHA 23, PHB 11, PHBV 5, and the rest), because
positives that act on one are ordinary positives. What is gone is the notion of PHA as a
withheld class.

**The exclusion rationale, corrected.** The user proposed stating that the other plastics are
excluded because they lack the ester bond and are harder to break. The first half is right; the
second half would be a factual error in print, because the source data contains **65
polyamide-degrading and 15 polyurethane-degrading enzymes** and **92 of the 118 excluded
entries have experimentally verified activity**. Nylonases and polyurethanases are real. The
exclusions therefore fall into two tiers that must be stated separately:

| plastic | backbone bond | why it is out |
|---|---|---|
| polyamide (nylon) | amide `–C(=O)–NH–` | hydrolysable, but by an **amidase** — a different catalytic machine |
| polyurethane | urethane `–NH–C(=O)–O–` | hydrolysable, different machine |
| polycarbonate | carbonate `–O–C(=O)–O–` | an ester, but not a carboxylic one; few enzymes |
| polyether (PEG) | ether `–C–O–C–` | very stable, needs oxidative chemistry |
| polyolefins/vinyls (PE, PP, PS, PVC) | C–C only | **no hydrolysable bond exists** — oxidative attack only |
| rubber | C–C / C=C | same |

Only the last two rows are excluded on chemistry alone. The rest are excluded because a
different bond needs a different enzyme, so pooling them with PETases would train one
classifier on unrelated families and call the mixture a plastic degrader.

**Why the ester is the bond that works** — now stated in the artifact rather than assumed. An
ester's carbonyl carbon is electrophilic with a good leaving group, so a serine nucleophile
attacks at a modest barrier. In an amide the nitrogen lone pair delocalises into the carbonyl,
giving the C–N bond partial double-bond character and raising the uncatalysed barrier by orders
of magnitude. In a C–C backbone there is no polarised bond at all, so hydrolysis is not
chemically available. Plus the evolutionary point: **cutin is a natural polyester**, so
cutinases existed before PET did and PET's ester is close enough that they partly work on it;
nothing in nature resembled polyethylene. (Supported by [R1] for the backbone split and [R3]
for "progress is limited to polyesters".)

**Artifacts.** `eda_dataset.html` v2 — PHA holdout language removed, held-out row dropped from
the splits table, new section "What is not in the positive set, and why" with both tiers and
the chemistry. `methods_detection.html` v5 — the "Generalisation to an unseen lineage" section
deleted, the constraint row relabelled. §14 and §25 keep their supersession notes so the
history stays readable.

**Note:** the "BioPET-SAE Dataset v1" artifact (`TrmNy5H5KtFTNRgkKcLS8y`) no longer resolves
for this account and is gone from the gallery. Its source survives as
`artifacts/dataset_card.html`, `DATASET_CARD.md` and `data/release/`; it has not been
republished.

---

### 27. Hand-off package for an external replication (2026-10-11)

Built so a second team can reproduce the experiment in a fresh repository with no access to
this one. Generated by `src/biopet_sae/build_handoff.py`; `handoff/` is 756 KB total.

| file | lines / size | purpose |
|---|---|---|
| `handoff/README.md` | 168 lines | the science — positive and negative definitions, the 21 polyester substrates with backbone bonds, what is excluded and why, splits, provenance, references |
| `handoff/AGENTS.md` | 104 lines | the contract for an AI agent — six hard rules, what to reproduce first, mistakes already made here |
| `handoff/EVALUATION.md` | 3.6 KB | five reporting rules, baselines, and what a result can and cannot support |
| `handoff/dataset.tsv` | 397 KB | 847 rows × 18 columns, sequences included |
| `handoff/dataset.fasta` | 341 KB | same sequences; header carries split, label, group, component |
| `handoff/schema.json` | 2.1 KB | per-column definitions, written for a loader to read |
| `handoff/manifest.json` | 2.6 KB | counts, split constraints, sha256, and the baselines to reproduce |

**Two changes from the internal release.** `split=heldout` is rewritten to `split=unused`, so
nothing in the package implies a generalisation claim retired in §26; and
`tRNA_processing_no_ser_nucleophile` is shortened to `trna_processing`. The internal files are
unchanged.

**The six hard rules in `AGENTS.md`**, which are the real deliverable: never fit or select on
test; bootstrap over `component` and never over proteins; report test decomposed by
`negative_group` and state n=10 for the hard-negative cell; use `negative_group` not
`class_label`; `unused` rows carry no claim; declare any tuning done on validation, which holds
one hard negative.

It also carries a "mistakes already made here" section so the second team does not repeat
them — an EC code is not a fold (§21), a name is not a fold either (the 16-word blocklist that
admitted a Proteinase K-like subtilisin, §26 context), absence of recorded activity is not
evidence of inactivity, and a large effect that misses a cutoff has not collapsed (§22b).

**Verification.** 22 assertions, all passing: row and column counts against `schema.json`
order; sha256 of both data files against the manifest; split, negative-group and
hard-negative-by-split counts against the manifest; `label` in {0,1}; `negative_group`
populated exactly when `label==0`; no stale `heldout` value anywhere; FASTA matched
record-by-record against the TSV (847 records, 0 mismatches); 74 positive components with
`BPS0190` at 256; every headline figure present in the prose; and the README's substrate
counts recomputed from the data.

---

### 28. Fold audit of the positives, BPS0016 reclassified, BPS0383, the three nulls, artifact bookkeeping (2026-10-12)

Written up on 2026-11-14 from the 2026-10-12 session. Everything below was re-derived from
`build_dataset.py`, `plastizymes_merged.tsv`, the splits file and live InterPro/UniProt lookups,
not recalled.

#### 28a. The fold of the positives was never verified

Prompted by "how do you verify the esterase fold?" The honest answer is that we do not.

**The code.** `build_dataset.py` classifies in this order: manual exclusion, then
`is_pet` → `1_pet` (returns immediately), then PHA group → `heldout_pha`, then the
`CLASS2_GROUPS` branch, which is the *only* place the fold check runs, then everything else →
"non-ester mechanism". The check, `is_esterase_fold_candidate`, is
`not any(label.lower() in NON_ESTERASE_ENZYMES ...)` — a **16-string blocklist** (protease,
serine protease, chitinase, peg dehydrogenase, laccase, peroxidase, manganese peroxidase, lignin
peroxidase, alkane hydroxylase, monooxygenase, dehydrogenase, oxidase, amidase, nitrilase,
urease) matched against the PlasticDB `Enzyme` field.

Three weaknesses, in increasing severity:

1. It is a deny-list: anything not on those 16 strings passes. Nothing is ever positively
   confirmed as an alpha/beta hydrolase.
2. **161 of 388 positives (41.5%) carry no enzyme label** (141 of 320 `1_pet`, 20 of 68
   `2_other_polyester`), so the list cannot fire on them. All 388 have
   `esterase_fold_candidate = 1`.
3. **It was only ever applied to 68 of 388 proteins.** All 320 `1_pet` returned before reaching
   it.

**Independent annotation.** Only 152 of 388 positives (39%) have a UniProt accession; fold is
checkable for those only.

| result | n | detail |
|---|---|---|
| carry `IPR029058` (alpha/beta hydrolase fold) | 111 | 73% of the 152 |
| no InterPro annotation at all | 38 | unreviewed TrEMBL |
| **carry InterPro showing a different fold** | **3** | below |

| id | accession | name | InterPro | actual fold |
|---|---|---|---|---|
| `BPS0013` | W0M2H4 | Esterase\|MGS0010 | IPR001466, IPR012338 | beta-lactamase-related |
| `BPS0394` | A0A4P1LYH5 | E4 | IPR013830, IPR036514 | SGNH hydrolase |
| `BPS0016` | A0A0U5AS91 | PAM | IPR034193, IPR000209, IPR023827 | **Proteinase K-like subtilisin (peptidase S8)** |

(Accession names resolved against the InterPro API: `IPR034193` = Proteinase K-like catalytic
domain, `IPR023827` = peptidase S8 subtilisin Asp active site, `IPR013830` = SGNH hydrolase-type
esterase domain, `IPR029058` = alpha/beta hydrolase fold.)

These are the *detectable* cases among 152, not a bound on the other 236. `BPS0016` is the
sharpest: a Proteinase K-like subtilisin degrading PLA sits in `2_other_polyester`, while
`BPS0018` — actual Proteinase K — was excluded as "not an esterase fold: Protease". Same family,
opposite fate, because one label reads "PAM" and did not match the string "protease". Also
suspect: `BPS0060`, enzyme label "Oxidoreductase", substrates PBAT|PU, activity unverified; it
passed only because that string is not on the list.

**Correction to this session's earlier statements.** `BPS0016` was repeatedly described as "a
PAM polyamidase". It is not: its substrate is PLA (a polyester) and its fold is subtilisin.
That wording is now fixed in §16, the atlas and the hand-off docs.

**Resolution (user's decision).** The positive definition drops any fold clause: *an enzyme
experimentally reported to degrade a plastic whose backbone is joined by carboxylic ester
bonds*. Fold becomes something reported, not something enforced. **Open:** the eight enzymes
excluded as "not an esterase fold" (six PLA-degrading proteases, two chitinases) genuinely
degrade polyesters and would qualify under that definition; they remain excluded and the
choice has not been made explicitly. InterProScan over all 388 sequences — which needs no
accession — would settle the fold question and has not been run.

#### 28b. BPS0016 is a correct rejection, not a localisation failure

§16 logged `BPS0016` as one of the two proteins where all three triad latents are silent (peak
activations 0.070, 0.054, 0.007) and treated it as a limitation. With the fold identified it is
the opposite: a subtilisin has no alpha/beta charge-relay geometry, so there is nothing for
those latents to detect and silence is the correct behaviour. It is the only non-alpha/beta
protein among the 59 (`BPS0013` and `BPS0394` are not in the localisation set). **On the 58
proteins that actually have the fold the protein-level rate is 57/58.** The remaining failure is
`Q9Y7G8` (cutinase pbc1), His only. Edited in §16, in `petase_structures.html` and in the
published atlas (v20).

#### 28c. BPS0383 — keep it, and say what the polyamide tag is

§24b left `BPS0383` (mechanism `amide|carboxylic_ester`, substrates `PA|PET`, verified) as a
decision. Resolved as far as the evidence goes:

- It is ***Fusarium solani* cutinase (FsC), UniProt P00590, PDB 1AGY** — the canonical
  cutinase, split `val`, component `BPS0054` (13 members).
- Its two sources: doi `10.1021/ma9005318`, "Cutinase-Catalyzed Hydrolysis of Poly(ethylene
  terephthalate)", *Macromolecules* 2009 (the PET record), and doi `10.1002/pola.20684`,
  "Cutinase — A new tool for biomodification of synthetic fibers", *J Polym Sci A* 2005 (the PA
  record). The polyamide tag therefore comes from a **fibre surface-modification** study, not
  from polyamide depolymerisation.
- Probe score 0.5657, well above the 0.1455 threshold, so removing it deletes a correct call.

| effect of removal | before | after |
|---|---|---|
| modelled positives | 388 | 387 |
| `1_pet` | 320 | 319 |
| val | 43 / 76, prevalence 0.361 | 42 / 76, prevalence 0.356 |
| val AU-ROC / AU-PRC | 0.9976 / 0.9954 | 0.9975 / 0.9952 |
| test | unchanged | unchanged |
| component `BPS0054` | 13 | 12 |

The one real cost is explainability: it is `view1[8]` of the 14 train+val proteins on which the
neighbourhood offsets were learned (Ser +1 13/14, Asp +3 9/14, His +1 12/14), so removing it
changes those denominators. **Recommendation: keep it and document the PA tag** (done in
`handoff/AGENTS.md`). The user has not ruled otherwise.

#### 28d. An abstract sentence audited against the run

A draft sentence read: *"…stable across 100 bootstrap resamples, localize to experimentally
verified catalytic residues in 172 of 176 plastizymes (97.7%), against a ~3.2% background rate
when all remaining latents are run through the identical pipeline."* Three figures were wrong,
two of them reversions to numbers already corrected:

| in the sentence | actual | note |
|---|---|---|
| 172 of 176 plastizymes (97.7%) | **57 of 59 proteins (96.6%)** | corrected 2026-09-29 at the user's direction; 176 is a count of residue outcomes, and the three latents agree pairwise at 1.0 |
| 100 bootstrap resamples | **200** | random-half draws |
| ~3.2% background | **4.4% median** (for the all-latents null) | 3.2% is a different null |

**There are three nulls and they answer different questions.**

| null | asks | result |
|---|---|---|
| random position (§16) | is the ±3 window doing the work? | **3.2%** (41/1,280); ±5 adds nothing |
| **random latent (§17)** | are these latents special among all 10,240? | **4.4% median**, mean ≈ 10–13%, 95th percentile 57–60%; **0 of 10,240 score higher** |
| shuffled position (§17) | — | 12.0 / 7.5 / 6.5% |

The sentence's clause "when all remaining latents are run through the identical pipeline"
describes the random-latent null, so the figure is 4.4%, not 3.2%.

**"None scores higher" needs its ties stated.** 9529 scores 100%, which cannot be beaten, and
**25 other latents tie it** on the Ser task (7 on Asp, 8 on His). What makes 9529 special is
that it was chosen from model behaviour alone before any catalytic residue was examined, and
none of the 25 ties passes that selection (best stability 0.695 against the 0.75 cutoff). State
the ties; do not leave a reviewer to find them.

**Like-for-like comparison.** The random-latent null was run on the **45 held-out** proteins
(offsets learned on the other 14). The 57/59 headline includes those 14. The matching figure is
the held-out one — 44/45 = 97.8% protein-level (test 2/2 + external 42/43), with the per-latent
rates 100 / 100 / 97.8% that appear in the null table. Quote the 45-protein figure next to the
null, or the 59-protein figure with the denominators stated.

**"Stable across 200 resamples" is literally true and means less than it sounds.** Stability is
the fraction of 200 random-half draws in which a latent stays in the top 10 of 10,240 by Cliff's
δ, with criterion ≥150/200 and δ ≥ 0.9 (9529 200, 5201 200, 7942 189, 5255 176, 2661 174, 7734
159, 5271 157). It is stability against a negative set that is 88% not fold-matched; against the
51 fold-matched negatives all seven fall to 0/200 (§22).

Suggested wording:

> We then interpret the probe using pretrained InterPLM sparse autoencoders: latent features
> selected from model behaviour alone, with no catalytic-residue supervision and stable across
> 200 bootstrap resamples, localize to experimentally verified catalytic residues in 57 of 59
> proteins (96.6%), against a 4.4% median rate when all 10,240 latents are run through the
> identical pipeline, where no latent scores higher and the 25 that tie on the serine task do
> not pass selection.

#### 28e. Artifact and repository bookkeeping

**Published artifacts** (all private — each needs sharing from its Share menu before anyone else
can open it):

| artifact | URL | version | local source |
|---|---|---|---|
| Catalytic Latent Atlas for Plastizymes | `claude.ai/artifact/KcdU4xp1QUSvuWzrL4pGwS` | **v20** | **`artifacts/petase_structures.html`** |
| Plastizyme Detection Methods | `claude.ai/artifact/US29Zu7ScaoAKSoepRQeMg` | **v6** | `artifacts/methods_detection.html` |
| Polyester Degrader Dataset | `claude.ai/artifact/299aZWX8LjJM5E7bTQ4k7r` | **v3** | `artifacts/eda_dataset.html` |
| Polyester Classifier Curves | `claude.ai/artifact/S1rzDp1cJYxA39xSTPZteE` | v1 | `artifacts/pr_curves.html` |
| Metagenomic Plastizyme Screen | `claude.ai/artifact/9UVa672uQUXExrESd2xvtv` | v1 | `artifacts/screen30k.html` or `screen30k_page.html` — **not verified, diff against the live page before republishing** |
| BioPET-SAE Dataset v1 | `claude.ai/artifact/TrmNy5H5KtFTNRgkKcLS8y` | — | **deleted from the account** (not by this session); source `artifacts/dataset_card.html` is unpublished |

**The atlas trap.** The live atlas is built from `petase_structures.html` (2.3 MB, embedded PDB
data), **not** `atlas_page.html` (42 KB), which was a different page that never backed it.
After a context compaction this session lost the path-to-artifact mapping, published
`atlas_page.html` without a `url`, and created a stray duplicate
(`57sw1mMGXwmsJc47TMxRLz`, deleted at the user's instruction). The correct identification came
from downloading the live page and diffing it against both candidates: it differed from
`petase_structures.html` only by the platform's publish skeleton. **Rule: before publishing to
an artifact from an earlier context, read it and diff against the candidate source; never
publish without `url` to "update" something.**

**Repository.** Two commits were made by the user after the last logged state — `b2356d2`
"adding metagenome predictions and cleaner dataset card" and `8d35185` "fixing atlas" (which
deleted `artifacts/atlas_page.html` and carried the BPS0016 fix). Both are pushed
(`main...origin/main`, nothing ahead). **`b2356d2` committed `data/mgnify/raw/` — six `.faa.gz`
files, 277 MB — plus `data/mgnify/sample_cds.faa` (99.9 MB), so `.git` is now 455 MB and that
data is on GitHub.** `.gitignore` has no `mgnify` entry. Fixing it going forward is
non-destructive:

```
git rm -r --cached data/mgnify/raw data/mgnify/sample_cds.faa
printf 'data/mgnify/raw/\ndata/mgnify/sample_cds.faa\n' >> .gitignore
```

Removing it from *history* needs `git filter-repo` or BFG plus a force-push, which rewrites the
public history and is the user's decision. Not done.

**Process.** The shell permission layer denied every form of `rm` in this session, so file
deletion had to be run by the user (`! rm …`) or arrive via their commit. Per CLAUDE.md this
session never ran `git commit` or `git push`.

#### 28f. Hand-off package corrected

Two corrections to `handoff/` (§27), neither touching the data files — `dataset.tsv`,
`dataset.fasta` and `manifest.json` are byte-identical to the committed copies:

- `schema.json` described `ab_hydrolase_fold` as "blank where no UniProt accession exists".
  It is in fact populated for **all 423 negatives and blank for every positive and unused
  row**, because the annotation pass was only ever run on negatives. Description fixed in
  `build_handoff.py`.
- `handoff/AGENTS.md` gained a **"Known data issues in the positives"** section: the unverified
  fold, `BPS0016` / `BPS0013` / `BPS0394`, the suspect `BPS0060`, the `BPS0383` polyamide tag,
  the 23 positives carrying PU/PUR/PC/PA tags (19 + 3 PC-only + 1 PA), and the fact that the 36
  `unused` rows are not a held-out test. A first draft of that section said "25"; the union is
  **23**, and the 25 elsewhere in this log is a different count (any non-carboxylic-ester
  mechanism tag, including two "unknown").

`DATASET_CARD.md` and `artifacts/dataset_card.html` had still described PHA as "an unseen-substrate
generalisation test"; both now say it is unused and carries no claim. `artifacts/SAE_Feature_Interpretation.md`
now opens with a pointer to §15–§18 and §22 and a statement that it documents only the superseded
PET-only run. The root `README.md`, previously a two-line stub, now describes the project and its
caveats.

#### 28g. A mislabelled cell and an invalid interval, both inside the hand-off package

Found on 2026-11-14 while re-deriving the numbers for the status section. Both were my errors.

**The cell.** §21 correctly reported AU-ROC 0.9141 for **"EC 3.1.1 non-tRNA", n = 23**. That
group is the 13 `ec311_wrong_fold` plus the 10 `ab_hydrolase_hard_negative` negatives in test.
It was then transcribed into the hand-off package and the dataset card as the figure for
`ec311_wrong_fold` alone, with n = 23, which is wrong on both counts. Recomputed from
`polyester_scores_650M.npz` with the 42 test positives held fixed:

| test negatives | n | AU-ROC | AU-PRC |
|---|---|---|---|
| all | 62 | 0.9681 | 0.9526 |
| `naive_control` | 36 | 1.0000 | 1.0000 |
| `trna_processing` | 3 | 1.0000 | 1.0000 |
| **`ec311_wrong_fold` alone** | **13** | **0.9890** | **0.9967** |
| `ab_hydrolase_hard_negative` | 10 | 0.8167 | 0.9550 |
| wrong-fold + alpha/beta combined (the old "23") | 23 | 0.9141 | 0.9526 |

So **the probe separates the wrong-fold group almost as easily as the naive controls**, and the
alpha/beta hard negatives are the only cell that is actually hard. That contradicts the
"intermediate difficulty" wording I had written into the card; the accurate statement is that
the probe finds them easy (0.989) while the SAE serine latent fires on only 13.9% of them, which
is what makes them a control for "serine versus fold" rather than a hard negative.

**The interval.** The 95% CI of [0.676, 0.926] on the 0.8167 was a *per-protein* bootstrap —
the form this project's own rule rejects. Resampling homology components instead (5,000 draws;
20 positive and 7 negative components) gives **[0.606, 0.963]**, wider on both sides. For the
wrong-fold cell the component interval is [0.951, 1.000] (20 and 8 components). The per-protein
interval is kept in the manifest under `ci95_protein_bootstrap_not_valid` so the difference is
visible rather than silently replaced. §21's intervals for the 23-protein and 10-protein cells
([0.840, 0.971] and [0.676, 0.926]) are per-protein and stand only as logged.

**Corrected in:** `handoff/EVALUATION.md`, `handoff/README.md`, `handoff/manifest.json` (via
`build_handoff.py`), `DATASET_CARD.md`, `artifacts/dataset_card.html`,
`artifacts/EDA_polyester_task.md`, `artifacts/eda_dataset.html` (published v3) and
`artifacts/methods_detection.html` (published v6). `dataset.tsv` and `dataset.fasta` are
byte-identical to the committed copies. The 39-of-62 "separated perfectly" figure (36 naive + 3
tRNA) is unaffected.

---

### 28h. Handoff drops the 36 PHA rows (2026-11-23)

User decision: unlabelled/unused rows should not ship. `build_handoff.py` now skips `split=heldout`, so `handoff/dataset.tsv` and `.fasta` have **811 rows** (388 pos / 423 neg; train 588, val 119, test 104), every row modelled; `manifest.json` records `removed_pha_not_shipped: 36`. README/AGENTS/schema updated; sha256 in the manifest refreshed. *Superseded the same day:* the release file was trimmed too (user: "trim everywhere"), see below; the handoff is now built from v1.1 (§28i, 826 rows). Not committed.

**Trimmed everywhere (same day).** `build_release.py` now filters out `heldout` rows; `data/release/biopet_sae_dataset_v1.{tsv,fasta}` have **811 rows** (verified = old file minus the 36 PHA rows, otherwise identical); handoff regenerated with the same sha256. `DATASET_CARD.md`, `artifacts/dataset_card.html`, `artifacts/EDA_polyester_task.md`, `artifacts/methods_detection.html` updated. Left alone on purpose: `data/processed/*` and `results/` (pipeline intermediates, 847 rows with class `heldout_pha`) and older PROGRESS sections (history). **Published artifacts not republished** (methods v6 still shows the old wording; dataset_card.html was already unpublished).

### 28i. PAZy update: 16 PET enzymes added as release v1.1 (2026-11-23)

Live PAZy lists 505 proteins; our 2026-09-27 export had 462 (diffed by PAZy id across all 11 pages; none missing from the export, 43 new = 16 PET, 19 PUR, 8 PA). Added the 16 PET enzymes (all `verified`, PAZy ids 523-538, DOIs recorded) as `biopet_sae_dataset_v1_1.{tsv,fasta}` via `src/biopet_sae/add_pazy_update.py`; **v1 left frozen** because all results hang on its 104-row test. Existing assignments untouched (checked: 0 changed v1 rows). Placement: component membership at >= 0.40 global identity (make_splits routine) to v1 sequences; unlinked go to train. Result: 7 train + 9 test (HaloPETase1-4, R75estA, BhEstB, LipSh1-3 share components with test), 0 val; 3 new singleton components; largest component 259/404 = 64%; max cross-split identity of additions 0.3967. Dropped negative `P37957` (hard negative, 94% identical to LipBv, D2 rule): 51 -> 50 hard negatives (39/1/10). v1.1: 826 rows, 404 pos / 422 neg. Test is now 51 pos / 62 neg; **nine new test positives unscored** (needs ESM-2 embedding). Handoff regenerated from v1.1 with `batch` column and baseline notes ("reproduce on batch == v1"); README/AGENTS/EVALUATION updated; DATASET_CARD has a v1.1 section, its tables stay v1. Not updated: published artifacts, EDA md (v1 numbers). Not committed. Open: UniProt/InterPro check of the 16; embed and score the nine; note baseline train set is 587 v1 rows (P37957 dropped), not 588.

### 28j. v1 results archived; UniProt check of the 16 additions (2026-11-24)

**Archive.** `archive/v1/` (122 MB, `MANIFEST.sha256`, `README.md`) is a byte-for-byte snapshot taken before re-running on v1.1: release files, `data/processed/`, `data/embeddings_650M/`, `data/sae_650M_L33/`, `results/`, the handoff as of today, and PROGRESS/DATASET_CARD. Untracked; commit or tag it yourself (`git tag v1-results` before the re-run is the lightest protection). `results/` is gitignored, so the archive is the only copy of those files besides local disk.

**UniProt check** (`src/biopet_sae/check_pazy_update_uniprot.py`, output `data/raw/pazy_update/uniprot_check.json`; only public MD5 checksums and accessions were sent). 15 of 16 sequences exist in UniParc with an exact MD5 match; **`bbPET0069` does not** (unidentified soil bacterium, 2025 paper; absent from UniProt). Only 4 have a live UniProtKB entry (LactoPETase, LipSh2, SM0281, R75estA): all four have identical sequence, matching organism, and **IPR029058 (alpha/beta hydrolase fold)**. The other 11 map only to DELETED UniProtKB entries, so UniProt gives no fold annotation; InterProScan (not run) is needed. No curated ACT_SITE on any of the 16, so none enters the 59-protein localisation set. Names: LactoPETase "Alpha/beta hydrolase" and "Cinnamoyl esterase"; SM0281 "Palmitoyl-protein thioesterase ABHD10"; LipSh2 "triacylglycerol lipase".

**Reproducibility gap found while planning the re-run.** No script in `src/` writes `polyester_results.json`, `task_polyester.json`, `selection_full.npz`, `null_random_latents.npz` or `polyester_scores_650M.npz`; PROGRESS cites `scratchpad/null_features.py` and `nested_demo.py`, which are gone. The headline numbers came from ad-hoc scripts that were not committed. Re-running requires rewriting them as committed scripts and first reproducing v1 numbers (0.9681 / 0.9526 / 0.8167 / 57 of 59 / 4.4%) on the frozen v1 data.

### 28k. Re-run, step 1a: the probe is now a committed script and reproduces v1 exactly (2026-11-24)

`src/biopet_sae/polyester_probe.py` (output `results/v1_repro/probe/`). Reverse-engineered from the stored v1 scores: **seed 42 reproduces `polyester_scores_650M.npz` with max |diff| = 0.0** (seeds 0-4 differ by up to 0.008, so the seed matters). Protocol: max-pooled layer 33; StandardScaler on train; SGD log-loss L2, alpha = 1/(C n_train), C=1e-2, eta0=1e-4, 5000 epochs, tol=None, class_weight balanced; **threshold = best-F1 on val**. Regression on frozen v1 (811 rows): threshold 0.14548159; val AU-ROC 0.9976 / AU-PRC 0.9954; **test AU-ROC 0.9681, AU-PRC 0.9526**; TP/FP/FN/TN 39/8/3/54; recall at precision >= 95/90/85/80% = 59.5/78.6/92.9/97.6%; by group naive 1.0000 (n=36), tRNA 1.0000 (3), wrong-fold 0.9890 (13), **alpha/beta 0.8167 (10)**. All match the logged values. Intervals: the v1 component intervals resample positive and negative components separately (20 + 7 for the hard cell); the script now does the same and gives [0.613, 0.962] vs logged [0.606, 0.963] (resampling noise); wrong-fold [0.950, 1.000]; the blended interval [0.917, 0.997] differs by 0.004 from the logged [0.921, 0.997] (the logged blended interval's exact procedure is not recorded). On v1.1 the script also reports the original 104 test rows and the 9 added positives separately.

Still to write as committed scripts, each to be regression-tested on v1 before running on v1.1: SAE feature extraction for the 16 new sequences; latent selection (Cliff's delta, top-10 stability over 200 half-draws; stored in `selection_full.npz`); the 10,240-latent null (`null_random_latents.npz`); localisation to curated sites (57 of 59); BLASTp baseline decomposed by negative group; screen re-score. Embeddings for the 16 new sequences are not yet computed (`polyester_probe.py` stops with a clear message if rows lack embeddings).

### 28l. Re-run, step 1b: probe, hyperparameter sweep and BLASTp on v1.1 (2026-11-24)

**New committed scripts** (`src/biopet_sae/`): `polyester_probe.py`, `polyester_sweep.py`, `polyester_blast.py`, plus `embed_esm2` run on the 16 new sequences (`data/embeddings_650M_new16/`, merged into `data/embeddings_650M_v1_1/embeddings.npz`, 863 rows; the original embeddings are untouched). **Embedding check:** four old sequences re-embedded on MPS match the stored embeddings with max |diff| = 0.0 on three checked layers, so the new embeddings are consistent with the old ones. **Regression on frozen v1** (all pass): probe (0.9681 / 0.9526, TP/FP/FN/TN 39/8/3/54, threshold 0.14548159, by-group 0.8167 / 0.9890 / 1.0 / 1.0, recall at 90% precision 78.6%); sweep (14 of 16 stored cells match to 4 decimals; the two that differ are in the unstable eta0=0.1 column; the stored grid also had C=1, now included); BLASTp (**score is -log10 of the best e-value, not the bitscore**; test AU-ROC 0.9040, AU-PRC 0.7552, e* 3.16e-2, TP/FP/FN/TN 40/8/2/54, recall at 90% precision 2.4%).

**Sweep (C x eta0, 5 x 4, val log-loss, seed 42).** v1.1 selects **C=1e-1, eta0=1e-3** (val log-loss 0.2144), where v1 selected C=1e-2, eta0=1e-4. The selection is not decisive: five cells lie within 0.025 log-loss (0.2144 / 0.2167 / 0.2265 / 0.2314 / 0.2369), and cells with small C or large eta0 changed a lot from v1 to v1.1 (e.g. C=1e-4, eta0=1e-2 val AU-ROC 0.88 -> 0.52), which says SGD at 5000 epochs on ~600 rows is sensitive to the data order. Full grid in `results/v1_1/sweep/sweep.json`.

**v1.1 test (51 pos / 62 neg; threshold = val best-F1; component bootstrap, positive and negative components resampled separately):**

| | probe, sweep-selected (C=1e-1, eta0=1e-3) | probe, frozen v1 (C=1e-2, eta0=1e-4) | BLASTp (db 310, e* 3.16e-3) | v1 probe (test 104) |
|---|---|---|---|---|
| blended AU-ROC | 0.9643 [0.905, 0.998] | 0.9763 [0.929, 1.000] | 0.9102 [0.813, 0.990] | 0.9681 |
| blended AU-PRC | 0.9544 [0.841, 0.998] | 0.9712 [0.887, 1.000] | 0.7964 [0.615, 0.988] | 0.9526 |
| recall at 90% precision | 92.2% | 94.1% | 2.0% | 78.6% (BLASTp 2.4%) |
| alpha/beta hard negatives (n=10) AU-ROC | 0.8216 [0.590, 0.976] | 0.8725 [0.688, 0.993] | **0.4627** | 0.8167 (BLASTp 0.4286) |
| wrong-fold (n=13) | 0.9744 | 0.9864 | 0.9940 | 0.9890 |
| naive (n=36) / tRNA (n=3) | 0.9973 / 1.0 | 0.9995 / 1.0 | 0.9967 / 1.0 | 1.0 / 1.0 |

**Rule for which probe is primary:** the sweep-selected cell (the v1 selection rule applied mechanically), with the frozen setting shown for sensitivity. Neither was chosen on test, but test has now been read twice per probe configuration; do not select on these numbers. On the 104 original test rows the retrained probes score 0.9627 / 0.9443 (selected) and 0.9747 / 0.9641 (frozen), against 0.9681 / 0.9526 for v1: **retraining on the +7 positives / -1 hard negative alone moves AU-ROC by about +-0.005 and the hyperparameter cell by about +-0.01, both smaller than the hard-cell interval**, so none of these differences is meaningful. BLASTp on the 104 rows: 0.9063 / 0.7602.

**The nine added test positives** (never scored before): the probe calls all 9 at its threshold with either setting; scores (selected) HaloPETase1-4 0.993-0.997, R75estA 0.977, BhEstB 0.686, LipSh1/2/3 **0.136 / 0.196 / 0.205** (threshold 0.1115): the three LipSh enzymes (766/755/755 aa, Staphylococcus hominis, ~0.40 identity to test relatives) barely pass. BLASTp also recalls all 9 at e*. Not independent evidence: all nine share components with old test proteins.

**Finding, closes Still-open #4 for the headline cell:** BLASTp is **below chance on the alpha/beta hard negatives (AU-ROC 0.43 on v1, 0.46 on v1.1, n=10)**: a hard negative's nearest training positive is, by fold, about as similar as a true positive's. The probe is at 0.82-0.87 there, so the probe-over-BLASTp advantage is concentrated exactly where the negatives are fold-matched, and BLASTp matches or beats the probe on wrong-fold and naive negatives (0.994 / 0.997). Paired bootstrap of the difference by group is not yet run, and the cell has only 7 negative components.

Outputs: `results/v1_repro/{probe,blast,sweep}` (regression on v1) and `results/v1_1/{probe_sel,probe_frozen,blast,sweep}` (gitignored). Not done: SAE steps, screen re-score, paired probe-vs-BLASTp test, decision about which numbers go in the paper.

### 28m. `paper/v1.2.md` started; v1.2 plan recorded (2026-11-25)

Created `paper/v1.2.md`, a working file of paper-ready snippets and the record behind them: task definition and wording rules, positives (substrate table, provenance, verification, known data issues), negatives (four groups, what EC 3.1.1 means), splits for v1 and v1.1, v1/v1.1 probe and BLASTp results with how-to-read caveats, explainability facts, PAZy update and UniProt check, the v1.2 plan, methods facts, a list of errors caught, and the Introduction references (18, Crossref-resolved) with the claims still needing a source. Plan for v1.2 (not built): fetch 300-400 UniProt reviewed IPR029058 hard negatives (UniProt counts 2026-11-25: 3,156 IPR029058; 2,454 with an active site; 1,172 with EC 3.1.1), drop >= 90% identical to a positive, one new 60/20/20 component split, then probe/sweep/BLASTp with 5 seeds and the figure tables on test. No code run this turn. Not committed.

### 28n. Dataset v1.2 built (2026-11-25)

Built per the refined prompt (see `paper/v1.2.md` section 12 for the full record). v1.1 archived in `archive/v1_1/`. 404 positives and all other negatives unchanged; **+400 new hard negatives** (UniProt reviewed, IPR029058, EC 3.1.1.-, curated Ser/His/Asp active site, not plastic-linked, < 90% to any positive, 40-90% dropped, one per 90% group; funnel 1,172 -> 628 -> 403 -> 400); total 1,226 rows, 450 hard negatives. New 40%-identity component split: train 789 (317 pos, 225 hard), val 237 (44 pos, 112 hard), test 200 (43 pos, 113 hard); max cross-split identity 0.3982. Verified independently: 30/30 live UniProt re-checks pass; no id/sequence duplication; 0 label changes; mmseqs 0.40 clustering shows 2 spanning clusters whose global identities are 0.386-0.391. EDA tables in `data/release/eda_v1_2/`. New scripts: `fetch_hard_negatives_v12.py`, `select_hard_negatives_v12.py`, `build_v12.py`, `verify_v12.py`, `eda_v12.py`. Bugs fixed on the way: the phylum field labelled animals "Eukaryota" (a domain); the name filter matched "alpha" and "chloroplastic" (now whole-word, acronyms upper-case only); a first split put 34 of 68 non-PET positives in test (positives are now balanced by class).

**Main finding: the new hard negatives are 76% eukaryotic (342/450) while positives are 77% bacterial.** Domain alone gives AU-ROC 0.88 (val) / 0.80 (test) on positives vs hard negatives; length alone 0.75 / 0.68. Probe results on the hard cell must be reported against these and on the bacteria-only hard cell (108: 66/20/22). Only 62 of the 400 new hard negatives are bacterial. Not done: probe, sweep and BLASTp on v1.2; handoff rebuild; a taxonomy-matched pool (options: add bacterial TrEMBL entries with the same criteria, or down-weight eukaryotes); an InterProScan of the 74 positives lacking phylum. Not committed.

### 28o. Can embeddings tell bacteria from eukaryotes? Yes, almost perfectly (2026-11-25)

`src/biopet_sae/domain_probe_v12.py` (outputs `results/v1_2/domain_probe/`). Embeddings for the 400 new hard negatives computed on MPS (`data/embeddings_650M_new400/`, merged with the v1.1 embeddings into `data/embeddings_650M_v1_2/`, 1,263 rows). A logistic regression predicts Bacteria vs Eukaryota (1,148 rows with a known domain: 780 / 368) from the sequence embedding only, trained on the v1.2 train split and read on test (homology-disjoint, <= 40% identity). Test: **ESM-2 max L33 AU-ROC 0.9958 (balanced accuracy 0.984); mean L33 0.9985 (0.989)**; amino-acid composition 0.796; length only 0.622. Within the hard negatives only (91 eukaryotic of 113 in test): 0.993 / 0.998. **Transfer:** trained on everything that is not a hard negative (positives, wrong-fold, tRNA, naive; train+val) and read on the test hard negatives: AU-ROC 0.991 (max) / 0.995 (mean) (the balanced accuracy at the 0.5 cut is 0.53 / 0.88, so the ranking transfers more than the calibration), composition 0.742. Conclusion: domain of life is encoded in the embedding far beyond what composition or length carry, and it generalises across protein families and across groups, so a polyester probe on the full v1.2 hard cell can use it. Not yet shown: that the polyester probe actually does use it (needs the probe run on v1.2 and a comparison within the bacteria-only hard cell). Caveats: domain labels for old rows come from the phylum name; 74 positives with no phylum and 4 with an unclassified phylum are excluded from this test.

### 28p. Probe and BLASTp on v1.2 by cell, with taxonomy controls (2026-11-25)

`src/biopet_sae/polyester_cells_v12.py`, outputs `results/v1_2/cells/`. Sweep on the new val split selects **C=0.01, eta0=0.0001** (val AU-ROC 0.972 / AU-PRC 0.899, vs 0.998 / 0.995 on v1 when val had 1 hard negative); five seeds, SE < 0.0002 everywhere. Test (n+ 43), AU-ROC / AU-PRC (floor): all negatives 0.932 / 0.844 (0.21); all hard negatives 0.921 / 0.859 (0.28); **bacterial hard negatives (22) 0.795 [0.651, 0.910] / 0.903 (0.66)**; eukaryotic hard negatives (91) 0.952 / 0.926 (0.32), where domain-only gets 0.930 / 0.905; **matched cell, bacterial positives vs bacterial hard negatives (23 / 22) 0.757 [0.554, 0.925] / 0.826 (0.51)** against BLASTp 0.684 / 0.727, length-only 0.678 / 0.705, domain-only 0.500 / 0.511. BLASTp on all hard negatives 0.727 / 0.552; wrong-fold 0.914 / 0.989 (probe 0.870 / 0.984); naive 0.960 / 0.964 (probe 0.970 / 0.975). Validation shows the same ordering. Conclusion: the headline hard-cell number on v1.2 is inflated by the taxonomy shortcut; the controlled estimate is 0.76 AU-ROC / 0.83 AU-PRC with a wide interval that overlaps the baselines. Full table in `paper/v1.2.md` section 12.8. Not done: v1.1 vs v1.2 comparison is not valid (different test); SAE steps; figure tables; paired probe-vs-BLASTp test; handoff rebuild. Standing user rule from this turn: always report AU-PRC beside AU-ROC with the prevalence floor.

### 28q. Phase one complete: v1.2 probe, five runs, threshold and curve tables (2026-11-25)

`src/biopet_sae/polyester_phase1_v12.py` (outputs `results/v1_2/phase1/`, copied to tracked `paper/v1_2_phase1/` because `results/` is gitignored). Five seeds on train only; per-run per-threshold precision/recall/FPR/F1 on val and test, means and SE; threshold = best mean val F1 = **0.15**; val 0.816 / 0.809 / 0.813 (P/R/F1), **test 0.727 / 0.744 / 0.736**. AU-ROC / AU-PRC: **val 0.972 / 0.899 (floor 0.19), test 0.932 / 0.844 (floor 0.21)**; BLASTp test 0.791 / 0.545. Test recall at val-carried precision targets: 95% -> 0.349, 90% -> 0.419, 85% -> 0.488, 80% -> 0.744 (test precision 1.000 / 0.947 / 0.955 / 0.727). SE < 0.001 nearly everywhere. Full tables in `paper/v1.2.md` section 13. Phase one (build the probe) ends here; SAE/ablation, figures, handoff rebuild and the paired probe-vs-BLASTp test remain. Standing rule: AU-PRC is always reported next to AU-ROC with its floor. BLASTp tables added the same day (`blast_phase1_v12.py`, `paper/v1_2_phase1/blast_*`): test precision 0.449 / recall 0.721 / F1 0.554 at the val-chosen cut-off (e <= 3.2e-6); recall at 85-95% precision 0.186 (precision 0.727) vs the probe's 0.349-0.488. Probe-vs-BLASTp seed-SE table and the caveat sentence ("SE = training variation only; BLASTp is deterministic") added to `paper/v1.2.md` 13.7, with the paired component-bootstrap differences (all negatives: val AU-ROC +0.109 [+0.039, +0.217], test +0.141 [+0.038, +0.256]; not saved to a file, hard cells not yet tested). Probe architecture and training documented for the methods section in `paper/v1.2.md` section 14 (frozen ESM-2 650M, max-pool L33, standardise, linear + sigmoid, class-balanced L2 log-loss by constant-step SGD, C=0.01, eta0=1e-4, 5 runs, threshold 0.15 from mean val F1; 1,281 trained parameters). What was swept and how each value was chosen is recorded in `paper/v1.2.md` section 15 (20-cell C x eta0 grid by val log-loss -> C=0.01, eta0=1e-4; 103 probability thresholds by mean val F1 -> 0.15; BLASTp half-decade e-value grid by val F1 -> 3.2e-6). Paper record completed: `paper/v1.2.md` now has sections 12-17 (v1.2 build, phase-one numbers, probe architecture, sweep choices, interpretation notes, glossary) with a contents line; `paper/v1_2_phase1/` (17+ files, small) holds every table behind them with a README index (sweep grid, five-run scores, BLASTp tables, paired differences, matched-cell and domain-probe results). **Class weighting recorded as a key methods fact** in `paper/v1.2.md` 14.2 (balanced weights 1.24 / 0.84; no group or family weighting; probabilities uncalibrated, threshold 0.15; not swept). v1 SAE latent identification reconstructed and verified: `src/biopet_sae/sae_select_latents.py`; features = MAX-pooled over residues (`feat_max`, not mean); Cliff's delta over all 346 x 361 train+val pairs; 200 random-half draws WITHOUT replacement (seed 0) top-10 stability; delta >= 0.9 and >= 150 hits gives the seven (9529, 5201, 7942, 5255, 2661, 5271, 7734); exact regression vs the stored arrays (delta 6e-8, hits identical). Record in `paper/v1.2.md` section 18; outputs in `paper/v1_sae_selection/`. Not committed.

### 28r. SAE latents on v1.2 recorded (2026-11-26)

Full record `paper/v1.2.md` section 19; files `paper/v1_2_sae/`. 1,000-draw selection on v1.2 train+val: only 1817 and 5325 pass delta >= 0.9 and >= 75%; v1 seven mostly drop out because half the negatives are now hard (they fire on the shared triad). v1 triad latents on held-out hard negatives: 9529 Ser 99%, 2661 Asp 72%, 7734 His 98%. Top-10 latents: no plastic specificity; mostly family features; only 2919 lands in the IsPETase binding site above chance. **Corrected artefact:** the first localisation run recorded a silent latent's peak as residue 0, so its results for the new latents (and its null) are invalid; `sae_biology_v12.py` excludes silent proteins (Ser corrected null: 7 of 3,758 eligible latents above 9529). **Fixed bug:** loading `npz["feat_max"][i]` per row re-read the whole array and ran out of memory. Not done: ablation; corrected null for Asp/His; per-family selection over all latents; re-running v1's 57/59 from committed code. Not committed.

### 28s. Catalytic-triad annotation finalised (2026-11-26)

Record: `paper/v1.2.md` section 20; files `paper/v1_2_triad/`. Rule (project lead): max activation >= 0.3 **on exactly** the annotated residue; latent with most correct train+val proteins; 1,000 half-draws. **Ser 2925, Asp 157, His 3927** (each first in 1000/1000). Test (105 proteins, 3 degraders) precision / recall: Ser 99.0% / 99.0%; His 99.0% / 97.1%; Asp 61.0% / 47.6%; all three 50/105 (47.6%). **Correction:** test had been inspected in earlier analyses and was wrongly called unread. **Ground truth:** only 43 of 1327 evidence tags on hard-negative active sites are experimental. Asp alternative (3 latents with offsets) reaches 73% on val; not run on test. Not done: ablation, calibration, structure check for Asp, experimental-only subset evaluation. Not committed.

### 28t. Aspartate classifier and end-to-end pipeline (2026-11-26)

Aspartate: L1 logistic ranking over Asp residues on SAE activations, fit on train, C=0.1 chosen on val (val 103/105), 9 latents; test (once) precision 94.1% / recall 91.4% at score >= 0.5 (157 alone 47.6%); whole triad 96/105. Pipeline `plastizyme_pipeline.py` with saved models `models/pipeline_v1_2/` (five-run probe reproduced exactly, thresholds 0.15 / 0.44, triad latents 2925 / 3927 + aspartate classifier); `--check` reproduces recorded test results from raw sequences. Record: `paper/v1.2.md` 20.12 and 21. Caveats: held-out not blind; annotations mostly inferred; 3 test degraders; unrelated folds untested. Open: ablation, calibration, false-call rate on unrelated proteins, experimental-annotation subset. Not committed.

### 28u. Ablation of the triad latents, and the headline framing (2026-11-26)

`ablation_v12.py`, record `paper/v1.2.md` sections 22-23. Removing the 11 triad latents: degrader score -0.068 [-0.088, -0.047], recall at 0.15 74% -> 60%; AU-ROC / AU-PRC changes not distinguishable from zero. Recorded as a modest finding: partial use, not what separates degraders from look-alikes. Headline (lead): probe beats BLASTp + SAE triad annotation + end-to-end pipeline. Open before claiming novelty: literature search. Not committed.

### 28v. Split algorithm documented (2026-11-26)

Plain-language algorithm of the v1.2 split (family-level, 40% identity, 20,000 seeded restarts) recorded in `paper/v1.2.md` section 24. Not committed.

### 20. Full metagenomic screen — 30,000 families, 165 ranked candidates (2026-10-03)

Scaled the 500-sequence pilot (section 19) to **30,000 protein families**. 13.5 h of ESM-2
650M + SAE inference on MPS. **BLAST was deliberately not run**, at the user's direction —
so this screen makes no homology claim, and the "homology search would miss it" framing from
the pilot is absent.

#### Source and curation — `src/biopet_sae/build_screen30k.py`

MGnify (EBI Metagenomics) API v1, three plastic-associated studies, two assemblies each,
cached locally (277 MB):

| study | biome | families | sampled |
|---|---|---|---|
| MGYS00005625 | Engineered — plastic metagenome (PRJEB15404) | 82,884 | **10,000** |
| MGYS00006544 | Marine — genomic signatures (PRJNA777294) | 73,047 | **10,000** |
| MGYS00005970 | Lab enrichment — PE film, in situ marine | 51,385 | **10,000** |

Sampling is **stratified equally** so no assembly dominates; natural proportions would have
been 40/35/25.

| funnel step | remaining |
|---|---|
| 1 · predicted CDS across 6 assemblies | 2,104,436 |
| 4 · length 50-1000, standard AAs, unique, contains **GxSxG** | 240,964 |
| 5 · not >=90% identical to any of the 542 known plastizymes | 240,961 (**3 dropped**) |
| 6 · families at 90% identity — **one representative each** | 207,316 |
| 7 · stratified sample | **30,000** |

**"30,000 families, not 30,000 sequences."** The elbow filter uses the *generic* motif;
filtering on our training variants would pre-select for what the model already knows.

#### Results — `src/biopet_sae/score_screen30k.py`

| probe threshold | candidates |
|---|---|
| >= 0.9 | **4** |
| >= 0.8 | 21 |
| >= 0.7 | 40 |
| >= 0.5 | **165** |
| >= 0.3 | 683 |

Median 0.0375, p90 0.132, p99 0.417, **max 0.9378**. Reference: our 42 test positives had
median 0.60, max 0.97.

**The pilot's ceiling was a sample-size artifact.** Section 19 reported max 0.582 and noted
the maximum could not be extrapolated from 500 draws — correctly, as it turned out. At
30,000 the best candidate reaches **0.930**, well inside the range of known plastizymes.

**Triad distribution across all 30,000:**

| triad residues located | n |
|---|---|
| 0/3 | 21,123 |
| 1/3 | 7,264 |
| 2/3 | 1,308 |
| **3/3** | **305** (1.0%) |

#### Ranked shortlist: 165

Tiers are evidence-based, with no homology component:

| tier | definition | n |
|---|---|---|
| **A** | probe >= 0.5 **and** 3/3 catalytic triad | **40** |
| **B** | probe >= 0.5 and 2/3 triad | 74 |
| **C** | probe >= 0.5, triad <= 1/3 | 51 |

Top five: probe 0.930 `GQSGG` 607 aa PE-film · 0.913 `GYSMG` 439 aa PE-film · 0.879 `GQSGG`
528 aa PE-film · 0.834 `GYSMG` 430 aa Marine · 0.832 `GFTNG` 552 aa PE-film. All 3/3 triad,
pocket activation 0.83-0.92.

#### Three findings worth keeping

**1 · The probe enriched for the plastizyme elbow without being told to.** Plastizyme-style
elbow variants (`G[HWYFDENQ]S[MQA]G`) are 2.9% of the 30,000 pool but **42.5% of Tier A** —
a **14.7x enrichment**. The motif was never an input to the probe, only to the generic
`GxSxG` pre-filter.

| set | plastizyme-style elbow |
|---|---|
| all 30,000 | 866 (2.9%) |
| tier A | 17/40 (**42.5%**) |
| tier B | 26/74 (35.1%) |
| tier C | 9/51 (17.6%) |

**2 · Tier A concentrates in the enrichment biomes.** From equal 10,000-sequence samples:
**21 PE-film, 15 Marine, 4 Engineered.** The polyethylene-film in-situ enrichment yields 5x
the Tier A candidates of the engineered-plastic metagenome. Tiers B and C are flat across
biomes (22/23/29 and 14/21/16), so the skew is specific to the strongest tier.

**3 · The two-stage filter separates the tiers on independent evidence.** Pocket-latent
activation, which is not part of the tier definition: **median 0.76 in Tier A against 0.49 in
Tier C**. Of 165 candidates above threshold only 40 have a complete triad; the 51 in Tier C
have probe confidence with no mechanistic support — the pattern that was noise in four of
five pilot cases.

#### The anomaly to chase

The **highest raw score in the entire run is 0.938 with only 2/3 triad** — Ser ok (act 0.97),
His ok (0.82), **Asp misses** (act 0.80, lands on G not D). 341 aa, engineered biome, elbow
`GYTEG`. It ranks 41st rather than 1st because tiering puts complete triads first.

Either a genuine variant triad (a Glu-for-Asp substitution would not be scored by our
`D/E` rule if the geometry differs), or a probe artifact. This is precisely the case the
mechanistic evidence exists to flag, and it is the first candidate worth looking at by hand.

#### Honest framing

- **Nominations, not discoveries.** No experimental validation exists for any of the 165.
- **Base rate.** The probe was trained at 47.8% prevalence; a metagenome's true rate is far
  lower, so precision here is much worse than the 0.830 benchmark. **Rank, not threshold.**
- **No homology comparison.** BLAST was not run, so we cannot say whether these would have
  been found by sequence search. That claim requires a separate run against the 542 known.
- **Nothing here is a novel fold.** Everything passed a GxSxG pre-filter, so all 30,000 are
  esterase-like by construction. The screen ranks within that family.

Artifact: `https://claude.ai/artifact/9UVa672uQUXExrESd2xvtv` — 165 ranked candidates,
filterable by tier and biome, per-candidate evidence cards, full provenance and funnel.
Data: `data/mgnify/screen30k_{set.tsv,scored.jsonl,funnel.json,payload.json}`.

**Next:** BLAST the 40 Tier A candidates against the 542 known to populate the
"probe yes, homology no" cell; ESMFold the top 10 to check the triad is spatially assembled;
hand-inspect the 0.938 anomaly.

---

### 19. Prospective metagenomic screen — MGnify plastisphere, 500 candidates (2026-09-29)

First application of the pipeline to sequences that are not in any curated database.
**Result: no candidate scores as high as a typical known plastizyme, and one candidate has
a full evidence stack that homology search misses.** Both halves matter.

#### Where the sequences came from

UniProt is not a viable source: it holds **28** proteins under metagenome taxa in total, and
`PF01083` (cutinase domain, unreviewed) is 96% already named "cutinase" and 78% fungal —
screening it would rediscover the training family. Metagenomic proteins live in MGnify/IMG-M.

MGnify has 14 plastic-related studies; three have assemblies with predicted CDS:

| study | biome | assemblies used |
|---|---|---|
| MGYS00005625 | Engineered — plastic metagenome (PRJEB15404) | 2 |
| MGYS00006544 | Marine — genomic signatures (PRJNA777294) | 2 |
| MGYS00005970 | Lab enrichment — polyethylene film, in situ marine | 2 |

#### Funnel — `src/biopet_sae/fetch_metagenome.py`

| step | remaining |
|---|---|
| 1 · predicted CDS across 6 assemblies | 2,104,436 |
| 2 · length 50-1000, standard amino acids | 1,864,531 |
| 3 · exact duplicates removed | 1,837,819 |
| 4 · contains a **GxSxG** nucleophile elbow | 240,964 |
| 5 · not >=90% identical to any of our 388 positives | 240,961 (**3 dropped**) |
| 6 · family-deduplicated at 90% (mmseqs, one rep per cluster) | 207,316 |
| 7 · **random sample, seed 0** | **500** |

**The elbow filter uses the generic motif, not our training variants.** Filtering on
`G[HWYFDENQ]S[MQA]G` would pre-select for what the model already knows and make any hit
partly circular. Of the 500 sampled, only **16 (3.2%)** carry a plastizyme-style elbow, and
there are **335 distinct** elbow 5-mers — so the set is fold-enriched but not motif-enriched.
Biome balance: 200 engineered / 163 marine / 137 PE-film.

**Only 3 of 240,964 were near-identical to a known positive.** The plastisphere does not
contain copies of our training set.

#### Scoring — `src/biopet_sae/score_screen.py`

Probe refit on train only (C=1e-2, eta0=1e-4), BLASTp against the 303 training positives,
SAE triad localisation with the learned offsets, pocket-latent activation.

| probe threshold | candidates |
|---|---|
| >= 0.9 | **0** |
| >= 0.7 | **0** |
| >= 0.5 | 4 |
| >= 0.3 | 15 |
| >= 0.1455 (shipped) | 42 |

**Median score 0.032, maximum 0.582.** For reference our 42 test positives had median 0.60
and max 0.97 — **the best metagenomic candidate scores below the median known plastizyme.**

#### The four cells

| | count |
|---|---|
| probe YES, BLAST no — *the interesting cell* | **5** |
| probe YES, BLAST yes — both agree | 10 |
| probe no, BLAST yes — homology only | 16 |
| probe no, BLAST no | 469 |

#### The one candidate with a complete evidence stack

`MGYA00572563 | ERZ1644704.2386-NODE-2386`, engineered-plastic biome, 477 aa:

| evidence | value |
|---|---|
| probe score | 0.324 |
| **BLASTp** | **e = 0.09, 31% identity — above the 3.16e-2 cutoff, so homology MISSES it** |
| catalytic Ser | latent 9529 peak 229, act **1.04**, +1 -> **S** |
| catalytic Asp | latent 2661 peak 427, act 0.69, +3 -> **D** |
| catalytic His | latent 7734 peak 456, act 0.78, +1 -> **H** |
| pocket latents | mean activation **0.98** — the highest in the whole screen |
| elbow | `GYSLG` |

Full triad located, strongest pocket signal of all 500, and BLASTp would not return it.
That is exactly the cell the detection result predicts the probe should win in.

**The other four in that cell are noise** — all 0/3 triad with activations of 0.01-0.08, i.e.
the latents are silent. The probe scored them 0.30-0.35 on dense embedding alone with no
mechanistic support. **This is the two-stage filter earning its place:** probe score alone
would have nominated five, the SAE evidence rejects four.

#### Honest reading

1. **No strong hit.** Max 0.582 against a test-positive median of 0.60. If the plastisphere
   contains an obvious novel plastizyme, this screen did not find it.
2. **One real nomination**, with probe + full triad + top pocket signal + BLAST disagreement.
   It is a hypothesis for wet-lab testing, not a discovery.
3. **The base-rate caveat stands.** At the shipped 0.1455 threshold this set yields 42
   candidates, most of which will be ordinary esterases (specificity 0.871 -> ~13% FP rate).
   Reporting the *ranking* rather than a thresholded call is the only defensible framing.
4. **7 of 500 have 3/3 triad**; 356 have 0/3. The SAE evidence is a strong filter and is
   nearly independent of the probe score.

Data: `data/mgnify/screen_set.tsv`, `screen_scored.json`, `screen_funnel.json`,
raw assemblies cached under `data/mgnify/raw/`.

**Next:** ESMFold the top ~10 to check the triad is spatially assembled; pull more
assemblies for breadth; check whether the top candidates skew to one phylum (the
P86325 failure mode).

---

### 18. Evidence table for the localisation claim, and what each stage rests on (2026-09-29)

#### The four-stage pipeline

| stage | question it answers | data | held out? | result |
|---|---|---|---|---|
| **1 · Selection** | does this latent separate degraders from non-degraders? | train+val, 707 seqs | no | 7 of 10,240 survive: delta >= 0.9 **and** bootstrap top-10 >= 75% |
| **2 · Offset** | is the catalytic residue a constant distance from the peak? | train+val, 14 curated | no | Ser **+1** (13/14), Asp **+3** (9/14), His **+1** (12/14) |
| **3 · Localisation** | does that offset transfer to unseen proteins? | 45 held-out | **yes** | **57/59 proteins = 96.6%** (residues: 164 exact, 8 recovered) |
| **4 · Null** | would an arbitrary latent do as well? | all 10,240 | **yes** | median latent 4.4%; ours 98-100%, nothing higher |

**Stage 2 is what assigns each latent to a residue, not stage 3.** Each latent has exactly one
target whose gap repeats (13/14, 9/14, 12/14); against its other two targets the modal gap
appears 5-6 times out of 14 — a mode of noise. See the cross-localisation caveat below.

#### Evidence table, ranked by strength

The 1,326-esterase result is **deliberately excluded here** — see the note after the table.

| evidence | n | ground truth | plastizymes? | held out? | result |
|---|---|---|---|---|---|
| train+val curated `ACT_SITE` | 18 | **experimental** | yes, ours | no | Ser 17/18, Asp 11/18, His 15/18 |
| **test curated `ACT_SITE`** | **2** | **experimental** | yes, ours | **yes** | 2/2, 2/2, 2/2 |
| **46 unseen curated cutinases** | **46** | **experimental** | family, not ours | **yes** | Ser **100%**, Asp **97.8%**, His **95.7%** |
| test GxSxG motif | 38 | motif (regex) | yes, ours | **yes** | **38/38** |
| random-latent null | 10,240 | — | — | **yes** | median 4.4% vs ours 98-100% |

**Why the 1,326 curated esterases are not in this table.** They support the claim *"9529 fires
at the catalytic serine"* — and for that they are the best evidence available, since only 20
of our 388 positives are curated at all. But they are **not plastizymes**, so they cannot
support the headline. If these latents localise the catalytic machinery of any esterase, they
are esterase features, not plastizyme features — which is the finding of 14d/15/16, and the
reason the probe reaches only 0.79 AU-ROC against genuine esterases. **That result belongs
under limitations, not results.**

**The uncomfortable fact the table exposes:** experimental ground truth on *held-out
plastizymes* is **2 sequences**. Everything larger is the wrong organisms (1,326), the wrong
strictness (38, motif-based), or not held out (18). The 46 cutinases are the best compromise —
curated, experimental, outside our data, right family — but they are curated *cutinases*, not
enzymes assayed on plastic, and calling them plastizymes would overstate it.

#### The motifs behind the 38/38

Latent 9529's peak-centred 5-mers across the 38 held-out test positives that carry an elbow
(4 of the 42 have no GxSxG at all):

| motif | n | in train+val? | | motif | n | in train+val? |
|---|---|---|---|---|---|---|
| GHSQG | 9 | yes | | GISLG | 2 | **no** |
| GHSMG | 8 | yes | | GNSVG | 1 | **no** |
| GDSAG | 3 | yes | | GFSLG | 1 | **no** |
| GYSQG | 3 | yes | | GHSFG | 1 | **no** |
| GESAG | 2 | yes | | GCSLG | 1 | **no** |
| GWSMG | 2 | yes | | GLSLG | 1 | **no** |
| GHSLG, GFSQG, GQSMG, GFSMG | 1 each | yes | | | | |

**7 of the 38 hits use a variant that appears in no train+val positive** — `GISLG`, `GNSVG`,
`GFSLG`, `GHSFG`, `GCSLG`, `GLSLG`. The latent is locating the structural position, not
matching a memorised string.

**CORRECTION:** earlier write-ups (including the atlas footer through v17) cited `GDSAG` and
`GESAG` as unseen variants. They are not — both occur in train+val. The list above is correct.

#### Cross-localisation caveat

Each selected latent localises more than one triad residue on the held-out set:

| latent | Ser loc | Asp loc | His loc | assigned |
|---|---|---|---|---|
| 9529 | **100.0%** | 4.5% | 4.4% | Ser — specific on both measures |
| 2661 | 86.7% | **100.0%** | 93.3% | Asp — on offset consistency only |
| 7734 | 84.4% | 4.5% | **97.8%** | His |

**Why 2661 scores 86.7% on Ser despite a 5/14 offset fit:** 43 of the 45 held-out entries are
external cutinases, one homogeneous family, where the triad separations collapse to a single
value so *any* fixed offset transfers. Across the diverse train+val set the separations are
bimodal (Ser->Asp `[45, 46x5, 55x5, 82]`, sd 60.8), which is why 9529's Asp gap fits only
6/14. **"Held out" here means unseen, not diverse** — that inflates every cross-target rate.
Only 9529 is specific on both measures; 2661 and 7734 rest on offset consistency plus motif.

---

### 17. THE NULL for SAE residue localisation — random latents under the identical procedure (2026-09-29)

**This is the control the localisation headline depends on. Nothing in sections 15-16 means
anything without it, and it was missing until the user asked for it.**

#### Why the earlier null was not enough

Section 16 reported a null that randomised the *position*: start the +/-3 search from a random
residue instead of the latent's peak, 3.2% hit rate. That tests **the search rule** — "is the
window just finding serines by luck?" It does not test **the interpretation** — "is this
latent special, or would any of the 10,240 do as well?" The headline claim is that particular
dictionary elements encode catalytic residues, and only the second question addresses it.

#### The procedure, applied identically to all 10,240 latents

1. One ESM-2 + SAE pass per protein gives a `residues x 10,240` activation table; take the
   **argmax down each column** -> one peak position per latent per protein.
2. **Gap** = `curated_residue_index - peak`. Note: the *curated* UniProt residue, never the
   nearest serine — using "nearest" would be circular, since these proteins are serine-rich.
3. **Learn the rule on train+val only** (the 14 entries with curated sites): the modal gap.
4. **Apply to the 45 held-out entries** (2 test + 43 external cutinases), scoring exact, or
   recovered by sliding +/-3 to the nearest residue of the right type.
5. Repeat for every latent -> a distribution to place the chosen ones against.

Every random latent gets **the same advantages**: its own fitted offset, the same window, and
no activation requirement (the real pipeline demands non-zero activation at the slid
position; the null does not). The null is therefore run under a *looser* rule than the real
result it is compared to.

#### Result

| latent | role | its rate | random mean | random median | 95th pct | scoring higher | tied |
|---|---|---|---|---|---|---|---|
| 9529 | Ser | **100.0%** | 12.0% | **4.4%** | 60.0% | **0 / 10,240** | 25 |
| 2661 | Asp | **100.0%** | 12.6% | **4.5%** | 56.8% | **0 / 10,240** | 7 |
| 7734 | His | **97.8%** | 9.7% | **4.4%** | 57.8% | **0 / 10,240** | 8 |

A typical latent localises at **4.4%**. The three chosen ones are at 98-100%, and nothing in
the dictionary beats them. **Percentiles: 99.75%, 99.92%, 99.91%.**

**Shuffled-position null** (real latents, protein A's peak applied to protein B, 200
permutations): Ser mean 12.0% / max 31.1%; Asp 7.5% / 31.8%; His 6.5% / 40.0%. Against
97.8-100% real.

#### Worked example of why the null was needed

| | latent 9529 | latent 77 (arbitrary) |
|---|---|---|
| gaps across the 14 train+val | `[1]x13, [18]x1` | `[-125,-111,-110,-105x3,-58,-51x4,26,67x2]` |
| modal gap | **+1** (13/14) | **-51** (4/14, a mode of noise) |
| held-out exact | **45/45 = 100%** | **26/45 = 58%** |

**A nonsense rule still scores 58%**, because a wrong position plus a +/-3 window often lands
on *some* serine. That is the whole reason the floor has to be measured: without it, 97% looks
decisive when the relevant comparison is not zero.

#### The caveat that must travel with the number

**These latents are not unique — they sit at a ceiling shared with others.** 26 latents tie
with 9529 on Ser, 8 with 2661, 9 with 7734; roughly 600 exceed 50%. Characterising the 26:

- only **2** peak where 9529 peaks (6590, at the same residue in 58/59 proteins, offset +1)
- **10** correlate above 0.8 with 9529 (897 at 0.983, 2804 at 0.980, 6778 at 0.975) —
  redundant copies, the SAE splitting one feature across dictionary elements
- the rest peak elsewhere and reach the serine by their own constant; learned offsets across
  the 26 span **-13 to +26**, i.e. a ~40-residue window around the elbow
- 9529 is **not** the strongest of them (2925 has higher mean activation, 0.513 vs 0.508).
  What distinguishes it is offset consistency: +1 in 13 of 14, landing on the `x` of GxSxG

**So the supportable sentence is a percentile, not a superlative:** *"latent 9529 localises
the catalytic serine better than 99.75% of the 10,240 dictionary elements"* — **not** "the SAE
learned a serine detector". The catalytic-serine neighbourhood is encoded redundantly by a
family of latents; 9529 is its most precisely positioned member. This is consistent with the
0.88 mean correlation already recorded among the six selected latents (section 16).

Data: `data/sae_650M_L33/null_random_latents.npz` (per-latent rates for all 10,240 x 3
targets), `null_random_latents_summary.json`, `all_latent_peaks.npy` (59 x 10,240 peaks).
Script: `scratchpad/null_features.py`.

---

### 16. Neighbourhood protocol — the latents do not fire on the catalytic residue (2026-09-29)

**Supersedes the framing in section 15.** Section 15's numbers are arithmetically right but
described wrongly, in two ways that a user's questions exposed.

**(a) Conditioning on "the latent fired" was invalid.** §15 reported accuracy over proteins
with activation > 0.3. Every protein in that set has a curated three-residue catalytic site,
so a latent that stays silent is a *miss*, not a case to exclude. Selecting on the model's
own output discards the failures. Honest denominators, all 1,326: **Ser 78.9%, Asp 35.9%,
His 63.1%**.

**(b) The +2 offset "correction" in §15 was wrong for this project's domain.** It was fitted
across 1,326 esterases, of which 1,266 are not cutinases. Per family:

| subset | n | offset +2 | offset +3 |
|---|---|---|---|
| cutinases (EC 3.1.1.74) | 60 | 13.6% | **86.4%** |
| all curated EC 3.1.1.- | 1326 | **35.9%** | 17.7% |
| non-cutinases | 1266 | **36.9%** | 14.5% |

The original **+3 is correct for cutinases**, which is what the project is about. Excluding
the 14 entries that are our own training positives, on **46 unseen curated cutinases**:
Ser 100%, Asp 97.8%, His 95.7%.

#### The finding that changes the claim

**None of the three latents fires on the catalytic residue.** Activation at the curated
residue, relative to the latent's own peak:

| latent | act at peak | act at the residue | ratio | >=50% of peak |
|---|---|---|---|---|
| 9529 Ser | 0.634 | 0.152 | 0.24 | **0/59** |
| 2661 Asp | 0.311 | 0.078 | 0.25 | **0/58** |
| 7734 His | 0.380 | 0.186 | 0.49 | 13/59 |

Each peaks one to three positions *before* the residue; by the residue itself the signal has
decayed to a quarter (Ser, Asp) or half (His). So the defensible claim is **"fires in a
catalytic-residue neighbourhood, from which the residue sits at a learned offset"** — not
"detects the catalytic serine". The localisation results stand; the mechanism is motif
detection plus a fixed offset, and that must be stated.

This also reframes the Asp problem: the mapping was always indirect, so whether the Asp lands
at +2 or +3 depends on local motif spacing, which differs by family. No single constant works.

#### Scoring protocol now used

1. Offsets learned on **train+val only**: Ser +1 (13/14), Asp +3 (9/14), His +1 (12/14).
2. **exact** — learned offset lands on the curated residue.
3. **recovered** — it does not, so slide **±3** to the nearest residue of the right type
   (S / D-E / H) with non-zero activation. Counted separately, never merged into exact.
4. **miss** — no such residue in the window.

**Null control:** the same ±3 rule from a *random* position hits the curated residue in
**3.2%** of trials (41/1,280). Widening to ±5 recovers nothing further, so the window is
correcting one-residue offsets rather than searching until something is found.

**The unit is the protein, not the residue.** Within a protein the three latents agree
**100% of the time** — either all three residues are located or none is, because a canonical
alpha/beta-hydrolase active site either is present or it is not. Counting 176 residue-level
outcomes treats one protein-level failure as three. Corrected to per-protein reporting:

| view | proteins | fully located | partial | none | rate |
|---|---|---|---|---|---|
| test (held out) | 2 | **2** | 0 | 0 | 100.0% |
| external cutinases | 43 | **42** | 1 | 0 | 97.7% |
| train+val (in-sample) | 14 | **13** | 0 | 1 | 92.9% |
| **all** | **59** | **57** | **1** | **1** | **96.6%** |

Residue-level breakdown beneath it, retained for detail but **not** as the headline:

| view | n pairs | exact | recovered | miss |
|---|---|---|---|---|
| test | 6 | 5 | 1 | 0 |
| external cutinases | 128 | 125 | 2 | 1 |
| train+val | 42 | 34 | 5 | 3 |
| **all** | **176** | **164** | **8** | **4** |

**CORRECTION (2026-09-29).** This was previously reported as **172/176 = 97.7%**, which
inflates the denominator roughly threefold: with pairwise agreement of 1.0 between the three
latents, 176 residue outcomes are not 176 independent observations. The headline is now
**57/59 proteins = 96.6%**. The rate barely moves; the sample size does. Same error class as
counting sequences instead of homology components, which was flagged for the test split and
then reproduced here.

The 2 failing proteins: **BPS0016** (PAM), which degrades **PLA** and is the only
non-alpha/beta-hydrolase among the 59. InterPro assigns it `IPR034193` (Proteinase K-like
catalytic domain) and `IPR023827` (peptidase S8, subtilisin, Asp active site) — it is a
**subtilisin protease**, not an esterase, and the same family as the six PLA-degrading
proteases that *were* excluded as "not an esterase fold". It entered the positive set because
the fold filter matched enzyme-name strings and its label reads "PAM" (see §26). All three
latents are silent on it (peak activations 0.070, 0.054, 0.007), which is the correct
behaviour: with no alpha/beta charge-relay geometry present there is nothing for them to
detect, so they signal "not my motif" rather than firing wrongly. **Counting it as a
localisation failure understates the latents** — on the 58 proteins that actually have the
fold, the rate is 57/58. And **Q9Y7G8** (cutinase pbc1), which misses on His only.

*(Corrected 2026-10-12. This entry previously called BPS0016 "a polyamidase", which is wrong
on both counts: its substrate is PLA, a polyester, and its fold is subtilisin. The revised
reading strengthens rather than weakens the result.)*

Data: `data/sae_650M_L33/atlas_views.json`, `learned_offsets.json`.
Atlas rebuilt around this protocol: `https://claude.ai/artifact/KcdU4xp1QUSvuWzrL4pGwS`
(v13) — three views, methods section, per-tier reporting.

#### Two UI bugs found and fixed, both from duplicated state

1. **Chip ordering.** `Object.keys()` orders integer-like keys numerically, not by insertion,
   so latent chips interleaved the triad and pocket groups and the page opened on the wrong
   latent. Fixed by carrying an explicit `order` field and sorting on it.
2. **Offset drift.** The caption read `latents[j].offset` while scoring used
   `protocol.offsets[j]`; the first was left at +2 from the reverted correction, so the page
   scored at +3 while displaying "+2". Both now derive from one source, and a build check
   fails if they diverge — it compares both config fields against the `off_used` recorded on
   every scored entry.

Same root cause in both: one fact stored twice. Worth checking for on any further edits.

#### Independent audit of the atlas (2026-09-29)

A separate agent audited the page against the payload. Its two HIGH findings were
re-verified here before acting, and both were real.

**1. The footer printed `undefined` nine times.** `line()` read `aggregate[view][j].hit`,
but the aggregate objects carry only `exact` / `recovered` / `miss` / `n`. The per-*entry*
records do have `hit`, and the top scoreboard computes `exact+recovered` correctly — so the
key existed in one place and not the other. Third instance of the duplicated-state bug in
this page. Fixed, and a build check now recomputes the footer from per-entry tiers.

**2. BPS0393's structure was the wrong protein.** PDB **4OYL** was attached to this cutinase
(A0A075B5G4), but only **57.7%** of mapped positions agree in residue identity — it is a
homolog, not this sequence. Comparison across all 14 view-1 entries:

| entry | PDB | residue-identity agreement |
|---|---|---|
| BPS0239 / BPS0054 / BPS0269 | 3VIS / 3GBS / 4EB0 | 100% |
| BPS0282 (IsPETase) | 5XFY | 99.6% |
| **BPS0393** | **4OYL** | **57.7%** |

The 3D view was drawing a label reading `Ser106` on a **glutamine**, `Asp161` on an
**alanine**, and `His175` on a **serine**, and colouring ~42% of the cartoon by the wrong
residue mapping. Its `dist_to_ser` values were also meaningless. The structure fields were
removed; the entry now renders sequence-only like the test and external views. **Scoring is
unaffected** — localisation runs on sequence, so BPS0393 remains 3/3 exact and the
aggregate stays 172/176. view1 keeps 14 entries, 13 of which now carry a structure.

**A finding that touches a scientific claim.** The "What survived" table read as the top six
latents by Cliff's delta. It was not:

| latent | delta | bootstrap top-10 | shown before? |
|---|---|---|---|
| 5271 | 0.93405 | 78.5% | **no — outranks 7734** |
| 7734 | 0.93381 | **79.5%** | yes |

Latent **5271 outranks 7734** and was omitted, and **7734 itself falls below the 80%
stability cutoff** — `polyester_latents.json` records the strictly stable set as
`[9529, 5201, 2661, 7942, 5255]`, five latents, without the His one. 5271 is now shown with
a note that it was never localised, and the page states plainly that 7734's selection is
weaker than the others' and that it is scored because its His assignment was established
independently.

**Also corrected:** the pocket geometry was stated as "8–10 Å" and "23–74 residues" as if
those were ranges; the true medians are 8.1/9.6/9.0 Å and 21.5/68/74 residues, with observed
ranges 6.0–24.5 Å and 11–128. Now reported as medians with ranges. Plus four presentation
bugs: `uniprot_shift` was applied in the side panel but not the sequence ruler or tooltip
(a 38-residue discrepancy on the three Q6A0I4 entries); the "N-terminal residues absent"
explanation was emitted for negative offsets, where it is backwards; the ruler's
`padStart(4)` was collapsed by HTML white-space handling so the monospace columns did not
line up; and recovered rows showed the curated residue in the predicted column, now
`T225 → D224`.

**Not applied:** the audit flagged a missing doctype/charset/viewport. The Artifact publish
skeleton supplies those, and the tool explicitly requires the file to omit them, so adding
them would break publishing. The checked-in file is only valid as artifact input.

**Verified clean by the audit** (independently recomputed): all 12 aggregate groups match the
per-entry tiers; `latents[j].offset` and `protocol.offsets[j]` agree after the earlier fix;
chip ordering holds; every activation array length equals its sequence length with no nulls
or NaNs; `pred == peak+1+off_used` for all 175 scored records; all view2/view3 guards for
missing structure fields are complete; max sequence length 386, so `Math.max.apply` is safe.

---

### 15. Triad latents verified at n=1,326 against curated UniProt active sites (2026-09-29)

> **Framing superseded by section 16 (2026-09-29).** The counts below are correct but were
> reported over a denominator conditioned on the latent firing, which hides genuine misses,
> and the +2 Asp offset recorded here is wrong for cutinases (+3 is right). Read section 16
> first; the table below is retained because the wrong version was stated before the right
> one.

The His and Asp assignments previously rested on 12 train/val sequences plus 2 in test.
Tested properly: every reviewed UniProt entry in `ec:3.1.1.-` carrying **three** curated
`ACT_SITE` residues — **1,326 proteins**, overwhelmingly ordinary esterases and lipases,
not plastizymes. ESM-2 650M L33 + SAE, 424 s on MPS.

**Counting only proteins where the latent actually fires (activation > 0.3):**

| latent | role | offset | exact | within 1 | n firing |
|---|---|---|---|---|---|
| 9529 | catalytic **Ser** | +1 | **865/866 (99.9%)** | 100% | 866 |
| 7734 | catalytic **His** | +1 | **556/595 (93.4%)** | 99% | 595 |
| 2661 | catalytic **Asp** | **+2** | **393/566 (69.4%)** | 85% | 566 |

**Accuracy tracks activation strength**, the key evidence that these are real detectors
rather than coincidences:

| latent | top 25% act | 2nd | 3rd | bottom 25% |
|---|---|---|---|---|
| 9529 | **100%** | 100% | — | 17% |
| 7734 | **98%** | 71% | — | 14% |
| 2661 | **77.7%** | 48.9% | 15.9% | 0.9% |

Bottom-quartile mean activation is 0.03 — the latent is silent there, so its argmax is
arbitrary. Pooling silent with firing proteins is what made the first pass look bad.

**CORRECTION — latent 2661's offset was wrong.** Published at **+3**, derived from
hand-checking three enzymes (LCC, IsPETase, Cut190) where +3 happened to fit. Across 1,326
proteins the correct offset is **+2**: aspartate is the modal residue at peak+2 in **522**
of them, against 299 at peak+3. At +3 it scores 18% exact overall; at +2, 36% overall and
69% among firing proteins.

*I also first reported this as "2661 is not really an Asp detector."* That was wrong — an
artefact of pooling firing and silent proteins. Conditioned on firing it is a genuine, if
less precise, Asp detector. Recorded because the wrong version was stated before the right
one.

**Why this matters.** These latents fire on the catalytic machinery of esterases in
general, not of plastizymes specifically — 1,326 mostly-non-degrading proteins is the
direct evidence. It confirms the reading in 14d: the interpretable features explain "is
this a hydrolase with a canonical active site," the *easy* half of the polyester decision.
They do not explain the probe's advantage over BLASTp on hard esterase negatives.

Data: `data/sae_650M_L33/triad_verification_uniprot.json` (summary + per-entry).
Atlas updated to the corrected offset: `https://claude.ai/artifact/KcdU4xp1QUSvuWzrL4pGwS`.

---

## Current status (checkpoint, 2026-11-23)

*Rewritten 2026-11-14, updated 2026-11-23 (PHA rows removed, release v1.1, PAZy update; see §28h-§28i). The previous version of this section was a 2026-09-28 checkpoint that
still described the explainability half as unported, PHA as generalising at 83%, and tier 3 as
~46% fold-matched. All three are superseded — see §21, §25–§26 and §22.*

**The task is locked (§26).** Positive: an enzyme experimentally reported to degrade a plastic
whose backbone is joined by carboxylic ester bonds — 388 enzymes in v1 (404 in v1.1), 21 polyester
substrates, PET the largest at 320 (336). Hard negative: an alpha/beta-hydrolase-fold protein with
the same Ser-His-Asp triad, never reported to degrade a plastic — 51 in v1 (50 in v1.1). Easy
negatives: 372. There is **no fold clause**
in the positive definition; fold is reported, not enforced (§28a). The PHA generalisation claim
is withdrawn (§25–§26).

**Dataset.** Two releases (§28i). **v1** (`biopet_sae_dataset_v1`) is frozen: 811 rows (388 positive,
423 negative); the 36 PHA depolymerases were **removed** from the release and the handoff on
2026-11-23 (§28h) and survive only in `data/processed/*`. **v1.1** (`biopet_sae_dataset_v1_1`,
`batch` column) adds 16 PAZy PET enzymes and drops negative `P37957`: 826 rows (404 / 422), 50 hard
negatives (39 / 1 / 10), 77 positive components, largest `BPS0190` 259 of 404 (64%), test 51 pos /
62 neg. `handoff/` is built from v1.1. **Every result below is on v1 (test n=104);** the nine
positives added to test have not been scored. The tables below are v1.

| split | positives | negatives | total | prevalence | hard negatives |
|---|---|---|---|---|---|
| train | 303 | 285 | 588 | 0.515 | 40 |
| val | 43 | 76 | 119 | 0.361 | **1** |
| test | 42 | 62 | 104 | 0.404 | 10 |

The negatives by what they actually are (§21; this supersedes the EC-based tiers, and tier 3 is
27% fold-matched, not 46%): 51 alpha/beta hydrolase, 36 ester-active in a different fold, 99
tRNA-processing hydrolases with no serine nucleophile, 237 naive controls. Positives: 74 homology
components, **66% in one** (`BPS0190`, 256 of 388). Max cross-split identity 0.3967 / 0.3800 /
0.3983.

**Detection.** ESM-2 650M layer 33, max-pooled, linear probe on val log-loss. Test, positives
fixed at 42:

| test negatives | n | AU-ROC | AU-PRC |
|---|---|---|---|
| all — blended headline | 62 | 0.9681 | 0.9526 |
| naive controls | 36 | 1.0000 | 1.0000 |
| tRNA-processing | 3 | 1.0000 | 1.0000 |
| ester-active, different fold | 13 | 0.9890 | 0.9967 |
| **alpha/beta hard negatives** | **10** | **0.8167** | 0.9550 |

The hard-negative interval is [0.606, 0.963] bootstrapped over homology components (§28g).
BLASTp on the same split: test AU-ROC 0.9040, AU-PRC 0.7552. The paired bootstrap over test
homology components puts the probe ahead on the *blended* comparison (AU-ROC +0.0603, AU-PRC
+0.1708, both significant), but **that comparison has not been decomposed by negative group**, so
it is not known whether the probe's advantage survives on the 10 hard negatives. 39 of 62 test
negatives are separated perfectly, so the blended number is mostly free weight. An amino-acid
composition baseline gets 0.7020 on this split and 0.8926 under random CV, a 0.19 inflation that
measures what a non-homology-aware split buys. **Test is spent**, and validation holds one hard
negative, so no further tuning on the hard task is possible without new data.

**Explainability.** Seven latents selected on train+val by Cliff's δ ≥ 0.9 and top-10 stability
≥ 150/200 random-half draws: 9529 (Ser), 2661 (Asp), 7734 (His), 7942/5201/5255 (pocket wall),
5271 (C-terminal second shell). They localise to curated catalytic residues in **57 of 59
proteins (96.6%)**, 44/45 on the held-out proteins, with offsets Ser +1, Asp +3, His +1 learned
on 14. `BPS0016` is a subtilisin and its silence is a correct rejection, so 57/58 on proteins that
have the fold (§28b). Nulls (§16, §17, §28d): 3.2% for a random position, **4.4% median for all
10,240 latents with none scoring higher** (25 tie 9529 on Ser, none passing selection).

**What the latents do not show.** They are a validated readout of serine-hydrolase catalytic
machinery: 100% of alpha/beta hard negatives fire the Ser latent, against 13.9% of
wrong-fold serine hydrolases and 0.0% of tRNA hydrolases (§22c). They are **not shown to separate
polyester degraders from fold-matched esterases.** Against the 41 fold-matched negatives in
train+val, δ falls to 0.506–0.716 per protein, 0.116 to −0.178 per homology component, and every
cluster-bootstrap interval includes zero; no latent anywhere in the dictionary reaches δ ≥ 0.9
there (max 0.847). The selection criterion was powered by easy negatives (§22, §22b). The honest
verdict is **undetermined, not refuted** — the test lacks power, not the effect. Single-latent
explanation of the hard-task margin does not exist at this layer; if the margin is real it is
distributed. No latent has been ablated, so everything here is correlational.

**Metagenomic screen (§19–§20).** 30,000 MGnify plastisphere families, probe ≥ 0.5, 165 shortlisted
(Tier A 40: probe ≥ 0.5 and 3/3 triad). No BLAST was run, so no homology claim is made. Caveat from
§22c: "3/3 triad" in Tier A now means alpha/beta-hydrolase-like, not polyester-specific. Follow-ups
named in §20 are not done (BLAST Tier A against the 542 known; ESMFold the top 10; hand-inspect
the 0.938 anomaly).

**Hand-off (§27, §28f–§28g).** `handoff/` holds `dataset.tsv`, `dataset.fasta`, `schema.json`,
`manifest.json` (sha256, counts, baselines to reproduce), and three documents: `README.md`,
`AGENTS.md` (six hard rules, known data issues in the positives, mistakes already made) and
`EVALUATION.md` (five reporting rules). Not yet given to the intern; it carries no license.

**Repository (§28e).** Latest commits `b2356d2` and `8d35185`, both pushed. Uncommitted as of
2026-11-14: `DATASET_CARD.md`, `PROGRESS.md`, `README.md`, `artifacts/{BLAST_vs_ESM2_650M_Polyester.md,
EDA_polyester_task.md, SAE_Feature_Interpretation.md, dataset_card.html, eda_dataset.html,
methods_detection.html}`,
`handoff/{AGENTS.md, EVALUATION.md, README.md, dataset.tsv, dataset.fasta, manifest.json, schema.json}`,
`data/release/biopet_sae_dataset_v1.{tsv,fasta}` (PHA removed), new `biopet_sae_dataset_v1_1.{tsv,fasta}` and
`_report.json`, `data/raw/pazy_update/`, `src/biopet_sae/{add_pazy_update,build_handoff,build_release}.py`
(updated 2026-11-23). **`data/mgnify/raw/` (277 MB) and `sample_cds.faa` (99.9 MB)
are committed and pushed; `.git` is 455 MB.** Never run `git commit` or `git push` without the
user's go-ahead for that specific commit.

**Published artifacts** (private; share from the Share menu):

| artifact | URL | version | local source |
|---|---|---|---|
| Catalytic Latent Atlas for Plastizymes | `claude.ai/artifact/KcdU4xp1QUSvuWzrL4pGwS` | v20 | **`artifacts/petase_structures.html`** (not `atlas_page.html`, now deleted) |
| Plastizyme Detection Methods | `claude.ai/artifact/US29Zu7ScaoAKSoepRQeMg` | v6 | `artifacts/methods_detection.html` |
| Polyester Degrader Dataset | `claude.ai/artifact/299aZWX8LjJM5E7bTQ4k7r` | v3 | `artifacts/eda_dataset.html` |
| Polyester Classifier Curves | `claude.ai/artifact/S1rzDp1cJYxA39xSTPZteE` | v1 | `artifacts/pr_curves.html` |
| Metagenomic Plastizyme Screen | `claude.ai/artifact/9UVa672uQUXExrESd2xvtv` | v1 | `screen30k.html` or `screen30k_page.html` — diff against the live page before republishing |

The "BioPET-SAE Dataset v1" artifact no longer exists on the account; its source
`artifacts/dataset_card.html` is unpublished. Live-update watches on all artifacts have ended.
**Before publishing to an artifact from an earlier context, read it and diff against the candidate
source; never publish without `url` to update something.**

**Data and written artifacts.**

| file | contents |
|---|---|
| `data/processed/dataset_splits_id40.tsv` | 847 rows incl. 36 PHA (not released): class, split, homology component |
| `data/processed/plastizymes_merged.tsv` | the 542 merged positives with substrates and sources |
| `data/release/biopet_sae_dataset_v1.{tsv,fasta}` | annotated release v1, 811 rows, frozen, with `negative_group` |
| `data/release/biopet_sae_dataset_v1_1.{tsv,fasta}` | v1 + 16 PAZy PET enzymes, 826 rows, `batch` column; report JSON alongside |
| `data/raw/pazy_update/` | raw PAZy records of the 16 added enzymes |
| `data/processed/uniprot_negative_annotation.json` | cached InterPro / Pfam / ACT_SITE for the 423 negatives |
| `data/embeddings_650M/embeddings.npz` | layers 1/9/18/24/30/33, mean and max pooled |
| `data/sae_650M_L33/` | SAE features, selection, localisation, nulls, atlas views |
| `data/mgnify/` | metagenome screen inputs and scored output |
| `handoff/` | the replication package |
| `DATASET_CARD.md`, `artifacts/EDA_polyester_task.md` | the dataset description |
| `artifacts/ESM2_650M_Polyester_Run.md`, `BLAST_vs_ESM2_650M_Polyester.md` | the polyester classifier and the BLAST comparison |
| `artifacts/{ESM2_8M_Dense_Run, ESM2_650M_Dense_Run, BLAST_vs_ESM2_8M, BLAST_vs_ESM2_650M}.md` | PET-only runs, superseded |
| `artifacts/SAE_Feature_Interpretation.md` | the PET-only SAE run, superseded; see its banner |

## Still open

Ordered by what the thesis needs. Numbering restarted 2026-11-14; older sections cite the
2026-09-28 list, which is no longer authoritative.

1. **Verify the fold of the positives (§28a).** Run InterProScan over all 404 sequences — it
   needs no accession, and only 152 have one. Then decide `BPS0016`, `BPS0013`, `BPS0394` and the
   suspect `BPS0060`, and decide explicitly whether the eight proteases and chitinases excluded as
   "not an esterase fold" belong in the positives under the substrate-only definition.
2. **More hard negatives.** Fetch 300–400 UniProt reviewed proteins carrying `IPR029058` plus three
   curated `ACT_SITE` residues, never assayed on a plastic, excluding cutinase and PETase
   families, then re-split so validation has real hard negatives and re-run detection and latent
   selection. The single highest-value fix, and it fixes one side only.
3. **Positive-side power.** 66% of positives are one component, so the comparison stays
   underpowered however many negatives are added. Needs one-vote-per-family weighting; diversity
   cannot be bought (fewer than 50 verified plastics-active enzymes existed at PAZy's publication).
4. **Decompose the BLASTp comparison by negative group.** Only blended metrics exist. Until the
   hard cell is compared, "the probe beats BLASTp" is a statement about the easy negatives.
5. **Explain the probe's hard-task margin.** Single-latent δ ranking cannot (ceiling 0.847).
   Needs a multi-latent or attribution method, at other layers (18/24 where structure dominates).
6. **Causal test.** No latent has been ablated. Ablate against a reconstruction baseline, not a
   raw baseline; the kinase project's precedent is greedy sequential ablation.
7. **`BPS0383` decision.** Currently kept and documented (§28c); the user has not ruled.
8. **HMMER baseline.** Prior art is the PAZy paper's profile HMM over ~3,000 PET-active homologs
   (R4). Needs `mafft` and `hmmbuild`, neither installed.
9. **PEZy-miner's 36 assayed non-degraders** remain unextracted from
   `~/Downloads/1-s2.0-S2214030124000178-mmc1.docx` — the only source of *measured* negatives.
10. **Screen follow-ups (§20).** BLAST Tier A against the 542 known; ESMFold the top 10; the 0.938
    anomaly (probe 0.938, 2/3 triad, Asp lands on a glycine, elbow `GYTEG`).
11. **Score the nine new test positives (§28i)** (HaloPETase1-4, R75estA, BhEstB, LipSh1-3): needs
    ESM-2 650M embeddings and the saved probe. Also UniProt/InterPro-check all 16 additions and
    decide whether to republish artifacts (they show v1 numbers; methods v6 has old PHA wording).
12. **More PAZy.** 505 live entries vs 462 exported; 19 PUR and 8 PA excluded as non-ester. Re-diff
    before any release, and consider re-checking PlasticDB for newer entries too.
13. **Housekeeping.** Add `data/mgnify/raw/` and `sample_cds.faa` to `.gitignore` and untrack them
    (commands in §28e); decide about purging history. Add a license to `handoff/`. Verify the
    local source of the metagenomic-screen artifact. 454 unannotated UniProt EC 3.1.1.101 entries
    were never triaged.

*Closed since 2026-09-28:* the SAE analysis on polyester labels; fold-based rebuild of the negative
description (§21, as a `negative_group` annotation rather than a rebuilt tier); the stale
`results/probe_embeddings_8M/` and `results/binary_embeddings_8M_dataset_splits_id40/`
directories; the `atlas_page.html` draft; latent 5271 (second shell, §16).

## Next steps

1. **InterProScan the 404 positives** and settle the fold question (Still open #1). Cheap, and it
   fixes the definition's weakest claim before anything is built on it.
2. **Fetch the 300–400 new hard negatives** (#2) and re-split. Do this before more modelling,
   because every hard-task number currently rests on 10 test proteins.
3. **Decompose BLASTp by negative group** (#4) on the existing split — no new data needed, and it
   says whether the central comparison holds where it matters.
4. **Give `handoff/` to the intern** once `.gitignore` is sorted and a license is added.
5. Then the attribution / ablation work (#5, #6).

## References

Bibliographic details verified against the Crossref API (2026-10-11), not quoted from memory.
Mechanistic summaries below come from abstracts and reviews of these works; **check the
wording in [R1] before quoting it as the basis for the exclusion rule.**

| id | citation | DOI | what it supports here |
|---|---|---|---|
| **R1** | Danso D, Chow J, Streit WR (2019). Plastics: environmental and biotechnological perspectives on microbial degradation. *Applied and Environmental Microbiology* 85, e01095-19. | `10.1128/AEM.01095-19` | The hydrolysable-backbone vs C–C-backbone split. **Cite for the 109 non-ester-mechanism exclusions.** |
| **R2** | Wei R, Zimmermann W (2017). Microbial enzymes for the recycling of recalcitrant petroleum-based plastics: how far are we? *Microbial Biotechnology* 10, 1308–1322. | `10.1111/1751-7915.12710` | PE, PS, PUR and PET compared side by side. Context for why PET is 82.5% of our positives — literature coverage, not our sampling. Open access, PMC5658625. |
| **R3** | Wei R, Tiso T, Bertling J, O'Connor K, Blank LM, Bornscheuer UT (2020). Possibilities and limitations of biotechnological plastic degradation and recycling. *Nature Catalysis* 3, 867–871. | `10.1038/s41929-020-00521-w` | "Progress is still limited to polyesters." **Cite for the task definition** — polyester degraders, not plastic degraders. |
| **R4** | Buchholz PCF, Feuerriegel G, Zhang H, Perez-Garcia P, Nover L-L, Chow J, Streit WR, Pleiss J (2022). Plastics degradation by hydrolytic enzymes: the plastics-active enzymes database — PAZy. *Proteins* 90, 1443–1456. | `10.1002/prot.26325` | **Cite for the positive set** — PAZy supplies 340 of 388 positives. Also derives PET-active sequence motifs by profile HMM over ~3,000 homologs, which is the published precedent for the unimplemented HMMER baseline (Still open #3). |

**Two points in [R4] worth carrying into the write-up.** It reports fewer than 50
experimentally verified plastics-active enzymes at publication, which is the cleanest external
statement of how small this field's ground truth is and why a homology-dense positive set is
unavoidable rather than careless. And its profile-HMM approach over PET-active homologs is
prior art for the HMMER baseline — worth reading before implementing it, since matching their
protocol makes the comparison interpretable.

**Still uncited.** No reference yet for: the α/β hydrolase fold definition behind `IPR029058`;
the catalytic mechanism of peptidyl-tRNA hydrolase and D-aminoacyl-tRNA deacylase asserted in
§22c (currently resting on UniProt's own `ACT_SITE` annotations, which is adequate but is not
a citation); PHA depolymerase lineage distinctness; and the identity-threshold choice behind
the 40% split ceiling.

---

## Environment notes

- **Installed and working** (verified 2026-09-28): `mmseqs`, `blastp`, `makeblastdb`
  (all `/opt/homebrew/bin`).
- **Not installed**: `mafft`, `hmmbuild`/`hmmsearch`/`hmmscan` (blocks the HMMER baseline),
  `cd-hit`, `diamond`.
- InterPLM SAE availability confirmed live on HuggingFace:
  `Elana/InterPLM-esm2-8m` layers 1-6 (320 dim -> 10,240 features);
  `Elana/InterPLM-esm2-650m` layers 1, 9, 18, 24, 30, 33 (1,280 dim -> 10,240 features).
  Each layer ships `ae_normalized.pt` and `ae_unnormalized.pt`.
- `interplm`'s installed package is missing its `train` subpackage, so `load_sae_from_hf`
  cannot work; load via `ReLUSAE.from_pretrained(hf_hub_download(...))`.
- Apple Silicon: SAE + ESM-2 run on `mps`. GPU only matters for embedding extraction, and
  all needed embeddings are already cached under `data/sae_650M_L33/`.
