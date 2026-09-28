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

## Current status (checkpoint, 2026-09-28)

Dataset built and verified; dense-embedding baselines complete at two model scales with a
BLAST comparison. **Test is spent** — further tuning must return to val or use a fresh set.
No SAE work yet.

**Assembled set: 847 sequences.** 320 `1_pet`, 68 `2_other_polyester`, 185
`3_fold_matched_esterase`, 238 `4_naive_control`, 36 `heldout_pha`, 118 excluded.

**Headline: polyester-degrader classifier, test AU-ROC 0.9681 / AU-PRC 0.9526, beating
BLASTp significantly on both (paired bootstrap over homology components). Generalises to a
held-out enzyme lineage at 83%.** See section 14.

**Best single interpretable feature: SAE latent 9529 alone, AU-ROC 0.9160 on the PET task** — a catalytic-serine detector, above
dense (0.8856), the full SAE probe (0.8900) and BLASTp (0.8778). **Best test F1: BLASTp
0.800.** No pairwise difference between methods is statistically significant.

**Main scientific result (section 13):** the model's best feature is an interpretable
nucleophile-elbow detector. It detects the alpha/beta-hydrolase fold, not PET specificity,
which explains the ~0.68 ceiling on the `2b_aliphatic` contrast across every model scale
and BLAST.

**Artifacts** (all numbers reproducible, each with a machine-readable `.json`):

| file | contents |
|---|---|
| `artifacts/dataset_splits_0.4.md` | dataset provenance, class rationale, split verification |
| `artifacts/ESM2_8M_Dense_Run.md` | 8M: C×eta sweep, 5-seed SEs, threshold curve, test |
| `artifacts/ESM2_650M_Dense_Run.md` | 650M layer 33: same four tables |
| `artifacts/BLAST_vs_ESM2_8M.md` | BLAST baseline, e-value sweep, 8M error analysis, rebalancing |
| `artifacts/BLAST_vs_ESM2_650M.md` | BLAST vs 650M, paired bootstrap method, 650M error analysis |
| `artifacts/SAE_Feature_Interpretation.md` | SAE latents, catalytic-triad mapping |
| `artifacts/EDA_polyester_task.md` | **polyester labels, confound floors** |
| `artifacts/ESM2_650M_Polyester_Run.md` | **the headline classifier** |
| `artifacts/BLAST_vs_ESM2_650M_Polyester.md` | **the significant win over BLAST** |
| `https://claude.ai/artifact/KcdU4xp1QUSvuWzrL4pGwS` | 3D structure viewer, 32 PETases |
| `https://claude.ai/artifact/S1rzDp1cJYxA39xSTPZteE` | **PR curves, val vs test, vs BLASTp** |
| `artifacts/pr_curves.html` | source for the curve figure |

**Data artifacts:**

| file | contents |
|---|---|
| `data/processed/dataset_splits_id40.tsv` | 847 rows: class, split, component |
| `data/processed/pair_identities.tsv` | 70,230 cached global alignments |
| `data/embeddings_8M/embeddings.npz` | layers 1–6, mean + max pooled |
| `data/embeddings_650M/embeddings.npz` | layers 1/9/18/24/30/33, mean + max pooled |
| `results/eda_report.md`, `results/split_frontier.json` | EDA, threshold sweep |

**Stale and pending deletion:** `results/probe_embeddings_8M/` (0.30 split) and
`results/binary_embeddings_8M_dataset_splits_id40/` (C=1.0, pre-regularization-fix). Both
superseded by the artifacts above.

## Still open

1. **SAE work — the main remaining question.** Both dense scales and BLAST fail together on
   `2b_aliphatic` (0.64 / 0.68 / 0.70). Dense dimensions are polysemantic, so this does not
   establish the feature is absent — only that it is not linearly decodable from dense
   activations. InterPLM SAEs exist for the exact layers already embedded
   (8M layers 1–6; 650M layers 1/9/18/24/30/33), and the pooled tensors are cached.
2. **HMMER is still unimplemented.** A profile HMM over the 257 training positives would be
   a stronger baseline than pairwise BLAST, and BLAST already matches or beats both ESM-2
   scales. Needs `mafft` + `hmmbuild`/`hmmsearch`.
3. **PEZy-miner's 36 assayed non-degraders** remain unextracted from
   `~/Downloads/1-s2.0-S2214030124000178-mmc1.docx` — the only known source of genuinely
   verified negatives, which neither PAZy nor PlasticDB provides.
4. **Framing for write-up.** The defensible claim is now narrow and mostly negative: dense
   ESM-2 embeddings do not outperform homology search on this benchmark at either scale.
   That is a legitimate result, and it sets up the SAE question rather than answering it.

## Next steps

1. Load InterPLM SAEs (`ReLUSAE.from_pretrained` + `hf_hub_download`; the `train` subpackage
   is missing upstream so `load_sae_from_hf` cannot work) and extract sparse features at
   650M layer 33, where the dense probe performed best.
2. Gate on SAE reconstruction fidelity before trusting any feature result — the kinase
   project's precedent. Compare against a **reconstruction baseline, not a raw baseline**.
3. Re-run the tiered evaluation on SAE features, with `2b_aliphatic` as the target contrast
   rather than the overall AU-PRC.
4. Map any discriminative features to residues and check against the structural determinants
   (W185 wobble, Y87/W185 aromatic clamp, DS1 disulfide) using the 32 PET entries with PDB
   structures. The catalytic triad, shared with the negatives, is the built-in control.
5. Optional: HMMER baseline; PEZy-miner verified negatives.

## Environment notes

- `mmseqs`, `blastp`, `makeblastdb`, `cd-hit`, `hmmscan`, `diamond` are all **not
  installed** on this machine as of 2026-09-27.
- InterPLM SAE availability confirmed live on HuggingFace:
  `Elana/InterPLM-esm2-8m` layers 1–6 (320 dim → 10,240 features);
  `Elana/InterPLM-esm2-650m` layers 1, 9, 18, 24, 30, 33 (1,280 dim → 10,240 features).
  Each layer ships `ae_normalized.pt` and `ae_unnormalized.pt`.
