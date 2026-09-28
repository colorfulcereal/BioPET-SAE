# BLASTp vs ESM-2 8M Probe

> **See also `BLAST_vs_ESM2_650M.md`.** The 650M run resolves the
> phylum-shortcut failure documented in Table 5 here: Actinomycetota class-3 false
> positives fall from 9/11 to 1/11 and total false positives from 20 to 8, without
> improving F1.

Generated 2026-09-27. Companion to `ESM2_8M_Dense_Run.md`. Same split, same val/test
sequences, matched tuning protocol, so the two methods are directly comparable.

## Protocol

| | BLASTp | ESM-2 8M probe |
|---|---|---|
| tool / model | `blastp` 2.17.0+ (BLAST+, arm64) | `facebook/esm2_t6_8M_UR50D`, max-pooled layer 6 |
| reference set | 257 **training PET sequences only** | trained on 588 train sequences |
| score | best e-value of any hit | P(PET) from L2 logistic regression |
| tuned hyperparameter | e-value cutoff | decision threshold |
| tuned on | val (F1) | val (F1) |
| applied to test | unchanged | unchanged |

**Both methods tune exactly one hyperparameter on val and apply it unchanged to test.**
This matters: an earlier version of this comparison let BLAST optimise its cutoff on test
itself, which inflated its F1 by 0.22 and produced the wrong conclusion. The BLAST
database contains only training PET sequences -- including val/test sequences or negatives
would leak.

---

## Table 1 — e-value tuned on val (43-point sweep, F1 criterion)

| e-value | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 1.00e+01 | 0.291 | 1.000 | 0.451 | 32 | 78 | 0 |
| 1.00e+00 | 0.527 | 0.906 | 0.667 | 29 | 26 | 3 |
| 3.16e-01 | 0.641 | 0.781 | 0.704 | 25 | 14 | 7 |
| 1.00e-01 | 0.759 | 0.688 | 0.721 | 22 | 7 | 10 |
| **3.16e-02** | **0.808** | **0.656** | **0.724** | 21 | 5 | 11 |  <- val optimum
| 1.00e-02 | 0.833 | 0.625 | 0.714 | 20 | 4 | 12 |
| 1.00e-03 | 0.842 | 0.500 | 0.627 | 16 | 3 | 16 |
| 1.00e-05 | 0.867 | 0.406 | 0.553 | 13 | 2 | 19 |
| 1.00e-10 | 0.778 | 0.219 | 0.341 | 7 | 2 | 25 |
| 1.00e-20 | 0.667 | 0.125 | 0.211 | 4 | 2 | 28 |

**Val-selected e-value = 3.16e-02** (val F1 0.724).

The sweep shows the plan's specified cutoff of 1e-5 is a **poor operating point**: it
reaches val F1 0.553, against 0.724 two orders of magnitude looser. 1e-5 trades away
recall for precision the task does not need.

### What an e-value is, and why the cutoff matters so much

An e-value is the number of alignments this good you would expect **by chance** in a
database of this size. e-value 1 means one such hit is expected at random (meaningless);
1e-5 means one in 100,000 (probably real homology). Lower is better.

Two consequences for reading these numbers:

1. **E-value scales with database size** (E ~ m*n*2^-S). Our database holds only
   257 sequences. Against all of UniProt (~250M) every e-value here would be
   roughly six orders of magnitude worse, so **these BLAST results are optimistic**
   relative to a real discovery scenario.
2. **"BLAST found a hit" is meaningless without a cutoff.** At e-value 10, BLAST hits
   31/31 test positives -- and also 65/73 negatives. Any claim about BLAST's recall must
   state the cutoff.

---

## Table 2 — Test set, matched tuning

| method | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| **BLASTp @ e-value 3.16e-02 (val-tuned)** | **0.828** | 0.774 | **0.800** | 24 | 5 | 7 | 68 |
| BLASTp @ e-value 1e-05 (plan spec) | 0.947 | 0.581 | 0.720 | 18 | 1 | 13 | 72 |
| ESM-2 8M probe @ 0.2223 (val-tuned) | 0.600 | **0.968** | 0.741 | 30 | 20 | 1 | 53 |

Threshold-free metrics (unaffected by tuning):

| | AU-ROC | AU-PRC |
|---|---|---|
| BLASTp | 0.8778 | **0.8500** |
| probe | **0.9015** | 0.7899 |

Paired bootstrap over homology components, delta = probe - BLAST:

| metric | delta | 95% CI | significant? |
|---|---|---|---|
| AU-ROC | +0.0226 | [-0.0533, +0.1181] | no |
| AU-PRC | -0.0697 | [-0.2008, +0.0280] | no |

**Neither threshold-free difference is significant** -- both intervals cross zero at
n=104 with 31 positives. The methods are statistically indistinguishable on ranking.

The test resamples **whole homology components** (66 of them across the 104 test
sequences), not individual sequences, because members of a component are >=40% identical
and are one family observed several times rather than independent observations. Both
methods are scored on the same resample and the difference recorded, which exploits their
correlated errors (r = 0.61) to cut comparison noise by ~36% relative to comparing two
separate intervals. See `BLAST_vs_ESM2_650M.md` Table 1 for the full method and the
per-method-versus-paired interval comparison.

"Not significant" here does not mean equivalent: BLAST's F1 advantage and its lower false
positive count are real differences in this test set. It means the set is too small to
establish that the ordering would hold on a fresh sample of PET enzymes.

### Headline finding

With matched tuning, **BLASTp reaches F1 0.800 against the probe's 0.741**, with
much better precision (0.828 vs 0.600) -- 5 false positives against 20.
The probe holds a clear recall advantage (0.968 vs 0.774): 30 of 31 recovered
against 24.

**A linear probe on dense ESM-2 8M embeddings does not beat sequence homology search on
this benchmark.** It trades a large precision loss for a recall gain and loses on F1.

---

## Table 3 — False positives by negative tier (test)

| method | 2a_PBAT | 2b_aliphatic | 3_fold_matched_esterase | 4_naive_control |
|---|---|---|---|---|
| BLASTp @ 3.2e-02 | 1/5 | 3/6 | 0/26 | 1/36 |
| BLASTp @ 1e-05 | 0/5 | 1/6 | 0/26 | 0/36 |
| ESM-2 probe | 2/5 | 6/6 | 10/26 | 2/36 |

**This column explains the entire precision gap.** BLAST makes **zero** false positives on
the 26 fold-matched UniProt esterases at either cutoff. The probe flags **10 of 26** --
half its false positives, and the whole of its precision deficit.

So the intuition that the probe handles hard negatives better is wrong. BLAST rejects
UniProt esterases cleanly and the probe cannot. Both flag the `2b_aliphatic` polyester
degraders (3/6 vs 6/6), which is the genuinely hard tier for both methods -- consistent
with PET specificity, not fold recognition, being the unsolved part.

### Per-tier AU-ROC (threshold-free)

| tier | n_neg | probe | BLASTp |
|---|---|---|---|
| 2a_PBAT | 5 | **0.9032** | 0.8000 |
| 2b_aliphatic | 6 | 0.6398 | **0.6989** |
| 3_fold_matched_esterase | 26 | 0.8437 | **0.8939** |
| 4_naive_control | 36 | **0.9866** | 0.9068 |

These per-tier differences were tested by paired bootstrap and **none reached
significance** -- the 2a and 2b tiers have 5 and 6 negatives respectively, far too few to
resolve a difference. They are reported as directions, not results.

---

## Table 4 — Test e-value sweep, for reference

| e-value | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 1.00e+00 | 0.400 | 0.903 | 0.554 | 28 | 42 | 3 |
| 1.00e-01 | 0.735 | 0.806 | 0.769 | 25 | 9 | 6 |
| 3.16e-02 | 0.828 | 0.774 | 0.800 | 24 | 5 | 7 |
| 1.00e-02 | 0.852 | 0.742 | 0.793 | 23 | 4 | 8 |
| 1.00e-03 | 0.952 | 0.645 | 0.769 | 20 | 1 | 11 |
| 1.00e-05 | 0.947 | 0.581 | 0.720 | 18 | 1 | 13 |
| 1.00e-10 | 0.929 | 0.419 | 0.578 | 13 | 1 | 18 |
| 1.00e-20 | 1.000 | 0.194 | 0.324 | 6 | 0 | 25 |

Shown for completeness. **Selecting from this table would be cheating** -- the reported
BLAST configuration is the val-selected one. Note its test optimum (F1 0.793 at 1e-2)
is close to the val-selected result (0.800 at 3.2e-2), so val tuning transferred well.

---

## Table 5 — Error analysis: where the probe's 20 false positives come from

The probe's false positives on test break down as: 2/5 `2a_PBAT`, 6/6 `2b_aliphatic`,
**10/26 `3_fold_matched_esterase`**, 2/36 `4_naive_control`. The 10 class-3 errors are the
largest group and the whole of the precision gap against BLAST, so they are worth naming.

| accession | score | EC | organism | phylum | protein |
|---|---|---|---|---|---|
| Q9L9D7 | 0.512 | 3.1.1.84 | Rhodococcus sp. (strain MB1 Bres | Actinomycetota | Cocaine esterase (EC 3.1.1.84) |
| P9WHR4 | 0.493 | 3.1.1.- | Mycobacterium tuberculosis (stra | Actinomycetota | Carboxylesterase B (EC 3.1.1.-) |
| I6Y2J4 | 0.445 | 3.1.1.3 | Mycobacterium tuberculosis (stra | Actinomycetota | Triacylglycerol lipase (EC 3.1.1.3) (Esteras |
| O32232 | 0.431 | 3.1.1.1 | Bacillus subtilis (strain 168) | Bacillota | Carboxylesterase (EC 3.1.1.1) |
| Q01470 | 0.393 | 3.1.1.- | Pseudarthrobacter oxydans (Arthr | Actinomycetota | Phenmedipham hydrolase (EC 3.1.1.-) (Phenylc |
| P95125 | 0.363 | 3.1.1.- | Mycobacterium tuberculosis (stra | Actinomycetota | Carboxylic ester hydrolase LipN (EC 3.1.1.-) |
| P9WK86 | 0.329 | 3.1.1.1 | Mycobacterium tuberculosis (stra | Actinomycetota | Carboxylesterase NlhH (EC 3.1.1.1) |
| Q47M62 | 0.315 | 3.1.1.1 | Thermobifida fusca (strain YX) | Actinomycetota | Carboxylesterase (EC 3.1.1.1) |
| P86325 | 0.314 | 3.1.1.1 | Thermobifida fusca (Thermomonosp | Actinomycetota | Carboxylesterase (EC 3.1.1.1) |
| P71668 | 0.305 | 3.1.1.- | Mycobacterium tuberculosis (stra | Actinomycetota | Esterase LipI (EC 3.1.1.-) |

### This is the phylum confound, measured directly

| class-3 esterases (test) | FP | TN | FP rate |
|---|---|---|---|
| Actinomycetota | 9 | 2 | 82% |
| Bacillota | 1 | 0 | 100% |
| Pseudomonadota | 0 | 9 | 0% |
| Bacteroidota | 0 | 3 | 0% |
| Basidiomycota | 0 | 1 | 0% |
| Ascomycota | 0 | 1 | 0% |

**9 of the 10 false positives are Actinomycetota. All 9 Pseudomonadota esterases were
correctly rejected.**

This is exactly what decision D6 predicted. The PET training class is 77% Actinomycetota,
so the probe learned a taxonomic signal alongside -- or instead of -- a functional one. The
planned within-Actinomycetota control (D6.2) could not be run because val held only 4
Actinomycetota PET sequences; this error analysis answers the same question more directly.

**Two of the false positives are *Thermobifida fusca* carboxylesterases** (Q47M62,
P86325). *T. fusca* is the canonical PETase-producing organism -- TfCut and TfH are in the
training positives. So the probe flags non-PET esterases from the very organism whose
PETases it was trained on. That is organism recognition, not function recognition.

### A second pattern: generic esterases are flagged, specialised ones rejected

| EC | name | FP | TN |
|---|---|---|---|
| 3.1.1.- | unspecified | 4 | 2 |
| 3.1.1.1 | carboxylesterase | 4 | 0 |
| 3.1.1.3 | triacylglycerol lipase | 1 | 0 |
| 3.1.1.84 | cocaine esterase | 1 | 0 |
| 3.1.1.29 | aminoacyl-tRNA hydrolase | 0 | 3 |
| 3.1.1.61 | protein-glutamate methylesterase | 0 | 1 |
| 3.1.1.106 | specialised | 0 | 3 |
| 3.1.1.32 | phospholipase A1 | 0 | 5 |
| 3.1.1.2 | arylesterase | 0 | 1 |
| 3.1.1.5 | lysophospholipase | 0 | 1 |

Every false positive is a **generic** carboxylesterase (3.1.1.1) or an unspecified
3.1.1.-, while the correctly rejected sequences carry **specific** EC numbers for narrow
activities. Biologically sensible -- generic carboxylesterases are the nearest functional
neighbours of cutinases -- but it confirms the probe is detecting "broad-specificity
alpha/beta-hydrolase esterase", which is precisely the generic-hydrolase shortcut the
four-class design was built to expose.

The separation is systematic rather than marginal: mean probe score 0.390 for the
false positives against 0.088 for the correctly rejected.

### Why BLAST does not make this mistake

BLAST scores alignment against the actual PET sequences, so a *T. fusca* carboxylesterase
that is not homologous to a *T. fusca* cutinase receives no significant hit. BLAST has no
mechanism for encoding "this organism makes PETases". The probe does, and uses it. That
asymmetry -- not model capacity -- is why BLAST makes 0/26 class-3 errors and the probe
makes 10/26.

### Table 5b — Phylum rebalancing was tested and does not fix it

The obvious remedy -- match the phylum distribution between positives and negatives -- was
run. Training positives are 76.7% Actinomycetota against the negatives' 58.9%, a 17.7-point
gap, with P(PET | Actinomycetota) = 0.503 versus P(PET | not Actinomycetota) = 0.306.

Three variants, each with the decision threshold re-tuned on val and applied to test:

| variant | n_train | PET | Precision | Recall | F1 | FP total | FP class-3 | of which Actino |
|---|---|---|---|---|---|---|---|---|
| A. baseline, no correction | 588 | 257 | 0.600 | 0.968 | 0.741 | 20 | **10/26** | 9/11 |
| B. phylum reweighting (all data kept) | 588 | 257 | 0.625 | 0.968 | 0.759 | 18 | **10/26** | 9/11 |
| C. subsample Actino positives | 477 | 146 | 0.596 | 0.903 | 0.718 | 19 | **10/26** | 9/11 |

**The class-3 error count is identical in all three -- 10/26, with 9 of 11 Actinomycetota.**
Reweighting removed only the two class-4 errors. Subsampling cost 111 positives and lowered
recall to 0.903 while fixing nothing.

### And the errors are not genuine PETase relatives either

The competing explanation -- that these esterases really are PETase-like and the "negative"
label is simply unverified -- was tested by BLASTing each against the training PET set:

| group | n | median best e-value | min | with e-value <= 1 |
|---|---|---|---|---|
| class-3 false positives | 10 | 0.47 | 0.067 | 7 |
| class-3 correctly rejected | 16 | 3.3 | 0.051 | 6 |
| PET positives (reference) | 31 | **1.7e-07** | 5.1e-45 | 28 |

The false positives' best alignments are chance-level (e-value 0.067-1.9), six orders of
magnitude worse than real positives. `P37446`, which the probe correctly rejected, has a
*better* e-value (0.051) than 9 of the 10 false positives. Percent identity overlaps
completely (FP 24-48%, rejected 23-48%). So these are not relatives the probe was right
about.

### Implication (revised)

Both candidate fixes fail, and that locates the problem in the representation rather than
in the data balance or the labels.

The probe never sees a phylum label -- it sees embedding dimensions. The compositional
signature correlated with Actinomycetota (the GC-driven Ala/Pro/Arg/Thr enrichment
quantified in `results/eda_report.md`) is *inside the features*. Reweighting the loss moves
the decision boundary; it cannot delete a feature. Given this representation, that
compositional signal remains the most discriminative thing available, so the probe keeps
using it.

**Conclusion: max-pooled ESM-2 8M layer 6 does not contain a linearly-decodable
PET-specificity feature.** It carries compositional and taxonomic information, which is
what a linear readout extracts. No amount of label rebalancing can create a feature that
is absent from the representation.

Three things follow, in increasing cost:

1. **Report stratified by phylum** rather than claiming a fix. The honest statement is that
   the probe achieves 0/15 false positives on non-Actinomycetota esterases and 9/11 on
   Actinomycetota ones.
2. **Try a larger backbone.** ESM-2 650M (embeddings already computed, layers 1/9/18/24/30/33)
   may carry the feature where 8M does not. This is cheap and untested.
3. **Try SAE features.** Dense dimensions are polysemantic, so "not linearly decodable from
   dense activations" does not imply "not present". A sparse decomposition may isolate what
   the dense readout smears -- which is the project's original hypothesis, now with a
   measured reason to test it.

---

## What this means for the project

The original plan targeted **">= 40% higher recall than BLASTp (e-value < 1e-5)"**.
Against that specific baseline the target is met: recall 0.968 vs 0.581, a
**67% relative improvement**. But 1e-5 is a weak baseline, and against a
val-tuned BLAST the probe's recall advantage narrows to 0.968 vs 0.774 while F1 reverses.

The defensible claims are narrow:

1. The probe recovers **30 of 31** PET enzymes against BLAST's 24. If missing a real
   plastizyme costs more than screening extra candidates -- which it does for wet-lab
   shortlisting -- the probe is the better tool for that purpose.
2. The probe assigns a score to every sequence, including the
   0 test sequences with no BLAST alignment at all. All 0 are true negatives,
   so this is not an accuracy advantage, only a coverage one.
3. Beyond that, **ESM-2 8M plus a linear probe does not outperform homology search.**

This is the strongest available argument for the SAE work. The open question is no longer
"can embeddings beat BLAST" -- measured, and largely no -- but **what representation would
have to exist for PET specificity to be separable at all**, which is an interpretability
question rather than a benchmarking one. The `2b_aliphatic` tier, where both methods fail
together, is where that question lives. See `dataset_splits_0.4.md` section 1 for why the
PBAT contrast is its sharpest form.

## Reproduce

```
brew install blast
uv run python -m biopet_sae.blast_baseline
```

| file | contents |
|---|---|
| `artifacts/BLAST_vs_ESM2_8M.json` | every number in this document, machine-readable |
| `artifacts/ESM2_8M_Dense_Run.md` | the probe being compared |
| `artifacts/dataset_splits_0.4.md` | dataset, split and class rationale |
