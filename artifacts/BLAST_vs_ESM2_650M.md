# BLASTp vs ESM-2 650M Probe (layer 33)

> **Task framing: PET-only (superseded).** This document evaluates the original
> **PET vs. everything else** classifier. The project's headline task was later redefined to
> **polyester degrader vs. non-degrader** (any of PET, PBAT, aliphatic polyester), which
> reclassifies cutinases as positives and is the framing all current results use.
> Current equivalents: `EDA_polyester_task.md`, `ESM2_650M_Polyester_Run.md`,
> `BLAST_vs_ESM2_650M_Polyester.md`. Numbers below remain valid *for the PET-only task* and
> are retained as the record of how the task came to be redefined.

Generated 2026-09-28. Companion to `ESM2_650M_Dense_Run.md`, and the 650M counterpart of
`BLAST_vs_ESM2_8M.md`. Same split, same test sequences, matched tuning protocol.

## Protocol

| | BLASTp | ESM-2 650M probe |
|---|---|---|
| tool / model | `blastp` 2.17.0+ | `facebook/esm2_t33_650M_UR50D`, max-pooled layer 33 |
| reference set | 257 **training PET sequences only** | trained on 588 train sequences |
| score | best e-value of any hit | P(PET) from L2 logistic regression |
| tuned hyperparameter | e-value cutoff | C, eta0, decision threshold |
| tuned on | val (F1) | val (AU-PRC for C/eta0, F1 for threshold) |
| applied to test | unchanged | unchanged |

Both methods tune on val and apply to test unchanged. The BLAST database holds only
training PET sequences; including val/test sequences or negatives would leak.

---

## Table 1 — Test set, matched tuning

| method | Precision | Recall | F1 | AU-ROC | AU-PRC | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| **BLASTp @ e-value 3.16e-02** | **0.828** | **0.774** | **0.800** | 0.8778 | **0.8500** | 24 | 5 | 7 | 68 |
| ESM-2 650M `max_L33` | 0.742 | 0.742 | 0.742 | **0.8856** | 0.8150 | 23 | 8 | 8 | 65 |
| ESM-2 8M `max_L6` (reference) | 0.600 | 0.968 | 0.741 | 0.9015 | 0.7899 | 30 | 20 | 1 | 53 |

**BLASTp wins on F1 (0.800 vs 0.742), precision and AU-PRC.** 650M edges it only on
AU-ROC. Scaling the backbone 80x did not close the gap to sequence homology search.

### Is that gap real? Paired bootstrap test

The two methods are compared on identical resamples so the difference is measured
directly rather than by eyeballing two separate intervals.

**What is resampled.** The 104 test sequences fall into **66 homology components** --
clusters of sequences >=40% identical, which were forced into the same split for that
reason. Component sizes run from 8 down to 1 (`BPS0038` holds 8 sequences, 7 of them PET).
Each bootstrap draw picks **66 components with replacement**, so in a typical draw ~25 are
absent entirely, ~24 appear once, and ~17 appear two or more times. The sequences of the
drawn components are pooled into a pseudo-test-set (~104-120 sequences), both methods are
scored on it, and the *difference* is recorded. Repeat 4,000 times; the CI is the
2.5th/97.5th percentiles of those differences.

Components rather than sequences, because the 8 members of `BPS0038` are one enzyme family
observed 8 times, not 8 independent observations. Resampling sequences would treat them as
independent and report an interval far narrower than is honest -- the same reasoning that
governs the split itself.

| metric | delta (650M - BLAST) | 95% CI | significant? |
|---|---|---|---|
| AU-ROC | +0.0074 | [-0.0959, +0.1232] | **no** |
| AU-PRC | -0.0412 | [-0.1899, +0.0756] | **no** |

Both intervals cross zero, so **the two methods are statistically indistinguishable on
threshold-free ranking.** BLAST is ahead on AU-PRC in 75% of resamples
(P(650M > BLAST) = 0.25), which is a consistent direction but not a resolvable difference
at n=104 with 31 positives.

The pairing is doing real work here. Per-method intervals are wide and overlap almost
completely:

| | point estimate | 95% CI | width |
|---|---|---|---|
| ESM-2 650M | 0.8150 | [0.613, 0.922] | 0.309 |
| BLASTp | 0.8500 | [0.699, 0.945] | 0.245 |
| **paired delta** | **-0.0412** | **[-0.190, +0.076]** | **0.266** |

Across resamples the two scores correlate at **r = 0.610** -- when a draw contains the hard
PETases, both methods drop together. That correlation reduces the standard deviation of the
difference to 0.066, against the 0.103 it would be if the methods were independent: a **36%
noise reduction** on the comparison. Comparing the two separate CIs would discard that and
make the comparison look hopeless rather than merely underpowered.

**What "not significant" does and does not mean here.** It does not mean the methods are
equivalent -- BLAST's F1 advantage (0.800 vs 0.742) and its 5-vs-8 false positives are real
differences in this test set. It means this test set is too small to establish that the
ordering would hold on a fresh sample of PET enzymes. The same verdict applies to the
650M-vs-8M comparison (dAU-PRC +0.0285, CI [-0.0833, +0.1312]), so **no pairwise ranking
among the three methods is statistically supported.** The claims that survive are the
error-profile differences in Tables 2 and 3, which are counts rather than estimates.

---

## Table 2 — False positives by negative tier

| tier | n_neg | ESM-2 650M | BLASTp | ESM-2 8M |
|---|---|---|---|---|
| 2a_PBAT | 5 | 1/5 | 1/5 | 2/5 |
| 2b_aliphatic | 6 | 2/6 | 3/6 | 6/6 |
| 3_fold_matched_esterase | 26 | 2/26 | 0/26 | 10/26 |
| 4_naive_control | 36 | 3/36 | 1/36 | 2/36 |
| **total** | 73 | **8** | **5** | **20** |

| | class-3 FP, Actinomycetota | class-3 FP, non-Actinomycetota |
|---|---|---|
| ESM-2 650M | 1/11 | 1/15 |
| ESM-2 8M | **9/11** | 1/15 |
| BLASTp | 0/11 | 0/15 |

**The taxonomic shortcut is gone.** The 8M probe made 9 of its 11 Actinomycetota class-3
errors -- an 82% false-positive rate on that group against 7% elsewhere. 650M makes 1/11
and 1/15: equal rates. Total false positives fall from 20 to 8, approaching BLAST's 5.

This confirms the conclusion of `BLAST_vs_ESM2_8M.md` Table 5b, where phylum reweighting
and Actinomycetota subsampling both failed to move the 8M error count (10/26 in all three
variants). The problem was the representation, not the label balance.

### Per-tier AU-ROC

| tier | n_neg | ESM-2 650M | BLASTp |
|---|---|---|---|
| 2a_PBAT | 5 | 0.7419 | **0.8000** |
| 2b_aliphatic | 6 | 0.6828 | **0.6989** |
| 3_fold_matched_esterase | 26 | **0.9218** | 0.8939 |
| 4_naive_control | 36 | **0.9131** | 0.9068 |

---

## Table 3 — Error analysis: the 8 false positives named

Decision threshold 0.0030, carried from val.

| accession | score | tier | phylum | BLAST e-value | protein / substrates |
|---|---|---|---|---|---|
| BPS0038 | 0.776 | 2b_aliphatic | Pseudomonadota | 1e-17 | HP PBS\|PHA |
| O33363 | 0.008 | 3_fold_matched_esterase | Actinomycetota | 2.6 | GDSL lipase Rv0518 (EC 3.1.1.-)  |
| A0QWG6 | 0.008 | 4_naive_control | Actinomycetota | 0.48 | Phosphatidyl-myo-inositol mannosyl  |
| P9WGK9 | 0.006 | 4_naive_control | Actinomycetota | 0.33 | Sensor histidine kinase MtrB (EC 2  |
| BPS0062 | 0.005 | 2a_PBAT | Bacillota | 0.005 | JW45_1534 PBAT\|PHA\|PLA\|PUR |
| BPS0025 | 0.005 | 2b_aliphatic | - | 0.009 | Lipase PCL\|PES\|PHBV\|PHO |
| P22637 | 0.004 | 4_naive_control | Actinomycetota | 0.17 | Cholesterol oxidase (CHOD) (EC 1.1  |
| O32232 | 0.004 | 3_fold_matched_esterase | Bacillota | 0.067 | Carboxylesterase (EC 3.1.1.1)  |

### These are marginal calls, not confident errors

**Seven of the eight score between 0.004 and 0.008 against a threshold of 0.0030** -- they sit
on the decision boundary. Contrast the 8M run, where the ten class-3 false positives scored
0.305-0.512 against a 0.2223 threshold, i.e. the model was confidently wrong about them.
650M is wrong marginally, which is a materially better failure mode: a small threshold
increase removes most of these errors, whereas 8M's could not be thresholded away without
losing true positives.

### The one confident error is arguably not an error

`BPS0038` scores **0.776**, two orders of magnitude above the rest, and has a BLAST e-value
of **1e-17** against the training PET set -- a genuine, strong homolog. It is labelled
`2b_aliphatic` because PlasticDB records it as a PBS/PHA degrader. BLAST flags it too.

Both databases record only *confirmed positives*, so "not PET-active" is an inference from
absence of annotation, never an assay result (see `dataset_splits_0.4.md` section 8.7). A
sequence this similar to known PETases, from an enzyme assayed on other polyesters, is
precisely the case where the negative label is least trustworthy. It may be a correct
prediction against a wrong label.

Three of the remaining seven are `4_naive_control` proteins that are not esterases at all
(a histidine kinase, a cholesterol oxidase, a mannosyltransferase). Those are real errors,
but all three score under 0.007.

---

## What this means for the project

| | 8M `max_L6` | 650M `max_L33` | BLASTp |
|---|---|---|---|
| F1 | 0.741 | 0.737 | **0.800** |
| total FP | 20 | 8 | **5** |
| class-3 FP (Actino) | 9/11 | **1/11** | **0/11** |
| failure mode | confidently wrong | marginally wrong | -- |

**Scale fixed the failure mode without improving the score.** The errors converted rather
than vanished: 8M caught 30 of 31 with 20 false positives, 650M catches 23 with 8. An 80x
parameter increase bought a better-behaved model, not a better-performing one, and neither
beats a properly tuned BLAST.

### The tier where everything still agrees

| method | 2b_aliphatic AU-ROC |
|---|---|
| ESM-2 8M `max_L6` | 0.6398 |
| ESM-2 650M `max_L33` | 0.6828 |
| BLASTp | 0.6989 |

Two model scales 80x apart and a sequence-alignment method all land near 0.68 on
distinguishing PET degraders from other polyester degraders. That agreement across
fundamentally different approaches is the strongest evidence that **this contrast, not fold
recognition, is the unsolved problem** -- and it is unchanged by scale.

Which is the case for the SAE work. Dense dimensions are polysemantic, so "not linearly
decodable from dense activations at either scale" does not imply the feature is absent. A
sparse decomposition may isolate what a dense readout smears. That was the project's
original hypothesis; it now has two scales of measured evidence behind the question rather
than an assumption.

## Reproduce

```
brew install blast
uv run python -m biopet_sae.embed_esm2 --model 650M
uv run python -m biopet_sae.blast_baseline
uv run python -m biopet_sae.train_binary_probe \
    --embeddings data/embeddings_650M/embeddings.npz \
    --splits data/processed/dataset_splits_id40.tsv --C 1e-1 --eta0 1e-3
```

| file | contents |
|---|---|
| `artifacts/BLAST_vs_ESM2_650M.json` | every number in this document, machine-readable |
| `artifacts/ESM2_650M_Dense_Run.md` | the 650M probe being compared |
| `artifacts/BLAST_vs_ESM2_8M.md` | the 8M counterpart, incl. the rebalancing experiment |
