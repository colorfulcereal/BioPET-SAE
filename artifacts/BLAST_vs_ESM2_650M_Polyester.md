# BLASTp vs ESM-2 650M — Polyester Degrader Task

Generated 2026-09-28. Companion to `ESM2_650M_Polyester_Run.md`. Same split, same test
sequences, matched tuning protocol.

> **Headline: the probe beats BLASTp significantly on both ranking metrics.** This is the
> first task in the project where that holds — on the PET-only framing the two methods
> were statistically indistinguishable and BLAST led on AU-PRC.

## Protocol

| | BLASTp | ESM-2 650M probe |
|---|---|---|
| tool / model | `blastp` 2.17.0+ | `esm2_t33_650M_UR50D`, max-pooled layer 33 |
| reference | **303 training polyester degraders only** | trained on 588 train sequences |
| score | best e-value of any hit | P(polyester) from L2 logistic regression |
| tuned | e-value cutoff | C, eta0, decision threshold |
| tuned on | val (F1) | val (log-loss for C/eta0, F1 for threshold) |
| applied to test | unchanged | unchanged |

The BLAST database contains only training positives; including val/test sequences or
negatives would leak. Both methods tune on val and touch test once.

---

## Table 1 — Test set

| method | AU-ROC | AU-PRC | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| **ESM-2 650M `max_L33`** | **0.9681** | **0.9526** | 0.830 | 0.929 | 0.876 | 39 | 8 | 3 | 54 |
| BLASTp @ e-value 3.16e-02 | 0.9040 | 0.7552 | **0.833** | **0.952** | **0.889** | 40 | 8 | 2 | 54 |
| amino-acid composition (floor) | 0.7020 | 0.6737 | | | | | | | |

**At the operating point the two are equivalent** — identical false-positive counts (8),
BLAST one true positive ahead, F1 0.889 vs 0.876. **The difference is in ranking.**

## Table 1b — Validation and test side by side

The same models and thresholds, scored on both splits.

| | VAL probe | VAL BLASTp | delta | TEST probe | TEST BLASTp | delta |
|---|---|---|---|---|---|---|
| AU-ROC | 0.9976 | 0.9910 | +0.0066 | **0.9681** | 0.9040 | **+0.0641** |
| AU-PRC | 0.9954 | 0.9895 | +0.0059 | **0.9526** | 0.7552 | **+0.1974** |
| Precision | 0.9773 | 0.9545 | +0.0228 | 0.8298 | 0.8333 | −0.0035 |
| Recall | 1.0000 | 0.9767 | +0.0233 | 0.9286 | 0.9524 | −0.0238 |
| F1 | 0.9885 | 0.9655 | +0.0230 | 0.8764 | 0.8889 | −0.0125 |
| TP/FP/FN | 43/1/0 | 42/2/1 | | 39/8/3 | 40/8/2 | |

**On validation the two methods are nearly identical** (+0.006 AU-PRC). **On test the probe
leads by +0.197.** The gap widens on held-out data rather than shrinking, because BLASTp
degrades far more:

```
val -> test   BLASTp AU-PRC   0.9895 -> 0.7552   -0.234
              probe  AU-PRC   0.9954 -> 0.9526   -0.043
```

### Why BLASTp falls off and the probe does not

Not because test positives are more remote — they are actually *closer* to the database
(median best e-value 3.7e-15 on test against 1e-09 on val). **The difference is in the
negatives:**

| split | negatives | median best e-value | clearing e ≤ 3.16e-2 |
|---|---|---|---|
| val | 76 | 1.9 | **2/76** |
| test | 62 | 1.0 | **8/62** |

Test drew esterases markedly more homologous to the training polyester degraders. Both
methods make 8 false positives at their operating points — but BLASTp scatters those cases
among its top-ranked hits, collapsing AU-PRC, while the probe keeps them near its decision
boundary.

**This tempers the headline claim.** Part of the measured gap reflects which negatives
happened to land in test. The paired bootstrap resamples *within* test and so cannot account
for that. The defensible statement is **"the probe degrades more gracefully as negatives get
harder"**, not "the probe is uniformly better" — on validation the two are tied.

### Reading the shape of the BLASTp test curve

The test panel of the PR figure shows BLASTp as a **flat shelf pinned at 0.844 precision
from recall 0.05 through 0.95**, with no high-precision region at the left edge. That shape
is not an artefact — it is set by the top three ranks.

| rank | seq_id | truth | tier | best e-value | running precision |
|---|---|---|---|---|---|
| 1 | BPS0080 | positive | 1_pet | 4e-79 | 1.000 |
| **2** | **P86325** | **negative** | **3_fold_matched_esterase** | **1e-74** | 0.500 |
| **3** | **Q47M62** | **negative** | **3_fold_matched_esterase** | **3e-74** | 0.333 |
| 4 | BPS0079 | positive | 1_pet | 3e-74 | 0.500 |
| **5** | **Q01470** | **negative** | **3_fold_matched_esterase** | **4e-55** | 0.400 |

BLASTp's second- and third-strongest hits in the whole test split are non-degraders, at
e-values interleaved with genuine PET hydrolases. **Six of 62 test negatives beat BLASTp's
own median positive** (e ≤ 3.7e-15). Once a false positive occupies rank 2, precision can
never recover: tightening the cutoff removes true positives from the bottom of the ranking
long before it reaches these three, so every operating point carries the same contamination.

Where each method first errs:

| split | method | rank of 1st false positive | recall already reached | AU-PRC |
|---|---|---|---|---|
| val | BLASTp | 41 | 93.0% | 0.9895 |
| val | probe | 36 | 81.4% | 0.9954 |
| test | BLASTp | **2** | **2.4%** | 0.7552 |
| test | probe | 25 | 57.1% | 0.9526 |

On validation BLASTp accumulates 40 correct calls before its first mistake, which is why its
val curve is textbook-shaped. On test it errs immediately.

Precision attainable at fixed recall (monotone PR envelope):

| recall | val BLAST | val probe | test BLAST | test probe |
|---|---|---|---|---|
| 0.05 | 1.000 | 1.000 | **0.844** | 1.000 |
| 0.25 | 1.000 | 1.000 | **0.844** | 1.000 |
| 0.50 | 1.000 | 1.000 | **0.844** | 1.000 |
| 0.75 | 1.000 | 1.000 | **0.844** | 0.941 |
| 0.90 | 1.000 | 0.977 | **0.844** | 0.867 |
| 0.95 | 0.977 | 0.977 | 0.833 | 0.816 |
| 1.00 | 0.597 | 0.977 | 0.750 | 0.792 |

**Interpretation.** These six negatives are tier-3 fold-matched esterases that are genuine
close homologs of the training degraders. Sequence similarity is the only signal BLASTp has,
so it cannot rank them down — this is a ceiling of the method, not a tuning failure. The
probe scores four of the six correctly (0.187–0.313) and is caught by the same two hardest
cases (P86325 0.525, Q47M62 0.522), which is the honest limit of the embedding approach.

*Caveat, same as above:* this is six specific sequences in one split. It illustrates the
mechanism behind the AU-PRC gap; it does not by itself establish the gap's size.

---

## Table 1c — The precision question, properly posed

At the chosen operating point BLASTp's precision is 0.8333 against the probe's 0.8298. That
0.0035 is **one sequence**: both make exactly 8 false positives, and BLASTp catches one more
true positive, so its denominator is 48 rather than 47. It is arithmetic, not a difference
in error behaviour.

A single threshold is one arbitrary point on a curve. The trade-off is the real comparison.

**Recall at a required precision (test, 42 positives):**

| precision required | ESM-2 650M | BLASTp | enzymes returned |
|---|---|---|---|
| ≥ 95% | **59.5%** | 2.4% | 25 vs 1 |
| ≥ 90% | **78.6%** | 2.4% | 33 vs 1 |
| ≥ 85% | **92.9%** | 2.4% | 39 vs 1 |
| ≥ 80% | **97.6%** | 95.2% | 41 vs 40 |
| ≥ 75% | **100.0%** | 100.0% | 42 vs 42 |

**At 90% precision BLASTp returns one enzyme out of 42; the probe returns 33.** BLASTp
plateaus near 84% precision and cannot be pushed higher — its highest-scoring hits already
include esterases genuinely homologous to polyester degraders, so tightening the cutoff
removes true positives before it removes those.

**Precision at a required recall (test):**

| recall required | ESM-2 650M | BLASTp | delta |
|---|---|---|---|
| ≥ 100% | **0.792** | 0.750 | +0.042 |
| ≥ 95% | **0.816** | 0.833 | -0.017 |
| ≥ 90% | **0.867** | 0.844 | +0.022 |
| ≥ 85% | **0.867** | 0.844 | +0.022 |
| ≥ 80% | **0.872** | 0.844 | +0.027 |
| ≥ 70% | **0.941** | 0.844 | +0.097 |
| ≥ 60% | **0.941** | 0.844 | +0.097 |
| ≥ 50% | **1.000** | 0.844 | +0.156 |

The probe leads at every recall level except 0.95, where it trails by 0.017.

**Curves:** `https://claude.ai/artifact/S1rzDp1cJYxA39xSTPZteE` — both splits side by side
with operating points and the prevalence floor.

---

## Table 2 — Paired bootstrap

Both methods scored on the same resamples of the 66 test homology components; the
*difference* is recorded per resample. Components rather than sequences, because members
of a component are ≥40% identical and are one family observed several times.

| metric | delta (probe − BLAST) | 95% CI | P(probe > BLAST) | verdict |
|---|---|---|---|---|
| AU-ROC | **+0.0603** | [+0.0058, +0.1237] | 0.988 | **SIGNIFICANT** |
| AU-PRC | **+0.1708** | [+0.0136, +0.2722] | 0.989 | **SIGNIFICANT** |

Both intervals clear zero. For contrast, the same test on the PET-only task:

| metric | delta | 95% CI | verdict |
|---|---|---|---|
| AU-ROC | +0.0074 | [−0.0959, +0.1232] | not significant |
| AU-PRC | −0.0412 | [−0.1899, +0.0756] | not significant, BLAST ahead |

---

## Table 3 — Generalisation to an unseen lineage

| method | PHA depolymerases flagged |
|---|---|
| ESM-2 650M | **30/36** (83%) |
| BLASTp | 28/36 (78%) |

PHA depolymerases were excluded from training and from the BLAST database, and are a
separate enzyme lineage. Both methods recognise most of them; the probe slightly ahead.
Neither is passing by flagging everything — the probe returns 0 false positives on the 36
non-esterase controls.

---

## Why the probe wins here and not on PET

The two methods fail differently, and the polyester task rewards the probe's mode.

**BLAST scores similarity to a known positive.** It is excellent when a query resembles
something in the database and blind when it does not — it cannot rank a sequence with no
alignment. On the PET task that was adequate because PETases are one dense homology
family; on the polyester task the positives span three substrate classes and several
families, and pairwise similarity to any single one of them is a weaker signal.

**The probe scores a learned property.** The SAE interpretation shows what that property
is: the Ser-Asp-His catalytic triad (`SAE_Feature_Interpretation.md`). A single latent,
9529, fires on the `GxSxG` nucleophile elbow and reaches test AU-ROC 0.92 by itself.

That is precisely why the framing matters. The catalytic apparatus **is** what defines a
polyester hydrolase, so detecting it answers this question well. It is **not** what
distinguishes PET activity from PCL activity, so the same feature cannot answer that one —
every polyester hydrolase carries it, and the models and BLAST alike sit near 0.68 on
that contrast.

A model that finds the catalytic triad is well matched to "is this a plastic-degrading
enzyme?" and badly matched to "which plastic?". The results follow.

## Caveats

1. **The test sequences are not virgin.** They are the PET task's test set relabelled. The
   split, the identity bound and all tuning were redone, but these sequences' behaviour
   under a different labelling has been seen.
2. **42 test positives.** The bootstrap intervals are wide — AU-ROC [0.921, 0.997] — and
   the significant deltas have lower bounds of +0.006 and +0.014, close to zero.
3. **Negatives are unverified.** `3_fold_matched_esterase` means "not annotated as
   plastic-degrading", not "assayed and inactive". All 8 false positives are in that tier
   and some may be correct predictions about untested enzymes.
4. **BLAST is not beaten at the operating point** — only in ranking. If the deliverable is
   a yes/no call at one fixed threshold, the two are equivalent here (8 false positives
   each). The advantage appears when precision is a requirement rather than an outcome.
5. **Part of the gap is split composition.** Test drew harder negatives than validation
   (8 of 62 clear BLASTp's cutoff versus 2 of 76). On validation the methods are within
   0.006 AU-PRC. See Table 1b.

## Reproduce

```
brew install blast
uv run python -m biopet_sae.embed_esm2 --model 650M
```

| file | contents |
|---|---|
| `data/sae_650M_L33/polyester_blast.json` | every number in this document |
| `artifacts/ESM2_650M_Polyester_Run.md` | the probe being compared |
| `artifacts/EDA_polyester_task.md` | dataset, labels, confound floors |
| `artifacts/SAE_Feature_Interpretation.md` | what the features detect |
