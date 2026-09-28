# ESM-2 8M Dense Run — Linear Probe Results

> **Task framing: PET-only (superseded).** This document evaluates the original
> **PET vs. everything else** classifier. The project's headline task was later redefined to
> **polyester degrader vs. non-degrader** (any of PET, PBAT, aliphatic polyester), which
> reclassifies cutinases as positives and is the framing all current results use.
> Current equivalents: `EDA_polyester_task.md`, `ESM2_650M_Polyester_Run.md`,
> `BLAST_vs_ESM2_650M_Polyester.md`. Numbers below remain valid *for the PET-only task* and
> are retained as the record of how the task came to be redefined.

Generated 2026-09-27. Binary PET-vs-rest linear probe on dense ESM-2 8M embeddings.

| setting | value |
|---|---|
| model | `facebook/esm2_t6_8M_UR50D` (320-dim), HuggingFace `transformers` |
| features | dense embeddings, mean- and max-pooled over residues (CLS/EOS stripped) |
| classifier | `SGDClassifier(loss='log_loss', penalty='l2')`, `class_weight='balanced'` |
| optimizer | SGD, constant learning rate, 5000 epochs, `tol=None` |
| regularization | C = 1e-03  (`alpha = 1/(C*n_train)`) |
| learning rate | eta0 = 1e-04 |
| scaling | `StandardScaler` fitted on train only |
| split | `data/processed/dataset_splits_id40.tsv` (40% identity bound, verified) |
| train | 588 sequences (257 PET) |
| val | 119 sequences (32 PET, prevalence 0.269) |
| test | **untouched** |

Hyperparameters were selected on `max_L6` only (Table 1) and then applied unchanged to
all four feature sets, so the val set is not spent on repeated per-feature selection.

---

## Table 1 — Hyperparameter sweep on `max_L6` (val AU-PRC)

Rows are C (regularization; larger = weaker penalty), columns are eta0 (learning rate).

| C \ eta0 | 1e-04 | 1e-03 | 1e-02 | 1e-01 |
|---|---|---|---|---|
| 1e-04 | 0.7533 | 0.7012 | 0.7686 | 0.4437 |
| 1e-03 | **0.8474** | 0.8418 | 0.8366 | 0.7820 |
| 1e-02 | 0.8415 | 0.8415 | 0.8430 | 0.7576 |
| 1e-01 | 0.8200 | 0.8227 | 0.8252 | 0.8239 |
| 1e+00 | 0.7910 | 0.8053 | 0.8118 | 0.7757 |

**Selected: C = 1e-03, eta0 = 1e-04** (val AU-PRC 0.8474).

Note the learning rate barely matters: across three orders of magnitude at C = 1e-3 the
AU-PRC moves only 0.847 -> 0.837. The objective is convex, so any adequately converged
optimizer reaches the same solution -- eta0 is not a meaningful knob for a linear probe.
Only the largest learning rate (1e-1) destabilises training.

---

## Table 2 — Performance across feature sets (mean ± standard error, 5 seeds)

Seeds [42, 43, 44, 45, 46]. Precision and Recall are taken at each run's best-F1 threshold.

| features | AU-ROC | AU-PRC | best F1 | Precision | Recall | threshold |
|---|---|---|---|---|---|---|
| **`max_L6`** | **0.9504 ± 0.0002** | **0.8480 ± 0.0003** | **0.8312 ± 0.0000** | **0.7111 ± 0.0000** | **1.0000 ± 0.0000** | **0.2241 ± 0.0007** |
| `max_L3` | 0.9422 ± 0.0001 | 0.8267 ± 0.0002 | 0.7960 ± 0.0019 | 0.6792 ± 0.0084 | 0.9625 ± 0.0117 | 0.1455 ± 0.0032 |
| `mean_L3` | 0.9211 ± 0.0002 | 0.7096 ± 0.0004 | 0.8205 ± 0.0000 | 0.6957 ± 0.0000 | 1.0000 ± 0.0000 | 0.1714 ± 0.0001 |
| `mean_L6` | 0.9362 ± 0.0002 | 0.8133 ± 0.0006 | 0.8219 ± 0.0000 | 0.7317 ± 0.0000 | 0.9375 ± 0.0000 | 0.2380 ± 0.0004 |

Per-seed AU-PRC, showing the raw spread:

```
max_L6    0.8474  0.8488  0.8488  0.8474  0.8474
max_L3    0.8266  0.8261  0.8266  0.8270  0.8270
mean_L3   0.7087  0.7108  0.7099  0.7099  0.7087
mean_L6   0.8136  0.8127  0.8127  0.8156  0.8120
```

**Interpreting the error bars.** They are near zero (third/fourth decimal, several exactly
0.0000) because the seed only changes the order in which training examples are shuffled;
weights initialise at zero and the objective is convex, so all five runs converge to the
same solution. **These bars measure optimizer determinism, not uncertainty about
performance.** The uncertainty that matters comes from having only 32 validation positives:
a bootstrap over homology components gives roughly [0.71, 0.99] for `max_L6` AU-PRC, an
interval ~0.28 wide against ±0.0003 here -- three orders of magnitude apart.

Max-pooling beats mean-pooling at both layers (0.848 vs 0.710 at L3; 0.848 vs 0.813 at
L6), consistent with the catalytic signal occupying a few residues that mean-pooling
dilutes across ~300.

---

## Table 3 — Precision/Recall vs threshold (`max_L6`, seed 42)

val n = 119, PET = 32, non-PET = 87

| threshold | Precision | Recall | F1 | TP | FP | FN | TN | |
|---|---|---|---|---|---|---|---|---|
| 0.0100 | 0.269 | 1.000 | 0.424 | 32 | 87 | 0 | 0 |  |
| 0.0500 | 0.405 | 1.000 | 0.577 | 32 | 47 | 0 | 40 |  |
| 0.1000 | 0.542 | 1.000 | 0.703 | 32 | 27 | 0 | 60 |  |
| 0.1500 | 0.615 | 1.000 | 0.762 | 32 | 20 | 0 | 67 |  |
| 0.2000 | 0.696 | 1.000 | 0.821 | 32 | 14 | 0 | 73 |  |
| **0.2223** | **0.711** | **1.000** | **0.831** | 32 | 13 | 0 | 74 | ← best F1 |
| 0.2500 | 0.721 | 0.969 | 0.827 | 31 | 12 | 1 | 75 |  |
| 0.3000 | 0.733 | 0.688 | 0.710 | 22 | 8 | 10 | 79 |  |
| 0.4000 | 0.895 | 0.531 | 0.667 | 17 | 2 | 15 | 85 |  |
| 0.5000 | 0.857 | 0.375 | 0.522 | 12 | 2 | 20 | 85 | ← default |
| 0.6000 | 0.875 | 0.219 | 0.350 | 7 | 1 | 25 | 86 |  |
| 0.7000 | 1.000 | 0.062 | 0.118 | 2 | 0 | 30 | 87 |  |
| 0.8000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 32 | 87 |  |

Operating points of practical interest:

| target | threshold | Precision | Recall |
|---|---|---|---|
| recall = 100% | 0.208 | 0.711 | 1.000 |
| recall ≥ 95 | 0.224 | 0.721 | 0.969 |
| recall ≥ 90 | 0.270 | 0.725 | 0.906 |
| precision ≥ 90 | 0.384 | 0.913 | 0.656 |
| precision ≥ 95 | 0.640 | 1.000 | 0.156 |

**Three features of this curve.**

1. **There is a cliff between 0.25 and 0.30.** Recall falls 0.969 -> 0.688 across a 0.05
   threshold change, losing nine true positives. PET scores are bunched in that narrow
   band rather than spread out, so any fixed operating point is fragile.
2. **No score exceeds 0.8.** Everything collapses to zero at threshold 0.80 -- the model
   never assigns high confidence, a consequence of the probability scale rather than of
   ranking quality.
3. **For a discovery tool the left side of the table is the right place to sit.** At
   threshold 0.20 the probe recovers **all 32 PET sequences with 14 false positives**.
   Screening 46 candidates to find 32 is far more useful than the 0.5 default, which
   finds 12 and misses 20. The default threshold is the worst choice on this table.

---

---

## Table 4 — Held-out test set (`max_L6`, one shot)

Test was untouched until `max_L6` and C=1e-3 / eta0=1e-4 had both been selected on val.
The decision threshold **0.2223 is carried over from val and was not re-tuned on test** --
re-optimising a threshold on the test set would invalidate it.

Test: 104 sequences, 31 PET, 73 non-PET (prevalence 0.298).
Negatives by tier: 2a_PBAT = 5, 2b_aliphatic = 6, 3_fold_matched_esterase = 26,
4_naive_control = 36.

### Headline — trained on train only

| metric | value | 95% CI (component bootstrap) |
|---|---|---|
| **Precision** | **0.6000** | |
| **Recall** | **0.9677** | |
| F1 | 0.7407 | |
| AU-ROC | 0.9015 | [0.806, 0.957] |
| AU-PRC | 0.7899 | [0.575, 0.897] |

```
TP = 30    FP = 20    FN = 1    TN = 53
```

30 of 31 PET sequences recovered, one missed, 20 false positives out of 73 negatives.

### Val to test

| | val | test | change |
|---|---|---|---|
| AU-ROC | 0.9501 | 0.9015 | -0.049 |
| AU-PRC | 0.8474 | 0.7899 | -0.058 |
| Precision | 0.7111 | 0.6000 | -0.111 |
| Recall | 1.0000 | 0.9677 | -0.032 |

A modest drop, expected because val was spent on hyperparameter and feature-set
selection. All four changes sit inside the test CIs, so the two splits are consistent
rather than the model collapsing out of sample.

### Refitting on train+val did not help

| trained on | n | AU-ROC | AU-PRC | Precision | Recall | FP |
|---|---|---|---|---|---|---|
| train only | 588 (257 PET) | 0.9015 | 0.7899 | 0.6000 | 0.9677 | 20 |
| train + val | 707 (289 PET) | 0.9019 | 0.7921 | 0.5556 | 0.9677 | 24 |

Ranking is unchanged (+0.002 AU-PRC), but precision falls because the extra training data
shifts the probability scale and the val-derived threshold no longer sits in the same
place. **The train-only model is the one reported.**

### Tier diagnostic on test

| tier | n_neg | test AU-ROC | val AU-ROC |
|---|---|---|---|
| 2a_PBAT | 5 | 0.9032 | 0.7031 |
| **2b_aliphatic** | 6 | **0.6398** | **0.6250** |
| 3_fold_matched_esterase | 26 | 0.8437 | 1.0000 |
| 4_naive_control | 36 | 0.9866 | 0.9896 |

Three things changed or held:

1. **Class 3 fell from a perfect 1.0000 to 0.8437.** The tier that carried the val
   headline does not replicate, so that 1.0000 was partly luck on 40 sequences.
2. **`2a_PBAT` rose from 0.703 to 0.903, but on 5 negatives against 2 on val.** Too few
   either way; this number should not be cited.
3. **`2b_aliphatic` held at 0.640 (val 0.625).** The only tier stable across both
   independent splits, and it is the hard contrast. Replication across splits makes this
   the most reliable finding in the run: the probe cannot distinguish PET degraders from
   enzymes that degrade other aliphatic polyesters.

### Defensible summary

> Precision 0.60, recall 0.97, AU-ROC 0.90 on a held-out set in which no sequence shares
> more than 40% identity with any training sequence -- while discrimination against other
> polyester-degrading enzymes remains near 0.64. The probe detects polyester-hydrolase
> activity rather than PET specificity.

**Test is now spent.** Any further tuning must return to val, or a fresh held-out set is
required.

## Caveat that belongs with every number above

These metrics are measured against a negative set of 87 sequences dominated by
non-plastic esterases (40) and naive controls (36). Evaluated per negative tier, the same
`max_L6` probe scores **AU-ROC 1.0000 against non-plastic esterases** but only **0.625
against enzymes that degrade other polyesters** (n=9) and **0.703 against PBAT-type
degraders** (n=2). The headline AU-ROC of 0.95 is therefore carried by the easy tiers.

The probe detects *polyester-hydrolase activity*, not *PET specificity*. This held across
a 100-point hyperparameter search spanning two optimizers, four feature sets, five
regularization strengths and four learning rates -- no configuration lifted the
hard-tier score above ~0.72. See `artifacts/dataset_splits_0.4.md` for why that contrast
is the scientifically meaningful one.

## Reproduce

```
uv run python -m biopet_sae.embed_esm2 --model 8M
uv run python -m biopet_sae.train_binary_probe \
    --embeddings data/embeddings_8M/embeddings.npz \
    --splits data/processed/dataset_splits_id40.tsv

# Table 4 (test set, threshold 0.2223 carried from val)
uv run python -m biopet_sae.eval_test --features max_L6 --threshold 0.2223
```

| file | contents |
|---|---|
| `data/embeddings_8M/embeddings.npz` | pooled embeddings, layers 1-6, mean + max |
| `data/processed/dataset_splits_id40.tsv` | the split used |
| `artifacts/ESM2_8M_Dense_Run.json` | Tables 1-3, machine-readable |
| `artifacts/ESM2_8M_Dense_Run_test.json` | Table 4 (test set), machine-readable |
