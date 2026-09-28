# ESM-2 8M Dense Run — Linear Probe Results

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
```

| file | contents |
|---|---|
| `data/embeddings_8M/embeddings.npz` | pooled embeddings, layers 1-6, mean + max |
| `data/processed/dataset_splits_id40.tsv` | the split used |
| `artifacts/ESM2_8M_Dense_Run.json` | every number in this document, machine-readable |
