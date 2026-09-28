# ESM-2 650M Dense Run — Linear Probe Results (layer 33)

Generated 2026-09-28. Binary PET-vs-rest linear probe on dense ESM-2 650M embeddings.
Structured to match `ESM2_8M_Dense_Run.md` so the two are directly comparable.

| setting | value |
|---|---|
| model | `facebook/esm2_t33_650M_UR50D` (**1280-dim**), HuggingFace `transformers` |
| layer | **33** (final; one of the six where InterPLM ships SAEs) |
| features | dense embeddings, mean- and max-pooled over residues (CLS/EOS stripped) |
| classifier | `SGDClassifier(loss='log_loss', penalty='l2')`, `class_weight='balanced'` |
| optimizer | SGD, constant learning rate, 5000 epochs, `tol=None` |
| regularization | C = 1e-01  (`alpha = 1/(C*n_train)`) |
| learning rate | eta0 = 1e-03 |
| scaling | `StandardScaler` fitted on train only |
| split | `data/processed/dataset_splits_id40.tsv` (40% identity bound, verified) |
| train | 588 sequences (257 PET) |
| val | 119 sequences (32 PET, prevalence 0.269) |
| test | 104 sequences (31 PET, prevalence 0.298) |
| seeds | [42, 43, 44, 45, 46] |

Hyperparameters were selected on **`max_L33` only** (Table 1) and then applied unchanged
to both poolings. `mean_L33` is therefore *not* at its own optimum -- its own best val cell
is C=1e-4, eta=1e-1, which reaches val AU-PRC 0.7987 but sits in the unconverged
large-learning-rate corner and transfers poorly (test 0.7318). Selecting per feature set
would spend the val set four times over; see the note under Table 1.

---

## Table 1 — Hyperparameter sweep on `max_L33` (val AU-PRC)

Rows are C (regularization; larger = weaker penalty), columns are eta0 (learning rate).

| C \ eta0 | 1e-04 | 1e-03 | 1e-02 | 1e-01 |
|---|---|---|---|---|
| 1e-04 | 0.5617 | 0.5238 | 0.4429 | 0.2689 |
| 1e-03 | 0.6539 | 0.6423 | 0.5903 | 0.4585 |
| 1e-02 | 0.6834 | 0.6830 | 0.6688 | 0.5037 |
| 1e-01 | 0.6858 | **0.7178** | 0.6630 | 0.5726 |
| 1e+00 | 0.7138 | 0.7148 | 0.7063 | 0.6911 |

**Selected: C = 1e-01, eta0 = 1e-03** (val AU-PRC 0.7178).

Two observations. The surface is smooth in the neighbourhood of the optimum (0.686 / 0.718
/ 0.663 across adjacent eta at C=1e-1), so the selection is stable rather than a lucky
cell. And the **largest learning rate degrades every row** -- 0.269 at C=1e-4 rising only
to 0.691 at C=1.0 -- because eta0=1e-1 does not converge on this objective. That is the
same trap that produced a spurious 0.911 for `mean_L3` in the 8M run.

---

## Table 2 — Validation performance, both poolings (mean ± SE, 5 seeds)

Precision and Recall are taken at each run's best-F1 threshold.

| features | AU-ROC | AU-PRC | best F1 | Precision | Recall | threshold |
|---|---|---|---|---|---|---|
| **`max_L33`** | **0.8979 ± 0.0001** | **0.7191 ± 0.0006** | **0.7714 ± 0.0000** | **0.7105 ± 0.0000** | **0.8438 ± 0.0000** | **0.0029 ± 0.0000** |
| `mean_L33` | 0.8341 ± 0.0006 | 0.6426 ± 0.0007 | 0.6923 ± 0.0000 | 0.5870 ± 0.0000 | 0.8438 ± 0.0000 | 0.0006 ± 0.0000 |

Per-seed val AU-PRC:

```
max_L33   0.7178  0.7197  0.7197  0.7206  0.7178
mean_L33  0.6406  0.6445  0.6441  0.6421  0.6416
```

**Max-pooling beats mean-pooling by a wide margin** -- AU-PRC 0.719 vs 0.643, AU-ROC 0.898
vs 0.834. The same direction as the 8M run and for the same reason: the catalytic signal
occupies a handful of residues that mean-pooling divides across ~300.

**Interpreting the error bars.** They are in the fourth decimal because the seed only
changes the order in which training examples are shuffled; weights initialise at zero and
the objective is convex, so every run converges to the same solution. **These bars measure
optimizer determinism, not uncertainty about performance.** The uncertainty that matters
comes from 32 validation positives and 31 test positives -- the component bootstrap on the
test set gives AU-PRC [0.603, 0.922] for `max_L33`, an interval
~0.28 wide against ±0.001 here.

---

## Table 3 — Precision/Recall vs threshold (`max_L33`, seed 42, val)

val n = 119, PET = 32, non-PET = 87

| threshold | Precision | Recall | F1 | TP | FP | FN | TN | |
|---|---|---|---|---|---|---|---|---|
| 0.0010 | 0.525 | 0.969 | 0.681 | 31 | 28 | 1 | 59 |  |
| 0.0020 | 0.651 | 0.875 | 0.747 | 28 | 15 | 4 | 72 |  |
| **0.0030** | **0.703** | **0.812** | **0.754** | 26 | 11 | 6 | 76 | ← best F1 |
| 0.0050 | 0.750 | 0.750 | 0.750 | 24 | 8 | 8 | 79 |  |
| 0.0100 | 0.684 | 0.406 | 0.510 | 13 | 6 | 19 | 81 |  |
| 0.0200 | 0.750 | 0.281 | 0.409 | 9 | 3 | 23 | 84 |  |
| 0.0500 | 0.800 | 0.250 | 0.381 | 8 | 2 | 24 | 85 |  |
| 0.1000 | 0.800 | 0.250 | 0.381 | 8 | 2 | 24 | 85 |  |
| 0.2000 | 0.778 | 0.219 | 0.341 | 7 | 2 | 25 | 85 |  |
| 0.5000 | 0.714 | 0.156 | 0.256 | 5 | 2 | 27 | 85 |  |

**The operating threshold is 0.0030 -- two orders of magnitude below 0.5.** This is the
saturation signature diagnosed in the 8M run: 1280 features over 588 training points is
separable, so weights grow until probabilities pin near 0 and 1, and held-out positives
land in the band reserved for negatives. Ranking metrics are threshold-free and unaffected,
but the probability scale is not calibrated and the default 0.5 threshold is unusable
(recall 0.156).

There is also a cliff between 0.005 and 0.010: recall falls 0.750 -> 0.406 across a 0.005
change. Any fixed operating point on this model is fragile.

---

## Table 4 — Held-out test set (mean ± SE, 5 seeds)

Threshold selected on val per seed, then applied to test unchanged. Test was untouched
until the hyperparameters were fixed.

| features | AU-ROC | AU-PRC | F1 | Precision | Recall | TP/FP/FN/TN |
|---|---|---|---|---|---|---|
| **`max_L33`** | **0.8883 ± 0.0012** | **0.8177 ± 0.0012** | **0.7372 ± 0.0029** | **0.7327 ± 0.0057** | **0.7419 ± 0.0000** | 23/8/8/65 |
| `mean_L33` | 0.8264 ± 0.0014 | 0.7344 ± 0.0016 | 0.6111 ± 0.0000 | 0.5366 ± 0.0000 | 0.7097 ± 0.0000 | 22/19/9/54 |

Component-bootstrap CIs for `max_L33` (seed 42): AU-ROC [0.765, 0.957], AU-PRC [0.603, 0.922].

### False positives by negative tier (test, seed 42)

| features | 2a_PBAT | 2b_aliphatic | 3_fold_matched_esterase | 4_naive_control |
|---|---|---|---|---|
| `max_L33` | 1/5 | 2/6 | 2/26 | 3/36 |
| `mean_L33` | 3/5 | 3/6 | 3/26 | 10/36 |

| features | class-3 FP, Actinomycetota | class-3 FP, non-Actinomycetota |
|---|---|---|
| `max_L33` | 1/11 | 1/15 |
| `mean_L33` | 2/11 | 1/15 |

---

## The main result: scale fixes the failure mode without improving the score

| | ESM-2 8M `max_L6` | **ESM-2 650M `max_L33`** | BLASTp (val-tuned) |
|---|---|---|---|
| Precision | 0.600 | **0.733** | 0.828 |
| Recall | **0.968** | 0.742 | 0.774 |
| F1 | 0.741 | 0.737 | **0.800** |
| AU-ROC | **0.9015** | 0.8883 | 0.8778 |
| AU-PRC | 0.790 | 0.818 | **0.850** |
| total false positives | 20 | **8** | 5 |
| class-3 FP | 10/26 | **2/26** | 0/26 |
| class-3 FP, Actinomycetota | **9/11** | **1/11** | 0/11 |

**The taxonomic shortcut is gone.** The 8M probe made 9 of its 11 Actinomycetota class-3
errors -- an 82% false-positive rate on that group against 7% for everything else. At 650M
that falls to 1/11 Actinomycetota and 1/15 non-Actinomycetota: **equal rates**. Total false
positives drop from 20 to 8.

This is direct evidence for the conclusion reached in `BLAST_vs_ESM2_8M.md` Table 5b, where
phylum reweighting and Actinomycetota subsampling both failed to move the 8M error count
(10/26 in all three variants). The problem was never the label balance; it was that 8M's
representation did not carry a linearly-decodable PET-specificity feature, so the probe
fell back on composition. Layer 33 of 650M carries more of it.

**But F1 is flat (0.741 -> 0.737).** The errors converted rather than disappeared: 8M caught
30 of 31 with 20 false positives, 650M catches 23 with 8. An 80x increase in parameters
bought a better-behaved model, not a better-scoring one.

Paired bootstrap over components on test, delta = 650M - 8M:

| metric | delta | 95% CI | significant? |
|---|---|---|---|
| AU-ROC | -0.0152 | [-0.0965, +0.0484] | no |
| AU-PRC | +0.0285 | [-0.0833, +0.1312] | no |

delta = 650M - BLASTp:

| metric | delta | 95% CI | significant? |
|---|---|---|---|
| AU-ROC | +0.0074 | [-0.0959, +0.1232] | no |
| AU-PRC | -0.0412 | [-0.1899, +0.0756] | no |

**Nothing separates the three methods statistically** at n=104 with 31 positives.

### The tier where everything agrees

| method | 2b_aliphatic AU-ROC |
|---|---|
| ESM-2 8M `max_L6` | 0.6398 |
| ESM-2 650M `max_L33` | 0.6828 |
| BLASTp | 0.6989 |

Two model scales 80x apart and a sequence-alignment method all land near 0.68 on
distinguishing PET degraders from other polyester degraders. That consistency across
fundamentally different approaches is the strongest available evidence that this contrast
-- not fold recognition -- is the unsolved problem. It is where the SAE work is aimed; see
`dataset_splits_0.4.md` section 1.1 for why the PBAT subgroup is its sharpest form.

## Reproduce

```
uv run python -m biopet_sae.embed_esm2 --model 650M
uv run python -m biopet_sae.train_binary_probe \
    --embeddings data/embeddings_650M/embeddings.npz \
    --splits data/processed/dataset_splits_id40.tsv \
    --C 1e-01 --eta0 1e-03
```

| file | contents |
|---|---|
| `artifacts/ESM2_650M_Dense_Run.json` | every number in this document, machine-readable |
| `artifacts/ESM2_8M_Dense_Run.md` | the 8M run this is compared against |
| `artifacts/BLAST_vs_ESM2_8M.md` | BLAST baseline and the 8M error analysis |
| `artifacts/dataset_splits_0.4.md` | dataset, split and class rationale |
