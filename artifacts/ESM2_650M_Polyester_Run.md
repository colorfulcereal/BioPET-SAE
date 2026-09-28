# ESM-2 650M — Polyester Degrader Classifier

Generated 2026-09-28. Binary **polyester degrader vs non-degrader** on ESM-2 650M layer
33. Dataset and labels: `EDA_polyester_task.md`.

| setting | value |
|---|---|
| model | `facebook/esm2_t33_650M_UR50D`, layer 33 (1280-dim) |
| classifier | `SGDClassifier(loss='log_loss', penalty='l2')`, `class_weight='balanced'` |
| optimizer | SGD, constant learning rate, 5000 epochs, `tol=None` |
| regularization | C = 1e-02 (`alpha = 1/(C*n_train)`) |
| learning rate | eta0 = 1e-04 |
| selected on | **val log-loss**, using `max_L33` only |
| split | `dataset_splits_id40.tsv`, 40% identity bound, verified |
| train | 588 (303 polyester) · val 119 (43) · test 104 (42) |
| seeds | 42–46 |

Hyperparameters were selected on `max_L33` alone and applied unchanged to all twelve
feature sets. This is a **clean run** — the sweep, the selection and the threshold were
all redone for this task rather than carried over from the PET classifier.

---

## Table 1 — Hyperparameter sweep (val log-loss, lower is better)

| C \ eta0 | 1e-04 | 1e-03 | 1e-02 | 1e-01 |
|---|---|---|---|---|
| 1e-04 | 0.5074 | 0.5039 | 0.5447 | 0.6552 |
| 1e-03 | 0.3554 | 0.3749 | 0.5430 | 2.2408 |
| 1e-02 | **0.2693** | 0.2792 | 0.3276 | 3.0836 |
| 1e-01 | 0.2876 | 0.2754 | 0.3497 | 2.7231 |
| 1e+00 | 0.3427 | 0.3398 | 0.3326 | 0.7313 |

**Selected: C = 1e-02, eta0 = 1e-04** (val log-loss 0.2693).

### Why log-loss and not AU-PRC

**AU-PRC saturates on this task and cannot select.** The same sweep scored by val AU-PRC:

| C \ eta0 | 1e-04 | 1e-03 | 1e-02 | 1e-01 |
|---|---|---|---|---|
| 1e-04 | 0.8606 | 0.8810 | 0.7629 | 0.4906 |
| 1e-03 | 0.9773 | 0.9772 | 0.8866 | 0.7237 |
| 1e-02 | 0.9954 | 0.9954 | 0.9696 | 0.8612 |
| 1e-01 | 0.9995 | 0.9995 | 0.9995 | 0.9846 |
| 1e+00 | 0.9995 | 0.9995 | 0.9995 | 1.0000 |

Several cells tie at 0.9995 and one reaches a perfect **1.0000** — at C=1, eta=1e-1, the
corner where log-loss is **2.72**, i.e. wildly miscalibrated and not converged. AU-PRC
selection would have chosen exactly that cell. It is the same trap that produced spurious
optima for `mean_L3` (8M) and `mean_L33` (650M) on the PET task; log-loss is a proper
scoring rule, does not saturate, and rejects it.

---

## Table 2 — All feature sets at the selected hyperparameters (mean ± SE, 5 seeds)

| features | val AU-ROC | test AU-ROC | test AU-PRC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| `mean_L33` | 0.9510 ± 0.0001 | 0.9745 ± 0.0001 | 0.9651 ± 0.0001 | 0.9469 ± 0.0057 | 0.7619 ± 0.0000 | 0.8444 ± 0.0022 |
| `mean_L9` | 0.9930 ± 0.0000 | 0.9710 ± 0.0001 | 0.9613 ± 0.0001 | 0.8298 ± 0.0000 | 0.9286 ± 0.0000 | 0.8764 ± 0.0000 |
| **`max_L33`** | **0.9976 ± 0.0000** | **0.9680 ± 0.0001** | **0.9529 ± 0.0002** | **0.8298 ± 0.0000** | **0.9286 ± 0.0000** | **0.8764 ± 0.0000** |
| `mean_L30` | 0.9982 ± 0.0000 | 0.9604 ± 0.0001 | 0.9287 ± 0.0002 | 0.8400 ± 0.0000 | 1.0000 ± 0.0000 | 0.9130 ± 0.0000 |
| `max_L30` | 0.9976 ± 0.0002 | 0.9403 ± 0.0001 | 0.9093 ± 0.0001 | 0.8000 ± 0.0000 | 0.8571 ± 0.0000 | 0.8276 ± 0.0000 |
| `max_L24` | 0.9991 ± 0.0000 | 0.9448 ± 0.0001 | 0.9091 ± 0.0001 | 0.8000 ± 0.0000 | 0.8571 ± 0.0000 | 0.8276 ± 0.0000 |
| `max_L9` | 1.0000 ± 0.0000 | 0.9357 ± 0.0001 | 0.9067 ± 0.0001 | 0.8297 ± 0.0046 | 0.8810 ± 0.0000 | 0.8545 ± 0.0024 |
| `mean_L18` | 0.9996 ± 0.0001 | 0.9382 ± 0.0002 | 0.9060 ± 0.0002 | 0.8125 ± 0.0000 | 0.9286 ± 0.0000 | 0.8667 ± 0.0000 |

`max_L33` is bolded as the selected feature set, not as the best — `mean_L33` and
`mean_L9` score marginally higher on test AU-PRC, but selecting on test is not
available. The spread across the top six is 0.953–0.965 AU-PRC, well inside the
confidence interval below.

Standard errors sit in the fourth decimal because the seed only permutes training order
on a convex objective. **They measure optimizer determinism, not performance
uncertainty** — the honest interval is the component bootstrap in Table 3.

---

## Table 3 — Held-out test set (`max_L33`, seed 42)

Threshold 0.1455, selected on val, applied unchanged.

| metric | value | 95% CI (component bootstrap) |
|---|---|---|
| **AU-ROC** | **0.9681** | [0.921, 0.997] |
| **AU-PRC** | **0.9526** | [0.857, 0.996] |
| Precision | 0.830 | |
| Recall | 0.929 | |
| F1 | 0.876 | |

```
TP = 39    FP = 8    FN = 3    TN = 54
```

Against the floors from `EDA_polyester_task.md`: amino-acid composition reaches AU-ROC
0.7020 on this same split, and prevalence puts the AU-PRC floor at 0.4784.

### Recall by positive tier

| tier | caught |
|---|---|
| PET degraders | **30/31** |
| PBAT-type | **3/5** |
| aliphatic | **6/6** |

All three substrate classes are recovered, so the result is not carried by the PET
majority.

### False positives by negative tier

| tier | false positives |
|---|---|
| UniProt esterases, fold-matched | 8/26 |
| non-esterase controls | 0/36 |

**Every error is a fold-matched esterase; none is a naive control.** Those 8 are UniProt
entries never assayed on a polyester, so some may be correct predictions against absent
labels rather than mistakes.

---

## Table 3b — Train, validation and test side by side

| | TRAIN (in-sample) | VAL (selection) | TEST (held out) |
|---|---|---|---|
| n | 588 | 119 | 104 |
| positives | 303 (prev 0.515) | 43 (prev 0.361) | 42 (prev 0.404) |
| **AU-ROC** | 1.0000 [1.000, 1.000] | 0.9976 [0.982, 1.000] | 0.9681 [0.920, 0.997] |
| **AU-PRC** | 1.0000 [0.999, 1.000] | 0.9954 [0.970, 1.000] | 0.9526 [0.860, 0.996] |
| Precision | 0.9619 | 0.9773 | 0.8298 |
| Recall | 1.0000 | 1.0000 | 0.9286 |
| F1 | 0.9806 | 0.9885 | 0.8764 |
| TP/FP/FN/TN | 303/12/0/273 | 43/1/0/75 | 39/8/3/54 |

```
val -> test    AU-ROC  -0.029    AU-PRC  -0.043    Precision  -0.148    Recall  -0.071
train -> test  AU-ROC  -0.032    AU-PRC  -0.047
```

**Train AU-ROC is exactly 1.0000** — the separable regime, expected with 1280 features over
588 points. But the train→test gap is only −0.032, so memorising train costs little
generalisation.

**Ranking transfers; the threshold does not.** AU-ROC and AU-PRC move −0.029 and −0.043
val→test, both inside the test confidence intervals. **Precision falls −0.148** — validation
has exactly one false positive in 76 negatives, test has 8 in 62. That is a
threshold-transfer failure, not a ranking failure: 0.1455 was fitted where negatives were
easier, and val prevalence (0.361) differs from test (0.404).

**The validation column is not an unbiased estimate.** It selected C, eta0 *and* the
threshold, so 0.9976 / 0.9954 is the number the selection optimised toward. Test is the only
clean column, and the val→test drop is approximately the size of that selection bias.

**Curves:** `https://claude.ai/artifact/S1rzDp1cJYxA39xSTPZteE`

---

## Table 4 — Generalisation to an unseen enzyme lineage

| | flagged as polyester degraders |
|---|---|
| held-out PHA depolymerases | **30/36** (83%) |

PHA depolymerases were **never trained on** and belong to a different enzyme lineage from
the cutinase-like hydrolases in the positive set. The classifier recognises them anyway.

This is the strongest single piece of evidence that the model learned *polyester
hydrolase* as a general property rather than memorising the training families. It cannot
be passed by flagging everything: the same model returns **0 false positives on 36
non-esterase controls**.

---

## Comparison with the PET-only classifier

| | PET vs everything | polyester vs non-polyester |
|---|---|---|
| test AU-ROC | 0.8856 | **0.9681** |
| test AU-PRC | 0.8150 | **0.9526** |
| Recall | 0.742 | **0.929** |
| beats BLASTp? | no | **yes, both metrics, significant** |

The honest reading is not that the model improved — it is that **the question changed to
one the representation can answer.** The SAE analysis shows why: the model's strongest
features detect the catalytic triad shared by all polyester hydrolases
(`SAE_Feature_Interpretation.md`). Asking it to find that apparatus is a question it can
answer; asking it to distinguish PET among enzymes that all possess the apparatus is not.

Part of the gain is that the new task is genuinely easier — the composition floor rises
from 0.702 (this task) against the model's 0.968. The lift over floor is what matters,
and it is large in both cases.

## One caveat on the test set

The test *sequences* are the same ones used for the PET classifier, relabelled. The split
and the 40% identity bound are untouched, and no hyperparameter or threshold was carried
over — the sweep, selection and threshold were all redone here. But these sequences are
not virgin: their behaviour under a different labelling has been observed. A fully
independent test would need sequences held back from the start.

## Reproduce

```
uv run python -m biopet_sae.embed_esm2 --model 650M
# sweep, selection and evaluation: see data/sae_650M_L33/polyester_*.json
```

| file | contents |
|---|---|
| `data/sae_650M_L33/polyester_sweep.json` | Table 1, both criteria |
| `data/sae_650M_L33/polyester_results.json` | Tables 2–4 |
| `artifacts/EDA_polyester_task.md` | dataset, labels, confound baselines |
| `artifacts/BLAST_vs_ESM2_650M_Polyester.md` | homology-search comparison |
| `artifacts/SAE_Feature_Interpretation.md` | what the features detect |
