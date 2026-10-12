# How to report results on this benchmark

Five rules. Breaking any of them produces a number that looks fine and means nothing.

## 1. Decompose test metrics by `negative_group`

A single blended figure is not interpretable, because **39 of the 62 test negatives are
separated perfectly** by any working model. The headline is mostly free weight.

Report this shape, always:

| negatives | n | AU-ROC | AU-PRC |
|---|---|---|---|
| all — headline | 62 | | |
| `naive_control` | 36 | | |
| `trna_processing` | 3 | | |
| `ec311_wrong_fold` | 23 | | |
| **`ab_hydrolase_hard_negative`** | **10** | | |

The reference probe gives 0.9681 blended, and 1.0000 / 1.0000 / 0.9141 / **0.8167** down the
rows. If your blended number improves while the last row does not, you have not improved at
the task.

## 2. Bootstrap over `component`, never over proteins

256 of the 388 positives (66%) are the single component `BPS0190`. Resampling proteins counts
that one enzyme family 256 times, producing intervals that are wrong rather than narrow.

Resample **components** with replacement, keeping every member of each drawn component:

```python
rng = np.random.default_rng(0)
pc, nc = list(pos_components), list(neg_components)
for _ in range(2000):
    p = np.concatenate([pos_idx[pc[i]] for i in rng.choice(len(pc), len(pc), True)])
    n = np.concatenate([neg_idx[nc[i]] for i in rng.choice(len(nc), len(nc), True)])
    stat.append(metric(scores[p], scores[n]))
```

What this changes in practice: a per-protein Cliff's delta of 0.716 for the strongest
interpretability feature became 0.116 with a cluster bootstrap, and its interval spanned zero.

## 3. State n for the hard-negative cell

It is **10** in test. Any claim there is low-powered and must carry an interval. The reference
probe's 0.8167 has a 95% CI of [0.676, 0.926].

## 4. Say whether you tuned on validation

Validation holds **1** hard negative. Model selection or threshold tuning done there was done
against the easy task. The reference probe's threshold of 0.1455 came from val, which is why
its test precision is 0.830 — the threshold was set where nothing hard was pushing back.

If you tune on val, say so, and report test decomposed anyway.

## 5. Report effect sizes against a reference distribution, not against a cutoff

A large effect that misses a pre-set threshold has not collapsed. Give the point estimate,
then an interval from a bootstrap that respects the dependence structure (rule 2), then where
it sits in a null or reference distribution. Only then say whether it clears the cutoff.

When power is the limit, the honest word is **undetermined**, not "no effect".

---

## Baselines on this split

| method | test AU-ROC | test AU-PRC |
|---|---|---|
| ESM-2 650M linear probe | 0.9681 | 0.9526 |
| BLASTp, matched protocol | 0.9040 | 0.7552 |
| amino-acid composition | 0.7020 | — |

The composition baseline reaches **0.8926 under random cross-validation** against 0.7020 on
the homology-bounded split. That 0.19 gap is a direct measure of what a non-homology-aware
split buys you, and it is the reason this benchmark ships with fixed splits rather than a
cross-validation recipe.

## What a result here can and cannot support

**Can**: that sequence representations separate polyester degraders from fold-matched
esterases better than homology search does, with the hard-negative cell reported and its n
stated.

**Cannot**: remote-homolog detection. 66% of positives are one component; this is
within-family discrimination and should be described that way.

**Cannot**: anything resting on validation performance. Validation has one hard negative.
