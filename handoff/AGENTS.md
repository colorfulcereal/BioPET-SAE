# Polyester-degrader benchmark — agent instructions

You are helping reproduce a protein-classification experiment in a fresh repository. This
file is the contract. Read `README.md` for the science; read this for the rules that must not
be broken.

## The task

Binary classification over protein sequences: **does this enzyme degrade a polyester?**

- **Positive** — an enzyme experimentally reported to degrade a plastic whose backbone is
  joined by carboxylic ester bonds. 388 of them.
- **Hard negative** — an α/β-hydrolase-fold protein carrying the same Ser–His–Asp catalytic
  triad as the positives, never reported to degrade a plastic. **51 of them.**
- **Easy negatives** — 372 further proteins in three tiers of decreasing similarity.

Separating positives from the 51 hard negatives *is* the problem. Everything else is
separated trivially and inflates any headline number.

## Files

| file | what it is |
|---|---|
| `dataset.tsv` | 847 rows, 18 columns, sequences included. The only data file you need. |
| `dataset.fasta` | Same sequences; header carries split, label, group, component. |
| `schema.json` | Column definitions. Read before writing a loader. |
| `manifest.json` | Counts, split constraints, and the baseline numbers to reproduce. |
| `EVALUATION.md` | How results must be reported. Non-negotiable. |

Verify `sha256` from `manifest.json` before use.

## Hard rules

1. **Never train, tune, or select on `split == "test"`.** Fit on `train`. Select on `val`.
   Touch `test` once, at the end.

2. **Bootstrap over the `component` column, never over individual proteins.** 256 of the 388
   positives (66%) are the single component `BPS0190`. A per-protein confidence interval
   counts that one enzyme family 256 times and is wrong, not merely narrow. Resample
   components with replacement, keeping all members of each.

3. **Report test metrics decomposed by `negative_group`.** A single blended number is not
   interpretable here: 39 of the 62 test negatives are separated perfectly by any working
   model, so the headline is mostly free weight. Always report the
   `ab_hydrolase_hard_negative` cell separately and state its n, which is **10**.

4. **Use `negative_group`, not `class_label`.** `class_label` is legacy. Its
   `3_fold_matched_esterase` tier is a misnomer — only 50 of its 185 members actually carry
   the α/β hydrolase fold; 99 are tRNA-processing hydrolases that share an EC reaction class
   but no fold and no serine nucleophile.

5. **`split == "unused"` rows carry no claim.** They are 36 PHA depolymerases. They are not a
   held-out test and not a generalisation set. Leave them out, or justify in writing why you
   included them.

6. **Do not tune a decision threshold on `val` without saying so.** Validation contains
   exactly **1** hard negative, because the tRNA family forms three homology components and
   identity-bounded splitting sent one of them wholly to val. A threshold tuned on val is
   tuned against the easy task; this is why the reference probe's test precision falls to
   0.830.

## Reproduce this first

Before trying anything new, reproduce the reference probe in `manifest.json`
→ `baselines_to_reproduce.probe`:

- embed with `facebook/esm2_t33_650M_UR50D`, **max-pool** over residues of `hidden_states[33]`
- `StandardScaler` fit on train only, persisted and reused — never refit at eval
- `SGDClassifier(loss="log_loss", penalty="l2", alpha=1/(1e-2*n_train),
  learning_rate="constant", eta0=1e-4, max_iter=5000, tol=None, random_state=0)`

Targets: test AU-ROC **0.9681**, AU-PRC **0.9526**; and the decomposition, including
`ab_hydrolase_hard_negative` AU-ROC **0.8167** on n=10.

If your blended number matches but the hard-negative cell does not, you have a bug in the
grouping, not a better model.

## Mistakes already made here — do not repeat them

- **An EC code is not a fold.** The negatives were originally gathered with `ec:3.1.1.-`,
  which returns D-aminoacyl-tRNA deacylases and peptidyl-tRNA hydrolases. They hydrolyse an
  ester bond by an unrelated mechanism, have no serine nucleophile, and are trivially
  separable. If you extend the negative set, filter on **InterPro `IPR029058`** plus curated
  `ACT_SITE` residues, not on EC.
- **A name is not a fold either.** The original fold filter was a 16-word blocklist matched
  against enzyme names. A Proteinase K-like subtilisin that degrades PLA passed it and sits
  in the positive set (`BPS0016`); actual Proteinase K was excluded. If fold matters to your
  analysis, verify it with InterProScan, which runs on sequence and needs no accession.
- **Absence of recorded activity is not evidence of inactivity.** No negative was ever
  assayed on a plastic. Every negative label is an assumption.
- **A large effect that fails a threshold has not "collapsed".** Report the effect size, then
  locate it in a null or reference distribution, then say whether it clears the cutoff.

## Extending the dataset

The single highest-value improvement is **more hard negatives**: UniProt reviewed, carrying
`IPR029058` plus three curated `ACT_SITE` residues, never assayed on a plastic, excluding the
cutinase and PETase families. Thousands are available; 300–400 would give every split real
hard negatives and make validation usable for the first time.

Note that this fixes only one side. With 66% of positives in one component the comparison
stays underpowered no matter how many negatives are added, and positive diversity cannot be
bought — fewer than 50 experimentally verified plastics-active enzymes existed in the
literature as of the PAZy paper (2022).
