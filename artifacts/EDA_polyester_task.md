# Polyester-Degrader Dataset

Generated 2026-09-28. The dataset under its revised labelling: **does this enzyme degrade
a polyester, or not?** Supersedes the PET-only framing in `dataset_splits_0.4.md`, which
remains the reference for provenance, deduplication and split construction.

## Why the task was redefined

The PET-only classifier plateaued at AU-ROC ~0.68 against other polyester degraders at
both model scales and for BLAST alike. The SAE interpretation explains why
(`SAE_Feature_Interpretation.md`): the model's strongest features locate the
**Ser-Asp-His catalytic triad**, which PETases and cutinases share almost exactly. A
PET-vs-cutinase boundary is not available in this representation.

Most "PETases" are in fact cutinases assayed on PET — LCC is *Leaf-branch Compost
**Cutinase***, Cut190 and TfCut are cutinases, and 20 of the 320 PET-class enzymes are
named "cutinase" outright. The PET/non-PET line runs through one protein family, and it
is drawn by which substrate somebody happened to test.

So the question was changed to one the biology supports and the data can answer.

---

## Labels

| tier | train | val | test | total | role |
|---|---|---|---|---|---|
| `1_pet` — PET degraders | 257 | 32 | 31 | 320 | **positive** |
| `2a_PBAT` — PBAT-type co-polyester | 13 | 2 | 5 | 20 | **positive** |
| `2b_aliphatic` — aliphatic polyester (PCL, PLA, PBS) | 33 | 9 | 6 | 48 | **positive** |
| `3_fold_matched_esterase` — UniProt esterases, no plastic annotation | 119 | 40 | 26 | 185 | negative |
| `4_naive_control` — non-esterase controls | 166 | 36 | 36 | 238 | negative |
| **POSITIVES** | **303** | **43** | **42** | **388** | |
| **NEGATIVES** | **285** | **76** | **62** | **423** | |
| `heldout_pha` — PHA depolymerases | — | — | — | 36 | **held out, never trained** |

**The split is unchanged.** Train/val/test were built from homology components under a
verified 40% identity bound; only the labels moved. Max cross-split identity remains
0.3967 (train/val) and 0.3800 (train/test) — see `dataset_splits_0.4.md` §6.

### Why PHA is held out

PHA depolymerases hydrolyse polyhydroxyalkanoates — a polyester, and a plastic — but they
are a **separate enzyme lineage** from the cutinase-like hydrolases in the positive set.
Excluding them from training turns them into a generalisation probe: a model that has
learned "polyester hydrolase" as a concept should recognise them; one that has memorised
the cutinase family should not. Because the negative set contains 238 non-esterase
controls, a model cannot pass this probe simply by flagging everything.

---

## Confound baselines

What the task yields to signals that have nothing to do with catalysis. Measured on the
**homology-bounded train → test split**, the same one the models are scored on.

| signal | test AU-ROC | test AU-PRC |
|---|---|---|
| amino-acid composition (20 features) | **0.7020** | 0.6737 |
| sequence length alone | 0.5823 | — |
| Actinomycetota indicator alone | 0.5842 | — |
| prevalence (AU-PRC floor) | — | 0.4784 |

**0.7020 is the floor any real result must clear**, not 0.5.

One number to treat with suspicion: the same composition baseline scores **0.8926** under
a random 5-fold CV, because homologous near-duplicates land on both sides of the split.
That figure is meaningless here and is reported only to show the size of the inflation —
0.19 AU-ROC — that homology leakage produces on this dataset.

## The phylum confound shrank, and reversed

| task | positives | negatives | gap |
|---|---|---|---|
| OLD — PET only | 72.6% Actinomycetota | 66.0% | **+6.6 pts** |
| NEW — any polyester | 66.0% | 70.4% | **−4.4 pts** |

Adding PBAT and aliphatic degraders — largely *Pseudomonadota* and *Bacillota* — dilutes
the actinomycete dominance of the PET class. The gap narrows and changes sign, so a model
can no longer gain by detecting "is this a high-GC actinomycete protein". An
Actinomycetota-only classifier now scores AU-ROC 0.5842.

This matters because the PET task's single largest error mode was taxonomic: the 8M probe
made 9 of its 11 Actinomycetota class-3 errors, including two *Thermobifida fusca*
carboxylesterases from the very organism whose PETases it trained on
(`BLAST_vs_ESM2_8M.md` Table 5).

## Sequence length

| | median | IQR |
|---|---|---|
| positives | 302 aa | 275–411 |
| negatives | 281 aa | 190–428 |

Heavily overlapping, which is why length alone reaches only 0.5823.

---

## Validation and test are not equally hard

A property of this split worth stating, because it affects every comparison drawn from it.

| split | negatives | median BLASTp e-value to the training positives | clearing e ≤ 3.16e-2 |
|---|---|---|---|
| val | 76 | 1.9 | **2/76** |
| test | 62 | 1.0 | **8/62** |

**Test drew esterases markedly more homologous to the training polyester degraders.** Four
times the rate of val, on a smaller negative set.

The positives run the other way — test positives are *closer* to the training set (median
best e-value 3.7e-15 against val's 1e-09), so this is not a remoteness effect. It is which
negatives happened to land where.

Consequence: any method comparison on this split is partly a measurement of how each method
handles hard negatives, and validation understates the difficulty. Both are reported in
`BLAST_vs_ESM2_650M_Polyester.md` Table 1b rather than test alone.

---

## Label provenance: where every label actually came from

Positives and negatives were established by **different kinds of evidence**, and the
asymmetry is the dataset's weakest premise. It is stated here rather than buried.

| class | n | source | basis for the label |
|---|---|---|---|
| `1_pet` | 320 | PAZy / PlasticDB | **301 experimentally verified**, 19 extrapolated |
| `2_other_polyester` | 68 | PAZy / PlasticDB | **39 verified**, 29 extrapolated |
| `heldout_pha` | 36 | PAZy / PlasticDB | 10 verified, 26 extrapolated |
| `3_fold_matched_esterase` | 185 | UniProt `ec:3.1.1.-` | **assumed** non-degrading — never assayed |
| `4_naive_control` | 238 | UniProt, outside EC 3.1 | **assumed** non-degrading — never assayed |

Merged positive set: 542 rows — PAZy only 232, PAZy + PlasticDB 210, PlasticDB only 100.
Each row carries a traceable `source_ids` (e.g. `PAZy:112|PlasticDB:00018`).

**Two consequences.**

1. **A positive means "somebody measured degradation and published it." A negative means
   "nobody has reported it as a plastic degrader"** — which is largely a statement about
   what has been tested. If any of the 423 negatives does degrade a polyester, the model is
   penalised for a correct prediction. PEZy-miner's 36 *assayed* non-degraders are the only
   known source of genuinely verified negatives and remain unextracted.
2. **48 of the 388 modelled positives (12%) are extrapolated, not verified** — 19 PET and
   29 other-polyester. `2_other_polyester` is the weaker half: **29 of 68** rest on
   homology-based annotation rather than measurement. Since that class is what the task
   redefinition *added* to the positive side, the headline partly rests on labels nobody
   measured. `MANUAL_EXCLUSIONS` already catches one such case by hand
   (`PlasticDB:00230` — "named PETase by homology, only ever assayed on PCL").

### Why the negatives are EC 3.1.1

**EC 3.1.1 is the carboxylic-ester hydrolase subclass — the class PETases themselves belong
to** (PET hydrolase is EC 3.1.1.101, MHETase 3.1.1.102, cutinase 3.1.1.74). Drawing
negatives from the same subclass yields proteins that perform the identical chemistry
(cleaving an ester bond), share the alpha/beta-hydrolase fold and the Ser-Asp-His triad, but
act on natural substrates rather than plastic.

This is the project's central methodological commitment: a classifier separating PETases
from *random* proteins has learned "is this a hydrolase," not "is this a plastizyme." For
that reason `fetch_negatives.py` deliberately does **not** filter class 3 at 30% identity —
homology to the positives is the point of the tier. Class 4 (outside EC 3.1) provides the
easy contrast, so the gap between tier-4 and tier-3 performance measures how much of the
score is fold detection rather than function detection.

**Query gotcha, handled.** `ec:3.1.1.*` is a loose string prefix that also matches
3.1.10-3.1.14 (nucleases, which cut DNA and RNA) — 6,174 reviewed entries against 4,294 for
the exact `ec:3.1.1.-`, so 1,880 would have been spurious. The exact form was used.

### Keeping PETases and cutinases out of the negatives

**Every positive in this dataset comes from PAZy or PlasticDB. Every negative comes from
UniProt.** The two sets never mix, and the separation is enforced rather than assumed.

This matters because the class-3 query is *deliberately* broad enough to include plastic
degraders: EC 3.1.1 is the subclass PETase (3.1.1.101), MHETase (3.1.1.102) and cutinase
(3.1.1.74) all belong to. Asking UniProt for "all carboxylic-ester hydrolases" asks for them
too. Three filters remove them, applied in `fetch_negatives.py` before anything is written:

1. **Accession exclusion.** Every UniProt accession appearing anywhere in the positive set is
   banned outright, so a known plastizyme cannot be re-drawn as a negative.
2. **The name filter (`PLASTIC_HINTS`).** A regex run over each candidate's *protein name,
   keywords and EC field* together; if it matches, the entry is dropped:

   ```
   polyethylene terephthalate | PETase | MHETase | terephthalate |
   cutinase | plastic | polyester hydrolase | polyurethan
   ```

   This is the one that does the real work. It catches plastic-active enzymes that PAZy and
   PlasticDB never recorded — anything UniProt itself names as a cutinase or a PET hydrolase
   is removed whether or not our positive set knows about it.
3. **Identity exclusion**, applied later during clustering: any negative >=90% identical to a
   positive is dropped as a probable duplicate accession of the same protein.

**Verified against live UniProt (2026-09-28), not assumed.** Every reviewed entry carrying a
plastic-degrading EC code was re-queried and checked against the filters:

| EC | reviewed entries matching the class-3 query | already in positives | caught by name filter | slipped through |
|---|---|---|---|---|
| 3.1.1.101 (PET hydrolase) | 10 | 8 | 10 / 10 | **0** |
| 3.1.1.102 (MHETase) | 1 | 0 | 1 / 1 | **0** |
| 3.1.1.74 (cutinase) | 61 | 9 | 61 / 61 | **0** |

And confirmed in the delivered data: **0 of the 423 negatives** carry any of those EC codes.

**Why the name filter carries the load.** Only **152 of 388** positives have a UniProt
accession at all — the other 236 come from PlasticDB records without one — so accession
exclusion can only protect against 40% of them. For the rest, the name filter is the sole
barrier. It holds because UniProt's naming convention is consistent: all 72 entries above
carry "cutinase", "terephthalate" or "MHET" in their protein name.

**Known fragility, worth fixing before any re-fetch.** The regex matches *names*, not EC
codes — `PLASTIC_HINTS` does not match the string `3.1.1.101`. So the guarantee currently
rests on the assumption that any plastic-degrading enzyme says so in its name. That is true
of all 72 reviewed entries today, but an entry annotated `EC 3.1.1.101` and named, say,
"Alpha/beta-hydrolase fold protein" would pass. Adding the three EC codes to the exclusion
pattern removes the dependency at no cost.


### Limitation found 2026-09-28: the "fold-matched" tier is only half fold-matched

EC number was used as a proxy for fold, and for this subclass **the proxy leaks**:

| EC | n | enzyme | alpha/beta-hydrolase fold? |
|---|---|---|---|
| 3.1.1.96 | 56 | D-aminoacyl-tRNA deacylase | **no** |
| 3.1.1.29 | 43 | peptidyl-tRNA hydrolase | **no** |
| 3.1.1.- | 27 | unspecified esterase | yes |
| 3.1.1.1 | 6 | carboxylesterase | yes |
| 3.1.1.3 | 6 | triacylglycerol lipase | yes |
| others | 47 | phospholipases, lactonases, etc. | mixed |

**99 of 185 (54%) are tRNA-processing hydrolases.** They sit in EC 3.1.1 by formal reaction
chemistry — they hydrolyse the ester linkage of an aminoacyl-tRNA — but structurally they
are unrelated to cutinases and lipases. Only ~30 entries in the tier are genuine
esterases/lipases/cutinases.

This is the mechanism behind the caveat already recorded in `SAE_Feature_Interpretation.md`
§7, that only 41% of tier-3 sequences have a catalytic serine at the peak position. **Tier 3
is easier than its name claims**, which understates the task's difficulty and correspondingly
overstates performance on it. Re-querying on fold (InterPro/Pfam alpha/beta-hydrolase clan)
rather than EC alone would fix it; until then, tier-3 numbers should be read as a
*mixed* rather than a fold-matched contrast.

---

## Why this is a hard dataset

Worth stating plainly, because most of the design choices in this document are consequences
of it rather than preferences. The difficulty is not that the data is small — though it is —
but that it is **structurally** awkward in four independent ways.

### 1. The positives are one dense family, not a diverse sample

Known plastizymes are overwhelmingly cutinase-like alpha/beta-hydrolases that happen to have
been assayed on a polyester. Across the 388 positives:

| measure | value |
|---|---|
| positive-positive pairs >= 30% identity | 45.0% of all 75,078 pairs |
| positive-positive pairs >= 50% identity | 15.1% |
| median identity among aligned pairs | 0.434 |
| median relatives at >= 50% identity, per positive | 57.5 |
| positives with **no** >= 50% relative | 71 / 388 |

**One homology component holds 256 of 303 training positives (84%)**; the remaining 32
components share 47. This single fact drives most of what follows: any procedure that
assigns whole components — splitting, k-fold CV, bootstrap — has to move 84% of the
positives as one indivisible block.

### 2. A stricter identity bound makes the benchmark *easier*, not harder

The intuitive fix — tighten the cross-split bound from 40% to 30% — backfires. A negative
that is 40% identical to a positive must join that positive's component, and therefore its
split. Tightening the bound systematically **evicts the hard negatives from the test set**:

| class-3 negatives >= 30% identical to a positive | train | val | test |
|---|---|---|---|
| at the 40% bound (current) | 18 | 0 | **8** |
| at a 30% bound | 14 | 11 | **1** |

All six negatives that BLASTp ranks above its own median positive move from test to val. The
30% split would report a stricter-sounding identity bound while measuring an easier
discrimination, on 40% fewer test positives. The 40% bound is the better instrument, and the
reason is worth stating rather than defending.

### 3. Effective sample size is far below the sequence count

The 42 test positives are not 42 independent observations — they occupy 20 homology
components, unevenly:

| effective n | 95% CI half-width, recall ~0.9 |
|---|---|
| 42 (treating sequences as independent — too optimistic) | +/- 9.1 pp |
| **10.6** (Kish, treating each component as one unit) | **+/- 18.0 pp** |

The truth lies between, nearer the lower figure. Cross-validation does not rescue this: the
giant component means a fold either holds it (and is almost all positive) or does not (and
has almost none). Tested directly — two of five folds produced no metric at all. See
PROGRESS.md D10.

### 4. The labels are asymmetric, and the hard negatives are unverifiable

A positive means somebody ran an assay. A negative means nobody has reported the enzyme as a
degrader, which is largely a statement about what has been tested. The closer a negative is
to the positives — i.e. the more useful it is as a hard case — the more likely it is to be a
genuinely untested plastizyme rather than a true negative. **The most informative negatives
are exactly the ones whose labels are least trustworthy.** There is no way to resolve this
from within the data; it needs assayed non-degraders (PEZy-miner).

### What follows from all this

- **Any headline accuracy figure is meaningless without the identity bound and the class
  balance beside it.** Both are quoted throughout this document for that reason.
- **The right claims are mechanistic and comparative, not absolute.** "The probe degrades
  more gracefully than BLASTp as negatives get harder, and here is the feature it uses"
  survives these constraints; "AU-PRC 0.95" does not, on its own.
- **This is within-family discrimination, not remote-homolog detection.** The remote-homolog
  claim cannot be evaluated on curated data, because the independent families do not exist
  in it — an honest negative result, and one the field shares (see `dataset_splits_0.4.md`
  on Balci et al. 2026, whose title states the same conclusion).

---

## What this dataset can and cannot support

**Can:** *does this enzyme degrade a polyester?* 388 positives spanning three substrate
classes (340 experimentally verified, 48 extrapolated), against 423 negatives — of which
185 are nominally fold-matched, though only ~30 are genuine esterases (see above).

**Cannot:** *does this enzyme degrade PET specifically?* That contrast has 48 aliphatic
and 20 PBAT degraders as its negative set — 6 and 5 in test respectively — and every
method tried lands near 0.68 on it. See `SAE_Feature_Interpretation.md` §5.

**One caveat on the negatives.** PAZy and PlasticDB record only confirmed *positives*, so
`3_fold_matched_esterase` means "not annotated as plastic-degrading", never "assayed and
shown inactive". A false positive in that tier may be a correct prediction about an
untested enzyme. Metrics are therefore reported per tier rather than pooled.

**A second caveat on the negatives.** 54% of that tier is not fold-matched at all
(tRNA hydrolases, see above), so tier-3 performance overstates how well the model handles a
true esterase contrast.

## Provenance

| file | contents |
|---|---|
| `artifacts/dataset_splits_0.4.md` | sources, dedup, class rationale, split verification |
| `data/processed/dataset_splits_id40.tsv` | 847 rows: class, split, component |
| `data/sae_650M_L33/polyester_eda.json` | every number in this document |
| `data/processed/plastizymes_merged.tsv` | 542 positives, `sources` + `source_ids` per row |
| `data/processed/uniprot_negatives.tsv` | 423 negatives, EC and phylum per row |
| `src/biopet_sae/fetch_negatives.py` | the UniProt queries, verbatim |
