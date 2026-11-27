# BioPET-SAE dataset v1 — polyester-degrading enzymes vs. hard negatives

A homology-bounded benchmark for **detecting polyester-degrading enzymes (plastizymes)
from protein sequence**, with the negative set annotated by **fold** rather than by EC
class. 811 proteins (388 positive, 423 negative). The 36 PHA depolymerases that carried no claim
were removed (see below). A self-contained package for a second team is in `handoff/`.

## v1.1 (2026-11-23): 16 PET enzymes added

`biopet_sae_dataset_v1_1.{tsv,fasta}` = v1 + 16 PET enzymes that PAZy published after our
2026-09-27 export, − 1 negative. **v1 is frozen** and every number in this card (and every
probe/SAE result) refers to it. v1.1 is built by `src/biopet_sae/add_pazy_update.py`
(report: `data/release/biopet_sae_dataset_v1_1_report.json`; raw PAZy records:
`data/raw/pazy_update/`). A `batch` column marks `v1` vs `v1.1` rows.

| | v1 | v1.1 |
|---|---|---|
| rows | 811 | 826 |
| positives / negatives | 388 / 423 | 404 / 422 |
| hard negatives (train / val / test) | 51 (40 / 1 / 10) | 50 (39 / 1 / 10) |
| positives train / val / test | 303 / 43 / 42 | 310 / 43 / 51 |
| positive components; largest | 74; `BPS0190` 256 (66%) | 77; `BPS0190` 259 (64%) |

- **Placement by homology, not at random.** Each new sequence was aligned (same global
  identity as the original splits) to all v1 sequences; one at ≥ 0.40 joins that component
  and split. Nine land in test components (HaloPETase1–4, R75estA, BhEstB, LipSh1–3);
  putting them in train would leak. Seven join train; three (LactoPETase, SM0281, NmCut)
  have no relative at ≥ 0.40 and go to train. Maximum cross-split identity of the additions:
  0.3967. **The nine new test positives have never been scored**; compare to published
  numbers on `batch == v1` only.
- **One negative dropped:** `P37957` (*B. subtilis* LipA, an α/β hard negative) is 94%
  identical to `LipBv`, which PAZy reports as PET-degrading. By the original rule (a negative
  ≥ 90% identical to a positive is a label conflict) it was removed. It is also a reminder
  that "never assayed on a plastic" is an assumption.
- All 16 are PAZy-verified, PET (SM0281 also PLA), with sequences and a DOI. 19 polyurethane
  and 8 polyamide entries from the same update are excluded as non-ester.
- Not done for the 16: UniProt cross-checks, InterPro fold annotation.

## v1.2 (2026-11-25): 400 more hard negatives, new split

`biopet_sae_dataset_v1_2.{tsv,fasta}`: 1,226 rows (404 positives; 450 hard negatives = 50 existing + 400
from UniProt; 36 wrong-fold, 99 tRNA-processing, 237 naive). New 40%-identity split: train 789 / val 237 /
test 200, hard negatives 225 / 112 / 113. **Test is not comparable to v1 or v1.1.** The new hard negatives are
76% eukaryotic while positives are mostly bacterial; taxonomy alone gives AU-ROC 0.80-0.88 on the hard cell,
so report the bacteria-only cell and the taxonomy baseline. Full record: `paper/v1.2.md` section 12 and
`data/release/eda_v1_2/EDA_v1_2.md`. The tables below describe v1.1.

Files in `data/release/`:

| file | contents |
|---|---|
| `biopet_sae_dataset_v1.tsv` | 811 rows × 20 columns, sequences included |
| `biopet_sae_dataset_v1.fasta` | same sequences, header carries split/class/group |

Rebuild with `uv run python -m biopet_sae.build_release [--refresh]`. The `--refresh` flag
re-queries UniProt; otherwise the annotation cache at
`data/processed/uniprot_negative_annotation.json` is reused.

---

## Read this first

Two properties of this dataset bound every claim you can make with it. They are not
defects to be worked around; they are the actual difficulty of the problem.

**1. There are only 51 genuinely fold-matched negatives, and 1 of them is in validation.**
A classifier that separates plastizymes from random proteins has learned "is this a
hydrolase." Only the α/β-hydrolase-fold negatives test the real question. Of 423 negatives,
51 qualify — 40 train, **1 val**, 10 test.

**2. 66% of the positives are one homology component.** 388 modelled positives collapse
into 74 components; the largest, `BPS0190`, holds 256 of them. Any per-protein metric
effectively counts that one cutinase family 256 times. **Use a cluster bootstrap over the
`component` column, not a per-protein one.** A per-protein confidence interval on this data
is wrong, not merely optimistic.

Both are why the headline AU-ROC of 0.9681 falls to 0.82 against fold-matched negatives
alone, with a component-bootstrap 95% CI of [0.61, 0.96].

---

## Columns

| column | meaning |
|---|---|
| `seq_id` | `BPS*` for positives (internal), UniProt accession for negatives |
| `split` | `train` / `val` / `test` |
| `y` | 1 = degrader, 0 = negative |
| `class_label` | `1_pet`, `2_other_polyester`, `3_fold_matched_esterase`, `4_naive_control` |
| `negative_group` | **the corrected, fold-based negative classification — prefer this over `class_label`** |
| `ab_hydrolase_fold` | 1 if UniProt/InterPro `IPR029058` |
| `gxsxg_elbow` | 1 if the sequence contains the `G.[ST].G` nucleophile elbow |
| `n_curated_active_sites` | count of curated UniProt `ACT_SITE` residues (negatives only) |
| `ec`, `pfam` | UniProt EC numbers and Pfam accessions (negatives only) |
| `activity_verified` | 1 = measured degradation in the literature |
| `extrapolated_only` | 1 = label inferred, never directly assayed |
| `component` | **homology component — group by this for any CV or bootstrap** |
| `cluster` | finer identity cluster |
| `substrates`, `source`, `organism`, `phylum`, `seq_len`, `sequence` | as named |

`class_label` is kept for continuity with published results. **`3_fold_matched_esterase` is
a misnomer** — only 50 of its 185 members carry the α/β hydrolase fold. Use
`negative_group`.

---

## Composition

### Positives — 388 modelled, all from assay literature (PAZy + PlasticDB)

| class | n | verified | unverified | extrapolated-only | train | val | test |
|---|---|---|---|---|---|---|---|
| `1_pet` | 320 | 301 | 19 | 15 | 257 | 32 | 31 |
| `2_other_polyester` | 68 | 39 | 29 | 14 | 46 | 11 | 11 |
| **total** | **388** | **340** | **48** | **29** | 303 | 43 | 42 |

Provenance: PAZy only 161, PAZy+PlasticDB 179, PlasticDB only 48.
Substrate: PET/aromatic polyester 284, aliphatic polyester 83, aromatic copolyester 21.
`2_other_polyester` is 43% unverified — the weakest part of the positive set.

**Removed: 36 PHA depolymerases.** They carried no claim. They were never used in training, tuning or feature selection, but they were *not* a generalisation test either: 14 of the 36 share a homology component with a trained positive, and 25 trained
positives already act on a PHA, so "unseen lineage" was never true (PROGRESS.md §25). The
generalisation claim was withdrawn (§26). They are no longer shipped in the release file or the handoff; intermediate files under `data/processed/` still hold them (class `heldout_pha`).

### Negatives — 423, from UniProt, **none ever assayed**

Every negative label is an assumption of non-degradation, not a measurement.

| `negative_group` | n | `class_label` = tier 3 | train | val | test |
|---|---|---|---|---|---|
| `ab_hydrolase_hard_negative` | **51** | 50 | 40 | **1** | 10 |
| `ec311_wrong_fold` | 36 | 36 | 23 | **0** | 13 |
| `tRNA_processing_no_ser_nucleophile` | 99 | 99 | 56 | **40** | 3 |
| `naive_control` | 237 | 0 | 166 | 35 | 36 |
| **total** | **423** | **185** | 285 | 76 | 62 |

**Why `tRNA_processing` is separated out.** 99 negatives are D-aminoacyl-tRNA deacylase
(EC 3.1.1.96, Pfam PF02580, n=56) and peptidyl-tRNA hydrolase (EC 3.1.1.29, PF01195, n=43).
They match on EC *reaction class* — carboxylic-ester hydrolase — but share neither the fold
nor the catalytic machinery: 43 have a single curated active site, always a His "proton
acceptor", and **not one has an annotated serine nucleophile**; the other 56 have no active
site annotation at all. They are trivially separable and inflate any metric computed against
them. They are in the dataset because `ec:3.1.1.-` returns them; they are kept, labelled, so
you can exclude them deliberately.

**Why `ec311_wrong_fold` is separated out.** 36 negatives are ester hydrolases with a serine
nucleophile in a *different* fold — GDSL/SGNH (PF13472, PF00657), patatin (PF01734),
β-lactamase-fold (PF00144). The probe separates them almost as easily as naive controls (test AU-ROC 0.989, n=13), but the
SAE serine latent fires on only 13.9% of them, so they remain a useful control for whether a
method detects a serine or a fold.

### Catalytic machinery by group — measured, not assumed

| group | n | curated catalytic residues | SAE Ser-neighbourhood latent 9529 fires (>0.3) |
|---|---|---|---|
| plastizymes | 388 | Ser–His–Asp triad | 96.1% |
| `ab_hydrolase_hard_negative` | 51 | Ser–His–Asp, identical | **100.0%** |
| `ec311_wrong_fold` | 36 | Ser nucleophile, other geometry | 13.9% |
| `tRNA_processing` | 99 | **His only, zero Ser** | **0.0%** |
| `naive_control` | 237 | scattered D/H/E/C | 0.0% |

The hard negatives are **not** distinguishable from the positives by catalytic machinery.
That is the point of the benchmark.

---

## Splits

Homology-bounded, built with identity-based partitioning at a 40% ceiling.

| split | pos | neg | total | prevalence | fold-matched negatives |
|---|---|---|---|---|---|
| train | 303 | 285 | 588 | 0.515 | 40 |
| val | 43 | 76 | 119 | 0.361 | **1** |
| test | 42 | 62 | 104 | 0.404 | **10** |

Maximum cross-split sequence identity: train/val 0.3967, train/test 0.3800,
val/test 0.3983. No pair across splits exceeds 40%.

**Validation is not a miniature of test.** Its 76 negatives are 40 tRNA hydrolases plus 35
naive controls plus 1 fold-matched esterase. Model selection or threshold tuning on val is
tuning against the easy task — in our own run this is why the operating threshold carried
from val gave test precision of 0.830. If you tune on val, say so, and report test
decomposed by `negative_group`.

---

## Reference results (ESM-2 650M, max-pooled layer 33, linear probe)

Test AU-ROC decomposed by negative group, positives fixed at 42:

| negatives | n | AU-ROC | AU-PRC |
|---|---|---|---|
| all — headline | 62 | 0.9681 | 0.9526 |
| `naive_control` | 36 | 1.0000 | 1.0000 |
| `tRNA_processing` | 3 | 1.0000 | 1.0000 |
| `ec311_wrong_fold` | 13 | 0.9890 | 0.9967 |
| `ab_hydrolase_hard_negative` | 10 | **0.8167** | 0.9550 |

39 of 62 test negatives are separated perfectly, so the headline is a weighted blend in
which most of the weight is free. The hard-negative cell's 95% interval, bootstrapped over
homology components (20 positive and 7 negative components), is **[0.606, 0.963]**. Earlier
drafts reported 0.9141 for `ec311_wrong_fold`; that figure belongs to the 23 proteins in
wrong-fold plus alpha/beta hard negatives combined (13 + 10). A BLASTp baseline on the same split gives test AU-ROC
0.9040 / AU-PRC 0.7552. An amino-acid-composition baseline gives 0.7020 (and 0.8926 under
random CV — a 0.19 inflation that shows what a non-homology-aware split buys you).

---

## How to report results on this dataset

1. **Report test decomposed by `negative_group`.** A single headline number is not
   interpretable here.
2. **Bootstrap over `component`, not over proteins.** With 66% of positives in one
   component, per-protein intervals are invalid.
3. **Say whether you tuned on val**, and remember val has 1 fold-matched negative.
4. **State n for the fold-matched cell.** It is 10 in test. Any claim there is low-powered.
5. The PHA depolymerases were removed and carry no claim; do not describe PHA as a generalisation set.

---

## Known limitations

- **51 fold-matched negatives is too few**, and the val allocation of 1 makes val unusable
  for the hard task. Expanding this is the single highest-value fix: UniProt reviewed,
  `IPR029058` plus three curated `ACT_SITE` residues, never assayed on a plastic, excluding
  the cutinase/PETase families. Thousands are available; 300–400 would make every split
  usable.
- **Positive homology density** (66% one component) limits power independently of the
  negative set. Needs one-vote-per-family weighting or genuine positive diversity beyond
  the dominant cutinase clade.
- **Negative labels are assumptions.** No negative was assayed for polyester degradation.
  Some may be weak degraders. PEZy-miner's 36 assayed non-degraders are not yet incorporated
  and would be a genuine measured-negative set.
- **48 of 388 positives are unverified**, 29 of those extrapolated-only. Filter on
  `activity_verified` for a stricter positive set at the cost of n.
- **The fold of the positives was never verified.** The original filter was a 16-string
  blocklist matched against PlasticDB enzyme names. It was only reached on the
  `2_other_polyester` branch (all 320 `1_pet` bypassed it), and 161 of 388 positives carry no
  enzyme label at all. Of the 152 positives with a UniProt accession, 111 carry `IPR029058`,
  38 have no InterPro annotation, and 3 carry a different fold: `BPS0013` (beta-lactamase),
  `BPS0394` (SGNH hydrolase) and `BPS0016` (Proteinase K-like subtilisin that degrades PLA).
  The positive definition is therefore substrate-based by design, and fold is a property to
  report. Running InterProScan over all 388 sequences would settle it.
- **`3_fold_matched_esterase` is mislabelled** (see above). Retained only for continuity.
- Tier assignment used EC codes; a rebuild on InterPro/Pfam fold is the right design and is
  what `negative_group` approximates post hoc.

---

## Provenance and reproduction

Positives: PAZy and PlasticDB, each row carrying traceable `source_ids` in
`data/processed/plastizymes_merged.tsv`. 118 candidates were excluded during assembly — 109
for a non-ester mechanism (polyamide, polyurethane, polyolefin, rubber, polycarbonate,
polyether), 8 for the wrong fold (proteases degrading PLA, chitinases), 1 manually
(`BPS0023`, named a PETase by homology but only ever assayed on PCL).

Negatives: UniProt queries in `src/biopet_sae/fetch_negatives.py`, with a phylum quota and
a filter removing anything resembling a known plastizyme.

Annotation in this release: UniProt REST, fields `ec`, `xref_pfam`, `xref_interpro`,
`ft_act_site`, fetched 2026-10-11 and cached.

## References

| citation | DOI |
|---|---|
| Danso D, Chow J, Streit WR (2019). Plastics: environmental and biotechnological perspectives on microbial degradation. *Appl Environ Microbiol* 85, e01095-19. — basis for the hydrolysable-backbone vs C–C-backbone split, i.e. the 109 non-ester-mechanism exclusions | `10.1128/AEM.01095-19` |
| Wei R, Tiso T, Bertling J, O'Connor K, Blank LM, Bornscheuer UT (2020). Possibilities and limitations of biotechnological plastic degradation and recycling. *Nature Catalysis* 3, 867–871. — basis for scoping the task to polyester degraders | `10.1038/s41929-020-00521-w` |
| Buchholz PCF, Feuerriegel G, Zhang H, Perez-Garcia P, Nover L-L, Chow J, Streit WR, Pleiss J (2022). Plastics degradation by hydrolytic enzymes: the plastics-active enzymes database — PAZy. *Proteins* 90, 1443–1456. — **cite this if you use the positives**; PAZy supplies 340 of 388 | `10.1002/prot.26325` |
| Wei R, Zimmermann W (2017). Microbial enzymes for the recycling of recalcitrant petroleum-based plastics: how far are we? *Microbial Biotechnology* 10, 1308–1322. — PE/PS/PUR/PET compared side by side; context for PET being 82.5% of the positives | `10.1111/1751-7915.12710` |

---

Full experimental history, including every correction to these numbers, is in
`PROGRESS.md` — §21 (negative set), §22/22b (latent selection and its homology confound),
§22c (catalytic machinery by group), §23 (this release), §24 (positive substrates),
§25–§26 (PHA claim retired, task definition), §27 (hand-off package), §28 (fold audit of the
positives, BPS0016, BPS0383, the three nulls).
