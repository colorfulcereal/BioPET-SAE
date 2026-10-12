# BioPET-SAE dataset v1 — polyester-degrading enzymes vs. hard negatives

A homology-bounded benchmark for **detecting polyester-degrading enzymes (plastizymes)
from protein sequence**, with the negative set annotated by **fold** rather than by EC
class. 847 proteins, 811 modelled plus 36 held out.

Files in `data/release/`:

| file | contents |
|---|---|
| `biopet_sae_dataset_v1.tsv` | 847 rows × 20 columns, sequences included |
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
alone, with a 95% CI reaching 0.68.

---

## Columns

| column | meaning |
|---|---|
| `seq_id` | `BPS*` for positives (internal), UniProt accession for negatives |
| `split` | `train` / `val` / `test` / `heldout` |
| `y` | 1 = degrader (includes `heldout_pha`), 0 = negative |
| `class_label` | `1_pet`, `2_other_polyester`, `3_fold_matched_esterase`, `4_naive_control`, `heldout_pha` |
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
| `heldout_pha` | 36 | 10 | 26 | 10 | — | — | — |

Provenance: PAZy only 161, PAZy+PlasticDB 179, PlasticDB only 48.
Substrate: PET/aromatic polyester 284, aliphatic polyester 83, aromatic copolyester 21.
`2_other_polyester` is 43% unverified — the weakest part of the positive set.

`heldout_pha` (polyhydroxyalkanoate depolymerases) was never used in training, tuning or
feature selection. It is available as an unseen-substrate generalisation test.

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
β-lactamase-fold (PF00144). Intermediate difficulty, and a useful control for whether a
method is detecting a serine or a fold.

### Catalytic machinery by group — measured, not assumed

| group | n | curated catalytic residues | SAE Ser-neighbourhood latent 9529 fires (>0.3) |
|---|---|---|---|
| plastizymes | 388 | Ser–His–Asp triad | 96.1% |
| `ab_hydrolase_hard_negative` | 51 | Ser–His–Asp, identical | **100.0%** |
| `ec311_wrong_fold` | 36 | Ser nucleophile, other geometry | 13.9% |
| `tRNA_processing` | 99 | **His only, zero Ser** | **0.0%** |
| `naive_control` | 237 | scattered D/H/E/C | 0.0% |
| `heldout_pha` | 36 | Ser–His–Asp triad | 97.2% |

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
| `ec311_wrong_fold` | 23 | 0.9141 | 0.9526 |
| `ab_hydrolase_hard_negative` | 10 | **0.8167** | 0.9550 |

39 of 62 test negatives are separated perfectly, so the headline is a weighted blend in
which most of the weight is free. A BLASTp baseline on the same split gives test AU-ROC
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
5. `heldout_pha` is unused — keep it unused until you want a generalisation number.

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
§22c (catalytic machinery by group), §23 (this release).
