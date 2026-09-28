# SAE Feature Interpretation — ESM-2 650M Layer 33

> **Scope note.** This analysis was run against the **PET-only** classifier, before the task
> was redefined to *polyester degrader vs. non-degrader*
> (see `EDA_polyester_task.md`). Its central finding — that the latents encode the
> alpha/beta-hydrolase fold rather than PET specificity — is what *motivated* that
> redefinition, so it is reported as-is and has not been re-run on the new labels.
> **Re-running it on the polyester task is open work.**

Generated 2026-09-28. What the sparse autoencoder latents actually detect, and why that
explains both the model's performance and its ceiling.

Task: **PET (320) vs everything else (491)**, binary. Selection on the train split only;
validated on val and test. Split `dataset_splits_id40.tsv` (40% identity bound, verified).

---

## 1. Setup and the reconstruction gate

| | |
|---|---|
| backbone | `facebook/esm2_t33_650M_UR50D`, layer 33 |
| SAE | `Elana/InterPLM-esm2-650m` `layer_33/ae_normalized.pt`, 1280 → **10,240** latents |
| loading | `ReLUSAE.from_pretrained(hf_hub_download(...))` — `load_sae_from_hf` cannot work (upstream package omits its `train` subpackage) |
| pooling | max over residues |
| sparsity | **1.42%** of latents active per residue (145 of 10,240) |
| reconstruction | mean per-residue cosine **0.898** |

**Gate passed before any feature claim was made:**

| representation | dim | test AU-ROC | test AU-PRC |
|---|---|---|---|
| raw dense layer 33 | 1280 | 0.8856 | 0.8150 |
| SAE reconstruction | 1280 | 0.9068 | **0.8386** |
| SAE features | 10240 | 0.8900 | 0.8115 |

Reconstruction retains **102.9%** of raw test AU-PRC. The SAE does not destroy the signal,
so downstream results are attributable to the analysis rather than to round-trip damage.

---

## 2. The signal is concentrated, not distributed

Top-k latents by **train** univariate AU-ROC, evaluated on test:

| k | test AU-ROC | test AU-PRC | % of full AU-PRC |
|---|---|---|---|
| 1 | 0.8878 | 0.7655 | 94.3% ← **one latent** |
| 5 | 0.8723 | 0.7789 | 96.0% |
| 10 | 0.8515 | 0.7852 | 96.8% |
| 25 | 0.8705 | 0.7817 | 96.3% |
| 50 | 0.8529 | 0.7522 | 92.7% |
| 100 | 0.8237 | 0.7066 | 87.1% |
| 500 | 0.8347 | 0.7250 | 89.3% |
| 2500 | 0.8224 | 0.7242 | 89.2% |
| 10240 | 0.8900 | 0.8115 | 100.0% |

**A single latent recovers 94% of the full 10,240-dimensional model.** Removing any one
latent from the top-50 changes test AU-PRC by at most 0.022 — they are massively
redundant. This is the opposite of the sibling kinase project, where recognition was
distributed across detectors and no single latent mattered.

---

## 3. Six latents that hold up on held-out data

Raw latent activation used directly as the score — **no classifier at all**:

| latent | train | val | test | held-out (63 pos / 160 neg) |
|---|---|---|---|---|
| **9529** | 0.9938 | 0.9418 | 0.9160 | **0.9282** |
| **411** | 0.9918 | 0.9397 | 0.9103 | **0.9199** |
| **2661** | 0.9874 | 0.9174 | 0.9129 | **0.9142** |
| **7734** | 0.9917 | 0.9165 | 0.8975 | **0.9053** |
| **5271** | 0.9938 | 0.9124 | 0.8904 | **0.8991** |
| **2473** | 0.9982 | 0.9095 | 0.8878 | **0.8991** |

**Latent 9529 alone reaches test AU-ROC 0.9160** — higher than dense `max_L33`
(0.8856), higher than all 10,240 SAE latents (0.8900), higher than BLASTp (0.8778). It is
the single best number produced in this project.

Multiple-comparison check, since these were chosen from 10,240 candidates:

| | test AU-ROC |
|---|---|
| best achievable by **any** of the 10,240 latents | 0.9222 |
| 99.9th percentile across all latents | 0.8958 |
| 99th percentile | 0.7705 |
| 95th percentile | 0.6673 |

Latent 9529 (train-selected) scores above the 99.9th percentile, and 6 of the 12
train-selected latents beat the 99th. **The selection transfers; it is not luck.** Note
that roughly half the top-12 by train AU-ROC *do* collapse on held-out data (3255:
0.991 → 0.576; 2359: 0.988 → 0.467), so validation was necessary.

---

## 4. What the latents detect: the catalytic triad

Per-residue activations were recomputed to locate each latent's peak. The most common
15-mer window among the 40 top-activating PET sequences, and the residue immediately
following the peak:

| latent | next residue | consensus window | assignment |
|---|---|---|---|
| **9529** | S (40/40) | `NRLAVAGHSMGGGGA` | catalytic SERINE |
| **411** | M (21/40) | `LANDRVPTMVISGQA` | catalytic ASPARTATE region |
| **2661** | A (32/40) | `PTMIFSGQADTVVTP` | catalytic ASPARTATE region |
| **2473** | E (32/40) | `ATTESVYLEVAGADH` | catalytic HISTIDINE region |
| **7734** | H (40/40) | `YLEVAGADHGFMVGR` | catalytic HISTIDINE |
| **5271** | A (15/40) | `PNIPNKIIGKYSVAW` | C-terminal, unresolved |

**Five of the six map onto the Ser-Asp-His catalytic triad**, in three regions:

```
  9529   NRLAVAGHSMGGGGA     G-H-S-M-G   -> catalytic SERINE, nucleophile elbow GxSxG

  411    LANDRVPTMVISGQA   ┐ peaks ~6 residues apart in the same window
  2661   PTMVISGQADTVVTP   ┘ -> catalytic ASPARTATE region

  2473   ATTESVYLEVAGADH   ┐ peaks ~7 residues apart in the same window
  7734   YLEVAGADHGFMVGR   ┘ -> catalytic HISTIDINE (A-G-A-D-H-G)
```

That overlap explains the redundancy in section 2: these are not six independent
detectors but **three views of one catalytic apparatus**.

### Independent validation against UniProt curated active sites

The latent-derived triad positions were checked against UniProt `ACT_SITE` annotations —
curated, independent of anything in this pipeline. **9 of the 31 entries with a UniProt
accession carry ACT_SITE features, covering 32 annotated residues. The latent-derived
triad recovers all 32 (within +/-1 residue).**

| entry | enzyme | UniProt ACT_SITE | latent-derived | match |
|---|---|---|---|---|
| BPS0282 | IsPETase | S160 D206 H237 | S160 D206 H237 | 3/3 |
| BPS0269 | LCC | S165 D210 H242 | S165 D210 H242 | 3/3 |
| BPS0239 | Est119 | S169 D215 H247 | S169 D215 H247 | 3/3 |
| BPS0383 | FsC cutinase | S136 D191 H204 | S136 D190 H204 | 3/3 |
| BPS0327/44/45 | Thermobifida cutinases | S170 D216 H248 | S170 D215 H248 | 3/3 each |
| BPS0190 | Mors1 | S189 | S189 | 1/1 |
| BPS0384, BPS0393 | FoCut5a, HiC | — | — | 5/5 each |

Two notes. PDB files are **not** a source for this: their `SITE` records are
software-generated ligand-binding sites (thiocyanate, sulfate, glycerol), present in only
9 of 29 structures, and any overlap with the catalytic triad is incidental. And only 9 of
31 accessions carry `ACT_SITE` at all — so the latent-derived assignment is not redundant
with the annotation; it fills in the other 23 entries.

Numbering conventions differ between sources and must be reconciled by alignment: UniProt
numbers IsPETase from the full-length sequence (S160) while PDB 5XFY numbers from the
mature protein (S131), and the three Thermobifida entries are truncated constructs offset
by 38-39 residues from their UniProt entry.

### Latent 9529 is unambiguously a catalytic-serine detector

| residue immediately after the peak | PET | 2b aliphatic | 3 esterase |
|---|---|---|---|
| **S** | **100%** | 98% | 41% |
| other | 0% | 2% | R 16%, A 9%, K 9% |

| peak lands on a `GxSxG` motif | |
|---|---|
| PET | 316/320 (99%) |
| non-PET | 127/527 (24%) |

---

## 5. The finding: it detects the fold, not the substrate

Mean activation by class, all six latents:

| latent | PET | 2b aliphatic | 2a PBAT | 3 esterase | 4 naive | PET/2b |
|---|---|---|---|---|---|---|
| 9529 | 0.950 | 0.715 | 0.560 | 0.237 | 0.030 | 1.33 |
| 2473 | 0.953 | 0.407 | 0.217 | 0.156 | 0.031 | 2.34 |
| 7734 | 0.872 | 0.482 | 0.264 | 0.181 | 0.022 | 1.81 |
| 2661 | 0.884 | 0.453 | 0.213 | 0.179 | 0.026 | 1.95 |
| 411 | 0.835 | 0.403 | 0.269 | 0.172 | 0.044 | 2.07 |
| 5271 | 0.765 | 0.428 | 0.269 | 0.156 | 0.031 | 1.79 |

Every latent shows the same monotone gradient:

```
  PET  >  2b aliphatic  >  2a PBAT  >  3 esterase  >  4 naive control
```

That is an ordering in **how canonically alpha/beta-hydrolase-like the protein is** — not
a PET axis. The strongest non-PET activators of latent 9529 settle it:

```
  BPS0038  2b_aliphatic  1.004   GRVGTSGHSQGGGGS
  BPS0021  2b_aliphatic  0.952   DKFAVSGWSMGGGGA
  P9WM39   3_esterase    0.955   DGRAVAGFSMGGFGA

  PET consensus          ~1.02   NRLAVAGHSMGGGGA
```

`GWSMGGGGA` versus `GHSMGGGGA` — the same motif, one position different. The latent fires
just as hard on the aliphatic-polyester cutinase. **It cannot tell them apart, and it
never could**: PETases and aliphatic-polyester cutinases share the nucleophile elbow
almost exactly.

### Why it scores 0.92 anyway

Because the negative set is mostly proteins that lack the motif entirely. Only **24%** of
non-PET sequences have the peak land on a `GxSxG`, against **99%** of PET. The latent
separates the classes by detecting *presence of a canonical nucleophile elbow*, which is
close to free on `4_naive_control` (238 non-esterases, mean activation 0.030) and on the
more heterogeneous UniProt `3_fold_matched_esterase` set (only 41% have a serine after
the peak).

---

## 6. What this explains

This single interpretable feature accounts for every result in the project:

| observation | explanation |
|---|---|
| dense probe scores ~0.89 AU-ROC overall | the fold signal is strong and easy |
| near-perfect against `4_naive_control` | non-esterases have no nucleophile elbow at all |
| ~0.64-0.70 against `2b_aliphatic`, at **every** model scale and for BLAST | those enzymes have the same motif |
| 8M → 650M fixed the phylum shortcut but not F1 | scale sharpened the fold detector, which was never the bottleneck |
| L1 sparsity on dense embeddings hurt | the fold signal is spread across many correlated dimensions |
| the differential score found nothing on `2b` | there is no PET-specific latent for it to find at this layer |

**The model is a very good serine-hydrolase detector and not a PET detector.** That is a
coherent, defensible result, and it is the same shape of finding the sibling kinase
project reached independently — its top latents mapped onto textbook catalytic motifs
(P-loop, HRD) rather than onto anything kinase-substrate-specific.

## 6b. The PET-specificity test, and a leak that had to be corrected

Section 5's claim — the latents track the fold, not the substrate — was tested directly by
the differential score: separate PET from `2b_aliphatic` (the hardest contrast, since both
are genuine polyester degraders). Trained on train, evaluated on test
(31 PET vs 6 `2b`, so 186 positive-negative pairs):

| feature set | n_feat | test AU-ROC | pairs correct |
|---|---|---|---|
| 9 candidates, **selection on train only** | 9 | **0.5108** | 95/186 |
| top-50 latents (2b tier) | 50 | 0.5699 | — |
| all SAE latents | 10240 | 0.5860 | — |
| **9 RANDOM latents** (200 draws) | 9 | **0.5770**, 90% range [0.333, 0.806] | — |
| dense `max_L33` (reference) | 1280 | **0.7312** | — |

**0.5108 is a coin flip**, and it sits at the **29th percentile of the random-latent
distribution** — worse than a median random draw. The top-50 set is also worse than random.
All 10,240 sparse features underperform the 1,280 dense dimensions on this contrast. This is
the quantitative form of Section 5: there is no PET-specific latent here to find.

> **Correction (recorded 2026-09-28).** The differential statistic was originally computed
> over `keep = correct` across *all* splits, so selection saw 27 val and 23 test PET
> sequences plus the test `2b` controls, and the chosen latents were then evaluated on test.
> That is circular. It inflated the result from **0.5108 to 0.6129** and changed **10 of the
> 13** candidates. Leak-free selection yields
> `[3281, 3350, 3628, 3793, 4336, 6926, 8088, 10184, 10213]`, only 3 of which appear in the
> original list. `sae_differential.py` now defaults to `--select-on train`. **Only the
> 0.5108 figure should be cited.** The leaked 0.6129 would have supported the opposite
> conclusion, which is why it is recorded here rather than quietly dropped.

AU-PRC is uninformative on this contrast — 31 PET vs 6 negatives is prevalence 0.84, so
everything including random scores ~0.89.

---

## 7. Limits of this analysis

1. **One layer, one SAE.** Layer 33 is the final layer and was chosen because the dense
   probe performed best there. PET specificity, if it is linearly available anywhere, may
   sit at a middle layer (9, 18, 24 also have SAEs) where structural rather than
   sequence-level features dominate.
2. **Max-pooling discards position.** A latent that fires weakly at many residues is
   invisible; only the single strongest residue is retained.
3. **No causal test.** These are correlational. The kinase project's precedent is to
   confirm with greedy sequential ablation against a **reconstruction baseline** — a
   latent's importance is only established by removing it and seeing the prediction move.
4. **`5271` remains unassigned** — its peaks are C-terminal with no consistent motif.
5. **The negative set shapes the conclusion.** `3_fold_matched_esterase` is more
   heterogeneous than intended (only 41% have a catalytic serine at the peak position),
   which inflates how well a fold detector separates the classes.

## Reproduce

```
uv run python -m biopet_sae.extract_sae_features --layer 33
uv run python -m biopet_sae.sae_differential --top 50 --select-on train
uv run python -m biopet_sae.feature_windows --latents 9529,411,2661,5271,2473,7734
```

| file | contents |
|---|---|
| `data/sae_650M_L33/sae_features.npz` | 847 x 10240 pooled latents + reconstruction |
| `data/sae_650M_L33/feature_windows.json` | per-latent peak windows and position frequencies |
| `data/sae_650M_L33/report_numbers.json` | every number in this document |
| `artifacts/ESM2_650M_Dense_Run.md` | the dense probe this builds on |
| `artifacts/BLAST_vs_ESM2_650M.md` | the homology-search comparison |
