# BioPET-SAE

Detecting **polyester-degrading enzymes** from protein-language-model embeddings, and asking
what the model is using to decide, via sparse autoencoders (SAEs).

Two halves, in order:

1. **Detection.** ESM-2 650M embeddings plus a linear probe, compared against BLASTp on a
   homology-bounded split.
2. **Explainability.** Which pretrained InterPLM SAE latents (ESM-2 650M, layer 33) the
   decision rests on, and whether they point at real catalytic residues.

## The task

- **Positive** — an enzyme experimentally reported to degrade a plastic whose backbone is
  joined by carboxylic ester bonds (PET, PBAT, PLA, PCL, PBS and the other 16 substrates listed
  in `DATASET_CARD.md`). 388 enzymes, from PAZy and PlasticDB.
- **Hard negative** — an alpha/beta-hydrolase-fold protein with the same Ser-His-Asp triad as
  the positives, never reported to degrade a plastic. **51** of them.
- **Easy negatives** — 372 further proteins that are progressively less like a polyester
  hydrolase, down to unrelated controls. No negative was ever assayed on a plastic.

Polyamide, polyurethane, polyolefin and the other non-ester plastics are excluded because a
different bond needs a different catalytic machine, not because they cannot be degraded.

## Where it stands

Test AU-ROC for the ESM-2 650M probe, positives fixed at 42, split by what the negatives are:

| test negatives | n | AU-ROC |
|---|---|---|
| all (the blended headline) | 62 | 0.9681 |
| naive controls | 36 | 1.0000 |
| tRNA-processing hydrolases | 3 | 1.0000 |
| ester-active, different fold | 13 | 0.9890 |
| **alpha/beta hard negatives** | **10** | **0.8167** (component-bootstrap 95% CI 0.606–0.963) |

BLASTp on the same split: test AU-ROC 0.9040, AU-PRC 0.7552. The blended headline is mostly
free weight: 39 of 62 test negatives are separated perfectly. The honest result is the last
row, on 10 proteins.

Two properties of the data bound every claim, and neither is fixable by modelling:

- **Only 51 hard negatives exist, and 1 of them is in validation.** Anything tuned on
  validation was tuned against the easy task.
- **66% of the positives are one homology component** (256 of 388). Intervals must be
  bootstrapped over the `component` column, never over proteins.

On the interpretability side, the selected latents localise to curated catalytic residues in
57 of 59 proteins (96.6%), against a 4.4% median rate for all 10,240 latents run through the
same pipeline. But they read serine-hydrolase machinery, and they are **not** shown to separate
polyester degraders from fold-matched esterases: that comparison is underpowered once homology
is respected. See `PROGRESS.md` §22–§22c and §28d.

## Layout

| path | what it is |
|---|---|
| `PROGRESS.md` | the project log and source of truth, with every correction recorded |
| `DATASET_CARD.md` | the dataset: definitions, composition, splits, limitations |
| `handoff/` | self-contained package for replicating the experiment elsewhere — start at `handoff/AGENTS.md` |
| `data/release/` | the annotated dataset (TSV and FASTA) |
| `data/processed/` | merged positives and the identity-bounded splits |
| `src/biopet_sae/` | pipeline code |
| `artifacts/` | written analyses and the HTML pages published as claude.ai artifacts |
| `CLAUDE.md` | conventions for working in this repository |

Rebuild the release and the hand-off package:

```
uv run python -m biopet_sae.build_release [--refresh]
uv run python -m biopet_sae.build_handoff
```

`--refresh` re-queries UniProt instead of reusing the cached annotation.

## Known limitations

- The fold of the **positives** was never verified, only their substrate. Three are known not to
  be alpha/beta hydrolases. See `PROGRESS.md` §28a.
- The SAE evidence is correlational. No latent has been ablated.
- Test has been read once and is spent; further tuning needs a fresh set.
- Fewer than 50 experimentally verified plastics-active enzymes existed in the literature when
  PAZy was published (2022), so positive diversity cannot be added by collecting more data.
