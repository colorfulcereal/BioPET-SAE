# Archive: v1 results (frozen 2026-11-24)

A byte-for-byte snapshot of everything the v1 experiment produced, taken before the
experiment is re-run on dataset v1.1. **Do not edit, and do not overwrite from the pipeline.**
Integrity: `shasum -a 256 -c MANIFEST.sha256` from this directory.

| dir | contents |
|---|---|
| `release/` | `biopet_sae_dataset_v1.{tsv,fasta}` (811 rows) and `biopet_sae_dataset_v1_1.*` as of the snapshot |
| `processed/` | merged positives, splits (`dataset_splits_id40.tsv`, 847 rows incl. PHA), pair identities, UniProt negative annotations |
| `embeddings_650M/` | ESM-2 650M embeddings, layers 1/9/18/24/30/33, mean and max pooled |
| `sae_650M_L33/` | SAE features, latent selection, localisation, nulls, probe scores, atlas views |
| `results/` | EDA report, split-frontier and external-split experiments |
| `handoff_as_of_2026-11-24/` | the replication package as it was (v1.1 build) |
| `PROGRESS.md`, `DATASET_CARD.md` | the logs at snapshot time |

v1 headline numbers (test n=104: 42 pos / 62 neg): blended AU-ROC 0.9681, AU-PRC 0.9526;
alpha/beta hard negatives (n=10) AU-ROC 0.8167, component-bootstrap CI [0.606, 0.963]; 57/59
catalytic localisation; null 4.4% median over 10,240 latents. Not archived (unchanged by v1.1 and
large): `data/mgnify/`, `data/pdb_cache/`, `data/embeddings_8M/`, `artifacts/` (tracked in git).
