# paper/v1_2_phase1: numbers behind the v1.2 probe results (generated 2026-11-25)

Tracked copies of files from `results/v1_2/` (gitignored). The narrative is in `paper/v1.2.md`, sections 12-17.
Probe = ESM-2 650M, max-pool layer 33, linear logistic regression; 5 runs (seeds 42-46); C=0.01, eta0=0.0001;
threshold 0.15 (best mean validation F1). Test = 200 proteins (43 positives); val = 237 (44 positives).

| file | contents | made by |
|---|---|---|
| `report.txt` | readable summary: threshold table, chosen threshold, precision targets, AU-ROC/AU-PRC by cell | `polyester_phase1_v12.py` |
| `phase1.json` | the same, machine-readable (means and SEs) | same |
| `threshold_table_{val,test}.csv` | per threshold (0.001-0.999): precision, recall, F1, FPR, TP/FP/FN/TN, each mean and SE over 5 runs | same |
| `roc_curve_{val,test}.csv` | mean TPR against mean FPR per threshold, with SEs (for the ROC plot) | same |
| `pr_curve_{val,test}.csv` | mean precision against mean recall per threshold, with SEs and the number of runs with defined precision (for the PR plot) | same |
| `blast_threshold_table_{val,test}.csv`, `blast_roc_curve_*.csv`, `blast_pr_curve_*.csv`, `blast_phase1.json` | BLASTp equivalents (one deterministic run, no SE; e-value cut-offs 10 to 1e-30) | `blast_phase1_v12.py` |
| `sweep_grid_C_eta0.json` | the 20-cell C x eta0 grid with validation log-loss, AU-PRC, AU-ROC, train log-loss; selected cell | `polyester_sweep.py` |
| `cells_matched_controls.json` | probe, BLASTp, domain-only and length-only baselines by cell (all hard negatives, bacterial hard negatives, matched bacterial cell, ...) on val and test | `polyester_cells_v12.py` |
| `domain_probe.json` | can ESM-2 embeddings predict bacteria vs eukaryote (AU-ROC 0.996 / 0.999) | `domain_probe_v12.py` |
| `paired_probe_vs_blast_all_negatives.json` | paired component-bootstrap difference, probe minus BLASTp, all negatives only | inline run (provenance in the file) |
| `scores_5_runs.npz` | per-protein probe probability from each of the 5 runs (to redo any plot) | `polyester_phase1_v12.py` |
| `blast_and_probe_seed_mean_scores.npz` | per-protein seed-mean probe score and BLASTp score | `polyester_cells_v12.py` |

Dataset behind these numbers: `data/release/biopet_sae_dataset_v1_2.tsv` (1,226 rows); distribution tables in
`data/release/eda_v1_2/`; build report `data/release/biopet_sae_dataset_v1_2_report.json`.

Reading rules: always read AU-PRC beside AU-ROC with the prevalence floor (test 0.21, val 0.19). The seed SE is
training variation only; BLASTp has no SE. v1.2 numbers are not comparable with v1 or v1.1 (different split).
