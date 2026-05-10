# Unsupervised Immunophenotyping of PrEP Users and People Living with HIV

Analysis code and per-patient data tables accompanying the manuscript submitted to eBioMedicine.

## Overview

This repository contains the custom scripts and processed data used to perform unsupervised clustering (MiniSom) and differential abundance analysis of CD3+ T cell subsets in:
- Healthy controls (HC, n=12)
- PrEP users (n=10)
- People living with HIV at baseline (W0, n=14) and after 48 weeks of ART (W48, n=14)

## Repository Structure

```
.
├── scripts/
│   ├── 01_normalization/       # Custom batch normalization (peak alignment + median shift)
│   ├── 02_clustering/          # MiniSom SOM + hierarchical metaclustering
│   ├── 03_differential_abundance/  # Kruskal-Wallis + pairwise Mann-Whitney / Wilcoxon
│   ├── 04_figures/             # Dendrograms (R pheatmap), UMAP, volcano plots
│   └── 05_sensitivity/        # Parameter robustness + IPR/IPNR recovery scoring
├── data/
│   ├── cd3/                    # CD3 20-metacluster frequencies, profiles, DA results
│   ├── cd4/                    # CD4 10-metacluster (split from CD3)
│   ├── cd8/                    # CD8 10-metacluster (split from CD3)
│   └── cd4_naive/              # CD4 Naive subclustering
└── README.md
```

## Script-to-Figure Mapping

| Script | Figures |
|--------|---------|
| `02_clustering/minisom_cd3_main.py` | Fig 1a-b (CD3 UMAP, metacluster overview) |
| `02_clustering/flowsom_cd4cd8_mcsweep.py` | Fig 2-3 (CD4/CD8 lineage-specific clustering) |
| `03_differential_abundance/flowsom_DA_full.py` | Fig 4 (differential abundance heatmaps, boxplots) |
| `04_figures/dendro_cd3_unified.R` | Fig 1c (patient dendrogram + heatmap, CD3 level) |
| `04_figures/dendro_cd4cd8_unified.R` | Fig 2-3 (CD4/CD8 patient dendrograms) |
| `04_figures/volcano_prep.R` | Fig 5 (PrEP vs HC / PrEP vs HIV volcano plots) |
| `04_figures/flowsom_umap.py` | Supplementary UMAP panels |
| `05_sensitivity/sensitivity_analysis.py` | Supplementary (parameter robustness) |
| `05_sensitivity/reproduce_recovery_score.py` | Supplementary (IPR/IPNR classification) |

## Data Files

### Metacluster Frequencies
Per-patient percentage of each metacluster (rows = patients, columns = MCs). Patient identifiers are pseudonymized (P1, P3, ...).

- `cd3/cd3_mc20_frequencies.csv` — 50 samples x 20 metaclusters
- `cd4/cd4_mc10_frequencies.csv` — CD4+ subset, 10 metaclusters
- `cd8/cd8_mc10_frequencies.csv` — CD8+ subset, 10 metaclusters
- `cd4_naive/cd4_naive_sub_frequencies.csv` — CD4 Naive subclusters

### Metacluster Profiles
Mean marker expression (z-scored) per metacluster.

### Differential Abundance Results
Pairwise group comparisons with Benjamini-Hochberg FDR correction.

## Analysis Pipeline

1. **Batch normalization** (`batch_norm_cd3_v11.R`): Peak alignment + median shift across 3 acquisition batches using bridge samples, applied to 9 markers on arcsinh-transformed (cofactor 6000) fluorescence values.

2. **Unsupervised clustering** (`minisom_cd3_main.py`): 14-marker MiniSom (10x10 SOM, seed 42) on 5000 cells/sample, Ward hierarchical linkage into 20 metaclusters. CD4/CD8 lineage split followed by independent 10-MC clustering.

3. **Differential abundance** (`flowsom_DA_full.py`): Kruskal-Wallis across 4 groups, pairwise Mann-Whitney U (unpaired) and Wilcoxon signed-rank (paired HIV W0 vs W48), arcsine-sqrt transformation for effect sizes, BH-FDR correction.

4. **Visualization**: Patient-level dendrograms (R `pheatmap`, Ward.D2, Euclidean distance), UMAP (Python `umap-learn`), volcano plots (R).

5. **Sensitivity analysis**: Bootstrap resampling (n=1000), SOM grid sweep, metacluster number sweep, cophenetic correlation.

## Requirements

### Python (>= 3.9)
```
numpy, pandas, scipy, scikit-learn, minisom, fcsparser, umap-learn, matplotlib
```

### R (>= 4.2)
```
flowCore, pheatmap, grid, ggplot2
```

## Usage

Scripts contain a `USER CONFIG` section at the top where input/output paths should be updated. The `data/` directory contains all processed tables needed to reproduce statistical analyses and figures without access to raw FCS files.

## License

This code is provided for academic reproducibility. Please cite the associated publication.
