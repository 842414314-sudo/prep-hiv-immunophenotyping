#!/usr/bin/env python3
"""
FlowSOM UMAP Visualization
===========================
Extends the FlowSOM pipeline (ORIGINAL_flowsom_analysis.py) with UMAP embedding
of individual cells. Reuses the saved SOM, StandardScaler, and metacluster
assignments from OUTPUT_DIR + '/flowsom_results.pkl' so the clustering result remains
identical to the FlowSOM run.

Figures generated:
  1. FlowSOM_UMAP_Metacluster.png   - UMAP colored by FlowSOM metacluster
  2. FlowSOM_UMAP_Group.png         - UMAP colored by clinical group
  3. FlowSOM_UMAP_Timepoint.png     - UMAP colored by timepoint (HIV only)
  4. FlowSOM_UMAP_Markers.png       - UMAP grid, one panel per marker
  5. FlowSOM_UMAP_Density_Group.png - per-group cell density facets
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import fcsparser
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ======================================================================
# USER CONFIG — update these paths to match your local setup
# FCS_DIR: directory containing batch-normalized CD3+ FCS files
# OUTPUT_DIR: where figures and results will be saved
# ======================================================================
FCS_DIR    = './normalized_fcs'
OUTPUT_DIR = './output'

# ============================================================
# Configuration
# ============================================================
base    = FCS_DIR
fig_dir = os.path.join(OUTPUT_DIR, 'Fig')
os.makedirs(fig_dir, exist_ok=True)

SEED              = 42
CELLS_PER_SAMPLE  = 2000   # subsample per FCS for UMAP (smaller than SOM for speed)
UMAP_N_NEIGHBORS  = 30
UMAP_MIN_DIST     = 0.3
UMAP_METRIC       = 'euclidean'

np.random.seed(SEED)

# ============================================================
# Step 1: Load saved FlowSOM results (SOM + scaler + MC assignments)
# ============================================================
print("=" * 60)
print("STEP 1: Loading FlowSOM pickle")
print("=" * 60)

with open(os.path.join(OUTPUT_DIR, 'flowsom_results.pkl'), 'rb') as pf:
    R = pickle.load(pf)

som                    = R['som']
scaler                 = R['scaler']
metacluster_assignments = R['metacluster_assignments']
mc_annotations         = R['mc_annotations']
clustering_markers     = R['clustering_markers']
group_colors           = R['group_colors']

N_METACLUSTERS = int(metacluster_assignments.max())
SOM_X, SOM_Y   = som.get_weights().shape[:2]
print(f"  SOM grid: {SOM_X}x{SOM_Y}  |  metaclusters: {N_METACLUSTERS}")
print(f"  markers : {clustering_markers}")

# ============================================================
# Step 2: Re-read FCS files and subsample for UMAP
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Reading FCS files and subsampling cells for UMAP")
print("=" * 60)

with open(os.path.join(OUTPUT_DIR, 'flowsom_filelist.json')) as jf:
    file_list = json.load(jf)

all_cells    = []
sample_labels = []
sample_info  = []

for i, fl in enumerate(file_list):
    fname = os.path.basename(fl['file'])
    print(f"  [{i+1:>2}/{len(file_list)}] P{fl['patient']} {fl['group']} {fl['timepoint']} - {fname}")

    _, data = fcsparser.parse(fl['file'], reformat_meta=True)
    available = [m for m in clustering_markers if m in data.columns]
    cell_data = data[available].dropna()

    n_cells = len(cell_data)
    if n_cells > CELLS_PER_SAMPLE:
        idx       = np.random.choice(n_cells, CELLS_PER_SAMPLE, replace=False)
        cell_data = cell_data.iloc[idx]

    all_cells.append(cell_data.values)
    sample_labels.extend([i] * len(cell_data))
    sample_info.append({
        'sample_idx' : i,
        'patient'    : fl['patient'],
        'group'      : fl['group'],
        'timepoint'  : fl['timepoint'],
        'batch'      : fl['batch'],
        'label'      : f"P{fl['patient']}_{fl['group']}_{fl['timepoint']}",
    })

combined      = np.vstack(all_cells)
sample_labels = np.array(sample_labels)
print(f"\n  Total cells for UMAP: {combined.shape[0]} ({combined.shape[1]} markers)")

# Apply saved scaler (do NOT refit - ensures consistency with SOM)
combined_scaled = scaler.transform(combined)

# ============================================================
# Step 3: Assign cells to SOM nodes and metaclusters
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Mapping cells to FlowSOM metaclusters")
print("=" * 60)

winners       = np.array([som.winner(x) for x in combined_scaled])
node_ids      = winners[:, 0] * SOM_Y + winners[:, 1]
cell_mc       = metacluster_assignments[node_ids]  # 1-based
print(f"  Metacluster distribution: "
      f"{dict(zip(*np.unique(cell_mc, return_counts=True)))}")

# ============================================================
# Step 4: Compute UMAP
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 4: Running UMAP (n_neighbors={UMAP_N_NEIGHBORS}, "
      f"min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC})")
print("=" * 60)

reducer = umap.UMAP(
    n_neighbors  = UMAP_N_NEIGHBORS,
    min_dist     = UMAP_MIN_DIST,
    metric       = UMAP_METRIC,
    random_state = SEED,
    verbose      = True,
)
emb = reducer.fit_transform(combined_scaled)
print(f"  UMAP embedding shape: {emb.shape}")

# Save embedding + annotations to disk for reuse
umap_df = pd.DataFrame({
    'UMAP1'      : emb[:, 0],
    'UMAP2'      : emb[:, 1],
    'sample_idx' : sample_labels,
    'metacluster': cell_mc,
    'node_id'    : node_ids,
})
for col_name in ('patient', 'group', 'timepoint', 'batch'):
    umap_df[col_name] = [sample_info[s][col_name] for s in sample_labels]
umap_df.to_csv(os.path.join(fig_dir, 'FlowSOM_UMAP_embedding.csv'), index=False)
print(f"  Saved embedding: FlowSOM_UMAP_embedding.csv")

# Cache embedding to pickle too
with open(os.path.join(OUTPUT_DIR, 'flowsom_umap.pkl'), 'wb') as pf:
    pickle.dump({
        'emb'                  : emb,
        'cell_mc'              : cell_mc,
        'sample_labels'        : sample_labels,
        'node_ids'             : node_ids,
        'sample_info'          : sample_info,
        'combined_scaled'      : combined_scaled,
        'clustering_markers'   : clustering_markers,
        'metacluster_assignments': metacluster_assignments,
        'mc_annotations'       : mc_annotations,
        'group_colors'         : group_colors,
    }, pf)

# ============================================================
# Helpers for plotting
# ============================================================
mc_cmap = plt.cm.get_cmap('tab20', N_METACLUSTERS)

# Shuffle index so overplotting isn't biased by file order
rng        = np.random.default_rng(SEED)
shuffle_ix = rng.permutation(len(emb))
emb_s      = emb[shuffle_ix]
cell_mc_s  = cell_mc[shuffle_ix]
samp_s     = sample_labels[shuffle_ix]
cs_s       = combined_scaled[shuffle_ix]

groups_per_cell     = np.array([sample_info[s]['group']     for s in samp_s])
timepoints_per_cell = np.array([sample_info[s]['timepoint'] for s in samp_s])

# Composite group label that splits HIV by timepoint
def _cohort(g, tp):
    if g == 'HIV':
        return f'HIV {tp}'
    return g
cohort_per_cell = np.array([_cohort(g, tp) for g, tp in
                            zip(groups_per_cell, timepoints_per_cell)])

cohort_colors = {
    'HIV W0' : '#E74C3C',
    'HIV W48': '#C0392B',
    'PrEP'   : '#3498DB',
    'HC'     : '#2ECC71',
}
cohort_order = ['HIV W0', 'HIV W48', 'PrEP', 'HC']

# ============================================================
# FIGURE 1: UMAP colored by metacluster
# ============================================================
print("\nGenerating Figure 1: UMAP by metacluster")
fig, ax = plt.subplots(figsize=(11, 9))
for mc in range(1, N_METACLUSTERS + 1):
    m = cell_mc_s == mc
    ax.scatter(emb_s[m, 0], emb_s[m, 1],
               s=1.2, c=[mc_cmap(mc - 1)], alpha=0.55,
               linewidths=0, rasterized=True)

# Label each metacluster at its centroid
for mc in range(1, N_METACLUSTERS + 1):
    m = cell_mc == mc
    if m.sum() == 0:
        continue
    cx, cy = emb[m, 0].mean(), emb[m, 1].mean()
    ax.text(cx, cy, f"MC{mc}", fontsize=9, fontweight='bold',
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='black',
                      boxstyle='round,pad=0.2', alpha=0.85))

ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
ax.set_title('FlowSOM Metaclusters on UMAP (CD3+ T cells)',
             fontsize=13, fontweight='bold')
# Legend with MC annotations
handles = [mpatches.Patch(color=mc_cmap(i), label=mc_annotations[i])
           for i in range(N_METACLUSTERS)]
ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
          fontsize=7, frameon=True, title='Metacluster', title_fontsize=9,
          ncol=1)
ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'FlowSOM_UMAP_Metacluster.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(fig_dir, 'FlowSOM_UMAP_Metacluster.pdf'),
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved: FlowSOM_UMAP_Metacluster.png/pdf")

# ============================================================
# FIGURE 2: UMAP colored by cohort (HIV split into W0 / W48)
# ============================================================
print("Generating Figure 2: UMAP by cohort (HIV W0 / HIV W48 / PrEP / HC)")
fig, ax = plt.subplots(figsize=(10, 8))
for grp in cohort_order:
    m = cohort_per_cell == grp
    ax.scatter(emb_s[m, 0], emb_s[m, 1],
               s=1.2, c=cohort_colors[grp], alpha=0.35, linewidths=0,
               label=f"{grp} (n={m.sum()})", rasterized=True)
ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
ax.set_title('UMAP by Cohort', fontsize=13, fontweight='bold')
leg = ax.legend(fontsize=10, loc='best', frameon=True, markerscale=6)
for lh in leg.legend_handles:
    lh.set_alpha(1.0)
ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'FlowSOM_UMAP_Group.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved: FlowSOM_UMAP_Group.png")

# ============================================================
# FIGURE 3: UMAP colored by timepoint (HIV only - paired W0/W48)
# ============================================================
print("Generating Figure 3: UMAP by timepoint (HIV)")
hiv_mask = (groups_per_cell == 'HIV')
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
tp_colors = {'W0': '#F39C12', 'W48': '#8E44AD'}
for ax, tp in zip(axes, ['W0', 'W48']):
    # background: all cells in gray
    ax.scatter(emb_s[:, 0], emb_s[:, 1],
               s=0.6, c='#DDDDDD', alpha=0.4, linewidths=0, rasterized=True)
    m = hiv_mask & (timepoints_per_cell == tp)
    ax.scatter(emb_s[m, 0], emb_s[m, 1],
               s=1.3, c=tp_colors[tp], alpha=0.6, linewidths=0,
               rasterized=True)
    ax.set_title(f'HIV {tp}  (n_cells={m.sum()})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
    ax.grid(True, alpha=0.2)
fig.suptitle('HIV Cohort: UMAP W0 vs W48', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(fig_dir, 'FlowSOM_UMAP_Timepoint.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved: FlowSOM_UMAP_Timepoint.png")

# ============================================================
# FIGURE 4: UMAP grid by marker expression
# ============================================================
print("Generating Figure 4: UMAP per-marker expression grid")
n_markers = len(clustering_markers)
ncols     = 4
nrows     = int(np.ceil(n_markers / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                         sharex=True, sharey=True)

marker_cmap = LinearSegmentedColormap.from_list(
    'expr', ['#2166AC', '#67A9CF', '#F7F7F7', '#EF8A62', '#B2182B'])

for idx, marker in enumerate(clustering_markers):
    ax       = axes.flat[idx]
    m_idx    = clustering_markers.index(marker)
    vals     = cs_s[:, m_idx]
    vmin, vmax = np.percentile(vals, [2, 98])
    sc = ax.scatter(emb_s[:, 0], emb_s[:, 1],
                    c=vals, cmap=marker_cmap,
                    s=0.7, alpha=0.65, linewidths=0,
                    vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(marker, fontsize=11, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.01)

# Hide spare axes
for j in range(n_markers, nrows * ncols):
    axes.flat[j].axis('off')

fig.suptitle('UMAP: Scaled Marker Expression per Cell',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(fig_dir, 'FlowSOM_UMAP_Markers.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved: FlowSOM_UMAP_Markers.png")

# ============================================================
# FIGURE 5: Per-cohort 2D density facets (HIV split W0 / W48)
# ============================================================
print("Generating Figure 5: UMAP per-cohort density (HIV W0 / W48 / PrEP / HC)")
fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharex=True, sharey=True)
xlim = (emb[:, 0].min() - 1, emb[:, 0].max() + 1)
ylim = (emb[:, 1].min() - 1, emb[:, 1].max() + 1)

for ax, grp in zip(axes, cohort_order):
    m  = cohort_per_cell == grp
    xy = emb_s[m]
    if len(xy) == 0:
        ax.set_title(f'{grp} (no cells)')
        continue
    h, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1],
                               bins=120, range=[xlim, ylim])
    ax.imshow(np.log1p(h.T),
              origin='lower',
              extent=[xe[0], xe[-1], ye[0], ye[-1]],
              aspect='auto', cmap='magma')
    ax.set_title(f'{grp}  (cells={m.sum()})',
                 fontsize=12, fontweight='bold',
                 color=cohort_colors[grp])
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')

fig.suptitle('UMAP Density by Cohort (log1p binned counts)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(fig_dir, 'FlowSOM_UMAP_Density_Group.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved: FlowSOM_UMAP_Density_Group.png")

print("\n" + "=" * 60)
print("UMAP PIPELINE COMPLETE")
print("=" * 60)
print(f"Output directory: {fig_dir}")
