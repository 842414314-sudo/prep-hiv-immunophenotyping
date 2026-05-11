#!/usr/bin/env python3
"""
CD3 MiniSom clustering — full pipeline
=======================================
Reads batch-normalized CD3+ FCS files (51 samples), trains a 14-marker
MiniSom (10x10 SOM, seed 42), metaclusters into 20 MCs via Ward linkage,
and saves complete intermediate artifacts (SOM model, scaler, per-cell
MC assignments, per-sample MC frequencies) for downstream scripts
(`minisom_umap.py`, dendrogram and volcano R scripts).

Sample set
----------
SOM training uses all 51 samples (HC n=13, including P88) to maximise
cluster stability. P88 is excluded prior to downstream differential
abundance analysis. See README for details.

Cohort definitions are hard-coded below to match the manuscript
(HIV n=14 with W0/W48, PrEP n=10, HC n=13 incl. P88; P44HC excluded;
bridge duplicates P27prep, P33prep removed).

Pipeline
--------
1. Scan FCS_DIR/{B1,B2,B3} and build the 51-file list
2. Read FCS, subsample 5000 cells/file
3. StandardScale -> train MiniSom 10x10 (PCA init, seed 42)
4. Assign cells to nodes
5. Ward hierarchical metaclustering into 20 MCs
6. Per-sample MC frequencies, MC marker profiles, heuristic lineage labels
7. Save outputs (CSVs + pickle) to OUTPUT_DIR
"""

import os, re, json, pickle, warnings
import numpy as np
import pandas as pd
import fcsparser
from minisom import MiniSom
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ======================================================================
# USER CONFIG -- update these paths to match your local setup
# FCS_DIR    : root directory with B1/, B2/, B3/ subdirectories of
#              batch-normalised CD3+ FCS files (output of
#              `batch_norm_cd3_v11.R`; not included in this repo)
# OUTPUT_DIR : where CSVs and the pickled results will be written
# ======================================================================
FCS_DIR    = './normalized_fcs'
OUTPUT_DIR = './output'

# ============================================================
# Pipeline configuration (do not edit unless reproducing a sweep)
# ============================================================
SEED             = 42
SOM_X, SOM_Y     = 10, 10
N_METACLUSTERS   = 20
CELLS_PER_SAMPLE = 5000

clustering_markers = ['CD4', 'CD8', 'CD45RA', 'CCR7', 'CD27', 'CD28',
                      'CD38', 'HLADR', 'TIM3', 'PD1', 'FOXP3', 'CD25',
                      'CD127', 'CD95']

# Cohort definitions (manuscript)
HIV_PATIENTS  = [1, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 20, 25]
PREP_PATIENTS = [18, 19, 22, 23, 27, 28, 30, 31, 32, 33]
HC_PATIENTS   = [39, 41, 43, 45, 46, 47, 70, 76, 80, 83, 85, 88, 100]
# Notes:
#   P44HC : excluded from clinical database
#   P88HC : CMV-seropositive HC; retained for SOM training (HC n=13);
#           excluded from primary downstream analyses (see README)
#   P27prep, P33prep in B2: bridge duplicates -> skip
#   P33, P43, P47 in B3   : duplicates of B1/B2 samples -> skip

group_colors = {'HIV_W0':  '#FFCD65', 'HIV_W48': '#808000',
                'PrEP':    '#3E0080', 'HC':      '#008000'}

np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Step 1: Build the file list from FCS_DIR/{B1,B2,B3}
# ============================================================
print("=" * 60)
print("STEP 1: Scanning FCS directories and building file list")
print("=" * 60)

file_list = []

# --- B1: HIV (W0 + W48) + PrEP ---
b1_dir = os.path.join(FCS_DIR, 'B1')
for f in sorted(os.listdir(b1_dir)):
    if f.startswith('.') or not f.endswith('.fcs'):
        continue
    m = re.match(r'norm_export_(\d+)\s*(Basal|W48)_CD3', f)
    if not m:
        continue
    pid = int(m.group(1))
    tp = 'W0' if m.group(2) == 'Basal' else 'W48'
    if pid in HIV_PATIENTS:
        group, timepoint = 'HIV', tp
    elif pid in PREP_PATIENTS:
        group, timepoint = 'PrEP', 'W0'
    else:
        continue
    file_list.append({'file': os.path.join(b1_dir, f),
                      'patient': pid, 'group': group,
                      'timepoint': timepoint, 'batch': 'B1'})

# --- B2: HIV P20/P25 + HC (skip bridge duplicates) ---
b2_dir = os.path.join(FCS_DIR, 'B2')
for f in sorted(os.listdir(b2_dir)):
    if f.startswith('.') or not f.endswith('.fcs'):
        continue
    # HIV BASAL / W48
    m = re.match(r'norm_export_(\d+)(BASAL|W48)_CD3', f)
    if m:
        pid = int(m.group(1))
        if pid not in HIV_PATIENTS:
            continue
        tp = 'W0' if m.group(2) == 'BASAL' else 'W48'
        file_list.append({'file': os.path.join(b2_dir, f),
                          'patient': pid, 'group': 'HIV',
                          'timepoint': tp, 'batch': 'B2'})
        continue
    # PrEP bridge duplicates (P27prep, P33prep) -> skip
    if re.match(r'norm_export_\d+prep_CD3', f):
        continue
    # HC
    m = re.match(r'norm_export_(\d+)HC_CD3', f)
    if m:
        pid = int(m.group(1))
        if pid == 44:
            continue
        if pid in HC_PATIENTS:
            file_list.append({'file': os.path.join(b2_dir, f),
                              'patient': pid, 'group': 'HC',
                              'timepoint': 'W0', 'batch': 'B2'})

# --- B3: HC (skip duplicates of B1/B2 samples) ---
b3_dir = os.path.join(FCS_DIR, 'B3')
for f in sorted(os.listdir(b3_dir)):
    if f.startswith('.') or not f.endswith('.fcs'):
        continue
    if 'VD subset' in f:
        continue
    m = re.match(r'norm_export_(\d+)_CD3 subset\.fcs', f)
    if not m:
        continue
    pid = int(m.group(1))
    if pid == 44 or pid in [43, 47, 33]:
        continue
    if pid in HC_PATIENTS:
        file_list.append({'file': os.path.join(b3_dir, f),
                          'patient': pid, 'group': 'HC',
                          'timepoint': 'W0', 'batch': 'B3'})

print(f"  Total samples found: {len(file_list)}")
for g in ['HC', 'PrEP', 'HIV']:
    n = sum(1 for fl in file_list if fl['group'] == g)
    print(f"    {g}: {n}")

with open(os.path.join(OUTPUT_DIR, 'flowsom_filelist.json'), 'w') as jf:
    json.dump(file_list, jf, indent=2)


# ============================================================
# Step 2: Read FCS, subsample cells
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Reading FCS files and subsampling cells")
print("=" * 60)

all_cells = []
sample_labels = []
sample_info = []

for i, fl in enumerate(file_list):
    fname = os.path.basename(fl['file'])
    print(f"  [{i+1:>2}/{len(file_list)}] "
          f"P{fl['patient']} {fl['group']} {fl['timepoint']} -- {fname}")

    _, data = fcsparser.parse(fl['file'], reformat_meta=True)
    available = [m for m in clustering_markers if m in data.columns]
    if len(available) < len(clustering_markers):
        missing = set(clustering_markers) - set(available)
        print(f"    WARNING: missing markers in this FCS: {missing}")
    cell_data = data[available].copy().dropna()

    n = len(cell_data)
    if n > CELLS_PER_SAMPLE:
        idx = np.random.choice(n, CELLS_PER_SAMPLE, replace=False)
        cell_data = cell_data.iloc[idx]

    all_cells.append(cell_data.values)
    sample_labels.extend([i] * len(cell_data))
    sample_info.append({
        'sample_idx': i,
        'patient': fl['patient'],
        'group': fl['group'],
        'timepoint': fl['timepoint'],
        'batch': fl['batch'],
        'n_cells_total': n,
        'n_cells_sampled': len(cell_data),
        'label': f"P{fl['patient']}_{fl['group']}_{fl['timepoint']}",
    })

combined = np.vstack(all_cells)
sample_labels = np.array(sample_labels)
print(f"\n  Pooled cells: {combined.shape[0]:,} | "
      f"{len(sample_info)} samples | {combined.shape[1]} markers")


# ============================================================
# Step 3: Standardise + train MiniSom
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 3: Training MiniSom ({SOM_X}x{SOM_Y} grid)")
print("=" * 60)

scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined)

som = MiniSom(SOM_X, SOM_Y, combined_scaled.shape[1],
              sigma=2.0, learning_rate=0.5,
              neighborhood_function='gaussian',
              random_seed=SEED)
som.pca_weights_init(combined_scaled)
print("  SOM initialised with PCA weights")

som.train(combined_scaled,
          num_iteration=combined_scaled.shape[0],
          random_order=True, verbose=False)

qe = som.quantization_error(combined_scaled)
print(f"  Quantization error: {qe:.4f}")


# ============================================================
# Step 4: Assign cells to SOM nodes
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Assigning cells to SOM nodes")
print("=" * 60)

winners = np.array([som.winner(x) for x in combined_scaled])
node_ids = winners[:, 0] * SOM_Y + winners[:, 1]
n_nodes = SOM_X * SOM_Y
node_counts = np.bincount(node_ids, minlength=n_nodes)
print(f"  {n_nodes} nodes | cells/node: "
      f"min={node_counts.min()}, max={node_counts.max()}, "
      f"median={int(np.median(node_counts))}")


# ============================================================
# Step 5: Ward metaclustering of SOM nodes
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 5: Hierarchical metaclustering into {N_METACLUSTERS} MCs")
print("=" * 60)

node_profiles_scaled = np.zeros((n_nodes, combined_scaled.shape[1]))
node_profiles_orig   = np.zeros((n_nodes, combined.shape[1]))
for nid in range(n_nodes):
    mask = node_ids == nid
    if mask.sum() > 0:
        node_profiles_scaled[nid] = combined_scaled[mask].mean(axis=0)
        node_profiles_orig[nid]   = combined[mask].mean(axis=0)

node_linkage = linkage(node_profiles_scaled, method='ward', metric='euclidean')
metacluster_assignments = fcluster(node_linkage,
                                   t=N_METACLUSTERS, criterion='maxclust')

for mc in range(1, N_METACLUSTERS + 1):
    nodes_in_mc = np.where(metacluster_assignments == mc)[0]
    cells_in_mc = sum(node_counts[n] for n in nodes_in_mc)
    print(f"  MC{mc:>2}: {len(nodes_in_mc):>3} nodes, {cells_in_mc:>6} cells")

cell_metaclusters = metacluster_assignments[node_ids]


# ============================================================
# Step 6: Per-sample frequencies + per-MC marker profiles
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Computing MC frequencies and marker profiles")
print("=" * 60)

n_samples = len(sample_info)
freq_matrix = np.zeros((n_samples, N_METACLUSTERS))
for i in range(n_samples):
    mask = sample_labels == i
    sample_mc = cell_metaclusters[mask]
    counts = np.bincount(sample_mc, minlength=N_METACLUSTERS + 1)[1:]
    freq_matrix[i] = counts / counts.sum() * 100

freq_df = pd.DataFrame(
    freq_matrix,
    index=[s['label'] for s in sample_info],
    columns=[f'MC{i+1}' for i in range(N_METACLUSTERS)],
)
print(f"  Frequency matrix: {freq_df.shape} | "
      f"row sums {freq_df.sum(axis=1).min():.1f}-{freq_df.sum(axis=1).max():.1f}%")

mc_profiles_scaled = np.zeros((N_METACLUSTERS, len(clustering_markers)))
mc_profiles_orig   = np.zeros((N_METACLUSTERS, len(clustering_markers)))
for mc in range(N_METACLUSTERS):
    mask = cell_metaclusters == mc + 1
    if mask.sum() > 0:
        mc_profiles_scaled[mc] = combined_scaled[mask].mean(axis=0)
        mc_profiles_orig[mc]   = combined[mask].mean(axis=0)

# Heuristic lineage labels (refined manually downstream)
mc_annotations = []
for mc in range(N_METACLUSTERS):
    profile = mc_profiles_scaled[mc]
    cd4_val = profile[clustering_markers.index('CD4')]
    cd8_val = profile[clustering_markers.index('CD8')]
    if cd4_val > 0.5 and cd8_val < -0.5:
        lineage = 'CD4+'
    elif cd8_val > 0.5 and cd4_val < -0.5:
        lineage = 'CD8+'
    elif cd4_val > 0 and cd8_val > 0:
        lineage = 'DP'
    else:
        lineage = 'DN/mix'
    mc_annotations.append(f"MC{mc+1} ({lineage})")


# ============================================================
# Save outputs
# ============================================================
freq_df.to_csv(os.path.join(OUTPUT_DIR, 'CD3_metacluster_frequencies.csv'))

pd.DataFrame(mc_profiles_scaled,
             index=[f'MC{i+1}' for i in range(N_METACLUSTERS)],
             columns=clustering_markers
             ).to_csv(os.path.join(OUTPUT_DIR, 'CD3_metacluster_profiles_zscored.csv'))

results = {
    'freq_matrix': freq_matrix,
    'freq_df': freq_df,
    'sample_info': sample_info,
    'mc_profiles_scaled': mc_profiles_scaled,
    'mc_profiles': mc_profiles_orig,
    'mc_annotations': mc_annotations,
    'clustering_markers': clustering_markers,
    'node_profiles_scaled': node_profiles_scaled,
    'node_counts': node_counts,
    'metacluster_assignments': metacluster_assignments,
    'som': som,
    'scaler': scaler,
    'group_colors': group_colors,
}
with open(os.path.join(OUTPUT_DIR, 'flowsom_results.pkl'), 'wb') as pf:
    pickle.dump(results, pf)

print(f"\n  Wrote frequencies, profiles, and complete pickle to {OUTPUT_DIR}/")
print("\nDone.")
