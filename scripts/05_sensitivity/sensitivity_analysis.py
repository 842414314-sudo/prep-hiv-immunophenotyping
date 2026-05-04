#!/usr/bin/env python3
"""
Sensitivity Analysis for FlowSOM Patient Clustering
Tests robustness across: SOM grid sizes, metacluster numbers, k values,
distance metrics, linkage methods, and bootstrap resampling.
"""
import numpy as np
import pandas as pd
import fcsparser
import os, re, sys, warnings
from minisom import MiniSom
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
np.random.seed(42)

# ======================================================================
# USER CONFIG — update these paths to match your local setup
# FCS_DIR: directory containing batch-normalized CD3+ FCS files
# OUTPUT_DIR: where figures and results will be saved
# ======================================================================
FCS_DIR    = './normalized_fcs'
OUTPUT_DIR = './output'
DATA_DIR   = '../data/cd3'



BASE = 'FCS_DIR'
FIG = os.path.join(BASE, 'Fig')

# Markers for clustering (same as original)
MARKERS = ['CD4', 'CD8', 'CD45RA', 'CCR7', 'CD27', 'CD28', 'CD38', 'HLADR',
           'TIM3', 'PD1', 'FOXP3', 'CD25', 'CD127', 'CD95']

# Key patients to track
IPR_W48 = ['P3_HIV_W48', 'P8_HIV_W48', 'P20_HIV_W48', 'P25_HIV_W48']
IPNR_W48 = ['P1_HIV_W48', 'P5_HIV_W48', 'P9_HIV_W48', 'P10_HIV_W48', 'P12_HIV_W48', 'P15_HIV_W48']

# File mapping: (batch_dir, filename_pattern) -> sample_label
# Exclude: P44HC, P27prep_bridge(B2), P33prep_bridge(B2), B3 bridges (P33,P43,P44,P47)
def build_file_map():
    files = {}
    # B1: HIV patients (paired W0/W48) and PrEP (single)
    b1 = os.path.join(BASE, 'B1')
    hiv_ids_b1 = [1,3,5,6,7,8,9,10,12,13,14,15]
    for pid in hiv_ids_b1:
        for tp, label in [('Basal', 'W0'), ('W48', 'W48')]:
            fname = f'norm_export_{pid} {tp}_CD3 subset.fcs'
            fpath = os.path.join(b1, fname)
            if os.path.exists(fpath):
                files[fpath] = f'P{pid}_HIV_{label}'
    prep_ids_b1 = [18,19,22,23,27,28,30,31,32,33]
    for pid in prep_ids_b1:
        fname = f'norm_export_{pid} Basal_CD3 subset.fcs'
        fpath = os.path.join(b1, fname)
        if os.path.exists(fpath):
            files[fpath] = f'P{pid}_PrEP_W0'

    # B2: HIV P20, P25 (paired) + HC P39,41,43,45,46,47
    b2 = os.path.join(BASE, 'B2')
    for pid in [20, 25]:
        for tp, label in [('BASAL', 'W0'), ('W48', 'W48')]:
            fname = f'norm_export_{pid}{tp}_CD3 subset.fcs'
            fpath = os.path.join(b2, fname)
            if os.path.exists(fpath):
                files[fpath] = f'P{pid}_HIV_{label}'
    for pid in [39, 41, 43, 45, 46, 47]:
        fname = f'norm_export_{pid}HC_CD3 subset.fcs'
        fpath = os.path.join(b2, fname)
        if os.path.exists(fpath):
            files[fpath] = f'P{pid}_HC_W0'

    # B3: HC P70,76,80,83,85,88,100
    b3 = os.path.join(BASE, 'B3')
    for pid in [70, 76, 80, 83, 85, 88, 100]:
        fname = f'norm_export_{pid}_CD3 subset.fcs'
        fpath = os.path.join(b3, fname)
        if os.path.exists(fpath):
            files[fpath] = f'P{pid}_HC_W0'

    return files

def load_data(file_map, subsample=3000):
    """Load and subsample FCS data"""
    all_data = []
    sample_labels = []
    sample_indices = []

    for fpath, label in sorted(file_map.items(), key=lambda x: x[1]):
        try:
            meta, data = fcsparser.parse(fpath)
            # Select markers
            available = [m for m in MARKERS if m in data.columns]
            if len(available) < len(MARKERS):
                missing = set(MARKERS) - set(available)
                print(f"  Warning: {label} missing {missing}")
                continue

            cells = data[MARKERS].values
            # Subsample
            if len(cells) > subsample:
                idx = np.random.choice(len(cells), subsample, replace=False)
                cells = cells[idx]

            # Arcsinh transform (cofactor=6000)
            cells = np.arcsinh(cells / 6000)

            start_idx = len(all_data[0]) if all_data else 0
            all_data.append(cells)
            sample_labels.append(label)
            sample_indices.append((sum(len(d) for d in all_data[:-1]),
                                   sum(len(d) for d in all_data)))
        except Exception as e:
            print(f"  Error loading {label}: {e}")

    combined = np.vstack(all_data)
    print(f"Loaded {len(sample_labels)} samples, {combined.shape[0]} total cells")
    return combined, sample_labels, sample_indices

def run_som(data, grid_x, grid_y, seed=42):
    """Train MiniSom"""
    som = MiniSom(grid_x, grid_y, data.shape[1],
                  sigma=max(grid_x, grid_y)/2, learning_rate=0.5,
                  random_seed=seed)
    som.random_weights_init(data)
    som.train_random(data, num_iteration=len(data)*2)
    return som

def get_bmu_labels(som, data):
    """Get BMU index for each cell"""
    grid_x = som._weights.shape[0]
    labels = np.array([som.winner(d)[0] * som._weights.shape[1] + som.winner(d)[1]
                       for d in data])
    return labels

def metacluster(som, n_clusters):
    """K-means metaclustering on SOM node weights"""
    weights = som._weights.reshape(-1, som._weights.shape[2])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    node_labels = km.fit_predict(weights)
    return node_labels

def build_frequency_matrix(bmu_labels, node_metacluster, sample_indices, sample_labels, n_mc):
    """Build patient × metacluster frequency matrix"""
    freq = np.zeros((len(sample_labels), n_mc))
    for i, (start, end) in enumerate(sample_indices):
        cell_bmus = bmu_labels[start:end]
        cell_mc = node_metacluster[cell_bmus]
        for mc in range(n_mc):
            freq[i, mc] = np.sum(cell_mc == mc) / len(cell_bmus) * 100
    return pd.DataFrame(freq, index=sample_labels,
                       columns=[f'MC{j+1}' for j in range(n_mc)])

def get_group(label):
    if 'HC' in label: return 'HC'
    if 'PrEP' in label: return 'PrEP'
    if 'W0' in label: return 'HIV_W0'
    return 'HIV_W48'

def classify_clusters(freq_df, k, method='ward', metric='euclidean'):
    """Hierarchical clustering and identify which cluster is HC-dominant"""
    Z = linkage(freq_df.values, method=method, metric=metric)
    clusters = fcluster(Z, t=k, criterion='maxclust')

    # Find HC-dominant cluster(s)
    cluster_comp = defaultdict(lambda: defaultdict(int))
    for label, cl in zip(freq_df.index, clusters):
        grp = get_group(label)
        cluster_comp[cl][grp] += 1

    # HC cluster = cluster with most HC samples
    hc_cluster = max(cluster_comp.keys(),
                     key=lambda c: cluster_comp[c].get('HC', 0))

    # Check where IPR and IPNR W48 patients are
    result = {}
    for label, cl in zip(freq_df.index, clusters):
        result[label] = cl

    ipr_in_hc = sum(1 for p in IPR_W48 if p in result and result[p] == hc_cluster)
    ipnr_in_hc = sum(1 for p in IPNR_W48 if p in result and result[p] == hc_cluster)

    return {
        'clusters': dict(zip(freq_df.index, clusters)),
        'hc_cluster': hc_cluster,
        'cluster_comp': dict(cluster_comp),
        'ipr_in_hc': ipr_in_hc,
        'ipnr_in_hc': ipnr_in_hc,
        'ipr_details': {p: result.get(p, -1) for p in IPR_W48},
        'ipnr_details': {p: result.get(p, -1) for p in IPNR_W48},
        'Z': Z
    }

def bootstrap_cocluster(freq_df, n_boot=500, method='ward', metric='euclidean', k=3):
    """Bootstrap co-clustering probability matrix"""
    n = len(freq_df)
    cocluster = np.zeros((n, n))
    count = np.zeros((n, n))

    for b in range(n_boot):
        # Resample with replacement
        idx = np.random.choice(n, n, replace=True)
        boot_df = freq_df.iloc[idx]

        try:
            Z = linkage(boot_df.values, method=method, metric=metric)
            clusters = fcluster(Z, t=k, criterion='maxclust')

            # Map back to original indices
            for i in range(n):
                for j in range(i+1, n):
                    orig_i = idx[i]
                    orig_j = idx[j]
                    count[orig_i, orig_j] += 1
                    count[orig_j, orig_i] += 1
                    if clusters[i] == clusters[j]:
                        cocluster[orig_i, orig_j] += 1
                        cocluster[orig_j, orig_i] += 1
        except:
            continue

    # Probability matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        prob = np.where(count > 0, cocluster / count, 0)
    np.fill_diagonal(prob, 1.0)
    return prob

# ============================================================
# PART 1: Sensitivity on existing frequency matrix
# ============================================================
print("=" * 70)
print("PART 1: Sensitivity analysis on existing 20-metacluster matrix")
print("=" * 70)

freq_existing = pd.read_csv(os.path.join(FIG, 'FlowSOM_metacluster_frequencies.csv'), index_col=0)
print(f"Existing matrix: {freq_existing.shape}")

# Test different k values, metrics, linkage methods
k_values = [2, 3, 4, 5, 6]
linkage_methods = ['ward', 'complete', 'average']
# Ward only works with euclidean
metrics_for_ward = ['euclidean']
metrics_for_others = ['euclidean', 'cosine', 'correlation']

results_table = []

for method in linkage_methods:
    metrics = metrics_for_ward if method == 'ward' else metrics_for_others
    for metric in metrics:
        for k in k_values:
            try:
                res = classify_clusters(freq_existing, k, method, metric)
                results_table.append({
                    'grid': '10x10', 'n_mc': 20,
                    'linkage': method, 'metric': metric, 'k': k,
                    'IPR_in_HC': res['ipr_in_hc'], 'IPNR_in_HC': res['ipnr_in_hc'],
                    'IPR_details': str(res['ipr_details']),
                    'IPNR_details': str(res['ipnr_details'])
                })
            except Exception as e:
                print(f"  Error: {method}/{metric}/k={k}: {e}")

print("\n--- k/metric/linkage sensitivity (existing 20-MC matrix) ---")
for r in results_table:
    print(f"  {r['linkage']:8s} | {r['metric']:12s} | k={r['k']} | "
          f"IPR→HC: {r['IPR_in_HC']}/4  IPNR→HC: {r['IPNR_in_HC']}/6")

# ============================================================
# PART 2: Bootstrap co-clustering stability
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Bootstrap co-clustering stability (500 resamples)")
print("=" * 70)

prob_matrix = bootstrap_cocluster(freq_existing, n_boot=500, method='ward', metric='euclidean', k=3)

# Key co-clustering probabilities
print("\nCo-clustering probabilities (with HC samples):")
hc_samples = [s for s in freq_existing.index if 'HC' in s]
for patient in IPR_W48 + IPNR_W48:
    if patient in freq_existing.index:
        pidx = list(freq_existing.index).index(patient)
        hc_probs = [prob_matrix[pidx, list(freq_existing.index).index(h)]
                     for h in hc_samples if h in freq_existing.index]
        mean_hc_prob = np.mean(hc_probs)
        group = "IPR " if patient in IPR_W48 else "IPNR"
        print(f"  {group} {patient:20s}: P(co-cluster with HC) = {mean_hc_prob:.3f}")

# ============================================================
# PART 3: Full re-run with different SOM parameters
# ============================================================
print("\n" + "=" * 70)
print("PART 3: Re-running SOM with different grid sizes & metacluster numbers")
print("=" * 70)

file_map = build_file_map()
print(f"Found {len(file_map)} files")

# Load data once
print("Loading FCS data...")
all_cells, sample_labels, sample_indices = load_data(file_map, subsample=3000)

# Different configurations
grid_sizes = [(8, 8), (10, 10), (12, 12), (15, 15)]
mc_numbers = [10, 15, 20, 25, 30]

full_results = []

for gx, gy in grid_sizes:
    print(f"\n--- Training SOM {gx}x{gy} ---")
    som = run_som(all_cells, gx, gy, seed=42)

    # Get BMU labels for all cells
    print(f"  Getting BMU labels for {len(all_cells)} cells...")
    bmu_labels = np.array([som.winner(d)[0] * gy + som.winner(d)[1]
                           for d in all_cells])

    for n_mc in mc_numbers:
        if n_mc > gx * gy:
            continue  # Can't have more metaclusters than nodes

        print(f"  Metaclustering into {n_mc} clusters...")
        node_mc = metacluster(som, n_mc)

        # Build frequency matrix
        freq = build_frequency_matrix(bmu_labels, node_mc, sample_indices,
                                      sample_labels, n_mc)

        # Test k=2,3,4,5
        for k in [2, 3, 4, 5]:
            try:
                res = classify_clusters(freq, k, 'ward', 'euclidean')
                full_results.append({
                    'grid': f'{gx}x{gy}', 'n_mc': n_mc, 'k': k,
                    'linkage': 'ward', 'metric': 'euclidean',
                    'IPR_in_HC': res['ipr_in_hc'], 'IPNR_in_HC': res['ipnr_in_hc'],
                    'IPR_details': res['ipr_details'],
                    'IPNR_details': res['ipnr_details']
                })
            except Exception as e:
                print(f"    Error k={k}: {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FULL SENSITIVITY RESULTS")
print("=" * 70)

all_results = results_table + full_results

print(f"\n{'Grid':>6s} | {'MC':>3s} | {'Linkage':>8s} | {'Metric':>12s} | {'k':>2s} | "
      f"{'IPR→HC':>7s} | {'IPNR→HC':>8s} | Separation")
print("-" * 85)

n_configs = 0
n_good_separation = 0  # IPR in HC ≥ 3 AND IPNR in HC ≤ 1

for r in all_results:
    ipr = r['IPR_in_HC']
    ipnr = r['IPNR_in_HC']
    sep = "✓ GOOD" if (ipr >= 3 and ipnr <= 1) else ("~ OK" if ipr > ipnr else "✗ BAD")

    if ipr >= 3 and ipnr <= 1:
        n_good_separation += 1
    n_configs += 1

    print(f"{r['grid']:>6s} | {r['n_mc']:>3d} | {r['linkage']:>8s} | {r['metric']:>12s} | "
          f"{r['k']:>2d} | {ipr:>4d}/4  | {ipnr:>5d}/6   | {sep}")

print(f"\n{'='*70}")
print(f"ROBUSTNESS SUMMARY: {n_good_separation}/{n_configs} configurations show good separation")
print(f"  (Good = ≥3/4 IPR in HC cluster AND ≤1/6 IPNR in HC cluster)")
print(f"{'='*70}")

# ============================================================
# VISUALIZATION
# ============================================================

# Figure 1: Heatmap of results across configurations
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Sensitivity Analysis: Patient Clustering Robustness', fontsize=16, fontweight='bold')

# Panel A: IPR in HC cluster across SOM grid × metacluster number (k=3)
ax = axes[0, 0]
k3_results = [r for r in full_results if r['k'] == 3]
grids = sorted(set(r['grid'] for r in k3_results))
mcs = sorted(set(r['n_mc'] for r in k3_results))
heatdata = np.full((len(grids), len(mcs)), np.nan)
for r in k3_results:
    gi = grids.index(r['grid'])
    mi = mcs.index(r['n_mc'])
    heatdata[gi, mi] = r['IPR_in_HC']

im = ax.imshow(heatdata, cmap='RdYlGn', vmin=0, vmax=4, aspect='auto')
ax.set_xticks(range(len(mcs)))
ax.set_xticklabels(mcs)
ax.set_yticks(range(len(grids)))
ax.set_yticklabels(grids)
ax.set_xlabel('Number of metaclusters')
ax.set_ylabel('SOM grid size')
ax.set_title('A) IPR patients in HC cluster (k=3)\n(4/4 = all IPR cluster with HC)')
for i in range(len(grids)):
    for j in range(len(mcs)):
        if not np.isnan(heatdata[i, j]):
            ax.text(j, i, f'{int(heatdata[i,j])}/4', ha='center', va='center', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='IPR in HC cluster')

# Panel B: IPNR in HC cluster (should be 0)
ax = axes[0, 1]
heatdata2 = np.full((len(grids), len(mcs)), np.nan)
for r in k3_results:
    gi = grids.index(r['grid'])
    mi = mcs.index(r['n_mc'])
    heatdata2[gi, mi] = r['IPNR_in_HC']

im2 = ax.imshow(heatdata2, cmap='RdYlGn_r', vmin=0, vmax=6, aspect='auto')
ax.set_xticks(range(len(mcs)))
ax.set_xticklabels(mcs)
ax.set_yticks(range(len(grids)))
ax.set_yticklabels(grids)
ax.set_xlabel('Number of metaclusters')
ax.set_ylabel('SOM grid size')
ax.set_title('B) IPNR patients in HC cluster (k=3)\n(0/6 = no IPNR misclassified)')
for i in range(len(grids)):
    for j in range(len(mcs)):
        if not np.isnan(heatdata2[i, j]):
            ax.text(j, i, f'{int(heatdata2[i,j])}/6', ha='center', va='center', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax, label='IPNR in HC cluster')

# Panel C: k sensitivity (existing matrix)
ax = axes[1, 0]
k_sens = [r for r in results_table if r['linkage'] == 'ward']
ks = [r['k'] for r in k_sens]
ipr_counts = [r['IPR_in_HC'] for r in k_sens]
ipnr_counts = [r['IPNR_in_HC'] for r in k_sens]
x = np.arange(len(ks))
w = 0.35
bars1 = ax.bar(x - w/2, ipr_counts, w, label='IPR W48 (want: in HC)', color='#4CAF50', alpha=0.8)
bars2 = ax.bar(x + w/2, ipnr_counts, w, label='IPNR W48 (want: NOT in HC)', color='#F44336', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'k={k}' for k in ks])
ax.set_ylabel('Patients in HC cluster')
ax.set_title('C) k sensitivity (20 MC, Ward/Euclidean)')
ax.legend()
ax.set_ylim(0, 7)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{int(bar.get_height())}', ha='center', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{int(bar.get_height())}', ha='center', fontsize=10)

# Panel D: Bootstrap co-clustering heatmap (key patients only)
ax = axes[1, 1]
key_patients = IPR_W48 + IPNR_W48
key_patients_present = [p for p in key_patients if p in freq_existing.index]
# Add a few HC for reference
hc_ref = [s for s in freq_existing.index if 'HC' in s][:4]
display_patients = key_patients_present + hc_ref

display_idx = [list(freq_existing.index).index(p) for p in display_patients]
sub_prob = prob_matrix[np.ix_(display_idx, display_idx)]

im3 = ax.imshow(sub_prob, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
short_labels = [p.replace('_HIV_W48', '').replace('_HC_W0', '_HC') for p in display_patients]
ax.set_xticks(range(len(display_patients)))
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(display_patients)))
ax.set_yticklabels(short_labels, fontsize=8)
ax.set_title('D) Bootstrap co-clustering probability\n(500 resamples, Ward/Euclidean, k=3)')
for i in range(len(display_patients)):
    for j in range(len(display_patients)):
        color = 'white' if sub_prob[i,j] > 0.5 else 'black'
        ax.text(j, i, f'{sub_prob[i,j]:.2f}', ha='center', va='center',
                fontsize=7, color=color)
plt.colorbar(im3, ax=ax, label='P(co-cluster)')

# Add separating lines between IPR/IPNR/HC
n_ipr = len([p for p in key_patients_present if p in IPR_W48])
n_ipnr = len([p for p in key_patients_present if p in IPNR_W48])
ax.axhline(y=n_ipr-0.5, color='blue', linewidth=2)
ax.axvline(x=n_ipr-0.5, color='blue', linewidth=2)
ax.axhline(y=n_ipr+n_ipnr-0.5, color='red', linewidth=2)
ax.axvline(x=n_ipr+n_ipnr-0.5, color='red', linewidth=2)

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'Sensitivity_Analysis.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(FIG, 'Sensitivity_Analysis.pdf'), bbox_inches='tight')
print(f"\nFigure saved: Sensitivity_Analysis.png/pdf")

# ============================================================
# Figure 2: Detailed per-patient tracking across configurations
# ============================================================
fig2, ax2 = plt.subplots(figsize=(16, 8))

# For each k=3 configuration, track whether each W48 patient clusters with HC
all_k3 = [r for r in full_results if r['k'] == 3]
config_labels = [f"{r['grid']}\n{r['n_mc']}MC" for r in all_k3]

# Build matrix: patients × configurations, value = 1 if in HC cluster, 0 otherwise
track_patients = IPR_W48 + IPNR_W48
track_matrix = np.zeros((len(track_patients), len(all_k3)))

for ci, r in enumerate(all_k3):
    for pi, patient in enumerate(track_patients):
        details = r.get('IPR_details', {}) if patient in IPR_W48 else r.get('IPNR_details', {})
        if isinstance(details, dict):
            # Find if this patient is in HC cluster
            # We need to check the cluster assignment
            pass

# Alternative: just use the IPR_in_HC count tracking which specific patients
for ci, r in enumerate(all_k3):
    for pi, patient in enumerate(track_patients):
        if patient in IPR_W48:
            details = r['IPR_details']
        else:
            details = r['IPNR_details']
        if isinstance(details, dict) and patient in details:
            # Get HC cluster ID from the result
            # A patient is "in HC" if their cluster matches the HC-dominant cluster
            # We stored this info... let me reconstruct
            pass

# Simpler: re-compute for the summary figure
print("\n--- Per-patient tracking across k=3 configs ---")
patient_stability = {p: [] for p in track_patients}

for r in all_k3:
    for p in IPR_W48:
        if isinstance(r['IPR_details'], dict) and p in r['IPR_details']:
            # Check if this patient's cluster is the HC cluster
            cl = r['IPR_details'][p]
            patient_stability[p].append(1 if cl == -1 else (1 if True else 0))
    for p in IPNR_W48:
        if isinstance(r['IPNR_details'], dict) and p in r['IPNR_details']:
            cl = r['IPNR_details'][p]

# Actually, we already have ipr_in_hc but not per-patient detail in a clean way.
# Let me redo the full_results tracking with per-patient detail

# Re-run classification for detailed per-patient tracking
print("\nDetailed per-patient stability tracking:")
patient_in_hc_count = {p: 0 for p in track_patients}
total_configs = 0

for gx, gy in grid_sizes:
    som = run_som(all_cells, gx, gy, seed=42)
    bmu_labels = np.array([som.winner(d)[0] * gy + som.winner(d)[1] for d in all_cells])

    for n_mc in mc_numbers:
        if n_mc > gx * gy:
            continue
        node_mc = metacluster(som, n_mc)
        freq = build_frequency_matrix(bmu_labels, node_mc, sample_indices, sample_labels, n_mc)

        # k=3 only
        Z = linkage(freq.values, method='ward', metric='euclidean')
        clusters = fcluster(Z, t=3, criterion='maxclust')

        cluster_map = dict(zip(freq.index, clusters))

        # Find HC cluster
        hc_counts = defaultdict(int)
        for label, cl in cluster_map.items():
            if 'HC' in label:
                hc_counts[cl] += 1
        hc_cluster = max(hc_counts, key=hc_counts.get) if hc_counts else -1

        total_configs += 1
        for p in track_patients:
            if p in cluster_map and cluster_map[p] == hc_cluster:
                patient_in_hc_count[p] += 1

print(f"\nAcross {total_configs} configurations (k=3, Ward/Euclidean):")
print(f"{'Patient':25s} | {'Times in HC cluster':>20s} | {'%':>6s}")
print("-" * 60)
for p in IPR_W48:
    pct = patient_in_hc_count[p] / total_configs * 100 if total_configs > 0 else 0
    print(f"IPR  {p:20s} | {patient_in_hc_count[p]:>10d}/{total_configs:<8d} | {pct:5.1f}%")
for p in IPNR_W48:
    pct = patient_in_hc_count[p] / total_configs * 100 if total_configs > 0 else 0
    print(f"IPNR {p:20s} | {patient_in_hc_count[p]:>10d}/{total_configs:<8d} | {pct:5.1f}%")

# Bar chart of per-patient stability
colors = ['#4CAF50']*4 + ['#F44336']*6
labels_short = [p.replace('_HIV_W48', '') for p in track_patients]
pcts = [patient_in_hc_count[p] / total_configs * 100 if total_configs > 0 else 0
        for p in track_patients]

ax2.bar(range(len(track_patients)), pcts, color=colors, alpha=0.8, edgecolor='black')
ax2.set_xticks(range(len(track_patients)))
ax2.set_xticklabels(labels_short, fontsize=12)
ax2.set_ylabel('% of configurations where patient clusters with HC', fontsize=12)
ax2.set_title(f'Per-Patient Stability: {total_configs} configs (grids × metaclusters, k=3, Ward/Euclidean)',
              fontsize=14, fontweight='bold')
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylim(0, 105)

for i, pct in enumerate(pcts):
    ax2.text(i, pct + 1.5, f'{pct:.0f}%', ha='center', fontsize=11, fontweight='bold')

# Legend
ipr_patch = mpatches.Patch(color='#4CAF50', label='IPR (expected: high)')
ipnr_patch = mpatches.Patch(color='#F44336', label='IPNR (expected: low)')
ax2.legend(handles=[ipr_patch, ipnr_patch], fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'Sensitivity_PerPatient.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(FIG, 'Sensitivity_PerPatient.pdf'), bbox_inches='tight')
print(f"\nFigure saved: Sensitivity_PerPatient.png/pdf")

print("\n\nDONE! Sensitivity analysis complete.")
