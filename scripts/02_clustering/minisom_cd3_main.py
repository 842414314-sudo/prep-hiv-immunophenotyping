#!/usr/bin/env python3
"""
CD3 Main MiniSom Analysis — 50 files (P88 CMV- excluded)
14 markers (incl CD4/CD8), 10x10 SOM, 20 MC
StandardScaler, PCA init, Ward hierarchical
Replaces the original ORIGINAL_flowsom_analysis.py
"""

import os, warnings, glob, re, pickle
import numpy as np
import pandas as pd
import fcsparser
from minisom import MiniSom
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram as dendro_plot
from scipy.stats import kruskal, mannwhitneyu
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

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



def bh_adjust(pvals):
    n = len(pvals)
    si = np.argsort(pvals)
    sp = np.array(pvals)[si]
    adj = np.zeros(n)
    cm = 1.0
    for i in range(n-1, -1, -1):
        cm = min(cm, sp[i]*n/(i+1))
        adj[si[i]] = min(cm, 1.0)
    return adj

# ============================================================
BASE = 'FCS_DIR + "/"'
outdir = 'OUTPUT_DIR'
os.makedirs(outdir, exist_ok=True)

# 14 clustering markers (including CD4/CD8 for lineage)
clustering_markers = ['CD4','CD8','CD45RA','CCR7','CD27','CD28',
                      'CD38','HLADR','TIM3','PD1','FOXP3','CD25',
                      'CD127','CD95']

CELLS_PER_SAMPLE = 5000
SOM_X, SOM_Y = 10, 10
N_MC = 20

# ============================================================
# Build file list in ORIGINAL order (B1 → B2 → B3)
print("="*60)
print("CD3 Main MiniSom — 14 markers, 20 MC, 50 files")
print("="*60)

file_list = []
# B1: HIV W0/W48
for pid in [1,3,5,6,7,8,9,10,12,13,14,15]:
    for tp, label in [('Basal','HIV_W0'),('W48','HIV_W48')]:
        f = os.path.join(BASE,'B1',f'norm_export_{pid} {tp}_CD3 subset.fcs')
        if os.path.exists(f): file_list.append({'file':f,'patient':f'P{pid}','group':label})
# B1: PrEP
for pid in [18,19,22,23,27,28,30,31,32,33]:
    f = os.path.join(BASE,'B1',f'norm_export_{pid} Basal_CD3 subset.fcs')
    if os.path.exists(f): file_list.append({'file':f,'patient':f'P{pid}','group':'PrEP'})
# B2: HIV P20/P25
for pid in [20,25]:
    for tp, label in [('BASAL','HIV_W0'),('W48','HIV_W48')]:
        f = os.path.join(BASE,'B2',f'norm_export_{pid}{tp}_CD3 subset.fcs')
        if os.path.exists(f): file_list.append({'file':f,'patient':f'P{pid}','group':label})
# B2: HC
for pid in [39,41,43,45,46,47]:
    f = os.path.join(BASE,'B2',f'norm_export_{pid}HC_CD3 subset.fcs')
    if os.path.exists(f): file_list.append({'file':f,'patient':f'P{pid}','group':'HC'})
# B3: HC (no P88)
for pid in [70,76,80,83,85,100]:
    f = os.path.join(BASE,'B3',f'norm_export_{pid}_CD3 subset.fcs')
    if os.path.exists(f): file_list.append({'file':f,'patient':f'P{pid}','group':'HC'})

print(f"Built file list: {len(file_list)} files")

all_cells = []
sample_info = []

for fl in file_list:
    meta, data = fcsparser.parse(fl['file'], reformat_meta=True)
    avail = [m for m in clustering_markers if m in data.columns]
    if len(avail) < len(clustering_markers): continue

    cell_data = data[clustering_markers].dropna().values

    n = len(cell_data)
    if n > CELLS_PER_SAMPLE:
        idx = np.random.choice(n, CELLS_PER_SAMPLE, replace=False)
        cell_data = cell_data[idx]

    all_cells.append(cell_data)
    sample_info.append({'patient': fl['patient'], 'group': fl['group'], 'file': os.path.basename(fl['file']), 'n_cells': len(cell_data)})
    print(f"  [{len(sample_info)}] {fl['patient']} ({fl['group']}): {n} -> {len(cell_data)}")

combined = np.vstack(all_cells)
sample_labels = []
for i, cells in enumerate(all_cells):
    sample_labels.extend([i] * len(cells))
sample_labels = np.array(sample_labels)

print(f"\nTotal: {combined.shape[0]} cells, {len(sample_info)} samples, {combined.shape[1]} markers")
for g in ['HC','PrEP','HIV_W0','HIV_W48']:
    n = sum(1 for s in sample_info if s['group'] == g)
    print(f"  {g}: {n}")

# ============================================================
print("\nTraining MiniSom...")
scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined)

som = MiniSom(SOM_X, SOM_Y, combined_scaled.shape[1],
              sigma=2.0, learning_rate=0.5,
              neighborhood_function='gaussian', random_seed=42)
som.pca_weights_init(combined_scaled)
som.train(combined_scaled, num_iteration=combined_scaled.shape[0],
          random_order=True, verbose=False)
print(f"  QE: {som.quantization_error(combined_scaled):.4f}")

# Nodes -> MC
winners = np.array([som.winner(x) for x in combined_scaled])
node_ids = winners[:, 0] * SOM_Y + winners[:, 1]
n_nodes = SOM_X * SOM_Y

node_profiles = np.zeros((n_nodes, combined_scaled.shape[1]))
for nid in range(n_nodes):
    mask = node_ids == nid
    if mask.sum() > 0:
        node_profiles[nid] = combined_scaled[mask].mean(axis=0)

node_link = linkage(node_profiles, method='ward', metric='euclidean')
mc_assign = fcluster(node_link, t=N_MC, criterion='maxclust')
cell_mc = mc_assign[node_ids]

# ============================================================
# Frequencies
n_samples = len(sample_info)
freq_matrix = np.zeros((n_samples, N_MC))
for i in range(n_samples):
    mask = sample_labels == i
    sample_mc = cell_mc[mask]
    counts = np.bincount(sample_mc, minlength=N_MC+1)[1:]
    freq_matrix[i] = counts / counts.sum() * 100

labels = [f"{s['patient']}_{s['group']}" for s in sample_info]
mc_cols = [f'MC{i+1}' for i in range(N_MC)]
freq_df = pd.DataFrame(freq_matrix, index=labels, columns=mc_cols)

# MC profiles (scaled)
mc_profiles = np.zeros((N_MC, len(clustering_markers)))
for mc in range(N_MC):
    mask = cell_mc == mc+1
    if mask.sum() > 0:
        mc_profiles[mc] = combined_scaled[mask].mean(axis=0)

# MC profiles (original scale)
mc_profiles_orig = np.zeros((N_MC, len(clustering_markers)))
for mc in range(N_MC):
    mask = cell_mc == mc+1
    if mask.sum() > 0:
        mc_profiles_orig[mc] = combined[mask].mean(axis=0)

# ============================================================
# Stats: KW + pairwise
groups_arr = np.array([s['group'] for s in sample_info])
results = []
for mc in range(N_MC):
    gd = {g: freq_matrix[groups_arr==g, mc] for g in ['HC','PrEP','HIV_W0','HIV_W48']}
    try:
        _, kw_p = kruskal(*[v for v in gd.values() if len(v) > 0])
    except:
        kw_p = 1.0

    pairwise = {}
    for g1, g2 in [('PrEP','HC'), ('PrEP','HIV_W0'), ('PrEP','HIV_W48'),
                    ('HIV_W0','HC'), ('HIV_W0','HIV_W48')]:
        try:
            _, p = mannwhitneyu(gd[g1], gd[g2], alternative='two-sided')
        except:
            p = 1.0
        pairwise[f'{g1}_vs_{g2}'] = p

    row = {'MC': f'MC{mc+1}', 'PrEP': gd['PrEP'].mean(), 'HC': gd['HC'].mean(),
           'HIV_W0': gd['HIV_W0'].mean(), 'HIV_W48': gd['HIV_W48'].mean(), 'KW_p': kw_p}
    row.update(pairwise)
    results.append(row)

res_df = pd.DataFrame(results)
res_df['KW_padj'] = bh_adjust(res_df['KW_p'].values)

# Adjust pairwise p-values
for col in [c for c in res_df.columns if '_vs_' in c]:
    res_df[col + '_adj'] = bh_adjust(res_df[col].values)

n_sig = (res_df['KW_padj'] < 0.05).sum()
print(f"\nKW significant: {n_sig}/{N_MC}")

# Print sorted results
print(f"\n{'MC':<6} {'PrEP':>6} {'HC':>6} {'W0':>6} {'W48':>6} {'KW_padj':>8}")
print("-"*45)
for _, r in res_df.sort_values('KW_p').iterrows():
    sig = '***' if r['KW_padj']<0.001 else '**' if r['KW_padj']<0.01 else '*' if r['KW_padj']<0.05 else ''
    print(f"{r['MC']:<6} {r['PrEP']:>6.1f} {r['HC']:>6.1f} {r['HIV_W0']:>6.1f} {r['HIV_W48']:>6.1f} {r['KW_padj']:>8.4f} {sig}")

# ============================================================
# Heatmap
fig, ax = plt.subplots(figsize=(14, 12))
cmap = LinearSegmentedColormap.from_list('c', ['#0A1F5D','white','#6A0624'])
order = res_df.sort_values('KW_padj').index.values
im = ax.imshow(mc_profiles[order], cmap=cmap, aspect='auto', vmin=-2, vmax=2)
ax.set_yticks(range(N_MC))
ylabels = []
for i in range(N_MC):
    mi = order[i]
    p = res_df.iloc[mi]['KW_padj']
    s = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    ylabels.append(f"MC{mi+1} {s}")
ax.set_yticklabels(ylabels, fontsize=10)
ax.set_xticks(range(len(clustering_markers)))
ax.set_xticklabels(clustering_markers, rotation=45, ha='right', fontsize=10)
ax.set_title('CD3+ MiniSom — 20 MC, 14 markers, 50 samples\nSorted by KW p-value', fontsize=13)
plt.colorbar(im, ax=ax, label='Z-scored expression', shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'CD3_Heatmap_MiniSom_50files.png'), dpi=200)
plt.close()

# ============================================================
# Dendrogram (raw Euclidean — ORIGINAL method, no CLR)
patient_link = linkage(freq_matrix, method='ward', metric='euclidean')
fig, ax = plt.subplots(figsize=(20, 7))
cm = {'PrEP':'#3E0080','HC':'#008000','HIV_W0':'#FFCD65','HIV_W48':'#808000'}
short_labels = [f"{s['patient']}_{s['group'].split('_')[-1]}" if 'HIV' in s['group'] else s['patient'] for s in sample_info]
dn = dendro_plot(patient_link, labels=short_labels, ax=ax, leaf_rotation=90, leaf_font_size=8)
label_to_idx = {sl: j for j, sl in enumerate(short_labels)}
for lbl in ax.get_xticklabels():
    txt = lbl.get_text()
    idx_m = label_to_idx[txt]
    lbl.set_color(cm[sample_info[idx_m]['group']])
    lbl.set_fontweight('bold')
ax.set_title('CD3+ Patient Dendrogram — MiniSom 20 MC, Euclidean + Ward, 50 samples', fontsize=13)
ax.set_ylabel('Ward Distance (Euclidean)')
ax.legend(handles=[Patch(facecolor=c, label=g) for g,c in cm.items()], loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'CD3_Dendrogram_MiniSom_50files.png'), dpi=200)
plt.close()

# ============================================================
# Volcano data for each pairwise comparison
for comp in [('PrEP','HC'), ('PrEP','HIV_W0'), ('PrEP','HIV_W48'), ('HIV_W0','HIV_W48'), ('HIV_W0','HC')]:
    g1, g2 = comp
    vol = pd.DataFrame()
    vol['MC'] = res_df['MC']
    vol[f'{g1}_mean'] = res_df[g1]
    vol[f'{g2}_mean'] = res_df[g2]
    vol['log2FC'] = np.log2((res_df[g1] + 0.01) / (res_df[g2] + 0.01))
    pcol = f'{g1}_vs_{g2}'
    vol['pvalue'] = res_df[pcol]
    vol['padj'] = res_df[pcol + '_adj']
    vol.to_csv(os.path.join(outdir, f'volcano_{g1}_vs_{g2}_50files.csv'), index=False)

# ============================================================
# Save everything
freq_df.to_csv(os.path.join(outdir, 'FlowSOM_CD3_frequencies_50files.csv'))
res_df.to_csv(os.path.join(outdir, 'stats_cd3_50files.csv'), index=False)
pd.DataFrame(mc_profiles, index=mc_cols, columns=clustering_markers).to_csv(
    os.path.join(outdir, 'FlowSOM_CD3_profiles_50files.csv'))
pd.DataFrame(mc_profiles_orig, index=mc_cols, columns=clustering_markers).to_csv(
    os.path.join(outdir, 'FlowSOM_CD3_profiles_orig_50files.csv'))

# Save pickle for downstream R scripts
results_pkl = {
    'freq_matrix': freq_matrix,
    'freq_df': freq_df,
    'sample_info': sample_info,
    'mc_profiles_scaled': mc_profiles,
    'mc_profiles': mc_profiles_orig,
    'clustering_markers': clustering_markers,
    'res_df': res_df,
}
with open(os.path.join(outdir, 'MiniSom_50files.pkl'), 'wb') as pf:
    pickle.dump(results_pkl, pf)

print(f"\nDONE — CD3 MiniSom 50 files → {outdir}")
