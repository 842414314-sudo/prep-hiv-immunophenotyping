#!/usr/bin/env python3
"""
CD4 / CD8 FlowSOM MC-count sweep
================================
Adapted from the v3 CD3 reproduce pipeline for CD4+ and CD8+ subsets.

Pipeline (per subset CD4 / CD8):
  1. Read 51 CD3 FCS files (v3 hardcoded order)
  2. arcsinh(x / 6000) to match existing CD4/CD8 convention
  3. Subsample 6000 cells per file
  4. Pool + compute global median of CD4 and CD8 → gate
      CD4+ : CD4 > med_CD4 & CD8 <= med_CD8
      CD8+ : CD8 > med_CD8 & CD4 <= med_CD4
  5. StandardScaler on 12 clustering markers (no CD4/CD8)
  6. Train ONE SOM (10x10, σ=2.0, lr=0.5, PCA init, seed=42)
  7. Ward linkage on 100 node profiles → ONE dendrogram
  8. For N_MC ∈ {10, 15, 20}:
      - fcluster(linkage, t=N_MC) → cell_mc
      - per-sample frequency matrix (51 rows)
      - patient Ward linkage on freq → patient dendrogram
      - cophenetic correlation
      - k=3 cut → cluster assignment
  9. Cross-MC stability: ARI between {10 vs 15, 15 vs 20, 10 vs 20}
  10. Plot 2x3 grid of dendrograms (subset × MC count)

Outputs in OUTPUT_DIR/
"""

import os, warnings, time, pickle
import numpy as np, pandas as pd
import fcsparser
from minisom import MiniSom
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# ======================================================================
# USER CONFIG — update these paths to match your local setup
# FCS_DIR: directory containing batch-normalized CD3+ FCS files
# OUTPUT_DIR: where figures and results will be saved
# ======================================================================
FCS_DIR    = './normalized_fcs'
OUTPUT_DIR = './output'
DATA_DIR   = '../data/cd3'

# ============================================================
# Config
# ============================================================
B       = FCS_DIR
OUTDIR  = OUTPUT_DIR
os.makedirs(OUTDIR, exist_ok=True)

SEED             = 42
SOM_X, SOM_Y     = 10, 10
CELLS_PER_SAMPLE = 6000          # → ~3000 CD4+ and ~3000 CD8+ per sample after gate
COFACTOR         = 6000
N_MC_SWEEP       = [10, 15, 20]
GROUP_COLORS     = {'HC':'#008000','PrEP':'#3E0080','HIV_W0':'#FFCD65','HIV_W48':'#808000'}

# 12 clustering markers (no CD4/CD8)
clust_markers = ['CD45RA','CCR7','CD27','CD28','CD38','HLADR','TIM3','PD1',
                 'FOXP3','CD25','CD127','CD95']
lineage_markers = ['CD4','CD8']
all_markers     = lineage_markers + clust_markers   # 14 total; index 0=CD4, 1=CD8

# ============================================================
# 51-file hardcoded list (v3 order, includes P88)
# ============================================================
files = [
    (f'{B}/B1/norm_export_1 Basal_CD3 subset.fcs',  'P1',  'HIV_W0'),
    (f'{B}/B1/norm_export_1 W48_CD3 subset.fcs',    'P1',  'HIV_W48'),
    (f'{B}/B1/norm_export_10 Basal_CD3 subset.fcs', 'P10', 'HIV_W0'),
    (f'{B}/B1/norm_export_10 W48_CD3 subset.fcs',   'P10', 'HIV_W48'),
    (f'{B}/B1/norm_export_12 Basal_CD3 subset.fcs', 'P12', 'HIV_W0'),
    (f'{B}/B1/norm_export_12 W48_CD3 subset.fcs',   'P12', 'HIV_W48'),
    (f'{B}/B1/norm_export_13 Basal_CD3 subset.fcs', 'P13', 'HIV_W0'),
    (f'{B}/B1/norm_export_13 W48_CD3 subset.fcs',   'P13', 'HIV_W48'),
    (f'{B}/B1/norm_export_14 Basal_CD3 subset.fcs', 'P14', 'HIV_W0'),
    (f'{B}/B1/norm_export_14 W48_CD3 subset.fcs',   'P14', 'HIV_W48'),
    (f'{B}/B1/norm_export_15 Basal_CD3 subset.fcs', 'P15', 'HIV_W0'),
    (f'{B}/B1/norm_export_15 W48_CD3 subset.fcs',   'P15', 'HIV_W48'),
    (f'{B}/B1/norm_export_18 Basal_CD3 subset.fcs', 'P18', 'PrEP'),
    (f'{B}/B1/norm_export_19 Basal_CD3 subset.fcs', 'P19', 'PrEP'),
    (f'{B}/B1/norm_export_22 Basal_CD3 subset.fcs', 'P22', 'PrEP'),
    (f'{B}/B1/norm_export_23 Basal_CD3 subset.fcs', 'P23', 'PrEP'),
    (f'{B}/B1/norm_export_27 Basal_CD3 subset.fcs', 'P27', 'PrEP'),
    (f'{B}/B1/norm_export_28 Basal_CD3 subset.fcs', 'P28', 'PrEP'),
    (f'{B}/B1/norm_export_3 Basal_CD3 subset.fcs',  'P3',  'HIV_W0'),
    (f'{B}/B1/norm_export_3 W48_CD3 subset.fcs',    'P3',  'HIV_W48'),
    (f'{B}/B1/norm_export_30 Basal_CD3 subset.fcs', 'P30', 'PrEP'),
    (f'{B}/B1/norm_export_31 Basal_CD3 subset.fcs', 'P31', 'PrEP'),
    (f'{B}/B1/norm_export_32 Basal_CD3 subset.fcs', 'P32', 'PrEP'),
    (f'{B}/B1/norm_export_33 Basal_CD3 subset.fcs', 'P33', 'PrEP'),
    (f'{B}/B1/norm_export_5 Basal_CD3 subset.fcs',  'P5',  'HIV_W0'),
    (f'{B}/B1/norm_export_5 W48_CD3 subset.fcs',    'P5',  'HIV_W48'),
    (f'{B}/B1/norm_export_6 Basal_CD3 subset.fcs',  'P6',  'HIV_W0'),
    (f'{B}/B1/norm_export_6 W48_CD3 subset.fcs',    'P6',  'HIV_W48'),
    (f'{B}/B1/norm_export_7 Basal_CD3 subset.fcs',  'P7',  'HIV_W0'),
    (f'{B}/B1/norm_export_7 W48_CD3 subset.fcs',    'P7',  'HIV_W48'),
    (f'{B}/B1/norm_export_8 Basal_CD3 subset.fcs',  'P8',  'HIV_W0'),
    (f'{B}/B1/norm_export_8 W48_CD3 subset.fcs',    'P8',  'HIV_W48'),
    (f'{B}/B1/norm_export_9 Basal_CD3 subset.fcs',  'P9',  'HIV_W0'),
    (f'{B}/B1/norm_export_9 W48_CD3 subset.fcs',    'P9',  'HIV_W48'),
    (f'{B}/B2/norm_export_20BASAL_CD3 subset.fcs',  'P20', 'HIV_W0'),
    (f'{B}/B2/norm_export_20W48_CD3 subset.fcs',    'P20', 'HIV_W48'),
    (f'{B}/B2/norm_export_25BASAL_CD3 subset.fcs',  'P25', 'HIV_W0'),
    (f'{B}/B2/norm_export_25W48_CD3 subset.fcs',    'P25', 'HIV_W48'),
    (f'{B}/B2/norm_export_39HC_CD3 subset.fcs',     'P39', 'HC'),
    (f'{B}/B2/norm_export_41HC_CD3 subset.fcs',     'P41', 'HC'),
    (f'{B}/B2/norm_export_43HC_CD3 subset.fcs',     'P43', 'HC'),
    (f'{B}/B2/norm_export_45HC_CD3 subset.fcs',     'P45', 'HC'),
    (f'{B}/B2/norm_export_46HC_CD3 subset.fcs',     'P46', 'HC'),
    (f'{B}/B2/norm_export_47HC_CD3 subset.fcs',     'P47', 'HC'),
    (f'{B}/B3/norm_export_100_CD3 subset.fcs',      'P100','HC'),
    (f'{B}/B3/norm_export_70_CD3 subset.fcs',       'P70', 'HC'),
    (f'{B}/B3/norm_export_76_CD3 subset.fcs',       'P76', 'HC'),
    (f'{B}/B3/norm_export_80_CD3 subset.fcs',       'P80', 'HC'),
    (f'{B}/B3/norm_export_83_CD3 subset.fcs',       'P83', 'HC'),
    (f'{B}/B3/norm_export_85_CD3 subset.fcs',       'P85', 'HC'),
    (f'{B}/B3/norm_export_88_CD3 subset.fcs',       'P88', 'HC'),
]

# ============================================================
# Read + arcsinh + subsample
# ============================================================
print("="*70)
print(f"CD4/CD8 FlowSOM MC-Sweep — reading {len(files)} files")
print("="*70)

np.random.seed(SEED)

all_cells_per_file = []   # list of (n_cells, 14) arrays
sample_info        = []

t0 = time.time()
for i, (fp, pid, grp) in enumerate(files):
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    _, data = fcsparser.parse(fp, reformat_meta=True)
    available = [m for m in all_markers if m in data.columns]
    if len(available) != len(all_markers):
        raise ValueError(f"{fp} missing: {set(all_markers) - set(available)}")
    cd = data[all_markers].dropna().values
    # arcsinh transform (match CD4 legacy)
    cd = np.arcsinh(cd / COFACTOR)
    n = len(cd)
    if n > CELLS_PER_SAMPLE:
        idx = np.random.choice(n, CELLS_PER_SAMPLE, replace=False)
        cd = cd[idx]
    all_cells_per_file.append(cd)
    sample_info.append({'patient':pid, 'group':grp,
                        'label':f'{pid}_{grp}', 'file':os.path.basename(fp),
                        'n_in_fcs':n, 'n_sampled':len(cd)})
    print(f"  [{i+1:>2}/51] {pid:>4} {grp:<7} {n:>7} -> {len(cd):>5}")

all_stack = np.vstack(all_cells_per_file)
print(f"\nTotal cells: {all_stack.shape[0]:,}  ({all_stack.shape[1]} markers)")
print(f"Read time: {time.time()-t0:.1f}s")

# Global medians
cd4_idx = all_markers.index('CD4')
cd8_idx = all_markers.index('CD8')
med_CD4 = np.median(all_stack[:, cd4_idx])
med_CD8 = np.median(all_stack[:, cd8_idx])
print(f"\nGlobal medians (arcsinh): CD4 = {med_CD4:.3f}, CD8 = {med_CD8:.3f}")

# ============================================================
# Per-subset analysis
# ============================================================
def run_subset(subset_name):
    """subset_name: 'CD4' or 'CD8'"""
    print(f"\n{'='*70}\n{subset_name}+ subset\n{'='*70}")
    # Gate each file's cells, keep track of per-sample count
    gated_cells       = []
    gated_labels_list = []
    sample_n_gated    = []
    for i, cd in enumerate(all_cells_per_file):
        if subset_name == 'CD4':
            mask = (cd[:, cd4_idx] > med_CD4) & (cd[:, cd8_idx] <= med_CD8)
        else:  # CD8
            mask = (cd[:, cd8_idx] > med_CD8) & (cd[:, cd4_idx] <= med_CD4)
        cd_sel = cd[mask]
        sample_n_gated.append(len(cd_sel))
        if len(cd_sel) == 0:
            print(f"  WARN: sample {sample_info[i]['label']} has 0 {subset_name}+ cells")
            continue
        # keep only 12 clustering markers
        clust = cd_sel[:, [all_markers.index(m) for m in clust_markers]]
        gated_cells.append(clust)
        gated_labels_list.extend([i] * len(clust))

    combined = np.vstack(gated_cells)
    sample_labels = np.array(gated_labels_list)
    print(f"  Total {subset_name}+ cells: {combined.shape[0]:,}  "
          f"(mean {np.mean(sample_n_gated):.0f}/sample)")

    # Scale
    X = StandardScaler().fit_transform(combined)

    # SOM
    print(f"  Training SOM...")
    t1 = time.time()
    som = MiniSom(SOM_X, SOM_Y, X.shape[1],
                  sigma=2.0, learning_rate=0.5,
                  neighborhood_function='gaussian', random_seed=SEED)
    som.pca_weights_init(X)
    som.train(X, num_iteration=X.shape[0], random_order=True, verbose=False)
    qe = som.quantization_error(X)
    print(f"    QE = {qe:.4f}  [{time.time()-t1:.1f}s]")

    winners = np.array([som.winner(x) for x in X])
    node_ids = winners[:,0]*SOM_Y + winners[:,1]
    n_nodes = SOM_X * SOM_Y

    node_profiles = np.zeros((n_nodes, X.shape[1]))
    for k in range(n_nodes):
        m = node_ids == k
        if m.sum() > 0:
            node_profiles[k] = X[m].mean(axis=0)

    # ONE Ward linkage on 100 nodes, multiple cuts
    node_link = linkage(node_profiles, method='ward', metric='euclidean')

    mc_results = {}
    for N_MC in N_MC_SWEEP:
        mc_assign = fcluster(node_link, t=N_MC, criterion='maxclust')
        cell_mc   = mc_assign[node_ids]

        # MC profiles
        mc_prof = np.zeros((N_MC, X.shape[1]))
        for mc in range(N_MC):
            m = cell_mc == mc+1
            if m.sum() > 0:
                mc_prof[mc] = X[m].mean(axis=0)

        # Per-sample freq matrix (all 51)
        freq = np.zeros((len(files), N_MC))
        for s in range(len(files)):
            m = sample_labels == s
            if m.sum() == 0:
                continue
            cnts = np.bincount(cell_mc[m], minlength=N_MC+1)[1:]
            freq[s] = cnts / cnts.sum() * 100

        # Patient dendrogram (Ward + Euclidean on freq)
        patient_link = linkage(freq, method='ward', metric='euclidean')
        # Cophenetic r
        D_raw  = pdist(freq)
        D_coph = cophenet(patient_link, D_raw)[0]
        # k=3 cut
        k3 = fcluster(patient_link, t=3, criterion='maxclust')

        mc_results[N_MC] = {
            'mc_assign'     : mc_assign,
            'cell_mc'       : cell_mc,
            'mc_profiles'   : mc_prof,
            'freq'          : freq,
            'patient_link'  : patient_link,
            'cophenetic_r'  : D_coph,
            'k3_assign'     : k3,
        }
        print(f"  N_MC={N_MC:<3}: cophenetic r = {D_coph:.4f}  "
              f"| k=3 sizes = {np.bincount(k3)[1:]}")

    return {'node_link':node_link, 'node_profiles':node_profiles, 'qe':qe,
            'n_cells':combined.shape[0], 'n_per_sample':sample_n_gated,
            'mc_results':mc_results}

RESULTS = {}
for subset in ['CD4', 'CD8']:
    RESULTS[subset] = run_subset(subset)

# ============================================================
# Cross-MC stability (ARI)
# ============================================================
print("\n" + "="*70)
print("Patient k=3 assignment stability across MC counts (ARI)")
print("="*70)
ari_table = []
for subset in ['CD4','CD8']:
    for i, n1 in enumerate(N_MC_SWEEP):
        for n2 in N_MC_SWEEP[i+1:]:
            k1 = RESULTS[subset]['mc_results'][n1]['k3_assign']
            k2 = RESULTS[subset]['mc_results'][n2]['k3_assign']
            ari = adjusted_rand_score(k1, k2)
            ari_table.append({'subset':subset, 'N_MC_A':n1, 'N_MC_B':n2, 'ARI_k3':ari})
            print(f"  {subset} | N_MC {n1:>2} vs {n2:>2}: ARI(k=3) = {ari:.4f}")
ari_df = pd.DataFrame(ari_table)
ari_df.to_csv(os.path.join(OUTDIR,'patient_k3_ARI.csv'), index=False)

# Cophenetic r table
coph_rows = []
for subset in ['CD4','CD8']:
    for n in N_MC_SWEEP:
        coph_rows.append({'subset':subset,'N_MC':n,
                          'cophenetic_r':RESULTS[subset]['mc_results'][n]['cophenetic_r']})
pd.DataFrame(coph_rows).to_csv(os.path.join(OUTDIR,'cophenetic_r.csv'), index=False)

# ============================================================
# Save outputs
# ============================================================
with open(os.path.join(OUTDIR, 'flowsom_cd4cd8_mcsweep.pkl'), 'wb') as pf:
    pickle.dump({'sample_info':sample_info, 'results':RESULTS,
                 'clust_markers':clust_markers,
                 'med_CD4':med_CD4, 'med_CD8':med_CD8,
                 'ari_df':ari_df}, pf)

# Freq CSVs
for subset in ['CD4','CD8']:
    for n in N_MC_SWEEP:
        freq = RESULTS[subset]['mc_results'][n]['freq']
        df = pd.DataFrame(freq,
                          index=[s['label'] for s in sample_info],
                          columns=[f'MC{i+1}' for i in range(n)])
        df.to_csv(os.path.join(OUTDIR, f'freq_{subset}_MC{n}.csv'))
        prof = RESULTS[subset]['mc_results'][n]['mc_profiles']
        pd.DataFrame(prof,
                     index=[f'MC{i+1}' for i in range(n)],
                     columns=clust_markers).to_csv(
            os.path.join(OUTDIR, f'profiles_{subset}_MC{n}.csv'))

# ============================================================
# Plot: 2x3 grid of dendrograms
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(28, 12))
labels_all = [s['patient'] for s in sample_info]
group_of   = [s['group']   for s in sample_info]

for ri, subset in enumerate(['CD4','CD8']):
    for ci, n_mc in enumerate(N_MC_SWEEP):
        ax = axes[ri, ci]
        link = RESULTS[subset]['mc_results'][n_mc]['patient_link']
        coph = RESULTS[subset]['mc_results'][n_mc]['cophenetic_r']
        k3   = RESULTS[subset]['mc_results'][n_mc]['k3_assign']
        dn = dendrogram(link, labels=labels_all, ax=ax, leaf_rotation=90,
                        leaf_font_size=7, color_threshold=0)
        # color leaf labels by group
        for lbl in ax.get_xticklabels():
            pid = lbl.get_text()
            gi = labels_all.index(pid)
            lbl.set_color(GROUP_COLORS[group_of[gi]])
            lbl.set_fontweight('bold')
        ax.set_title(f'{subset}+  —  N_MC = {n_mc}\n'
                     f'cophenetic r = {coph:.3f}',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Ward distance' if ci==0 else '')

# Shared legend
handles = [Patch(facecolor=c, label=g) for g,c in GROUP_COLORS.items()]
fig.legend(handles=handles, loc='upper right', fontsize=11,
           bbox_to_anchor=(0.99, 0.99))
fig.suptitle('Patient-level Dendrogram Stability across MC counts (CD4 & CD8)',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.985])
for ext in ('pdf','png'):
    fig.savefig(os.path.join(OUTDIR, f'patient_dendrogram_grid.{ext}'),
                bbox_inches='tight', facecolor='white',
                dpi=200 if ext=='png' else None)
plt.close(fig)

# ARI heatmap
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))
for i, subset in enumerate(['CD4','CD8']):
    M = np.ones((3, 3))
    for _, r in ari_df[ari_df['subset']==subset].iterrows():
        ai = N_MC_SWEEP.index(r['N_MC_A'])
        bj = N_MC_SWEEP.index(r['N_MC_B'])
        M[ai, bj] = r['ARI_k3']
        M[bj, ai] = r['ARI_k3']
    im = axes2[i].imshow(M, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    axes2[i].set_xticks(range(3)); axes2[i].set_xticklabels(N_MC_SWEEP)
    axes2[i].set_yticks(range(3)); axes2[i].set_yticklabels(N_MC_SWEEP)
    axes2[i].set_title(f'{subset}+  —  Patient k=3 ARI')
    for a in range(3):
        for b in range(3):
            axes2[i].text(b, a, f"{M[a,b]:.3f}", ha='center', va='center',
                          fontsize=11, fontweight='bold',
                          color='white' if M[a,b] < 0.75 else 'black')
    plt.colorbar(im, ax=axes2[i], shrink=0.7)
plt.tight_layout()
for ext in ('pdf','png'):
    fig2.savefig(os.path.join(OUTDIR, f'k3_ARI_heatmap.{ext}'),
                 bbox_inches='tight', facecolor='white',
                 dpi=200 if ext=='png' else None)
plt.close(fig2)

print(f"\n{'='*70}\nDONE -> {OUTDIR}\n{'='*70}")
print(f"  patient_dendrogram_grid.pdf/png  (2x3 dendrogram grid)")
print(f"  k3_ARI_heatmap.pdf/png           (ARI matrices)")
print(f"  patient_k3_ARI.csv               (ARI values)")
print(f"  cophenetic_r.csv                 (cophenetic r table)")
print(f"  freq_{{CD4,CD8}}_MC{{10,15,20}}.csv    (6 freq matrices)")
print(f"  profiles_{{CD4,CD8}}_MC{{10,15,20}}.csv (6 profile matrices)")
print(f"  flowsom_cd4cd8_mcsweep.pkl       (full pickle)")
