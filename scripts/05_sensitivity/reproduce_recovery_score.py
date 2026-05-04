#!/usr/bin/env python3
"""
Reproduce IPR/IPNR recovery score analysis with HC n=12 (P88 excluded).
Matches sensitivity_analysis.py parameters exactly:
- 3000 cells/file subsampled, arcsinh(x/6000)
- MiniSom 10x10, sigma=5, lr=0.5, random_weights_init, num_iter=2*n_cells
- KMeans (20 clusters) on SOM node weights for metaclustering
- Bootstrap 1000 resamples, Ward linkage, k=3
"""

import os, warnings, time
import numpy as np
import pandas as pd
import fcsparser
from minisom import MiniSom
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict

warnings.filterwarnings('ignore')

BASE = 'FCS_DIR'
OUT  = 'OUTPUT_DIR'
os.makedirs(OUT, exist_ok=True)

MARKERS = ['CD4','CD8','CD45RA','CCR7','CD28','CD27','CD95',
           'HLADR','CD38','PD1','TIM3','CD25','CD127','FOXP3']
SEED = 42
N_BOOT = 1000

# Build file map
def build_files():
    files = {}
    # B1: HIV 1,3,5,6,7,8,9,10,12,13,14,15 (W0+W48); PrEP 18,19,22,23,27,28,30,31,32,33
    b1 = os.path.join(BASE, 'B1')
    for pid in [1,3,5,6,7,8,9,10,12,13,14,15]:
        for tp, label in [(' Basal','W0'), (' W48','W48')]:
            fname = f'norm_export_{pid}{tp}_CD3 subset.fcs'
            fpath = os.path.join(b1, fname)
            if os.path.exists(fpath):
                files[fpath] = f'P{pid}_HIV_{label}'
    for pid in [18,19,22,23,27,28,30,31,32,33]:
        fname = f'norm_export_{pid} Basal_CD3 subset.fcs'
        fpath = os.path.join(b1, fname)
        if os.path.exists(fpath):
            files[fpath] = f'P{pid}_PrEP_W0'
    # B2: HIV 20,25 + HC 39,41,43,45,46,47
    b2 = os.path.join(BASE, 'B2')
    for pid in [20, 25]:
        for tp, label in [('BASAL','W0'), ('W48','W48')]:
            fname = f'norm_export_{pid}{tp}_CD3 subset.fcs'
            fpath = os.path.join(b2, fname)
            if os.path.exists(fpath):
                files[fpath] = f'P{pid}_HIV_{label}'
    for pid in [39,41,43,45,46,47]:
        fname = f'norm_export_{pid}HC_CD3 subset.fcs'
        fpath = os.path.join(b2, fname)
        if os.path.exists(fpath):
            files[fpath] = f'P{pid}_HC_W0'
    # B3: HC 70,76,80,83,85,88,100
    b3 = os.path.join(BASE, 'B3')
    for pid in [70, 76, 80, 83, 85, 88, 100]:
        fname = f'norm_export_{pid}_CD3 subset.fcs'
        fpath = os.path.join(b3, fname)
        if os.path.exists(fpath):
            files[fpath] = f'P{pid}_HC_W0'
    return files

def load_data(file_map, n_per=3000, seed=SEED):
    np.random.seed(seed)
    all_data, labels, idxs = [], [], []
    cumu = 0
    for fpath, label in sorted(file_map.items(), key=lambda x: x[1]):
        meta, data = fcsparser.parse(fpath)
        avail = [m for m in MARKERS if m in data.columns]
        if len(avail) < len(MARKERS):
            print(f"  SKIP {label}: missing markers")
            continue
        cells = data[MARKERS].values
        if len(cells) > n_per:
            idx = np.random.choice(len(cells), n_per, replace=False)
            cells = cells[idx]
        cells = np.arcsinh(cells / 6000)
        all_data.append(cells)
        labels.append(label)
        idxs.append((cumu, cumu + len(cells)))
        cumu += len(cells)
    combined = np.vstack(all_data)
    print(f"Loaded {len(labels)} samples, {combined.shape[0]} cells")
    return combined, labels, idxs

def run_som(data, gx=10, gy=10, seed=SEED):
    som = MiniSom(gx, gy, data.shape[1],
                  sigma=max(gx, gy)/2, learning_rate=0.5, random_seed=seed)
    som.random_weights_init(data)
    som.train_random(data, num_iteration=len(data)*2)
    return som

def bmu_labels(som, data, gy=10):
    winners = np.array([som.winner(x) for x in data])
    return winners[:,0]*gy + winners[:,1]

def freq_matrix(bmu, node_mc, idxs, labels, n_mc):
    f = np.zeros((len(labels), n_mc))
    for i,(s,e) in enumerate(idxs):
        cell_mc = node_mc[bmu[s:e]]
        for k in range(n_mc):
            f[i,k] = (cell_mc==k).sum() / len(cell_mc) * 100
    return pd.DataFrame(f, index=labels, columns=[f'MC{j+1}' for j in range(n_mc)])

def get_group(l):
    if 'HC' in l: return 'HC'
    if 'PrEP' in l: return 'PrEP'
    if 'W0' in l: return 'HIV_W0'
    return 'HIV_W48'

def bootstrap_cocluster(freq_df, n_boot=1000, k=3, seed=SEED):
    """Per-pair co-clustering probability over bootstrap resamples."""
    np.random.seed(seed)

# ======================================================================
# USER CONFIG — update these paths to match your local setup
# FCS_DIR: directory containing batch-normalized CD3+ FCS files
# OUTPUT_DIR: where figures and results will be saved
# ======================================================================
FCS_DIR    = './normalized_fcs'
OUTPUT_DIR = './output'
DATA_DIR   = '../data/cd3'


    n = len(freq_df)
    cc = np.zeros((n,n))
    ct = np.zeros((n,n))
    X = freq_df.values
    for b in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        uniq = np.unique(idx)
        Z = linkage(X[idx], method='ward', metric='euclidean')
        cls = fcluster(Z, t=k, criterion='maxclust')
        # Map boot positions back to originals
        for pos_i, oi in enumerate(idx):
            for pos_j, oj in enumerate(idx):
                if pos_i >= pos_j: continue
                ct[oi, oj] += 1; ct[oj, oi] += 1
                if cls[pos_i] == cls[pos_j]:
                    cc[oi, oj] += 1; cc[oj, oi] += 1
    prob = np.where(ct>0, cc/ct, 0)
    return pd.DataFrame(prob, index=freq_df.index, columns=freq_df.index)

# ==========================================================
print("="*70)
print("Reproducing IPR/IPNR recovery score with HC n=12 (P88 excluded)")
print("="*70)

t0 = time.time()
file_map = build_files()
combined, labels, idxs = load_data(file_map, n_per=3000)

print("\nTraining SOM (10x10, sigma=5, lr=0.5)...")
som = run_som(combined, 10, 10)
bmu = bmu_labels(som, combined)

print("Metaclustering 20 MC via KMeans...")
weights = som._weights.reshape(-1, combined.shape[1])
km = KMeans(n_clusters=20, random_state=SEED, n_init=10)
node_mc = km.fit_predict(weights)

freq_all = freq_matrix(bmu, node_mc, idxs, labels, 20)
print(f"Full freq matrix: {freq_all.shape} (includes P88)")

# ----------------------------------------------------------
# Scenario A: Original (51 samples, HC n=13 including P88)
# ----------------------------------------------------------
print("\n" + "="*70)
print("Scenario A: Original analysis (HC n=13, includes P88)")
print("="*70)

cc_A = bootstrap_cocluster(freq_all, n_boot=N_BOOT, k=3, seed=SEED)
hc_cols_A = [l for l in freq_all.index if 'HC' in l]
w48_rows  = [l for l in freq_all.index if 'HIV_W48' in l]
print(f"HC n={len(hc_cols_A)}, W48 n={len(w48_rows)}")

scores_A = {p: cc_A.loc[p, hc_cols_A].mean() for p in w48_rows}
scores_A_sorted = sorted(scores_A.items(), key=lambda x: -x[1])
print(f"\n{'Patient':<18} {'Score (A, n=13)':<16}")
for p, s in scores_A_sorted:
    print(f"{p:<18} {s:.3f}")

# ----------------------------------------------------------
# Scenario B: P88 excluded (HC n=12)
# ----------------------------------------------------------
print("\n" + "="*70)
print("Scenario B: CMV-matched HC (P88 excluded, HC n=12)")
print("="*70)

freq_B = freq_all.drop(index=[l for l in freq_all.index if 'P88' in l])
cc_B = bootstrap_cocluster(freq_B, n_boot=N_BOOT, k=3, seed=SEED)
hc_cols_B = [l for l in freq_B.index if 'HC' in l]
print(f"HC n={len(hc_cols_B)}, W48 n={len(w48_rows)}")

scores_B = {p: cc_B.loc[p, hc_cols_B].mean() for p in w48_rows}
scores_B_sorted = sorted(scores_B.items(), key=lambda x: -x[1])

# ----------------------------------------------------------
# Comparison + docx reference
# ----------------------------------------------------------
docx_scores = {
    'P25_HIV_W48':0.86, 'P3_HIV_W48':0.83, 'P20_HIV_W48':0.74, 'P8_HIV_W48':0.73,
    'P13_HIV_W48':0.65, 'P6_HIV_W48':0.58, 'P14_HIV_W48':0.27, 'P7_HIV_W48':0.26,
    'P10_HIV_W48':0.20, 'P12_HIV_W48':0.08, 'P5_HIV_W48':0.08, 'P15_HIV_W48':0.08,
    'P1_HIV_W48':0.06,  'P9_HIV_W48':0.05,
}

print("\n" + "="*80)
print("RECOVERY SCORE comparison: docx vs A (n=13) vs B (n=12 CMV-matched)")
print("="*80)
print(f"{'Patient':<18} {'docx':>7} {'A (n=13)':>9} {'B (n=12)':>9} {'Δ B-docx':>10}  class(docx)")
print("-"*80)

def clsA(s):
    if s >= 0.70: return 'IPR'
    if s >= 0.22: return 'IPR-p'
    return 'IPNR'

all_pts = sorted(docx_scores.keys(), key=lambda p: -docx_scores[p])
for p in all_pts:
    d = docx_scores[p]
    a = scores_A.get(p, np.nan)
    b = scores_B.get(p, np.nan)
    cls_d = 'IPR' if d>=0.70 else ('IPR-p' if d>=0.22 else 'IPNR')
    print(f"{p:<18} {d:>7.2f} {a:>9.3f} {b:>9.3f} {b-d:>+10.3f}  {cls_d}")

# Save
out_df = pd.DataFrame([
    {'patient':p, 'docx':docx_scores[p],
     'A_HC_n13':scores_A.get(p, np.nan),
     'B_HC_n12':scores_B.get(p, np.nan),
     'docx_class': 'IPR' if docx_scores[p]>=0.70 else ('IPR-p' if docx_scores[p]>=0.22 else 'IPNR')}
    for p in all_pts
])
out_path = os.path.join(OUT, 'Recovery_Score_Reproduction.csv')
out_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"\nTotal time: {time.time()-t0:.1f}s")
