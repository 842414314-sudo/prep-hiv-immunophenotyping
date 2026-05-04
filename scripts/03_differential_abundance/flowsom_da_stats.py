#!/usr/bin/env python3
"""
FlowSOM: Differential abundance statistics
==========================================
Contrasts tested:
  1. HIV W0 vs HC          - Mann-Whitney U (unpaired)
  2. HIV W0 vs PrEP        - Mann-Whitney U (unpaired)
  3. PrEP    vs HC         - Mann-Whitney U (unpaired)
  4. HIV W0 vs HIV W48     - Wilcoxon signed-rank (paired, matched patient)

Compositional handling: percentages are arcsine-sqrt transformed (variance-
stabilizing for proportions) for effect-size computation; raw percentages
are used for the rank tests since ranks are invariant to monotonic transform.

Multiple testing: Benjamini-Hochberg FDR per contrast (20 MCs).

Outputs:
  Fig/DA_results_<contrast>.csv
  Fig/DA_volcano_panel.pdf            - 4-panel volcano plot
  Fig/DA_boxplots_topMC.pdf           - per-MC boxplots for top hits
  Fig/DA_summary_heatmap.pdf          - contrast x MC signed -log10(p_adj) heatmap
"""

import os, pickle, warnings, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

# ======================================================================
# USER CONFIG — update these paths to match your local setup
# FCS_DIR: directory containing batch-normalized CD3+ FCS files
# OUTPUT_DIR: where figures and results will be saved
# DATA_DIR: root of the data/ directory
# ======================================================================
FCS_DIR    = './normalized_fcs'
OUTPUT_DIR = './output'
DATA_DIR   = '../data'

fig_dir = os.path.join(OUTPUT_DIR, 'Fig')
os.makedirs(fig_dir, exist_ok=True)

# ---------- load data ----------
with open(os.path.join(OUTPUT_DIR, 'flowsom_results.pkl'), 'rb') as f:
    R = pickle.load(f)
with open(os.path.join(OUTPUT_DIR, 'flowsom_mc_annotation.pkl'), 'rb') as f:
    A = pickle.load(f)

freq        = R['freq_matrix']              # (50, 20)
sample_info = R['sample_info']
ann_df      = A['ann_df']                   # refined labels
labels_full = {r['MC']: r['label'] for _, r in ann_df.iterrows()}
mc_subset   = {r['MC']: f"{r['lineage']} {r['subset']}"
               for _, r in ann_df.iterrows()}
N_MC = freq.shape[1]
mc_names = [f'MC{i+1}' for i in range(N_MC)]

meta = pd.DataFrame(sample_info)
meta['cohort'] = np.where(meta['group']=='HIV',
                          'HIV '+meta['timepoint'], meta['group'])

print("Sample breakdown:")
print(meta['cohort'].value_counts().to_string(), '\n')

# ---------- helpers ----------
def asin_sqrt(p):  # p in [0,1]
    p = np.clip(p, 0, 1)
    return np.arcsin(np.sqrt(p))

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonic
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(p)
    out[order] = adj
    return out

def rank_biserial(u, n1, n2):
    # effect size for MW
    return 1 - 2*u / (n1*n2)

def paired_r(stat, n_nonzero):
    # z-approximation r for Wilcoxon
    # stat = W (sum of positive ranks); we don't use directly, use z from p
    return None  # computed inline from z

# ---------- build per-cohort frequency tables ----------
def freqs_for(cohort):
    idx = meta.index[meta['cohort']==cohort].tolist()
    return freq[idx] / 100.0, idx  # proportions

HIV_W0_p,  hiv_w0_idx  = freqs_for('HIV W0')
HIV_W48_p, hiv_w48_idx = freqs_for('HIV W48')
PrEP_p,    prep_idx    = freqs_for('PrEP')
HC_p,      hc_idx      = freqs_for('HC')

# paired HIV: match by patient
hiv_w0_pat  = meta.loc[hiv_w0_idx,  'patient'].tolist()
hiv_w48_pat = meta.loc[hiv_w48_idx, 'patient'].tolist()
common_pat  = sorted(set(hiv_w0_pat) & set(hiv_w48_pat))
w0_order  = [hiv_w0_pat.index(p)  for p in common_pat]
w48_order = [hiv_w48_pat.index(p) for p in common_pat]
HIV_W0_paired  = HIV_W0_p[w0_order]
HIV_W48_paired = HIV_W48_p[w48_order]
print(f"Paired HIV patients: n={len(common_pat)}  IDs={common_pat}")

# ---------- run contrasts ----------
def mw_contrast(A, B, a_name, b_name):
    rows = []
    A_a = asin_sqrt(A); B_a = asin_sqrt(B)
    for j in range(N_MC):
        a, b = A[:, j], B[:, j]
        u, p = mannwhitneyu(a, b, alternative='two-sided')
        mean_a = np.mean(a)*100
        mean_b = np.mean(b)*100
        med_a  = np.median(a)*100
        med_b  = np.median(b)*100
        # fold change on arcsine-sqrt transformed means (avoid 0-div)
        diff_as = np.mean(A_a[:, j]) - np.mean(B_a[:, j])  # arcsine-sqrt diff
        # direction based on raw difference in proportions
        direction = np.sign(mean_a - mean_b)
        rbc = rank_biserial(u, len(a), len(b))
        rows.append({
            'MC'        : mc_names[j],
            'subset'    : mc_subset[mc_names[j]],
            f'mean_{a_name}_%': round(mean_a, 3),
            f'mean_{b_name}_%': round(mean_b, 3),
            f'median_{a_name}_%': round(med_a, 3),
            f'median_{b_name}_%': round(med_b, 3),
            'delta_%'   : round(mean_a - mean_b, 3),
            'asin_diff' : round(diff_as, 4),
            'rank_biserial': round(rbc, 3),
            'p_value'   : p,
        })
    df = pd.DataFrame(rows)
    df['p_adj'] = bh_fdr(df['p_value'].values)
    df['signed_nlog10_padj'] = np.sign(df['delta_%']) * -np.log10(df['p_adj'].clip(lower=1e-300))
    df['contrast'] = f'{a_name}_vs_{b_name}'
    return df

def wilcoxon_paired(A, B, a_name, b_name):
    rows = []
    A_a = asin_sqrt(A); B_a = asin_sqrt(B)
    for j in range(N_MC):
        d = A[:, j] - B[:, j]
        try:
            if np.all(d == 0):
                W, p = 0.0, 1.0
            else:
                W, p = wilcoxon(A[:, j], B[:, j], zero_method='wilcox',
                                alternative='two-sided')
        except Exception:
            W, p = np.nan, 1.0
        mean_a = np.mean(A[:, j])*100
        mean_b = np.mean(B[:, j])*100
        med_a  = np.median(A[:, j])*100
        med_b  = np.median(B[:, j])*100
        diff_as = np.mean(A_a[:, j]) - np.mean(B_a[:, j])
        rows.append({
            'MC'        : mc_names[j],
            'subset'    : mc_subset[mc_names[j]],
            f'mean_{a_name}_%': round(mean_a, 3),
            f'mean_{b_name}_%': round(mean_b, 3),
            f'median_{a_name}_%': round(med_a, 3),
            f'median_{b_name}_%': round(med_b, 3),
            'delta_%'   : round(mean_a - mean_b, 3),
            'asin_diff' : round(diff_as, 4),
            'W'         : W,
            'p_value'   : p,
        })
    df = pd.DataFrame(rows)
    df['p_adj'] = bh_fdr(df['p_value'].values)
    df['signed_nlog10_padj'] = np.sign(df['delta_%']) * -np.log10(df['p_adj'].clip(lower=1e-300))
    df['contrast'] = f'{a_name}_vs_{b_name}'
    return df

contrasts = [
    ('HIV_W0', 'HC',   mw_contrast(HIV_W0_p,  HC_p,    'HIV_W0', 'HC')),
    ('HIV_W0', 'PrEP', mw_contrast(HIV_W0_p,  PrEP_p,  'HIV_W0', 'PrEP')),
    ('PrEP',   'HC',   mw_contrast(PrEP_p,    HC_p,    'PrEP',   'HC')),
    ('HIV_W48','HIV_W0', wilcoxon_paired(HIV_W48_paired, HIV_W0_paired,
                                         'HIV_W48', 'HIV_W0')),
]

# save per-contrast CSVs
for a, b, df in contrasts:
    fn = os.path.join(fig_dir, f'DA_results_{a}_vs_{b}.csv')
    df.sort_values('p_adj').to_csv(fn, index=False)
    n_sig = int((df['p_adj'] < 0.05).sum())
    print(f"  {a} vs {b}:  n_sig(p_adj<0.05) = {n_sig}  -> {os.path.basename(fn)}")

# ---------- volcano panel ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for ax, (a, b, df) in zip(axes.flat, contrasts):
    x = df['asin_diff'].values
    y = -np.log10(df['p_adj'].clip(lower=1e-300).values)
    sig = df['p_adj'] < 0.05
    ax.scatter(x[~sig], y[~sig], s=38, c='#AAAAAA', alpha=0.6, linewidths=0.3,
               edgecolors='black')
    ax.scatter(x[sig],  y[sig],  s=62, c='#E74C3C', alpha=0.9, linewidths=0.4,
               edgecolors='black')
    # annotate sig + top10
    order = np.argsort(df['p_adj'].values)
    labeled = 0
    for k in order:
        if sig.iloc[k] or labeled < 5:
            ax.annotate(f"{df['MC'].iloc[k]}\n{df['subset'].iloc[k]}",
                        (x[k], y[k]),
                        fontsize=6.5, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')
            labeled += 1
        if labeled >= 10 and not sig.iloc[k]:
            break
    ax.axhline(-np.log10(0.05), ls='--', color='k', lw=0.8, alpha=0.5)
    ax.axvline(0, color='k', lw=0.5, alpha=0.4)
    ax.set_xlabel(f'Δ arcsin√p  (positive = {a} higher)')
    ax.set_ylabel('-log10(FDR)')
    ax.set_title(f'{a}  vs  {b}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)

fig.suptitle('Differential Abundance: Volcano (FDR-adjusted)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(fig_dir, 'DA_volcano_panel.pdf'),
            bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(fig_dir, 'DA_volcano_panel.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: DA_volcano_panel.pdf/png")

# ---------- contrast x MC heatmap ----------
hm = np.zeros((len(contrasts), N_MC))
contrast_names = []
for r, (a, b, df) in enumerate(contrasts):
    contrast_names.append(f'{a} vs {b}')
    hm[r] = df.sort_values('MC', key=lambda s:s.str.replace('MC','').astype(int))['signed_nlog10_padj'].values

cmap = LinearSegmentedColormap.from_list(
    'div',['#2166AC','#67A9CF','#F7F7F7','#EF8A62','#B2182B'])
fig, ax = plt.subplots(figsize=(14, 4.3))
vmax = max(2, np.percentile(np.abs(hm), 98))
im = ax.imshow(hm, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax,
               interpolation='nearest')
ax.set_yticks(range(len(contrasts)))
ax.set_yticklabels(contrast_names, fontsize=10)
mc_labels_subset = [f"{m}\n{mc_subset[m]}" for m in mc_names]
ax.set_xticks(range(N_MC))
ax.set_xticklabels(mc_labels_subset, fontsize=7, rotation=90)
cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('signed -log10(FDR)\n(+ = first-higher)', fontsize=9)
# mark significant cells
for r in range(len(contrasts)):
    for j in range(N_MC):
        padj = contrasts[r][2].sort_values('MC',
                    key=lambda s:s.str.replace('MC','').astype(int)
                   )['p_adj'].values[j]
        if padj < 0.05:
            ax.text(j, r, '*', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='black')
ax.set_title('DA summary: signed -log10(FDR)  (★ FDR<0.05)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(fig_dir,'DA_summary_heatmap.pdf'),
            bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(fig_dir,'DA_summary_heatmap.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: DA_summary_heatmap.pdf/png")

# ---------- boxplots for top hits (per contrast) ----------
fig, axes = plt.subplots(len(contrasts), 5, figsize=(20, 3.3*len(contrasts)),
                         sharey=False)
cohort_order  = ['HIV W0','HIV W48','PrEP','HC']
cohort_colors = {'HIV W0':'#E74C3C','HIV W48':'#C0392B',
                 'PrEP':'#3498DB','HC':'#2ECC71'}

for r, (a, b, df) in enumerate(contrasts):
    top = df.nsmallest(5, 'p_adj').reset_index(drop=True)
    for c, row in top.iterrows():
        ax = axes[r, c]
        mc_j = int(row['MC'].replace('MC',''))-1
        data = []
        for coh in cohort_order:
            idx = meta.index[meta['cohort']==coh].tolist()
            data.append(freq[idx, mc_j])
        bp = ax.boxplot(data, widths=0.55, patch_artist=True,
                        medianprops=dict(color='black'))
        for patch, coh in zip(bp['boxes'], cohort_order):
            patch.set_facecolor(cohort_colors[coh])
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(cohort_order)+1))
        ax.set_xticklabels(cohort_order, rotation=30, fontsize=8)
        ax.set_title(f"{row['MC']} {row['subset']}\np_adj={row['p_adj']:.2e}",
                     fontsize=8.5, fontweight='bold')
        ax.set_ylabel('%' if c == 0 else '')
        ax.grid(True, axis='y', alpha=0.25)
    axes[r, 0].annotate(f"{a} vs {b}",
                        xy=(-0.35, 0.5), xycoords='axes fraction',
                        rotation=90, ha='center', va='center',
                        fontsize=11, fontweight='bold')

fig.suptitle('Top-5 differentially abundant metaclusters per contrast',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0.02, 0, 1, 0.96])
fig.savefig(os.path.join(fig_dir,'DA_boxplots_topMC.pdf'),
            bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(fig_dir,'DA_boxplots_topMC.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: DA_boxplots_topMC.pdf/png")

# ---------- print concise summary ----------
print("\n================ DA SUMMARY ================")
for a, b, df in contrasts:
    print(f"\n--- {a} vs {b} ---")
    sig = df[df['p_adj'] < 0.05].sort_values('p_adj')
    if sig.empty:
        print("  No MC reaches FDR<0.05")
    else:
        print(sig[['MC','subset','delta_%','p_value','p_adj']]
              .to_string(index=False))
