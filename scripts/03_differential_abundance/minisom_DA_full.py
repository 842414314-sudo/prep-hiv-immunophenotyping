#!/usr/bin/env python3
"""
Full Differential Abundance (DA) analysis pipeline
===================================================
Runs 6 pairwise contrasts per subset (CD3, CD4, CD8) on MC frequencies.

Contrasts:
  1. HIV_W0  vs HC       (MW, unpaired)
  2. HIV_W0  vs PrEP     (MW)
  3. PrEP    vs HC       (MW)
  4. HIV_W48 vs HIV_W0   (Wilcoxon signed-rank, PAIRED)
  5. HIV_W48 vs HC       (MW)
  6. HIV_W48 vs PrEP     (MW)

Method:
  - arcsin-sqrt transform: x' = arcsin(sqrt(p/100))  (variance-stabilising for %)
  - Mann-Whitney U (two-sided) for unpaired; Wilcoxon signed-rank for paired
  - BH FDR correction WITHIN each contrast (n = N_MC tests)
  - Effect size: Δ arcsin√p = mean(arcsin√% group_A) - mean(arcsin√% group_B)

Inputs:
  - CD3: /Volumes/.../Fig/Data/FlowSOM_metacluster_frequencies.csv  (50 samples × 20 MC)
        + /Volumes/.../02_Normalized_V11/Fig/FlowSOM_MC_Annotation_Table.csv
  - CD4: /Volumes/.../MiniSom_Final/CD4_CD8_MCSweep/freq_CD4_MC10.csv  (51 × 10)
        + profiles_CD4_MC10.csv  (auto-label by signature)
  - CD8: idem for CD8

Outputs (under DA/<subset>/):
  - DA_results_all.csv          all contrasts × MC (Δ, p, p_adj)
  - heatmap_signedFDR.pdf/png   contrasts × MC matrix, signed -log10(FDR)
  - volcano_grid.pdf/png        2×3 grid (adjustText for labels)
  - top5_boxplots.pdf/png       6 rows × 5 cols, top-5 per contrast
"""

import os, re, warnings
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from adjustText import adjust_text

warnings.filterwarnings('ignore')

# ======================================================================
# USER CONFIG — update these paths to match your local setup
# DATA_DIR: root of the data/ directory.
#           Default resolves to <repo>/data based on this script's location,
#           so the script runs from any working directory. Override if your
#           data lives elsewhere.
# OUTPUT_DIR: where DA results and figures will be saved
# ======================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'data'))
OUTPUT_DIR  = './output'

# ============================================================
# Config
# ============================================================
OUT_ROOT = OUTPUT_DIR
os.makedirs(OUT_ROOT, exist_ok=True)

GROUP_COLORS = {'HIV W0':'#FFCD65','HIV W48':'#808000','PrEP':'#3E0080','HC':'#008000'}
# Internal keys (no space) used in CSV labels
GK = {'HIV_W0':'HIV W0','HIV_W48':'HIV W48','PrEP':'PrEP','HC':'HC'}

CONTRASTS = [
    ('HIV_W0',  'HC',       'unpaired'),
    ('HIV_W0',  'PrEP',     'unpaired'),
    ('PrEP',    'HC',       'unpaired'),
    ('HIV_W48', 'HIV_W0',   'paired'),
    ('HIV_W48', 'HC',       'unpaired'),
    ('HIV_W48', 'PrEP',     'unpaired'),
]

# ============================================================
# Utilities
# ============================================================
def bh_adjust(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n)
    cum = 1.0
    for i in range(n-1, -1, -1):
        cum = min(cum, ranked[i] * n / (i+1))
        adj[order[i]] = min(cum, 1.0)
    return adj

def arcsin_sqrt(p_pct):
    """p_pct in 0..100. arcsin-sqrt of proportion."""
    p = np.clip(np.asarray(p_pct, float)/100.0, 0, 1)
    return np.arcsin(np.sqrt(p))

def parse_group(label):
    """Labels come in two formats:
      CD3 (original ref):  P1_HIV_W0, P1_HIV_W48, P18_PrEP_W0, P39_HC_W0
      CD4/CD8 (my sweep):  P1_HIV_W0, P1_HIV_W48, P18_PrEP,    P39_HC
    Handle both.
    """
    if re.search(r'_HIV_W0(_|$)',  label): return 'HIV_W0'
    if re.search(r'_HIV_W48(_|$)', label): return 'HIV_W48'
    if re.search(r'_PrEP(_|$)',    label): return 'PrEP'
    if re.search(r'_HC(_|$)',      label): return 'HC'
    return 'Unknown'

def parse_patient(label):
    m = re.match(r'P(\d+)', label)
    return m.group(1) if m else label

# ============================================================
# Auto-label CD4/CD8 MCs from z-score profile (used when no annotation CSV given)
# ============================================================
def label_cd4(prof_row, markers):
    z = dict(zip(markers, prof_row))
    f = {m: z.get(m, 0) for m in markers}
    def hi(m, t=0.5):  return f[m] > t
    def lo(m, t=-0.3): return f[m] < t
    # Treg
    if f['FOXP3'] > 1.0 and f['CD25'] > 0.5 and f['CD127'] < 0:
        return 'aTreg' if f['HLADR'] > 0.5 else 'Treg'
    # Activated
    if f['HLADR'] > 1.0 or (f['HLADR'] > 0.4 and f['PD1'] > 0.5):
        return 'Activated(HLADR+)'
    # Naive subsets
    if hi('CD45RA') and hi('CCR7') and hi('CD27') and lo('CD95'):
        if f['CD38'] > 0.4:   return 'Tnaive(CD38hi)'
        if f['CD38'] < -0.3:  return 'Tnaive(CD38lo)'
        return 'Tnaive'
    # Tscm (CD45RA+CD95+)
    if hi('CD45RA') and hi('CD95') and hi('CCR7'):
        return 'Tscm'
    # Temra
    if hi('CD45RA') and lo('CCR7') and lo('CD28'):
        return 'Temra'
    # Tcm
    if lo('CD45RA') and hi('CCR7') and hi('CD27') and hi('CD28'):
        return 'Tcm'
    # Tem
    if lo('CD45RA') and lo('CCR7'):
        return 'Tem'
    return 'Memory'

def label_cd8(prof_row, markers):
    z = dict(zip(markers, prof_row))
    f = {m: z.get(m, 0) for m in markers}
    def hi(m, t=0.5):  return f[m] > t
    def lo(m, t=-0.3): return f[m] < t
    # Activated
    if f['HLADR'] > 0.8 and f['CD38'] > 0.3:
        return 'Activated(CD38+HLADR+)'
    if f['PD1'] > 1.0 and f['TIM3'] > 0.5:
        return 'Exhausted(PD1+TIM3+)'
    if f['TIM3'] > 1.5:
        return 'TIM3+'
    # Naive
    if hi('CD45RA') and hi('CCR7') and hi('CD27') and lo('CD95'):
        return 'Tnaive(CD38hi)' if f['CD38'] > 0.4 else 'Tnaive'
    if hi('CD45RA') and hi('CCR7') and hi('CD95'):
        return 'Tscm'
    # Temra
    if hi('CD45RA') and lo('CCR7'):
        return 'Temra'
    # Effector memory
    if lo('CD45RA') and lo('CCR7') and lo('CD28'):
        return 'Tem(senescent)'
    if lo('CD45RA') and lo('CCR7'):
        return 'Tem'
    if lo('CD45RA') and hi('CCR7'):
        return 'Tcm'
    return 'Memory'

# ============================================================
# Load a subset's freq matrix + MC labels
# ============================================================
def load_subset(name):
    if name == 'CD3':
        freq_csv = os.path.join(DATA_DIR, 'cd3', 'cd3_mc20_frequencies.csv')
        da_csv = os.path.join(DATA_DIR, 'cd3', 'cd3_da_results.csv')
        df = pd.read_csv(freq_csv, index_col=0)
        anno = pd.read_csv(da_csv)
        # Build label dict: 'MC1' → 'MC1 CD8+ Exh(TIM3+)'
        label_map = dict(zip(anno['MC'], anno['label'])) if 'label' in anno.columns else {f'MC{i+1}': f'MC{i+1}' for i in range(20)}
    else:
        subset_lower = name.lower()
        freq_csv = os.path.join(DATA_DIR, subset_lower, f'{subset_lower}_mc10_frequencies.csv')
        prof_csv = os.path.join(DATA_DIR, subset_lower, f'{subset_lower}_mc10_profiles.csv')
        df = pd.read_csv(freq_csv, index_col=0)
        prof = pd.read_csv(prof_csv, index_col=0)
        markers = list(prof.columns)
        labeler = label_cd4 if name == 'CD4' else label_cd8
        label_map = {}
        for mc in prof.index:
            pheno = labeler(prof.loc[mc].values, markers)
            label_map[mc] = f'{mc} {name}+ {pheno}'
    # Attach metadata
    df = df.copy()
    df['__group']   = [parse_group(s)   for s in df.index]
    df['__patient'] = [parse_patient(s) for s in df.index]
    return df, label_map

# ============================================================
# Compute DA for one contrast
# ============================================================
def run_contrast(df, mc_cols, grpA, grpB, kind):
    """Return per-MC dict of {MC, delta, p, direction}."""
    out = []
    if kind == 'paired':
        # Match patients present in both groups
        pa = set(df[df['__group'] == grpA]['__patient'])
        pb = set(df[df['__group'] == grpB]['__patient'])
        common = sorted(pa & pb, key=lambda x: int(x) if x.isdigit() else x)
        if len(common) < 3:
            raise ValueError(f"Too few paired samples for {grpA} vs {grpB}: {len(common)}")
        A = df[df['__group'] == grpA].set_index('__patient').loc[common]
        B = df[df['__group'] == grpB].set_index('__patient').loc[common]
        for mc in mc_cols:
            a = arcsin_sqrt(A[mc].values)
            b = arcsin_sqrt(B[mc].values)
            diff = a - b
            if np.allclose(diff, 0):
                p, stat = 1.0, 0.0
            else:
                try:
                    stat, p = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
                except Exception:
                    p, stat = 1.0, 0.0
            out.append({'MC': mc,
                        'mean_A_%': A[mc].mean(), 'mean_B_%': B[mc].mean(),
                        'median_A_%': A[mc].median(), 'median_B_%': B[mc].median(),
                        'delta_arcsin': a.mean() - b.mean(),
                        'n_A': len(A), 'n_B': len(B), 'p': p})
    else:
        A = df[df['__group'] == grpA]
        B = df[df['__group'] == grpB]
        for mc in mc_cols:
            a = arcsin_sqrt(A[mc].values)
            b = arcsin_sqrt(B[mc].values)
            if len(a) < 2 or len(b) < 2:
                p = 1.0
            else:
                try:
                    _, p = mannwhitneyu(a, b, alternative='two-sided')
                except Exception:
                    p = 1.0
            out.append({'MC': mc,
                        'mean_A_%': A[mc].mean(), 'mean_B_%': B[mc].mean(),
                        'median_A_%': A[mc].median(), 'median_B_%': B[mc].median(),
                        'delta_arcsin': a.mean() - b.mean(),
                        'n_A': len(A), 'n_B': len(B), 'p': p})
    res = pd.DataFrame(out)
    res['p_adj'] = bh_adjust(res['p'].values)
    res['signed_logFDR'] = np.sign(res['delta_arcsin']) * (-np.log10(np.clip(res['p_adj'], 1e-300, 1.0)))
    res['grpA'] = grpA; res['grpB'] = grpB; res['test'] = kind
    return res

# ============================================================
# Figures
# ============================================================
def make_heatmap(all_res, label_map, mc_order, subset, outpath):
    """signed -log10(FDR) heatmap; rows=contrasts, cols=MCs.  Square cells."""
    # Matrix
    contrast_names = [f'{a.replace("_"," ")} vs {b.replace("_"," ")}' for a,b,_ in CONTRASTS]
    M = np.zeros((len(CONTRASTS), len(mc_order)))
    S = np.zeros((len(CONTRASTS), len(mc_order)), dtype=bool)  # star for FDR<0.05
    for i, (a,b,_) in enumerate(CONTRASTS):
        key = f'{a}_vs_{b}'
        sub = all_res[all_res['contrast'] == key].set_index('MC')
        for j, mc in enumerate(mc_order):
            if mc in sub.index:
                M[i,j] = sub.loc[mc, 'signed_logFDR']
                S[i,j] = sub.loc[mc, 'p_adj'] < 0.05

    # Shorten long labels for x-axis
    def shorten(mc):
        s = label_map.get(mc, mc).replace(mc, '').strip()
        s = s.replace(f'{subset}+ ', '')
        # Abbreviations
        for long, short in [
            ('Activated(CD38+HLADR+)', 'Act.'),
            ('Activated(HLADR+)',      'Act.'),
            ('Activated',              'Act.'),
            ('Exhausted',              'Exh.'),
            ('Exh(TIM3+)',             'Exh.'),
            ('Exhausted(PD1+TIM3+)',   'Exh.'),
            ('Temra',                  'TemRA'),
            ('Tnaive(CD38hi)',         'Tn(CD38hi)'),
            ('Tnaive(CD38lo)',         'Tn(CD38lo)'),
            ('Tnaive',                 'Tn'),
            ('Tem(senescent)',         'Tem(sen)'),
            ('(CD38+HLADR+)',          ''),  # safety net
        ]:
            s = s.replace(long, short)
        return s

    vmax = max(3.0, np.ceil(np.nanmax(np.abs(M))))
    ncols = len(mc_order)

    # Square cells — pick a size and dimension the figure from it
    cellsize = 0.52 if ncols >= 15 else 0.68
    left_pad = 2.6    # inches for y-labels
    right_pad = 1.8   # inches for colorbar
    top_pad = 0.8     # title
    bot_pad = 3.0     # x-labels (multi-line)

    plot_w = ncols * cellsize
    plot_h = len(CONTRASTS) * cellsize
    fig_w = plot_w + left_pad + right_pad
    fig_h = plot_h + top_pad + bot_pad

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(M, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_yticks(range(len(contrast_names)))
    ax.set_yticklabels(contrast_names, fontsize=10.5, fontweight='bold')
    ax.set_xticks(range(ncols))
    xlabels = [f'{mc}\n{shorten(mc)}' for mc in mc_order]
    ax.set_xticklabels(xlabels, rotation=90, fontsize=8, fontweight='bold')
    ax.tick_params(axis='x', pad=2)
    # Stars
    for i in range(len(contrast_names)):
        for j in range(ncols):
            if S[i,j]:
                ax.text(j, i, '★', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if abs(M[i,j])>vmax*0.35 else 'black')
    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.015)
    cb.set_label('signed -log10(FDR)\n(+ = first-higher)', fontsize=9.5)
    cb.ax.tick_params(labelsize=9)
    ax.set_title(f'{subset}  DA summary:  signed -log10(FDR)   (★ FDR<0.05)',
                 fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    for ext in ('pdf','png'):
        fig.savefig(f'{outpath}.{ext}', bbox_inches='tight', facecolor='white',
                    dpi=260 if ext=='png' else None)
    plt.close(fig)

def make_volcano(all_res, label_map, subset, outpath):
    """2x3 grid, 6 contrasts."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for ax, (a,b,_) in zip(axes.flat, CONTRASTS):
        key = f'{a}_vs_{b}'
        sub = all_res[all_res['contrast'] == key]
        x = sub['delta_arcsin'].values
        y = -np.log10(np.clip(sub['p_adj'].values, 1e-300, 1.0))
        sig = sub['p_adj'].values < 0.05
        ax.scatter(x[~sig], y[~sig], c='lightgrey', s=40, edgecolors='none', rasterized=True)
        ax.scatter(x[sig], y[sig], c='#E04040', s=55, edgecolors='black',
                   linewidths=0.5, rasterized=True)
        ax.axhline(-np.log10(0.05), ls='--', lw=0.8, c='grey')
        ax.axvline(0, lw=0.8, c='black')
        # adjustText labels (only significant + top-8 by |delta|)
        texts = []
        rows = sub[sub['p_adj'] < 0.05].sort_values(by='delta_arcsin',
                                                    key=lambda s: s.abs(),
                                                    ascending=False).head(12)
        for _, r in rows.iterrows():
            full = label_map.get(r['MC'], r['MC'])
            # Compact label: MC# + phenotype short
            short_ph = full.replace(r['MC'], '').replace(f'{subset}+','').strip()
            lbl = f"{r['MC']} {short_ph}" if short_ph else r['MC']
            texts.append(ax.text(r['delta_arcsin'],
                                 -np.log10(max(r['p_adj'], 1e-300)),
                                 lbl, fontsize=9, fontweight='bold'))
        if texts:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='grey', lw=0.5),
                        expand=(1.15, 1.25),
                        force_points=(0.3, 0.4))
        ax.set_xlabel(f'Δ arcsin√p  (positive = {a} higher)', fontsize=11, fontweight='bold')
        ax.set_ylabel('-log10(FDR)', fontsize=11, fontweight='bold')
        ax.set_title(f'{a.replace("_"," ")}  vs  {b.replace("_"," ")}',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2)
        for s_ in ['top','right']: ax.spines[s_].set_visible(False)
    fig.suptitle(f'{subset} — Differential Abundance (FDR-adjusted, 6 contrasts)',
                 fontsize=17, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    for ext in ('pdf','png'):
        fig.savefig(f'{outpath}.{ext}', bbox_inches='tight', facecolor='white',
                    dpi=200 if ext=='png' else None)
    plt.close(fig)

def make_top5_boxplots(df, all_res, label_map, subset, outpath):
    """6 rows (contrasts) × 5 cols (top-5 by p_adj)."""
    fig, axes = plt.subplots(len(CONTRASTS), 5, figsize=(22, 4.2*len(CONTRASTS)))
    group_order = ['HIV_W0','HIV_W48','PrEP','HC']
    box_colors  = [GROUP_COLORS[GK[g]] for g in group_order]
    for ri, (a,b,_) in enumerate(CONTRASTS):
        key = f'{a}_vs_{b}'
        sub = all_res[all_res['contrast'] == key].sort_values('p_adj').head(5)
        for ci in range(5):
            ax = axes[ri, ci] if axes.ndim == 2 else axes[ci]
            if ci >= len(sub):
                ax.axis('off'); continue
            row = sub.iloc[ci]
            mc = row['MC']
            full = label_map.get(mc, mc)
            data = [df[df['__group']==g][mc].values for g in group_order]
            bp = ax.boxplot(data, labels=[GK[g] for g in group_order],
                            patch_artist=True, widths=0.6,
                            medianprops=dict(color='black', linewidth=1.2))
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            # jitter points
            for gi, g in enumerate(group_order):
                vals = df[df['__group']==g][mc].values
                jit = np.random.normal(gi+1, 0.05, len(vals))
                ax.scatter(jit, vals, s=12, c='black', alpha=0.55, zorder=3)
            title = f"{full}\np_adj = {row['p_adj']:.2e}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_ylabel('%' if ci==0 else '')
            ax.tick_params(axis='x', labelsize=9, rotation=25)
            ax.tick_params(axis='y', labelsize=9)
            for s_ in ['top','right']: ax.spines[s_].set_visible(False)
        # Left-side contrast label
        axes[ri, 0].annotate(f'{a.replace("_"," ")}\nvs\n{b.replace("_"," ")}',
                              xy=(-0.32, 0.5), xycoords='axes fraction',
                              fontsize=12, fontweight='bold',
                              ha='center', va='center', rotation=0)
    fig.suptitle(f'{subset} — Top-5 differentially abundant MCs per contrast',
                 fontsize=17, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0.035, 0, 1, 0.985])
    for ext in ('pdf','png'):
        fig.savefig(f'{outpath}.{ext}', bbox_inches='tight', facecolor='white',
                    dpi=180 if ext=='png' else None)
    plt.close(fig)

# ============================================================
# Driver
# ============================================================
def run_subset(name):
    print(f"\n{'='*70}\n{name} DA analysis\n{'='*70}")
    df, label_map = load_subset(name)
    mc_cols = [c for c in df.columns if c.startswith('MC')]
    mc_order = sorted(mc_cols, key=lambda s: int(s.replace('MC','')))
    print(f"  {len(df)} samples × {len(mc_cols)} MCs")
    print(f"  group counts: {df['__group'].value_counts().to_dict()}")

    outdir = os.path.join(OUT_ROOT, name)
    os.makedirs(outdir, exist_ok=True)

    all_results = []
    for a, b, kind in CONTRASTS:
        res = run_contrast(df, mc_order, a, b, kind)
        res['contrast'] = f'{a}_vs_{b}'
        all_results.append(res)
        n_sig = (res['p_adj'] < 0.05).sum()
        print(f"  [{kind:<8}] {a:<8} vs {b:<8}  → {n_sig:>2}/{len(mc_order)} FDR<0.05")
    all_res = pd.concat(all_results, ignore_index=True)
    all_res.to_csv(os.path.join(outdir, 'DA_results_all.csv'), index=False)

    # Save label map
    pd.DataFrame([{'MC':k, 'label':v} for k,v in label_map.items()]
                 ).to_csv(os.path.join(outdir, 'MC_labels.csv'), index=False)

    # Figures
    make_heatmap(all_res, label_map, mc_order, name,
                 os.path.join(outdir, f'{name}_DA_heatmap'))
    make_volcano(all_res, label_map, name,
                 os.path.join(outdir, f'{name}_DA_volcano'))
    make_top5_boxplots(df, all_res, label_map, name,
                       os.path.join(outdir, f'{name}_DA_top5_boxplot'))
    print(f"  → outputs in {outdir}")

np.random.seed(42)
for name in ['CD3', 'CD4', 'CD8']:
    run_subset(name)

print("\nALL DONE.")
