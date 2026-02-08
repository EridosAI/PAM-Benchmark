"""Generate publication-quality figures for the Predictive Associative Memory Benchmark.

Updated for faithfulness evaluation paradigm with ablation controls.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'results', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Muted colour palette
C_PRED = '#4878A8'    # steel blue
C_COS = '#B0B0B0'     # grey
C_BILIN = '#D4A574'   # tan
C_DECAY = '#4878A8'   # steel blue
C_NODECAY = '#C07070' # muted red
C_SHUF = '#C07070'    # muted red (shuffle ablation)
C_50K = '#4878A8'
C_200K = '#70A870'    # muted green
C_ACCENT = '#D4A574'  # annotation colour

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
})


def fig1_faithfulness_overview():
    """Overview of all four faithfulness metrics: predictor vs cosine."""
    metrics = [
        'AP@20',
        'CBR@20',
        'AUC\n(all)',
        'AUC\n(x-room)',
        'Specificity\n@20',
    ]
    predictor = [0.2155, 0.4194, 0.9158, 0.8526, 0.3398]
    cosine = [0.0454, 0.0003, 0.7892, 0.5033, 0.0000]

    x = np.arange(len(metrics))
    w = 0.32

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_p = ax.bar(x - w/2, predictor, w, label='Predictor (4L)', color=C_PRED,
                    edgecolor='white', linewidth=0.5)
    bars_c = ax.bar(x + w/2, cosine, w, label='Cosine baseline', color=C_COS,
                    edgecolor='white', linewidth=0.5)

    for i, v in enumerate(predictor):
        ax.text(x[i] - w/2, v + 0.015, f'{v:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    for i, v in enumerate(cosine):
        if v > 0.01:
            ax.text(x[i] + w/2, v + 0.015, f'{v:.3f}', ha='center', va='bottom',
                    fontsize=9, color='#666666')

    ax.set_ylabel('Score')
    ax.set_title('Faithfulness Metrics: Predictor vs Cosine Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    ax.axhline(y=0.5, color='#CCCCCC', linestyle=':', linewidth=1, zorder=0)
    ax.text(len(metrics) - 0.5, 0.51, 'chance', fontsize=9, color='#AAAAAA')

    path = os.path.join(FIGURES_DIR, 'faithfulness_overview.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def fig2_cross_boundary_recall():
    """Cross-boundary recall at various k: the headline metric."""
    k_values = [5, 10, 20, 50]
    predictor = [0.3302, 0.3936, 0.4194, 0.4415]
    cosine = [0.0000, 0.0000, 0.0003, 0.0003]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_values, predictor, 'o-', color=C_PRED, linewidth=2.5, markersize=8,
            label='Predictor (4L)')
    ax.plot(k_values, cosine, 's--', color=C_COS, linewidth=2, markersize=7,
            label='Cosine baseline')

    for k, v in zip(k_values, predictor):
        ax.text(k, v + 0.015, f'{v:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.fill_between(k_values, cosine, predictor, alpha=0.1, color=C_PRED)

    ax.set_xlabel('k (top-k retrieved)')
    ax.set_ylabel('Cross-Boundary Recall@k')
    ax.set_title('Cross-Room Association Retrieval\n(cosine has zero signal here)')
    ax.set_xticks(k_values)
    ax.set_ylim(-0.02, 0.55)
    ax.legend(loc='lower right')

    # MRR annotation
    ax.annotate('MRR = 0.631', xy=(5, 0.33), xytext=(15, 0.20),
                fontsize=11, color=C_PRED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_PRED, lw=1.5))

    path = os.path.join(FIGURES_DIR, 'cross_boundary_recall.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def fig3_temporal_shuffle_ablation():
    """Side-by-side: normal vs temporal shuffle across metrics."""
    metrics = ['CBR@20', 'AUC (x-room)', 'Specificity@20', 'AP@20']
    normal = [0.4194, 0.8526, 0.3398, 0.2155]
    shuffled = [0.0442, 0.5691, 0.1147, 0.0195]
    collapse_pct = [(1 - s/max(n, 1e-6)) * 100 for n, s in zip(normal, shuffled)]

    x = np.arange(len(metrics))
    w = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_n = ax.bar(x - w/2, normal, w, label='Normal (temporal order)',
                    color=C_PRED, edgecolor='white', linewidth=0.5)
    bars_s = ax.bar(x + w/2, shuffled, w, label='Shuffled (random order)',
                    color=C_SHUF, edgecolor='white', linewidth=0.5)

    for i in range(len(metrics)):
        ax.text(x[i] - w/2, normal[i] + 0.015, f'{normal[i]:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(x[i] + w/2, shuffled[i] + 0.015, f'{shuffled[i]:.3f}',
                ha='center', va='bottom', fontsize=10, color=C_SHUF)
        # Collapse annotation
        ax.annotate(f'-{collapse_pct[i]:.0f}%',
                    xy=(x[i] + w/2, shuffled[i] + 0.06),
                    ha='center', fontsize=9, color='#888888', style='italic')

    ax.set_ylabel('Score')
    ax.set_title('Temporal Shuffle Ablation\n(proves model learned temporal structure, not embedding artifacts)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    ax.axhline(y=0.5, color='#CCCCCC', linestyle=':', linewidth=1, zorder=0)

    path = os.path.join(FIGURES_DIR, 'temporal_shuffle_ablation.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def fig4_similarity_matched_discrimination():
    """Bar chart: predictor vs cosine on same-room discrimination."""
    methods = ['Predictor', 'Cosine']
    aucs = [0.8480, 0.7323]
    colors = [C_PRED, C_COS]

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(methods, aucs, color=colors, edgecolor='white', linewidth=0.5, width=0.5)

    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Discrimination AUC')
    ax.set_title('Same-Room Discrimination\n(associates vs non-associated room-mates)')
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color='#CCCCCC', linestyle=':', linewidth=1, zorder=0)
    ax.text(1.3, 0.505, 'chance', fontsize=9, color='#AAAAAA')

    path = os.path.join(FIGURES_DIR, 'similarity_matched_discrimination.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def fig5_improvement_progression():
    """Bar chart showing R@20 progression from 0.037 to 0.396 with annotations."""
    configs = [
        'NumPy\n200ep/256h',
        'PyTorch\n500ep/512h',
        'PyTorch\n2000ep/512h',
        '3L\n500ep/1024h',
        '3L\n2000ep/1024h',
        '4L+200k\n500ep/1024h',
    ]
    values = [0.037, 0.089, 0.107, 0.218, 0.220, 0.396]
    colors = [C_COS, C_COS, C_COS, C_PRED, C_PRED, '#3A6890']

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(range(len(configs)), values, color=colors, edgecolor='white',
                  linewidth=0.5, width=0.65)

    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008, f'{v:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.annotate('GPU + capacity\n+141%', xy=(1, 0.089), xytext=(1.3, 0.16),
                fontsize=9, color='#666666', ha='center',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1))
    ax.annotate('1024 hidden\n+104%', xy=(3, 0.218), xytext=(3.3, 0.30),
                fontsize=9, color='#666666', ha='center',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1))
    ax.annotate('4L + 200k pairs\n+80%', xy=(5, 0.396), xytext=(4.3, 0.36),
                fontsize=9, color='#666666', ha='center',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1))

    ax.annotate('', xy=(3, 0.228), xytext=(4, 0.228),
                arrowprops=dict(arrowstyle='<->', color='#CC8888', lw=1.5))
    ax.text(3.5, 0.235, 'plateau', ha='center', fontsize=9, color='#CC8888')

    ax.set_ylabel('Cross-Boundary Recall@20')
    ax.set_title('Improvement Progression: 10.7x Total Gain')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylim(0, 0.48)

    path = os.path.join(FIGURES_DIR, 'improvement_progression.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def fig6_scaling_analysis():
    """2-panel: R@20 vs hidden dim; R@20 vs training pairs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: R@20 vs hidden dim
    hidden_dims = [256, 512, 1024]
    r20_by_hidden = [0.037, 0.089, 0.218]

    ax1.plot(hidden_dims, r20_by_hidden, 'o-', color=C_PRED, linewidth=2.5, markersize=8)
    for h, r in zip(hidden_dims, r20_by_hidden):
        ax1.text(h, r + 0.008, f'{r:.3f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    ax1.set_xlabel('Hidden Dimension')
    ax1.set_ylabel('Cross-Boundary Recall@20')
    ax1.set_title('Effect of Model Capacity')
    ax1.set_xticks(hidden_dims)
    ax1.set_ylim(0, 0.45)

    # Panel 2: R@20 vs training pairs
    pairs = [100, 200]
    r20_by_pairs_4l = [0.222, 0.396]
    r20_by_pairs_3l = [0.218, 0.305]

    ax2.plot(pairs, r20_by_pairs_4l, 'o-', color='#3A6890', linewidth=2.5,
             markersize=8, label='4-layer')
    ax2.plot(pairs, r20_by_pairs_3l, 's--', color=C_PRED, linewidth=2.5,
             markersize=8, label='3-layer')

    for p, r in zip(pairs, r20_by_pairs_4l):
        ax2.text(p, r + 0.008, f'{r:.3f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    for p, r in zip(pairs, r20_by_pairs_3l):
        ax2.text(p, r - 0.022, f'{r:.3f}', ha='center', va='top', fontsize=10)

    ax2.text(100, 0.04, '41% coverage', ha='center', fontsize=9, color='#888888')
    ax2.text(200, 0.04, '82% coverage', ha='center', fontsize=9, color='#888888')

    ax2.set_xlabel('Training Pairs (thousands)')
    ax2.set_ylabel('Cross-Boundary Recall@20')
    ax2.set_title('Effect of Data Coverage')
    ax2.set_xticks(pairs)
    ax2.set_xticklabels(['100k', '200k'])
    ax2.set_ylim(0, 0.45)
    ax2.legend(loc='upper left')

    fig.suptitle('Scaling Analysis', fontsize=15, y=1.02)

    path = os.path.join(FIGURES_DIR, 'scaling_analysis.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def fig7_generalisation_stress_test():
    """Bar chart: train vs test performance on 70/30 split."""
    conditions = ['Train (70%)\nassociations', 'Held-out (30%)\nassociations', 'Cosine\nbaseline']
    r20_values = [0.578, 0.023, 0.000]
    colors = [C_PRED, '#3A6890', C_COS]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(conditions)), r20_values, color=colors,
                  edgecolor='white', linewidth=0.5, width=0.55)

    for bar, v in zip(bars, r20_values):
        label = f'{v:.3f}' if v > 0 else '0.000'
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, label,
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.annotate('Large gap expected:\nassociative memory\nis faithful recall,\nnot generalisation',
                xy=(1.5, 0.15), fontsize=9, color='#888888', ha='center',
                style='italic')

    ax.set_ylabel('Cross-Boundary Recall@20')
    ax.set_title('Generalisation Stress Test (70/30 edge-disjoint split)\n'
                 'Secondary analysis -- memorisation is correct behaviour')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylim(0, 0.7)

    path = os.path.join(FIGURES_DIR, 'generalisation_stress_test.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


if __name__ == '__main__':
    print('Generating figures...')
    fig1_faithfulness_overview()
    fig2_cross_boundary_recall()
    fig3_temporal_shuffle_ablation()
    fig4_similarity_matched_discrimination()
    fig5_improvement_progression()
    fig6_scaling_analysis()
    fig7_generalisation_stress_test()
    print('Done. All figures saved to results/figures/')
