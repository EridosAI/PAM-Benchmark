"""Generate publication-quality figures for the Predictive Associative Memory Benchmark."""

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


def fig1_association_vs_similarity():
    """Grouped bar chart: R@5, R@10, R@20, MRR across 3 methods."""
    metrics = ['R@5', 'R@10', 'R@20', 'MRR']
    predictor = [0.392, 0.378, 0.396, 0.635]
    cosine = [0.000, 0.000, 0.000, 0.000]
    bilinear = [0.000, 0.000, 0.001, 0.001]

    x = np.arange(len(metrics))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, predictor, w, label='Predictor (4L)', color=C_PRED, edgecolor='white', linewidth=0.5)
    ax.bar(x, cosine, w, label='Cosine', color=C_COS, edgecolor='white', linewidth=0.5)
    ax.bar(x + w, bilinear, w, label='Bilinear', color=C_BILIN, edgecolor='white', linewidth=0.5)

    # Value labels on predictor bars
    for i, v in enumerate(predictor):
        ax.text(x[i] - w, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title('Test 1: Association vs Similarity (Cross-Room)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.75)
    ax.legend(loc='upper left')

    path = os.path.join(FIGURES_DIR, 'test1_association_vs_similarity.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def fig2_transitive_crossroom():
    """Grouped bar chart: 1/2/3-hop R@20 across 3 methods."""
    hops = ['1-hop', '2-hop', '3-hop']
    predictor = [0.455, 0.355, 0.280]
    cosine = [0.000, 0.000, 0.000]
    bilinear = [0.000, 0.000, 0.000]

    x = np.arange(len(hops))
    w = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w, predictor, w, label='Predictor (4L)', color=C_PRED, edgecolor='white', linewidth=0.5)
    ax.bar(x, cosine, w, label='Cosine', color=C_COS, edgecolor='white', linewidth=0.5)
    ax.bar(x + w, bilinear, w, label='Bilinear', color=C_BILIN, edgecolor='white', linewidth=0.5)

    for i, v in enumerate(predictor):
        ax.text(x[i] - w, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('R@20')
    ax.set_title('Test 2: Transitive Association (Cross-Room)')
    ax.set_xticks(x)
    ax.set_xticklabels(hops)
    ax.set_ylim(0, 0.55)
    ax.legend(loc='upper right')

    path = os.path.join(FIGURES_DIR, 'test2_transitive_crossroom.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def fig3_decay_ablation():
    """2-panel line chart: with/without decay over steps for 50k and 200k."""
    steps = [50, 100, 150, 200, 250, 300, 350, 400, 450]

    # 50k states
    decay_50k = [0.467, 0.533, 0.350, 0.383, 0.550, 0.483, 0.550, 0.517, 0.450]
    nodecay_50k = [0.283, 0.250, 0.233, 0.217, 0.283, 0.183, 0.283, 0.217, 0.233]

    # 200k states
    decay_200k = [0.433, 0.550, 0.533, 0.433, 0.417, 0.450, 0.633, 0.567, 0.467]
    nodecay_200k = [0.050, 0.067, 0.183, 0.033, 0.050, 0.100, 0.033, 0.050, 0.033]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # 50k panel
    ax1.plot(steps, decay_50k, 'o-', color=C_DECAY, label='With decay', linewidth=2, markersize=5)
    ax1.plot(steps, nodecay_50k, 's--', color=C_NODECAY, label='Without decay', linewidth=2, markersize=5)
    ax1.set_xlabel('Trajectory Step')
    ax1.set_ylabel('Precision')
    ax1.set_title('50k States (+96%)')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 0.75)

    # Fill between
    ax1.fill_between(steps, nodecay_50k, decay_50k, alpha=0.1, color=C_DECAY)

    # 200k panel
    ax2.plot(steps, decay_200k, 'o-', color=C_DECAY, label='With decay', linewidth=2, markersize=5)
    ax2.plot(steps, nodecay_200k, 's--', color=C_NODECAY, label='Without decay', linewidth=2, markersize=5)
    ax2.set_xlabel('Trajectory Step')
    ax2.set_title('200k States (+647%)')
    ax2.legend(loc='upper right')

    ax2.fill_between(steps, nodecay_200k, decay_200k, alpha=0.1, color=C_DECAY)

    fig.suptitle('Test 3: Decay Ablation', fontsize=15, y=1.02)

    path = os.path.join(FIGURES_DIR, 'test3_decay_ablation.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def fig4_improvement_progression():
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
    bars = ax.bar(range(len(configs)), values, color=colors, edgecolor='white', linewidth=0.5, width=0.65)

    # Value labels
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008, f'{v:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Annotations for key transitions
    ax.annotate('GPU + capacity\n+141%', xy=(1, 0.089), xytext=(1.3, 0.16),
                fontsize=9, color='#666666', ha='center',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1))

    ax.annotate('1024 hidden\n+104%', xy=(3, 0.218), xytext=(3.3, 0.30),
                fontsize=9, color='#666666', ha='center',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1))

    ax.annotate('4L + 200k pairs\n+80%', xy=(5, 0.396), xytext=(4.3, 0.36),
                fontsize=9, color='#666666', ha='center',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1))

    # Plateau bracket
    ax.annotate('', xy=(3, 0.228), xytext=(4, 0.228),
                arrowprops=dict(arrowstyle='<->', color='#CC8888', lw=1.5))
    ax.text(3.5, 0.235, 'plateau', ha='center', fontsize=9, color='#CC8888')

    ax.set_ylabel('T1 R@20')
    ax.set_title('Improvement Progression: 10.7x Total Gain')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylim(0, 0.48)

    path = os.path.join(FIGURES_DIR, 'improvement_progression.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def fig5_scaling_analysis():
    """2-panel: R@20 vs hidden dim; R@20 vs training pairs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: R@20 vs hidden dim (3-layer, 100k pairs, 500ep)
    hidden_dims = [256, 512, 1024]
    r20_by_hidden = [0.037, 0.089, 0.218]

    ax1.plot(hidden_dims, r20_by_hidden, 'o-', color=C_PRED, linewidth=2.5, markersize=8)
    for h, r in zip(hidden_dims, r20_by_hidden):
        ax1.text(h, r + 0.008, f'{r:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Hidden Dimension')
    ax1.set_ylabel('T1 R@20')
    ax1.set_title('Effect of Model Capacity')
    ax1.set_xticks(hidden_dims)
    ax1.set_ylim(0, 0.45)

    # Panel 2: R@20 vs training pairs (4-layer, 1024h, 500ep)
    pairs = [100, 200]  # in thousands
    r20_by_pairs_4l = [0.222, 0.396]
    r20_by_pairs_3l = [0.218, 0.305]

    ax2.plot(pairs, r20_by_pairs_4l, 'o-', color='#3A6890', linewidth=2.5, markersize=8, label='4-layer')
    ax2.plot(pairs, r20_by_pairs_3l, 's--', color=C_PRED, linewidth=2.5, markersize=8, label='3-layer')

    for p, r in zip(pairs, r20_by_pairs_4l):
        ax2.text(p, r + 0.008, f'{r:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for p, r in zip(pairs, r20_by_pairs_3l):
        ax2.text(p, r - 0.022, f'{r:.3f}', ha='center', va='top', fontsize=10)

    # Coverage annotations
    ax2.text(100, 0.04, '41% coverage', ha='center', fontsize=9, color='#888888')
    ax2.text(200, 0.04, '82% coverage', ha='center', fontsize=9, color='#888888')

    ax2.set_xlabel('Training Pairs (thousands)')
    ax2.set_ylabel('T1 R@20')
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


if __name__ == '__main__':
    print('Generating figures...')
    fig1_association_vs_similarity()
    fig2_transitive_crossroom()
    fig3_decay_ablation()
    fig4_improvement_progression()
    fig5_scaling_analysis()
    print('Done. All figures saved to results/figures/')
