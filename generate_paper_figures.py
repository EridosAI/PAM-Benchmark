"""Generate publication-quality figures for the PAM paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'results', 'figures', 'paper')
os.makedirs(OUTDIR, exist_ok=True)

# Colour scheme
C_PRED = '#2166AC'   # blue
C_COS  = '#999999'   # grey
C_BIL  = '#E08040'   # orange
C_PRED_LIGHT = '#92C5DE'  # lighter blue for held-out
C_SHUF = '#D6604D'   # muted red for shuffle

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})


def _save(fig, name):
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'{name}.{ext}')
        fig.savefig(path)
    plt.close(fig)
    print(f'  Saved {name}.png / .pdf')


def _bar_label(ax, bars, fmt='{:.3f}', fontsize=9, offset=0.015, bold=False):
    for bar in bars:
        h = bar.get_height()
        if h < 0.001:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset, fmt.format(h),
                ha='center', va='bottom', fontsize=fontsize,
                fontweight='bold' if bold else 'normal')


# =========================================================================
# Figure 1: Precision gradient
# =========================================================================
def fig1_precision_gradient():
    groups = ['AP@1', 'AP@5', 'AP@20']
    pred   = [0.970, 0.703, 0.216]
    cos    = [0.000, 0.085, 0.045]
    bil    = [0.015, 0.037, 0.022]
    pred_sd = [0.005, 0.001, 0.001]

    x = np.arange(len(groups))
    w = 0.24

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    b1 = ax.bar(x - w, pred, w, label='Predictor', color=C_PRED,
                edgecolor='white', linewidth=0.5,
                yerr=pred_sd, capsize=3, error_kw={'linewidth': 1, 'color': '#333333'})
    b2 = ax.bar(x, cos, w, label='Cosine', color=C_COS,
                edgecolor='white', linewidth=0.5)
    b3 = ax.bar(x + w, bil, w, label='Bilinear', color=C_BIL,
                edgecolor='white', linewidth=0.5)

    _bar_label(ax, b1, bold=True)
    _bar_label(ax, b2, fontsize=8)
    _bar_label(ax, b3, fontsize=8)

    ax.set_ylabel('Association Precision')
    ax.set_title('Association Precision at Retrieval Depths k = 1, 5, 20')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.08)
    ax.legend(loc='upper right', frameon=False)

    fig.tight_layout()
    _save(fig, 'fig1_precision_gradient')


# =========================================================================
# Figure 2: Discrimination AUC comparison
# =========================================================================
def fig2_discrimination_auc():
    groups = ['Overall', 'Cross-Room']
    pred   = [0.916, 0.849]
    cos    = [0.789, 0.503]
    bil    = [0.791, 0.514]
    pred_sd = [0.000, 0.004]

    x = np.arange(len(groups))
    w = 0.25

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    b1 = ax.bar(x - w, pred, w, label='Predictor', color=C_PRED,
                edgecolor='white', linewidth=0.5,
                yerr=pred_sd, capsize=3, error_kw={'linewidth': 1, 'color': '#333333'})
    b2 = ax.bar(x, cos, w, label='Cosine', color=C_COS,
                edgecolor='white', linewidth=0.5)
    b3 = ax.bar(x + w, bil, w, label='Bilinear', color=C_BIL,
                edgecolor='white', linewidth=0.5)

    _bar_label(ax, b1, bold=True)
    _bar_label(ax, b2, fontsize=8)
    _bar_label(ax, b3, fontsize=8)

    ax.axhline(y=0.5, color='#BBBBBB', linestyle='--', linewidth=0.8, zorder=0)
    ax.text(-0.42, 0.508, 'chance', fontsize=8, color='#999999', va='bottom')

    ax.set_ylabel('Discrimination AUC')
    ax.set_title('Discrimination AUC: Overall vs Cross-Room')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.08)
    ax.legend(loc='upper right', frameon=False)

    fig.tight_layout()
    _save(fig, 'fig2_discrimination_auc')


# =========================================================================
# Figure 3: Temporal shuffle ablation
# =========================================================================
def fig3_temporal_shuffle():
    groups = ['CBR@20', 'AP@20']
    normal   = [0.421, 0.216]
    shuffled = [0.044, 0.017]  # 92% reduction from 0.216 -> 0.017
    drop_pct = ['-90%', '-92%']

    x = np.arange(len(groups))
    w = 0.30

    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    b1 = ax.bar(x - w/2, normal, w, label='Normal', color=C_PRED,
                edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w/2, shuffled, w, label='Shuffled', color=C_SHUF,
                edgecolor='white', linewidth=0.5)

    _bar_label(ax, b1, bold=True)
    _bar_label(ax, b2, fontsize=9)

    # Drop percentage annotations
    for i, pct in enumerate(drop_pct):
        mid_x = x[i] + w/2
        top_y = shuffled[i] + 0.045
        ax.annotate(pct, xy=(mid_x, top_y), ha='center', fontsize=10,
                    fontweight='bold', color=C_SHUF)

    ax.set_ylabel('Score')
    ax.set_title('Temporal Shuffle Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 0.55)
    ax.legend(loc='upper right', frameon=False)

    fig.tight_layout()
    _save(fig, 'fig3_temporal_shuffle')


# =========================================================================
# Figure 4: Held-out query-state evaluation
# =========================================================================
def fig4_held_out_query():
    labels = ['Train-Anchor\n(80%)', 'Held-Out\n(20%)']
    values = [0.508, 0.000]

    fig, ax = plt.subplots(figsize=(3.2, 3.5))
    bars = ax.bar(labels, values, width=0.55,
                  color=[C_PRED, C_PRED_LIGHT],
                  edgecolor='white', linewidth=0.5,
                  hatch=[None, '///'])
    # Force hatch colour
    bars[1].set_edgecolor('#666666')

    ax.text(bars[0].get_x() + bars[0].get_width()/2, 0.508 + 0.015, '0.508',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(bars[1].get_x() + bars[1].get_width()/2, 0.000 + 0.015, '0.000',
            ha='center', va='bottom', fontsize=10, color='#666666')

    ax.set_ylabel('Cross-Boundary Recall@20')
    ax.set_title('Held-Out Query-State Recall')
    ax.set_ylim(0, 0.65)

    fig.tight_layout()
    _save(fig, 'fig4_held_out_query')


# =========================================================================
# Figure 5: Transitive chain traversal
# =========================================================================
def fig5_transitive_chains():
    hops = [1, 2, 3]
    pred = [0.455, 0.355, 0.280]
    cos  = [0.000, 0.000, 0.000]

    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    ax.plot(hops, pred, 'o-', color=C_PRED, linewidth=2.0, markersize=7,
            label='Predictor', zorder=3)
    ax.plot(hops, cos, 's--', color=C_COS, linewidth=1.5, markersize=6,
            label='Cosine', zorder=2)

    for h, v in zip(hops, pred):
        ax.text(h, v + 0.018, f'{v:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=C_PRED)

    ax.fill_between(hops, cos, pred, alpha=0.08, color=C_PRED)

    ax.set_xlabel('Hops')
    ax.set_ylabel('Cross-Room R@20')
    ax.set_title('Transitive Chain Recall by Hop Depth')
    ax.set_xticks(hops)
    ax.set_ylim(-0.02, 0.55)
    ax.legend(loc='upper right', frameon=False)

    fig.tight_layout()
    _save(fig, 'fig5_transitive_chains')


# =========================================================================
# Figure 6: Model selection progression
# =========================================================================
def fig6_model_selection():
    configs = [
        'Baseline\n(3L/100k)',
        '+Capacity\n(3L/1024h)',
        '+Depth\n(4L/100k)',
        '+Coverage\n(3L/200k)',
        '+Both\nOnline(D)',
        '+Both\nFixed(D2)',
    ]
    values = [0.037, 0.218, 0.222, 0.305, 0.398, 0.421]
    colors = [C_COS, C_COS, C_COS, C_COS, C_COS, C_PRED]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar(range(len(configs)), values, color=colors,
                  edgecolor='white', linewidth=0.5, width=0.6)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9,
                fontweight='bold' if v == max(values) else 'normal')

    # Highlight D2
    bars[-1].set_edgecolor(C_PRED)
    bars[-1].set_linewidth(1.5)

    # Gain annotation
    ax.annotate(f'11.4x', xy=(5, 0.421), xytext=(4.0, 0.45),
                fontsize=10, fontweight='bold', color=C_PRED,
                arrowprops=dict(arrowstyle='->', color=C_PRED, lw=1.2))
    ax.annotate('', xy=(0, 0.047), xytext=(0, 0.037),
                arrowprops=dict(arrowstyle='-', color='none'))

    ax.set_ylabel('Cross-Room R@20')
    ax.set_title('Architecture Selection: Cross-Room R@20')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylim(0, 0.52)

    fig.tight_layout()
    _save(fig, 'fig6_model_selection')


if __name__ == '__main__':
    print('Generating paper figures...')
    fig1_precision_gradient()
    fig2_discrimination_auc()
    fig3_temporal_shuffle()
    fig4_held_out_query()
    fig5_transitive_chains()
    fig6_model_selection()
    print(f'Done. All figures in {OUTDIR}')
