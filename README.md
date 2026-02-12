# Predictive Associative Memory

**Retrieval Beyond Similarity Through Temporal Co-occurrence**

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18595537.svg)](https://doi.org/10.5281/zenodo.18595537)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Standard embedding-based retrieval finds what *looks like* the query. But biological memory retrieves what *happened with* the query — states linked by temporal co-occurrence, not representational similarity. Stairs do not resemble a slip, yet one reliably evokes the other.

**Predictive Associative Memory (PAM)** trains a JEPA-style predictor on temporal co-occurrence within a continuous experience stream. An *Inward* predictor — the complement to the standard *Outward* JEPA that predicts future sensory states — learns which past states are associatively reachable from the current state. The result retrieves associations across representational boundaries where similarity-based methods score zero.

## Key Results

Faithfulness of associative recall (mean ± SD across training seeds 42, 123, 456):

| Metric | Predictor | Cosine Similarity | Bilinear |
|---|---|---|---|
| Association Precision@1 | **0.970 ± 0.005** | 0.000 | 0.015 |
| Association Precision@5 | **0.703 ± 0.001** | 0.085 | 0.037 |
| Cross-Boundary Recall@20 | **0.421 ± 0.002** | 0.000 | 0.000 |
| Discrimination AUC (overall) | **0.916 ± 0.000** | 0.789 | 0.791 |
| Discrimination AUC (cross-room) | **0.849 ± 0.004** | 0.503 | 0.514 |
| Specificity@20 | **0.338 ± 0.005** | 0.000 | 0.000 |

The predictor's top retrieval is a true temporal associate **97% of the time**. On cross-room pairs — where embedding similarity is uninformative — it achieves AUC = 0.849 while cosine similarity is at chance.

### Ablation Controls

| Control | Result | Interpretation |
|---|---|---|
| Temporal shuffle | CBR@20 collapses 90% (0.421 → 0.044) | Signal is learned temporal structure, not embedding geometry |
| Similarity-matched negatives | Predictor AUC = 0.848 vs Cosine AUC = 0.732 | Predictor discriminates true associates from same-room distractors |

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- NumPy, Matplotlib

```bash
pip install -r requirements.txt
```

### Reproduce Results

```bash
python experiments/run_faithfulness.py
```

Trains on all associations, runs faithfulness metrics, ablation controls, and generalisation stress test. Takes ~29 minutes on an RTX 4080 Super. Results saved to `results/faithfulness_evaluation/results.json`.

### Generate Figures

```bash
python generate_figures.py
```

Produces publication-quality figures in `results/figures/`.

## Architecture

**Predictor (4-layer MLP, 2.36M parameters):**

```
Input:  128-dim embedding
fc1:    Linear(128 → 1024)  + GELU
fc2:    Linear(1024 → 1024) + GELU + residual
fc3:    Linear(1024 → 1024) + GELU + residual
fc4:    Linear(1024 → 128)  + LayerNorm
Output: 128-dim predicted embedding
```

**Training:** InfoNCE loss with in-batch negatives (batch size 512), cosine LR schedule (5×10⁻⁴ → 1×10⁻⁵), temperature annealing (0.15 → 0.05), 500 epochs over 200,000 training pairs covering 82% of 242,264 temporal associations.

**Synthetic world:** 20 rooms, 50 objects, 128-dimensional embeddings, 500 trajectories of 100 timesteps (50,000 total states), temporal co-occurrence window τ = 5.

## Evaluation Paradigm

PAM is evaluated as an **associative recall** system — testing faithfulness of recall for experienced associations — not as a retrieval system evaluated on generalisation to unseen associations. Memorisation of experienced associations is correct behaviour, not a flaw. The failure modes are false association (hallucinating links) and loss of specificity (retrieving categories instead of specific experiences).

See Section 4.1 of the paper for the full discussion of recall vs retrieval evaluation.

## Citation

```bibtex
@article{dury2025pam,
  title={Predictive Associative Memory: Retrieval Beyond Similarity Through Temporal Co-occurrence},
  author={Dury, Jason},
  year={2025},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

## License

MIT
