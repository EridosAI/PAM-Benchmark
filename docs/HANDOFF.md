# Predictive Associative Memory Benchmark -- Handoff Document

## Overview

A benchmark testing whether a learned predictor can faithfully recall **associatively linked** memories (temporal co-occurrence) rather than merely **similar** ones (cosine proximity). The predictor is an MLP trained with contrastive learning (InfoNCE / CLIP-style in-batch negatives) to map a query embedding to the region of embedding space where its associated memories live.

**Key framing:** This is an associative memory system, not a general retrieval system. Memorisation of experienced associations is correct behaviour -- that is what memory IS. The primary evaluation measures faithfulness of recall, validated by ablation controls. Generalisation to unseen edges is a secondary stress test.

Ported to **PyTorch + CUDA** (RTX 4080 Super). Training runs at ~85 epochs/min on GPU.

---

## File Inventory

### Core Files
- **`world.py`** -- Synthetic world generator. 128-dim embeddings, 20 rooms, 50 objects, 500 trajectories x 100 steps = 50,000 states. Temporal co-occurrence associations within 5-step window.
- **`models_torch.py`** -- PyTorch models. 3-layer and 4-layer MLP architectures with GELU, residual, LayerNorm. AdamW optimizer, cosine LR schedule, temperature annealing.
- **`evaluate_torch.py`** -- **Evaluation framework.** Contains:
  - Faithfulness metrics: `faithfulness_association_precision()`, `faithfulness_cross_boundary_recall()`, `faithfulness_discrimination_auc()`, `faithfulness_specificity()`, `run_faithfulness()`
  - Legacy tests: `test_association_vs_similarity()`, `test_transitive_association()`, `test_decay_ablation()`, `test_creative_bridging()`, `test_familiarity_normalisation()`
- **`generate_figures.py`** -- 7 publication-quality figures

### Experiment Scripts
- **`experiments/run_faithfulness.py`** -- **PRIMARY: Faithfulness evaluation + ablation controls + generalisation stress test** (~29 min)
- **`experiments/run_exp_d2.py`** -- D2 config with 70/30 split
- **`experiments/run_plateau_experiments.py`** -- Experiments A-D (contains `AssociativePredictor4Layer`)
- **`experiments/run_full_benchmark.py`** -- Full 5-test benchmark (legacy)
- Other experiment scripts for historical reference

### Results
- **`results/faithfulness_evaluation/results.json`** -- Primary results (current)
- `results/exp_d2_4layer_200k/results.json` -- D2 with 70/30 split
- `results/plateau_experiments/results.json` -- Experiments A-D
- `results/figures/` -- 7 publication PNGs

---

## Current Best Config

```
Architecture: 4-layer MLP (128 -> 1024 -> 1024 -> 1024 -> 128)
  fc1: Linear(128 -> 1024) + GELU
  fc2: Linear(1024 -> 1024) + GELU + residual
  fc3: Linear(1024 -> 1024) + GELU + residual
  fc4: Linear(1024 -> 128) + LayerNorm

Training pairs: 200,000 (82% association coverage, from ALL associations)
Epochs: 500
Batch size: 512
LR: 5e-4 -> 1e-5 (cosine schedule)
Temp: 0.15 -> 0.05 (cosine annealing)
Parameters: 2,362,752
Training time: ~353s on RTX 4080 Super
Final loss: 0.409
```

---

## Primary Results: Faithfulness Metrics

| Metric | Predictor | Cosine |
|--------|-----------|--------|
| Association Precision@5 | **0.702** | 0.085 |
| Association Precision@20 | **0.216** | 0.045 |
| Cross-Boundary Recall@20 | **0.419** | 0.000 |
| Cross-Boundary MRR | **0.631** | 0.000 |
| Discrimination AUC (all) | **0.916** | 0.789 |
| Discrimination AUC (x-room) | **0.853** | 0.503 |
| Specificity@20 | **0.340** | 0.000 |

## Ablation Controls

### Temporal Shuffle
Randomly permute temporal ordering within trajectories, retrain. Performance should collapse.

| Metric | Normal | Shuffled | Collapse |
|--------|--------|----------|----------|
| CBR@20 | 0.419 | 0.044 | -90% |
| AUC (x-room) | 0.853 | 0.569 | -33% |
| AP@20 | 0.216 | 0.020 | -91% |

Confirms: model learned temporal structure, not embedding artifacts.

### Similarity-Matched Negatives
Predictor discriminates true associates from same-room non-associates.

| Method | Discrimination AUC |
|--------|-------------------|
| Predictor | **0.848** |
| Cosine | 0.732 |

Confirms: predictor adds value beyond room-level clustering.

## Secondary: Generalisation Stress Test (70/30 split)

| Condition | R@20 | MRR |
|-----------|------|-----|
| Train (70%) | 0.578 | 0.678 |
| Held-out (30%) | 0.023 | 0.014 |
| Cosine | 0.000 | 0.000 |

Large gap expected for a memory system. Held-out R@20=0.023 vs cosine 0.000 is a bonus finding.

---

## Improvement Progression

| Config | CBR@20 | Loss | MRR | Time |
|--------|--------|------|-----|------|
| Original (200ep/256h, NumPy) | 0.037 | 2.668 | 0.060 | ~20min CPU |
| 500ep/512h (PyTorch) | 0.089 | 1.613 | ~0.2 | 149s GPU |
| 2000ep/512h | 0.107 | 1.352 | 0.205 | 599s |
| 500ep/1024h (3-layer) | 0.218 | 0.287 | 0.530 | 147s |
| **500ep/1024h (4-layer, 200k)** | **0.419** | **0.409** | **0.631** | **353s** |

Total: **11.3x** improvement (0.037 -> 0.419)

---

## Test 4 (Creative Bridging) -- Structurally Unsolvable

Oracle analysis proved 0% reachability across trajectory boundaries. Zero cross-trajectory edges in the association graph. 8 spreading approaches x 3 fanouts all at chance. Root cause: predictor maps to single point; cross-trajectory is one-to-many. Requires architectural redesign.

---

## What Could Come Next

### Push Further on Faithfulness
- 1000 epochs (loss=0.409 suggests room to improve)
- Full association coverage (currently 82% -- try 242k pairs directly)
- Curriculum learning: easy (same-room) then hard (cross-room)

### Scale to 200k States
- Previous 200k attempts used 3-layer/100k. Try 4-layer with proportional pairs
- The coverage insight applies here too

### Architecture Experiments
- Attention-based predictor (multi-head for set-valued outputs)
- Test 4 redesign: mixture-of-experts for one-to-many associations

---

## Environment Notes

- Windows, Python 3.13, PyTorch 2.6.0+cu124
- RTX 4080 Super, CUDA 12.4
- Use `python -u` for unbuffered output when running background tasks
- Windows cp1252 cannot print Unicode characters
- `WorldConfig` parameter is `num_trajectories` (not `n_trajectories`)
- Training data generation is CPU-bound: ~143s for 200k pairs
- 4-layer predictor is in `run_plateau_experiments.py` as `AssociativePredictor4Layer`
