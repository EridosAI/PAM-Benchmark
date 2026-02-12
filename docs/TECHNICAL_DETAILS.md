# PAM-Benchmark

**Predictive Associative Memory Benchmark** -- a benchmark for evaluating whether learned predictors can faithfully recall *associatively linked* memories (temporal co-occurrence) rather than merely *similar* ones (cosine proximity).

DOI: 10.5281/zenodo.18595537.

## Abstract

Standard embedding-based retrieval finds what *looks like* the query. But biological memory retrieves what *happened with* the query -- states linked by temporal co-occurrence across distinct contexts. This benchmark tests whether a learned predictor (MLP trained with contrastive learning) can bridge that gap.

A synthetic world generates 50,000 states across 20 rooms with 128-dimensional embeddings. Trajectories create temporal associations between states that may share no embedding similarity (different rooms, different objects). The predictor is trained on ALL associations with InfoNCE loss (CLIP-style in-batch negatives) to map a query embedding to the embedding-space region where its associated memories live.

**This is an associative memory system, not a general retrieval system.** Faithful recall of experienced associations is the correct behaviour. The failure modes are false association (hallucinating links that don't exist) or loss of specificity (retrieving whole categories instead of specific experiences).

## Key Results

### Primary: Faithfulness Metrics (train on ALL associations)

| Metric | Predictor | Cosine | What it measures |
|--------|-----------|--------|-----------------|
| **Association Precision@20** | **0.216** | 0.045 | Of top-20, what fraction are true associates? |
| **Cross-Boundary Recall@20** | **0.419** | 0.000 | Recall for associations crossing room boundaries |
| **Cross-Boundary MRR** | **0.631** | 0.000 | Mean reciprocal rank for cross-room associates |
| **Discrimination AUC (all)** | **0.916** | 0.789 | ROC-AUC separating associates from non-associates |
| **Discrimination AUC (x-room)** | **0.853** | 0.503 | Same, restricted to cross-room (cosine = chance) |
| **Specificity@20** | **0.340** | 0.000 | Retrieves specific associates, not whole categories |

The predictor retrieves the correct cross-room association in its top-2 predictions on average (MRR 0.631). Cosine similarity scores effectively zero on cross-room retrieval because cross-room states share no embedding similarity -- only temporal co-occurrence links them.

### Ablation Controls

| Ablation | Result | Interpretation |
|----------|--------|---------------|
| **Temporal shuffle** | CBR@20 collapses 90% (0.419 -> 0.044) | Model learned temporal structure, not embedding artifacts |
| **Similarity-matched negatives** | Predictor AUC=0.848 vs Cosine AUC=0.732 | Predictor discriminates associates from same-room non-associates |

### Secondary: Generalisation Stress Test (70/30 edge-disjoint split)

| Condition | R@20 | MRR |
|-----------|------|-----|
| Train associations (70%) | 0.578 | 0.678 |
| Held-out associations (30%) | 0.023 | 0.014 |
| Cosine baseline | 0.000 | 0.000 |

The large train/test gap is expected and correct: associative memory is faithful recall of experienced associations, not generalisation to unseen edges. The held-out R@20=0.023 vs cosine's 0.000 still demonstrates non-trivial cross-room retrieval on never-seen associations.

### Improvement Progression

| Config | CBR@20 | MRR | Gain |
|--------|--------|-----|------|
| NumPy baseline (256h, 200ep) | 0.037 | 0.060 | -- |
| PyTorch GPU (512h, 500ep) | 0.089 | ~0.2 | +141% |
| 1024 hidden (3-layer) | 0.218 | 0.530 | +145% |
| **4-layer + 200k pairs (D2)** | **0.419** | **0.631** | **+92%** |

Total improvement: **11.3x** from original baseline.

## Reproducing Results

### Requirements

```
Python 3.10+
PyTorch 2.0+ (CUDA recommended)
NumPy
Matplotlib (for figures only)
```

```bash
pip install -r requirements.txt
```

### Run the Faithfulness Evaluation (recommended)

```bash
python experiments/run_faithfulness.py
```

Trains on ALL associations, runs four faithfulness metrics, two ablation controls, and the generalisation stress test. Takes ~29min on an RTX 4080 Super. Results saved to `results/faithfulness_evaluation/results.json`.

### Generate Figures

```bash
python generate_figures.py
```

Produces 7 publication-quality PNGs in `results/figures/`.

### Run Individual Components

```bash
# Best config training only
python experiments/run_exp_d2.py

# Plateau-breaking experiments A-D
python experiments/run_plateau_experiments.py
```

## Architecture

### Best Config (D2): 4-Layer MLP

```
Input:  128-dim embedding
fc1:    Linear(128 -> 1024)  + GELU
fc2:    Linear(1024 -> 1024) + GELU + residual
fc3:    Linear(1024 -> 1024) + GELU + residual
fc4:    Linear(1024 -> 128)  + LayerNorm
Output: 128-dim predicted embedding

Parameters: 2,362,752
```

### Training

```
Loss:        InfoNCE with in-batch negatives (CLIP-style)
Batch size:  512 (511 negatives per positive)
Optimizer:   AdamW (weight_decay=1e-4)
LR:          5e-4 -> 1e-5 (cosine schedule)
Temperature: 0.15 -> 0.05 (cosine annealing)
Epochs:      500
Data:        200,000 training pairs (82% coverage of 242k associations)
```

### Synthetic World

```
Rooms:             20
Objects per room:  5 (some shared across rooms)
Embedding dim:     128
Trajectories:      500
Steps/trajectory:  100
Total states:      50,000
Association window: 5 steps (temporal co-occurrence)
```

## Evaluation Paradigm

### Primary: Faithfulness (train on ALL associations)

This is an associative memory system. The correct evaluation asks: *how faithfully does the predictor recall the associations it was trained on?* Four metrics capture different aspects:

1. **Association Precision@k** -- Of the top-k retrieved items, what fraction are true temporal associates? Measures precision of the learned mapping.
2. **Cross-Boundary Recall@k** -- Recall restricted to associations that cross room boundaries. This is the headline differentiator: cosine similarity has zero signal for cross-room associations, so any non-zero performance demonstrates learned temporal structure.
3. **Discrimination AUC** -- ROC-AUC for separating true associates from non-associates. Measures the predictor's ability to rank associates above non-associates across the full memory store.
4. **Specificity** -- Does the predictor retrieve the *specific* associated item or the whole category? Measures whether the model learned fine-grained temporal associations rather than broad room-level patterns.

### Ablation Controls

1. **Temporal shuffle** -- Randomly permute temporal ordering within trajectories, retrain. If performance collapses, the model learned temporal structure, not embedding artifacts.
2. **Similarity-matched negatives** -- For each query, measure discrimination between true associates and same-room non-associates. This controls for the possibility that the predictor merely learned room-level clustering.

### Secondary: Generalisation Stress Test

A 70/30 edge-disjoint split tests whether the predictor can retrieve associations it has never seen. The large gap (R@20: 0.578 train vs 0.023 test) is expected for a memory system -- memorisation of experienced associations is the correct behaviour, not a flaw.

## Project Structure

```
PAM-Benchmark/
  world.py              # Synthetic world generator (pure NumPy)
  models_torch.py       # PyTorch predictor architectures
  evaluate_torch.py     # Faithfulness evaluation framework
  generate_figures.py   # Publication figure generation
  analyze_results.py    # Results comparison utility
  requirements.txt
  experiments/
    run_faithfulness.py        # Primary evaluation + ablations (recommended)
    run_exp_d2.py              # Best config (4L + 200k pairs)
    run_plateau_experiments.py # Experiments A-D
    run_full_benchmark.py      # Full 5-test benchmark (legacy)
    run_combined_best.py       # 1024h + 2000ep
    run_200k_scaled.py         # 200k states scaling
    run_scaleup_exps.py        # Capacity scaling
    run_test4_investigation.py # T4 bridge diagnostics
    run_test4_improved.py      # T4 cross-trajectory v2
    run_test4_v3.py            # T4 targeted objects v3
    run_d2_split_t2.py         # D2 with split evaluation
  results/
    faithfulness_evaluation/   # Primary results
    exp_d2_4layer_200k/        # Best config results
    plateau_experiments/       # Experiments A-D results
    figures/                   # Publication PNGs
    ...                        # Other experiment results
  docs/
    EXPERIMENT_SUMMARY.md      # Full experiment writeup
    HANDOFF.md                 # Technical handoff document
```

## Key Findings

1. **Faithful associative recall across embedding boundaries.** The predictor achieves CBR@20=0.419 and Discrimination AUC=0.853 on cross-room associations where cosine similarity is at chance (0.503 AUC).
2. **Temporal structure, not embedding artifacts.** Temporal shuffle ablation collapses CBR@20 by 90%, confirming the model learned genuine temporal associations.
3. **Discriminates associates from room-mates.** Similarity-matched negatives ablation shows AUC=0.848 for distinguishing true associates from same-room non-associates (cosine: 0.732).
4. **Data coverage is the primary bottleneck.** Going from 41% to 82% coverage of temporal associations nearly doubled performance.
5. **Model capacity and data interact multiplicatively.** The 4-layer network only helps with sufficient data (200k pairs), not with 100k.
6. **Cross-trajectory bridging (Test 4) is structurally unsolvable** in the current framework. See [docs/EXPERIMENT_SUMMARY.md](docs/EXPERIMENT_SUMMARY.md) for the full investigation.

## License

Research use. Part of the [Eridos](https://github.com/EridosAI) project.
