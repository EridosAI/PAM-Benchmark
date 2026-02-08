# PAM-Benchmark

**Predictive Associative Memory Benchmark** -- a benchmark for evaluating whether learned predictors can retrieve *associatively linked* memories (temporal co-occurrence) rather than merely *similar* ones (cosine proximity).

## Abstract

Standard embedding-based retrieval finds what *looks like* the query. But biological memory retrieves what *happened with* the query -- states linked by temporal co-occurrence across distinct contexts. This benchmark tests whether a learned predictor (MLP trained with contrastive learning) can bridge that gap.

A synthetic world generates 50,000 states across 20 rooms with 128-dimensional embeddings. Trajectories create temporal associations between states that may share no embedding similarity (different rooms, different objects). The predictor is trained with InfoNCE loss (CLIP-style in-batch negatives) to map a query embedding to the embedding-space region where its associated memories live.

**Key result:** The predictor achieves **R@20 = 0.396** and **MRR = 0.635** on cross-room retrieval, where cosine similarity scores exactly **0.000**. Cosine cannot retrieve what it cannot see; the predictor retrieves what it has learned to associate.

## Results

### Test 1: Association vs Similarity (Cross-Room)

| Metric | Predictor | Cosine | Bilinear |
|--------|-----------|--------|----------|
| R@5    | **0.392** | 0.000  | 0.000    |
| R@10   | **0.378** | 0.000  | 0.001    |
| R@20   | **0.396** | 0.000  | 0.001    |
| MRR    | **0.635** | 0.000  | 0.001    |

### Test 2: Transitive Association (Multi-Hop, Cross-Room)

| Hops   | Predictor R@20 | Cosine | Bilinear |
|--------|----------------|--------|----------|
| 1-hop  | **0.455**      | 0.000  | 0.000    |
| 2-hop  | **0.355**      | 0.000  | 0.000    |
| 3-hop  | **0.280**      | 0.000  | 0.000    |

### Test 3: Decay Ablation

| Scale      | With Decay | Without Decay | Improvement |
|------------|------------|---------------|-------------|
| 50k states | **0.476**  | 0.243         | +96%        |
| 200k states| **0.498**  | 0.067         | +647%       |

### Improvement Progression

| Config                          | R@20      | MRR   | Gain     |
|---------------------------------|-----------|-------|----------|
| NumPy baseline (256h, 200ep)    | 0.037     | 0.060 | --       |
| PyTorch GPU (512h, 500ep)       | 0.089     | ~0.2  | +141%    |
| 1024 hidden (3-layer)           | 0.218     | 0.530 | +145%    |
| **4-layer + 200k pairs (D2)**   | **0.396** | **0.635** | **+82%** |

Total improvement: **10.7x** from original baseline.

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

### Run the Best Config (D2)

```bash
python experiments/run_exp_d2.py
```

This trains a 4-layer MLP with 200k training pairs and evaluates on Tests 1-2. Takes ~350s on an RTX 4080 Super. Results are saved to `results/exp_d2_4layer_200k/results.json`.

### Run the Full 5-Test Benchmark

```bash
python experiments/run_full_benchmark.py
```

Runs all 5 tests (association, transitive, decay, bridging, familiarity) with the 3-layer baseline config. Results saved to `results/full_benchmark_1024h_500ep/results.json`.

### Run Plateau-Breaking Experiments (A-D)

```bash
python experiments/run_plateau_experiments.py
```

Runs experiments A (online sampling), B (4-layer), C (200k pairs), D (combined) sequentially. ~23 min total. Results saved to `results/plateau_experiments/results.json`.

### Generate Figures

```bash
python generate_figures.py
```

Produces 5 publication-quality PNGs in `results/figures/`.

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

## Project Structure

```
PAM-Benchmark/
  world.py              # Synthetic world generator (pure NumPy)
  models_torch.py       # PyTorch predictor architectures
  evaluate_torch.py     # GPU evaluation framework (all 5 tests)
  generate_figures.py   # Publication figure generation
  analyze_results.py    # Results comparison utility
  requirements.txt
  experiments/
    run_exp_d2.py              # Best config (4L + 200k pairs)
    run_full_benchmark.py      # Full 5-test benchmark
    run_plateau_experiments.py # Experiments A-D
    run_combined_best.py       # 1024h + 2000ep
    run_200k_scaled.py         # 200k states scaling
    run_scaleup_exps.py        # Capacity scaling
    run_test4_investigation.py # T4 bridge diagnostics
    run_test4_improved.py      # T4 cross-trajectory v2
    run_test4_v3.py            # T4 targeted objects v3
  results/
    exp_d2_4layer_200k/        # Best config results
    full_benchmark_1024h_500ep/# Full benchmark results
    plateau_experiments/       # Experiments A-D results
    figures/                   # Publication PNGs
    ...                        # Other experiment results
  docs/
    EXPERIMENT_SUMMARY.md      # Full experiment writeup
    HANDOFF.md                 # Technical handoff document
```

## Key Findings

1. **Data coverage is the primary bottleneck.** Going from 41% to 82% coverage of temporal associations nearly doubled R@20.
2. **Model capacity and data interact multiplicatively.** The 4-layer network only helps with sufficient data (200k pairs), not with 100k.
3. **Fixed pairs beat online sampling.** Temporal associations are facts to memorize, not a distribution to approximate.
4. **Decay weighting is the strongest complementary signal.** +96% at 50k states, +647% at 200k.
5. **Cross-trajectory bridging (Test 4) is structurally unsolvable** in the current framework. See [docs/EXPERIMENT_SUMMARY.md](docs/EXPERIMENT_SUMMARY.md) for the full investigation.

## License

Research use. Part of the [Eridos](https://github.com/EridosAI) project.
