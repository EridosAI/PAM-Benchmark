# Predictive Associative Memory Benchmark -- Handoff Document

## Overview

A benchmark testing whether a learned predictor can retrieve **associatively linked** memories (temporal co-occurrence) rather than merely **similar** ones (cosine proximity). The predictor is an MLP trained with contrastive learning (InfoNCE / CLIP-style in-batch negatives) to map a query embedding to the region of embedding space where its associated memories live.

Ported to **PyTorch + CUDA** (RTX 4080 Super). Training runs at ~150-200 epochs/min on GPU.

---

## File Inventory

### Core Files
- **`world.py`** -- Synthetic world generator. 128-dim embeddings, 20 rooms, 50 objects, 500 trajectories x 100 steps = 50,000 states. Temporal co-occurrence associations within 5-step window. Optional `cross_trajectory=True` for object-mediated cross-trajectory associations.
- **`models.py`** -- Original NumPy implementation (kept for reference)
- **`models_torch.py`** -- **PyTorch port**. 3-layer MLP architecture with GELU, residual, LayerNorm. AdamW optimizer, cosine LR schedule, temperature annealing. All tensors on CUDA.
- **`evaluate.py`** -- Original NumPy evaluator (kept for reference)
- **`evaluate_torch.py`** -- **PyTorch evaluator with all 5 tests**. Includes GPU-optimized batched spreading activation for Test 4.

### Experiment Scripts
- **`run_benchmark.py`** -- Original NumPy orchestrator
- **`run_improved.py`** -- NumPy with cosine LR + temp annealing
- **`run_scaleup_exps.py`** -- Scale-up experiments (3 configs)
- **`run_combined_best.py`** -- Combined best config (1024h + 2000ep)
- **`run_full_benchmark.py`** -- Full 5-test benchmark (1024h, 500ep)
- **`run_200k_scaled.py`** -- 200k states with 400k training pairs
- **`run_test4_investigation.py`** -- **Test 4 deep investigation** (bridge diagnostics, multi-approach comparison)
- **`run_test4_improved.py`** -- Test 4 with cross-trajectory associations (v2)
- **`run_test4_v3.py`** -- Test 4 with targeted salient-object bridges (v3)
- **`run_plateau_experiments.py`** -- **Plateau-breaking experiments A-D**
- **`run_exp_d2.py`** -- **Experiment D2: 4-layer + 200k fixed pairs (NEW BEST)**

### Results
- `results/full_benchmark_1024h_500ep/results.json` -- Full 5-test benchmark (previous best)
- `results/200k_scaled_400kpairs_1024h/results.json` -- 200k states scaled
- `results/test4_investigation/results.json` -- Test 4 bridge diagnostics
- `results/test4_improved/results.json` -- Test 4 with cross-traj associations
- `results/test4_v3/results.json` -- Test 4 with salient objects
- `results/plateau_experiments/results.json` -- Experiments A-D
- `results/exp_d2_4layer_200k/results.json` -- **New best config results**

---

## Current Best Config (NEW -- 2026-02-08)

```
architecture: 4-layer MLP (128 -> 1024 -> 1024 -> 1024 -> 128)
  fc1: Linear(128 -> 1024) + GELU
  fc2: Linear(1024 -> 1024) + GELU + residual
  fc3: Linear(1024 -> 1024) + GELU + residual
  fc4: Linear(1024 -> 128) + LayerNorm

training_pairs: 200,000 (82% association coverage)
epochs: 500
batch_size: 512
lr: 5e-4 -> 1e-5 (cosine schedule)
temp: 0.15 -> 0.05 (cosine annealing)
parameters: 2,362,752
training time: ~350s on RTX 4080 Super
```

**Key change from previous best:** 4 layers instead of 3, and 200k training pairs instead of 100k. The data coverage (41% -> 82%) was the main bottleneck.

---

## Evolution of Results

### Scale-Up History

| Config | T1 R@20 | Loss | MRR | Time | Notes |
|---|---|---|---|---|---|
| Original (200ep/256h, NumPy) | 0.037 | 2.668 | 0.060 | ~20min CPU | Baseline proof of concept |
| 500ep/512h (PyTorch) | 0.089 | 1.613 | ~0.2 | 149s GPU | First GPU run |
| 2000ep/512h | 0.107 | 1.352 | 0.205 | 599s | +20%, more training helps modestly |
| 500ep/1024h (3-layer) | 0.218 | 0.287 | 0.530 | 147s | +145%, model capacity was bottleneck |
| 2000ep/1024h | 0.220 | 0.194 | 0.536 | 583s | Marginal over 500ep |
| **500ep/1024h (4-layer, 200k pairs)** | **0.396** | **0.409** | **0.635** | **350s** | **+82%, data coverage was bottleneck** |

### Plateau-Breaking Experiments (2026-02-08)

| Experiment | T1 R@20 | T1 MRR | T2 XRoom | Loss | Notes |
|---|---|---|---|---|---|
| Baseline (3L/100k) | 0.218 | 0.530 | 0.185 | 0.287 | Previous best |
| A: Online sampling (3L/100k per ep) | 0.149 | 0.206 | 0.115 | 2.315 | WORSE -- can't converge |
| B: 4-layer (100k fixed) | 0.222 | 0.567 | 0.200 | 0.092 | +2% R@20, +7% MRR, +8% T2 |
| C: 200k pairs (3-layer) | 0.305 | 0.469 | 0.170 | 0.969 | +40% R@20, -12% MRR |
| D: 4L + online 200k/ep | 0.398 | 0.423 | 0.255 | 1.468 | +83% R@20, -20% MRR |
| **D2: 4L + 200k fixed** | **0.396** | **0.635** | **0.355** | **0.409** | **+82% R@20, +20% MRR, +92% T2** |

**Key Findings:**
1. **Data coverage was the main bottleneck** (not model capacity or training time). Going from 41% to 82% coverage of the 242k associations nearly doubled R@20.
2. **4-layer network adds capacity to exploit the larger dataset.** With 100k pairs (B), 4-layer barely helps (+2%). With 200k pairs (D2), it extracts much more value (+30% over C).
3. **Online sampling hurts.** Fresh pairs each epoch prevent the model from deeply learning any specific association. This reduces MRR from 0.635 to 0.423 even though R@20 is similar. Fixed pairs with high coverage are strictly better.
4. **T2 cross-room nearly doubled** (0.185 -> 0.355). The deeper model with more data learns better transitive associations.

---

## Test 4 (Creative Bridging) -- Deep Investigation

### Summary: Structurally Unsolvable

Test 4 was designed to test cross-trajectory bridging: given state s1 in trajectory t1, retrieve state s2 in trajectory t2 via shared objects. After extensive investigation across 3 versions, this task is **structurally unsolvable** with the current world and architecture.

### Investigation (run_test4_investigation.py)

**Bridge Diagnostics:**
- Shared objects per bridge: mean=1.08 (very sparse)
- Embedding similarity (s1, s2): mean=0.125 (low -- room centroids dominate)
- Bridge states in t1 associated with query: 56.7%
- **Oracle (association-hop from bridge states) reaches target: 0/100 (0.0%)**
- Oracle (cosine from bridge states, top-50) reaches target: 2/100 (2.0%)

The oracle result is definitive: even a perfect oracle that finds ALL bridge states in t1 and follows ALL their temporal associations NEVER reaches the target in t2.

**Multi-Approach Comparison (8 approaches x 3 fanout values):**
All approaches performed at chance (~0.3-1.2% R@50):
- Predictor weighted spreading: 0.005
- Unweighted convergence: 0.006
- Softmax convergence: 0.004
- Hybrid (predictor + cosine) hop2: 0.005
- Cosine-only hop2: 0.004
- Full cosine spreading: 0.006
- Direct cosine: 0.005

### Root Causes

1. **No cross-trajectory paths in association graph.** Temporal associations are strictly within-trajectory. The association graph has zero edges between trajectories. No spreading strategy can cross this boundary.

2. **Adding cross-trajectory associations fails (v2).** Adding object-mediated cross-trajectory associations created 30M new links (127x the 242k temporal ones), overwhelming the training signal. T1 R@20 dropped from 0.218 to 0.207, and bridging still failed.

3. **Sparse targeted associations also fail (v3).** Even with only ~1M targeted cross-trajectory links for top-10 salient objects and boosted object scale (2.5x), bridge states were found in hop1 only 8-13% of the time, and target reachable from bridge at 0.2%.

4. **Architectural mismatch.** The predictor maps one embedding to a single point in embedding space. Cross-trajectory associations are one-to-many (one object appears in 450+ trajectories). InfoNCE loss pushes the prediction toward ONE specific positive, but the correct answer is a DISTRIBUTION across many rooms.

5. **Embedding structure.** Room embeddings (scale 2.0) dominate object embeddings (scale 1.5). Cross-room states sharing objects have cosine similarity ~0.12, buried in noise from 50k states.

### What Would Make It Solvable

Creative bridging would require one of:
- An architecture supporting set-valued predictions (e.g., mixture-of-experts output)
- A training regime with explicit curriculum for cross-trajectory links
- A hybrid retrieval system combining temporal predictor with object-identity matching
- A fundamentally different world structure where cross-trajectory paths exist naturally (e.g., recurring visit patterns)

---

## Full Benchmark Results (Previous: 1024h/500ep/3-layer/100k pairs)

| Test | Predictor | Cosine | Bilinear | Notes |
|---|---|---|---|---|
| T1: Association vs Similarity | **0.218** | 0.000 | 0.000 | MRR=0.530, core thesis proven |
| T2: Transitive (Multi-Hop) | **0.185** | 0.000 | 0.000 | 1-hop > 2-hop > 3-hop |
| T3: Decay Ablation | **0.476** | 0.243 | 0.000 | **+96.2%** from decay weighting |
| T4: Creative Bridging | 0.005 | 0.003 | 0.003 | Structurally unsolvable (see above) |
| T5: Familiarity/EMA | 0.000 | 0.000 | 0.000 | 1.08x separation, EMA too weak |

## New Best Results (4-layer/500ep/200k pairs)

| Test | Result | Change vs Prev | Notes |
|---|---|---|---|
| T1: R@20 | **0.396** | **+82%** | From 0.218. Data coverage was bottleneck |
| T1: MRR | **0.635** | **+20%** | From 0.530. 4-layer improves ranking quality |
| T2: Cross-room R@20 | **0.355** | **+92%** | From 0.185. Deeper model learns transitivity better |
| T2: 1-hop R@20 | **0.455** | -- | Direct retrieval now 45.5% |

---

## 200k States Scaling (for reference)

- 200k/400k pairs/1024h: R@20=0.040, loss=1.801
- Data starvation partially explains old failure, but search space expansion is the bigger issue
- T3 decay ablation = 0.498 (+647%) -- recency gating even more critical at scale
- Training data gen is slow: 1172s for 400k pairs (CPU-bound hard negative mining)
- To push further at 200k scale: try 4-layer with proportionally more pairs

---

## What Needs to Be Done Next

### Priority 1: Push 4-Layer Further
- The 4-layer model with 200k pairs still has loss=0.409 (higher than 3-layer's 0.287 on 100k). This suggests more training time or epochs could help.
- Try 1000 epochs with 200k pairs (the model may not have fully converged)
- Try 2048 hidden (more capacity for the larger dataset)

### Priority 2: Investigate the MRR Pattern
- With 100k pairs, 4-layer gets MRR=0.567 (good top-1 precision)
- With 200k pairs, 4-layer gets MRR=0.635 (even better)
- Online sampling gets MRR=0.423 (worse top-1 despite good top-20)
- The model benefits from MEMORIZING specific associations, not just learning the distribution

### Priority 3: Scale to 200k States with New Config
- Previous 200k attempts used 3-layer/100k. Try 4-layer/proportional pairs
- The coverage insight suggests the bottleneck at 200k was also data coverage
- Predict: 4-layer with 800k+ pairs at 200k states should beat 0.040 significantly

### T4 and T5 -- Redesign Needed
- T4 requires a fundamentally different approach (see investigation section)
- T5 EMA signal (1.08x separation) is an architectural limitation of L2 distance in 128-dim space

---

## Environment Notes

- Windows, Python 3.13, PyTorch 2.6.0+cu124
- RTX 4080 Super, CUDA 12.4
- Use `python -u` for unbuffered output when running background tasks
- Windows cp1252 cannot print Unicode characters -- use ASCII alternatives
- `WorldConfig` parameter is `num_trajectories` (not `n_trajectories`)
- Training data generation is CPU-bound: ~72s for 100k pairs, ~143s for 200k pairs
- `get_training_pairs(max_pairs=N)` controls pair count; default 100k
- 4-layer predictor is in `run_plateau_experiments.py` as `AssociativePredictor4Layer`

---

## Architecture

### 3-Layer (Original)
```
AssociativePredictor (nn.Module):
  fc1: Linear(128 -> 1024)  + GELU
  fc2: Linear(1024 -> 1024) + GELU + residual
  fc3: Linear(1024 -> 128)  + LayerNorm
  Parameters: 1,313,152
```

### 4-Layer (New Best)
```
AssociativePredictor4Layer (nn.Module):
  fc1: Linear(128 -> 1024)  + GELU
  fc2: Linear(1024 -> 1024) + GELU + residual
  fc3: Linear(1024 -> 1024) + GELU + residual
  fc4: Linear(1024 -> 128)  + LayerNorm
  Parameters: 2,362,752
```

### Training
```
Training: InfoNCE with in-batch negatives (CLIP-style)
  - B x B similarity matrix (batch=512 -> 511 negatives)
  - Cosine similarity / temperature
  - Cross-entropy loss on each row

Optimizer: AdamW (weight_decay=1e-4)
LR Schedule: Cosine decay 5e-4 -> 1e-5
Temperature: Cosine anneal 0.15 -> 0.05
```

---

## evaluate_torch.py Test Methods

All 5 tests are implemented in the PyTorch evaluator:

- `test_association_vs_similarity()` -- T1: Cross-room R@K and MRR
- `test_transitive_association()` -- T2: Multi-hop chain retrieval (1/2/3 hops)
- `test_decay_ablation()` -- T3: Recency-weighted retrieval via DecayingMemoryStore
- `test_creative_bridging()` -- T4: Dendritic spreading activation with batched GPU inference
- `test_familiarity_normalisation()` -- T5: Running EMA novelty detection
- `run_all()` -- Runs all 5 tests and prints summary

---

## Key Takeaway

The system has proven the core thesis: a learned predictor retrieves cross-room associations that cosine similarity fundamentally cannot. The latest 4-layer model with 200k training pairs achieves **R@20 = 0.396** (vs 0.000 for cosine) and **T2 cross-room = 0.355**.

The 11x improvement from original (0.037 -> 0.396) came from three factors:
1. GPU-enabled capacity scale-up (256 -> 1024 hidden): 6x gain
2. Data coverage (41% -> 82% of associations): 1.8x gain
3. Deeper architecture (3 -> 4 layers): adds capacity to exploit broader coverage

Decay weighting remains the strongest complementary signal (+96%). Test 4 (creative bridging) is structurally unsolvable in the current framework and requires architectural redesign. The current plateau suggests the next gains will come from even higher data coverage, larger models, or applying the 4-layer insight to the 200k-state scale.
