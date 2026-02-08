# Predictive Associative Memory Benchmark -- Experiment Summary

## Section A -- Final Results

Best configuration: **4-layer MLP, 200k training pairs, 1024 hidden dim, 500 epochs** (Experiment D2)

### Test 1: Association vs Similarity (Cross-Room Retrieval)

The core thesis test: can the predictor retrieve temporally associated states from different rooms, where cosine similarity provides zero signal?

| Metric | Predictor | Cosine | Bilinear |
|---|---|---|---|
| R@5 | **0.392** | 0.000 | 0.000 |
| R@10 | **0.378** | 0.000 | 0.001 |
| R@20 | **0.396** | 0.000 | 0.001 |
| R@50 | **0.413** | 0.000 | 0.001 |
| MRR | **0.635** | 0.000 | 0.001 |

The predictor retrieves the correct cross-room association in its top-2 predictions on average (MRR 0.635). Cosine similarity and bilinear baselines score effectively zero because cross-room states share no embedding similarity -- only temporal co-occurrence links them.

### Test 2: Transitive Association (Multi-Hop Chain Retrieval)

Can the predictor follow multi-hop association chains across rooms?

| Hops | Predictor R@20 | Cosine R@20 | Bilinear R@20 |
|---|---|---|---|
| 1-hop | **0.455** | 0.000 | 0.000 |
| 2-hop | **0.355** | 0.000 | 0.000 |
| 3-hop | **0.280** | 0.000 | 0.000 |

The predictor sustains retrieval across multiple hops with graceful degradation (45.5% -> 35.5% -> 28.0%). The monotonic decrease confirms the predictor has learned genuine transitive structure rather than a shortcut. Baselines remain at zero for all hop depths.

### Test 3: Decay Ablation

Does recency-weighted retrieval outperform uniform retrieval?

| Condition | Mean Precision |
|---|---|
| With decay (50k states) | **0.476** |
| Without decay (50k states) | 0.243 |
| Improvement | **+96.2%** |
| With decay (200k states) | **0.498** |
| Without decay (200k states) | 0.067 |
| Improvement | **+647%** |

Recency gating nearly doubles precision at 50k states and improves it 7.5x at 200k states. The effect is more pronounced at larger scale because the search space is larger, making temporal recency a stronger discriminative signal.

### Test 4: Creative Bridging -- Structurally Unsolvable

Cross-trajectory bridging via shared objects was investigated across 3 versions with 8 spreading activation approaches, 3 fanout values, and oracle analysis. All approaches performed at chance (~0.5% R@50).

**Root cause:** The association graph contains zero cross-trajectory edges. Temporal associations are strictly within-trajectory. An oracle that perfectly finds all bridge states and follows all their associations still reaches the target 0% of the time. The task is structurally impossible in the current world, not merely hard.

Adding cross-trajectory associations (30M links) overwhelmed the 242k temporal associations. Targeted sparse approaches also failed due to architectural mismatch: the predictor maps to a single point in embedding space, but cross-trajectory associations are one-to-many.

### Test 5: Familiarity/EMA Normalisation

EMA novelty scores showed 1.08x separation between novel and familiar states -- insufficient for meaningful reranking. This is an architectural limitation of L2 distance in 128-dimensional space.

---

## Section B -- Improvement Progression

### Scale-Up History

| Config | Layers | Hidden | Pairs | R@20 | MRR | Loss | Time | Delta |
|---|---|---|---|---|---|---|---|---|
| Original (NumPy, 200ep) | 3 | 256 | 100k | 0.037 | 0.060 | 2.668 | ~20min CPU | -- |
| 500ep/512h (PyTorch) | 3 | 512 | 100k | 0.089 | ~0.2 | 1.613 | 149s GPU | +141% |
| 2000ep/512h | 3 | 512 | 100k | 0.107 | 0.205 | 1.352 | 599s | +20% |
| 500ep/1024h | 3 | 1024 | 100k | 0.218 | 0.530 | 0.287 | 147s | +104% |
| 2000ep/1024h | 3 | 1024 | 100k | 0.220 | 0.536 | 0.194 | 583s | +1% |
| **500ep/1024h (D2)** | **4** | **1024** | **200k** | **0.396** | **0.635** | **0.409** | **350s** | **+80%** |

Total improvement: **10.7x** (0.037 -> 0.396)

### Plateau-Breaking Experiments (A-D)

Starting from the 3-layer/100k baseline (R@20=0.218), four experiments isolated the contribution of each factor:

| Experiment | Change | T1 R@20 | T1 MRR | T2 XRoom | Loss | Verdict |
|---|---|---|---|---|---|---|
| Baseline | 3L, 100k fixed | 0.218 | 0.530 | 0.185 | 0.287 | -- |
| A: Online sampling | Fresh 100k/epoch | 0.149 | 0.206 | 0.115 | 2.315 | Harmful |
| B: 4-layer | 4L, 100k fixed | 0.222 | 0.567 | 0.200 | 0.092 | Marginal |
| C: 200k pairs | 3L, 200k fixed | 0.305 | 0.469 | 0.170 | 0.969 | Major gain |
| D: 4L + online 200k | 4L, 200k/epoch | 0.398 | 0.423 | 0.255 | 1.468 | Good R@20, bad MRR |
| **D2: 4L + 200k fixed** | **4L, 200k fixed** | **0.396** | **0.635** | **0.355** | **0.409** | **Best overall** |

---

## Section C -- Key Findings

### 1. Data coverage was the primary bottleneck

The world contains 242,264 temporal associations. With 100k training pairs, only 41% of associations are covered. Increasing to 200k pairs (82% coverage) nearly doubled R@20 from 0.218 to 0.305 even with the same 3-layer architecture (Experiment C). This was the single largest improvement factor.

### 2. Model capacity and data coverage interact multiplicatively

With 100k pairs, a 4-layer network barely helps (+2% R@20, Experiment B). With 200k pairs, the 4-layer network extracts substantially more value (+30% over 3-layer, comparing C vs D2). The deeper model has the capacity to learn the richer signal in the larger dataset. Neither factor alone accounts for the full gain.

### 3. Fixed pairs beat online sampling

Online sampling (fresh pairs each epoch) prevents deep memorization of specific associations. Despite seeing more total pairs, the model converges poorly (loss 2.315 vs 0.409) and MRR drops from 0.635 to 0.423. Temporal associations are facts to be memorized, not a distribution to be approximated. The fixed-pair regime lets the model repeatedly reinforce the same associations to high confidence.

### 4. Test 4 (creative bridging) is structurally unsolvable

Oracle analysis proved 0% reachability across trajectory boundaries. Three progressive attempts (raw spreading, cross-trajectory associations, targeted salient objects) all confirmed the same conclusion. The fundamental issues are: (a) no cross-trajectory edges in the association graph, (b) adding them overwhelms temporal signal, (c) the predictor architecture maps to a single point, but cross-trajectory associations are one-to-many. Solving this requires architectural redesign (e.g., mixture-of-experts output, hybrid retrieval).

### 5. Decay weighting is the strongest complementary signal

Recency-weighted retrieval improves precision by 96% at 50k states and 647% at 200k states (Test 3). This effect grows with scale because larger search spaces make temporal recency an increasingly powerful discriminator. Decay weighting is orthogonal to model improvements and should always be used.

### 6. The 10.7x improvement came from three distinct factors

| Factor | Contribution | Mechanism |
|---|---|---|
| GPU-enabled capacity (256->1024 hidden) | ~6x | Wider bottleneck allows richer representations |
| Data coverage (41%->82% of associations) | ~1.8x | More associations seen = more retrievable |
| Deeper architecture (3->4 layers) | Multiplier on data | Extra capacity exploits broader coverage |

---

## Section D -- Reproducible Config Spec

### World

```python
from world import WorldConfig, SyntheticWorld

config = WorldConfig(
    num_rooms=20,
    objects_per_room=5,       # 50 objects total (some shared)
    embedding_dim=128,
    room_embedding_scale=2.0,
    object_embedding_scale=1.5,
    num_trajectories=500,
    steps_per_trajectory=100,  # 50,000 states total
    association_window=5,
    seed=42
)
world = SyntheticWorld(config)
```

### Training Data

```python
pairs = world.get_training_pairs(max_pairs=200000)
# Returns ~200k (query, target) index pairs
# Coverage: ~82% of 242,264 temporal associations
# Generation time: ~143s (CPU-bound hard negative mining)
```

### Architecture

```python
# AssociativePredictor4Layer from run_plateau_experiments.py
class AssociativePredictor4Layer(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=1024, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)   # 128 -> 1024
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)       # 1024 -> 1024
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)       # 1024 -> 1024
        self.fc4 = nn.Linear(hidden_dim, embedding_dim)    # 1024 -> 128
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        h1 = F.gelu(self.fc1(x))
        h2 = F.gelu(self.fc2(h1)) + h1   # residual
        h3 = F.gelu(self.fc3(h2)) + h2   # residual
        output = self.layer_norm(self.fc4(h3))
        return output

# Parameters: 2,362,752
```

### Training

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

# Cosine LR schedule: 5e-4 -> 1e-5
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

# Temperature annealing: 0.15 -> 0.05 (cosine schedule)
temp_start, temp_end = 0.15, 0.05

# InfoNCE with in-batch negatives (CLIP-style)
# batch_size=512 -> 511 negatives per positive
# B x B cosine similarity matrix / temperature
# Cross-entropy loss on each row

epochs = 500
batch_size = 512
```

### Training Performance

```
Hardware: NVIDIA RTX 4080 Super, CUDA 12.4
Training time: ~350 seconds
Throughput: ~85 epochs/min
Final loss: 0.409
```

### Expected Results

```
T1 Association vs Similarity:
  R@20 = 0.396    (cosine: 0.000, bilinear: 0.001)
  MRR  = 0.635    (cosine: 0.000, bilinear: 0.001)

T2 Transitive (Cross-Room):
  1-hop R@20 = 0.455
  2-hop R@20 = 0.355
  3-hop R@20 = 0.280

T3 Decay Ablation:
  With decay    = 0.476
  Without decay = 0.243
  Improvement   = +96.2%
```
