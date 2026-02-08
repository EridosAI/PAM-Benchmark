"""
D2 re-run with split-aware T2 transitive test.

Trains on 70% associations, evaluates:
  - T1 on held-out 30% (same as before)
  - T2 pure_test: A->B test, B->C test (fully held-out chains)
  - T2 train_to_test: A->B train, B->C test (chain generalization)
"""

import time
import torch
import numpy as np
from pathlib import Path
import json

BATCH_SIZE = 512
LR_START = 5e-4
LR_END = 1e-5
TEMP_START = 0.15
TEMP_END = 0.05
SEED = 42
EPOCHS = 500
HIDDEN_DIM = 1024
NUM_TRAJECTORIES = 500

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from world import SyntheticWorld, WorldConfig
from models_torch import (
    LearnedBilinearBaseline, train_predictor, DEVICE
)
from evaluate_torch import BenchmarkEvaluator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from run_plateau_experiments import AssociativePredictor4Layer

print("=" * 70)
print("D2 with Split-Aware T2 Transitive Test")
print("=" * 70)
t_start = time.time()

# Generate world
config = WorldConfig(num_trajectories=NUM_TRAJECTORIES)
world = SyntheticWorld(config)
world.generate_trajectories()
world.compute_association_ground_truth()

# Split
train_assoc, test_assoc = world.split_associations(train_ratio=0.7, seed=SEED)

# Train on train split only
print("\nGenerating 200k training pairs from TRAIN split...")
t0 = time.time()
anchors, positives, _ = world.get_training_pairs(max_pairs=200000, associations=train_assoc)
print(f"Pair generation: {time.time()-t0:.1f}s")

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

pred = AssociativePredictor4Layer(
    embedding_dim=config.embedding_dim,
    hidden_dim=HIDDEN_DIM,
    seed=SEED
)
print(f"Parameters: {pred.count_parameters():,}")

t0 = time.time()
best_loss = train_predictor(
    pred, anchors, positives,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    lr_start=LR_START, lr_end=LR_END,
    temp_start=TEMP_START, temp_end=TEMP_END,
    print_every=100
)
train_time = time.time() - t0
print(f"Training: {train_time:.1f}s, Loss: {best_loss:.4f}")

bilinear = LearnedBilinearBaseline(config.embedding_dim, seed=123)
bilinear.train(anchors, positives, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

# --- T1 on held-out test ---
print("\n" + "=" * 70)
print("T1: Held-out test associations")
print("=" * 70)
eval_test = BenchmarkEvaluator(world, pred, bilinear, test_associations=test_assoc)
t1_test = eval_test.test_association_vs_similarity()

# --- T1 on train (for comparison) ---
print("\n" + "=" * 70)
print("T1: Train associations (comparison)")
print("=" * 70)
eval_train = BenchmarkEvaluator(world, pred, bilinear, test_associations=train_assoc)
t1_train = eval_train.test_association_vs_similarity()

# --- T2 split-aware ---
print("\n" + "=" * 70)
print("T2: Split-Aware Transitive Chains")
print("=" * 70)
t2_split = eval_test.test_transitive_split(train_assoc, test_assoc)

# --- Summary ---
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")

test_r20 = t1_test.predictor_score
test_mrr = t1_test.details['mrr']['predictor']
train_r20 = t1_train.predictor_score
train_mrr = t1_train.details['mrr']['predictor']

print(f"\nT1 Association vs Similarity:")
print(f"  Train R@20: {train_r20:.3f}  MRR: {train_mrr:.3f}")
print(f"  Test  R@20: {test_r20:.3f}  MRR: {test_mrr:.3f}")
print(f"  Gap:  R@20 {train_r20 - test_r20:+.3f}  MRR {train_mrr - test_mrr:+.3f}")

print(f"\nT2 Transitive (cross-room R@20):")
for label in ['pure_test', 'train_to_test']:
    if label in t2_split:
        r = t2_split[label]
        print(f"  {label}:")
        print(f"    1-hop: {r[20]['pred_1hop']:.3f}  2-hop: {r[20]['pred_2hop']:.3f}  "
              f"3-hop: {r[20]['pred_3hop']:.3f}  cosine: {r[20]['cosine']:.3f}")

print(f"\nLoss: {best_loss:.4f}  Time: {train_time:.0f}s")

# Save
output_dir = Path("results") / "d2_split_t2"
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "results.json", 'w') as f:
    json.dump({
        'split': '70/30 edge-disjoint',
        'T1_test': {'R@20': test_r20, 'MRR': test_mrr, 'details': t1_test.details},
        'T1_train': {'R@20': train_r20, 'MRR': train_mrr, 'details': t1_train.details},
        'T2_split': t2_split,
        'loss': float(best_loss),
        'time': float(train_time),
        'train_associations': len(train_assoc),
        'test_associations': len(test_assoc),
    }, f, indent=2, default=str)

print(f"\nTotal time: {time.time()-t_start:.1f}s")
