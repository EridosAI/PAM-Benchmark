"""
Experiment D2: 4-layer + 200k fixed pairs (combining B and C).

The original D used 4-layer + online sampling at 200k/epoch.
This tests 4-layer + 200k FIXED pairs to see if fixed pairs
with higher capacity outperforms online sampling.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    AssociativePredictor, LearnedBilinearBaseline,
    train_predictor, DEVICE
)
from evaluate_torch import BenchmarkEvaluator
from run_plateau_experiments import AssociativePredictor4Layer

print("=" * 70)
print("EXPERIMENT D2: 4-Layer + 200k Fixed Pairs")
print("=" * 70)
t_start = time.time()

# Generate world
config = WorldConfig(num_trajectories=NUM_TRAJECTORIES)
world = SyntheticWorld(config)
world.generate_trajectories()
world.compute_association_ground_truth()

# Generate 200k pairs
print("\nGenerating 200k training pairs...")
t0 = time.time()
anchors, positives, _ = world.get_training_pairs(max_pairs=200000)
print(f"Time: {time.time()-t0:.1f}s")

# Train 4-layer predictor
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
    print_every=50
)
train_time = time.time() - t0
print(f"Training: {train_time:.1f}s, Loss: {best_loss:.4f}")

# Bilinear
bilinear = LearnedBilinearBaseline(config.embedding_dim, seed=123)
bilinear.train(anchors, positives, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

# Evaluate T1 and T2
evaluator = BenchmarkEvaluator(world, pred, bilinear)
t1 = evaluator.test_association_vs_similarity()
t2 = evaluator.test_transitive_association()

r20 = t1.predictor_score
mrr_val = t1.details['mrr']['predictor']
t2_cross = t2.predictor_score

print(f"\n{'=' * 70}")
print("EXPERIMENT D2 RESULTS")
print(f"{'=' * 70}")
print(f"T1 R@20: {r20:.3f} (baseline 0.218)")
print(f"T1 MRR:  {mrr_val:.3f} (baseline 0.530)")
print(f"T2 XRoom: {t2_cross:.3f} (baseline 0.185)")
print(f"Loss: {best_loss:.4f}")
print(f"Time: {train_time:.0f}s")
print(f"Change vs baseline: R@20 {(r20-0.218)/0.218*100:+.1f}%, "
      f"MRR {(mrr_val-0.530)/0.530*100:+.1f}%, "
      f"T2 {(t2_cross-0.185)/0.185*100:+.1f}%")

# Save
output_dir = Path("results") / "exp_d2_4layer_200k"
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "results.json", 'w') as f:
    json.dump({
        'T1_R@20': r20, 'T1_MRR': mrr_val,
        'T2_cross_room_R@20': t2_cross,
        'loss': float(best_loss), 'time': float(train_time),
        'parameters': pred.count_parameters(),
        'T1_details': t1.details,
        'T2_details': t2.details,
    }, f, indent=2, default=str)

print(f"\nTotal time: {time.time()-t_start:.1f}s")
