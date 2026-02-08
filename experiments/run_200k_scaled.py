"""
200k states with proportionally scaled training pairs (400k).

Previous 200k experiment failed (R@20=0.019) because it used only 100k training pairs
for 200k states -- 50% coverage vs the baseline's 200% (100k pairs / 50k states).

Fix: scale training pairs to 400k (maintains 200% coverage ratio).
Also use 1024 hidden dim (proven best).
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
NUM_TRAJECTORIES = 2000  # -> 200k states
MAX_PAIRS = 400000  # 4x the default, proportional to states

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from world import SyntheticWorld, WorldConfig
from models_torch import AssociativePredictor, LearnedBilinearBaseline, train_predictor, DEVICE
from evaluate_torch import BenchmarkEvaluator

print("=" * 80)
print("200K STATES + 400K TRAINING PAIRS + 1024 HIDDEN")
print("=" * 80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Hypothesis: previous 200k failure was data starvation, not fundamental scaling")
print("=" * 80)

# --- Step 1: Generate world ---
print("\nSTEP 1: Generating World (200k states)")
t0 = time.time()
config = WorldConfig(num_trajectories=NUM_TRAJECTORIES)
world = SyntheticWorld(config)
world.generate_trajectories()
world.compute_association_ground_truth()
print(f"States: {len(world.all_states)}, Associations: {len(world._associations)}")
print(f"Time: {time.time()-t0:.1f}s")

# --- Step 2: Training data (400k pairs) ---
print(f"\nSTEP 2: Generating Training Data ({MAX_PAIRS // 1000}k pairs)")
t0 = time.time()
anchors, positives, _ = world.get_training_pairs(max_pairs=MAX_PAIRS)
print(f"Training pairs: {len(anchors)}")
print(f"Coverage ratio: {len(anchors) / len(world.all_states):.1f}x")
print(f"Time: {time.time()-t0:.1f}s")

# --- Step 3: Train predictor ---
print(f"\nSTEP 3: Training Predictor ({EPOCHS} epochs, {HIDDEN_DIM} hidden)")
predictor = AssociativePredictor(
    embedding_dim=config.embedding_dim,
    hidden_dim=HIDDEN_DIM,
    seed=SEED
)
print(f"Parameters: {predictor.count_parameters():,}")

t0 = time.time()
best_loss = train_predictor(
    predictor, anchors, positives,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr_start=LR_START,
    lr_end=LR_END,
    temp_start=TEMP_START,
    temp_end=TEMP_END,
    print_every=50
)
training_time = time.time() - t0
print(f"Training time: {training_time:.1f}s ({EPOCHS / (training_time / 60):.1f} epochs/min)")

# --- Step 4: Train bilinear baseline ---
print("\nSTEP 4: Training Bilinear Baseline")
bilinear = LearnedBilinearBaseline(config.embedding_dim, seed=123)
t0 = time.time()
bilinear.train(anchors, positives, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)
print(f"Time: {time.time()-t0:.1f}s")

# --- Step 5: Run ALL tests ---
print("\nSTEP 5: Full Evaluation (5 Tests)")
evaluator = BenchmarkEvaluator(world, predictor, bilinear)
all_results = evaluator.run_all()

# --- Save results ---
output_dir = Path("results") / "200k_scaled_400kpairs_1024h"
output_dir.mkdir(parents=True, exist_ok=True)

results_dict = {
    'name': '200k_scaled_400kpairs_1024h',
    'config': {
        'epochs': EPOCHS,
        'hidden_dim': HIDDEN_DIM,
        'num_trajectories': NUM_TRAJECTORIES,
        'num_states': len(world.all_states),
        'max_pairs': MAX_PAIRS,
        'actual_pairs': len(anchors),
        'coverage_ratio': len(anchors) / len(world.all_states),
        'batch_size': BATCH_SIZE,
        'lr_start': LR_START,
        'lr_end': LR_END,
        'temp_start': TEMP_START,
        'temp_end': TEMP_END,
    },
    'training': {
        'best_loss': float(best_loss),
        'training_time_sec': float(training_time),
        'epochs_per_min': float(EPOCHS / (training_time / 60)),
        'parameters': predictor.count_parameters(),
    },
    'results': {}
}

for r in all_results:
    results_dict['results'][r.test_name] = {
        'predictor_score': r.predictor_score,
        'cosine_baseline_score': r.cosine_baseline_score,
        'bilinear_baseline_score': r.bilinear_baseline_score,
        'details': r.details,
    }

output_file = output_dir / "results.json"
with open(output_file, 'w') as f:
    json.dump(results_dict, f, indent=2, default=str)

# --- Final summary ---
print("\n" + "=" * 80)
print("200K SCALED EXPERIMENT RESULTS")
print("=" * 80)
print(f"Config: {HIDDEN_DIM}h, {EPOCHS}ep, {len(anchors)//1000}k pairs, {len(world.all_states)//1000}k states")
print(f"Training loss: {best_loss:.4f}, Time: {training_time:.1f}s")
print()
for r in all_results:
    print(f"{r.test_name}:")
    print(f"  Predictor={r.predictor_score:.3f}  Cosine={r.cosine_baseline_score:.3f}  Bilinear={r.bilinear_baseline_score:.3f}")
print()
print("Comparison:")
print(f"  200k/100k pairs/512h (old):  R@20=0.019")
print(f"  50k/100k pairs/1024h:        R@20=0.218")
t1 = [r for r in all_results if r.test_name == 'association_vs_similarity'][0]
print(f"  200k/400k pairs/1024h (NEW): R@20={t1.predictor_score:.3f}")
print(f"\nResults saved to: {output_file}")
print("=" * 80)
