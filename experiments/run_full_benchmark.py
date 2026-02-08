"""
Full benchmark: 1024 hidden, 500 epochs, all 5 tests.

Best config from scale-up experiments (500ep/1024h gave R@20=0.218).
Now running all 5 tests for a complete picture.
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
from models_torch import AssociativePredictor, LearnedBilinearBaseline, train_predictor, DEVICE
from evaluate_torch import BenchmarkEvaluator

print("=" * 80)
print("FULL BENCHMARK: 1024 hidden, 500 epochs, ALL 5 TESTS")
print("=" * 80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 80)

# --- Step 1: Generate world ---
print("\nSTEP 1: Generating World")
t0 = time.time()
config = WorldConfig(num_trajectories=NUM_TRAJECTORIES)
world = SyntheticWorld(config)
world.generate_trajectories()
world.compute_association_ground_truth()
print(f"States: {len(world.all_states)}, Associations: {len(world._associations)}")
print(f"Time: {time.time()-t0:.1f}s")

# --- Step 2: Training data ---
print("\nSTEP 2: Generating Training Data")
t0 = time.time()
anchors, positives, _ = world.get_training_pairs()
print(f"Training pairs: {len(anchors)}")
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
output_dir = Path("results") / "full_benchmark_1024h_500ep"
output_dir.mkdir(parents=True, exist_ok=True)

results_dict = {
    'name': 'full_benchmark_1024h_500ep',
    'config': {
        'epochs': EPOCHS,
        'hidden_dim': HIDDEN_DIM,
        'num_trajectories': NUM_TRAJECTORIES,
        'num_states': len(world.all_states),
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
print("FULL BENCHMARK RESULTS")
print("=" * 80)
print(f"Config: {HIDDEN_DIM} hidden, {EPOCHS} epochs, {predictor.count_parameters():,} params")
print(f"Training loss: {best_loss:.4f}, Time: {training_time:.1f}s")
print()
for r in all_results:
    print(f"{r.test_name}:")
    print(f"  Predictor={r.predictor_score:.3f}  Cosine={r.cosine_baseline_score:.3f}  Bilinear={r.bilinear_baseline_score:.3f}")
print(f"\nResults saved to: {output_file}")
print("=" * 80)
