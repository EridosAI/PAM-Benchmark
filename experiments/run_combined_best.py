"""
Combined best config: 1024 hidden + 2000 epochs

Previous results:
- Baseline (500ep/512h):  R@20=0.089, loss=1.613
- 2000 epochs (512h):     R@20=0.107, loss=1.352  (+20%)
- 1024 hidden (500ep):    R@20=0.218, loss=0.287  (+145%)
- 200k states:            R@20=0.019, loss=1.631  (-79%)

Expected: combining 1024 hidden + 2000 epochs should push well past 0.218.
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
print("COMBINED BEST CONFIG: 1024 hidden + 2000 epochs")
print("=" * 80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 80)

# Config
EPOCHS = 2000
HIDDEN_DIM = 1024
NUM_TRAJECTORIES = 500

# Generate world
print("\nSTEP 1: Generating World")
t0 = time.time()
config = WorldConfig(num_trajectories=NUM_TRAJECTORIES)
world = SyntheticWorld(config)
world.generate_trajectories()
world.compute_association_ground_truth()
print(f"States: {len(world.all_states)}, Associations: {len(world._associations)}")
print(f"Time: {time.time()-t0:.1f}s")

# Training data
print("\nSTEP 2: Generating Training Data")
t0 = time.time()
anchors, positives, _ = world.get_training_pairs()
print(f"Training pairs: {len(anchors)}")
print(f"Time: {time.time()-t0:.1f}s")

# Train predictor
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
    print_every=40
)
training_time = time.time() - t0
print(f"Training time: {training_time:.1f}s ({EPOCHS / (training_time / 60):.1f} epochs/min)")

# Train bilinear
print("\nSTEP 4: Training Bilinear Baseline")
bilinear = LearnedBilinearBaseline(config.embedding_dim, seed=123)
t0 = time.time()
bilinear.train(anchors, positives, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)
print(f"Time: {time.time()-t0:.1f}s")

# Evaluate
print("\nSTEP 5: Evaluation")
evaluator = BenchmarkEvaluator(world, predictor, bilinear)
result = evaluator.test_association_vs_similarity()

# Save results
output_dir = Path("results") / "combined_best_1024h_2000ep"
output_dir.mkdir(parents=True, exist_ok=True)

results_dict = {
    'name': 'combined_best_1024h_2000ep',
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
    'results': {
        'test1_r20_predictor': result.predictor_score,
        'test1_r20_cosine': result.cosine_baseline_score,
        'test1_r20_bilinear': result.bilinear_baseline_score,
        'details': result.details,
    }
}

output_file = output_dir / "results.json"
with open(output_file, 'w') as f:
    json.dump(results_dict, f, indent=2)

# Final summary
print("\n" + "=" * 80)
print("COMBINED BEST CONFIG RESULTS")
print("=" * 80)
print(f"R@20 Predictor: {result.predictor_score:.3f}")
print(f"R@5:  {result.details['recall_at_k'][5]['predictor']:.3f}")
print(f"R@10: {result.details['recall_at_k'][10]['predictor']:.3f}")
print(f"R@50: {result.details['recall_at_k'][50]['predictor']:.3f}")
print(f"MRR:  {result.details['mrr']['predictor']:.3f}")
print(f"Training loss: {best_loss:.4f}")
print(f"Training time: {training_time:.1f}s")
print()
print("Comparison:")
print(f"  Baseline (500ep/512h):  R@20=0.089")
print(f"  2000ep/512h:            R@20=0.107  (+20%)")
print(f"  500ep/1024h:            R@20=0.218  (+145%)")
print(f"  2000ep/1024h (THIS):    R@20={result.predictor_score:.3f}  ({(result.predictor_score - 0.089) / 0.089 * 100:+.1f}%)")
print(f"\nResults saved to: {output_file}")
print("=" * 80)
