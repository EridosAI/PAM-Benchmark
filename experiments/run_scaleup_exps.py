"""
Scale-up experiments with PyTorch + GPU

Baseline: exp1_500epochs (R@20=0.089, 500 epochs, 512 hidden, 50k states)

Experiments:
1. 2000 epochs (same config) - test if more training helps
2. 1024 hidden dim - test if larger model helps
3. 200k states - test if more data helps

Run sequentially to see what helps most.
"""

import time
import torch
import numpy as np
from pathlib import Path
import json
import sys

# Baseline config
BASE_EPOCHS = 500
BASE_HIDDEN = 512
BASE_STATES = 50000  # trajectories=500, steps=100

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

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from world import SyntheticWorld, WorldConfig
from models_torch import AssociativePredictor, LearnedBilinearBaseline, train_predictor
from evaluate_torch import BenchmarkEvaluator


def run_experiment(name, epochs, hidden_dim, num_trajectories=500):
    """Run a single scale-up experiment."""
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {name}")
    print("=" * 80)
    print(f"Config: epochs={epochs}, hidden={hidden_dim}, trajectories={num_trajectories}")
    print(f"        states={num_trajectories * 100}")
    print("=" * 80 + "\n")

    # Generate world
    print("STEP 1: Generating World")
    t0 = time.time()
    config = WorldConfig(num_trajectories=num_trajectories)
    world = SyntheticWorld(config)
    world.generate_trajectories()
    world.compute_association_ground_truth()
    print(f"States: {len(world.all_states)}, Associations: {len(world._associations)}")
    print(f"Time: {time.time()-t0:.1f}s\n")

    # Training data
    print("STEP 2: Generating Training Data")
    t0 = time.time()
    anchors, positives, _ = world.get_training_pairs()
    print(f"Training pairs: {len(anchors)}")
    print(f"Time: {time.time()-t0:.1f}s\n")

    # Train predictor
    print(f"STEP 3: Training Predictor ({epochs} epochs, {hidden_dim} hidden)")
    predictor = AssociativePredictor(
        embedding_dim=config.embedding_dim,
        hidden_dim=hidden_dim,
        seed=SEED
    )
    print(f"Parameters: {predictor.count_parameters():,}")

    t0 = time.time()
    best_loss = train_predictor(
        predictor, anchors, positives,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        lr_start=LR_START,
        lr_end=LR_END,
        temp_start=TEMP_START,
        temp_end=TEMP_END,
        print_every=max(10, epochs // 50)
    )
    training_time = time.time() - t0
    print(f"Training time: {training_time:.1f}s ({epochs / (training_time / 60):.1f} epochs/min)\n")

    # Train bilinear
    print("STEP 4: Training Bilinear Baseline")
    bilinear = LearnedBilinearBaseline(config.embedding_dim, seed=123)
    t0 = time.time()
    bilinear.train(anchors, positives, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)
    print(f"Time: {time.time()-t0:.1f}s\n")

    # Evaluate
    print("STEP 5: Evaluation")
    evaluator = BenchmarkEvaluator(world, predictor, bilinear)
    result = evaluator.test_association_vs_similarity()

    # Save results
    output_dir = Path("results") / f"scaleup_{name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {
        'name': name,
        'config': {
            'epochs': epochs,
            'hidden_dim': hidden_dim,
            'num_trajectories': num_trajectories,
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
            'epochs_per_min': float(epochs / (training_time / 60)),
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

    print("\n" + "=" * 80)
    print(f"RESULTS: {name}")
    print("=" * 80)
    print(f"R@20 Predictor: {result.predictor_score:.3f}")
    print(f"Training loss: {best_loss:.4f}")
    print(f"Training time: {training_time:.1f}s")
    print(f"Results saved to: {output_file}")
    print("=" * 80)

    return results_dict


# Run experiments sequentially
print("\n" + "=" * 80)
print("SCALE-UP EXPERIMENTS - PyTorch + GPU")
print("=" * 80)
print(f"Baseline: exp1_500epochs (R@20=0.089, {BASE_EPOCHS} epochs, {BASE_HIDDEN} hidden, {BASE_STATES} states)")
print("=" * 80)

all_results = []

# Exp 1: 2000 epochs
exp1 = run_experiment("exp1_2000epochs", epochs=2000, hidden_dim=512)
all_results.append(exp1)

# Exp 2: 1024 hidden
exp2 = run_experiment("exp2_1024hidden", epochs=500, hidden_dim=1024)
all_results.append(exp2)

# Exp 3: 200k states (2000 trajectories)
exp3 = run_experiment("exp3_200kstates", epochs=500, hidden_dim=512, num_trajectories=2000)
all_results.append(exp3)

# Summary
print("\n\n" + "=" * 80)
print("SCALE-UP SUMMARY")
print("=" * 80)
print(f"{'Experiment':<25} {'R@20':<10} {'Loss':<10} {'Time(s)':<12} {'Improvement'}")
print("-" * 80)
print(f"{'Baseline (500ep/512h)':<25} {'0.089':<10} {'1.613':<10} {'149':<12} {'-'}")
for r in all_results:
    imp = ((r['results']['test1_r20_predictor'] - 0.089) / 0.089 * 100)
    print(f"{r['name']:<25} {r['results']['test1_r20_predictor']:<10.3f} "
          f"{r['training']['best_loss']:<10.4f} {r['training']['training_time_sec']:<12.1f} "
          f"{imp:+.1f}%")

print("\nTo compare: python analyze_scaleup.py")
