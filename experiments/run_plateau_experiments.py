"""
Plateau-Breaking Experiments for Tests 1-3.

Baseline: 1024h/500ep -> T1 R@20=0.218, MRR=0.530, T2 cross-room R@20=0.185

Experiment A: Online pair sampling (fresh pairs each epoch)
Experiment B: Deeper network (4 layers instead of 3)
Experiment C: Higher coverage (200k training pairs)
Experiment D: Combine best improvements from A-C
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from copy import deepcopy

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


# =============================================================================
# 4-Layer Predictor for Experiment B
# =============================================================================

class AssociativePredictor4Layer(nn.Module):
    """4-layer MLP predictor with residual connections."""

    def __init__(self, embedding_dim=128, hidden_dim=1024, seed=42):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.to(DEVICE)

    def forward(self, x):
        h1 = F.gelu(self.fc1(x))
        h2 = F.gelu(self.fc2(h1)) + h1  # residual
        h3 = F.gelu(self.fc3(h2)) + h2  # residual
        output = self.layer_norm(self.fc4(h3))
        return output

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def association_scores(self, query, memory_bank):
        if query.dim() == 1:
            query = query.unsqueeze(0)
        predicted = F.normalize(self.predict(query), dim=-1)
        mem_norm = F.normalize(memory_bank, dim=-1)
        scores = (predicted @ mem_norm.T).squeeze(0)
        return scores

    def multi_hop_retrieval(self, query, memory_bank, num_hops=3,
                           continuous=True, decay=0.7):
        if query.dim() == 1:
            query = query.unsqueeze(0)
        current = query
        mem_norm = F.normalize(memory_bank, dim=-1)
        aggregated = torch.zeros(memory_bank.shape[0], device=DEVICE)
        all_hops = []
        with torch.no_grad():
            for hop in range(num_hops):
                pred = F.normalize(self.predict(current), dim=-1)
                scores = (pred @ mem_norm.T).squeeze(0)
                aggregated += scores * (decay ** hop)
                all_hops.append(scores.cpu().numpy())
                if continuous:
                    current = self.predict(current)
                else:
                    idx = torch.argmax(scores).item()
                    current = memory_bank[idx:idx+1]
        return aggregated.cpu().numpy(), all_hops

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Online Pair Sampling Training
# =============================================================================

def train_predictor_online(predictor, world, epochs=500, batch_size=512,
                           lr_start=5e-4, lr_end=1e-5,
                           temp_start=0.15, temp_end=0.05,
                           pairs_per_epoch=100000, print_every=10):
    """
    Train with fresh pairs sampled each epoch.

    Instead of pre-generating fixed pairs, sample from the full association
    structure every epoch. This gives the model different views of the 242k
    associations each time.
    """
    associations = world._associations
    assoc_list = [(i, j, s) for (i, j), s in associations.items() if s >= 0.2]
    n_assoc = len(assoc_list)
    print(f"  Online sampling from {n_assoc} associations ({n_assoc*2} directed)")

    embeddings_t = torch.from_numpy(world.all_embeddings).float().to(DEVICE)
    N = len(world.all_states)

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr_start, weight_decay=1e-4)
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # Cosine annealing
        progress = (epoch - 1) / max(epochs - 1, 1)
        lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + np.cos(np.pi * progress))
        temp = temp_end + 0.5 * (temp_start - temp_end) * (1 + np.cos(np.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Sample fresh pairs this epoch
        n_pairs = min(pairs_per_epoch, n_assoc * 2)
        sampled_indices = np.random.choice(n_assoc, size=min(n_pairs, n_assoc), replace=False)

        anchor_indices = []
        positive_indices = []
        for idx in sampled_indices:
            i, j, s = assoc_list[idx]
            # Use both directions
            anchor_indices.append(i)
            positive_indices.append(j)
            if len(anchor_indices) < n_pairs:
                anchor_indices.append(j)
                positive_indices.append(i)

        anchor_indices = anchor_indices[:n_pairs]
        positive_indices = positive_indices[:n_pairs]

        anchors_t = embeddings_t[anchor_indices]
        positives_t = embeddings_t[positive_indices]

        # Training
        predictor.train()
        n = len(anchors_t)
        indices = torch.randperm(n, device=DEVICE)
        total_loss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bi = indices[start:end]

            a_batch = anchors_t[bi]
            p_batch = positives_t[bi]
            B = a_batch.shape[0]

            predicted = predictor(a_batch)
            pred_norm = F.normalize(predicted, dim=-1)
            pos_norm = F.normalize(p_batch, dim=-1)

            sim_matrix = (pred_norm @ pos_norm.T) / temp
            labels = torch.arange(B, device=DEVICE)
            loss = F.cross_entropy(sim_matrix, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % print_every == 0 or epoch == epochs:
            print(f"  Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}  "
                  f"Best: {best_loss:.4f}  LR: {lr:.2e}  Temp: {temp:.3f}")

    return best_loss


# =============================================================================
# Evaluation Helper
# =============================================================================

def evaluate_t1_t2(world, predictor, bilinear, label=""):
    """Run T1 and T2 and return key metrics."""
    evaluator = BenchmarkEvaluator(world, predictor, bilinear)

    t1 = evaluator.test_association_vs_similarity()
    t2 = evaluator.test_transitive_association()

    r20 = t1.predictor_score
    mrr_val = t1.details['mrr']['predictor']
    t2_cross = t2.predictor_score  # cross-room R@20

    print(f"\n  {label} Summary: T1 R@20={r20:.3f}, MRR={mrr_val:.3f}, "
          f"T2 cross-room R@20={t2_cross:.3f}")

    return {
        'T1_R@20': r20,
        'T1_MRR': mrr_val,
        'T2_cross_room_R@20': t2_cross,
        'T1_details': t1.details,
        'T2_details': t2.details,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("PLATEAU-BREAKING EXPERIMENTS")
    print("Baseline: T1 R@20=0.218, MRR=0.530, T2 cross-room=0.185")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    t_total = time.time()

    # --- Generate world (shared across all experiments) ---
    print("\nGenerating World...")
    config = WorldConfig(num_trajectories=NUM_TRAJECTORIES)
    world = SyntheticWorld(config)
    world.generate_trajectories()
    world.compute_association_ground_truth()
    print(f"States: {len(world.all_states)}, Associations: {len(world._associations)}")

    # Generate fixed pairs for B and C (A uses online sampling)
    print("\nGenerating fixed training pairs (100k for baseline/B)...")
    anchors_100k, positives_100k, _ = world.get_training_pairs(max_pairs=100000)

    all_results = {}

    # =========================================================================
    # Experiment A: Online Pair Sampling
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Online Pair Sampling")
    print("=" * 70)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    pred_a = AssociativePredictor(
        embedding_dim=config.embedding_dim,
        hidden_dim=HIDDEN_DIM,
        seed=SEED
    )
    print(f"Parameters: {pred_a.count_parameters():,}")

    t0 = time.time()
    loss_a = train_predictor_online(
        pred_a, world,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr_start=LR_START, lr_end=LR_END,
        temp_start=TEMP_START, temp_end=TEMP_END,
        pairs_per_epoch=100000,
        print_every=50
    )
    time_a = time.time() - t0
    print(f"Training time: {time_a:.1f}s, Loss: {loss_a:.4f}")

    bilinear_a = LearnedBilinearBaseline(config.embedding_dim, seed=123)
    bilinear_a.train(anchors_100k, positives_100k, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

    results_a = evaluate_t1_t2(world, pred_a, bilinear_a, "Exp A (Online)")
    results_a['loss'] = float(loss_a)
    results_a['time'] = float(time_a)
    all_results['exp_a_online'] = results_a

    # =========================================================================
    # Experiment B: Deeper Network (4 layers)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT B: 4-Layer Network")
    print("=" * 70)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    pred_b = AssociativePredictor4Layer(
        embedding_dim=config.embedding_dim,
        hidden_dim=HIDDEN_DIM,
        seed=SEED
    )
    print(f"Parameters: {pred_b.count_parameters():,}")

    t0 = time.time()
    loss_b = train_predictor(
        pred_b, anchors_100k, positives_100k,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr_start=LR_START, lr_end=LR_END,
        temp_start=TEMP_START, temp_end=TEMP_END,
        print_every=50
    )
    time_b = time.time() - t0
    print(f"Training time: {time_b:.1f}s, Loss: {loss_b:.4f}")

    bilinear_b = LearnedBilinearBaseline(config.embedding_dim, seed=123)
    bilinear_b.train(anchors_100k, positives_100k, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

    results_b = evaluate_t1_t2(world, pred_b, bilinear_b, "Exp B (4-Layer)")
    results_b['loss'] = float(loss_b)
    results_b['time'] = float(time_b)
    results_b['parameters'] = pred_b.count_parameters()
    all_results['exp_b_4layer'] = results_b

    # =========================================================================
    # Experiment C: Higher Coverage (200k pairs)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT C: 200k Training Pairs")
    print("=" * 70)

    print("Generating 200k training pairs...")
    t0 = time.time()
    anchors_200k, positives_200k, _ = world.get_training_pairs(max_pairs=200000)
    print(f"Pair generation time: {time.time()-t0:.1f}s")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    pred_c = AssociativePredictor(
        embedding_dim=config.embedding_dim,
        hidden_dim=HIDDEN_DIM,
        seed=SEED
    )
    print(f"Parameters: {pred_c.count_parameters():,}")

    t0 = time.time()
    loss_c = train_predictor(
        pred_c, anchors_200k, positives_200k,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr_start=LR_START, lr_end=LR_END,
        temp_start=TEMP_START, temp_end=TEMP_END,
        print_every=50
    )
    time_c = time.time() - t0
    print(f"Training time: {time_c:.1f}s, Loss: {loss_c:.4f}")

    bilinear_c = LearnedBilinearBaseline(config.embedding_dim, seed=123)
    bilinear_c.train(anchors_200k, positives_200k, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

    results_c = evaluate_t1_t2(world, pred_c, bilinear_c, "Exp C (200k pairs)")
    results_c['loss'] = float(loss_c)
    results_c['time'] = float(time_c)
    all_results['exp_c_200k'] = results_c

    # =========================================================================
    # Experiment D: Combine Best
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Combined Best")
    print("=" * 70)

    # Determine best approach from A-C
    best_r20 = 0
    best_exp = None
    for name, r in all_results.items():
        if r['T1_R@20'] > best_r20:
            best_r20 = r['T1_R@20']
            best_exp = name

    print(f"Best single experiment: {best_exp} with R@20={best_r20:.3f}")

    # Combine: use best architecture + online sampling + 200k pairs/epoch
    use_4layer = all_results.get('exp_b_4layer', {}).get('T1_R@20', 0) > \
                 all_results.get('exp_a_online', {}).get('T1_R@20', 0)

    print(f"Using {'4-layer' if use_4layer else '3-layer'} + online sampling")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    if use_4layer:
        pred_d = AssociativePredictor4Layer(
            embedding_dim=config.embedding_dim,
            hidden_dim=HIDDEN_DIM,
            seed=SEED
        )
    else:
        pred_d = AssociativePredictor(
            embedding_dim=config.embedding_dim,
            hidden_dim=HIDDEN_DIM,
            seed=SEED
        )
    print(f"Parameters: {pred_d.count_parameters():,}")

    t0 = time.time()
    loss_d = train_predictor_online(
        pred_d, world,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr_start=LR_START, lr_end=LR_END,
        temp_start=TEMP_START, temp_end=TEMP_END,
        pairs_per_epoch=200000,
        print_every=50
    )
    time_d = time.time() - t0
    print(f"Training time: {time_d:.1f}s, Loss: {loss_d:.4f}")

    bilinear_d = LearnedBilinearBaseline(config.embedding_dim, seed=123)
    bilinear_d.train(anchors_100k, positives_100k, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

    results_d = evaluate_t1_t2(world, pred_d, bilinear_d, "Exp D (Combined)")
    results_d['loss'] = float(loss_d)
    results_d['time'] = float(time_d)
    results_d['architecture'] = '4-layer' if use_4layer else '3-layer'
    results_d['parameters'] = pred_d.count_parameters()
    all_results['exp_d_combined'] = results_d

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PLATEAU-BREAKING EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"{'Experiment':<25s} {'T1 R@20':>8s} {'T1 MRR':>8s} {'T2 XRoom':>8s} {'Loss':>8s} {'Time':>6s}")
    print("-" * 70)
    print(f"{'Baseline (reference)':<25s} {'0.218':>8s} {'0.530':>8s} {'0.185':>8s} {'0.287':>8s} {'147s':>6s}")

    for name, r in all_results.items():
        label = {
            'exp_a_online': 'A: Online sampling',
            'exp_b_4layer': 'B: 4-layer network',
            'exp_c_200k': 'C: 200k pairs',
            'exp_d_combined': 'D: Combined best',
        }.get(name, name)
        print(f"{label:<25s} {r['T1_R@20']:>8.3f} {r['T1_MRR']:>8.3f} "
              f"{r['T2_cross_room_R@20']:>8.3f} {r['loss']:>8.3f} {r['time']:>5.0f}s")

    # Best result
    best_name = max(all_results, key=lambda k: all_results[k]['T1_R@20'])
    best = all_results[best_name]
    improvement = (best['T1_R@20'] - 0.218) / 0.218 * 100
    print(f"\nBest: {best_name} -> T1 R@20={best['T1_R@20']:.3f} ({improvement:+.1f}% vs baseline)")

    # --- Save results ---
    output_dir = Path("results") / "plateau_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'baseline': {'T1_R@20': 0.218, 'T1_MRR': 0.530, 'T2_cross_room_R@20': 0.185},
        'experiments': all_results,
        'config': {
            'epochs': EPOCHS,
            'hidden_dim': HIDDEN_DIM,
            'batch_size': BATCH_SIZE,
        },
        'total_time_sec': time.time() - t_total,
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir / 'results.json'}")
    print(f"Total time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
