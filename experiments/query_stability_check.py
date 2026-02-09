"""
Query sample stability check: resample query sets with 5 different seeds,
keeping the same trained model and world. Reports mean +/- SD for each metric.
"""
import time
import torch
import torch.nn.functional as F
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from world import SyntheticWorld, WorldConfig
from models_torch import (
    AssociativePredictor, LearnedBilinearBaseline,
    train_predictor, DEVICE
)
from evaluate_torch import (
    association_precision_at_k, discrimination_auc_single, recall_at_k, mrr
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_plateau_experiments import AssociativePredictor4Layer

BATCH_SIZE = 512
EPOCHS = 500
HIDDEN_DIM = 1024


def train_model(anchors, positives, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    pred = AssociativePredictor4Layer(embedding_dim=128, hidden_dim=HIDDEN_DIM, seed=seed)
    best_loss = train_predictor(
        pred, anchors, positives,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr_start=5e-4, lr_end=1e-5,
        temp_start=0.15, temp_end=0.05,
        print_every=100
    )
    print(f"  Loss: {best_loss:.4f}, Params: {pred.count_parameters():,}")
    return pred


def evaluate_with_query_seed(world, predictor, query_seed):
    """Evaluate CBR@20, AP@1, AP@20, Disc AUC with a specific query selection seed."""
    associations = world._associations
    assoc_by_state = {}
    for (i, j), s in associations.items():
        assoc_by_state.setdefault(i, set()).add(j)
        assoc_by_state.setdefault(j, set()).add(i)

    memory_bank = torch.from_numpy(world.all_embeddings).float().to(DEVICE)

    def pred_scores(qi):
        query = memory_bank[qi]
        return predictor.association_scores(query, memory_bank).cpu().numpy()

    rng = np.random.RandomState(query_seed)

    # --- AP@1 and AP@20: 500 queries with >=3 associates ---
    ap_candidates = [idx for idx, nbrs in assoc_by_state.items() if len(nbrs) >= 3]
    rng.shuffle(ap_candidates)
    ap_queries = ap_candidates[:500]

    ap1_vals, ap20_vals = [], []
    for qi in ap_queries:
        ta = assoc_by_state[qi]
        ps = pred_scores(qi)
        ap1_vals.append(association_precision_at_k(ps, ta, 1))
        ap20_vals.append(association_precision_at_k(ps, ta, 20))

    # --- CBR@20: 500 queries with >=3 cross-room associates ---
    cross_queries = []
    for idx, nbrs in assoc_by_state.items():
        my_room = world.all_states[idx].room_id
        cross = {n for n in nbrs if world.all_states[n].room_id != my_room}
        if len(cross) >= 3:
            cross_queries.append((idx, cross))
    rng2 = np.random.RandomState(query_seed)  # fresh RNG for this metric
    rng2.shuffle(cross_queries)

    cbr20_vals = []
    for idx, gt in cross_queries[:500]:
        ps = pred_scores(idx)
        cbr20_vals.append(recall_at_k(ps, gt, 20))

    # --- Discrimination AUC (all): 300 queries with >=5 associates ---
    auc_candidates = [idx for idx, nbrs in assoc_by_state.items() if len(nbrs) >= 5]
    rng3 = np.random.RandomState(query_seed)
    rng3.shuffle(auc_candidates)
    auc_queries = auc_candidates[:300]

    auc_vals = []
    for qi in auc_queries:
        ta = assoc_by_state[qi]
        ps = pred_scores(qi)
        auc_vals.append(discrimination_auc_single(ps, ta, qi))

    return {
        'AP@1': float(np.mean(ap1_vals)),
        'AP@20': float(np.mean(ap20_vals)),
        'CBR@20': float(np.mean(cbr20_vals)),
        'AUC': float(np.mean(auc_vals)),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("QUERY SAMPLE STABILITY CHECK")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Build world and train model (once)
    np.random.seed(42)
    config = WorldConfig(num_trajectories=500, seed=42)
    world = SyntheticWorld(config)
    world.generate_trajectories()
    world.compute_association_ground_truth()

    anchors, positives, _ = world.get_training_pairs(max_pairs=200000)
    print("Training model (seed=42)...")
    predictor = train_model(anchors, positives, seed=42)

    # Evaluate with 5 different query seeds
    query_seeds = [100, 101, 102, 103, 104]
    all_results = []

    for qs in query_seeds:
        print(f"\nEvaluating with query_seed={qs}...")
        t1 = time.time()
        res = evaluate_with_query_seed(world, predictor, qs)
        elapsed = time.time() - t1
        print(f"  AP@1={res['AP@1']:.4f}  AP@20={res['AP@20']:.4f}  "
              f"CBR@20={res['CBR@20']:.4f}  AUC={res['AUC']:.4f}  ({elapsed:.1f}s)")
        all_results.append(res)

    # Also run with default seed 42 for comparison
    print(f"\nEvaluating with query_seed=42 (default)...")
    t1 = time.time()
    res42 = evaluate_with_query_seed(world, predictor, 42)
    elapsed = time.time() - t1
    print(f"  AP@1={res42['AP@1']:.4f}  AP@20={res42['AP@20']:.4f}  "
          f"CBR@20={res42['CBR@20']:.4f}  AUC={res42['AUC']:.4f}  ({elapsed:.1f}s)")

    # Summary
    print(f"\n{'=' * 60}")
    print("QUERY STABILITY SUMMARY (seeds 100-104)")
    print(f"{'=' * 60}")
    metrics = ['AP@1', 'AP@20', 'CBR@20', 'AUC']
    print(f"{'Metric':<10s}", end="")
    for qs in query_seeds:
        print(f"  {qs:>7d}", end="")
    print(f"  {'Mean':>8s}  {'SD':>8s}  {'seed42':>8s}")
    print("-" * 80)

    for m in metrics:
        vals = [r[m] for r in all_results]
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1)
        print(f"{m:<10s}", end="")
        for v in vals:
            print(f"  {v:>7.4f}", end="")
        print(f"  {mean:>8.4f}  {sd:>8.4f}  {res42[m]:>8.4f}")

    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    main()
