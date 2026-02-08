"""
Faithfulness evaluation with ablation controls.

Primary evaluation: train on ALL associations, measure faithfulness of recall.
Secondary: 70/30 generalisation stress test.
Ablation 1: temporal shuffle (permute temporal ordering, retrain).
Ablation 2: similarity-matched negatives (same-room non-associates as distractors).
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import copy

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

# Import 4-layer architecture
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from run_plateau_experiments import AssociativePredictor4Layer


def train_model(anchors, positives, seed=SEED):
    """Train a 4-layer predictor on given data."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    pred = AssociativePredictor4Layer(
        embedding_dim=128, hidden_dim=HIDDEN_DIM, seed=seed
    )
    print(f"  Parameters: {pred.count_parameters():,}")

    t0 = time.time()
    best_loss = train_predictor(
        pred, anchors, positives,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr_start=LR_START, lr_end=LR_END,
        temp_start=TEMP_START, temp_end=TEMP_END,
        print_every=100
    )
    elapsed = time.time() - t0
    print(f"  Training: {elapsed:.1f}s, Loss: {best_loss:.4f}")
    return pred, best_loss, elapsed


def temporal_shuffle_ablation(world):
    """
    Ablation 1: randomly permute temporal ordering within each trajectory,
    preserving embeddings. Recompute associations, retrain predictor.
    Performance should collapse -- proves model learned temporal structure.
    """
    print("\n" + "#" * 60)
    print("ABLATION: Temporal Shuffle")
    print("(Randomly permute time ordering within each trajectory)")
    print("#" * 60)

    # Deep copy the world and shuffle temporal ordering
    shuffled_world = copy.deepcopy(world)
    rng = np.random.RandomState(999)

    for traj in shuffled_world.trajectories:
        # Shuffle the states within this trajectory
        n = len(traj.states)
        perm = rng.permutation(n)
        original_states = [traj.states[i] for i in range(n)]
        for new_pos, old_pos in enumerate(perm):
            traj.states[new_pos] = original_states[old_pos]
            traj.states[new_pos].timestep = new_pos

    # Recompute associations on the shuffled world
    shuffled_world._associations = None
    shuffled_world.compute_association_ground_truth()

    # Generate training pairs from shuffled associations
    print("Generating 200k training pairs from shuffled associations...")
    anchors, positives, _ = shuffled_world.get_training_pairs(max_pairs=200000)

    # Train a fresh predictor
    pred_shuffled, loss_shuffled, time_shuffled = train_model(anchors, positives, seed=SEED)

    # Evaluate on the ORIGINAL world's associations
    # (does the shuffled-trained model recover the real temporal structure?)
    bilinear = LearnedBilinearBaseline(128, seed=123)
    bilinear.train(anchors, positives, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

    evaluator = BenchmarkEvaluator(world, pred_shuffled, bilinear)
    results = evaluator.run_faithfulness()
    results['loss'] = float(loss_shuffled)
    results['time'] = float(time_shuffled)

    print("\n--- Temporal Shuffle Ablation Complete ---")
    print("If faithfulness metrics collapse, model learned temporal structure,")
    print("not embedding artifacts.")

    return results


def similarity_matched_negatives_ablation(world, predictor):
    """
    Ablation 2: for each query, find same-room same-category states that
    were NEVER co-present. Measure whether predictor correctly ranks actual
    associates above these distractors. Report as discrimination AUC.
    """
    print("\n" + "#" * 60)
    print("ABLATION: Similarity-Matched Negatives")
    print("(Same-room non-associates as hard distractors)")
    print("#" * 60)

    associations = world._associations
    assoc_by_state = {}
    for (i, j), s in associations.items():
        assoc_by_state.setdefault(i, set()).add(j)
        assoc_by_state.setdefault(j, set()).add(i)

    # Group states by room
    room_states = {}
    for s in world.all_states:
        room_states.setdefault(s.room_id, set()).add(s.global_index)

    memory_bank = torch.from_numpy(world.all_embeddings).float().to(DEVICE)

    rng = np.random.RandomState(42)
    candidates = [idx for idx, nbrs in assoc_by_state.items() if len(nbrs) >= 5]
    rng.shuffle(candidates)
    test_queries = candidates[:300]

    pred_aucs = []
    cos_baseline_aucs = []

    cosine_baseline = torch.from_numpy(world.all_embeddings).float().to(DEVICE)
    cosine_baseline = F.normalize(cosine_baseline, dim=-1)

    for qi in test_queries:
        my_room = world.all_states[qi].room_id
        true_assoc = assoc_by_state[qi]

        # Same-room non-associates = hard distractors
        same_room = room_states[my_room] - true_assoc - {qi}
        if len(same_room) < 5:
            continue

        # True same-room associates
        same_room_assoc = true_assoc & room_states[my_room]
        if len(same_room_assoc) < 3:
            continue

        # Score all same-room states (associates + non-associates)
        eval_set = list(same_room_assoc | same_room)
        eval_labels = np.array([1.0 if idx in same_room_assoc else 0.0 for idx in eval_set])

        # Predictor scores
        query = memory_bank[qi]
        with torch.no_grad():
            predicted = F.normalize(predictor.predict(query.unsqueeze(0)), dim=-1)
            eval_embs = F.normalize(memory_bank[eval_set], dim=-1)
            pred_scores = (predicted @ eval_embs.T).squeeze(0).cpu().numpy()

        # Cosine scores
        q_norm = cosine_baseline[qi:qi+1]
        cos_scores = (q_norm @ cosine_baseline[eval_set].T).squeeze(0).cpu().numpy()

        # Compute AUC
        pos_mask = eval_labels == 1
        neg_mask = eval_labels == 0
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        # Predictor AUC
        pos_s = pred_scores[pos_mask]
        neg_s = pred_scores[neg_mask]
        count = sum(np.sum(ps > neg_s) + 0.5 * np.sum(ps == neg_s) for ps in pos_s)
        pred_aucs.append(float(count / (len(pos_s) * len(neg_s))))

        # Cosine AUC (within same room, cosine should do well)
        pos_s = cos_scores[pos_mask]
        neg_s = cos_scores[neg_mask]
        count = sum(np.sum(ps > neg_s) + 0.5 * np.sum(ps == neg_s) for ps in pos_s)
        cos_baseline_aucs.append(float(count / (len(pos_s) * len(neg_s))))

    mean_pred = float(np.mean(pred_aucs)) if pred_aucs else 0.0
    mean_cos = float(np.mean(cos_baseline_aucs)) if cos_baseline_aucs else 0.0

    print(f"\nSimilarity-Matched Discrimination AUC:")
    print(f"  Predictor: {mean_pred:.4f}  (n={len(pred_aucs)})")
    print(f"  Cosine:    {mean_cos:.4f}")
    print(f"\nNote: within same room, cosine has signal. Predictor should still")
    print(f"discriminate true temporal associates from mere room-mates.")

    return {
        'predictor_auc': mean_pred,
        'cosine_auc': mean_cos,
        'n_queries': len(pred_aucs),
    }


def main():
    t_total = time.time()

    print("=" * 70)
    print("FAITHFULNESS EVALUATION + ABLATION CONTROLS")
    print("Best config: 4-layer, 1024 hidden, 200k pairs, 500 epochs")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # =====================================================================
    # 1. Generate world and train on ALL associations
    # =====================================================================
    print("\n--- Generating World ---")
    config = WorldConfig(num_trajectories=NUM_TRAJECTORIES)
    world = SyntheticWorld(config)
    world.generate_trajectories()
    world.compute_association_ground_truth()
    print(f"States: {len(world.all_states)}, Associations: {len(world._associations)}")

    print("\n--- Generating 200k training pairs (ALL associations) ---")
    t0 = time.time()
    anchors, positives, _ = world.get_training_pairs(max_pairs=200000)
    print(f"Pair generation: {time.time()-t0:.1f}s")

    print("\n--- Training Primary Model ---")
    predictor, best_loss, train_time = train_model(anchors, positives)

    # Bilinear baseline
    bilinear = LearnedBilinearBaseline(128, seed=123)
    bilinear.train(anchors, positives, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

    # =====================================================================
    # 2. PRIMARY: Faithfulness metrics (train on ALL, evaluate on ALL)
    # =====================================================================
    evaluator = BenchmarkEvaluator(world, predictor, bilinear)
    faithfulness = evaluator.run_faithfulness()
    faithfulness['training'] = {
        'loss': float(best_loss),
        'time': float(train_time),
        'pairs': len(anchors),
        'parameters': predictor.count_parameters(),
    }

    # =====================================================================
    # 3. ABLATION 1: Temporal shuffle
    # =====================================================================
    shuffle_results = temporal_shuffle_ablation(world)

    # =====================================================================
    # 4. ABLATION 2: Similarity-matched negatives
    # =====================================================================
    sim_matched = similarity_matched_negatives_ablation(world, predictor)

    # =====================================================================
    # 5. SECONDARY: Generalisation stress test (70/30 split)
    # =====================================================================
    print("\n" + "#" * 60)
    print("SECONDARY: Generalisation Stress Test (70/30 split)")
    print("#" * 60)

    train_assoc, test_assoc = world.split_associations(train_ratio=0.7, seed=SEED)

    print("\nTraining on 70% split (200k pairs)...")
    anchors_split, positives_split, _ = world.get_training_pairs(
        max_pairs=200000, associations=train_assoc)
    pred_split, loss_split, time_split = train_model(anchors_split, positives_split, seed=99)

    bilinear_split = LearnedBilinearBaseline(128, seed=123)
    bilinear_split.train(anchors_split, positives_split, epochs=200,
                         batch_size=BATCH_SIZE, lr=1e-3)

    # Evaluate on held-out 30%
    eval_test = BenchmarkEvaluator(world, pred_split, bilinear_split,
                                    test_associations=test_assoc)
    t1_test = eval_test.test_association_vs_similarity()
    t2_test = eval_test.test_transitive_association()

    # Also evaluate on train associations
    eval_train = BenchmarkEvaluator(world, pred_split, bilinear_split,
                                     test_associations=train_assoc)
    t1_train = eval_train.test_association_vs_similarity()

    gen_stress = {
        'test': {
            'T1_R@20': float(t1_test.predictor_score),
            'T1_MRR': float(t1_test.details['mrr']['predictor']),
            'T2_cross_room_R@20': float(t2_test.predictor_score),
            'T1_cosine_R@20': float(t1_test.cosine_baseline_score),
            'details': t1_test.details,
        },
        'train': {
            'T1_R@20': float(t1_train.predictor_score),
            'T1_MRR': float(t1_train.details['mrr']['predictor']),
        },
        'gap': {
            'R@20': float(t1_train.predictor_score - t1_test.predictor_score),
            'MRR': float(t1_train.details['mrr']['predictor'] - t1_test.details['mrr']['predictor']),
        },
        'loss': float(loss_split),
        'time': float(time_split),
    }

    # =====================================================================
    # 6. FINAL SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    ap20 = faithfulness['association_precision'][20]['predictor']
    cbr20 = faithfulness['cross_boundary_recall']['recall_at_k'][20]['predictor']
    cbr20_cos = faithfulness['cross_boundary_recall']['recall_at_k'][20]['cosine']
    cb_mrr = faithfulness['cross_boundary_recall']['mrr']['predictor']
    auc_all = faithfulness['discrimination_auc']['all']['predictor']
    auc_cross = faithfulness['discrimination_auc']['cross_room']['predictor']
    spec = faithfulness['specificity']['predictor']

    shuf_cbr20 = shuffle_results['cross_boundary_recall']['recall_at_k'][20]['predictor']
    shuf_auc = shuffle_results['discrimination_auc']['cross_room']['predictor']

    print(f"\n--- Primary: Faithfulness (train=ALL, eval=ALL) ---")
    print(f"  Association Precision@20:    {ap20:.4f}")
    print(f"  Cross-Boundary Recall@20:    {cbr20:.4f}  (cosine: {cbr20_cos:.4f})")
    print(f"  Cross-Boundary MRR:          {cb_mrr:.4f}")
    print(f"  Discrimination AUC (all):    {auc_all:.4f}")
    print(f"  Discrimination AUC (x-room): {auc_cross:.4f}")
    print(f"  Specificity@20:              {spec:.4f}")

    print(f"\n--- Ablation: Temporal Shuffle ---")
    print(f"  Cross-Boundary Recall@20:    {shuf_cbr20:.4f}  (vs {cbr20:.4f} normal)")
    print(f"  Discrimination AUC (x-room): {shuf_auc:.4f}  (vs {auc_cross:.4f} normal)")
    cbr_collapse = (1 - shuf_cbr20 / max(cbr20, 1e-6)) * 100
    print(f"  CBR@20 collapse: {cbr_collapse:.1f}%")

    print(f"\n--- Ablation: Similarity-Matched Negatives ---")
    print(f"  Predictor AUC: {sim_matched['predictor_auc']:.4f}")
    print(f"  Cosine AUC:    {sim_matched['cosine_auc']:.4f}")

    print(f"\n--- Secondary: Generalisation Stress Test (70/30) ---")
    print(f"  Held-out R@20: {gen_stress['test']['T1_R@20']:.4f}  "
          f"(cosine: {gen_stress['test']['T1_cosine_R@20']:.4f})")
    print(f"  Held-out MRR:  {gen_stress['test']['T1_MRR']:.4f}")
    print(f"  Train R@20:    {gen_stress['train']['T1_R@20']:.4f}")
    print(f"  Gap: R@20={gen_stress['gap']['R@20']:+.4f}, MRR={gen_stress['gap']['MRR']:+.4f}")

    total_time = time.time() - t_total
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # =====================================================================
    # Save all results
    # =====================================================================
    output_dir = Path("results") / "faithfulness_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'primary_faithfulness': faithfulness,
        'ablation_temporal_shuffle': shuffle_results,
        'ablation_similarity_matched': sim_matched,
        'secondary_generalisation_stress_test': gen_stress,
        'config': {
            'architecture': '4-layer MLP (128->1024->1024->1024->128)',
            'hidden_dim': HIDDEN_DIM,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': f'{LR_START}->{LR_END}',
            'temp': f'{TEMP_START}->{TEMP_END}',
            'training_pairs': len(anchors),
            'total_associations': len(world._associations),
        },
        'total_time': total_time,
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
