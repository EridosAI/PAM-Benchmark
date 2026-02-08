"""
Test 4 Deep Investigation: Bridge Solvability & Dendritic Spreading Activation

Diagnoses why creative bridging gets R@50=0.005, then tests improved approaches.

Key question: Is the task solvable at all? The predictor learns within-trajectory
temporal associations. Bridges require cross-trajectory retrieval via shared objects.
The predictor was never trained on cross-trajectory pairs, so hop2 from a bridge
state B (in trajectory t1) has no reason to retrieve target C (in trajectory t2).

Approaches tested:
1. Pure predictor spreading (current approach, baseline)
2. Hybrid hop2: alpha*predictor + (1-alpha)*cosine at second hop
3. Cosine-only hop2: predictor for hop1, cosine for hop2
4. Full cosine spreading: cosine for both hops
5. Dendritic convergence: unweighted path counting
"""

import time
import torch
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
    train_predictor, CosineSimilarityBaseline, DEVICE
)
from evaluate_torch import BenchmarkEvaluator, recall_at_k


def diagnose_bridges(world, bridges, assoc_by_state):
    """Deep analysis of bridge structure and solvability."""
    print("\n" + "=" * 70)
    print("BRIDGE SOLVABILITY DIAGNOSTICS")
    print("=" * 70)

    n_test = min(len(bridges), 300)

    # --- 1. Shared object analysis ---
    print("\n--- 1. Shared Object Analysis ---")
    obj_per_bridge = []
    embedding_sims = []
    room_same = 0

    for i in range(n_test):
        s1, s2, obj_id, t1, t2 = bridges[i]
        st1 = world.all_states[s1]
        st2 = world.all_states[s2]

        # How many objects do s1 and s2 share?
        shared = set(st1.object_ids) & set(st2.object_ids)
        obj_per_bridge.append(len(shared))

        # Embedding similarity
        e1 = world.all_embeddings[s1]
        e2 = world.all_embeddings[s2]
        sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
        embedding_sims.append(sim)

        if st1.room_id == st2.room_id:
            room_same += 1

    print(f"  Bridges tested: {n_test}")
    print(f"  Shared objects per bridge: mean={np.mean(obj_per_bridge):.2f}, "
          f"min={np.min(obj_per_bridge)}, max={np.max(obj_per_bridge)}")
    print(f"  Same room: {room_same}/{n_test} ({100*room_same/n_test:.1f}%)")
    print(f"  Embedding similarity (s1,s2): mean={np.mean(embedding_sims):.3f}, "
          f"std={np.std(embedding_sims):.3f}")

    # --- 2. Bridge state density ---
    print("\n--- 2. Bridge State Density ---")
    # For each bridge, how many states in t1 share the bridge object?
    # And how many states in t2 share it?
    t1_bridge_counts = []
    t2_bridge_counts = []
    bridge_state_sims_to_target = []

    for i in range(n_test):
        s1, s2, obj_id, t1, t2 = bridges[i]

        t1_with_obj = [s.global_index for s in world.trajectories[t1].states
                       if obj_id in s.object_ids]
        t2_with_obj = [s.global_index for s in world.trajectories[t2].states
                       if obj_id in s.object_ids]

        t1_bridge_counts.append(len(t1_with_obj))
        t2_bridge_counts.append(len(t2_with_obj))

        # Cosine similarity between bridge states in t1 and target s2
        for b_idx in t1_with_obj:
            eb = world.all_embeddings[b_idx]
            e2 = world.all_embeddings[s2]
            sim = np.dot(eb, e2) / (np.linalg.norm(eb) * np.linalg.norm(e2) + 1e-8)
            bridge_state_sims_to_target.append(sim)

    print(f"  States in t1 with bridge object: mean={np.mean(t1_bridge_counts):.1f}, "
          f"median={np.median(t1_bridge_counts):.0f}")
    print(f"  States in t2 with bridge object: mean={np.mean(t2_bridge_counts):.1f}, "
          f"median={np.median(t2_bridge_counts):.0f}")
    print(f"  Cosine sim (bridge_state_in_t1 -> target_s2): "
          f"mean={np.mean(bridge_state_sims_to_target):.3f}, "
          f"std={np.std(bridge_state_sims_to_target):.3f}")

    # --- 3. Are bridge states temporally near s1? ---
    print("\n--- 3. Temporal Proximity of Bridge States to Query ---")
    temporal_dists = []
    bridge_is_associated = []

    for i in range(n_test):
        s1, s2, obj_id, t1, t2 = bridges[i]
        st1 = world.all_states[s1]

        t1_with_obj = [s for s in world.trajectories[t1].states
                       if obj_id in s.object_ids]

        for b_state in t1_with_obj:
            temporal_dists.append(abs(b_state.timestep - st1.timestep))
            # Is this bridge state associated with s1?
            key = (min(s1, b_state.global_index), max(s1, b_state.global_index))
            is_assoc = key in world._associations
            bridge_is_associated.append(is_assoc)

    print(f"  Temporal distance (query s1 -> bridge states in t1): "
          f"mean={np.mean(temporal_dists):.1f}, median={np.median(temporal_dists):.0f}")
    print(f"  Bridge states associated with query: "
          f"{sum(bridge_is_associated)}/{len(bridge_is_associated)} "
          f"({100*np.mean(bridge_is_associated):.1f}%)")

    # --- 4. Is the task FUNDAMENTALLY solvable? ---
    print("\n--- 4. Oracle Analysis ---")
    # If we had a perfect oracle that:
    # a) Finds ALL bridge states in t1 (states with shared object)
    # b) Then finds ALL states in t2 associated with any bridge state in t2
    # Can we reach s2?

    oracle_reachable = 0
    oracle_via_cosine = 0
    N = len(world.all_states)

    for i in range(min(n_test, 100)):  # Limit for speed
        s1, s2, obj_id, t1, t2 = bridges[i]

        # Oracle hop 1: all states in t1 with shared object
        t1_bridges = {s.global_index for s in world.trajectories[t1].states
                      if obj_id in s.object_ids}

        # Oracle hop 2: all states associated with any bridge state
        reachable = set()
        for b in t1_bridges:
            if b in assoc_by_state:
                reachable |= assoc_by_state[b]

        if s2 in reachable:
            oracle_reachable += 1

        # Cosine oracle: from bridge states, find states most similar
        # (this tests if embedding similarity can bridge the gap)
        if t1_bridges:
            bridge_embs = world.all_embeddings[list(t1_bridges)]
            target_emb = world.all_embeddings[s2]
            sims = bridge_embs @ target_emb / (
                np.linalg.norm(bridge_embs, axis=1) * np.linalg.norm(target_emb) + 1e-8
            )
            # Is s2 in top-50 most similar to ANY bridge state?
            all_sims = world.all_embeddings @ target_emb / (
                np.linalg.norm(world.all_embeddings, axis=1) * np.linalg.norm(target_emb) + 1e-8
            )
            rank_of_s2 = np.sum(all_sims > all_sims[s2]) + 1
            # Can any bridge state rank s2 in top-50 by cosine?
            best_bridge_rank = N
            for b in t1_bridges:
                b_emb = world.all_embeddings[b]
                b_all_sims = world.all_embeddings @ b_emb / (
                    np.linalg.norm(world.all_embeddings, axis=1) * np.linalg.norm(b_emb) + 1e-8
                )
                rank = np.sum(b_all_sims > b_all_sims[s2]) + 1
                best_bridge_rank = min(best_bridge_rank, rank)
            if best_bridge_rank <= 50:
                oracle_via_cosine += 1

    tested = min(n_test, 100)
    print(f"  Oracle (association-hop from bridge states) reaches target: "
          f"{oracle_reachable}/{tested} ({100*oracle_reachable/tested:.1f}%)")
    print(f"  Oracle (cosine from bridge states, top-50) reaches target: "
          f"{oracle_via_cosine}/{tested} ({100*oracle_via_cosine/tested:.1f}%)")

    return {
        'n_bridges': n_test,
        'shared_objects_mean': float(np.mean(obj_per_bridge)),
        'embedding_sim_mean': float(np.mean(embedding_sims)),
        'same_room_pct': 100 * room_same / n_test,
        't1_bridge_count_mean': float(np.mean(t1_bridge_counts)),
        'temporal_dist_mean': float(np.mean(temporal_dists)),
        'bridge_associated_pct': 100 * np.mean(bridge_is_associated),
        'oracle_assoc_reachable_pct': 100 * oracle_reachable / tested,
        'oracle_cosine_reachable_pct': 100 * oracle_via_cosine / tested,
    }


def test4_approaches(evaluator, world, bridges, assoc_by_state, k_values=[20, 50, 100]):
    """Test multiple spreading activation approaches."""
    print("\n" + "=" * 70)
    print("TEST 4: MULTI-APPROACH COMPARISON")
    print("=" * 70)

    n_test = min(len(bridges), 300)
    N = len(world.all_states)
    memory_bank = evaluator.memory_bank
    predictor = evaluator.predictor

    fanout_values = [50, 200, 500]

    all_approach_results = {}

    for fanout_k in fanout_values:
        print(f"\n--- Fanout K={fanout_k} ---")

        results = {k: {} for k in k_values}
        approaches = [
            'pred_spreading',       # Current: predictor hop1, predictor hop2, weighted
            'pred_unweighted',      # Predictor hop1+hop2 but unweighted (convergence count)
            'pred_softmax',         # Softmax weights instead of raw scores
            'hybrid_hop2_0.3',      # 0.3*predictor + 0.7*cosine at hop2
            'hybrid_hop2_0.5',      # 0.5*predictor + 0.5*cosine at hop2
            'cosine_hop2',          # Predictor hop1, cosine hop2
            'full_cosine',          # Cosine both hops
            'cosine_baseline',      # Direct cosine (no spreading)
        ]
        for k in k_values:
            for a in approaches:
                results[k][a] = []

        # Precompute cosine similarity matrix (N x N is too large, use batched)
        mem_norm = F.normalize(memory_bank, dim=-1)

        for i in range(n_test):
            s1, s2, obj_id, t1, t2 = bridges[i]

            # Build relaxed ground truth
            s1_neighbors = assoc_by_state.get(s1, set())
            s2_neighbors = assoc_by_state.get(s2, set())
            t1_bridge_states = {
                s.global_index for s in world.trajectories[t1].states
                if obj_id in s.object_ids
            }
            t2_bridge_states = {
                s.global_index for s in world.trajectories[t2].states
                if obj_id in s.object_ids
            }
            intermediates = (s1_neighbors & t1_bridge_states) | (s2_neighbors & t2_bridge_states)
            gt_relaxed = {s2}
            for interm in intermediates:
                gt_relaxed |= (assoc_by_state.get(interm, set()) & s2_neighbors)

            # --- Hop 1: predictor scores ---
            query = memory_bank[s1]
            with torch.no_grad():
                pred_out = F.normalize(predictor.predict(query.unsqueeze(0)), dim=-1)
                hop1_pred_scores = (pred_out @ mem_norm.T).squeeze(0)  # [N]

            # --- Hop 1: cosine scores ---
            query_norm = F.normalize(query.unsqueeze(0), dim=-1)
            hop1_cos_scores = (query_norm @ mem_norm.T).squeeze(0)  # [N]

            # --- Get top-K for predictor hop1 ---
            top_k_pred = torch.topk(hop1_pred_scores, fanout_k).indices  # [K]
            top_k_pred_scores = hop1_pred_scores[top_k_pred]  # [K]

            # --- Get top-K for cosine hop1 ---
            top_k_cos = torch.topk(hop1_cos_scores, fanout_k).indices
            top_k_cos_scores = hop1_cos_scores[top_k_cos]

            # --- Hop 2 from predictor top-K: predictor scores ---
            with torch.no_grad():
                intermediates_emb = memory_bank[top_k_pred]  # [K, D]
                hop2_pred = F.normalize(predictor.predict(intermediates_emb), dim=-1)
                hop2_pred_scores = (hop2_pred @ mem_norm.T)  # [K, N]

            # --- Hop 2 from predictor top-K: cosine scores ---
            intermediates_norm = F.normalize(intermediates_emb, dim=-1)
            hop2_cos_scores = (intermediates_norm @ mem_norm.T)  # [K, N]

            # --- Hop 2 from cosine top-K: cosine scores ---
            cos_intermediates_emb = memory_bank[top_k_cos]
            cos_intermediates_norm = F.normalize(cos_intermediates_emb, dim=-1)
            hop2_cos_from_cos = (cos_intermediates_norm @ mem_norm.T)  # [K, N]

            # Move to numpy for scoring
            h1_pred_w = top_k_pred_scores.cpu().numpy()
            h2_pred = hop2_pred_scores.cpu().numpy()  # [K, N]
            h2_cos = hop2_cos_scores.cpu().numpy()  # [K, N]
            h1_cos_w = top_k_cos_scores.cpu().numpy()
            h2_cos_cos = hop2_cos_from_cos.cpu().numpy()

            # === Approach 1: Current weighted predictor spreading ===
            scores_pred_spread = h1_pred_w @ h2_pred  # [N]

            # === Approach 2: Unweighted convergence (pure path counting) ===
            scores_pred_unweighted = np.ones(fanout_k) @ h2_pred  # [N]

            # === Approach 3: Softmax weights ===
            softmax_w = np.exp(h1_pred_w * 10)  # temperature=0.1
            softmax_w = softmax_w / softmax_w.sum()
            scores_pred_softmax = softmax_w @ h2_pred

            # === Approach 4: Hybrid hop2 (0.3 pred + 0.7 cosine) ===
            h2_hybrid_03 = 0.3 * h2_pred + 0.7 * h2_cos
            scores_hybrid_03 = h1_pred_w @ h2_hybrid_03

            # === Approach 5: Hybrid hop2 (0.5 pred + 0.5 cosine) ===
            h2_hybrid_05 = 0.5 * h2_pred + 0.5 * h2_cos
            scores_hybrid_05 = h1_pred_w @ h2_hybrid_05

            # === Approach 6: Cosine hop2 only ===
            scores_cos_hop2 = h1_pred_w @ h2_cos

            # === Approach 7: Full cosine spreading ===
            scores_full_cos = h1_cos_w @ h2_cos_cos

            # === Approach 8: Direct cosine baseline ===
            scores_cos_direct = hop1_cos_scores.cpu().numpy()

            approach_scores = {
                'pred_spreading': scores_pred_spread,
                'pred_unweighted': scores_pred_unweighted,
                'pred_softmax': scores_pred_softmax,
                'hybrid_hop2_0.3': scores_hybrid_03,
                'hybrid_hop2_0.5': scores_hybrid_05,
                'cosine_hop2': scores_cos_hop2,
                'full_cosine': scores_full_cos,
                'cosine_baseline': scores_cos_direct,
            }

            for k in k_values:
                for a_name, a_scores in approach_scores.items():
                    results[k][a_name].append(recall_at_k(a_scores, gt_relaxed, k))

            if (i + 1) % 100 == 0:
                print(f"  processed {i+1}/{n_test}")

        # Print results for this fanout
        print(f"\n  Results (K={fanout_k}):")
        approach_results = {}
        for k in k_values:
            print(f"  R@{k}:")
            for a in approaches:
                mean_r = float(np.mean(results[k][a]))
                print(f"    {a:25s}: {mean_r:.4f}")
                if a not in approach_results:
                    approach_results[a] = {}
                approach_results[a][f'R@{k}'] = mean_r

        all_approach_results[f'K={fanout_k}'] = approach_results

    return all_approach_results


def test4_with_denser_bridges(world, predictor, bilinear, assoc_by_state):
    """
    Test with manually constructed denser bridges.

    Instead of using the pre-built bridge test set, find bridges where:
    - The shared object appears MANY times in both trajectories
    - Multiple shared objects exist between the trajectory pair
    """
    print("\n" + "=" * 70)
    print("TEST 4b: DENSER BRIDGE CONSTRUCTION")
    print("=" * 70)

    # Build object -> trajectory -> states map
    obj_traj_states = {}
    for state in world.all_states:
        for obj_id in state.object_ids:
            obj_traj_states.setdefault(obj_id, {}).setdefault(
                state.trajectory_id, []).append(state.global_index)

    # Find trajectory pairs with MULTIPLE shared objects, each appearing many times
    traj_pair_bridges = {}  # (t1, t2) -> [(obj_id, count_t1, count_t2), ...]

    for obj_id, traj_dict in obj_traj_states.items():
        traj_ids = list(traj_dict.keys())
        for i in range(len(traj_ids)):
            for j in range(i + 1, len(traj_ids)):
                t1, t2 = traj_ids[i], traj_ids[j]
                key = (min(t1, t2), max(t1, t2))
                count1 = len(traj_dict[t1])
                count2 = len(traj_dict[t2])
                traj_pair_bridges.setdefault(key, []).append(
                    (obj_id, count1, count2))

    # Rank trajectory pairs by total bridge density
    pair_density = []
    for (t1, t2), bridges_list in traj_pair_bridges.items():
        total_count = sum(c1 + c2 for _, c1, c2 in bridges_list)
        n_objects = len(bridges_list)
        pair_density.append((t1, t2, n_objects, total_count, bridges_list))

    pair_density.sort(key=lambda x: -x[3])  # Sort by total density

    print(f"  Total trajectory pairs with shared objects: {len(pair_density)}")
    print(f"  Top 10 densest pairs:")
    for t1, t2, n_obj, total, bl in pair_density[:10]:
        print(f"    Traj {t1}-{t2}: {n_obj} shared objects, "
              f"{total} total bridge states")

    # Build dense bridge test cases
    dense_bridges = []
    for t1, t2, n_obj, total, bridges_list in pair_density[:50]:
        if n_obj < 2 or total < 10:
            continue

        # Pick states: one from t1, one from t2, both containing a shared object
        for obj_id, c1, c2 in bridges_list[:3]:
            t1_states = obj_traj_states[obj_id][t1]
            t2_states = obj_traj_states[obj_id][t2]

            s1 = np.random.choice(t1_states)
            s2 = np.random.choice(t2_states)

            # Check not directly associated
            key = (min(s1, s2), max(s1, s2))
            if key not in world._associations:
                dense_bridges.append((s1, s2, obj_id, t1, t2, n_obj, total))

    print(f"  Dense bridge test cases: {len(dense_bridges)}")
    if not dense_bridges:
        print("  No dense bridges found!")
        return {}

    # Run spreading activation on dense bridges
    memory_bank = torch.from_numpy(world.all_embeddings).float().to(DEVICE)
    mem_norm = F.normalize(memory_bank, dim=-1)

    k_values = [20, 50, 100]
    approaches = ['pred_spreading', 'cosine_hop2', 'hybrid_0.5', 'full_cosine', 'cosine_direct']
    results = {k: {a: [] for a in approaches} for k in k_values}

    fanout_k = 200

    for idx, (s1, s2, obj_id, t1, t2, n_obj, total) in enumerate(dense_bridges):
        # Ground truth: s2 and its neighbors
        s2_neighbors = assoc_by_state.get(s2, set())
        gt = {s2} | s2_neighbors

        query = memory_bank[s1]
        with torch.no_grad():
            pred_out = F.normalize(predictor.predict(query.unsqueeze(0)), dim=-1)
            hop1_scores = (pred_out @ mem_norm.T).squeeze(0)

        top_k = torch.topk(hop1_scores, fanout_k).indices
        top_k_w = hop1_scores[top_k].cpu().numpy()

        with torch.no_grad():
            inter_emb = memory_bank[top_k]
            h2_pred = F.normalize(predictor.predict(inter_emb), dim=-1)
            h2_pred_scores = (h2_pred @ mem_norm.T).cpu().numpy()

        inter_norm = F.normalize(inter_emb, dim=-1)
        h2_cos_scores = (inter_norm @ mem_norm.T).cpu().numpy()

        # Cosine hop1
        q_norm = F.normalize(query.unsqueeze(0), dim=-1)
        cos_h1 = (q_norm @ mem_norm.T).squeeze(0)
        cos_top_k = torch.topk(cos_h1, fanout_k).indices
        cos_inter = memory_bank[cos_top_k]
        cos_inter_norm = F.normalize(cos_inter, dim=-1)
        h2_cos_cos = (cos_inter_norm @ mem_norm.T).cpu().numpy()
        cos_h1_w = cos_h1[cos_top_k].cpu().numpy()

        scores = {
            'pred_spreading': top_k_w @ h2_pred_scores,
            'cosine_hop2': top_k_w @ h2_cos_scores,
            'hybrid_0.5': top_k_w @ (0.5 * h2_pred_scores + 0.5 * h2_cos_scores),
            'full_cosine': cos_h1_w @ h2_cos_cos,
            'cosine_direct': cos_h1.cpu().numpy(),
        }

        for k in k_values:
            for a in approaches:
                results[k][a].append(recall_at_k(scores[a], gt, k))

    print(f"\n  Results (dense bridges, K={fanout_k}):")
    dense_results = {}
    for k in k_values:
        print(f"  R@{k}:")
        for a in approaches:
            m = float(np.mean(results[k][a]))
            print(f"    {a:25s}: {m:.4f}")
            if a not in dense_results:
                dense_results[a] = {}
            dense_results[a][f'R@{k}'] = m

    return dense_results


def main():
    print("=" * 70)
    print("TEST 4 DEEP INVESTIGATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    t_start = time.time()

    # --- Step 1: Generate world ---
    print("\nSTEP 1: Generating World")
    t0 = time.time()
    config = WorldConfig(num_trajectories=NUM_TRAJECTORIES)
    world = SyntheticWorld(config)
    world.generate_trajectories()
    world.compute_association_ground_truth()
    print(f"Time: {time.time()-t0:.1f}s")

    # --- Step 2: Training data ---
    print("\nSTEP 2: Generating Training Data")
    t0 = time.time()
    anchors, positives, _ = world.get_training_pairs()
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
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr_start=LR_START, lr_end=LR_END,
        temp_start=TEMP_START, temp_end=TEMP_END,
        print_every=100
    )
    print(f"Training time: {time.time()-t0:.1f}s, Loss: {best_loss:.4f}")

    # --- Step 4: Train bilinear ---
    print("\nSTEP 4: Training Bilinear Baseline")
    bilinear = LearnedBilinearBaseline(config.embedding_dim, seed=123)
    bilinear.train(anchors, positives, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

    # --- Step 5: Build test sets and associations ---
    test_sets = world.build_test_sets()
    bridges = test_sets['creative_bridge']

    associations = world._associations
    assoc_by_state = {}
    for (i, j), s in associations.items():
        if s >= 0.3:
            assoc_by_state.setdefault(i, set()).add(j)
            assoc_by_state.setdefault(j, set()).add(i)

    # --- Step 6: Bridge diagnostics ---
    diag_results = diagnose_bridges(world, bridges, assoc_by_state)

    # --- Step 7: Multi-approach comparison ---
    evaluator = BenchmarkEvaluator(world, predictor, bilinear)
    approach_results = test4_approaches(evaluator, world, bridges, assoc_by_state)

    # --- Step 8: Dense bridge test ---
    dense_results = test4_with_denser_bridges(
        world, predictor, bilinear, assoc_by_state)

    # --- Save results ---
    output_dir = Path("results") / "test4_investigation"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'diagnostics': diag_results,
        'approach_comparison': approach_results,
        'dense_bridges': dense_results,
        'config': {
            'epochs': EPOCHS,
            'hidden_dim': HIDDEN_DIM,
            'best_loss': float(best_loss),
        },
        'total_time_sec': time.time() - t_start,
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"INVESTIGATION COMPLETE")
    print(f"Total time: {time.time() - t_start:.1f}s")
    print(f"Results saved to: {output_dir / 'results.json'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
