"""
Test 4 Improved: Cross-Trajectory Object-Mediated Bridging

The original Test 4 was structurally unsolvable: temporal associations are
strictly within-trajectory, so no spreading strategy can cross trajectory
boundaries. Oracle analysis confirmed 0% reachability.

Fix: Add cross-trajectory associations for states sharing objects. This models
an agent recognizing previously-seen objects in new contexts. The creative
bridging test then becomes: from a query state s0, can you reach a target in
another trajectory via 2-hop: s0 -> bridge_state (temporal) -> target (object-mediated)?

Tests multiple spreading activation approaches:
1. Pure predictor (current baseline)
2. Dendritic convergence: fan out broadly, score by convergence of many weak paths
3. Hybrid hop2: predictor hop1, mixed predictor+cosine hop2
4. Various fanout sizes

Also tests whether cross-trajectory associations actually help training.
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
from evaluate_torch import recall_at_k, mrr


def build_bridging_test_cases(world, assoc_by_state, n_cases=500):
    """
    Build proper 2-hop creative bridging test cases.

    Each case: (s0, target, bridge_states, obj_id, t1, t2) where:
    - s0 is in trajectory t1, does NOT contain the bridge object
    - target is in trajectory t2, contains the bridge object
    - bridge_states are in t1, contain the bridge object, and are
      temporally associated with s0
    - The path is: s0 -> bridge_state (temporal) -> target (object-mediated)
    - s0 and target must be in DIFFERENT rooms (cross-room requirement)
    """
    # Build object -> trajectory -> states map
    obj_traj_states = {}
    for s in world.all_states:
        for obj_id in s.object_ids:
            obj_traj_states.setdefault(obj_id, {}).setdefault(
                s.trajectory_id, []).append(s.global_index)

    cases = []
    tried = 0

    obj_ids = list(obj_traj_states.keys())
    np.random.shuffle(obj_ids)

    for obj_id in obj_ids:
        traj_dict = obj_traj_states[obj_id]
        if len(traj_dict) < 2:
            continue

        traj_ids = list(traj_dict.keys())

        for ti in range(len(traj_ids)):
            for tj in range(ti + 1, len(traj_ids)):
                if len(cases) >= n_cases:
                    break
                tried += 1

                t1, t2 = traj_ids[ti], traj_ids[tj]
                t1_obj_states = set(traj_dict[t1])
                t2_obj_states = set(traj_dict[t2])

                # Find s0 in t1: temporally near a bridge state, but NOT containing obj
                t1_all_states = world.trajectories[t1].states
                bridge_in_t1 = [s for s in t1_all_states
                                if s.global_index in t1_obj_states]

                candidates_s0 = []
                for b_state in bridge_in_t1:
                    # Look for states within temporal window that DON'T have the object
                    for dt in range(-5, 6):
                        t_idx = b_state.timestep + dt
                        if 0 <= t_idx < len(t1_all_states) and dt != 0:
                            candidate = t1_all_states[t_idx]
                            if obj_id not in candidate.object_ids:
                                # Check it's associated with the bridge state
                                if b_state.global_index in assoc_by_state.get(
                                        candidate.global_index, set()):
                                    candidates_s0.append(
                                        (candidate.global_index, b_state.global_index))

                if not candidates_s0:
                    continue

                # Pick target in t2
                target_idx = np.random.choice(list(t2_obj_states))
                target_room = world.all_states[target_idx].room_id

                # Pick s0 that's in a different room from target
                valid_s0 = [(s0, b) for s0, b in candidates_s0
                            if world.all_states[s0].room_id != target_room]

                if not valid_s0:
                    continue

                s0_idx, bridge_idx = valid_s0[np.random.randint(len(valid_s0))]

                # All bridge states in t1 associated with s0
                s0_neighbors = assoc_by_state.get(s0_idx, set())
                reachable_bridges = s0_neighbors & t1_obj_states

                if not reachable_bridges:
                    continue

                cases.append({
                    's0': s0_idx,
                    'target': target_idx,
                    'bridge_states': list(reachable_bridges),
                    'obj_id': obj_id,
                    't1': t1,
                    't2': t2,
                    'n_shared_objects': len(
                        set(world.all_states[s0_idx].object_ids) &
                        set(world.all_states[target_idx].object_ids)),
                })

            if len(cases) >= n_cases:
                break
        if len(cases) >= n_cases:
            break

    print(f"Built {len(cases)} bridging test cases (tried {tried} pairs)")

    # Stats
    if cases:
        n_bridges = [len(c['bridge_states']) for c in cases]
        print(f"  Bridge states per case: mean={np.mean(n_bridges):.1f}, "
              f"median={np.median(n_bridges):.0f}, max={np.max(n_bridges)}")

    return cases


def test4_dendritic(world, predictor, test_cases, assoc_by_state,
                    k_values=[20, 50, 100]):
    """
    Run dendritic spreading activation on the improved test cases.

    Multiple approaches compared:
    1. Direct predictor (1-hop)
    2. Predictor spreading (weighted, current approach)
    3. Dendritic convergence (unweighted path counting)
    4. Dendritic with softmax normalization
    5. Hybrid hop2 (predictor + cosine)
    6. Cosine hop2 only
    7. Direct cosine baseline
    """
    print(f"\n{'=' * 70}")
    print("DENDRITIC SPREADING ACTIVATION COMPARISON")
    print(f"{'=' * 70}")

    n_test = len(test_cases)
    N = len(world.all_states)
    memory_bank = torch.from_numpy(world.all_embeddings).float().to(DEVICE)
    mem_norm = F.normalize(memory_bank, dim=-1)

    fanout_values = [50, 200]
    all_results = {}

    for fanout_k in fanout_values:
        print(f"\n--- Fanout K={fanout_k} ---")

        approaches = [
            'pred_1hop',
            'pred_weighted',
            'dendritic_unweighted',
            'dendritic_softmax',
            'hybrid_hop2_0.5',
            'cosine_hop2',
            'cosine_direct',
        ]
        results = {k: {a: [] for a in approaches} for k in k_values}
        mrr_results = {a: [] for a in approaches}

        # Track diagnostics
        bridge_found_in_hop1 = 0
        target_found_in_hop2_from_bridge = 0

        for ci, case in enumerate(test_cases):
            s0 = case['s0']
            target = case['target']
            bridge_states = set(case['bridge_states'])

            # Relaxed ground truth: target + its temporal neighbors
            gt = {target}
            for n_idx in assoc_by_state.get(target, set()):
                if world.all_states[n_idx].trajectory_id == case['t2']:
                    gt.add(n_idx)

            query = memory_bank[s0]

            with torch.no_grad():
                # Predictor hop1
                pred_out = F.normalize(predictor.predict(query.unsqueeze(0)), dim=-1)
                hop1_pred = (pred_out @ mem_norm.T).squeeze(0)  # [N]

            # Cosine hop1
            q_norm = F.normalize(query.unsqueeze(0), dim=-1)
            hop1_cos = (q_norm @ mem_norm.T).squeeze(0)  # [N]

            # Top-K from predictor
            top_k = torch.topk(hop1_pred, fanout_k).indices
            top_k_w = hop1_pred[top_k].cpu().numpy()

            # Check if any bridge state is in top-K
            top_k_set = set(top_k.cpu().numpy())
            if bridge_states & top_k_set:
                bridge_found_in_hop1 += 1

            # Hop2 from predictor top-K
            with torch.no_grad():
                inter_emb = memory_bank[top_k]
                h2_pred = F.normalize(predictor.predict(inter_emb), dim=-1)
                h2_pred_scores = (h2_pred @ mem_norm.T).cpu().numpy()  # [K, N]

            # Hop2 cosine from predictor top-K
            inter_norm = F.normalize(inter_emb, dim=-1)
            h2_cos_scores = (inter_norm @ mem_norm.T).cpu().numpy()  # [K, N]

            # Check if target reachable from bridge states via hop2
            for b in bridge_states & top_k_set:
                b_pos = (top_k.cpu().numpy() == b).argmax()
                if target in set(np.argsort(h2_pred_scores[b_pos])[::-1][:100]):
                    target_found_in_hop2_from_bridge += 1
                    break

            # === Approach 1: Direct predictor (1-hop) ===
            scores_1hop = hop1_pred.cpu().numpy()

            # === Approach 2: Weighted predictor spreading ===
            scores_weighted = top_k_w @ h2_pred_scores

            # === Approach 3: Dendritic unweighted convergence ===
            scores_unweighted = np.ones(fanout_k) @ h2_pred_scores

            # === Approach 4: Dendritic softmax convergence ===
            sw = np.exp(top_k_w * 10)
            sw = sw / sw.sum()
            scores_softmax = sw @ h2_pred_scores

            # === Approach 5: Hybrid hop2 ===
            h2_hybrid = 0.5 * h2_pred_scores + 0.5 * h2_cos_scores
            scores_hybrid = top_k_w @ h2_hybrid

            # === Approach 6: Cosine hop2 ===
            scores_cos_h2 = top_k_w @ h2_cos_scores

            # === Approach 7: Direct cosine ===
            scores_cos_direct = hop1_cos.cpu().numpy()

            approach_scores = {
                'pred_1hop': scores_1hop,
                'pred_weighted': scores_weighted,
                'dendritic_unweighted': scores_unweighted,
                'dendritic_softmax': scores_softmax,
                'hybrid_hop2_0.5': scores_hybrid,
                'cosine_hop2': scores_cos_h2,
                'cosine_direct': scores_cos_direct,
            }

            for k in k_values:
                for a_name, a_scores in approach_scores.items():
                    results[k][a_name].append(recall_at_k(a_scores, gt, k))

            for a_name, a_scores in approach_scores.items():
                mrr_results[a_name].append(mrr(a_scores, gt))

            if (ci + 1) % 100 == 0:
                print(f"  processed {ci+1}/{n_test}")

        # Diagnostics
        print(f"\n  Diagnostics (K={fanout_k}):")
        print(f"    Bridge found in hop1 top-K: {bridge_found_in_hop1}/{n_test} "
              f"({100*bridge_found_in_hop1/n_test:.1f}%)")
        print(f"    Target reachable from bridge via hop2 (top-100): "
              f"{target_found_in_hop2_from_bridge}/{n_test} "
              f"({100*target_found_in_hop2_from_bridge/n_test:.1f}%)")

        # Results
        print(f"\n  Results (K={fanout_k}):")
        approach_data = {}
        for a in approaches:
            m_mrr = float(np.mean(mrr_results[a]))
            r_strs = []
            for k in k_values:
                m_r = float(np.mean(results[k][a]))
                r_strs.append(f"R@{k}={m_r:.4f}")
            print(f"    {a:25s}: {', '.join(r_strs)}, MRR={m_mrr:.4f}")
            approach_data[a] = {
                'MRR': m_mrr,
                **{f'R@{k}': float(np.mean(results[k][a])) for k in k_values}
            }

        all_results[f'K={fanout_k}'] = {
            'approaches': approach_data,
            'diagnostics': {
                'bridge_found_pct': 100 * bridge_found_in_hop1 / n_test,
                'target_reachable_pct': 100 * target_found_in_hop2_from_bridge / n_test,
            }
        }

    return all_results


def main():
    print("=" * 70)
    print("TEST 4 IMPROVED: CROSS-TRAJECTORY BRIDGING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    t_start = time.time()

    # --- Step 1: Generate world WITH cross-trajectory associations ---
    print("\nSTEP 1: Generating World (with cross-trajectory associations)")
    config = WorldConfig(num_trajectories=NUM_TRAJECTORIES)
    world = SyntheticWorld(config)
    world.generate_trajectories()

    # Compute associations WITH cross-trajectory links
    world.compute_association_ground_truth(
        cross_trajectory=True, cross_traj_strength=0.3)

    n_assoc = len(world._associations)
    print(f"Total associations: {n_assoc}")

    # Build assoc lookup
    assoc_by_state = {}
    for (i, j), s in world._associations.items():
        if s >= 0.3:
            assoc_by_state.setdefault(i, set()).add(j)
            assoc_by_state.setdefault(j, set()).add(i)

    # --- Step 2: Training data ---
    print("\nSTEP 2: Generating Training Data")
    t0 = time.time()
    anchors, positives, _ = world.get_training_pairs(max_pairs=100000)
    print(f"Training pairs: {len(anchors)}, Time: {time.time()-t0:.1f}s")

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
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s, Loss: {best_loss:.4f}")

    # --- Step 4: Quick T1 check (did cross-traj associations hurt T1?) ---
    print("\nSTEP 4: Quick T1 Check")
    bilinear = LearnedBilinearBaseline(config.embedding_dim, seed=123)
    bilinear.train(anchors, positives, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

    from evaluate_torch import BenchmarkEvaluator
    evaluator = BenchmarkEvaluator(world, predictor, bilinear)
    t1_result = evaluator.test_association_vs_similarity()
    print(f"  T1 R@20 = {t1_result.predictor_score:.3f} "
          f"(baseline was 0.218, check for degradation)")

    # --- Step 5: Build bridging test cases ---
    print("\nSTEP 5: Building Bridging Test Cases")
    test_cases = build_bridging_test_cases(world, assoc_by_state, n_cases=500)

    if len(test_cases) < 50:
        print("WARNING: Too few test cases, trying with relaxed constraints...")
        test_cases = build_bridging_test_cases(world, assoc_by_state, n_cases=1000)

    # --- Step 6: Run dendritic spreading activation ---
    print("\nSTEP 6: Dendritic Spreading Activation")
    spreading_results = test4_dendritic(
        world, predictor, test_cases, assoc_by_state)

    # --- Save results ---
    output_dir = Path("results") / "test4_improved"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'config': {
            'epochs': EPOCHS,
            'hidden_dim': HIDDEN_DIM,
            'cross_traj_strength': 0.3,
            'best_loss': float(best_loss),
            'training_time': float(train_time),
        },
        'T1_check': {
            'R@20': t1_result.predictor_score,
            'MRR': t1_result.details['mrr']['predictor'],
        },
        'n_test_cases': len(test_cases),
        'spreading_results': spreading_results,
        'total_time_sec': time.time() - t_start,
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("TEST 4 IMPROVED - COMPLETE")
    print(f"T1 R@20: {t1_result.predictor_score:.3f}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Total time: {time.time() - t_start:.1f}s")
    print(f"Results saved to: {output_dir / 'results.json'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
