"""
Test 4 v3: Targeted Cross-Trajectory Bridging

Previous attempts failed because:
v1 (original): 0% oracle reachability - no cross-traj associations exist
v2 (add cross-traj): 30M associations overwhelmed 242k temporal (127:1 ratio)

This version:
- Creates SPARSE cross-trajectory associations (cap at ~5000 total)
- Only for "salient objects" (top 10 most cross-trajectory objects)
- Uses higher object scale (2.5) so shared-object states are more similar
- Tests the dendritic spreading activation approach

Also runs a "direct cross-traj retrieval" test: can the predictor learn
to retrieve cross-trajectory associated states at ALL? (1-hop, not 2-hop)
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
    train_predictor, DEVICE
)
from evaluate_torch import recall_at_k, mrr, BenchmarkEvaluator


def create_targeted_world():
    """Create world with salient objects that bridge trajectories."""
    config = WorldConfig(
        num_trajectories=NUM_TRAJECTORIES,
        object_scale=2.5,  # Boost from 1.5 -> objects matter more in embedding
    )
    world = SyntheticWorld(config)
    world.generate_trajectories()

    # First compute temporal-only associations
    world.compute_association_ground_truth(cross_trajectory=False)
    temporal_assoc = dict(world._associations)
    temporal_count = len(temporal_assoc)
    print(f"Temporal associations: {temporal_count}")

    # Identify salient objects: those appearing in the most trajectories
    obj_trajs = {}
    obj_states = {}
    for s in world.all_states:
        for obj_id in s.object_ids:
            obj_trajs.setdefault(obj_id, set()).add(s.trajectory_id)
            obj_states.setdefault(obj_id, {}).setdefault(
                s.trajectory_id, []).append(s.global_index)

    # Sort objects by trajectory coverage
    obj_coverage = [(obj_id, len(trajs)) for obj_id, trajs in obj_trajs.items()]
    obj_coverage.sort(key=lambda x: -x[1])

    print("\nObject trajectory coverage (top 15):")
    for obj_id, n_trajs in obj_coverage[:15]:
        total_states = sum(len(v) for v in obj_states[obj_id].values())
        print(f"  Object {obj_id}: {n_trajs} trajectories, {total_states} states")

    # Select top-10 salient objects
    salient_objects = [obj_id for obj_id, _ in obj_coverage[:10]]
    print(f"\nSalient objects (top 10): {salient_objects}")

    # Create sparse cross-trajectory associations for salient objects only
    cross_count = 0
    rng = np.random.RandomState(SEED)

    for obj_id in salient_objects:
        traj_dict = obj_states[obj_id]
        traj_ids = list(traj_dict.keys())

        for ti in range(len(traj_ids)):
            for tj in range(ti + 1, len(traj_ids)):
                t1, t2 = traj_ids[ti], traj_ids[tj]

                # Pick ONE state from each trajectory
                s1 = rng.choice(traj_dict[t1])
                s2 = rng.choice(traj_dict[t2])

                key = (min(s1, s2), max(s1, s2))
                # Strong association (1.0) - comparable to close temporal co-occurrence
                temporal_assoc[key] = temporal_assoc.get(key, 0.0) + 1.0
                cross_count += 1

    world._associations = temporal_assoc
    print(f"\nCross-trajectory associations added: {cross_count}")
    print(f"Total associations: {len(temporal_assoc)} "
          f"(cross-traj is {100*cross_count/len(temporal_assoc):.1f}% of total)")

    return world, salient_objects, obj_states


def test_direct_cross_traj_retrieval(world, predictor, salient_objects, obj_states):
    """
    Test 1-hop: Can the predictor directly retrieve cross-trajectory states
    that share a salient object? This is the prerequisite for 2-hop bridging.
    """
    print(f"\n{'=' * 70}")
    print("DIRECT CROSS-TRAJECTORY RETRIEVAL TEST")
    print("=' * 70")

    memory_bank = torch.from_numpy(world.all_embeddings).float().to(DEVICE)
    mem_norm = F.normalize(memory_bank, dim=-1)

    assoc_by_state = {}
    for (i, j), s in world._associations.items():
        if s >= 0.3:
            assoc_by_state.setdefault(i, set()).add(j)
            assoc_by_state.setdefault(j, set()).add(i)

    # For each salient object, test retrieval from state in t1 to state in t2
    k_values = [20, 50, 100]
    pred_recall = {k: [] for k in k_values}
    cos_recall = {k: [] for k in k_values}
    pred_mrr = []
    cos_mrr = []
    n_test = 0

    for obj_id in salient_objects:
        traj_dict = obj_states[obj_id]
        traj_ids = list(traj_dict.keys())

        for ti in range(min(len(traj_ids), 20)):
            for tj in range(ti + 1, min(len(traj_ids), 20)):
                t1, t2 = traj_ids[ti], traj_ids[tj]

                s1 = np.random.choice(traj_dict[t1])
                s2_candidates = traj_dict[t2]

                # Ground truth: all states in t2 with this object
                gt = set(s2_candidates)

                # Predictor scores
                query = memory_bank[s1]
                with torch.no_grad():
                    pred_out = F.normalize(predictor.predict(query.unsqueeze(0)), dim=-1)
                    p_scores = (pred_out @ mem_norm.T).squeeze(0).cpu().numpy()

                # Cosine scores
                q_norm = F.normalize(query.unsqueeze(0), dim=-1)
                c_scores = (q_norm @ mem_norm.T).squeeze(0).cpu().numpy()

                for k in k_values:
                    pred_recall[k].append(recall_at_k(p_scores, gt, k))
                    cos_recall[k].append(recall_at_k(c_scores, gt, k))

                pred_mrr.append(mrr(p_scores, gt))
                cos_mrr.append(mrr(c_scores, gt))
                n_test += 1

                if n_test >= 500:
                    break
            if n_test >= 500:
                break
        if n_test >= 500:
            break

    print(f"\nDirect cross-trajectory retrieval ({n_test} test cases):")
    for k in k_values:
        p = np.mean(pred_recall[k])
        c = np.mean(cos_recall[k])
        print(f"  R@{k}: Predictor={p:.4f}, Cosine={c:.4f}")
    print(f"  MRR: Predictor={np.mean(pred_mrr):.4f}, Cosine={np.mean(cos_mrr):.4f}")

    return {
        'n_test': n_test,
        'recall': {k: {'pred': float(np.mean(pred_recall[k])),
                       'cos': float(np.mean(cos_recall[k]))}
                   for k in k_values},
        'mrr': {'pred': float(np.mean(pred_mrr)),
                'cos': float(np.mean(cos_mrr))},
    }


def test_2hop_bridging(world, predictor, salient_objects, obj_states, assoc_by_state):
    """
    Test 2-hop bridging: s0 -> bridge (temporal) -> target (cross-traj)

    s0: state in t1 that does NOT contain the salient object
    bridge: state in t1 that DOES contain the salient object (temporally near s0)
    target: state in t2 that contains the salient object
    """
    print(f"\n{'=' * 70}")
    print("2-HOP BRIDGING TEST (DENDRITIC SPREADING)")
    print(f"{'=' * 70}")

    memory_bank = torch.from_numpy(world.all_embeddings).float().to(DEVICE)
    mem_norm = F.normalize(memory_bank, dim=-1)
    N = len(world.all_states)

    # Build test cases
    cases = []
    rng = np.random.RandomState(SEED + 1)

    for obj_id in salient_objects:
        traj_dict = obj_states[obj_id]
        traj_ids = list(traj_dict.keys())

        for ti in range(min(len(traj_ids), 30)):
            for tj in range(ti + 1, min(len(traj_ids), 30)):
                if len(cases) >= 500:
                    break

                t1, t2 = traj_ids[ti], traj_ids[tj]
                bridge_states_in_t1 = set(traj_dict[t1])
                target_states_in_t2 = traj_dict[t2]

                # Find s0: temporally near a bridge state, but NOT containing obj
                t1_states_list = world.trajectories[t1].states
                candidates = []
                for b_gi in bridge_states_in_t1:
                    b_state = world.all_states[b_gi]
                    for dt in range(-5, 6):
                        if dt == 0:
                            continue
                        t_idx = b_state.timestep + dt
                        if 0 <= t_idx < len(t1_states_list):
                            cand = t1_states_list[t_idx]
                            if (obj_id not in cand.object_ids and
                                    cand.global_index not in bridge_states_in_t1):
                                # Check temporal association exists
                                if b_gi in assoc_by_state.get(cand.global_index, set()):
                                    candidates.append((cand.global_index, b_gi))

                if not candidates:
                    continue

                s0, bridge = candidates[rng.randint(len(candidates))]
                target = rng.choice(target_states_in_t2)

                # Cross-room requirement
                if world.all_states[s0].room_id == world.all_states[target].room_id:
                    continue

                # All bridge states reachable from s0
                s0_neighbors = assoc_by_state.get(s0, set())
                reachable_bridges = s0_neighbors & bridge_states_in_t1

                cases.append({
                    's0': s0,
                    'target': target,
                    'bridge_states': list(reachable_bridges),
                    'obj_id': obj_id,
                    't1': t1,
                    't2': t2,
                })
            if len(cases) >= 500:
                break
        if len(cases) >= 500:
            break

    print(f"Test cases: {len(cases)}")
    if not cases:
        print("No test cases found!")
        return {}

    n_bridges = [len(c['bridge_states']) for c in cases]
    print(f"Bridge states per case: mean={np.mean(n_bridges):.1f}, "
          f"median={np.median(n_bridges):.0f}")

    # Run multiple approaches
    k_values = [20, 50, 100]
    fanout_values = [50, 200]
    all_results = {}

    for fanout_k in fanout_values:
        print(f"\n--- Fanout K={fanout_k} ---")

        approaches = [
            'pred_1hop', 'pred_weighted', 'dendritic_unweighted',
            'dendritic_softmax', 'hybrid_hop2_0.5', 'cosine_hop2', 'cosine_direct',
        ]
        results = {k: {a: [] for a in approaches} for k in k_values}
        mrr_results = {a: [] for a in approaches}

        bridge_in_top_k = 0
        target_from_bridge = 0

        for ci, case in enumerate(cases):
            s0 = case['s0']
            target = case['target']
            bridge_set = set(case['bridge_states'])

            # Ground truth: target + neighbors in t2
            gt = {target}
            for n_idx in assoc_by_state.get(target, set()):
                if world.all_states[n_idx].trajectory_id == case['t2']:
                    gt.add(n_idx)

            query = memory_bank[s0]

            # Predictor hop1
            with torch.no_grad():
                pred_out = F.normalize(predictor.predict(query.unsqueeze(0)), dim=-1)
                hop1_pred = (pred_out @ mem_norm.T).squeeze(0)

            # Cosine hop1
            q_norm = F.normalize(query.unsqueeze(0), dim=-1)
            hop1_cos = (q_norm @ mem_norm.T).squeeze(0)

            # Top-K predictor
            top_k = torch.topk(hop1_pred, fanout_k).indices
            top_k_w = hop1_pred[top_k].cpu().numpy()
            top_k_set = set(top_k.cpu().numpy())

            if bridge_set & top_k_set:
                bridge_in_top_k += 1

            # Hop2 predictor
            with torch.no_grad():
                inter_emb = memory_bank[top_k]
                h2_pred = F.normalize(predictor.predict(inter_emb), dim=-1)
                h2_pred_s = (h2_pred @ mem_norm.T).cpu().numpy()

            # Hop2 cosine
            inter_norm = F.normalize(inter_emb, dim=-1)
            h2_cos_s = (inter_norm @ mem_norm.T).cpu().numpy()

            # Check if target reachable from any found bridge
            for b in bridge_set & top_k_set:
                b_pos = (top_k.cpu().numpy() == b).argmax()
                if target in set(np.argsort(h2_pred_s[b_pos])[::-1][:100]):
                    target_from_bridge += 1
                    break

            # Score approaches
            scores = {
                'pred_1hop': hop1_pred.cpu().numpy(),
                'pred_weighted': top_k_w @ h2_pred_s,
                'dendritic_unweighted': np.ones(fanout_k) @ h2_pred_s,
                'dendritic_softmax': (lambda w: (np.exp(w*10)/np.exp(w*10).sum()) @ h2_pred_s)(top_k_w),
                'hybrid_hop2_0.5': top_k_w @ (0.5 * h2_pred_s + 0.5 * h2_cos_s),
                'cosine_hop2': top_k_w @ h2_cos_s,
                'cosine_direct': hop1_cos.cpu().numpy(),
            }

            for k in k_values:
                for a, sc in scores.items():
                    results[k][a].append(recall_at_k(sc, gt, k))
            for a, sc in scores.items():
                mrr_results[a].append(mrr(sc, gt))

            if (ci + 1) % 100 == 0:
                print(f"  processed {ci+1}/{len(cases)}")

        n_test = len(cases)
        print(f"\n  Diagnostics (K={fanout_k}):")
        print(f"    Bridge in top-K: {bridge_in_top_k}/{n_test} "
              f"({100*bridge_in_top_k/n_test:.1f}%)")
        print(f"    Target from bridge (top-100): {target_from_bridge}/{n_test} "
              f"({100*target_from_bridge/n_test:.1f}%)")

        print(f"\n  Results (K={fanout_k}):")
        approach_data = {}
        for a in approaches:
            m_mrr = float(np.mean(mrr_results[a]))
            r_strs = [f"R@{k}={np.mean(results[k][a]):.4f}" for k in k_values]
            print(f"    {a:25s}: {', '.join(r_strs)}, MRR={m_mrr:.4f}")
            approach_data[a] = {
                'MRR': m_mrr,
                **{f'R@{k}': float(np.mean(results[k][a])) for k in k_values}
            }

        all_results[f'K={fanout_k}'] = {
            'approaches': approach_data,
            'diagnostics': {
                'bridge_in_topk_pct': 100 * bridge_in_top_k / n_test,
                'target_from_bridge_pct': 100 * target_from_bridge / n_test,
            }
        }

    return all_results


def main():
    print("=" * 70)
    print("TEST 4 v3: TARGETED CROSS-TRAJECTORY BRIDGING")
    print("=" * 70)
    t_start = time.time()

    # --- Create targeted world ---
    world, salient_objects, obj_states = create_targeted_world()

    assoc_by_state = {}
    for (i, j), s in world._associations.items():
        if s >= 0.3:
            assoc_by_state.setdefault(i, set()).add(j)
            assoc_by_state.setdefault(j, set()).add(i)

    # --- Generate training data ---
    print("\nGenerating Training Data...")
    t0 = time.time()
    anchors, positives, _ = world.get_training_pairs(max_pairs=100000)
    print(f"Time: {time.time()-t0:.1f}s")

    # --- Train predictor ---
    print(f"\nTraining Predictor ({EPOCHS}ep, {HIDDEN_DIM}h)")
    predictor = AssociativePredictor(
        embedding_dim=world.config.embedding_dim,
        hidden_dim=HIDDEN_DIM,
        seed=SEED
    )
    t0 = time.time()
    best_loss = train_predictor(
        predictor, anchors, positives,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr_start=LR_START, lr_end=LR_END,
        temp_start=TEMP_START, temp_end=TEMP_END,
        print_every=100
    )
    train_time = time.time() - t0
    print(f"Training: {train_time:.1f}s, Loss: {best_loss:.4f}")

    # --- Quick T1 check ---
    print("\nT1 Check...")
    bilinear = LearnedBilinearBaseline(world.config.embedding_dim, seed=123)
    bilinear.train(anchors, positives, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)
    evaluator = BenchmarkEvaluator(world, predictor, bilinear)
    t1 = evaluator.test_association_vs_similarity()
    print(f"T1 R@20={t1.predictor_score:.3f} (baseline 0.218)")

    # --- Test direct cross-traj retrieval ---
    direct_results = test_direct_cross_traj_retrieval(
        world, predictor, salient_objects, obj_states)

    # --- Test 2-hop bridging ---
    bridging_results = test_2hop_bridging(
        world, predictor, salient_objects, obj_states, assoc_by_state)

    # --- Save ---
    output_dir = Path("results") / "test4_v3"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'config': {
            'epochs': EPOCHS,
            'hidden_dim': HIDDEN_DIM,
            'object_scale': 2.5,
            'n_salient': len(salient_objects),
            'best_loss': float(best_loss),
        },
        'T1_check': {'R@20': t1.predictor_score},
        'direct_cross_traj': direct_results,
        'bridging_2hop': bridging_results,
        'total_time': time.time() - t_start,
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("COMPLETE")
    print(f"Total time: {time.time() - t_start:.1f}s")
    print(f"Results: {output_dir / 'results.json'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
