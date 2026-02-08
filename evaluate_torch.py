"""
Evaluation Framework for Predictive Associative Memory -- PyTorch Compatible

Primary evaluation: faithfulness metrics (train on ALL associations, measure
how faithfully the predictor recalls experienced associations).

Secondary evaluation: generalisation stress test (70/30 edge-disjoint split).
"""

import numpy as np
import torch
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from world import SyntheticWorld
from models_torch import (
    AssociativePredictor, CosineSimilarityBaseline,
    LearnedBilinearBaseline, DecayingMemoryStore, DEVICE
)


@dataclass
class TestResult:
    test_name: str
    predictor_score: float
    cosine_baseline_score: float
    bilinear_baseline_score: float
    details: Dict


def recall_at_k(scores: np.ndarray, gt_indices: set, k: int) -> float:
    if not gt_indices:
        return 0.0
    top_k = set(np.argsort(scores)[::-1][:k])
    return len(top_k & gt_indices) / len(gt_indices)


def mrr(scores: np.ndarray, gt_indices: set) -> float:
    if not gt_indices:
        return 0.0
    for rank, idx in enumerate(np.argsort(scores)[::-1], 1):
        if idx in gt_indices:
            return 1.0 / rank
    return 0.0


def association_precision_at_k(scores: np.ndarray, true_associates: set, k: int) -> float:
    """Of top-k retrieved, what fraction are true temporal associates?"""
    if not true_associates:
        return 0.0
    top_k = set(np.argsort(scores)[::-1][:k])
    return len(top_k & true_associates) / k


def discrimination_auc_single(scores: np.ndarray, true_associates: set,
                               query_idx: int) -> float:
    """ROC-AUC for separating true associates from non-associates for one query."""
    n = len(scores)
    if not true_associates or len(true_associates) >= n - 1:
        return 0.5
    labels = np.zeros(n, dtype=np.float32)
    for idx in true_associates:
        labels[idx] = 1.0
    labels[query_idx] = -1  # exclude self
    mask = labels >= 0
    y = labels[mask]
    s = scores[mask]
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    # Wilcoxon-Mann-Whitney statistic (fast AUC)
    pos_scores = s[y == 1]
    neg_scores = s[y == 0]
    # Sample negatives if too many (for speed)
    if len(neg_scores) > 2000:
        rng = np.random.RandomState(42)
        neg_scores = rng.choice(neg_scores, 2000, replace=False)
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Count how often a positive score > negative score
    count = 0
    for ps in pos_scores:
        count += np.sum(ps > neg_scores) + 0.5 * np.sum(ps == neg_scores)
    return float(count / (n_pos * n_neg))


class BenchmarkEvaluator:
    def __init__(self, world: SyntheticWorld, predictor: AssociativePredictor,
                 bilinear: LearnedBilinearBaseline,
                 test_associations: Dict = None):
        """
        Args:
            test_associations: If provided, T1/T2 evaluate only on these held-out
                associations (edge-disjoint from training). If None, uses all
                associations (legacy behavior, NOT recommended).
        """
        self.world = world
        self.predictor = predictor
        self.bilinear = bilinear
        self.test_associations = test_associations

        # Keep memory bank on GPU
        self.memory_bank = torch.from_numpy(world.all_embeddings).float().to(DEVICE)
        self.cosine = CosineSimilarityBaseline(world.all_embeddings)
        self.test_sets = world.build_test_sets()

    def _pred_scores(self, query_idx: int) -> np.ndarray:
        """Get predictor scores for a query."""
        query = self.memory_bank[query_idx]
        scores = self.predictor.association_scores(query, self.memory_bank)
        return scores.cpu().numpy()

    def _cos_scores(self, query_idx: int) -> np.ndarray:
        """Get cosine baseline scores."""
        return self.cosine.all_scores(self.world.all_embeddings[query_idx])

    def _bil_scores(self, query_idx: int) -> np.ndarray:
        """Get bilinear baseline scores."""
        query = self.memory_bank[query_idx]
        return self.bilinear.all_scores(query, self.memory_bank)

    def _pred_multihop(self, query_idx: int, hops: int, continuous: bool = True) -> np.ndarray:
        """Multi-hop retrieval."""
        query = self.memory_bank[query_idx]
        aggregated, _ = self.predictor.multi_hop_retrieval(
            query, self.memory_bank, num_hops=hops, continuous=continuous
        )
        return aggregated

    def _pred_scores_batch(self, query_indices) -> np.ndarray:
        """Get predictor scores for multiple queries at once. Returns [K, N]."""
        if isinstance(query_indices, np.ndarray):
            query_indices = torch.from_numpy(query_indices.copy()).long().to(DEVICE)
        queries = self.memory_bank[query_indices]  # [K, D]
        with torch.no_grad():
            predicted = F.normalize(self.predictor.predict(queries), dim=-1)
            mem_norm = F.normalize(self.memory_bank, dim=-1)
            scores = predicted @ mem_norm.T  # [K, N]
        return scores.cpu().numpy()

    def test_association_vs_similarity(self, k_values=[5, 10, 20, 50]) -> TestResult:
        print("\n" + "=" * 60)
        print("TEST 1: Association != Similarity")
        print("=" * 60)

        # Use test-only associations if provided (avoids train/test leakage)
        associations = self.test_associations if self.test_associations is not None else self.world._associations
        split_label = "HELD-OUT test" if self.test_associations is not None else "ALL (no split)"
        print(f"Evaluating on {len(associations)} {split_label} associations")

        # Build per-state association sets
        assoc_by_state = {}
        for (i, j), s in associations.items():
            assoc_by_state.setdefault(i, set()).add(j)
            assoc_by_state.setdefault(j, set()).add(i)

        # Find queries where associated items are in DIFFERENT rooms
        cross_room_queries = []
        for idx, neighbors in assoc_by_state.items():
            my_room = self.world.all_states[idx].room_id
            cross_room_neighbors = {n for n in neighbors
                                   if self.world.all_states[n].room_id != my_room}
            if len(cross_room_neighbors) >= 3:
                cross_room_queries.append((idx, cross_room_neighbors))

        n_test = min(len(cross_room_queries), 300)
        print(f"Testing {n_test} queries with cross-room associations...")

        results = {k: {'predictor': [], 'cosine': [], 'bilinear': []} for k in k_values}
        mrr_scores = {'predictor': [], 'cosine': [], 'bilinear': []}

        for idx, cross_room_gt in cross_room_queries[:n_test]:
            pred_scores = self._pred_scores(idx)
            cos_scores = self._cos_scores(idx)
            bil_scores = self._bil_scores(idx)

            for k in k_values:
                results[k]['predictor'].append(recall_at_k(pred_scores, cross_room_gt, k))
                results[k]['cosine'].append(recall_at_k(cos_scores, cross_room_gt, k))
                results[k]['bilinear'].append(recall_at_k(bil_scores, cross_room_gt, k))

            mrr_scores['predictor'].append(mrr(pred_scores, cross_room_gt))
            mrr_scores['cosine'].append(mrr(cos_scores, cross_room_gt))
            mrr_scores['bilinear'].append(mrr(bil_scores, cross_room_gt))

            if (len(results[k_values[0]]['predictor'])) % 100 == 0:
                print(f"  processed {len(results[k_values[0]]['predictor'])}/{n_test}")

        details = {'recall_at_k': {}, 'mrr': {}}
        print("\n--- Results: Cross-Room Association Retrieval ---")
        print("Recall@K:")
        for k in k_values:
            d = {}
            for m in ['predictor', 'cosine', 'bilinear']:
                d[m] = float(np.mean(results[k][m]))
            print(f"  R@{k}: Pred={d['predictor']:.3f}  Cos={d['cosine']:.3f}  Bil={d['bilinear']:.3f}")
            details['recall_at_k'][k] = d

        print("Mean Reciprocal Rank:")
        for m in ['predictor', 'cosine', 'bilinear']:
            v = float(np.mean(mrr_scores[m]))
            print(f"  {m}: {v:.3f}")
            details['mrr'][m] = v

        return TestResult('association_vs_similarity',
                         float(np.mean(results[20]['predictor'])),
                         float(np.mean(results[20]['cosine'])),
                         float(np.mean(results[20]['bilinear'])),
                         details)

    # =========================================================================
    # Test 2: Transitive Association
    # =========================================================================
    def test_transitive_association(self, k_values=[10, 20, 50]) -> TestResult:
        print("\n" + "=" * 60)
        print("TEST 2: Transitive Association (Multi-Hop)")
        print("=" * 60)

        chains = self.test_sets['transitive_2hop']

        cross_room_chains = [
            (a, b, c) for a, b, c in chains
            if self.world.all_states[a].room_id != self.world.all_states[c].room_id
        ]
        same_room_chains = [
            (a, b, c) for a, b, c in chains
            if self.world.all_states[a].room_id == self.world.all_states[c].room_id
        ]

        print(f"Total chains: {len(chains)}")
        print(f"Cross-room chains (a,c different rooms): {len(cross_room_chains)}")
        print(f"Same-room chains: {len(same_room_chains)}")

        all_results = {}
        for label, test_chains in [('cross_room', cross_room_chains),
                                   ('same_room', same_room_chains)]:
            n_test = min(len(test_chains), 200)
            if n_test == 0:
                continue
            print(f"\nTesting {n_test} {label} 2-hop chains...")

            results = {k: {m: [] for m in ['pred_1hop', 'pred_2hop', 'pred_3hop', 'cosine', 'bilinear']}
                       for k in k_values}

            for a, b, c in test_chains[:n_test]:
                gt = {c}
                p1 = self._pred_scores(a)
                p2 = self._pred_multihop(a, hops=2)
                p3 = self._pred_multihop(a, hops=3)
                cs = self._cos_scores(a)
                bs = self._bil_scores(a)

                for k in k_values:
                    results[k]['pred_1hop'].append(recall_at_k(p1, gt, k))
                    results[k]['pred_2hop'].append(recall_at_k(p2, gt, k))
                    results[k]['pred_3hop'].append(recall_at_k(p3, gt, k))
                    results[k]['cosine'].append(recall_at_k(cs, gt, k))
                    results[k]['bilinear'].append(recall_at_k(bs, gt, k))

            all_results[label] = results

            print(f"  Results ({label}):")
            for k in k_values:
                print(f"  R@{k}:", end="")
                for m in ['pred_1hop', 'pred_2hop', 'pred_3hop', 'cosine', 'bilinear']:
                    print(f"  {m}={np.mean(results[k][m]):.3f}", end="")
                print()

        details = {label: {k: {m: float(np.mean(r[k][m])) for m in r[k]}
                           for k in k_values}
                   for label, r in all_results.items()}

        cr = all_results.get('cross_room', {})
        pred_score = float(np.mean(cr.get(20, {}).get('pred_2hop', [0])))
        cos_score = float(np.mean(cr.get(20, {}).get('cosine', [0])))
        bil_score = float(np.mean(cr.get(20, {}).get('bilinear', [0])))

        return TestResult('transitive_association', pred_score, cos_score, bil_score, details)

    # =========================================================================
    # Test 2b: Transitive Association with Split-Aware Chain Construction
    # =========================================================================
    def test_transitive_split(self, train_associations: Dict, test_associations: Dict,
                              k_values=[10, 20, 50], max_chains=500) -> Dict:
        """
        Build transitive chains with explicit edge provenance.

        Returns results for two chain types:
          - 'pure_test': A->B test edge, B->C test edge (fully held-out)
          - 'train_to_test': A->B train edge, B->C test edge (chain generalization)

        Both require A and C in different rooms (cross-room only).
        """
        print("\n" + "=" * 60)
        print("TEST 2b: Transitive Association (Split-Aware Chains)")
        print("=" * 60)

        # Build adjacency from each split (all associations, no strength filter)
        def build_adj(assoc):
            adj = {}
            for (i, j), s in assoc.items():
                adj.setdefault(i, set()).add(j)
                adj.setdefault(j, set()).add(i)
            return adj

        train_adj = build_adj(train_associations)
        test_adj = build_adj(test_associations)
        all_assoc = {**train_associations, **test_associations}

        print(f"Train adj nodes: {len(train_adj)}, Test adj nodes: {len(test_adj)}")

        rng = np.random.RandomState(123)
        states = self.world.all_states

        # Build chains: iterate over B nodes
        pure_test_chains = []
        train_to_test_chains = []

        all_nodes = list(set(list(train_adj.keys()) + list(test_adj.keys())))
        rng.shuffle(all_nodes)

        for b in all_nodes:
            if len(pure_test_chains) >= max_chains and len(train_to_test_chains) >= max_chains:
                break

            b_test_neighbors = test_adj.get(b, set())
            b_train_neighbors = train_adj.get(b, set())

            # Pure test: A and C both from test neighbors, in different rooms
            if len(pure_test_chains) < max_chains and len(b_test_neighbors) >= 2:
                by_room = {}
                for n in b_test_neighbors:
                    by_room.setdefault(states[n].room_id, []).append(n)
                room_ids = list(by_room.keys())
                if len(room_ids) >= 2:
                    for ri in range(len(room_ids)):
                        for rj in range(ri + 1, len(room_ids)):
                            a = rng.choice(by_room[room_ids[ri]])
                            c = rng.choice(by_room[room_ids[rj]])
                            key_ac = (min(a, c), max(a, c))
                            if key_ac not in all_assoc:
                                pure_test_chains.append((a, b, c))
                            if len(pure_test_chains) >= max_chains:
                                break
                        if len(pure_test_chains) >= max_chains:
                            break

            # Train->test: A from train neighbors, C from test neighbors, different rooms
            if len(train_to_test_chains) < max_chains and b_train_neighbors and b_test_neighbors:
                train_by_room = {}
                for n in b_train_neighbors:
                    train_by_room.setdefault(states[n].room_id, []).append(n)
                test_by_room = {}
                for n in b_test_neighbors:
                    test_by_room.setdefault(states[n].room_id, []).append(n)

                for tr_room, a_pool in train_by_room.items():
                    for te_room, c_pool in test_by_room.items():
                        if tr_room != te_room:
                            a = rng.choice(a_pool)
                            c = rng.choice(c_pool)
                            key_ac = (min(a, c), max(a, c))
                            if key_ac not in all_assoc:
                                train_to_test_chains.append((a, b, c))
                        if len(train_to_test_chains) >= max_chains:
                            break
                    if len(train_to_test_chains) >= max_chains:
                        break

        print(f"Pure test chains (A->B test, B->C test): {len(pure_test_chains)}")
        print(f"Train->test chains (A->B train, B->C test): {len(train_to_test_chains)}")

        all_results = {}
        for label, chains in [('pure_test', pure_test_chains),
                               ('train_to_test', train_to_test_chains)]:
            n_test = min(len(chains), 200)
            if n_test == 0:
                print(f"\n  {label}: no chains found, skipping")
                continue
            print(f"\nTesting {n_test} {label} cross-room chains...")

            results = {k: {m: [] for m in ['pred_1hop', 'pred_2hop', 'pred_3hop', 'cosine', 'bilinear']}
                       for k in k_values}

            for a, b, c in chains[:n_test]:
                gt = {c}
                p1 = self._pred_scores(a)
                p2 = self._pred_multihop(a, hops=2)
                p3 = self._pred_multihop(a, hops=3)
                cs = self._cos_scores(a)
                bs = self._bil_scores(a)

                for k in k_values:
                    results[k]['pred_1hop'].append(recall_at_k(p1, gt, k))
                    results[k]['pred_2hop'].append(recall_at_k(p2, gt, k))
                    results[k]['pred_3hop'].append(recall_at_k(p3, gt, k))
                    results[k]['cosine'].append(recall_at_k(cs, gt, k))
                    results[k]['bilinear'].append(recall_at_k(bs, gt, k))

            all_results[label] = {k: {m: float(np.mean(results[k][m])) for m in results[k]}
                                  for k in k_values}

            print(f"  Results ({label}):")
            for k in k_values:
                print(f"  R@{k}:", end="")
                for m in ['pred_1hop', 'pred_2hop', 'pred_3hop', 'cosine', 'bilinear']:
                    print(f"  {m}={all_results[label][k][m]:.3f}", end="")
                print()

        return all_results

    # =========================================================================
    # Test 3: Decay Ablation
    # =========================================================================
    def test_decay_ablation(self, num_steps=500, queries_per_checkpoint=10,
                            decay_rate=0.99) -> TestResult:
        print("\n" + "=" * 60)
        print("TEST 3: Decay Ablation (Context-Relevant Queries)")
        print("=" * 60)

        associations = self.test_associations if self.test_associations is not None else self.world._associations
        N = len(self.world.all_states)

        assoc_by_state = {}
        for (i, j), s in associations.items():
            if s >= 0.3:
                assoc_by_state.setdefault(i, set()).add(j)
                assoc_by_state.setdefault(j, set()).add(i)

        store_decay = DecayingMemoryStore(self.world.all_embeddings, decay_rate=decay_rate)
        store_flat = DecayingMemoryStore(self.world.all_embeddings, decay_rate=1.0)

        prec_decay, prec_flat, steps = [], [], []
        recently_active = []

        for step in range(num_steps):
            store_decay.step()
            store_flat.step()

            traj_id = np.random.randint(len(self.world.trajectories))
            traj = self.world.trajectories[traj_id]
            start_t = np.random.randint(max(1, len(traj.states) - 20))

            segment_states = []
            for t in range(start_t, min(start_t + 15, len(traj.states))):
                state_idx = traj.states[t].global_index
                segment_states.append(state_idx)

            activation_strength = 0.7
            store_decay.activate(np.array(segment_states),
                                 np.ones(len(segment_states)) * activation_strength)
            store_flat.activate(np.array(segment_states),
                                np.ones(len(segment_states)) * activation_strength)

            recently_active.extend(segment_states)
            if len(recently_active) > 200:
                recently_active = recently_active[-200:]

            if step % 50 == 0 and step > 0 and len(recently_active) >= 50:
                pd_list, pf_list = [], []

                query_candidates = [s for s in recently_active if s in assoc_by_state]
                if len(query_candidates) < queries_per_checkpoint:
                    continue

                # Get warmth as numpy for scoring
                warmth_decay = store_decay.warmth.cpu().numpy()
                warmth_flat = store_flat.warmth.cpu().numpy()

                for _ in range(queries_per_checkpoint):
                    qi = np.random.choice(query_candidates)
                    gt = assoc_by_state.get(qi, set())
                    if not gt:
                        continue

                    raw = self._pred_scores(qi)

                    decay_scores = raw * warmth_decay
                    flat_scores = raw * warmth_flat

                    top20_d = set(np.argsort(decay_scores)[::-1][:20])
                    top20_f = set(np.argsort(flat_scores)[::-1][:20])

                    pd_list.append(len(top20_d & gt) / min(20, len(gt)))
                    pf_list.append(len(top20_f & gt) / min(20, len(gt)))

                if pd_list:
                    prec_decay.append(np.mean(pd_list))
                    prec_flat.append(np.mean(pf_list))
                    steps.append(step)
                    print(f"  Step {step}: decay={prec_decay[-1]:.3f}  no_decay={prec_flat[-1]:.3f}")

        details = dict(steps=steps, with_decay=prec_decay, without_decay=prec_flat)

        mean_decay = float(np.mean(prec_decay)) if prec_decay else 0
        mean_flat = float(np.mean(prec_flat)) if prec_flat else 0

        print(f"\nMean precision - Decay: {mean_decay:.3f}  No-decay: {mean_flat:.3f}")
        print(f"Improvement: {((mean_decay - mean_flat) / (mean_flat + 1e-6)) * 100:+.1f}%")

        return TestResult('decay_ablation', mean_decay, mean_flat, 0.0, details)

    # =========================================================================
    # Test 4: Creative Bridging (Dendritic Spreading Activation)
    # =========================================================================
    def test_creative_bridging(self, k_values=[20, 50, 100], fanout_k=50) -> TestResult:
        print("\n" + "=" * 60)
        print("TEST 4: Creative Bridging (Dendritic Spreading Activation)")
        print("=" * 60)

        bridges = self.test_sets['creative_bridge']
        n_test = min(len(bridges), 300)
        print(f"Testing {n_test} cross-trajectory bridges...")
        print(f"Using spreading activation with K={fanout_k} fanout")

        associations = self.world._associations
        assoc_by_state = {}
        for (i, j), s in associations.items():
            if s >= 0.3:
                assoc_by_state.setdefault(i, set()).add(j)
                assoc_by_state.setdefault(j, set()).add(i)

        results = {k: {m: [] for m in ['pred_spreading', 'pred_serial', 'cosine', 'bilinear']}
                   for k in k_values}

        convergence_count = 0
        N = len(self.world.all_states)

        for i in range(n_test):
            s1, s2, obj_id, t1, t2 = bridges[i]

            # Find intermediate states
            s1_neighbors = assoc_by_state.get(s1, set())
            s2_neighbors = assoc_by_state.get(s2, set())

            t1_bridge_states = {
                s.global_index for s in self.world.trajectories[t1].states
                if obj_id in s.object_ids
            }
            t2_bridge_states = {
                s.global_index for s in self.world.trajectories[t2].states
                if obj_id in s.object_ids
            }

            intermediates = (s1_neighbors & t1_bridge_states) | (s2_neighbors & t2_bridge_states)

            # Relaxed ground truth
            gt_relaxed = {s2}
            for interm in intermediates:
                gt_relaxed |= (assoc_by_state.get(interm, set()) & s2_neighbors)

            # === SPREADING ACTIVATION (batched GPU) ===
            hop1_scores = self._pred_scores(s1)
            top_k_indices = np.argsort(hop1_scores)[::-1][:fanout_k]

            # Batch all K intermediates in one GPU forward pass
            hop2_all = self._pred_scores_batch(top_k_indices)  # [K, N]
            hop1_weights = hop1_scores[top_k_indices]  # [K]

            # Weighted sum: spreading_scores[n] = sum_k(hop2[k,n] * weight[k])
            spreading_scores = hop1_weights @ hop2_all  # [N]

            # Track convergence: count states reached by multiple paths in top-100
            top100_per_intermediate = np.argsort(hop2_all, axis=1)[:, ::-1][:, :100]
            path_counts = np.zeros(N)
            for row in top100_per_intermediate:
                path_counts[row] += 1
            if np.sum(path_counts >= 2) > 0:
                convergence_count += 1

            # === SERIAL MULTI-HOP (for comparison) ===
            serial_2hop = self._pred_multihop(s1, hops=2, continuous=False)

            # Baselines
            cs = self._cos_scores(s1)
            bs = self._bil_scores(s1)

            for k in k_values:
                results[k]['pred_spreading'].append(recall_at_k(spreading_scores, gt_relaxed, k))
                results[k]['pred_serial'].append(recall_at_k(serial_2hop, gt_relaxed, k))
                results[k]['cosine'].append(recall_at_k(cs, {s2}, k))
                results[k]['bilinear'].append(recall_at_k(bs, {s2}, k))

            if (i+1) % 100 == 0:
                print(f"  processed {i+1}/{n_test}")

        print(f"\nDiagnostics:")
        print(f"  Queries with convergent paths (>=2 paths to same state): {convergence_count}/{n_test} ({100*convergence_count/n_test:.1f}%)")

        details = {'recall_at_k': {}, 'diagnostics': {
            'convergence_pct': 100*convergence_count/n_test,
            'fanout_k': fanout_k
        }}

        print("\n--- Results (Spreading vs Serial) ---")
        for k in k_values:
            print(f"Recall@{k}:")
            d = {}
            for m in ['pred_spreading', 'pred_serial', 'cosine', 'bilinear']:
                d[m] = float(np.mean(results[k][m]))
                print(f"  {m}: {d[m]:.3f}")
            details['recall_at_k'][k] = d

        return TestResult('creative_bridging',
                         float(np.mean(results[50]['pred_spreading'])),
                         float(np.mean(results[50]['cosine'])),
                         float(np.mean(results[50]['bilinear'])),
                         details)

    # =========================================================================
    # Test 5: Familiarity Normalisation (Running EMA)
    # =========================================================================
    def test_familiarity_normalisation(self, trajectory_length=500, novel_events=20, ema_alpha=0.05) -> TestResult:
        print("\n" + "=" * 60)
        print("TEST 5: Familiarity Normalisation (Running EMA)")
        print("=" * 60)
        print("Testing whether the system selectively encodes novel vs. familiar co-occurrences")
        print(f"Using exponential moving average with alpha={ema_alpha}")

        N = len(self.world.all_states)
        embeddings_np = self.world.all_embeddings  # numpy for EMA math
        embedding_dim = embeddings_np.shape[1]

        # Pick a "home" room and a "novel" room
        room_ids = list(set(s.room_id for s in self.world.all_states))
        if len(room_ids) < 2:
            print("  Warning: Not enough rooms for novelty test")
            return TestResult('familiarity_normalisation', 0.0, 0.0, 0.0, {})

        home_room = room_ids[0]
        novel_room = room_ids[1]

        home_states = [s.global_index for s in self.world.all_states if s.room_id == home_room]
        novel_states_pool = [s.global_index for s in self.world.all_states if s.room_id == novel_room]

        print(f"  Home environment (room {home_room}): {len(home_states)} states")
        print(f"  Novel environment (room {novel_room}): {len(novel_states_pool)} states")

        # Phase 1: Build running EMA of familiar environment
        print(f"  Phase 1: Building familiarity baseline ({trajectory_length} steps in home environment)...")

        ema_baseline = np.zeros(embedding_dim)
        trajectory_states = []
        novelty_scores = []

        familiar_steps = int(trajectory_length * 0.85)
        for step in range(familiar_steps):
            state_idx = np.random.choice(home_states)
            trajectory_states.append(('familiar', state_idx))

            state_embedding = embeddings_np[state_idx]
            novelty = np.linalg.norm(state_embedding - ema_baseline)
            novelty_scores.append(('familiar', novelty))

            ema_baseline = (1 - ema_alpha) * ema_baseline + ema_alpha * state_embedding

        # Phase 2: Inject novel events
        print(f"  Phase 2: Injecting {novel_events} novel events (cross-environment intrusions)...")

        novel_state_indices = []
        for i in range(novel_events):
            novel_idx = np.random.choice(novel_states_pool)
            novel_state_indices.append(novel_idx)
            trajectory_states.append(('novel', novel_idx))

            novel_embedding = embeddings_np[novel_idx]
            novelty = np.linalg.norm(novel_embedding - ema_baseline)
            novelty_scores.append(('novel', novelty))

            ema_baseline = (1 - ema_alpha) * ema_baseline + ema_alpha * novel_embedding

            # Intersperse with familiar states
            for _ in range(max(1, (trajectory_length - familiar_steps) // novel_events - 1)):
                if np.random.rand() < 0.7:
                    state_idx = np.random.choice(home_states)
                    trajectory_states.append(('familiar_after_novel', state_idx))
                else:
                    state_idx = np.random.choice(novel_states_pool)
                    trajectory_states.append(('novel_continuation', state_idx))

                state_embedding = embeddings_np[state_idx]
                novelty = np.linalg.norm(state_embedding - ema_baseline)
                novelty_scores.append((trajectory_states[-1][0], novelty))

                ema_baseline = (1 - ema_alpha) * ema_baseline + ema_alpha * state_embedding

        # Phase 3: Evaluate retrieval with EMA-based novelty filtering
        print("  Phase 3: Evaluating retrieval with EMA-based novelty weighting...")

        novel_ranks_raw = []
        novel_ranks_weighted = []
        familiar_ranks_raw = []
        familiar_ranks_weighted = []

        max_dev = 10.0

        # Test retrieval for novel states
        for novel_idx in novel_state_indices[:10]:
            # Rebuild EMA up to this point
            test_ema = np.zeros(embedding_dim)
            for step_type, state_idx in trajectory_states:
                if state_idx == novel_idx:
                    break
                test_ema = (1 - ema_alpha) * test_ema + ema_alpha * embeddings_np[state_idx]

            query_idx = np.random.choice(home_states)
            raw_scores = self._pred_scores(query_idx)

            # Vectorized novelty weights
            deviations = np.linalg.norm(embeddings_np - test_ema, axis=1)
            novelty_weights = 1.0 + np.minimum(1.0, deviations / max_dev)

            weighted_scores = raw_scores * novelty_weights

            sorted_raw = np.argsort(raw_scores)[::-1]
            sorted_weighted = np.argsort(weighted_scores)[::-1]
            rank_raw = np.where(sorted_raw == novel_idx)[0]
            rank_weighted = np.where(sorted_weighted == novel_idx)[0]
            novel_ranks_raw.append(int(rank_raw[0]) if len(rank_raw) > 0 else N)
            novel_ranks_weighted.append(int(rank_weighted[0]) if len(rank_weighted) > 0 else N)

        # Test retrieval for familiar states
        familiar_test = np.random.choice(home_states, size=min(10, len(home_states)), replace=False)
        for fam_idx in familiar_test:
            test_ema = np.zeros(embedding_dim)
            for step_type, state_idx in trajectory_states[:familiar_steps]:
                test_ema = (1 - ema_alpha) * test_ema + ema_alpha * embeddings_np[state_idx]

            query_idx = np.random.choice(home_states)
            raw_scores = self._pred_scores(query_idx)

            deviations = np.linalg.norm(embeddings_np - test_ema, axis=1)
            novelty_weights = 1.0 + np.minimum(1.0, deviations / max_dev)

            weighted_scores = raw_scores * novelty_weights

            sorted_raw = np.argsort(raw_scores)[::-1]
            sorted_weighted = np.argsort(weighted_scores)[::-1]
            rank_raw = np.where(sorted_raw == fam_idx)[0]
            rank_weighted = np.where(sorted_weighted == fam_idx)[0]
            familiar_ranks_raw.append(int(rank_raw[0]) if len(rank_raw) > 0 else N)
            familiar_ranks_weighted.append(int(rank_weighted[0]) if len(rank_weighted) > 0 else N)

        # Results
        mean_novel_raw = float(np.mean(novel_ranks_raw)) if novel_ranks_raw else 0
        mean_novel_weighted = float(np.mean(novel_ranks_weighted)) if novel_ranks_weighted else 0
        mean_familiar_raw = float(np.mean(familiar_ranks_raw)) if familiar_ranks_raw else 0
        mean_familiar_weighted = float(np.mean(familiar_ranks_weighted)) if familiar_ranks_weighted else 0

        novel_novelties = [score for (cat, score) in novelty_scores if cat == 'novel']
        familiar_novelties = [score for (cat, score) in novelty_scores if cat == 'familiar']
        mean_novel_novelty = float(np.mean(novel_novelties)) if novel_novelties else 0
        mean_familiar_novelty = float(np.mean(familiar_novelties)) if familiar_novelties else 0

        print(f"\n--- Results ---")
        print(f"Novelty scores (L2 distance from EMA):")
        print(f"  Novel events mean deviation:    {mean_novel_novelty:.3f}")
        print(f"  Familiar events mean deviation: {mean_familiar_novelty:.3f}")
        print(f"  Separation: {mean_novel_novelty / (mean_familiar_novelty + 1e-6):.2f}x")

        print(f"\nRetrieval ranks for novel states (lower = better):")
        print(f"  Raw predictor mean rank:      {mean_novel_raw:.1f}")
        print(f"  EMA-weighted mean rank:       {mean_novel_weighted:.1f}")
        print(f"  Improvement: {mean_novel_raw - mean_novel_weighted:+.1f}")

        print(f"\nRetrieval ranks for familiar states (should not improve):")
        print(f"  Raw predictor mean rank:      {mean_familiar_raw:.1f}")
        print(f"  EMA-weighted mean rank:       {mean_familiar_weighted:.1f}")
        print(f"  Change: {mean_familiar_weighted - mean_familiar_raw:+.1f}")

        details = {
            'novel_raw_rank': mean_novel_raw,
            'novel_weighted_rank': mean_novel_weighted,
            'familiar_raw_rank': mean_familiar_raw,
            'familiar_weighted_rank': mean_familiar_weighted,
            'mean_novel_novelty': mean_novel_novelty,
            'mean_familiar_novelty': mean_familiar_novelty,
            'ema_alpha': ema_alpha
        }

        novel_improvement = (mean_novel_raw - mean_novel_weighted) / (mean_novel_raw + 1)
        familiar_degradation = max(0, (mean_familiar_weighted - mean_familiar_raw) / (mean_familiar_raw + 1))
        predictor_score = max(0, novel_improvement - familiar_degradation)

        return TestResult('familiarity_normalisation',
                         float(predictor_score), 0.0, 0.0, details)

    # =========================================================================
    # PRIMARY METRICS: Faithfulness evaluation (train on ALL associations)
    # =========================================================================

    def faithfulness_association_precision(self, k_values=[5, 10, 20, 50],
                                           n_queries=500) -> Dict:
        """
        Association Precision@k: of top-k retrieved, what fraction are true
        temporal associates? Evaluated on ALL associations (no holdout).
        """
        print("\n" + "=" * 60)
        print("FAITHFULNESS: Association Precision@k")
        print("=" * 60)

        associations = self.world._associations
        assoc_by_state = {}
        for (i, j), s in associations.items():
            assoc_by_state.setdefault(i, set()).add(j)
            assoc_by_state.setdefault(j, set()).add(i)

        # Sample queries that have associations
        candidates = [idx for idx, nbrs in assoc_by_state.items() if len(nbrs) >= 3]
        rng = np.random.RandomState(42)
        rng.shuffle(candidates)
        test_queries = candidates[:n_queries]

        results = {k: {'predictor': [], 'cosine': []} for k in k_values}

        for qi in test_queries:
            true_assoc = assoc_by_state[qi]
            pred_scores = self._pred_scores(qi)
            cos_scores = self._cos_scores(qi)

            for k in k_values:
                results[k]['predictor'].append(
                    association_precision_at_k(pred_scores, true_assoc, k))
                results[k]['cosine'].append(
                    association_precision_at_k(cos_scores, true_assoc, k))

        print(f"Evaluated {len(test_queries)} queries")
        print(f"\nAssociation Precision@k:")
        details = {}
        for k in k_values:
            p_pred = float(np.mean(results[k]['predictor']))
            p_cos = float(np.mean(results[k]['cosine']))
            print(f"  AP@{k}: Pred={p_pred:.4f}  Cos={p_cos:.4f}")
            details[k] = {'predictor': p_pred, 'cosine': p_cos}

        return details

    def faithfulness_cross_boundary_recall(self, k_values=[5, 10, 20, 50],
                                            n_queries=500) -> Dict:
        """
        Cross-Boundary Recall@k: recall restricted to associations that cross
        room boundaries. This is the headline differentiator -- cosine = 0 here.
        """
        print("\n" + "=" * 60)
        print("FAITHFULNESS: Cross-Boundary Recall@k")
        print("=" * 60)

        associations = self.world._associations
        assoc_by_state = {}
        for (i, j), s in associations.items():
            assoc_by_state.setdefault(i, set()).add(j)
            assoc_by_state.setdefault(j, set()).add(i)

        # Build cross-boundary queries
        cross_room_queries = []
        for idx, neighbors in assoc_by_state.items():
            my_room = self.world.all_states[idx].room_id
            cross_room = {n for n in neighbors
                         if self.world.all_states[n].room_id != my_room}
            if len(cross_room) >= 3:
                cross_room_queries.append((idx, cross_room))

        rng = np.random.RandomState(42)
        rng.shuffle(cross_room_queries)
        n_test = min(len(cross_room_queries), n_queries)
        print(f"Testing {n_test} queries with cross-room associations...")

        results = {k: {'predictor': [], 'cosine': []} for k in k_values}
        mrr_scores = {'predictor': [], 'cosine': []}

        for idx, cross_room_gt in cross_room_queries[:n_test]:
            pred_scores = self._pred_scores(idx)
            cos_scores = self._cos_scores(idx)

            for k in k_values:
                results[k]['predictor'].append(recall_at_k(pred_scores, cross_room_gt, k))
                results[k]['cosine'].append(recall_at_k(cos_scores, cross_room_gt, k))

            mrr_scores['predictor'].append(mrr(pred_scores, cross_room_gt))
            mrr_scores['cosine'].append(mrr(cos_scores, cross_room_gt))

        details = {'recall_at_k': {}, 'mrr': {}}
        print(f"\nCross-Boundary Recall@k:")
        for k in k_values:
            p_pred = float(np.mean(results[k]['predictor']))
            p_cos = float(np.mean(results[k]['cosine']))
            print(f"  CBR@{k}: Pred={p_pred:.4f}  Cos={p_cos:.4f}")
            details['recall_at_k'][k] = {'predictor': p_pred, 'cosine': p_cos}

        for m in ['predictor', 'cosine']:
            details['mrr'][m] = float(np.mean(mrr_scores[m]))
        print(f"  MRR: Pred={details['mrr']['predictor']:.4f}  Cos={details['mrr']['cosine']:.4f}")

        return details

    def faithfulness_discrimination_auc(self, n_queries=300) -> Dict:
        """
        Discrimination AUC: ROC-AUC for separating true associates from
        non-associates across the full memory store.
        """
        print("\n" + "=" * 60)
        print("FAITHFULNESS: Discrimination AUC")
        print("=" * 60)

        associations = self.world._associations
        assoc_by_state = {}
        for (i, j), s in associations.items():
            assoc_by_state.setdefault(i, set()).add(j)
            assoc_by_state.setdefault(j, set()).add(i)

        candidates = [idx for idx, nbrs in assoc_by_state.items() if len(nbrs) >= 5]
        rng = np.random.RandomState(42)
        rng.shuffle(candidates)
        test_queries = candidates[:n_queries]

        pred_aucs = []
        cos_aucs = []

        for qi in test_queries:
            true_assoc = assoc_by_state[qi]
            pred_scores = self._pred_scores(qi)
            cos_scores = self._cos_scores(qi)

            pred_aucs.append(discrimination_auc_single(pred_scores, true_assoc, qi))
            cos_aucs.append(discrimination_auc_single(cos_scores, true_assoc, qi))

        mean_pred = float(np.mean(pred_aucs))
        mean_cos = float(np.mean(cos_aucs))

        print(f"Evaluated {len(test_queries)} queries")
        print(f"  Predictor AUC: {mean_pred:.4f}")
        print(f"  Cosine AUC:    {mean_cos:.4f}")

        # Also compute cross-room-only AUC
        cross_pred_aucs = []
        cross_cos_aucs = []
        for qi in test_queries:
            my_room = self.world.all_states[qi].room_id
            cross_assoc = {n for n in assoc_by_state[qi]
                          if self.world.all_states[n].room_id != my_room}
            if len(cross_assoc) >= 3:
                cross_pred_aucs.append(
                    discrimination_auc_single(self._pred_scores(qi), cross_assoc, qi))
                cross_cos_aucs.append(
                    discrimination_auc_single(self._cos_scores(qi), cross_assoc, qi))

        mean_cross_pred = float(np.mean(cross_pred_aucs)) if cross_pred_aucs else 0.0
        mean_cross_cos = float(np.mean(cross_cos_aucs)) if cross_cos_aucs else 0.0
        print(f"  Cross-room Predictor AUC: {mean_cross_pred:.4f}")
        print(f"  Cross-room Cosine AUC:    {mean_cross_cos:.4f}")

        return {
            'all': {'predictor': mean_pred, 'cosine': mean_cos},
            'cross_room': {'predictor': mean_cross_pred, 'cosine': mean_cross_cos},
            'n_queries': len(test_queries),
            'n_cross_room_queries': len(cross_pred_aucs),
        }

    def faithfulness_specificity(self, k=20, n_queries=300) -> Dict:
        """
        Specificity: does the predictor retrieve the *specific* associated item
        or the whole category? Measures whether top-k hits cluster in the same
        room as the true associate or spread across the whole room.

        For each cross-room query:
          - Get the specific target room(s) of true associates
          - Check how many of the top-k are in those target rooms but NOT true associates
          - High specificity = few false positives from the target room
        """
        print("\n" + "=" * 60)
        print("FAITHFULNESS: Specificity (target vs category)")
        print("=" * 60)

        associations = self.world._associations
        assoc_by_state = {}
        for (i, j), s in associations.items():
            assoc_by_state.setdefault(i, set()).add(j)
            assoc_by_state.setdefault(j, set()).add(i)

        # Group states by room
        room_states = {}
        for s in self.world.all_states:
            room_states.setdefault(s.room_id, set()).add(s.global_index)

        # Queries with cross-room associations
        cross_queries = []
        for idx, neighbors in assoc_by_state.items():
            my_room = self.world.all_states[idx].room_id
            cross = {n for n in neighbors if self.world.all_states[n].room_id != my_room}
            if len(cross) >= 3:
                cross_queries.append((idx, cross))

        rng = np.random.RandomState(42)
        rng.shuffle(cross_queries)
        n_test = min(len(cross_queries), n_queries)

        pred_specificity = []
        cos_specificity = []

        for idx, cross_assoc in cross_queries[:n_test]:
            # Target rooms = rooms containing true cross-room associates
            target_rooms = {self.world.all_states[n].room_id for n in cross_assoc}
            # All states in target rooms
            target_room_states = set()
            for rid in target_rooms:
                target_room_states |= room_states[rid]
            # Category distractors = states in target rooms that are NOT true associates
            category_distractors = target_room_states - cross_assoc - {idx}

            pred_scores = self._pred_scores(idx)
            cos_scores = self._cos_scores(idx)

            top_k_pred = set(np.argsort(pred_scores)[::-1][:k])
            top_k_cos = set(np.argsort(cos_scores)[::-1][:k])

            # Specificity = true associates in top-k / (true associates + category distractors in top-k)
            pred_true_hits = len(top_k_pred & cross_assoc)
            pred_distractor_hits = len(top_k_pred & category_distractors)
            if pred_true_hits + pred_distractor_hits > 0:
                pred_specificity.append(pred_true_hits / (pred_true_hits + pred_distractor_hits))

            cos_true_hits = len(top_k_cos & cross_assoc)
            cos_distractor_hits = len(top_k_cos & category_distractors)
            if cos_true_hits + cos_distractor_hits > 0:
                cos_specificity.append(cos_true_hits / (cos_true_hits + cos_distractor_hits))

        mean_pred = float(np.mean(pred_specificity)) if pred_specificity else 0.0
        mean_cos = float(np.mean(cos_specificity)) if cos_specificity else 0.0

        print(f"Evaluated {n_test} cross-room queries")
        print(f"  Predictor specificity: {mean_pred:.4f}")
        print(f"  Cosine specificity:    {mean_cos:.4f}")
        print(f"  (1.0 = retrieves only specific associates, 0.0 = retrieves whole category)")

        return {
            'predictor': mean_pred,
            'cosine': mean_cos,
            'n_queries': n_test,
            'k': k,
        }

    def run_faithfulness(self) -> Dict:
        """Run all four faithfulness metrics. Returns combined results dict."""
        print("\n" + "#" * 60)
        print("PRIMARY EVALUATION: Faithfulness Metrics")
        print("(Trained on ALL associations -- no holdout)")
        print("#" * 60)

        t0 = time.time()
        results = {}
        results['association_precision'] = self.faithfulness_association_precision()
        results['cross_boundary_recall'] = self.faithfulness_cross_boundary_recall()
        results['discrimination_auc'] = self.faithfulness_discrimination_auc()
        results['specificity'] = self.faithfulness_specificity()

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print("FAITHFULNESS SUMMARY")
        print(f"{'=' * 60}")
        ap20 = results['association_precision'][20]['predictor']
        cbr20 = results['cross_boundary_recall']['recall_at_k'][20]['predictor']
        cbr20_cos = results['cross_boundary_recall']['recall_at_k'][20]['cosine']
        cb_mrr = results['cross_boundary_recall']['mrr']['predictor']
        auc_all = results['discrimination_auc']['all']['predictor']
        auc_cross = results['discrimination_auc']['cross_room']['predictor']
        spec = results['specificity']['predictor']

        print(f"  Association Precision@20:    {ap20:.4f}")
        print(f"  Cross-Boundary Recall@20:    {cbr20:.4f}  (cosine: {cbr20_cos:.4f})")
        print(f"  Cross-Boundary MRR:          {cb_mrr:.4f}")
        print(f"  Discrimination AUC (all):    {auc_all:.4f}")
        print(f"  Discrimination AUC (x-room): {auc_cross:.4f}")
        print(f"  Specificity@20:              {spec:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        results['time'] = elapsed
        return results

    # =========================================================================
    def run_all(self) -> List[TestResult]:
        """Run all 5 tests (legacy)."""
        t0 = time.time()

        results = [
            self.test_association_vs_similarity(),
            self.test_transitive_association(),
            self.test_decay_ablation(),
            self.test_creative_bridging(),
            self.test_familiarity_normalisation(),
        ]

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for result in results:
            print(f"\n{result.test_name}:")
            print(f"  Predictor:  {result.predictor_score:.3f}")
            print(f"  Cosine:     {result.cosine_baseline_score:.3f}")
            print(f"  Bilinear:   {result.bilinear_baseline_score:.3f}")

        print(f"\nTotal time: {time.time()-t0:.1f}s")

        return results
