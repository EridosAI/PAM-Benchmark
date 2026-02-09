"""
Paper reviewer experiments: multi-seed CIs, held-out query-state eval,
P@1/P@5, bilinear baseline, transitive chains, decay ablation.

Tasks:
  2. Multi-seed runs (seeds 42, 123, 456) with all primary metrics
  3. Held-out query-state: 80/20 anchor partition, evaluate on both
  4. P@1 and P@5 (included in all runs via k_values=[1,5,10,20,50])
  5. Bilinear baseline on all faithfulness metrics
  + Pull transitive chain and decay ablation numbers
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json

BATCH_SIZE = 512
LR_START = 5e-4
LR_END = 1e-5
TEMP_START = 0.15
TEMP_END = 0.05
EPOCHS = 500
HIDDEN_DIM = 1024
NUM_TRAJECTORIES = 500

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from world import SyntheticWorld, WorldConfig
from models_torch import (
    AssociativePredictor, LearnedBilinearBaseline,
    train_predictor, DEVICE
)
from evaluate_torch import (
    BenchmarkEvaluator, recall_at_k, mrr, association_precision_at_k,
    discrimination_auc_single
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from run_plateau_experiments import AssociativePredictor4Layer


def train_model(anchors, positives, seed=42):
    """Train a 4-layer predictor."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    pred = AssociativePredictor4Layer(embedding_dim=128, hidden_dim=HIDDEN_DIM, seed=seed)

    t0 = time.time()
    best_loss = train_predictor(
        pred, anchors, positives,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr_start=LR_START, lr_end=LR_END,
        temp_start=TEMP_START, temp_end=TEMP_END,
        print_every=100
    )
    elapsed = time.time() - t0
    print(f"  Training: {elapsed:.1f}s, Loss: {best_loss:.4f}, Params: {pred.count_parameters():,}")
    return pred, best_loss, elapsed


def run_full_faithfulness(world, predictor, bilinear, k_values=[1, 5, 10, 20, 50],
                          n_ap_queries=500, n_cbr_queries=500, n_auc_queries=300,
                          n_spec_queries=300, query_filter=None, label=""):
    """
    Run all faithfulness metrics with optional query filter.

    query_filter: if provided, a set of state indices. Only these states
                  will be used as queries (for held-out anchor eval).
    """
    print(f"\n{'#' * 60}")
    print(f"FAITHFULNESS EVAL: {label}")
    print(f"{'#' * 60}")

    associations = world._associations
    assoc_by_state = {}
    for (i, j), s in associations.items():
        assoc_by_state.setdefault(i, set()).add(j)
        assoc_by_state.setdefault(j, set()).add(i)

    memory_bank = torch.from_numpy(world.all_embeddings).float().to(DEVICE)

    def _pred_scores(qi):
        query = memory_bank[qi]
        scores = predictor.association_scores(query, memory_bank)
        return scores.cpu().numpy()

    def _cos_scores(qi):
        q = torch.from_numpy(world.all_embeddings[qi]).float().to(DEVICE).unsqueeze(0)
        q_norm = F.normalize(q, dim=-1)
        mem_norm = F.normalize(memory_bank, dim=-1)
        return (q_norm @ mem_norm.T).squeeze(0).cpu().numpy()

    def _bil_scores(qi):
        query = memory_bank[qi]
        return bilinear.all_scores(query, memory_bank)

    # ---- Association Precision@k ----
    print(f"\n--- Association Precision@k ---")
    candidates = [idx for idx, nbrs in assoc_by_state.items() if len(nbrs) >= 3]
    if query_filter is not None:
        candidates = [c for c in candidates if c in query_filter]
    rng = np.random.RandomState(42)
    rng.shuffle(candidates)
    ap_queries = candidates[:n_ap_queries]

    ap_results = {k: {'predictor': [], 'cosine': [], 'bilinear': []} for k in k_values}
    for qi in ap_queries:
        ta = assoc_by_state[qi]
        ps = _pred_scores(qi)
        cs = _cos_scores(qi)
        bs = _bil_scores(qi)
        for k in k_values:
            ap_results[k]['predictor'].append(association_precision_at_k(ps, ta, k))
            ap_results[k]['cosine'].append(association_precision_at_k(cs, ta, k))
            ap_results[k]['bilinear'].append(association_precision_at_k(bs, ta, k))

    ap_summary = {}
    for k in k_values:
        d = {m: float(np.mean(ap_results[k][m])) for m in ['predictor', 'cosine', 'bilinear']}
        ap_summary[k] = d
        print(f"  AP@{k}: Pred={d['predictor']:.4f}  Cos={d['cosine']:.4f}  Bil={d['bilinear']:.4f}")
    print(f"  ({len(ap_queries)} queries)")

    # ---- Cross-Boundary Recall@k ----
    print(f"\n--- Cross-Boundary Recall@k ---")
    cross_queries = []
    for idx, nbrs in assoc_by_state.items():
        if query_filter is not None and idx not in query_filter:
            continue
        my_room = world.all_states[idx].room_id
        cross = {n for n in nbrs if world.all_states[n].room_id != my_room}
        if len(cross) >= 3:
            cross_queries.append((idx, cross))
    rng = np.random.RandomState(42)
    rng.shuffle(cross_queries)
    n_cbr = min(len(cross_queries), n_cbr_queries)

    cbr_results = {k: {'predictor': [], 'cosine': [], 'bilinear': []} for k in k_values}
    mrr_scores = {'predictor': [], 'cosine': [], 'bilinear': []}

    for idx, gt in cross_queries[:n_cbr]:
        ps = _pred_scores(idx)
        cs = _cos_scores(idx)
        bs = _bil_scores(idx)
        for k in k_values:
            cbr_results[k]['predictor'].append(recall_at_k(ps, gt, k))
            cbr_results[k]['cosine'].append(recall_at_k(cs, gt, k))
            cbr_results[k]['bilinear'].append(recall_at_k(bs, gt, k))
        mrr_scores['predictor'].append(mrr(ps, gt))
        mrr_scores['cosine'].append(mrr(cs, gt))
        mrr_scores['bilinear'].append(mrr(bs, gt))

    cbr_summary = {}
    for k in k_values:
        d = {m: float(np.mean(cbr_results[k][m])) for m in ['predictor', 'cosine', 'bilinear']}
        cbr_summary[k] = d
        print(f"  CBR@{k}: Pred={d['predictor']:.4f}  Cos={d['cosine']:.4f}  Bil={d['bilinear']:.4f}")
    mrr_summary = {m: float(np.mean(mrr_scores[m])) for m in ['predictor', 'cosine', 'bilinear']}
    print(f"  MRR: Pred={mrr_summary['predictor']:.4f}  Cos={mrr_summary['cosine']:.4f}  Bil={mrr_summary['bilinear']:.4f}")
    print(f"  ({n_cbr} queries)")

    # ---- Discrimination AUC ----
    print(f"\n--- Discrimination AUC ---")
    auc_candidates = [idx for idx, nbrs in assoc_by_state.items() if len(nbrs) >= 5]
    if query_filter is not None:
        auc_candidates = [c for c in auc_candidates if c in query_filter]
    rng = np.random.RandomState(42)
    rng.shuffle(auc_candidates)
    auc_queries = auc_candidates[:n_auc_queries]

    pred_aucs, cos_aucs, bil_aucs = [], [], []
    cross_pred_aucs, cross_cos_aucs, cross_bil_aucs = [], [], []

    for qi in auc_queries:
        ta = assoc_by_state[qi]
        ps = _pred_scores(qi)
        cs = _cos_scores(qi)
        bs = _bil_scores(qi)
        pred_aucs.append(discrimination_auc_single(ps, ta, qi))
        cos_aucs.append(discrimination_auc_single(cs, ta, qi))
        bil_aucs.append(discrimination_auc_single(bs, ta, qi))

        my_room = world.all_states[qi].room_id
        cross = {n for n in ta if world.all_states[n].room_id != my_room}
        if len(cross) >= 3:
            cross_pred_aucs.append(discrimination_auc_single(_pred_scores(qi), cross, qi))
            cross_cos_aucs.append(discrimination_auc_single(_cos_scores(qi), cross, qi))
            cross_bil_aucs.append(discrimination_auc_single(_bil_scores(qi), cross, qi))

    auc_all = {m: float(np.mean(v)) for m, v in
               [('predictor', pred_aucs), ('cosine', cos_aucs), ('bilinear', bil_aucs)]}
    auc_xroom = {m: float(np.mean(v)) for m, v in
                 [('predictor', cross_pred_aucs), ('cosine', cross_cos_aucs), ('bilinear', cross_bil_aucs)]}

    print(f"  All: Pred={auc_all['predictor']:.4f}  Cos={auc_all['cosine']:.4f}  Bil={auc_all['bilinear']:.4f}")
    print(f"  X-room: Pred={auc_xroom['predictor']:.4f}  Cos={auc_xroom['cosine']:.4f}  Bil={auc_xroom['bilinear']:.4f}")
    print(f"  ({len(auc_queries)} queries, {len(cross_pred_aucs)} with x-room)")

    # ---- Specificity@20 ----
    print(f"\n--- Specificity@20 ---")
    room_states = {}
    for s in world.all_states:
        room_states.setdefault(s.room_id, set()).add(s.global_index)

    spec_queries = []
    for idx, nbrs in assoc_by_state.items():
        if query_filter is not None and idx not in query_filter:
            continue
        my_room = world.all_states[idx].room_id
        cross = {n for n in nbrs if world.all_states[n].room_id != my_room}
        if len(cross) >= 3:
            spec_queries.append((idx, cross))
    rng = np.random.RandomState(42)
    rng.shuffle(spec_queries)
    n_spec = min(len(spec_queries), n_spec_queries)

    pred_spec, cos_spec, bil_spec = [], [], []
    for idx, cross_assoc in spec_queries[:n_spec]:
        target_rooms = {world.all_states[n].room_id for n in cross_assoc}
        target_room_states = set()
        for rid in target_rooms:
            target_room_states |= room_states[rid]
        distractors = target_room_states - cross_assoc - {idx}

        for method, scores_fn, spec_list in [
            ('pred', _pred_scores, pred_spec),
            ('cos', _cos_scores, cos_spec),
            ('bil', _bil_scores, bil_spec),
        ]:
            scores = scores_fn(idx)
            top_k = set(np.argsort(scores)[::-1][:20])
            true_hits = len(top_k & cross_assoc)
            distractor_hits = len(top_k & distractors)
            if true_hits + distractor_hits > 0:
                spec_list.append(true_hits / (true_hits + distractor_hits))

    spec_summary = {
        'predictor': float(np.mean(pred_spec)) if pred_spec else 0.0,
        'cosine': float(np.mean(cos_spec)) if cos_spec else 0.0,
        'bilinear': float(np.mean(bil_spec)) if bil_spec else 0.0,
    }
    print(f"  Pred={spec_summary['predictor']:.4f}  Cos={spec_summary['cosine']:.4f}  Bil={spec_summary['bilinear']:.4f}")
    print(f"  ({n_spec} queries)")

    # ---- Median rank of nearest true associate ----
    print(f"\n--- Nearest Associate Rank ---")
    rank_queries = cross_queries[:min(200, n_cbr)]
    pred_ranks = []
    cos_ranks = []
    for idx, gt in rank_queries:
        ps = _pred_scores(idx)
        cs = _cos_scores(idx)
        sorted_pred = np.argsort(ps)[::-1]
        sorted_cos = np.argsort(cs)[::-1]
        for rank, sid in enumerate(sorted_pred, 1):
            if sid in gt:
                pred_ranks.append(rank)
                break
        for rank, sid in enumerate(sorted_cos, 1):
            if sid in gt:
                cos_ranks.append(rank)
                break

    rank_summary = {
        'predictor_median': float(np.median(pred_ranks)) if pred_ranks else 0,
        'predictor_mean': float(np.mean(pred_ranks)) if pred_ranks else 0,
        'cosine_median': float(np.median(cos_ranks)) if cos_ranks else 0,
        'cosine_mean': float(np.mean(cos_ranks)) if cos_ranks else 0,
    }
    print(f"  Pred: median={rank_summary['predictor_median']:.0f}, mean={rank_summary['predictor_mean']:.1f}")
    print(f"  Cos:  median={rank_summary['cosine_median']:.0f}, mean={rank_summary['cosine_mean']:.1f}")

    return {
        'association_precision': ap_summary,
        'cross_boundary_recall': cbr_summary,
        'mrr': mrr_summary,
        'discrimination_auc_all': auc_all,
        'discrimination_auc_xroom': auc_xroom,
        'specificity': spec_summary,
        'nearest_rank': rank_summary,
        'n_queries': {
            'ap': len(ap_queries), 'cbr': n_cbr,
            'auc': len(auc_queries), 'auc_xroom': len(cross_pred_aucs),
            'spec': n_spec,
        },
    }


def run_single_seed(seed, world_seed=42):
    """Run full pipeline for one seed. Returns all results."""
    print(f"\n{'=' * 70}")
    print(f"SEED {seed}")
    print(f"{'=' * 70}")

    np.random.seed(world_seed)
    config = WorldConfig(num_trajectories=NUM_TRAJECTORIES, seed=world_seed)
    world = SyntheticWorld(config)
    world.generate_trajectories()
    world.compute_association_ground_truth()

    print(f"\nGenerating 200k training pairs...")
    t0 = time.time()
    anchors, positives, _ = world.get_training_pairs(max_pairs=200000)
    pair_time = time.time() - t0
    print(f"Pair generation: {pair_time:.1f}s")

    print(f"\nTraining predictor (seed={seed})...")
    predictor, loss, train_time = train_model(anchors, positives, seed=seed)

    bilinear = LearnedBilinearBaseline(128, seed=seed + 81)
    bilinear.train(anchors, positives, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

    results = run_full_faithfulness(world, predictor, bilinear,
                                    label=f"Seed {seed}")
    results['loss'] = float(loss)
    results['train_time'] = float(train_time)
    results['seed'] = seed

    return results, world, predictor, bilinear, anchors, positives


def run_held_out_query_state(world, anchors, positives, seed=42):
    """
    Task 3: Held-out query-state evaluation.

    Partition states 80/20. During training, only use 80% states as anchors
    (query side). Targets can be any state. Evaluate on both sets separately.
    """
    print(f"\n{'=' * 70}")
    print("TASK 3: HELD-OUT QUERY-STATE EVALUATION")
    print(f"{'=' * 70}")

    N = len(world.all_states)
    rng = np.random.RandomState(seed)
    all_indices = np.arange(N)
    rng.shuffle(all_indices)
    split = int(N * 0.8)
    train_anchors_set = set(all_indices[:split].tolist())
    heldout_anchors_set = set(all_indices[split:].tolist())
    print(f"Train anchors: {len(train_anchors_set)}, Held-out anchors: {len(heldout_anchors_set)}")

    # Build training pairs: only use train-anchor states as queries
    associations = world._associations
    assoc_list = [(i, j, s) for (i, j), s in associations.items() if s >= 0.2]
    rng.shuffle(assoc_list)

    filtered_pairs = []
    for i, j, s in assoc_list:
        # Only include pair if the anchor is in train set
        if i in train_anchors_set:
            filtered_pairs.append((i, j))
        if j in train_anchors_set:
            filtered_pairs.append((j, i))
    rng.shuffle(filtered_pairs)
    filtered_pairs = filtered_pairs[:200000]

    anchor_embs = np.stack([world.all_embeddings[p[0]] for p in filtered_pairs])
    positive_embs = np.stack([world.all_embeddings[p[1]] for p in filtered_pairs])

    print(f"Training pairs (anchor-filtered): {len(filtered_pairs)}")
    # Verify no held-out anchors leaked in
    anchor_indices_used = {p[0] for p in filtered_pairs}
    leaked = anchor_indices_used & heldout_anchors_set
    print(f"Held-out anchors in training queries: {len(leaked)} (should be 0)")

    # Train
    print("\nTraining on anchor-filtered pairs...")
    predictor, loss, train_time = train_model(anchor_embs, positive_embs, seed=seed)

    bilinear = LearnedBilinearBaseline(128, seed=seed + 81)
    bilinear.train(anchor_embs, positive_embs, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

    # Evaluate on train-anchor queries
    print("\n--- Evaluating on TRAIN-anchor queries ---")
    train_results = run_full_faithfulness(
        world, predictor, bilinear,
        query_filter=train_anchors_set,
        label="Train-anchor queries (80%)")

    # Evaluate on held-out-anchor queries
    print("\n--- Evaluating on HELD-OUT-anchor queries ---")
    heldout_results = run_full_faithfulness(
        world, predictor, bilinear,
        query_filter=heldout_anchors_set,
        label="Held-out-anchor queries (20%)")

    return {
        'train_anchors': train_results,
        'heldout_anchors': heldout_results,
        'n_train_anchors': len(train_anchors_set),
        'n_heldout_anchors': len(heldout_anchors_set),
        'n_training_pairs': len(filtered_pairs),
        'loss': float(loss),
        'train_time': float(train_time),
        'leaked_anchors': len(leaked),
    }


def run_transitive_and_decay(world, predictor, bilinear):
    """Pull transitive chain and decay ablation numbers from the existing model."""
    print(f"\n{'=' * 70}")
    print("LEGACY METRICS: Transitive Chains + Decay Ablation")
    print(f"{'=' * 70}")

    evaluator = BenchmarkEvaluator(world, predictor, bilinear)
    t2 = evaluator.test_transitive_association()
    t3 = evaluator.test_decay_ablation()

    return {
        'transitive': t2.details,
        'transitive_headline': {
            'cross_room_2hop_R@20': t2.predictor_score,
            'cosine': t2.cosine_baseline_score,
        },
        'decay': {
            'with_decay': t3.predictor_score,
            'without_decay': t3.cosine_baseline_score,
        },
    }


def main():
    t_total = time.time()

    print("=" * 70)
    print("PAPER REVIEWER EXPERIMENTS")
    print("Tasks 2-5 + transitive/decay numbers")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    all_results = {}

    # =====================================================================
    # TASK 2: Multi-seed runs (seeds 42, 123, 456)
    # + TASK 4: P@1/P@5 (included via k_values=[1,5,10,20,50])
    # + TASK 5: Bilinear (included in all faithfulness evals)
    # =====================================================================
    seeds = [42, 123, 456]
    seed_results = {}

    for seed in seeds:
        res, world, predictor, bilinear, anchors, positives = run_single_seed(seed)
        seed_results[seed] = res

        # Run transitive + decay on first seed only
        if seed == seeds[0]:
            legacy = run_transitive_and_decay(world, predictor, bilinear)
            all_results['legacy_metrics'] = legacy

    all_results['multi_seed'] = seed_results

    # Compute mean +/- CI
    print(f"\n{'=' * 70}")
    print("MULTI-SEED SUMMARY")
    print(f"{'=' * 70}")

    metrics_to_summarise = [
        ('CBR@20', lambda r: r['cross_boundary_recall'][20]['predictor']),
        ('AP@1', lambda r: r['association_precision'][1]['predictor']),
        ('AP@5', lambda r: r['association_precision'][5]['predictor']),
        ('AP@20', lambda r: r['association_precision'][20]['predictor']),
        ('AUC (all)', lambda r: r['discrimination_auc_all']['predictor']),
        ('AUC (x-room)', lambda r: r['discrimination_auc_xroom']['predictor']),
        ('Specificity@20', lambda r: r['specificity']['predictor']),
        ('MRR', lambda r: r['mrr']['predictor']),
    ]

    summary_table = {}
    print(f"\n{'Metric':<20s}", end="")
    for seed in seeds:
        print(f"  {'Seed '+str(seed):>10s}", end="")
    print(f"  {'Mean +/- SE':>14s}")
    print("-" * 75)

    for name, extractor in metrics_to_summarise:
        vals = [extractor(seed_results[s]) for s in seeds]
        mean = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        ci95 = 1.96 * se
        summary_table[name] = {'mean': mean, 'se': se, 'ci95': ci95, 'per_seed': vals}
        print(f"{name:<20s}", end="")
        for v in vals:
            print(f"  {v:>10.4f}", end="")
        print(f"  {mean:.4f} +/- {ci95:.4f}")

    all_results['summary'] = summary_table

    # =====================================================================
    # TASK 3: Held-out query-state evaluation
    # =====================================================================
    # Use world from seed=42 run
    np.random.seed(42)
    config = WorldConfig(num_trajectories=NUM_TRAJECTORIES, seed=42)
    world = SyntheticWorld(config)
    world.generate_trajectories()
    world.compute_association_ground_truth()
    anchors, positives, _ = world.get_training_pairs(max_pairs=200000)

    heldout = run_held_out_query_state(world, anchors, positives, seed=42)
    all_results['held_out_query_state'] = heldout

    # Print comparison
    print(f"\n{'=' * 70}")
    print("HELD-OUT QUERY-STATE COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Metric':<20s}  {'Train-Anchor':>14s}  {'Held-Out':>14s}  {'Cosine':>10s}")
    print("-" * 65)

    tr = heldout['train_anchors']
    ho = heldout['heldout_anchors']
    comparisons = [
        ('CBR@20', tr['cross_boundary_recall'][20]['predictor'],
         ho['cross_boundary_recall'][20]['predictor'],
         ho['cross_boundary_recall'][20]['cosine']),
        ('Disc-AUC (x-room)', tr['discrimination_auc_xroom']['predictor'],
         ho['discrimination_auc_xroom']['predictor'],
         ho['discrimination_auc_xroom']['cosine']),
        ('Specificity@20', tr['specificity']['predictor'],
         ho['specificity']['predictor'],
         ho['specificity']['cosine']),
        ('AP@20', tr['association_precision'][20]['predictor'],
         ho['association_precision'][20]['predictor'],
         ho['association_precision'][20]['cosine']),
        ('MRR', tr['mrr']['predictor'],
         ho['mrr']['predictor'],
         ho['mrr']['cosine']),
        ('NearestRank(med)', tr['nearest_rank']['predictor_median'],
         ho['nearest_rank']['predictor_median'],
         ho['nearest_rank']['cosine_median']),
    ]
    for name, tv, hv, cv in comparisons:
        print(f"{name:<20s}  {tv:>14.4f}  {hv:>14.4f}  {cv:>10.4f}")

    # =====================================================================
    # TASK 2 (supplement): Temporal shuffle on seed 123
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("TEMPORAL SHUFFLE ON SEED 123")
    print(f"{'=' * 70}")
    import copy
    np.random.seed(42)
    config123 = WorldConfig(num_trajectories=NUM_TRAJECTORIES, seed=42)
    world123 = SyntheticWorld(config123)
    world123.generate_trajectories()
    world123.compute_association_ground_truth()

    shuffled_world = copy.deepcopy(world123)
    rng_shuf = np.random.RandomState(999)
    for traj in shuffled_world.trajectories:
        n = len(traj.states)
        perm = rng_shuf.permutation(n)
        orig = [traj.states[i] for i in range(n)]
        for new_pos, old_pos in enumerate(perm):
            traj.states[new_pos] = orig[old_pos]
            traj.states[new_pos].timestep = new_pos
    shuffled_world._associations = None
    shuffled_world.compute_association_ground_truth()

    shuf_a, shuf_p, _ = shuffled_world.get_training_pairs(max_pairs=200000)
    shuf_pred, shuf_loss, shuf_time = train_model(shuf_a, shuf_p, seed=123)
    shuf_bil = LearnedBilinearBaseline(128, seed=204)
    shuf_bil.train(shuf_a, shuf_p, epochs=200, batch_size=BATCH_SIZE, lr=1e-3)

    # Evaluate shuffled model on REAL world associations
    shuf_results = run_full_faithfulness(
        world123, shuf_pred, shuf_bil,
        label="Temporal Shuffle (seed 123, eval on real)")
    all_results['shuffle_seed123'] = shuf_results

    # =====================================================================
    # FINAL SAVE
    # =====================================================================
    total_time = time.time() - t_total

    output_dir = Path("results") / "paper_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results['total_time'] = total_time
    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"ALL TASKS COMPLETE")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Results saved to: {output_dir / 'results.json'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
