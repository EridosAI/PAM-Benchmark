"""
Analysis script to compare results across different benchmark runs
"""

import json
import sys
from pathlib import Path
import numpy as np


def load_results(result_path):
    """Load results from a JSON file."""
    with open(result_path, 'r') as f:
        return json.load(f)


def print_comparison(results_dict):
    """
    Print a comparison table of results across multiple runs.

    results_dict: {run_name: results_json, ...}
    """
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    # Extract test names from first result
    first_result = list(results_dict.values())[0]
    test_names = [r['test_name'] for r in first_result['results']]

    for test_name in test_names:
        print(f"\n{test_name.upper().replace('_', ' ')}")
        print("-" * 80)

        # Header
        print(f"{'Run Name':<30} {'Predictor':<12} {'Cosine':<12} {'Bilinear':<12} {'Loss':<12}")
        print("-" * 80)

        for run_name, results in results_dict.items():
            test_result = next((r for r in results['results'] if r['test_name'] == test_name), None)
            if test_result:
                pred = test_result['predictor']
                cos = test_result['cosine']
                bil = test_result['bilinear']
                loss = results['training']['predictor_best_loss']

                print(f"{run_name:<30} {pred:<12.4f} {cos:<12.4f} {bil:<12.4f} {loss:<12.4f}")

        # Print detailed breakdown for key tests
        if test_name == 'association_vs_similarity':
            print("\n  Detailed Recall@K (Predictor only):")
            print(f"  {'Run Name':<30} {'R@5':<10} {'R@10':<10} {'R@20':<10} {'R@50':<10} {'MRR':<10}")
            for run_name, results in results_dict.items():
                test_result = next((r for r in results['results'] if r['test_name'] == test_name), None)
                if test_result:
                    details = test_result['details']
                    if 'recall_at_k' in details:
                        r_at_k = details['recall_at_k']
                        r5 = r_at_k.get('5', {}).get('predictor', 0)
                        r10 = r_at_k.get('10', {}).get('predictor', 0)
                        r20 = r_at_k.get('20', {}).get('predictor', 0)
                        r50 = r_at_k.get('50', {}).get('predictor', 0)
                        mrr = details.get('mrr', {}).get('predictor', 0)
                        print(f"  {run_name:<30} {r5:<10.4f} {r10:<10.4f} {r20:<10.4f} {r50:<10.4f} {mrr:<10.4f}")

        elif test_name == 'decay_ablation':
            print("\n  With decay vs Without decay:")
            print(f"  {'Run Name':<30} {'With Decay':<15} {'Without Decay':<15} {'Improvement':<15}")
            for run_name, results in results_dict.items():
                test_result = next((r for r in results['results'] if r['test_name'] == test_name), None)
                if test_result:
                    with_decay = test_result['predictor']
                    without_decay = test_result['cosine']
                    improvement = ((with_decay - without_decay) / (without_decay + 1e-6)) * 100
                    print(f"  {run_name:<30} {with_decay:<15.4f} {without_decay:<15.4f} {improvement:<+15.1f}%")

        elif test_name == 'creative_bridging':
            print("\n  Diagnostics:")
            print(f"  {'Run Name':<30} {'Hop1 finds bridge':<20} {'Hop2 finds target':<20}")
            for run_name, results in results_dict.items():
                test_result = next((r for r in results['results'] if r['test_name'] == test_name), None)
                if test_result:
                    details = test_result.get('details', {})
                    diag = details.get('diagnostics', {})
                    h1 = diag.get('hop1_finds_bridge_pct', 0)
                    h2 = diag.get('hop2_finds_target_pct', 0)
                    print(f"  {run_name:<30} {h1:<20.1f}% {h2:<20.1f}%")

    # Training summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"{'Run Name':<30} {'Epochs':<10} {'Hidden':<10} {'Best Loss':<12} {'LR Range':<20} {'Temp Range':<20}")
    print("-" * 80)

    for run_name, results in results_dict.items():
        training = results['training']
        epochs = training.get('num_epochs', 0)
        hidden = training.get('hidden_dim', 512)
        best_loss = training['predictor_best_loss']

        lr_start = training.get('lr_start', 3e-4)
        lr_end = training.get('lr_end', 3e-4)
        temp_start = training.get('temp_start', 0.07)
        temp_end = training.get('temp_end', 0.07)

        lr_range = f"{lr_start:.1e} → {lr_end:.1e}" if lr_start != lr_end else f"{lr_start:.1e}"
        temp_range = f"{temp_start:.3f} → {temp_end:.3f}" if temp_start != temp_end else f"{temp_start:.3f}"

        print(f"{run_name:<30} {epochs:<10} {hidden:<10} {best_loss:<12.4f} {lr_range:<20} {temp_range:<20}")


def main():
    results_dir = Path('results')

    if len(sys.argv) > 1:
        # Specific result files provided
        result_files = [Path(p) for p in sys.argv[1:]]
    else:
        # Find all result files
        result_files = list(results_dir.glob('**/results.json'))
        result_files.extend(list(results_dir.glob('benchmark_results.json')))

    if not result_files:
        print("No result files found.")
        print("Usage: python analyze_results.py [result1.json result2.json ...]")
        return

    print(f"Found {len(result_files)} result file(s)")

    results_dict = {}
    for path in result_files:
        try:
            results = load_results(path)
            run_name = results.get('run_name', path.parent.name if path.parent.name != 'results' else 'baseline')
            timestamp = results.get('timestamp', '')
            full_name = f"{run_name}_{timestamp}" if timestamp else run_name
            results_dict[full_name] = results
        except Exception as e:
            print(f"Error loading {path}: {e}")

    if results_dict:
        print_comparison(results_dict)
    else:
        print("No valid result files could be loaded.")


if __name__ == '__main__':
    main()
