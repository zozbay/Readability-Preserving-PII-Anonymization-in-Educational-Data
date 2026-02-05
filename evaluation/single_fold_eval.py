# single_fold_eval.py

"""
You can use this to evaluate any method (rule-based, presidio or DeBERTa) on a specific fold.

Usage:
    python single_fold_eval.py <fold_idx>
    
Example:
    python single_fold_eval.py 3
"""

import os
import sys

# Adding project root to path so that evaluation scripts be ran from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
from pipeline.anonymizer import run_method
from pipeline.evaluator import evaluate_predictions
from pipeline.readability import compute_readability
from methods import custom_deberta


"""
    Configurations
"""
FOLD_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0

VAL_DATA_PATH = f"data/fold_{FOLD_IDX}_val.json"
OUTPUT_FILE = f"fold_{FOLD_IDX}_comparison.json"

METHODS = ["rule", "presidio", "customdeberta"]


"""
    Convert numpy types to JSON serializable types.
"""
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)  # Convert numpy bool to Python bool
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


"""
    Compute average readability change across all documents
"""
def average_readability_change(originals, anonymized):
    if not originals or not anonymized:
        return {}
    
    deltas = []
    for orig, anon in zip(originals, anonymized):
        orig_scores = compute_readability(orig)
        anon_scores = compute_readability(anon)
        delta = {k: anon_scores[k] - orig_scores[k] for k in orig_scores}
        deltas.append(delta)
    
    avg = {k: sum(d[k] for d in deltas) / len(deltas) for k in deltas[0]}
    return avg


"""
    Evaluate a single method on the validation data.
"""
def evaluate_method(method_name, val_file):
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {method_name.upper()}")
    print(f"{'='*60}")
    
    # Update DeBERTa model path for this fold
    if method_name == "customdeberta":
        custom_deberta.MODEL_PATH = os.path.join(custom_deberta.PROJECT_ROOT, "models", f"deberta_fold_{FOLD_IDX}")
        print(f"Using model: {custom_deberta.MODEL_PATH}")
    
    # Run method
    true_labels, pred_labels, originals, anonymized = run_method(method_name, val_file)
    
    # Evaluate detection accuracy
    if not true_labels:
        print("No gold labels found!")
        return None
    
    metrics = evaluate_predictions(true_labels, pred_labels)
    
    if "error" in metrics:
        print(f"ERROR: {metrics['error']}")
        return None
    
    # Compute readability change
    readability = average_readability_change(originals, anonymized)
    
    # Print results
    print(f"\n--- Detection Metrics ---")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Support:   {metrics['support']} entities")
    
    print(f"\n--- Readability Change (avg) ---")
    print(f"Flesch:    {readability.get('flesch', 0):.4f}")
    print(f"FK Grade:  {readability.get('fk_grade', 0):.4f}")
    print(f"FOG:       {readability.get('fog', 0):.4f}")
    
    # Convert all values to native Python types for JSON serialization
    return convert_to_serializable({
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "support": metrics["support"],
        "readability_change": readability,
        "detailed": metrics.get("detailed", {})
    })


"""
    Print comparison table of all methods
"""
def print_comparison_table(results):
    print(f"\n\n{'='*80}")
    print(f"FOLD {FOLD_IDX} - COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Method':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Support':<10}")
    print(f"{'-'*80}")
    
    # Rows
    for method_name, result in results.items():
        if result is None:
            print(f"{method_name:<20} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'N/A':<10}")
        else:
            print(f"{method_name:<20} "
                  f"{result['precision']:<12.4f} "
                  f"{result['recall']:<12.4f} "
                  f"{result['f1']:<12.4f} "
                  f"{result['support']:<10}")
    
    print(f"{'-'*80}\n")
    
    # Readability comparison
    print(f"{'Method':<20} {'Delta Flesch':<15} {'Delta FK Grade':<15} {'Delta FOG':<15}")
    print(f"{'-'*80}")
    
    for method_name, result in results.items():
        if result is None:
            print(f"{method_name:<20} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15}")
        else:
            read = result['readability_change']
            print(f"{method_name:<20} "
                  f"{read.get('flesch', 0):<15.4f} "
                  f"{read.get('fk_grade', 0):<15.4f} "
                  f"{read.get('fog', 0):<15.4f}")
    
    print(f"{'-'*80}\n")
    
    # Find best F1
    best_method = None
    best_f1 = -1
    for method_name, result in results.items():
        if result and result['f1'] > best_f1:
            best_f1 = result['f1']
            best_method = method_name
    
    if best_method:
        print(f"Best F1 Score: {best_method.upper()} ({best_f1:.4f})")
    
    print(f"\n{'='*80}\n")



"""
    Main function to run evaluation for one single fold
"""
def main():
    print(f"\n{'#'*80}")
    print(f"#{'':^78}#")
    print(f"#{'FOLD ' + str(FOLD_IDX) + ' EVALUATION - ALL METHODS':^78}#")
    print(f"#{'':^78}#")
    print(f"{'#'*80}\n")
    
    print(f"Validation data: {VAL_DATA_PATH}")
    
    # Evaluate all methods
    results = {}
    
    for method_name in METHODS:
        try:
            result = evaluate_method(method_name, VAL_DATA_PATH)
            results[method_name] = result
        except Exception as e:
            print(f"\n ERROR evaluating {method_name}: {e}")
            import traceback
            traceback.print_exc()
            results[method_name] = None
    
    # Print comparison
    print_comparison_table(results)
    
    # Save results
    output_data = {
        "fold": FOLD_IDX,
        "validation_file": VAL_DATA_PATH,
        "methods": results
    }
    
    # Ensure everything is JSON serializable
    output_data = convert_to_serializable(output_data)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {OUTPUT_FILE}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python single_fold_eval.py <fold_idx>")
        print("Example: python single_fold_eval.py 3")
        print("Folds available: 0, 1, 2, 3, 4\n")
        sys.exit(1)
    
    main()