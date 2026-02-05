# main.py

"""
    This is the script to run anonymization and evaluation.
    It supports multiple PII detection methods and computes NER metrics and readability changes.
"""

import sys
import json
from pipeline.anonymizer import run_method
from pipeline.evaluator import evaluate_predictions
from pipeline.readability import compute_readability


RESULTS_FILE = "results.json"

def average_readability_change(originals, anonymized):
    deltas = []
    for orig, anon in zip(originals, anonymized):
        orig_scores = compute_readability(orig)
        anon_scores = compute_readability(anon)
        delta = {k: anon_scores[k] - orig_scores[k] for k in orig_scores}
        deltas.append(delta)

    avg = {k: sum(d[k] for d in deltas) / len(deltas) for k in deltas[0]}
    return avg


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "rule"
    input_file = sys.argv[2] if len(sys.argv) > 2 else "data/train.json"

    true_labels, predicted_labels, originals, anonymized = run_method(method, input_file)

    result = {}

    if true_labels:
        metrics = evaluate_predictions(true_labels, predicted_labels)
        
        if "error" in metrics:
            print(f"\n❌ ERROR: {metrics['error']}")
            sys.exit(1)

        result["precision"] = metrics["precision"]
        result["recall"] = metrics["recall"]
        result["f1"] = metrics["f1"]
        result["support"] = metrics["support"]

        print("\n=== NER METRICS (seqeval) ===")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1:        {metrics['f1']:.4f}")
        print(f"Support:   {metrics['support']} entities")
        print()

    # Readability
    result["readability_change"] = average_readability_change(originals, anonymized)

    # Preview examples
    print("=== READABILITY EXAMPLES ===")
    for i in range(min(3, len(originals))):
        print(f"\nExample {i+1}:")
        print("  Original:  ", compute_readability(originals[i]))
        print("  Anonymized:", compute_readability(anonymized[i]))

    # Save results
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}

    all_results[method] = result

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved for '{method}' in {RESULTS_FILE}")