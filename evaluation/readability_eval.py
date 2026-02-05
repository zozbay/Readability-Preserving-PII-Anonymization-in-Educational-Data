# evaluate_readability.py

"""
Comprehensive readability evaluation for all methods across all folds.
Computes Δ values between anonymized and original versions for Flesch, FK Grade, and Gunning Fog.

"""

import json
import os
import sys

# Adding project root to path so that evaluation scripts be ran from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pipeline.anonymizer import run_method
from pipeline.readability import compute_readability


"""
    Configurations
"""
NUM_FOLDS = 5
METHODS = ["rule", "presidio", "customdeberta"]
METHOD_NAMES = {
    "rule": "Rule-based",
    "presidio": "Presidio",
    "customdeberta": "DeBERTa-v3"
}
OUTPUT_FILE = "readability_results.json"



"""
    Compute Δ readability for each document.
    Returns list of deltas and summary statistics.
"""
def compute_readability_changes(originals, anonymized):
    deltas = []
    
    for orig, anon in zip(originals, anonymized):
        # Compute scores
        orig_scores = compute_readability(orig)
        anon_scores = compute_readability(anon)
        
        # Calculate delta between anonymized and original
        delta = {
            "flesch": anon_scores["flesch"] - orig_scores["flesch"],
            "fk_grade": anon_scores["fk_grade"] - orig_scores["fk_grade"],
            "fog": anon_scores["fog"] - orig_scores["fog"]
        }
        deltas.append(delta)
    
    # Compute summary statistics
    metrics = ["flesch", "fk_grade", "fog"]
    summary = {}
    
    for metric in metrics:
        values = [d[metric] for d in deltas]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "abs_mean": float(np.mean(np.abs(values))),  # Mean absolute change
            "values": [float(v) for v in values]  # All individual deltas
        }
    
    return summary

def banner(text):
    print("\n" + "=" * 90)
    print(text.center(90))
    print("=" * 90 + "\n")


"""
    Main evaluation loop
"""
def run_readability_evaluation():
    banner("READABILITY EVALUATION STARTED")
    
    # Store results
    all_results = {
        "per_fold": {},
        "summary": {}
    }
    
    # Evaluate each fold
    for fold_idx in range(NUM_FOLDS):
        fold_key = f"fold_{fold_idx}"
        all_results["per_fold"][fold_key] = {}
        
        val_file = f"data/fold_{fold_idx}_val.json"
        
        banner(f"FOLD {fold_idx}")
        print(f"Validation file: {val_file}\n")
        
        # Evaluate each method
        for method_key in METHODS:
            method_name = METHOD_NAMES[method_key]
            
            print(f"Running {method_name}...")
            
            try:
                # Update model path for DeBERTa
                if method_key == "customdeberta":
                    from methods import custom_deberta
                    custom_deberta.MODEL_PATH = os.path.join(custom_deberta.PROJECT_ROOT, "models", f"deberta_fold_{fold_idx}")
                    print(f"  Model: {custom_deberta.MODEL_PATH}")
                
                # Run method
                _, _, originals, anonymized = run_method(method_key, val_file)
                
                # Compute readability changes
                readability_summary = compute_readability_changes(originals, anonymized)
                
                # Store
                all_results["per_fold"][fold_key][method_key] = readability_summary
                
                # Print summary
                print(f"  ✓ Processed {len(originals)} documents")
                print(f"    Δ Flesch:   {readability_summary['flesch']['mean']:+.3f} ± {readability_summary['flesch']['std']:.3f}")
                print(f"    Δ FK Grade: {readability_summary['fk_grade']['mean']:+.3f} ± {readability_summary['fk_grade']['std']:.3f}")
                print(f"    Δ Fog:      {readability_summary['fog']['mean']:+.3f} ± {readability_summary['fog']['std']:.3f}\n")
                
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                all_results["per_fold"][fold_key][method_key] = None
    
    """
    Compute overall averages across folds"""
    banner("COMPUTING AVERAGES")
    
    metrics = ["flesch", "fk_grade", "fog"]
    
    for method_key in METHODS:
        method_name = METHOD_NAMES[method_key]
        print(f"\n{method_name}:")
        print("-" * 80)
        
        all_results["summary"][method_key] = {}
        
        for metric in metrics:
            # Collect values across folds
            means = []
            stds = []
            all_deltas = []
            
            for fold_idx in range(NUM_FOLDS):
                fold_key = f"fold_{fold_idx}"
                fold_data = all_results["per_fold"][fold_key].get(method_key)
                
                if fold_data and metric in fold_data:
                    means.append(fold_data[metric]["mean"])
                    stds.append(fold_data[metric]["std"])
                    all_deltas.extend(fold_data[metric]["values"])
            
            # Compute aggregate statistics
            if means:
                all_results["summary"][method_key][metric] = {
                    "mean_of_means": float(np.mean(means)),
                    "std_of_means": float(np.std(means)),
                    "pooled_std": float(np.mean(stds)),
                    "overall_mean": float(np.mean(all_deltas)),
                    "overall_std": float(np.std(all_deltas)),
                    "abs_mean": float(np.mean(np.abs(all_deltas))),
                    "min": float(np.min(all_deltas)),
                    "max": float(np.max(all_deltas)),
                    "per_fold_means": [float(m) for m in means]
                }
                
                # Print
                print(f"  {metric:12} Δ = {np.mean(means):+.3f} ± {np.std(means):.3f}  (|Δ| = {np.mean(np.abs(all_deltas)):.3f})")
    
    """
     Save results to JSON
     
    """
    banner("SAVING RESULTS")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Results saved to: {OUTPUT_FILE}")
    print(f"  File size: {len(json.dumps(all_results)) / 1024:.1f} KB")
    
    # Print file structure
    print("\nFile structure:")
    print("  per_fold/")
    print("    fold_0/")
    print("      rule/")
    print("        flesch: {mean, std, min, max, values}")
    print("        fk_grade: {...}")
    print("        fog: {...}")
    print("      presidio/")
    print("      customdeberta/")
    print("    fold_1/")
    print("    ...")
    print("  summary/")
    print("    rule/")
    print("      flesch: {mean_of_means, std, overall_mean, ...}")
    print("      ...")
    
    banner("EVALUATION COMPLETE")
    
    return all_results


if __name__ == "__main__":
    results = run_readability_evaluation()
    
    # Summary table
    print("\nREADABILITY PRESERVATION SUMMARY")
    print("="*80)
    print(f"{'Method':<20} {'Δ Flesch':<15} {'Δ FK Grade':<15} {'Δ Fog':<15}")
    print("-"*80)
    
    for method_key in METHODS:
        method_name = METHOD_NAMES[method_key]
        
        if method_key in results["summary"]:
            flesch = results["summary"][method_key]["flesch"]["overall_mean"]
            fk = results["summary"][method_key]["fk_grade"]["overall_mean"]
            fog = results["summary"][method_key]["fog"]["overall_mean"]
            
            print(f"{method_name:<20} {flesch:+.3f}           {fk:+.3f}           {fog:+.3f}")
    
    print("-"*80)
    print("\nInterpretation:")
    print("  Δ near 0 = better readability preservation")
    