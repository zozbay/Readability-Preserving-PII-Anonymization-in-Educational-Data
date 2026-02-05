# categorical_entity_level.py

"""
ENTITY-LEVEL categorical evaluation for all methods across all folds.
Uses seqeval to match cross_validation_results.json metrics.
This generates the data for LaTeX tables in the thesis.

Computes and saves precision, recall, F1, support, and detection counts.
"""

import json
import os
import sys

# Adding project root to path so that evaluation scripts be ran from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from seqeval.metrics import classification_report
from pipeline.anonymizer import run_method
from methods import custom_deberta
from collections import defaultdict


"""
    Configurations
"""
NUM_FOLDS = 5
METHODS = ["rule", "presidio", "customdeberta"]
METHOD_NAMES = {"rule": "Rule-based", "presidio": "Presidio", "customdeberta": "DeBERTa-v3"}
OUTPUT_FILE = "category_entity_level_results.json"

# PII categories
CATEGORIES = ['NAME_STUDENT', 'EMAIL', 'PHONE_NUM', 'ID_NUM', 
              'STREET_ADDRESS', 'USERNAME', 'URL_PERSONAL']


"""
    Helper to extract entity level metrics per category.
    Uses seqeval which evaluates complete entity spans.
"""
def extract_category_metrics(true_labels, pred_labels, categories):
    
    # Get classification report from seqeval
    report = classification_report(
        true_labels,
        pred_labels,
        output_dict=True,
        zero_division=0
    )
    
    # Extract metrics for each category
    results = {}
    
    for category in categories:
        # seqeval uses category names directly (without B-/I- prefix)
        if category in report:
            cat_report = report[category]
            
            precision = cat_report["precision"]
            recall = cat_report["recall"]
            f1 = cat_report["f1-score"]
            support = int(cat_report["support"])  # Number of true entities
            
            # Calculate derived metrics from P, R, Support
            # TP = recall * support (entities correctly detected)
            true_positives = int(round(recall * support)) if support > 0 else 0
            
            # predicted = TP / precision (total entities predicted for this category)
            # Handle division by zero
            if precision > 0 and true_positives > 0:
                predicted = int(round(true_positives / precision))
            else:
                predicted = 0
            
            # FP = predicted - TP
            false_positives = max(0, predicted - true_positives)
            
            # FN = support - TP
            false_negatives = max(0, support - true_positives)
            
            results[category] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "predicted": predicted,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            }
        else:
            # Category not detected at all
            # Count support from true_labels
            support_count = 0
            for doc in true_labels:
                for label in doc:
                    if label == f"B-{category}":
                        support_count += 1
            
            results[category] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": support_count,
                "predicted": 0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": support_count
            }
    
    return results


def banner(text):
    print("\n" + "=" * 90)
    print(text.center(90))
    print("=" * 90 + "\n")


"""
    Run entity-level categorical evaluation across all folds and methods.    
"""
def run_per_category_evaluation():
    banner("PER-CATEGORY ENTITY-LEVEL EVALUATION STARTED")
    print("Using seqeval for entity-level metrics (matching cross_validation_results.json)")
    print("This generates data for LaTeX tables in the thesis.\n")
    
    # Store results
    all_results = {
        "per_fold": {},
        "summary": {},
        "evaluation_type": "entity_level",
        "note": "Uses seqeval - complete entity spans must match"
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
                    custom_deberta.MODEL_PATH = os.path.join(
                        custom_deberta.PROJECT_ROOT, 
                        "models", 
                        f"deberta_fold_{fold_idx}"
                    )
                    print(f"  Model: {custom_deberta.MODEL_PATH}")
                
                # Run method - returns List[List[str]]
                true_labels, pred_labels, _, _ = run_method(method_key, val_file)
                
                # Extract entity-level metrics
                category_metrics = extract_category_metrics(true_labels, pred_labels, CATEGORIES)
                
                # Store
                all_results["per_fold"][fold_key][method_key] = category_metrics
                
                # Print summary
                total_support = sum(m["support"] for m in category_metrics.values())
                total_detected = sum(m["predicted"] for m in category_metrics.values())
                total_correct = sum(m["true_positives"] for m in category_metrics.values())
                
                overall_p = total_correct / total_detected * 100 if total_detected > 0 else 0
                overall_r = total_correct / total_support * 100 if total_support > 0 else 0
                overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
                
                print(f"  ✓ Processed {total_support} PII entities")
                print(f"    Detected: {total_detected}, Correct: {total_correct}")
                print(f"    Overall P={overall_p:.1f}%, R={overall_r:.1f}%, F1={overall_f1:.1f}%\n")
                
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                import traceback
                traceback.print_exc()
                all_results["per_fold"][fold_key][method_key] = None
    
    
    """
    Compute averages across folds
    """
    banner("COMPUTING AVERAGES")
    
    for method_key in METHODS:
        method_name = METHOD_NAMES[method_key]
        print(f"\n{method_name}:")
        print("-" * 80)
        
        all_results["summary"][method_key] = {}
        
        for category in CATEGORIES:
            # Collect values across folds
            precisions = []
            recalls = []
            f1s = []
            supports = []
            predicteds = []
            tps = []
            fps = []
            fns = []
            
            for fold_idx in range(NUM_FOLDS):
                fold_key = f"fold_{fold_idx}"
                fold_data = all_results["per_fold"][fold_key].get(method_key)
                
                if fold_data and category in fold_data:
                    cat_data = fold_data[category]
                    precisions.append(cat_data["precision"])
                    recalls.append(cat_data["recall"])
                    f1s.append(cat_data["f1"])
                    supports.append(cat_data["support"])
                    predicteds.append(cat_data["predicted"])
                    tps.append(cat_data["true_positives"])
                    fps.append(cat_data["false_positives"])
                    fns.append(cat_data["false_negatives"])
            
            # Compute statistics
            if precisions:
                all_results["summary"][method_key][category] = {
                    "precision": {
                        "mean": float(np.mean(precisions)),
                        "std": float(np.std(precisions)),
                        "values": [float(p) for p in precisions]
                    },
                    "recall": {
                        "mean": float(np.mean(recalls)),
                        "std": float(np.std(recalls)),
                        "values": [float(r) for r in recalls]
                    },
                    "f1": {
                        "mean": float(np.mean(f1s)),
                        "std": float(np.std(f1s)),
                        "values": [float(f) for f in f1s]
                    },
                    "support": {
                        "total": int(np.sum(supports)),
                        "per_fold": [int(s) for s in supports]
                    },
                    "predicted": {
                        "total": int(np.sum(predicteds)),
                        "per_fold": [int(p) for p in predicteds]
                    },
                    "true_positives": {
                        "total": int(np.sum(tps)),
                        "per_fold": [int(t) for t in tps]
                    },
                    "false_positives": {
                        "total": int(np.sum(fps)),
                        "per_fold": [int(f) for f in fps]
                    },
                    "false_negatives": {
                        "total": int(np.sum(fns)),
                        "per_fold": [int(f) for f in fns]
                    }
                }
                
                # Print
                print(f"  {category:<20} F1={np.mean(f1s)*100:5.1f}% "
                      f"(Support: {np.sum(supports)}, "
                      f"Detected: {np.sum(predicteds)}, "
                      f"Correct: {np.sum(tps)})")
    
    """
    Save results to file
    """
    banner("SAVING RESULTS")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Results saved to: {OUTPUT_FILE}")
    print(f"  File size: {len(json.dumps(all_results)) / 1024:.1f} KB")
    
    # Print file structure
    print("\nFile structure:")
    print("  evaluation_type: 'entity_level'")
    print("  per_fold/")
    print("    fold_0/")
    print("      rule/")
    print("        NAME_STUDENT: {precision, recall, f1, support, predicted, ...}")
    print("        EMAIL: {...}")
    print("        ...")
    print("      presidio/")
    print("      customdeberta/")
    print("    fold_1/")
    print("    ...")
    print("  summary/")
    print("    rule/")
    print("      NAME_STUDENT: {mean, std, total_support, ...}")
    print("      ...")
    
    banner("EVALUATION COMPLETE")
    
    return all_results



if __name__ == "__main__":
    results = run_per_category_evaluation()
    
    # Quick summary
    print("\nQUICK SUMMARY (ENTITY-LEVEL METRICS)")
    print("="*80)
    print("These metrics match the bar chart (cross_validation_results.json)")
    print("="*80)
    
    for method_key in METHODS:
        method_name = METHOD_NAMES[method_key]
        print(f"\n{method_name}:")
        
        if method_key in results["summary"]:
            total_support = sum(
                results["summary"][method_key][cat]["support"]["total"]
                for cat in CATEGORIES
                if cat in results["summary"][method_key]
            )
            total_detected = sum(
                results["summary"][method_key][cat]["predicted"]["total"]
                for cat in CATEGORIES
                if cat in results["summary"][method_key]
            )
            total_correct = sum(
                results["summary"][method_key][cat]["true_positives"]["total"]
                for cat in CATEGORIES
                if cat in results["summary"][method_key]
            )
            
            overall_p = total_correct / total_detected * 100 if total_detected > 0 else 0
            overall_r = total_correct / total_support * 100 if total_support > 0 else 0
            overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
            
            print(f"  Total PII entities in data: {total_support}")
            print(f"  Total detected: {total_detected}")
            print(f"  Correct detections: {total_correct}")
            print(f"  Precision: {overall_p:.1f}%")
            print(f"  Recall: {overall_r:.1f}%")
            print(f"  F1-Score: {overall_f1:.1f}%")
    
    print("\n" + "="*80)
    print("="*80)