#categorical_token_level.py

"""
Detailed categorical evaluation for all methods across all folds.
Computes and saves precision, recall, F1, support, and detection counts.
"""

import json
import os
import sys

# Adding project root to path so that evaluation scripts be ran from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from pipeline.anonymizer import run_method
from methods import custom_deberta
from collections import defaultdict


"""
    Configurations
"""
NUM_FOLDS = 5
METHODS = ["rule", "presidio", "customdeberta"]
METHOD_NAMES = {"rule": "Rule-based", "presidio": "Presidio", "customdeberta": "DeBERTa-v3"}
OUTPUT_FILE = "category_token_level_results.json"

# PII categories
CATEGORIES = ['NAME_STUDENT', 'EMAIL', 'PHONE_NUM', 'ID_NUM', 
              'STREET_ADDRESS', 'USERNAME', 'URL_PERSONAL']


"""
    Helper to extract metrics per category including counts.
"""
def extract_category_metrics(true_labels, pred_labels, categories):
    # Flatten
    true_flat = [label for doc in true_labels for label in doc]
    pred_flat = [label for doc in pred_labels for label in doc]
    
    # Get all unique labels
    all_labels = sorted(list(set(true_flat + pred_flat)))
    
    # Classification report
    report = classification_report(
        true_flat,
        pred_flat,
        labels=all_labels,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix for detailed counts
    cm = confusion_matrix(true_flat, pred_flat, labels=all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    # Extract metrics
    results = {}
    
    for category in categories:
        b_label = f"B-{category}"
        
        # Default values
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0,  # How many actually exist
            "predicted": 0,  # How many we predicted
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
        
        # If category exists in predictions
        if b_label in report:
            metrics["precision"] = report[b_label]["precision"]
            metrics["recall"] = report[b_label]["recall"]
            metrics["f1"] = report[b_label]["f1-score"]
            metrics["support"] = int(report[b_label]["support"])
        
        # Count predictions and true positives from confusion matrix
        if b_label in label_to_idx:
            idx = label_to_idx[b_label]
            
            # True positives: diagonal element
            metrics["true_positives"] = int(cm[idx, idx])
            
            # False positives: column sum minus diagonal
            metrics["false_positives"] = int(cm[:, idx].sum() - cm[idx, idx])
            
            # False negatives: row sum minus diagonal
            metrics["false_negatives"] = int(cm[idx, :].sum() - cm[idx, idx])
            
            # Total predicted as this category
            metrics["predicted"] = int(cm[:, idx].sum())
        
        results[category] = metrics
    
    return results

def banner(text):
    print("\n" + "=" * 90)
    print(text.center(90))
    print("=" * 90 + "\n")


"""
    Run categorical evaluation across all folds and methods.    
"""
def run_per_category_evaluation():
    banner("PER-CATEGORY EVALUATION STARTED")
    
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
                    custom_deberta.MODEL_PATH = os.path.join(custom_deberta.PROJECT_ROOT, "models", f"deberta_fold_{fold_idx}")
                    print(f"  Model: {custom_deberta.MODEL_PATH}")
                
                # Run method
                true_labels, pred_labels, _, _ = run_method(method_key, val_file)
                
                # Extract the metrics
                category_metrics = extract_category_metrics(true_labels, pred_labels, CATEGORIES)
                
                # Store
                all_results["per_fold"][fold_key][method_key] = category_metrics
                
                # Print summary
                total_support = sum(m["support"] for m in category_metrics.values())
                total_detected = sum(m["predicted"] for m in category_metrics.values())
                total_correct = sum(m["true_positives"] for m in category_metrics.values())
                
                print(f"  ✓ Processed {total_support} PII entities")
                print(f"    Detected: {total_detected}, Correct: {total_correct}")
                print(f"    Overall P={total_correct/total_detected*100:.1f}%, R={total_correct/total_support*100:.1f}%\n")
                
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
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
    print("\nQUICK SUMMARY")
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
            
            print(f"  Total PII in data: {total_support}")
            print(f"  Total detected: {total_detected}")
            print(f"  Correct detections: {total_correct}")
            print(f"  Precision: {overall_p:.1f}%")
            print(f"  Recall: {overall_r:.1f}%")
            print(f"  F1-Score: {overall_f1:.1f}%")