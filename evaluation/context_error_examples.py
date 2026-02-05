# extract_errors_all_folds.py


"""
Extract error examples from all folds for a extensive detailed analysis
"""

import json
import os
import sys

# Adding project root to path so that evaluation scripts be ran from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.reconstruct import reconstruct_text
from methods import custom_deberta
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


"""
    Configurations
"""
NUM_FOLDS = 5
OUTPUT_FILE = "results/error_examples_all_folds.json"


# Aggregate errors
all_errors = {
    "false_negatives": [],
    "false_positives": [],
    "summary": {
        "total_fn": 0,
        "total_fp": 0,
        "fn_by_category": {},
        "fp_by_category": {}
    }
}

print("Extracting errors from ALL folds...")
print("="*80 + "\n")

for fold_idx in range(NUM_FOLDS):
    print(f"\n{'='*80}")
    print(f"PROCESSING FOLD {fold_idx}")
    print(f"{'='*80}\n")
    
    VAL_FILE = f"data/fold_{fold_idx}_val.json"
    
    # Update model path
    custom_deberta.MODEL_PATH = os.path.join(custom_deberta.PROJECT_ROOT, "models", f"deberta_fold_{fold_idx}")
    
    # Reload model
    custom_deberta.tokenizer = AutoTokenizer.from_pretrained(
        custom_deberta.MODEL_PATH, add_prefix_space=True
    )
    custom_deberta.model = AutoModelForTokenClassification.from_pretrained(
        custom_deberta.MODEL_PATH
    )
    custom_deberta.model.to(custom_deberta.device)
    custom_deberta.model.eval()
    
    print(f"Model loaded: {custom_deberta.MODEL_PATH}")
    
    # Load validation data
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} documents...\n")
    
    # Extract errors for this fold
    for doc_idx, doc in enumerate(data):
        if (doc_idx + 1) % 100 == 0:
            print(f"  Document {doc_idx + 1}/{len(data)}")
        
        tokens = doc["tokens"]
        true_labels = doc["labels"]
        whitespace = doc["trailing_whitespace"]
        
        # Get predictions
        pred_labels = custom_deberta.label_tokens(tokens)
        
        if len(pred_labels) != len(tokens):
            print(f"Doc {doc_idx}: length mismatch!")
            continue
        
        # Compare token by token
        for token_idx, (token, true_label, pred_label) in enumerate(
            zip(tokens, true_labels, pred_labels)
        ):
            # Skip whitespace
            if not token.strip():
                continue
            
            # False negative
            if true_label.startswith("B-") and pred_label == "O":
                pii_type = true_label.split("-")[1]
                
                # Get context
                start_idx = max(0, token_idx - 5)
                end_idx = min(len(tokens), token_idx + 6)
                context = reconstruct_text(
                    tokens[start_idx:end_idx],
                    whitespace[start_idx:end_idx]
                )
                
                all_errors["false_negatives"].append({
                    "fold": fold_idx,
                    "doc_id": doc_idx,
                    "token": token,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "pii_type": pii_type,
                    "context": context,
                    "token_idx": token_idx
                })
                
                all_errors["summary"]["total_fn"] += 1
                all_errors["summary"]["fn_by_category"][pii_type] = \
                    all_errors["summary"]["fn_by_category"].get(pii_type, 0) + 1
            
            # False positive
            elif pred_label.startswith("B-") and not true_label.startswith("B-"):
                pii_type = pred_label.split("-")[1]
                
                # Get context
                start_idx = max(0, token_idx - 5)
                end_idx = min(len(tokens), token_idx + 6)
                context = reconstruct_text(
                    tokens[start_idx:end_idx],
                    whitespace[start_idx:end_idx]
                )
                
                all_errors["false_positives"].append({
                    "fold": fold_idx,
                    "doc_id": doc_idx,
                    "token": token,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "pii_type": pii_type,
                    "context": context,
                    "token_idx": token_idx
                })
                
                all_errors["summary"]["total_fp"] += 1
                all_errors["summary"]["fp_by_category"][pii_type] = \
                    all_errors["summary"]["fp_by_category"].get(pii_type, 0) + 1
    
    print(f"\n✓ Fold {fold_idx} complete:")
    print(f"  FN so far: {all_errors['summary']['total_fn']}")
    print(f"  FP so far: {all_errors['summary']['total_fp']}")

# Save aggregated results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_errors, f, indent=2, ensure_ascii=False)

print(f"\n{'='*80}")
print("AGGREGATION COMPLETE")
print(f"{'='*80}\n")
print(f"✓ Saved to {OUTPUT_FILE}")
print(f"\nFinal Summary:")
print(f"  Total False Negatives: {all_errors['summary']['total_fn']}")
print(f"  Total False Positives: {all_errors['summary']['total_fp']}")

print(f"\n  FN by category:")
for cat, count in sorted(all_errors['summary']['fn_by_category'].items(), 
                          key=lambda x: x[1], reverse=True):
    print(f"    {cat}: {count}")

print(f"\n  FP by category:")
for cat, count in sorted(all_errors['summary']['fp_by_category'].items(), 
                          key=lambda x: x[1], reverse=True):
    print(f"    {cat}: {count}")