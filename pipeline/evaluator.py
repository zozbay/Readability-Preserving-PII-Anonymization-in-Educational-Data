# pipeline/evaluator.py

"""
    Evaluate using seqeval with BIO entity boundaries.
    Both inputs should be: List[List[str]] list of documents, each containing token labels.
"""

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

def evaluate_predictions(true_labels, pred_labels):
    
    
    if not true_labels or not pred_labels:
        return {"error": "Missing labels for evaluation"}
    
    if len(true_labels) != len(pred_labels):
        return {"error": f"Length mismatch: {len(true_labels)} vs {len(pred_labels)}"}
    
    # Compute metrics using seqeval
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    # Count total PII entities (not tokens)
    support = sum(
        sum(1 for l in doc if l.startswith("B-"))  # Count entities, not tokens
        for doc in true_labels
    )
    
    # Detailed per-entity metrics
    detailed = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "detailed": detailed
    }