# pipeline/anonymizer.py

""""
    Main anonymization pipeline.
    Supports multiple PII detection methods.

"""

import json
from pipeline.reconstruct import reconstruct_text
from methods import rule_based, presidio, custom_deberta
from pipeline.obfuscator import obfuscate_token


METHODS = {
    "rule": rule_based.label_tokens,
    "presidio": presidio.label_tokens,
    "customdeberta": custom_deberta.label_tokens,
}

def run_method(method_name, input_file):
    label_fn = METHODS[method_name]

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predicted = []
    gold_labels = []
    full_texts = []
    anonymized_texts = []

    for i, doc in enumerate(data):
        print(f"Processing document {i + 1}/{len(data)}", end="\r")

        tokens = doc["tokens"]
        whitespace = doc["trailing_whitespace"]
        labels = doc.get("labels")

        # Get predictions for all tokens
        pred_labels = label_fn(tokens)
        
        assert len(pred_labels) == len(tokens), \
            f"Label length mismatch in {method_name}: {len(pred_labels)} vs {len(tokens)}"

        # Filter for evaluation (with whitespace removed)
        filtered_pred = []
        filtered_gold = []
        
        for tok, pred_label, gold_label in zip(tokens, pred_labels, labels if labels else ["O"] * len(tokens)):
            if tok.strip():
                filtered_pred.append(pred_label)
                if labels:
                    filtered_gold.append(gold_label)
        
        predicted.append(filtered_pred)
        if labels:
            gold_labels.append(filtered_gold)

        # Reconstruct original text
        original_text = reconstruct_text(tokens, whitespace)
        
        # Only obfuscate non-whitespace tokens
        anonymized_tokens = []
        for token, label in zip(tokens, pred_labels):
            if token.strip() and (label.startswith("B-") or label.startswith("I-")):
                # Token is non whitespace and labeled as PII
                pii_type = label.split("-")[1]
                anonymized_tokens.append(obfuscate_token(pii_type, token))
            else:
                # Keep as it is if either whitespace or non-PII
                anonymized_tokens.append(token)
        
        anonymized_text = reconstruct_text(anonymized_tokens, whitespace)

        full_texts.append(original_text)
        anonymized_texts.append(anonymized_text)

    print()
    return gold_labels, predicted, full_texts, anonymized_texts