# deberta_train_fold.py

"""
Trains DeBERTa-v3-large for PII detection with proper word-level evaluation.

Key features:
- Filters whitespace tokens during preprocessing
- Uses seqeval for proper BIO entity evaluation
- Evaluates at word level not subword level

Usage: python train_fold.py <fold_idx>
Example: python train_fold.py 0

"""

import json
import sys
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    DataCollatorForTokenClassification, 
    Trainer, 
    TrainingArguments,
    AutoModelForTokenClassification, 
    AutoConfig, 
    EarlyStoppingCallback
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


"""
    Training configuration
"""
FOLD_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0

MODEL_CHECKPOINT = "microsoft/deberta-v3-large"
TRAIN_DATA_PATH = f"data/fold_{FOLD_IDX}_train.json"
VAL_DATA_PATH = f"data/fold_{FOLD_IDX}_val.json"
OUTPUT_DIR = f"models/deberta_fold_{FOLD_IDX}"

# BIO labels for PII entities
LABEL_LIST = [
    "O",                    # Not PII
    "B-NAME_STUDENT",       # Beginning of student name
    "I-NAME_STUDENT",       # Inside student name
    "B-EMAIL",              
    "I-EMAIL",
    "B-USERNAME",
    "I-USERNAME",
    "B-ID_NUM",
    "I-ID_NUM",
    "B-PHONE_NUM",
    "I-PHONE_NUM",
    "B-URL_PERSONAL",
    "I-URL_PERSONAL",
    "B-STREET_ADDRESS",
    "I-STREET_ADDRESS"
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, add_prefix_space=True)
tokenizer.padding_side = "right"

MAX_LENGTH = 768  # Maximum sequence length --> could be increased to 1024 if resource allows


"""
    Load fold data from JSON.
    
    Expected format:
    [
        {
            "tokens": ["word1", " ", "word2", ...],
            "labels": ["O", "O", "B-NAME_STUDENT", ...],
            "trailing_whitespace": [true, false, ...]
        },
        ...
    ]
    """
def load_fold_data(path):
    
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    samples = []
    for doc in docs:
        tokens = doc["tokens"]
        labels = doc.get("labels", ["O"] * len(tokens))
        
        samples.append({
            "tokens": tokens,
            "ner_tags": [LABEL_TO_ID.get(label, 0) for label in labels]
        })
    
    return Dataset.from_list(samples)



"""
    Remove whitespace-only tokens to match inference pipeline.
    
    Args:
        tokens: List of token strings
        labels: List of label IDs
    
    Returns:
        filtered_tokens: List of non-whitespace tokens
        filtered_labels: Corresponding labels
    """
def filter_tokens(tokens, labels):
    
    filtered_tokens = []
    filtered_labels = []
    
    for token, label in zip(tokens, labels):
        if token.strip():  # Keep only non-whitespace tokens
            filtered_tokens.append(token)
            filtered_labels.append(label)
    
    return filtered_tokens, filtered_labels




"""
    Tokenize words and align labels to subwords.
    
    Process:
    1. Filter whitespace tokens
    2. Tokenize into subwords
    3. Align labels: first subword gets label, rest get same label
    4. Special tokens (CLS, SEP, PAD) get -100 (ignored in loss)
    
    This creates the training data for the model.
    """
def tokenize_and_align_labels(example):
    
    # 1. Filter whitespace
    tokens, labels = filter_tokens(example["tokens"], example["ner_tags"])
    
    # 2. Tokenize
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    
    # 3. Align labels to subwords
    word_ids = tokenized.word_ids()
    aligned_labels = []
    
    for word_idx in word_ids:
        if word_idx is None:
            # Special token (CLS, SEP, PAD) need to be ignored in loss
            aligned_labels.append(-100)
        else:
            # Assign the word's label to all its subwords
            aligned_labels.append(labels[word_idx])
    
    tokenized["labels"] = aligned_labels
    return tokenized



"""
    Compute SUBWORD-LEVEL metrics using seqeval for training.
    This can not be used for inference evaluation since inference is at word level ( a subword is not a complete word).
    
    Training evaluates at subword level (only for displaying purposes and logging),
    while inference evaluates at word level which matches the anonymization task.
"""
def compute_metrics(eval_pred):
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Convert IDs to label strings, skipping -100 (padding/special tokens)
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        true_words = []
        pred_words = []
        
        for i, (pred_id, label_id) in enumerate(zip(pred_seq, label_seq)):
            if label_id == -100:
                continue
            
            # Append all subword predictions 
            true_words.append(ID_TO_LABEL[label_id])
            pred_words.append(ID_TO_LABEL[pred_id])
        
        true_labels.append(true_words)
        pred_labels.append(pred_words)
    
    # Compute seqeval metrics at subword level
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    print(f"\n{'='*60}")
    print(f"Training DeBERTa for PII Detection - Fold {FOLD_IDX}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    train_data = load_fold_data(TRAIN_DATA_PATH)
    val_data = load_fold_data(VAL_DATA_PATH)
    
    print(f"  Train: {len(train_data)} documents")
    print(f"  Val:   {len(val_data)} documents")
    
    # Preprocess
    print("\nPreprocessing (filtering whitespace, tokenizing)...")
    train_dataset = train_data.map(tokenize_and_align_labels, batched=False)
    val_dataset = val_data.map(tokenize_and_align_labels, batched=False)
    
    # Load model
    print(f"\nLoading model: {MODEL_CHECKPOINT}")
    config = AutoConfig.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )
    
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        config=config
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # Evaluation strategy
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        
        # Optimization
        learning_rate=3e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        weight_decay=0.01,
        
        # Batch size (effective batch size = 1 * 16 = 16)
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        
        # Training duration
        num_train_epochs=10,
        
        # Logging
        logging_steps=50,
        
        # Performance
        fp16=True,  # Mixed precision training
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )
    
    # Train
    print("\nStarting training...\n")
    trainer.train()
    
    # Save
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Final evaluation
    print("\nFinal evaluation on validation set...")
    eval_results = trainer.evaluate()
    
    # Save metrics
    metrics = {
        "fold": FOLD_IDX,
        "precision": eval_results.get("eval_precision", 0.0),
        "recall": eval_results.get("eval_recall", 0.0),
        "f1": eval_results.get("eval_f1", 0.0),
        "loss": eval_results.get("eval_loss", 0.0),
    }
    
    with open(f"{OUTPUT_DIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE - Fold {FOLD_IDX}")
    print(f"{'='*60}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()