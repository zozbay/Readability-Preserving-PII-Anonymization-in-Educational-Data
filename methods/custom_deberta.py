# methods/custom_deberta.py

"""
DeBERTa inference for PII detection.
Uses a fine-tuned DeBERTa model for token classification.
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "deberta_fold_0")

#MODEL_PATH = "models/deberta_fold_0"  # Path to the model

LABEL_LIST = [
    "O", "B-NAME_STUDENT", "I-NAME_STUDENT", "B-EMAIL", "I-EMAIL",
    "B-USERNAME", "I-USERNAME", "B-ID_NUM", "I-ID_NUM", "B-PHONE_NUM",
    "I-PHONE_NUM", "B-URL_PERSONAL", "I-URL_PERSONAL",
    "B-STREET_ADDRESS", "I-STREET_ADDRESS"
]

ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


"""
    Filter whitespace tokens (same as training).
"""
def filter_tokens(tokens):
   
    filtered_tokens = []
    original_indices = []
    
    for idx, token in enumerate(tokens):
        if token.strip():
            filtered_tokens.append(token)
            original_indices.append(idx)
    
    return filtered_tokens, original_indices


"""
    Predict BIO labels for tokens and returns labels for all original tokens 
    including whitespace as "O".
"""
def label_tokens(tokens):
    
    # 1. Filter whitespace (same as training)
    filtered_tokens, original_indices = filter_tokens(tokens)
    
    if not filtered_tokens:
        return ["O"] * len(tokens)
    
    # 2. Tokenize (same as training)
    encoding = tokenizer(
        filtered_tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=768,
        return_tensors="pt"
    )
    
    word_ids = encoding.word_ids()
    
    # 3. Predict
    encoding_gpu = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding_gpu)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
    
    # 4. Align predictions to filtered tokens (first subword only since BIO tags)
    filtered_labels = ["O"] * len(filtered_tokens)
    previous_word_idx = None
    
    for i, word_idx in enumerate(word_ids):
        if word_idx is None:  # Special token
            continue
        
        
        if word_idx != previous_word_idx:  # First subword of this word
            filtered_labels[word_idx] = ID_TO_LABEL.get(predictions[i], "O")
        
        previous_word_idx = word_idx
    
    # 5. Map back to original tokens (whitespace gets "O")
    final_labels = ["O"] * len(tokens)
    
    for filtered_idx, original_idx in enumerate(original_indices):
        final_labels[original_idx] = filtered_labels[filtered_idx]
    
    return final_labels