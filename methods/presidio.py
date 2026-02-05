# methods/presidio.py


"""
    PII detection using Presidio.
    Uses spaCy model + Presidio's pre-trained models.
"""

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from pipeline.obfuscator import obfuscate_token
import spacy


# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Create NLP engine
nlp_engine = SpacyNlpEngine(models=[{
    "lang_code": "en",
    "model": nlp,
    "model_name": "en_core_web_lg"
}])

# Create analyzer without custom patterns
analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["en"]
)


def label_tokens(tokens):
    # Simple reconstruction by joining with spaces
    text = " ".join(tokens)
    
    # Build token spans
    token_spans = []
    current_pos = 0
    
    for token in tokens:
        start = current_pos
        end = current_pos + len(token)
        token_spans.append((start, end))
        current_pos = end + 1  # +1 for the space
    
    # Run Presidio
    results = analyzer.analyze(
        text=text, 
        language="en",
        score_threshold=0.2  # Moderate threshold
    )
    
    # Start with all O labels
    labels = ["O"] * len(tokens)
    
    # Map entities with B-/I- tagging
    for entity in results:
        ent_start = entity.start
        ent_end = entity.end
        ent_type = normalize_entity_type(entity.entity_type)
        
        if ent_type == "O":
            continue
        
        # Find all tokens that overlap with this entity
        first_token_in_entity = True
        
        for i, (tok_start, tok_end) in enumerate(token_spans):
            if tok_end > ent_start and tok_start < ent_end:
                # Skip if token is too short (tokenization artifact like "cour")
                if len(tokens[i].strip()) < 2:
                    continue

                if first_token_in_entity:
                    labels[i] = f"B-{ent_type}"
                    first_token_in_entity = False
                else:
                    labels[i] = f"I-{ent_type}"
    
    return labels


def anonymize_tokens(tokens):
    labels = label_tokens(tokens)
    anonymized = []

    for token, label in zip(tokens, labels):
        if label.startswith("B-") or label.startswith("I-"):
            pii_type = label.split("-")[1]
            anonymized.append(obfuscate_token(pii_type, original_token=token))
        else:
            anonymized.append(token)

    return anonymized


"""Strict mapping using only clear PII types"""
def normalize_entity_type(presidio_type):
    
    mapping = {
        "PERSON": "NAME_STUDENT",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE_NUM",
        "URL": "URL_PERSONAL",
        "US_SSN": "ID_NUM",
        "CREDIT_CARD": "ID_NUM",
        "US_DRIVER_LICENSE": "ID_NUM",
        "US_PASSPORT": "ID_NUM",
        "US_BANK_NUMBER": "ID_NUM",
        "USERNAME": "USERNAME",
        "ADDRESS": "STREET_ADDRESS",
    }
    
    return mapping.get(presidio_type.upper(), "O")