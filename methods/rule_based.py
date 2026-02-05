# methods/rule_based.py

"""
    Rule-based PII detection.
    Uses function words filter + 2 consecutive capitals for names.
"""

import re


def label_tokens(tokens):
    
    labels = []
    
    # Function words
    FUNCTION_WORDS = {
        "The", "A", "An", "I", "We", "You", "He", "She", "It", "They",
        "Me", "Us", "Him", "Her", "Them", "My", "Your", "His", "Her",
        "Its", "Our", "Their", "This", "That", "These", "Those",
        "Who", "Whom", "What", "Which", "In", "At", "On", "To",
        "For", "With", "By", "From", "Of", "As", "And", "Or", "But"
    }
    
    for i, token in enumerate(tokens):
        if not token.strip():
            labels.append("O")
            continue
        
        # Patterns
        
        if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", token):
            labels.append("B-EMAIL")
        
        elif re.fullmatch(r"\d{3}-\d{3}-\d{4}", token):
            labels.append("B-PHONE_NUM")
        
        elif re.fullmatch(r"\d{5,}", token):
            labels.append("B-ID_NUM")
        
        elif re.fullmatch(r"https?://\S+", token):
            labels.append("B-URL_PERSONAL")
        
        elif re.fullmatch(r"@[A-Za-z0-9_]+", token):
            labels.append("B-USERNAME")
        
        # Name Detection
        
        elif is_name_candidate(token, FUNCTION_WORDS):
            # Check if previous non-whitespace token was a name
            prev_was_name = False
            for j in range(i - 1, -1, -1):
                if tokens[j].strip():
                    prev_was_name = labels[j].endswith("NAME_STUDENT")
                    break
            
            if prev_was_name:
                # Continue the name
                labels.append("I-NAME_STUDENT")
            else:
                # Check if next non whitespace token is also a name candidate
                next_token = None
                for j in range(i + 1, len(tokens)):
                    if tokens[j].strip():
                        next_token = tokens[j]
                        break
                
                if next_token and is_name_candidate(next_token, FUNCTION_WORDS):
                    # Start a new name (2 or more consecutive capitals)
                    labels.append("B-NAME_STUDENT")
                else:
                    # Single capitalized word is not a name
                    labels.append("O")
        
        else:
            labels.append("O")
    
    return labels

"""Check if token could be part of a name."""
def is_name_candidate(token, function_words):
    # Minimum length
    if len(token) < 3:
        return False
    
    # Must be title case
    if not token[0].isupper():
        return False
    
    # Exclude function words
    if token in function_words:
        return False
    
    # Must match name pattern (Title case only)
    if not re.match(r"^[A-Z][a-z\-']+$", token):
        return False
    
    # Exclude all caps words
    if token.isupper():
        return False
    
    return True


detect = label_tokens