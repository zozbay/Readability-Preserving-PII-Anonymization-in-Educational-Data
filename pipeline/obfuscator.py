# pipeline/obfuscator.py

"""
    Obfuscation functions for different PII types.
    Uses Faker to generate realistic surrogates.
"""


from faker import Faker
import random


USE_PLACEHOLDERS = False
fake = Faker()

"""Trims or regenerates text to match original token length ±2."""
def match_length(text, original_len):
    
    if len(text) <= original_len + 2:
        return text
    return text[:original_len]


def safe_name(original):
    if not original or not original.strip():
        return "[NAME]"
    return match_length(fake.first_name(), len(original))

def safe_email(original):
    if not original or not original.strip():
        return "[EMAIL]"
    email = fake.user_name() + "@example.com"
    return match_length(email, len(original))

def safe_username(original):
    if not original or not original.strip():
        return "[USERNAME]"
    return match_length(fake.user_name(), len(original))

def safe_id(original):
    if not original or not original.strip():
        return "[ID]"
    return match_length(str(random.randint(100000, 999999)), len(original))

def safe_phone(original):
    if not original or not original.strip():
        return "[PHONE]"
    return match_length(fake.numerify("555-###-####"), len(original))

def safe_url(original):
    if not original or not original.strip():
        return "[URL]"
    return match_length("https://example.com", len(original))

def safe_address(original):
    if not original or not original.strip():
        return "[ADDRESS]"
    return match_length(fake.street_address(), len(original))


OBFUSCATION_FUNCTIONS = {
    "NAME_STUDENT": safe_name,
    "EMAIL": safe_email,
    "USERNAME": safe_username,
    "ID_NUM": safe_id,
    "PHONE_NUM": safe_phone,
    "URL_PERSONAL": safe_url,
    "STREET_ADDRESS": safe_address,
}

PLACEHOLDER_MAPPING = {
    "NAME_STUDENT": "[NAME]",
    "EMAIL": "[EMAIL]",
    "USERNAME": "[USERNAME]",
    "ID_NUM": "[ID]",
    "PHONE_NUM": "[PHONE]",
    "URL_PERSONAL": "[URL]",
    "STREET_ADDRESS": "[ADDRESS]",
}


"""
    Obfuscates a token based on its PII label.
    
    Args:
        label: PII type (e.g., "NAME_STUDENT")
        original_token: Original token text (or None)
    
    Returns:
        The obfuscated token
    """
_surrogate_cache = {}

def obfuscate_token(label, original_token=None):
    
    # Safety check
    if not original_token or not original_token.strip():
        return original_token  # Return whitespace unchanged
    
    # Use placeholders if configured
    if USE_PLACEHOLDERS:
        return PLACEHOLDER_MAPPING.get(label, "[REDACTED]")
    
    # Check cache
    if original_token in _surrogate_cache:
        return _surrogate_cache[original_token]
    
    # Generate surrogate
    generator = OBFUSCATION_FUNCTIONS.get(label)
    if generator:
        surrogate = generator(original_token)
        _surrogate_cache[original_token] = surrogate
        return surrogate
    
    return "[REDACTED]"