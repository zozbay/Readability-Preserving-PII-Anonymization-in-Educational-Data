# pipeline/readability.py

"""
    Computes readability metrics for a given text.
    The metrics include Flesch Reading Ease, Flesch-Kincaid Grade Level, and Gunning Fog Index.
"""

import textstat

def compute_readability(text):
    return {
        "flesch": textstat.flesch_reading_ease(text),
        "fk_grade": textstat.flesch_kincaid_grade(text),
        "fog": textstat.gunning_fog(text)
    }