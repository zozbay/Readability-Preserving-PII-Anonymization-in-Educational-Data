# pipeline/reconstruct_text.py

"""
    Reconstructs text exactly as it was in the original essay.
    token + space_if_true
"""

def reconstruct_text(tokens, whitespace):
    
    assert len(tokens) == len(whitespace), \
        f"Token/whitespace length mismatch: {len(tokens)} vs {len(whitespace)}"

    out = []
    for tok, ws in zip(tokens, whitespace):
        if ws:
            out.append(tok + " ")
        else:
            out.append(tok)

    return "".join(out)
