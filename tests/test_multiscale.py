"""Tests for multi-scale encoding."""
import pytest
from semantic_probing.encoding import (
    HadamardBasis,
    SentenceScaleBasis,
    encode_word_full,
    SENTENCE_PRIMITIVES,
    COMP_METADATA,
)

def test_sentence_basis_orthogonal_to_word_basis():
    """Sentence and word bases should be orthogonal."""
    word_basis = HadamardBasis().generate()
    sent_basis = SentenceScaleBasis().generate()

    # Sample cross-space cosines should be ~0
    # Check a few random intersections
    for w_name, w_vec in list(word_basis.primitives.items())[:5]:
        for s_name, s_vec in list(sent_basis.primitives.items())[:5]:
            # They use different generation seeds/mechanisms, so should be orthogonal
            # But "generate_hadamard_pattern" vs "generate_sentence_hadamard_pattern"
            # use distinct seeds ("" vs "SENTENCE_").
            # However, the dimension is large (8192), so collision is unlikely.
            # Ideal orthogonality is 0.0 only if partitions don't overlap.
            # Since they are in the same physical space (dim 0-8191) but generated structurally differently,
            # they are statistically orthogonal (cos ~ 0).
            assert abs(w_vec.cosine(s_vec)) < 0.15

def test_compositional_operators_have_metadata():
    """All COMP_* operators should have arity and operation."""
    for prim in SENTENCE_PRIMITIVES:
        if prim.startswith("COMP_"):
            assert prim in COMP_METADATA
            assert "arity" in COMP_METADATA[prim]
            assert "operation" in COMP_METADATA[prim]

def test_encode_word_full_composition():
    """Multi-scale composition should work."""
    word_basis = HadamardBasis().generate()
    sent_basis = SentenceScaleBasis().generate()

    word_vec = word_basis.get_primitive("GOOD")
    # If "GOOD" is not a direct primitive in this basis (might be bipolar pole), get it via basis
    if word_vec is None:
        # Try getting via bipolar lookup logic or just pick known primitive "SOMEONE"
        word_vec = word_basis.get_primitive("SOMEONE")
    
    assert word_vec is not None, "Could not find a testing word primitive"

    sent_vec = sent_basis.get_primitive("SYN_SUBJ")
    assert sent_vec is not None

    composed = encode_word_full(word_vec, sent_vec)

    # Composed should be dissimilar to both inputs (bind property)
    assert abs(composed.cosine(word_vec)) < 0.3
    assert abs(composed.cosine(sent_vec)) < 0.3
