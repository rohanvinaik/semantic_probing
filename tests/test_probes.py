"""Tests for probes module."""

import pytest
from semantic_probing.encoding import (
    SparseVector,
    HadamardBasis,
    bundle,
    bind,
)
from semantic_probing.probes import (
    PrimitiveProbe,
    SemanticProbe,
    ReasoningProbe,
    Fact,
)

@pytest.fixture(scope="module")
def basis():
    """Shared basis for tests (expensive to generate)."""
    # Use smaller dims to speed up tests? 
    # Default 8192 is fast enough (~ms)
    return HadamardBasis().generate()

def test_primitive_probe(basis):
    """Test primitive detection."""
    probe = PrimitiveProbe(basis)
    
    # Create vector: I + WANT + GOOD
    v_i = basis.get_primitive("I")
    v_want = basis.get_primitive("WANT")
    v_good = basis.get_primitive("GOOD")
    
    vec = bundle([v_i, v_want, v_good])
    
    activations = probe.probe_vector(vec, threshold=0.1)
    
    assert "I" in activations
    assert "WANT" in activations
    assert "GOOD" in activations
    assert "BAD" in activations # BAD is activated negatively (cos ~ -1)
    assert activations["BAD"] < -0.8
    
    # Check decomposition
    decomp = probe.decompose(vec, top_k=5)
    names = [x[0] for x in decomp]
    assert "I" in names
    assert "WANT" in names
    assert "GOOD" in names

def test_bipolar_probe(basis):
    """Test bipolar activation."""
    probe = PrimitiveProbe(basis)
    
    v_good = basis.get_primitive("GOOD")
    v_bad = basis.get_primitive("BAD")
    
    # Strong positive
    acts_good = probe.probe_bipolar(v_good)
    eval_act = next((a for a in acts_good if a.pair.name == "EVALUATION"), None)
    assert eval_act is not None
    assert eval_act.value > 0.9 # Should be 1.0
    
    # Strong negative
    acts_bad = probe.probe_bipolar(v_bad)
    eval_act = next((a for a in acts_bad if a.pair.name == "EVALUATION"), None)
    assert eval_act is not None
    assert eval_act.value < -0.9 # Should be -1.0

def test_semantic_probe_analogy(basis):
    """Test semantic analogy."""
    probe = SemanticProbe()
    
    # Analogy: GOOD is to BAD as TRUE is to ? (FALSE)
    p_good = basis.get_primitive("GOOD")
    p_bad = basis.get_primitive("BAD")
    p_true = basis.get_primitive("TRUE")
    p_false = basis.get_primitive("FALSE")
    
    # Analogy: GOOD is to BAD as BAD is to ? (GOOD)
    # A=GOOD, B=BAD, C=BAD
    # D = C + (B - A) = BAD + (BAD - GOOD) = -G + (-G - G) = -3G
    # -3G is aligned with -G (BAD) -> cos(BAD) = 1.0, cos(GOOD) = -1.0
    
    # Let's just test that the vector math produces the expected vector
    # target = bundle([c, b, a.negate()])
    target = probe.analogy(p_good, p_bad, p_true)
    
    # In orthogonal space: TRUE + (BAD - GOOD)
    # Expected: Proj(TRUE) = 1, Proj(BAD) = 1, Proj(GOOD) = -1
    assert target.cosine(p_true) > 0.3
    assert target.cosine(p_bad) > 0.3
    assert target.cosine(p_good) < -0.3

def test_reasoning_probe_fact(basis):
    """Test fact encoding and querying."""
    probe = ReasoningProbe(basis)
    
    # Fact: LOVE(I, YOU)
    # Predicate: LOVE (symbol)
    # Args: I, YOU
    
    # We need to register LOVE as a symbol or let probe generate it
    # We'll rely on auto-generation in probe
    
    i_vec = basis.get_primitive("I")
    you_vec = basis.get_primitive("YOU")
    
    fact = Fact("LOVE", ["I", "YOU"])
    entity_vectors = {"I": i_vec, "YOU": you_vec}
    
    fact_vec = probe.encode_fact(fact, entity_vectors)
    
    # Query exact fact
    score = probe.query_fact(fact_vec, fact_vec)
    assert score > 0.9
    
    # Query wrong fact
    fact_wrong = Fact("HATE", ["I", "YOU"])
    wrong_vec = probe.encode_fact(fact_wrong, entity_vectors)
    
    score_wrong = probe.query_fact(wrong_vec, fact_vec)
    assert score_wrong < 0.5

def test_reasoning_probe_relation(basis):
    """Test relation probing."""
    probe = ReasoningProbe(basis)
    
    # Imagine a transform R that maps A to B
    # B ~ Bind(A, R)
    
    vec_a = basis.get_primitive("I")
    vec_b = basis.get_primitive("YOU") 
    
    # Create artificial relation R that maps I to YOU exactly
    # R = Bind(I, YOU) (since I*I=1 assuming binary... wait, ternary)
    # In ternary: Bind(A, B) * A -> B?
    # (+1 * +1) * +1 = +1
    # (-1 * -1) * -1 = +1 * -1 = -1
    # (+1 * -1) * +1 = -1 * +1 = -1
    # Yes, Bind(A, B) * A recovers B approx (or exactly if no zeros?)
    # Zeros mess it up.
    
    # Let's see if probe_relation logic matches
    
    rel_name = "REL_X"
    rel_vec = probe._get_vector(rel_name)
    
    # Construct object that IS related
    # obj = bind(subj, rel)
    obj_vec = bind(vec_a, rel_vec)
    
    # Probe should return high score
    score = probe.probe_relation(vec_a, obj_vec, rel_name)
    assert score > 0.9
