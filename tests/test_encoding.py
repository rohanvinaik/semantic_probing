"""Tests for encoding module."""

import numpy as np
import pytest
from semantic_probing.encoding.sparse_ternary import (
    SparseVector,
    bind,
    bundle,
    permute,
    HadamardBasis,
)
from semantic_probing.encoding.semantic_dimensions import (
    D,
    SemanticDimension,
    BIPOLAR_PAIRS,
)

def test_sparse_vector_init():
    """Test SparseVector initialization."""
    v = SparseVector(dimension=100, positive_indices=[1, 5], negative_indices=[2, 8])
    assert v.dimension == 100
    assert v.nnz == 4
    assert v.sparsity == 0.04
    np.testing.assert_array_equal(v.positive_indices, [1, 5])
    np.testing.assert_array_equal(v.negative_indices, [2, 8])

def test_sparse_vector_dense_conversion():
    """Test conversion to/from dense arrays."""
    dense = np.zeros(10, dtype=np.int8)
    dense[1] = 1
    dense[5] = -1
    
    v = SparseVector.from_dense(dense)
    assert v.dimension == 10
    np.testing.assert_array_equal(v.positive_indices, [1])
    np.testing.assert_array_equal(v.negative_indices, [5])
    
    dense_back = v.to_dense()
    np.testing.assert_array_equal(dense, dense_back)

def test_dot_product():
    """Test dot product."""
    # Orthogonal
    v1 = SparseVector(10, [1], [2])
    v2 = SparseVector(10, [3], [4])
    assert v1.dot(v2) == 0
    
    # Overlap same sign
    v3 = SparseVector(10, [1, 3], [2])
    assert v1.dot(v3) == 2 # 1 match in pos (idx 1), 1 match in neg (idx 2)
    
    # Overlap opposite sign
    v4 = SparseVector(10, [2], [1])
    assert v1.dot(v4) == -2 # idx 1 (+ vs -), idx 2 (- vs +)

def test_cosine_similarity():
    """Test cosine similarity."""
    v1 = SparseVector(10, [1, 2], [])
    v2 = SparseVector(10, [1, 2], [])
    assert np.isclose(v1.cosine(v2), 1.0)
    
    v3 = SparseVector(10, [], [1, 2])
    assert np.isclose(v1.cosine(v3), -1.0)
    
    v4 = SparseVector(10, [3, 4], [])
    assert np.isclose(v1.cosine(v4), 0.0)

def test_negate():
    """Test vector negation."""
    v = SparseVector(10, [1], [2])
    neg = v.negate()
    np.testing.assert_array_equal(neg.positive_indices, [2])
    np.testing.assert_array_equal(neg.negative_indices, [1])
    assert v.cosine(neg) == pytest.approx(-1.0)

def test_bind():
    """Test binding operation."""
    # (+1) * (+1) = +1
    v1 = SparseVector(10, [1], [])
    v2 = SparseVector(10, [1], [])
    res = bind(v1, v2)
    np.testing.assert_array_equal(res.positive_indices, [1])
    assert len(res.negative_indices) == 0
    
    # (+1) * (-1) = -1
    v3 = SparseVector(10, [], [1])
    res2 = bind(v1, v3)
    np.testing.assert_array_equal(res2.negative_indices, [1])
    assert len(res2.positive_indices) == 0

def test_bundle():
    """Test bundling (superposition)."""
    v1 = SparseVector(10, [1], [])
    v2 = SparseVector(10, [2], [])
    v3 = SparseVector(10, [1], []) # reinforce v1
    
    res = bundle([v1, v2, v3])
    # idx 1: +1 + 0 + 1 => +2 => +1
    # idx 2: 0 + 1 + 0 => +1 => +1
    
    np.testing.assert_array_equal(res.positive_indices, [1, 2])
    
    # Cancelation
    v4 = SparseVector(10, [], [1]) # -1 at idx 1
    res2 = bundle([v1, v4]) # +1 and -1 cancel to 0
    assert 1 not in res2.positive_indices
    assert 1 not in res2.negative_indices

def test_permute():
    """Test permutation."""
    v = SparseVector(10, [1], [2])
    p = permute(v, shift=1)
    
    np.testing.assert_array_equal(p.positive_indices, [2])
    np.testing.assert_array_equal(p.negative_indices, [3])
    
    # Wrap around
    v2 = SparseVector(10, [9], [])
    p2 = permute(v2, shift=1)
    np.testing.assert_array_equal(p2.positive_indices, [0])

def test_serialization():
    """Test msgpack serialization."""
    v = SparseVector(100, [1, 5], [2, 8], bank=3)
    data = v.serialize()
    v2 = SparseVector.deserialize(data)
    
    assert v.dimension == v2.dimension
    assert v.bank == v2.bank
    np.testing.assert_array_equal(v.positive_indices, v2.positive_indices)
    np.testing.assert_array_equal(v.negative_indices, v2.negative_indices)

def test_hadamard_basis_generation():
    """Test Hadamard basis generation."""
    # Use smaller dimension for speed
    # But hadamard requires power of 2
    # Current code defaults to global params, so we might test the global defaults
    # or mock constants if possible.
    # We will trust the defaults but just check a few properties.
    
    basis = HadamardBasis(dimension=2048, dims_per_bank=256).generate()
    
    # Check primitive count
    assert len(basis.primitives) > 0
    
    # Check bipolar pairs
    for pair in BIPOLAR_PAIRS:
        pos = basis.get_primitive(pair.positive)
        neg = basis.get_primitive(pair.negative)
        
        assert pos is not None
        assert neg is not None
        
        # Check orthogonality
        assert np.isclose(pos.cosine(neg), -1.0)

def test_hadamard_orthogonality():
    """Test orthogonality of primitives in same dimension."""
    # This is the key "Semantic Probing" feature
    basis = HadamardBasis(dimension=2048, dims_per_bank=256).generate()
    
    # Find two primitives in the same dimension (bank)
    # E.g. "I" and "YOU" in SUBSTANTIVES (Dim 0)
    p1 = basis.get_primitive("I")
    p2 = basis.get_primitive("YOU")
    
    if p1 and p2:
        # Should be exactly 0 due to Hadamard partitioning
        assert p1.dot(p2) == 0
