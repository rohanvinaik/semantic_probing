# Semantic Probing

**Structured behavioral probes for model cognition.**

This library provides tools for behavioral fingerprinting of language models using sparse ternary semantic encoding organized around 62 NSM (Natural Semantic Metalanguage) primitives.

---

## Headline Results

The multi-scale encoding system achieves mathematically guaranteed properties that enable precise semantic probing:

| Property | Result | Significance |
|----------|--------|--------------|
| **Intra-dimension orthogonality** | 229/229 pairs at cos=0.0 | Primitives within the same semantic dimension are *exactly* orthogonal (not approximately) |
| **Bipolar antonymy** | 14/14 pairs at cos=-1.0 | Antonym pairs (GOOD/BAD, TRUE/FALSE, etc.) achieve *perfect* structural opposition |
| **Cross-scale separation** | mean\|cos\|=0.003 | Word and Sentence bases are statistically independent (max\|cos\|=0.077) |
| **Antonym detection** | 100% accuracy | Zero false positives on bipolar pair detection |

### Set-Theoretic Operations

The encoding supports three core operations with provable algebraic properties:

```
Bind (⊗):   SOMEONE ⊗ ARG0  →  cos(result, inputs) ≈ 0.1  [dissimilar to both]
Bundle (⊕): GOOD ⊕ BIG ⊕ X  →  cos(result, GOOD) = 0.67   [similar to all]
Permute (ρ): ρ¹⁰⁰(DO)       →  cos(DO, ρ¹⁰⁰(DO)) = 0.12   [position encoding]
```

### Architecture

- **8,192-dimensional** sparse ternary vectors
- **8 semantic dimensions** with non-overlapping Hadamard partitions
- **4 scales**: Word → Sentence → Paragraph → Text
- **73 word primitives**, **24 sentence primitives**, **10 compositional operators**
- **1.79% sparsity** (~147 non-zero elements per vector)

### Performance (Consumer Hardware)

Benchmarked on Apple M-series laptop (single-threaded):

| Operation | Time | Throughput |
|-----------|------|------------|
| Cosine similarity | **59 μs** | 17,000/sec |
| Bind (⊗) | **69 μs** | 14,500/sec |
| Bundle (⊕) 3 vectors | **41 μs** | 24,400/sec |
| Antonym detection | **451 μs** | 2,200/sec |
| Full sentence encoding | **273 μs** | **3,665 sentences/sec** |

**Memory footprint**: ~21 KB for entire basis (298 bytes/vector)

**vs. LLM inference**: Semantic operations are **10,000-100,000x faster** than equivalent GPT-4 API calls, enabling real-time analysis at scale without GPU requirements.

---

## Philosophy

Instead of training probes on massive datasets to predict arbitrary properties, `semantic-probing` constructs a **theoretically grounded basis** of semantic primitives. By projecting model representations (or synthesized vectors) into this space, we can measure:

1.  **Antonymy**: Using structurally opposed dimensions (e.g., Good/Bad).
2.  **Logic**: Detecting activation of logical primitives (IF, BECAUSE, NOT).
3.  **Coverage**: Measuring how much of the semantic space is "active" in a representation.

## Installation

```bash
pip install -e .
```

## Quick Start

-   **Orthogonal Semantic Basis**: Uses 8192-dimensional Hadamard vectors for interference-free superposition.
-   **Multi-Scale Architecture**: Supports Word, Sentence, Paragraph, and Text scales with distinct orthogonal bases.
-   **Bipolar Dimensions**: Explicitly models antonyms as antipodal vectors (cosine = -1.0).
-   **Compositional Grounding**: Binds concepts with syntactic roles and compositional operators.
-   **Semantic Probes**: Tools to detect primitives, measure similarity, and solve analogies (including bipolar).
-   **Analyzers**: Utilities for analyzing semantic signatures and cross-model correlations.

### Antonym Detection

Detect if two vectors are antonyms by checking for structural opposition in bipolar dimensions.

```python
from semantic_probing import HadamardBasis, AntonymDetectionProbe

# Initialize basis (this creates the 8192-d sparse vectors for all primitives)
basis = HadamardBasis().generate()
probe = AntonymDetectionProbe(basis)

# Get primitive vectors (in real usage, these would come from decoding a model)
good = basis.get_primitive("GOOD")
bad = basis.get_primitive("BAD")

# Detect relationship
result = probe.detect_antonym(good, bad)

if result.is_antonym:
    print(f"Antonym detected! Pair: {result.bipolar_pair}")
    print(f"Confidence: {result.confidence:.2f}")
```

### Multi-Scale Composition

Encode phrases with syntactic structure using the 4-scale system:

```python
from semantic_probing import HadamardBasis, SentenceScaleBasis, bind, bundle

# Initialize both scales
word_basis = HadamardBasis().generate()
sent_basis = SentenceScaleBasis().generate()

# Encode "the big person" with grammatical structure
big = word_basis.get_primitive("BIG")
someone = word_basis.get_primitive("SOMEONE")
adj_mod = sent_basis.get_primitive("SYN_ADJ_MOD")
subj = sent_basis.get_primitive("SYN_SUBJ")

# Compose: [BIG ⊗ ADJ_MOD] ⊕ [SOMEONE ⊗ SUBJ]
phrase = bundle([bind(big, adj_mod), bind(someone, subj)])
```

### Semantic Analogy

Perform vector arithmetic using the sparse ternary representation.

```python
from semantic_probing import SparseVector, SemanticProbe

probe = SemanticProbe()

# Solve: A is to B as C is to ?
target = probe.analogy(vec_a, vec_b, vec_c)
```

## Directory Structure

-   `src/semantic_probing/encoding`: Core sparse ternary vector implementation and Hadamard basis.
-   `src/semantic_probing/grounding`: Mapping text and entities to the vector space.
-   `src/semantic_probing/probes`: Tools for measuring semantic properties (primitives, logic, reasoning).
-   `src/semantic_probing/analysis`: High-level analysis of model signatures.
-   `experiments`: Demo scripts and experiments.
-   `tests`: Comprehensive test suite including headline results validation.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run headline results tests with output
pytest tests/test_headline_results.py -v -s
```

## Theoretical Foundation

The encoding is grounded in:

- **Natural Semantic Metalanguage (NSM)**: 62 universal semantic primitives
- **Hyperdimensional Computing**: Sparse ternary vectors with bind/bundle/permute
- **Hadamard Matrices**: Guaranteed orthogonality via non-overlapping partitions

## License

MIT License
