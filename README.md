# Semantic Probing

**Structured behavioral probes for model cognition.**

This library provides tools for behavioral fingerprinting of language models using sparse ternary semantic encoding organized around 62 NSM (Natural Semantic Metalanguage) primitives.

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

### Semantic Analogy

Perform vector arithmetic using the sparse ternary representation.

```python
from semantic_probing import SparseVector, SemanticProbe

probe = SemanticProbe()

# Solve: A is to B as C is to ?
target = probe.analogy(vec_a, vec_b, vec_c)
```

## directory Structure

-   `src/semantic_probing/encoding`: Core sparse ternary vector implementation and Hadamard basis.
-   `src/semantic_probing/grounding`: Mapping text and entities to the vector space.
-   `src/semantic_probing/probes`: Tools for measuring semantic properties (primitives, logic, reasoning).
-   `src/semantic_probing/analysis`: High-level analysis of model signatures.
-   `experiments`: Demo scripts and experiments.
