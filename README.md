# Semantic Probing

**A mathematically grounded framework for probing semantic structure in neural representations.**

This library implements sparse ternary encoding over 62 universal semantic primitives (NSM), enabling precise behavioral fingerprinting of language models without training probes on massive datasets.

---

## Why This Matters for Interpretability

Current interpretability methods often rely on:
- Trained linear probes (require labeled data, overfit to specific tasks)
- Activation patching (costly, hard to interpret systematically)
- SAE features (emergent, not semantically grounded)

This framework offers a **complementary approach**: project representations into a pre-defined semantic basis with known mathematical properties, enabling:

1. **Compositional analysis** - decompose representations into interpretable primitives
2. **Zero-shot probing** - no training required, basis is constructed a priori
3. **Exact algebraic guarantees** - orthogonality and opposition are mathematically proven, not approximate

---

## Headline Results

### Mathematical Guarantees (Verified)

| Property | Result | Implication |
|----------|--------|-------------|
| **Intra-dimension orthogonality** | 229/229 pairs at cos = 0.000 | Primitives don't interfere within dimensions |
| **Bipolar opposition** | 14/14 pairs at cos = -1.000 | Antonyms are structurally opposed, not just distant |
| **Cross-dimension independence** | mean\|cos\| = 0.000 | Semantic dimensions are perfectly disentangled |
| **Cross-scale separation** | mean\|cos\| = 0.003 | Word/Sentence bases don't interfere |

### Capabilities (Demonstrated)

| Capability | Result | Significance |
|------------|--------|--------------|
| **Bipolar analogy solving** | 100% accuracy | GOOD:BAD :: ALIVE:? → DEAD (cos = 1.0 to target) |
| **Superposition capacity** | 73/73 primitives | Bundle ALL primitives, retrieve ANY with 100% accuracy |
| **Compositional unbinding** | 100% recovery | Extract concepts from bound role-filler structures |
| **Logical primitive detection** | 100% / 0% FP | Detect IF, BECAUSE, NOT in bundles without false positives |
| **Sequence encoding** | 100% position recovery | Encode and query by position using permutation |

### Performance (M-series Mac, single-threaded)

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Cosine similarity | 59 μs | 17,000/sec |
| Bind (⊗) | 69 μs | 14,500/sec |
| Bundle (⊕) 3 vectors | 41 μs | 24,400/sec |
| Antonym detection | 451 μs | 2,200/sec |
| Sentence encoding | 273 μs | **3,665/sec** |

**Memory**: 21 KB for entire basis (298 bytes/vector)

**vs. LLM inference**: 10,000–100,000× faster for equivalent semantic operations

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    8,192-dimensional space                   │
├─────────┬─────────┬─────────┬─────────┬─────────┬──────────┤
│  DIM 0  │  DIM 1  │  DIM 2  │  DIM 3  │  ...    │  DIM 7   │
│ SUBST.  │ QUANT.  │  EVAL.  │ MENTAL  │         │ LOGICAL  │
│ 1024d   │ 1024d   │ 1024d   │ 1024d   │         │  1024d   │
├─────────┴─────────┴─────────┴─────────┴─────────┴──────────┤
│  Hadamard partitions → EXACT orthogonality within dims     │
│  Bipolar pairs → cos(GOOD, BAD) = -1.0 EXACTLY             │
└─────────────────────────────────────────────────────────────┘

4 Scales:  WORD ──⊗──> SENTENCE ──⊗──> PARAGRAPH ──⊗──> TEXT
           73 prims    24 prims      15 prims       12 prims

Operations:
  ⊗ Bind    - role-filler binding (dissimilar to inputs)
  ⊕ Bundle  - superposition (similar to all inputs)
  ρ Permute - positional encoding (shift-based)
```

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from semantic_probing import HadamardBasis, AntonymDetectionProbe

# Generate the semantic basis (one-time, ~200ms)
basis = HadamardBasis().generate()

# Antonyms have cosine = -1.0 (exact, not approximate)
good = basis.get_primitive("GOOD")
bad = basis.get_primitive("BAD")
print(f"cos(GOOD, BAD) = {good.cosine(bad)}")  # -1.0

# Detect antonymy in arbitrary vectors
probe = AntonymDetectionProbe(basis)
result = probe.detect_antonym(good, bad)
print(f"Antonym pair: {result.bipolar_pair}")  # "EVALUATION"
```

### Solving Analogies

```python
from semantic_probing import HadamardBasis, SemanticProbe

basis = HadamardBasis().generate()
probe = SemanticProbe()

# GOOD:BAD :: ALIVE:?
good, bad = basis.get_primitive("GOOD"), basis.get_primitive("BAD")
alive = basis.get_primitive("ALIVE")

result = probe.compositional_analogy(good, bad, alive)

# Find nearest primitive to result
best_match = max(basis.primitives.items(), key=lambda x: result.cosine(x[1]))
print(f"Answer: {best_match[0]}")  # "DEAD" with cos = 1.0
```

### Multi-Scale Composition

```python
from semantic_probing import HadamardBasis, SentenceScaleBasis, bind, bundle

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

---

## The 62 NSM Primitives

Organized into 8 semantic dimensions:

| Dimension | Primitives |
|-----------|------------|
| **SUBSTANTIVES** | I, YOU, SOMEONE, SOMETHING, PEOPLE, BODY, KIND, PART + SAME/OTHER |
| **QUANTITY** | ONE, TWO, QUANTITY, NUMBER |
| **EVALUATORS** | GOOD/BAD, BIG/SMALL |
| **MENTAL** | THINK, KNOW, WANT, FEEL, SEE, HEAR, INFER |
| **ACTION** | SAY, WORDS, DO, HAPPEN, MOVE, BE, HAVE, EXIST + ALIVE/DEAD |
| **TEMPORAL** | MOMENT, WHEN + BEFORE/AFTER, THEN/NOW, START/END, LONG/SHORT_TIME |
| **SPATIAL** | WHERE, HERE, SIDE, TOUCH + ABOVE/BELOW, FAR/NEAR, INSIDE/OUTSIDE, TOWARD/AWAY, WITH/WITHOUT |
| **LOGICAL** | NOT, MAYBE, CAN, MUST, BECAUSE, CAUSE, IF, LIKE + TRUE/FALSE |

---

## Running Tests

```bash
# Verify mathematical properties
pytest tests/test_headline_results.py -v

# Run capability showcase
python tests/showcase_capabilities.py

# Performance benchmarks
python tests/benchmark_performance.py
```

---

## Theoretical Foundation

- **Natural Semantic Metalanguage (NSM)**: Wierzbicka's 62 universal primitives
- **Hyperdimensional Computing**: Kanerva's sparse distributed representations
- **Hadamard Matrices**: Walsh-Hadamard transform for guaranteed orthogonality

---

## Project Structure

```
src/semantic_probing/
├── encoding/          # Sparse ternary vectors, Hadamard basis
├── grounding/         # Text → vector mapping
├── probes/            # Antonym detection, analogy, similarity
├── analysis/          # Signatures, cross-model correlation
└── validation/        # SAE comparison utilities

tests/
├── test_headline_results.py   # Mathematical property verification
├── showcase_capabilities.py   # Capability demonstrations
└── benchmark_performance.py   # Performance benchmarks
```

---

## License

MIT
