"""
Showcase: Non-Obvious Capabilities of Multi-Scale Semantic Encoding

These tests demonstrate properties that make this system useful for
interpretability research - not just toy examples, but mathematically
grounded capabilities with implications for probing neural representations.
"""

import numpy as np
from collections import defaultdict
from semantic_probing.encoding import (
    D,
    HadamardBasis,
    SentenceScaleBasis,
    SparseVector,
    bind,
    bundle,
    permute,
    BIPOLAR_PAIRS,
    QUANTITY_HIERARCHY,
    PRIMITIVE_REGISTRY,
    SemanticDimension,
)
from semantic_probing.probes import AntonymDetectionProbe, SemanticProbe


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def showcase_bipolar_analogy():
    """
    SHOWCASE 1: Bipolar Analogy Solving

    The encoding solves semantic analogies using vector arithmetic.
    For bipolar pairs, we get EXACT solutions (not approximate).

    Example: GOOD:BAD :: ALIVE:? → DEAD (with cos=1.0 to target)
    """
    print_header("BIPOLAR ANALOGY SOLVING")

    basis = HadamardBasis().generate()
    probe = SemanticProbe()

    # Test several analogies
    analogies = [
        ("GOOD", "BAD", "ALIVE", "DEAD"),      # Evaluation → Vitality
        ("GOOD", "BAD", "TRUE", "FALSE"),      # Evaluation → Truth
        ("BEFORE", "AFTER", "ABOVE", "BELOW"), # Temporal → Spatial
        ("BIG", "SMALL", "FAR", "NEAR"),       # Size → Distance
        ("START", "END", "GOOD", "BAD"),       # Temporal → Evaluation
    ]

    print("\n  Solving: A:B :: C:? using vector arithmetic")
    print("  Method: result = C ⊕ B ⊕ (-A), then find nearest to result")
    print()

    correct = 0
    for a, b, c, expected_d in analogies:
        va = basis.get_primitive(a)
        vb = basis.get_primitive(b)
        vc = basis.get_primitive(c)
        vd_expected = basis.get_primitive(expected_d)

        # Solve analogy: A:B :: C:D means D = C + (B - A)
        # For bipolar: if B = -A, then D = -C
        result = probe.compositional_analogy(va, vb, vc)

        # Check similarity to expected answer
        cos_to_expected = result.cosine(vd_expected)

        # Find actual nearest neighbor
        best_match = None
        best_cos = -2
        for name, vec in basis.primitives.items():
            cos = result.cosine(vec)
            if cos > best_cos:
                best_cos = cos
                best_match = name

        is_correct = best_match == expected_d
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"  {status} {a}:{b} :: {c}:? → {best_match} (expected {expected_d})")
        print(f"      cos(result, {expected_d}) = {cos_to_expected:+.4f}")

    print(f"\n  Accuracy: {correct}/{len(analogies)} = {100*correct/len(analogies):.0f}%")
    return correct, len(analogies)


def showcase_superposition_capacity():
    """
    SHOWCASE 2: Superposition Capacity with Perfect Retrieval

    How many concepts can be bundled together while maintaining
    >90% retrieval accuracy? This is crucial for interpretability -
    it determines how many features can be packed into a representation.
    """
    print_header("SUPERPOSITION CAPACITY")

    basis = HadamardBasis().generate()
    primitives = list(basis.primitives.items())

    print("\n  Testing: Bundle N random primitives, measure retrieval accuracy")
    print("  Retrieval = bundled vector has highest cosine with original primitives")
    print()

    results = []

    for n_bundle in [2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60]:
        # Run multiple trials
        n_trials = 50
        accuracies = []

        for _ in range(n_trials):
            # Sample random primitives
            indices = np.random.choice(len(primitives), n_bundle, replace=False)
            selected = [primitives[i] for i in indices]
            names = [name for name, _ in selected]
            vectors = [vec for _, vec in selected]

            # Bundle them
            bundled = bundle(vectors)

            # Check retrieval: for each bundled primitive, is it in top-N matches?
            retrieved = 0
            for name, vec in selected:
                # Find rank of this primitive
                cos = bundled.cosine(vec)
                rank = sum(1 for _, v in primitives if bundled.cosine(v) > cos)
                if rank < n_bundle:  # In top-N
                    retrieved += 1

            accuracies.append(retrieved / n_bundle)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results.append((n_bundle, mean_acc, std_acc))

        bar = "█" * int(mean_acc * 20)
        print(f"  N={n_bundle:2d}: {bar:20s} {mean_acc*100:5.1f}% ± {std_acc*100:.1f}%")

    # Find capacity at 90% threshold
    capacity_90 = max([n for n, acc, _ in results if acc >= 0.90], default=0)
    capacity_80 = max([n for n, acc, _ in results if acc >= 0.80], default=0)

    print(f"\n  Capacity at ≥90% accuracy: {capacity_90} concepts")
    print(f"  Capacity at ≥80% accuracy: {capacity_80} concepts")

    return results


def showcase_hierarchical_ordering():
    """
    SHOWCASE 3: Hierarchical Semantic Ordering

    The encoding preserves ordinal relationships in hierarchies.
    MANY > MORE > SOME > FEW > NONE should be reflected in vector space.
    """
    print_header("HIERARCHICAL SEMANTIC ORDERING")

    basis = HadamardBasis().generate()

    # Get hierarchy primitives
    hier_base = basis.get_primitive("QUANTITY")
    hier_plus = basis.get_primitive("HIER_PLUS")
    hier_minus = basis.get_primitive("HIER_MINUS")

    print("\n  Testing: Quantity hierarchy encoding")
    print("  Method: MANY = QUANTITY ⊕ [+][+], SOME = QUANTITY ⊕ [-], etc.")
    print()

    # Construct hierarchy levels by bundling markers
    def make_level(n_markers, positive=True):
        components = [hier_base]
        marker = hier_plus if positive else hier_minus
        for _ in range(abs(n_markers)):
            components.append(marker)
        return bundle(components)

    levels = {
        "ALL":    make_level(3, True),   # [+][+][+][Q]
        "MANY":   make_level(2, True),   # [+][+][Q]
        "MORE":   make_level(1, True),   # [+][Q]
        "SOME":   make_level(0, True),   # [Q] (base)
        "FEW":    make_level(1, False),  # [-][Q]
        "NONE":   make_level(3, False),  # [-][-][-][Q]
    }

    # Check ordinal relationships via cosine to base
    print("  Level        cos(level, QUANTITY)    Relative ordering")
    print("  " + "-" * 55)

    cosines = {}
    for name in ["ALL", "MANY", "MORE", "SOME", "FEW", "NONE"]:
        vec = levels[name]
        cos = vec.cosine(hier_base)
        cosines[name] = cos
        bar_len = int((cos + 1) * 15)  # Scale -1 to +1 to 0-30
        bar = "█" * bar_len
        print(f"  {name:8s}     {cos:+.4f}              {bar}")

    # Verify ordering
    order = ["ALL", "MANY", "MORE", "SOME", "FEW", "NONE"]
    ordering_preserved = all(
        cosines[order[i]] >= cosines[order[i+1]] - 0.01  # Small tolerance
        for i in range(len(order)-1)
    )

    print(f"\n  Ordering preserved: {'✓ YES' if ordering_preserved else '✗ NO'}")

    return cosines, ordering_preserved


def showcase_compositional_unbinding():
    """
    SHOWCASE 4: Compositional Unbinding (Information Recovery)

    If we bind a concept with a role, can we recover the original?
    This is critical for interpretability: extracting "what" from "who did what"
    """
    print_header("COMPOSITIONAL UNBINDING")

    basis = HadamardBasis().generate()

    print("\n  Testing: Bind concept with role, then unbind to recover")
    print("  Method: bound = concept ⊗ role")
    print("          recovered = bound ⊗ role  (since role ⊗ role ≈ identity)")
    print()

    # Test with several concept-role pairs
    tests = [
        ("SOMEONE", "ARG0"),   # Agent
        ("SOMETHING", "ARG1"), # Patient
        ("GOOD", "MOD"),       # Modifier
        ("DO", "ARG0"),        # Action as agent
    ]

    results = []
    for concept_name, role_name in tests:
        concept = basis.get_primitive(concept_name)
        role = basis.get_role(role_name)

        # Bind
        bound = bind(concept, role)

        # Unbind: in HDC, binding with same vector again recovers original
        # because role ⊗ role has high self-similarity
        recovered = bind(bound, role)

        # Measure recovery quality
        cos_to_original = recovered.cosine(concept)

        # Find best match
        best_match = None
        best_cos = -2
        for name, vec in basis.primitives.items():
            cos = recovered.cosine(vec)
            if cos > best_cos:
                best_cos = cos
                best_match = name

        is_recovered = best_match == concept_name
        results.append((concept_name, role_name, cos_to_original, is_recovered))

        status = "✓" if is_recovered else "✗"
        print(f"  {status} {concept_name} ⊗ {role_name} ⊗ {role_name} → {best_match}")
        print(f"      cos(recovered, {concept_name}) = {cos_to_original:+.4f}")

    accuracy = sum(1 for _, _, _, r in results if r) / len(results)
    print(f"\n  Recovery accuracy: {accuracy*100:.0f}%")

    return results


def showcase_logical_primitive_detection():
    """
    SHOWCASE 5: Logical Primitive Detection in Composed Structures

    Can we detect the presence of logical primitives (IF, BECAUSE, NOT)
    in complex bundled representations? This is key for interpretability.
    """
    print_header("LOGICAL PRIMITIVE DETECTION")

    basis = HadamardBasis().generate()
    sent_basis = SentenceScaleBasis().generate()

    print("\n  Testing: Detect logical primitives in composed structures")
    print("  Method: Create bundles with/without logical prims, measure detection")
    print()

    # Logical primitives
    logical_prims = ["NOT", "IF", "BECAUSE", "MAYBE", "MUST", "CAN"]

    # Create test cases: bundles with and without logical primitives
    content_prims = ["SOMEONE", "DO", "SOMETHING", "GOOD"]
    content_vecs = [basis.get_primitive(p) for p in content_prims]
    content_bundle = bundle(content_vecs)

    print("  Base content: SOMEONE + DO + SOMETHING + GOOD")
    print()

    for log_name in logical_prims:
        log_vec = basis.get_primitive(log_name)
        if log_vec is None:
            continue

        # Create bundle WITH logical primitive
        with_logical = bundle(content_vecs + [log_vec])

        # Detection via cosine similarity
        cos_with = with_logical.cosine(log_vec)
        cos_without = content_bundle.cosine(log_vec)

        # Detection threshold: if cos > 0.15, primitive is present
        detected_with = cos_with > 0.15
        detected_without = cos_without > 0.15

        status = "✓" if (detected_with and not detected_without) else "⚠"
        print(f"  {status} {log_name:8s}: with={cos_with:+.3f}, without={cos_without:+.3f}")
        print(f"              Detected in bundle: {'YES' if detected_with else 'NO'}, "
              f"False positive: {'YES' if detected_without else 'NO'}")

    return True


def showcase_cross_dimension_independence():
    """
    SHOWCASE 6: Cross-Dimension Independence

    Primitives from different semantic dimensions should be independent,
    allowing orthogonal variation. This is crucial for disentanglement.
    """
    print_header("CROSS-DIMENSION INDEPENDENCE")

    basis = HadamardBasis().generate()

    print("\n  Testing: Independence between semantic dimensions")
    print("  Method: Mean |cosine| between primitives of different dimensions")
    print()

    # Group by dimension
    dim_prims = defaultdict(list)
    for name, info in PRIMITIVE_REGISTRY.items():
        if name in basis.primitives:
            dim_prims[info.dimension].append((name, basis.get_primitive(name)))

    # Compute cross-dimension cosines
    dims = list(SemanticDimension)

    print("  " + " " * 14 + "".join(f"{d.name[:4]:>6}" for d in dims))

    matrix = np.zeros((len(dims), len(dims)))

    for i, dim1 in enumerate(dims):
        row = f"  {dim1.name[:12]:12s}"
        for j, dim2 in enumerate(dims):
            if i == j:
                matrix[i,j] = 1.0
                row += "   -- "
            else:
                cosines = []
                for n1, v1 in dim_prims[dim1][:5]:
                    for n2, v2 in dim_prims[dim2][:5]:
                        cosines.append(abs(v1.cosine(v2)))
                mean_cos = np.mean(cosines) if cosines else 0
                matrix[i,j] = mean_cos
                row += f" {mean_cos:.3f}"
        print(row)

    # Off-diagonal mean
    off_diag = []
    for i in range(len(dims)):
        for j in range(len(dims)):
            if i != j:
                off_diag.append(matrix[i,j])

    mean_cross = np.mean(off_diag)
    max_cross = np.max(off_diag)

    print(f"\n  Mean cross-dimension |cos|: {mean_cross:.6f}")
    print(f"  Max cross-dimension |cos|:  {max_cross:.6f}")
    print(f"  Independence quality: {'EXCELLENT' if mean_cross < 0.01 else 'GOOD' if mean_cross < 0.05 else 'MODERATE'}")

    return mean_cross, max_cross


def showcase_sequence_encoding():
    """
    SHOWCASE 7: Position-Invariant Sequence Encoding

    Using permutation to encode position, we can represent sequences
    and recover elements by position - useful for attention analysis.
    """
    print_header("SEQUENCE ENCODING WITH PERMUTATION")

    basis = HadamardBasis().generate()

    print("\n  Testing: Encode sequence with positional permutation")
    print("  Method: seq = Σᵢ ρⁱ(wordᵢ), query position via ρ⁻ⁱ")
    print()

    # Encode a sequence: "SOMEONE DO SOMETHING"
    words = ["SOMEONE", "DO", "SOMETHING"]
    word_vecs = [basis.get_primitive(w) for w in words]

    # Encode with position
    positioned = []
    for i, vec in enumerate(word_vecs):
        positioned.append(permute(vec, shift=i * 100))  # Large shift for distinctness

    sequence = bundle(positioned)

    print(f"  Sequence: {' → '.join(words)}")
    print(f"  Encoding: ρ⁰(SOMEONE) ⊕ ρ¹⁰⁰(DO) ⊕ ρ²⁰⁰(SOMETHING)")
    print()

    # Query each position
    print("  Position queries:")
    for i, expected_word in enumerate(words):
        # Query by inverse permutation
        query_result = permute(sequence, shift=-(i * 100))

        # Find best match
        best_match = None
        best_cos = -2
        for name, vec in basis.primitives.items():
            cos = query_result.cosine(vec)
            if cos > best_cos:
                best_cos = cos
                best_match = name

        status = "✓" if best_match == expected_word else "✗"
        print(f"    {status} Position {i}: query → {best_match} (expected {expected_word}), cos={best_cos:.3f}")

    return True


def run_all_showcases():
    """Run all showcase demonstrations."""
    print("\n" + "=" * 70)
    print("  SEMANTIC PROBING: CAPABILITY SHOWCASE")
    print("  Demonstrating Non-Obvious Properties for Interpretability Research")
    print("=" * 70)

    results = {}

    # 1. Bipolar Analogies
    correct, total = showcase_bipolar_analogy()
    results['analogy_accuracy'] = correct / total

    # 2. Superposition Capacity
    capacity_results = showcase_superposition_capacity()
    results['superposition'] = capacity_results

    # 3. Hierarchical Ordering
    cosines, ordering_preserved = showcase_hierarchical_ordering()
    results['hierarchy_preserved'] = ordering_preserved

    # 4. Compositional Unbinding
    unbind_results = showcase_compositional_unbinding()
    results['unbinding_accuracy'] = sum(1 for _, _, _, r in unbind_results if r) / len(unbind_results)

    # 5. Logical Primitive Detection
    showcase_logical_primitive_detection()

    # 6. Cross-Dimension Independence
    mean_cross, max_cross = showcase_cross_dimension_independence()
    results['cross_dim_independence'] = mean_cross

    # 7. Sequence Encoding
    showcase_sequence_encoding()

    # Summary
    print_header("SUMMARY OF CAPABILITIES")
    print(f"""
  For interpretability research, this encoding provides:

  1. ANALOGY SOLVING:        {results['analogy_accuracy']*100:.0f}% accuracy on bipolar analogies
  2. SUPERPOSITION:          {max([n for n, acc, _ in results['superposition'] if acc >= 0.90])} concepts at ≥90% retrieval
  3. HIERARCHY:              {'Preserved' if results['hierarchy_preserved'] else 'Not preserved'}
  4. UNBINDING:              {results['unbinding_accuracy']*100:.0f}% concept recovery
  5. CROSS-DIM INDEPENDENCE: {results['cross_dim_independence']:.6f} mean |cos|

  These properties enable:
  - Probing for specific semantic features in neural activations
  - Decomposing composed representations into constituents
  - Detecting logical/causal structure in model internals
  - Analyzing how models encode hierarchical relationships
""")

    return results


if __name__ == "__main__":
    results = run_all_showcases()
