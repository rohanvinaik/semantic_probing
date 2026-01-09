#!/usr/bin/env python3
"""
Demonstration: Antonym detection via bipolar semantic dimensions.
Shows cos(GOOD, BAD) = -1.0 structural property and solves bipolar analogies.
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from semantic_probing import (
    HadamardBasis, 
    BIPOLAR_PAIRS, 
    SparseVector, 
    AntonymDetectionProbe, 
    SemanticProbe,
    bundle
)

def main():
    print("ANTONYM DETECTION & BIPOLAR ANALOGY DEMO")
    print("=" * 60)

    # 1. Structural Validation
    print("\n--- TEST 1: Structural Validation ---")
    basis = HadamardBasis().generate()

    for pair in BIPOLAR_PAIRS:
        pos_vec = basis.get_primitive(pair.positive)
        neg_vec = basis.get_primitive(pair.negative)
        if pos_vec and neg_vec:
            cos = pos_vec.cosine(neg_vec)
            status = "✅" if abs(cos + 1.0) < 0.001 else "❌"
            print(f"  {pair.positive:12} <-> {pair.negative:12} : cos = {cos:+.3f} [{status}]")

    # 2. Antonym Detection
    print("\n--- TEST 2: Antonym Detection Probe ---")
    probe = AntonymDetectionProbe(basis)
    
    good = basis.get_primitive("GOOD")
    bad = basis.get_primitive("BAD")
    
    result = probe.detect_antonym(good, bad, "GOOD", "BAD")
    print(f"Query: GOOD vs BAD")
    print(f"  Is Antonym: {result.is_antonym}")
    print(f"  Explanation: {result.explanation}")
    print(f"  Confidence: {result.confidence:.3f}")

    # 3. Compositional Analogy
    print("\n--- TEST 3: Compositional Analogy (The Fix) ---")
    semantic_probe = SemanticProbe()
    
    # Query: GOOD is to BAD as TRUE is to ? 
    # Standard: TRUE + (BAD - GOOD) = TRUE + (-1 - 1) = TRUE - 2?? No, vectors don't work like scalar math exactly.
    # It failed before because D = C + B - A. If B = -A, then D = C - 2A.
    # We want D = FALSE = -TRUE.
    
    true_vec = basis.get_primitive("TRUE")
    
    print(f"Analogy: GOOD : BAD :: TRUE : ?")
    
    # Run compositional analogy
    target = semantic_probe.compositional_analogy(good, bad, true_vec)
    
    # Check against expected FALSE
    false_vec = basis.get_primitive("FALSE")
    sim = target.cosine(false_vec)
    
    print(f"  Target Cosine with FALSE: {sim:+.3f}")
    if sim > 0.9:
        print("  ✅ Analogy Solved: TRUE -> FALSE via bipolar negation.")
    else:
        print("  ❌ Analogy Failed.")


if __name__ == "__main__":
    main()
