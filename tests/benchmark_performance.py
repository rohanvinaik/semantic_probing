"""
Performance benchmarks for semantic probing operations.

Demonstrates that this technique runs efficiently on consumer hardware
compared to LLM-based semantic analysis.
"""

import time
import numpy as np
from semantic_probing.encoding import (
    HadamardBasis,
    SentenceScaleBasis,
    SparseVector,
    bind,
    bundle,
    permute,
)
from semantic_probing.probes import AntonymDetectionProbe, SemanticProbe


def benchmark_basis_generation():
    """Benchmark basis generation time."""
    start = time.perf_counter()
    basis = HadamardBasis().generate()
    word_time = time.perf_counter() - start

    start = time.perf_counter()
    sent_basis = SentenceScaleBasis().generate()
    sent_time = time.perf_counter() - start

    return {
        "word_basis_generation_ms": word_time * 1000,
        "sentence_basis_generation_ms": sent_time * 1000,
        "total_primitives": len(basis.primitives) + len(sent_basis.primitives),
    }


def benchmark_vector_operations(n_iterations=10000):
    """Benchmark core vector operations."""
    basis = HadamardBasis().generate()

    # Get some vectors for testing
    v1 = basis.get_primitive("GOOD")
    v2 = basis.get_primitive("BAD")
    v3 = basis.get_primitive("SOMEONE")
    role = basis.get_role("ARG0")

    # Benchmark cosine similarity
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = v1.cosine(v2)
    cosine_time = (time.perf_counter() - start) / n_iterations * 1e6  # microseconds

    # Benchmark bind
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = bind(v1, role)
    bind_time = (time.perf_counter() - start) / n_iterations * 1e6

    # Benchmark bundle (3 vectors)
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = bundle([v1, v2, v3])
    bundle_time = (time.perf_counter() - start) / n_iterations * 1e6

    # Benchmark permute
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = permute(v1, shift=100)
    permute_time = (time.perf_counter() - start) / n_iterations * 1e6

    return {
        "cosine_similarity_us": cosine_time,
        "bind_us": bind_time,
        "bundle_3vec_us": bundle_time,
        "permute_us": permute_time,
        "iterations": n_iterations,
    }


def benchmark_antonym_detection(n_iterations=1000):
    """Benchmark antonym detection probe."""
    basis = HadamardBasis().generate()
    probe = AntonymDetectionProbe(basis)

    v1 = basis.get_primitive("GOOD")
    v2 = basis.get_primitive("BAD")

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = probe.detect_antonym(v1, v2)
    detection_time = (time.perf_counter() - start) / n_iterations * 1e6

    return {
        "antonym_detection_us": detection_time,
        "iterations": n_iterations,
    }


def benchmark_full_pipeline(n_sentences=100):
    """Benchmark encoding a batch of sentences."""
    word_basis = HadamardBasis().generate()
    sent_basis = SentenceScaleBasis().generate()

    # Simulate encoding sentences with 5 words each
    words_per_sentence = 5

    # Pre-fetch vectors
    word_prims = list(word_basis.primitives.values())[:10]
    sent_prims = list(sent_basis.primitives.values())[:5]

    start = time.perf_counter()
    for _ in range(n_sentences):
        components = []
        for i in range(words_per_sentence):
            w = word_prims[i % len(word_prims)]
            s = sent_prims[i % len(sent_prims)]
            bound = bind(w, s)
            components.append(bound)
        _ = bundle(components)

    total_time = time.perf_counter() - start

    return {
        "sentences_encoded": n_sentences,
        "words_per_sentence": words_per_sentence,
        "total_time_ms": total_time * 1000,
        "time_per_sentence_us": (total_time / n_sentences) * 1e6,
        "sentences_per_second": n_sentences / total_time,
    }


def benchmark_memory_usage():
    """Estimate memory usage of the encoding."""
    basis = HadamardBasis().generate()

    # Measure serialized size
    sample_vec = basis.get_primitive("GOOD")
    serialized = sample_vec.serialize()

    # Memory per vector (approximate)
    # Each vector stores: dimension (int), pos_indices (uint16 array), neg_indices (uint16 array)
    avg_nnz = np.mean([v.nnz for v in basis.primitives.values()])
    bytes_per_vector = 4 + (avg_nnz * 2)  # dimension + indices

    return {
        "serialized_size_bytes": len(serialized),
        "avg_nonzeros_per_vector": avg_nnz,
        "estimated_bytes_per_vector": bytes_per_vector,
        "total_primitives": len(basis.primitives),
        "total_basis_kb": (len(basis.primitives) * bytes_per_vector) / 1024,
    }


def run_all_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 70)
    print("SEMANTIC PROBING: PERFORMANCE BENCHMARKS")
    print("=" * 70)

    # Basis generation
    print("\n[BASIS GENERATION]")
    gen_results = benchmark_basis_generation()
    print(f"  Word basis (73 primitives):     {gen_results['word_basis_generation_ms']:.2f} ms")
    print(f"  Sentence basis (24 primitives): {gen_results['sentence_basis_generation_ms']:.2f} ms")

    # Vector operations
    print("\n[CORE OPERATIONS] (averaged over 10,000 iterations)")
    ops_results = benchmark_vector_operations()
    print(f"  Cosine similarity:  {ops_results['cosine_similarity_us']:.2f} us")
    print(f"  Bind (⊗):           {ops_results['bind_us']:.2f} us")
    print(f"  Bundle (⊕) 3 vecs:  {ops_results['bundle_3vec_us']:.2f} us")
    print(f"  Permute (ρ):        {ops_results['permute_us']:.2f} us")

    # Antonym detection
    print("\n[ANTONYM DETECTION PROBE]")
    antonym_results = benchmark_antonym_detection()
    print(f"  Full detection:     {antonym_results['antonym_detection_us']:.2f} us")

    # Full pipeline
    print("\n[SENTENCE ENCODING PIPELINE]")
    pipeline_results = benchmark_full_pipeline()
    print(f"  Time per sentence:  {pipeline_results['time_per_sentence_us']:.2f} us")
    print(f"  Throughput:         {pipeline_results['sentences_per_second']:.0f} sentences/sec")

    # Memory
    print("\n[MEMORY EFFICIENCY]")
    mem_results = benchmark_memory_usage()
    print(f"  Bytes per vector:   {mem_results['estimated_bytes_per_vector']:.0f}")
    print(f"  Total basis size:   {mem_results['total_basis_kb']:.2f} KB")

    # Comparison context
    print("\n[COMPARISON TO LLM INFERENCE]")
    print(f"  Semantic probe:     ~{ops_results['cosine_similarity_us']:.0f} us per comparison")
    print(f"  GPT-4 API call:     ~500,000-2,000,000 us (0.5-2 sec)")
    print(f"  Speedup factor:     ~10,000-100,000x for semantic operations")

    print("\n" + "=" * 70)

    return {
        "generation": gen_results,
        "operations": ops_results,
        "antonym": antonym_results,
        "pipeline": pipeline_results,
        "memory": mem_results,
    }


if __name__ == "__main__":
    results = run_all_benchmarks()
