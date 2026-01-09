"""
O(1) cache for semantic probe results using experience-memory.

This integration provides instant retrieval of previously computed
semantic analyses, dramatically reducing latency for repeated queries.
"""

import sys
import json
import hashlib
import time
from typing import Optional, Dict, Callable
from dataclasses import dataclass, asdict

# Local imports
from semantic_probing.encoding.sparse_ternary import HadamardBasis
from semantic_probing.probes.primitives import PrimitiveProbe
from semantic_probing.encoding.text_encoder import TextEncoder
from semantic_probing.analysis.signatures import SignatureAnalyzer

# Integration: experience-memory
sys.path.insert(0, "/Users/rohanvinaik/experience-memory")
from experience_memory import FixRegistry
from experience_memory import ErrorSignature, Fix, ErrorType, FixType, ErrorSeverity


@dataclass
class CachedSemanticResult:
    """Cached semantic analysis result."""
    dimension_profile: Dict[str, float]
    entropy: float
    primary_dimension: str
    active_primitives: list
    coherence_score: float
    compute_time_ms: float


class CachedSemanticProbe:
    """
    Semantic probe with O(1) caching via experience-memory.

    Provides significant speedup for repeated queries while
    maintaining full semantic analysis capability.
    """

    def __init__(self, cache_path: str = "/tmp/semantic_probe_cache.db"):
        # Semantic probing
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

        # Caching via experience-memory
        self.cache = FixRegistry(cache_path)
        self._stats = {
            "hits": 0,
            "misses": 0,
            "total_compute_ms": 0,
            "total_cache_ms": 0,
        }

    def _text_to_signature(self, text: str) -> ErrorSignature:
        """Convert text to a cache key signature."""
        from experience_memory import ErrorSeverity
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return ErrorSignature(
            severity=ErrorSeverity.MINOR,
            error_type=ErrorType.PARTIAL_DEFINITION,
            context=f"semantic:{text_hash}",
            affected_categories=["semantic_cache"],
            delta=0.0,
        )

    def _result_to_fix(self, result: CachedSemanticResult) -> Fix:
        """Convert result to cacheable Fix."""
        return Fix(
            fix_type=FixType.COMPLETE_DEFINITION,
            definition_supplement=json.dumps(asdict(result)),
        )

    def _fix_to_result(self, fix: Fix) -> CachedSemanticResult:
        """Convert Fix back to result."""
        data = json.loads(fix.definition_supplement)
        return CachedSemanticResult(**data)

    def analyze(self, text: str) -> CachedSemanticResult:
        """Analyze text, using cache when available."""
        start = time.perf_counter()

        # Check cache
        sig = self._text_to_signature(text)
        cached_fix = self.cache.lookup(sig)

        if cached_fix is not None:
            self._stats["hits"] += 1
            self._stats["total_cache_ms"] += (time.perf_counter() - start) * 1000
            return self._fix_to_result(cached_fix)

        # Cache miss - compute
        self._stats["misses"] += 1
        compute_start = time.perf_counter()

        vector = self.encoder.encode(text)
        analysis = self.analyzer.compute_signature(vector)

        compute_time = (time.perf_counter() - compute_start) * 1000
        self._stats["total_compute_ms"] += compute_time

        # Create result
        result = CachedSemanticResult(
            dimension_profile=analysis.dimension_profile,
            entropy=analysis.entropy,
            primary_dimension=analysis.primary_dimension,
            active_primitives=list(activations.keys())[:20],
            coherence_score=1.0 - min(1.0, analysis.entropy / 4.0),
            compute_time_ms=compute_time,
        )

        # Store in cache
        self.cache.register(sig, self._result_to_fix(result))

        return result

    def batch_analyze(self, texts: list) -> list:
        """Analyze multiple texts efficiently."""
        return [self.analyze(text) for text in texts]

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0

    @property
    def avg_cache_latency_ms(self) -> float:
        """Average cache hit latency."""
        if self._stats["hits"] == 0:
            return 0.0
        return self._stats["total_cache_ms"] / self._stats["hits"]

    @property
    def avg_compute_latency_ms(self) -> float:
        """Average compute latency."""
        if self._stats["misses"] == 0:
            return 0.0
        return self._stats["total_compute_ms"] / self._stats["misses"]

    @property
    def speedup(self) -> float:
        """Speedup factor from caching."""
        if self.avg_cache_latency_ms == 0:
            return 1.0
        return self.avg_compute_latency_ms / self.avg_cache_latency_ms

    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            **self._stats,
            "hit_rate": self.hit_rate,
            "avg_cache_latency_ms": self.avg_cache_latency_ms,
            "avg_compute_latency_ms": self.avg_compute_latency_ms,
            "speedup": self.speedup,
        }


def benchmark_caching():
    """Benchmark the caching system."""
    import random

    probe = CachedSemanticProbe("/tmp/semantic_cache_benchmark.db")

    # Generate test queries
    base_queries = [
        "What is the capital of France?",
        "Calculate 15 times 23.",
        "If A implies B and B implies C, what can we conclude?",
        "The sky is blue because of light scattering.",
        "Water boils at 100 degrees Celsius.",
    ]

    # Extend with variations
    queries = base_queries * 20  # 100 queries with repeats
    random.shuffle(queries)

    # Run benchmark
    print("Running cache benchmark...")
    for q in queries:
        probe.analyze(q)

    stats = probe.stats()
    print(f"\n=== Cache Benchmark Results ===")
    print(f"Total queries: {stats['hits'] + stats['misses']}")
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.1%}")
    print(f"Avg cache latency: {stats['avg_cache_latency_ms']:.3f}ms")
    print(f"Avg compute latency: {stats['avg_compute_latency_ms']:.3f}ms")
    print(f"Speedup: {stats['speedup']:.1f}x")


if __name__ == "__main__":
    benchmark_caching()
