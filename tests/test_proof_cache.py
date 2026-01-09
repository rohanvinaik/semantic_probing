"""Tests for proof cache integration."""

import pytest
import tempfile
from semantic_probing.integration.experience_memory import (
    CachedSemanticProbe,
    CachedSemanticResult,
)


class TestCachedSemanticProbe:
    @pytest.fixture
    def probe(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            yield CachedSemanticProbe(f.name)

    def test_basic_analysis(self, probe):
        """Test basic semantic analysis."""
        result = probe.analyze("What is 2 + 2?")
        assert result.primary_dimension is not None
        assert result.entropy >= 0
        assert result.coherence_score >= 0

    def test_cache_hit(self, probe):
        """Second query should be a cache hit."""
        query = "The sum of two and three is five."

        # First query - miss
        result1 = probe.analyze(query)
        assert probe._stats["misses"] == 1
        assert probe._stats["hits"] == 0

        # Second query - hit
        result2 = probe.analyze(query)
        assert probe._stats["hits"] == 1

        # Results should match
        assert result1.primary_dimension == result2.primary_dimension
        assert result1.entropy == result2.entropy

    def test_cache_speedup(self, probe):
        """Cache should provide speedup."""
        query = "A complex query about mathematics and logic."

        # Warm up
        probe.analyze(query)

        # Measure
        import time
        n = 100

        # Cache hits
        start = time.perf_counter()
        for _ in range(n):
            probe.analyze(query)
        cache_time = time.perf_counter() - start

        # The cache time should be very fast
        avg_cache_ms = (cache_time / n) * 1000
        assert avg_cache_ms < 10  # Should be well under 10ms

    def test_stats(self, probe):
        """Test statistics tracking."""
        probe.analyze("Query 1")
        probe.analyze("Query 2")
        probe.analyze("Query 1")  # Repeat

        stats = probe.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(1/3)
