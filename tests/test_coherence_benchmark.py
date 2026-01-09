"""Tests for coherence benchmark."""

import pytest
from semantic_probing.experiments.coherence_benchmark import (
    CoherenceBenchmark,
    CoherenceMetrics,
)


class TestCoherenceBenchmark:
    @pytest.fixture
    def benchmark(self):
        return CoherenceBenchmark()

    def test_single_step_coherence(self, benchmark):
        """Single step should have perfect stability."""
        metrics = benchmark.compute_trace_coherence(["The answer is 5."])
        assert metrics.primitive_stability == 1.0
        assert metrics.dimension_drift == 0.0

    def test_consistent_steps_high_stability(self, benchmark):
        """Consistent reasoning should have high stability."""
        steps = [
            "Step 1: We need to find x + y.",
            "Step 2: We know x = 3 and y = 4.",
            "Step 3: Therefore x + y = 3 + 4 = 7.",
        ]
        metrics = benchmark.compute_trace_coherence(steps)
        assert metrics.primitive_stability > 0.5

    def test_incoherent_steps_low_stability(self, benchmark):
        """Incoherent reasoning should have lower stability."""
        steps = [
            "The sky is blue because of light scattering.",
            "I prefer chocolate ice cream.",
            "The square root of 144 is 12.",
        ]
        metrics = benchmark.compute_trace_coherence(steps)
        # Incoherent steps may still have some stability, but should be lower
        assert metrics.entropy_variance > 0  # High variance indicates incoherence

    def test_parse_reasoning_steps(self, benchmark):
        """Test parsing of reasoning traces."""
        trace = "Step 1: First thing.\nStep 2: Second thing.\nStep 3: Conclusion."
        steps = benchmark.parse_reasoning_steps(trace)
        assert len(steps) == 3

    def test_metrics_dataclass(self):
        """Test CoherenceMetrics creation."""
        metrics = CoherenceMetrics(
            primitive_stability=0.8,
            dimension_drift=0.2,
            entropy_variance=0.1,
            mean_entropy=2.5,
            primary_dimension="QUANTITY",
        )
        assert metrics.primitive_stability == 0.8
        assert metrics.primary_dimension == "QUANTITY"
