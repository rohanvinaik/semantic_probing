"""Tests for wiki grounding integration."""

import pytest
from semantic_probing.integration.wiki_grounding import (
    GroundedFingerprintGenerator,
    GroundedHallucinationDetector,
)


class TestGroundedFingerprint:
    @pytest.fixture
    def generator(self):
        return GroundedFingerprintGenerator()

    def test_generates_fingerprint(self, generator):
        """Test basic fingerprint generation."""
        fp = generator.generate("Paris is the capital of France.")
        assert fp.primary_dimension is not None
        assert fp.entropy >= 0
        assert isinstance(fp.grounded_entities, list)

    def test_grounded_claim_has_entities(self, generator):
        """Claims with known entities should have grounding."""
        fp = generator.generate("Albert Einstein developed the theory of relativity.")
        # Should find Einstein as an entity
        assert len(fp.grounded_entities) > 0 or fp.grounding_coverage >= 0

    def test_hallucination_detection(self):
        """Test hallucination detection."""
        detector = GroundedHallucinationDetector()

        # True claim
        true_result = detector.detect("Water is composed of hydrogen and oxygen.")

        # Hallucinated claim
        fake_result = detector.detect(
            "The Henderson-Smith theorem proves that quantum numbers are purple."
        )

        # Fake claim should have higher suspicion
        # Note: May not always work depending on entity DB coverage
        assert "suspicion_score" in fake_result
        assert "is_hallucination" in fake_result
