"""
Combine semantic primitives with entity grounding for richer fingerprints.

This integration enhances semantic signatures with grounded entity information
from sparse-wiki-grounding, creating a more powerful hallucination detector.
"""

import sys
from typing import List, Dict, Optional
from dataclasses import dataclass

# Local imports
from semantic_probing.encoding.sparse_ternary import HadamardBasis
from semantic_probing.probes.primitives import PrimitiveProbe
from semantic_probing.encoding.text_encoder import TextEncoder
from semantic_probing.analysis.signatures import SignatureAnalyzer

# Integration: sparse-wiki-grounding
sys.path.insert(0, "/Users/rohanvinaik/sparse-wiki-grounding")
from wiki_grounding.store import EntityStore
from wiki_grounding.spreading import SpreadingActivation


@dataclass
class GroundedFingerprint:
    """A semantic fingerprint enhanced with entity grounding."""
    # Semantic signature
    dimension_profile: Dict[str, float]
    primary_dimension: str
    entropy: float
    active_primitives: List[str]

    # Entity grounding
    grounded_entities: List[str]
    entity_positions: List[Dict[str, float]]  # Position in 5D space
    entity_epa: List[Dict[str, float]]        # Evaluation-Potency-Activity
    grounding_coverage: float                  # % of claims grounded

    # Combined metrics
    semantic_entity_alignment: float  # How well semantic and entity info align


class GroundedFingerprintGenerator:
    """Generate fingerprints combining semantic probing with entity grounding."""

    def __init__(self, entity_db_path: str = None):
        # Semantic probing
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

        # Entity grounding
        db_path = entity_db_path or "/Users/rohanvinaik/sparse-wiki-grounding/data/wiki_grounding.db"
        self.entity_store = EntityStore(db_path)
        self.spreader = SpreadingActivation(self.entity_store)

    def generate(self, text: str) -> GroundedFingerprint:
        """Generate a grounded fingerprint for text."""
        # 1. Compute semantic signature
        vector = self.encoder.encode(text)
        sig = self.analyzer.compute_signature(vector)

        # 2. Extract and ground entities
        entities = self.entity_store.search(text, limit=10)

        entity_names = [e.name for e in entities]
        entity_positions = [
            {
                "spatial": e.positions.spatial,
                "temporal": e.positions.temporal,
                "taxonomic": e.positions.taxonomic,
                "scale": e.positions.scale,
                "domain": e.positions.domain,
            }
            for e in entities
        ]
        entity_epa = [
            {
                "evaluation": e.epa.evaluation,
                "potency": e.epa.potency,
                "activity": e.epa.activity,
            }
            for e in entities
        ]

        # 3. Compute grounding coverage
        # How many important words in text are grounded?
        words = set(text.lower().split())
        grounded_words = set(name.lower() for name in entity_names)
        coverage = len(words & grounded_words) / len(words) if words else 0

        # 4. Compute semantic-entity alignment
        alignment = self._compute_alignment(sig.dimension_profile, entity_positions)

        return GroundedFingerprint(
            dimension_profile=sig.dimension_profile,
            primary_dimension=sig.primary_dimension,
            entropy=sig.entropy,
            active_primitives=list(activations.keys())[:10],
            grounded_entities=entity_names,
            entity_positions=entity_positions,
            entity_epa=entity_epa,
            grounding_coverage=coverage,
            semantic_entity_alignment=alignment,
        )

    def _compute_alignment(
        self,
        dim_profile: Dict[str, float],
        positions: List[Dict[str, float]],
    ) -> float:
        """Compute alignment between semantic dimensions and entity positions."""
        if not positions:
            return 0.0

        # Heuristic alignment rules:
        # - SPATIAL dimension should correlate with spatial position variance
        # - TEMPORAL dimension should correlate with temporal position variance
        # - QUANTITY dimension should correlate with scale position
        alignment_score = 0.0
        n_checks = 0

        # Spatial alignment
        if "SPATIAL" in dim_profile and len(positions) > 1:
            spatial_vals = [p["spatial"] for p in positions]
            spatial_var = sum((v - sum(spatial_vals)/len(spatial_vals))**2 for v in spatial_vals)
            # High spatial dimension + high spatial variance = good alignment
            alignment_score += dim_profile["SPATIAL"] * min(1.0, spatial_var)
            n_checks += 1

        # Temporal alignment
        if "TEMPORAL" in dim_profile and len(positions) > 1:
            temporal_vals = [p["temporal"] for p in positions]
            temporal_var = sum((v - sum(temporal_vals)/len(temporal_vals))**2 for v in temporal_vals)
            alignment_score += dim_profile["TEMPORAL"] * min(1.0, temporal_var)
            n_checks += 1

        return alignment_score / n_checks if n_checks > 0 else 0.0


class GroundedHallucinationDetector:
    """Detect hallucinations using grounded fingerprints."""

    def __init__(self, entity_db_path: str = None):
        self.generator = GroundedFingerprintGenerator(entity_db_path)

    def detect(self, claim: str) -> Dict:
        """Detect if a claim is likely hallucinated."""
        fp = self.generator.generate(claim)

        # Detection heuristics:
        # 1. Low grounding coverage + high confidence = suspicious
        # 2. High entropy + low alignment = suspicious
        # 3. Specific claims (low entropy) with no grounding = suspicious

        suspicion_score = 0.0

        # Check 1: Ungrounded confident claims
        if fp.grounding_coverage < 0.1 and fp.entropy < 2.0:
            suspicion_score += 0.4

        # Check 2: High entropy + poor alignment
        if fp.entropy > 3.0 and fp.semantic_entity_alignment < 0.3:
            suspicion_score += 0.3

        # Check 3: Claims about entities that don't exist
        if len(fp.grounded_entities) == 0:
            suspicion_score += 0.3

        is_hallucination = suspicion_score > 0.5

        return {
            "is_hallucination": is_hallucination,
            "suspicion_score": suspicion_score,
            "grounding_coverage": fp.grounding_coverage,
            "semantic_entropy": fp.entropy,
            "alignment": fp.semantic_entity_alignment,
            "grounded_entities": fp.grounded_entities,
        }
