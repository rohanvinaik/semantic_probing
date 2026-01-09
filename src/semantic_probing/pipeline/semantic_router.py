"""
Fuzzy semantic router for directing queries to appropriate specialists.

Routes based on semantic dimension profile rather than explicit rules,
enabling emergent coordination without a homunculus.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from semantic_probing.encoding.sparse_ternary import HadamardBasis
from semantic_probing.probes.primitives import PrimitiveProbe
from semantic_probing.encoding.text_encoder import TextEncoder
from semantic_probing.analysis.signatures import SignatureAnalyzer


class SpecialistType(Enum):
    """Available specialist types."""
    LOGICAL = "logical"
    QUANTITY = "quantity"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    MENTAL = "mental"
    GROUNDING = "grounding"  # Wiki-based fact checking
    GENERALIST = "generalist"  # Fallback


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    primary_specialist: SpecialistType
    secondary_specialists: List[SpecialistType]
    dimension_profile: Dict[str, float]
    confidence: float
    reasoning: str


class SemanticRouter:
    """
    Route queries to specialists based on semantic dimension profile.

    No explicit rules - uses fuzzy matching of semantic signatures
    to determine which specialists should handle a query.
    """

    # Dimension to specialist mapping
    DIMENSION_TO_SPECIALIST = {
        "LOGICAL": SpecialistType.LOGICAL,
        "QUANTITY": SpecialistType.QUANTITY,
        "TEMPORAL": SpecialistType.TEMPORAL,
        "SPATIAL": SpecialistType.SPATIAL,
        "MENTAL": SpecialistType.MENTAL,
    }

    # Thresholds
    PRIMARY_THRESHOLD = 0.4  # Minimum score for primary specialist
    SECONDARY_THRESHOLD = 0.2  # Minimum score for secondary
    GROUNDING_THRESHOLD = 0.3  # When to include grounding specialist

    def __init__(self):
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

    def route(self, query: str) -> RoutingDecision:
        """Route a query to appropriate specialists."""
        # Compute semantic signature
        vector = self.encoder.encode(query)
        sig = self.analyzer.compute_signature(vector)

        dim_profile = sig.dimension_profile

        # Sort dimensions by score
        sorted_dims = sorted(
            dim_profile.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Determine primary specialist
        primary = SpecialistType.GENERALIST
        primary_score = 0.0

        if sorted_dims and sorted_dims[0][1] >= self.PRIMARY_THRESHOLD:
            dim_name = sorted_dims[0][0]
            if dim_name in self.DIMENSION_TO_SPECIALIST:
                primary = self.DIMENSION_TO_SPECIALIST[dim_name]
                primary_score = sorted_dims[0][1]

        # Determine secondary specialists
        secondary = []
        for dim_name, score in sorted_dims[1:]:
            if score >= self.SECONDARY_THRESHOLD:
                if dim_name in self.DIMENSION_TO_SPECIALIST:
                    secondary.append(self.DIMENSION_TO_SPECIALIST[dim_name])

        # Add grounding specialist for factual queries
        if self._needs_grounding(query, dim_profile):
            if SpecialistType.GROUNDING not in secondary:
                secondary.append(SpecialistType.GROUNDING)

        # Compute confidence
        confidence = primary_score if primary != SpecialistType.GENERALIST else 0.5

        # Generate reasoning
        reasoning = self._generate_reasoning(primary, secondary, sorted_dims)

        return RoutingDecision(
            primary_specialist=primary,
            secondary_specialists=secondary,
            dimension_profile=dim_profile,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _needs_grounding(self, query: str, profile: Dict[str, float]) -> bool:
        """Determine if query needs entity grounding."""
        # Heuristics for grounding:
        # 1. Contains proper nouns (capital letters mid-sentence)
        # 2. Contains specific claims
        # 3. Low LOGICAL but high SUBSTANTIVES

        words = query.split()
        has_proper_nouns = any(
            w[0].isupper() and i > 0
            for i, w in enumerate(words)
            if w and w[0].isalpha()
        )

        substantive_heavy = profile.get("SUBSTANTIVES", 0) > self.GROUNDING_THRESHOLD
        low_logical = profile.get("LOGICAL", 0) < 0.3

        return has_proper_nouns or (substantive_heavy and low_logical)

    def _generate_reasoning(
        self,
        primary: SpecialistType,
        secondary: List[SpecialistType],
        sorted_dims: List[Tuple[str, float]],
    ) -> str:
        """Generate human-readable routing reasoning."""
        top_dims = [f"{d}={s:.2f}" for d, s in sorted_dims[:3]]

        if primary == SpecialistType.GENERALIST:
            return f"No dominant dimension ({', '.join(top_dims)}). Using generalist."

        secondary_str = ", ".join(s.value for s in secondary) if secondary else "none"
        return (
            f"Primary dimension: {sorted_dims[0][0]} ({sorted_dims[0][1]:.2f}). "
            f"Routing to {primary.value} specialist. "
            f"Secondary specialists: {secondary_str}."
        )


class MultiSpecialistOrchestrator:
    """Orchestrate multiple specialists for complex queries."""

    def __init__(self, specialists: Dict[SpecialistType, object] = None):
        self.router = SemanticRouter()
        self.specialists = specialists or {}

    def process(self, query: str) -> Dict:
        """Process query through appropriate specialists."""
        # Route
        decision = self.router.route(query)

        # Collect responses
        responses = {}

        # Primary specialist
        if decision.primary_specialist in self.specialists:
            specialist = self.specialists[decision.primary_specialist]
            responses["primary"] = {
                "specialist": decision.primary_specialist.value,
                "response": specialist.process(query),
            }

        # Secondary specialists
        for spec_type in decision.secondary_specialists:
            if spec_type in self.specialists:
                specialist = self.specialists[spec_type]
                responses[spec_type.value] = specialist.process(query)

        return {
            "routing": {
                "primary": decision.primary_specialist.value,
                "secondary": [s.value for s in decision.secondary_specialists],
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
            },
            "responses": responses,
            "dimension_profile": decision.dimension_profile,
        }
