"""
Semantic Probing: Interpretable semantic encoding and probing.

This library provides tools for inspecting the semantic content of
sparse vector representations, grounded in linguistic and world knowledge.
"""

from .encoding import (
    SparseVector,
    HadamardBasis,
    BIPOLAR_PAIRS,
    BipolarPair,
    bind,
    bundle,
    permute,
)
from .grounding import (
    LinguisticGrounder,
    EntityGrounder,
    GroundingResult,
    LinguisticProfile,
    EntityProfile,
    GroundedEntity,
    EntityEnhancement,
)
from .probes import (
    PrimitiveProbe,
    SemanticProbe,
    AntonymDetectionProbe,
    ReasoningProbe,
)
from .analysis import (
    SemanticSignature,
    SignatureAnalyzer,
    CorrelationAnalyzer,
)

__version__ = "0.1.0"

__all__ = [
    "SparseVector",
    "HadamardBasis",
    "BIPOLAR_PAIRS",
    "BipolarPair",
    "bind",
    "bundle",
    "permute",
    "LinguisticGrounder",
    "EntityGrounder",
    "GroundingResult",
    "LinguisticProfile",
    "EntityProfile",
    "GroundedEntity",
    "EntityEnhancement",
    "PrimitiveProbe",
    "SemanticProbe",
    "ReasoningProbe",
    "SemanticSignature",
    "SignatureAnalyzer",
    "CorrelationAnalyzer",
]
