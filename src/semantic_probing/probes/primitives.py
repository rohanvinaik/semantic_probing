"""Primitive probing logic."""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..encoding import (
    SparseVector,
    HadamardBasis,
    PRIMITIVE_REGISTRY,
    SemanticDimension,
    BIPOLAR_PAIRS,
    BipolarPair,
)

@dataclass
class BipolarActivation:
    """Activation of a bipolar pair."""
    pair: BipolarPair
    value: float  # -1.0 to +1.0 (towards negative or positive pole)
    confidence: float # Absolute value


class PrimitiveProbe:
    """Probes vectors for primitive activations."""

    def __init__(self, basis: Optional[HadamardBasis] = None):
        if basis is None:
            basis = HadamardBasis().generate()
        self.basis = basis

    def probe_vector(
        self,
        vector: SparseVector,
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Probe vector against all primitives in the basis.
        
        Args:
            vector: Vector to probe
            threshold: Minimum absolute cosine similarity to return
            
        Returns:
            Dict mapping primitive name to cosine similarity
        """
        activations = {}
        
        # Optimization: Could use matrix multiplication if basis cached as matrix
        # For now, simple iteration is fine for research code
        for name, prim_vec in self.basis.primitives.items():
            sim = vector.cosine(prim_vec)
            if abs(sim) >= threshold:
                activations[name] = sim
                
        return activations

    def probe_bipolar(
        self,
        vector: SparseVector
    ) -> List[BipolarActivation]:
        """
        Probe Bipolar pairs specifically.
        
        Return activation along each bipolar axis.
        """
        results = []
        for pair in BIPOLAR_PAIRS:
            # Check positive pole
            pos_vec = self.basis.get_primitive(pair.positive)
            if pos_vec:
                sim = vector.cosine(pos_vec)
                # If significant
                if abs(sim) > 0.05:
                    results.append(BipolarActivation(
                        pair=pair,
                        value=sim,
                        confidence=abs(sim)
                    ))
        
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def get_dimension_profile(
        self,
        vector: SparseVector,
        threshold: float = 0.05,
    ) -> Dict[str, float]:
        """
        Get aggregate activation mass per semantic dimension.
        """
        profile = {dim.name: 0.0 for dim in SemanticDimension}
        
        # Probe all (filtered by low threshold to avoid noise)
        activations = self.probe_vector(vector, threshold)
        
        for name, value in activations.items():
            info = PRIMITIVE_REGISTRY.get(name)
            if info:
                # Add absolute activation (energy)
                profile[info.dimension.name] += abs(value)
                
        return profile

    def decompose(
        self,
        vector: SparseVector,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Decompose vector into its constituent primitives (sparse coding).
        
        Returns top K primitives by similarity.
        """
        activations = self.probe_vector(vector, threshold=0.01)
        # Sort by absolute value
        sorted_acts = sorted(activations.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_acts[:top_k]
