"""Semantic signature analysis."""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from ..encoding import SparseVector
from ..probes.primitives import PrimitiveProbe

@dataclass
class SemanticSignature:
    """Statistical signature of a semantic vector."""
    dimension_profile: Dict[str, float]
    active_primitives: int
    sparsity: float
    entropy: float
    magnitude: float
    primary_dimension: str
    
    def __str__(self):
        return (
            f"SemanticSignature(prim={self.primary_dimension}, "
            f"active={self.active_primitives}, "
            f"entropy={self.entropy:.2f}, "
            f"mag={self.magnitude:.2f})"
        )


class SignatureAnalyzer:
    """Analyzes semantic signatures."""
    
    def __init__(self, probe: Optional[PrimitiveProbe] = None):
        if probe is None:
            from ..probes.primitives import PrimitiveProbe
            probe = PrimitiveProbe()
        self.probe = probe
        
    def compute_signature(self, vector: SparseVector) -> SemanticSignature:
        """Compute semantic signature from vector."""
        # Get primitive activations
        activations = self.probe.probe_vector(vector, threshold=0.05)
        
        # Get dimension profile
        dim_profile = self.probe.get_dimension_profile(vector, threshold=0.05)
        
        # Calculate stats
        values = np.array(list(activations.values()))
        active_count = len(values)
        
        # Entropy of activation distribution (energy distribution)
        if active_count > 0:
            energy = np.abs(values)
            # Normalize to prob distribution
            total_energy = np.sum(energy)
            if total_energy > 0:
                probs = energy / total_energy
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
            else:
                entropy = 0.0
        else:
            entropy = 0.0
            
        # Sparsity (fraction of ZERO primitives relative to ACTIVE ones? No, relative to Basis size)
        total_primitives_in_basis = len(self.probe.basis.primitives)
        sparsity = 1.0 - (active_count / total_primitives_in_basis)
        
        # Primary dimension
        if dim_profile:
            primary_dim = max(dim_profile.items(), key=lambda x: x[1])[0]
        else:
            primary_dim = "NONE"
            
        return SemanticSignature(
            dimension_profile=dim_profile,
            active_primitives=active_count,
            sparsity=sparsity,
            entropy=entropy,
            magnitude=vector.magnitude(),
            primary_dimension=primary_dim,
        )
        
    def compare_signatures(
        self,
        s1: SemanticSignature,
        s2: SemanticSignature
    ) -> Dict[str, float]:
        """Compare two signatures."""
        # Overlap of primary dimensions?
        # Absolute difference in entropy?
        
        dim_diff = 0.0
        all_dims = set(s1.dimension_profile.keys()) | set(s2.dimension_profile.keys())
        for dim in all_dims:
            d1 = s1.dimension_profile.get(dim, 0.0)
            d2 = s2.dimension_profile.get(dim, 0.0)
            dim_diff += abs(d1 - d2)
            
        return {
            "entropy_diff": abs(s1.entropy - s2.entropy),
            "magnitude_diff": abs(s1.magnitude - s2.magnitude),
            "dimension_vector_distance": dim_diff,
            "complexity_ratio": s1.active_primitives / s2.active_primitives if s2.active_primitives > 0 else 0.0
        }
