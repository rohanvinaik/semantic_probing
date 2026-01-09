"""Correlation analysis."""

from typing import List, Dict, Optional
import numpy as np
import random

from ..encoding import SparseVector
from ..probes.primitives import PrimitiveProbe

class CorrelationAnalyzer:
    """Analyzes correlations between semantic features."""
    
    def __init__(self, probe: PrimitiveProbe):
        self.probe = probe
        
    def correlate_features(
        self, 
        vectors: List[SparseVector], 
        features: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise correlations between features across a set of vectors.
        
        Args:
            vectors: List of vectors to analyze
            features: List of primitive names to correlate
            
        Returns:
            Correlation matrix (as dict of dicts)
        """
        if not vectors or len(vectors) < 2:
            return {}
            
        # Collect activations matrix: (n_vectors, n_features)
        activations = []
        for vec in vectors:
            acts = self.probe.probe_vector(vec, threshold=0.0)
            row = [acts.get(f, 0.0) for f in features]
            activations.append(row)
            
        data = np.array(activations) # shape (N, F)
        
        # Standardize features (zero mean, unit variance) to handle different scales safely
        # But np.corrcoef handles this. 
        # Check for constant features (zero covariance)
        std = np.std(data, axis=0)
        valid_indices = np.where(std > 1e-9)[0]
        
        if len(valid_indices) == 0:
            return {}
            
        # Calculate correlation matrix
        # Transpose to get (F, N) so corrcoef computes variable correlations
        try:
            # Only use valid features for calculation to avoid warnings
            feature_data = data[:, valid_indices].T
            corr_matrix = np.corrcoef(feature_data)
        except Exception:
            return {}
            
        # Map back to original features
        result = {}
        # Pre-fill with 0
        for f in features:
            result[f] = {f2: 0.0 for f2 in features}
            
        # Fill computed values
        for idx1, real_idx1 in enumerate(valid_indices):
            f1 = features[real_idx1]
            for idx2, real_idx2 in enumerate(valid_indices):
                f2 = features[real_idx2]
                
                if np.ndim(corr_matrix) == 0:
                    val = float(corr_matrix)
                else:
                    val = float(corr_matrix[idx1, idx2])
                
                if np.isnan(val):
                    val = 0.0
                result[f1][f2] = val
                
        return result

    def analyze_structure(
        self,
        vectors: List[SparseVector],
        sample_size: int = 100
    ) -> Dict[str, float]:
        """
        Analyze structural properties of the vector space.
        
        Returns metrics like average cosine distance.
        """
        if len(vectors) < 2:
            return {"average_cosine_distance": 0.0}
            
        dists = []
        
        # Sample pairs randomly if too many
        to_sample = vectors
        if len(vectors) > sample_size:
            to_sample = random.sample(vectors, sample_size)
            
        n = len(to_sample)
        # Compute pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                v1 = to_sample[i]
                v2 = to_sample[j]
                # Cosine distance = 1 - sim
                dists.append(1.0 - v1.cosine(v2))
        
        avg_dist = np.mean(dists) if dists else 0.0
        
        return {
            "average_cosine_distance": float(avg_dist),
            "max_distance": float(np.max(dists)) if dists else 0.0,
            "min_distance": float(np.min(dists)) if dists else 0.0,
            "sample_size": n
        }
