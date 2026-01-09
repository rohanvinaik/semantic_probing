"""Semantic probing logic (similarity, relationships)."""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np

from dataclasses import dataclass
from ..encoding import SparseVector, bundle, BIPOLAR_PAIRS, BipolarPair

@dataclass
class AntonymProbeResult:
    """Result of antonym detection probe."""
    word1: str
    word2: str
    cosine_similarity: float
    is_antonym: bool
    bipolar_pair: Optional[str]
    confidence: float
    explanation: str

class AntonymDetectionProbe:
    """
    Demonstrate antonym detection via bipolar semantic dimensions.

    Key insight: Antonyms encode with opposite signs in the SAME
    dimensional space, yielding cos(antonym1, antonym2) = -1.0
    """

    def __init__(self, basis):
        self.basis = basis

    def detect_antonym(
        self,
        vec1: SparseVector,
        vec2: SparseVector,
        word1: str = "word1",
        word2: str = "word2"
    ) -> AntonymProbeResult:
        """
        Check if two vectors represent antonyms via bipolar opposition.

        Returns result with cosine similarity and identified bipolar pair.
        """
        cosine = vec1.cosine(vec2)

        # Check each bipolar pair for opposition pattern
        for pair in BIPOLAR_PAIRS:
            pos_vec = self.basis.primitives.get(pair.positive)
            neg_vec = self.basis.primitives.get(pair.negative)

            if pos_vec is None or neg_vec is None:
                continue

            v1_pos_sim = vec1.cosine(pos_vec)
            v1_neg_sim = vec1.cosine(neg_vec)
            v2_pos_sim = vec2.cosine(pos_vec)
            v2_neg_sim = vec2.cosine(neg_vec)

            # Pattern: vec1 aligns positive, vec2 aligns negative (or vice versa)
            # Threshold 0.3 for robust detection in bundles
            if (v1_pos_sim > 0.3 and v2_neg_sim > 0.3) or \
               (v1_neg_sim > 0.3 and v2_pos_sim > 0.3):
                return AntonymProbeResult(
                    word1=word1,
                    word2=word2,
                    cosine_similarity=cosine,
                    is_antonym=True,
                    bipolar_pair=pair.name,
                    confidence=abs(cosine),
                    explanation=f"Detected opposition on {pair.name} dimension"
                )

        return AntonymProbeResult(
            word1=word1,
            word2=word2,
            cosine_similarity=cosine,
            is_antonym=cosine < -0.5,
            bipolar_pair=None,
            confidence=abs(cosine) if cosine < -0.5 else 0.0,
            explanation="No specific bipolar dimension identified" if cosine < -0.5 else "Not antonyms"
        )

    def validate_bipolar_pairs(self) -> List[AntonymProbeResult]:
        """
        Validate that all 14 bipolar pairs have cos = -1.0
        """
        results = []
        for pair in BIPOLAR_PAIRS:
            pos_vec = self.basis.primitives.get(pair.positive)
            neg_vec = self.basis.primitives.get(pair.negative)

            if pos_vec is None or neg_vec is None:
                continue

            result = self.detect_antonym(
                pos_vec, neg_vec,
                word1=pair.positive,
                word2=pair.negative
            )
            results.append(result)

        return results

class SemanticProbe:
    """Probes semantic relationships between vectors."""

    def similarity(self, v1: SparseVector, v2: SparseVector) -> float:
        """Compute cosine similarity."""
        return v1.cosine(v2)

    def analogy(
        self,
        a: SparseVector,
        b: SparseVector,
        c: SparseVector,
    ) -> SparseVector:
        """
        Solve analogy: a is to b as c is to ?
        
        Vector arithmetic: d = c + (b - a)
        Implemented as: bundle([c, b, a.negate()])
        """
        # We want vector close to c and b, but far from a
        # c + b - a
        return bundle([c, b, a.negate()])

    def compositional_analogy(
        self,
        a: SparseVector,
        b: SparseVector,
        c: SparseVector,
    ) -> SparseVector:
        """
        Bipolar analogy using multiplicative negation.

        For bipolar pairs: if B = -A, then D = -C (not C + B - A)
        """
        # Detect if A and B are bipolar opposites
        if a.cosine(b) < -0.9:  # They're antonyms
            return c.negate()  # D = -C (multiplicative)
        else:
            return bundle([c, b, a.negate()])  # Fallback to additive


    def find_nearest(
        self,
        query: SparseVector,
        candidates: Dict[str, SparseVector],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find nearest neighbors in a candidate set.
        
        Args:
            query: Query vector
            candidates: Dict mapping word -> vector
            top_k: Number of results
            
        Returns:
            List of (word, score) tuples
        """
        scores = []
        for word, vec in candidates.items():
            sim = self.similarity(query, vec)
            scores.append((word, sim))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def rank_similarity(
        self,
        query: SparseVector,
        targets: List[str],
        candidate_vectors: Dict[str, SparseVector],
    ) -> List[Tuple[str, float]]:
        """Rank specific targets by similarity to query."""
        scores = []
        for target in targets:
            vec = candidate_vectors.get(target)
            if vec:
                sim = self.similarity(query, vec)
                scores.append((target, sim))
            else:
                scores.append((target, -1.0)) # Missing
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
