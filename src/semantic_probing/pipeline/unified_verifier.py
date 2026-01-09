"""
Unified verification pipeline coordinating all five components.

This is the main integration point combining:
1. Semantic probing (semantic_probing)
2. Entity grounding (sparse-wiki-grounding)
3. Negative learning (negative-learning-censor)
4. Orthogonal validation (orthogonal-validators)
5. Proof caching (experience-memory)
"""

import sys
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Core semantic probing
from semantic_probing.encoding.sparse_ternary import HadamardBasis
from semantic_probing.probes.primitives import PrimitiveProbe
from semantic_probing.encoding.text_encoder import TextEncoder
from semantic_probing.analysis.signatures import SignatureAnalyzer

# Integrations
sys.path.insert(0, "/Users/rohanvinaik/sparse-wiki-grounding/src")
sys.path.insert(0, "/Users/rohanvinaik/negative-learning-censor/src")
sys.path.insert(0, "/Users/rohanvinaik/orthogonal-validators")
sys.path.insert(0, "/Users/rohanvinaik/experience-memory")

from wiki_grounding.store import EntityStore
from wiki_grounding.verifier import ClaimVerifier
from negative_learning import CensorRegistry, CensorContext
from orthogonal_validators.core.committee import ValidatorCommittee
from orthogonal_validators.validators.semantic import SemanticValidator
from orthogonal_validators.validators.entity import EntityValidator
from experience_memory import FixRegistry
from experience_memory import ErrorSignature, Fix, ErrorType, FixType, ErrorSeverity


class VerificationStatus(Enum):
    """Verification outcome status."""
    VERIFIED = "verified"
    REJECTED = "rejected"
    SUPPRESSED = "suppressed"  # Known bad pattern
    REVIEW = "review"  # Needs human review
    CACHED = "cached"  # Retrieved from cache


@dataclass
class VerificationResult:
    """Result of unified verification."""
    status: VerificationStatus
    confidence: float
    details: Dict
    latency_ms: float


class UnifiedVerifier:
    """
    Main verification pipeline coordinating all components.

    Flow:
    1. Check proof cache (O(1))
    2. Check censors (should we even try?)
    3. Run validator committee
    4. Handle outcome (cache success, learn from failure)
    """

    def __init__(
        self,
        wiki_db_path: str = None,
        cache_path: str = "/tmp/unified_verifier_cache.db",
    ):
        # Core semantic probing
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

        # Wiki grounding
        wiki_path = wiki_db_path or "/Users/rohanvinaik/sparse-wiki-grounding/data/wiki_grounding.db"
        self.wiki_store = EntityStore(wiki_path)
        self.claim_verifier = ClaimVerifier(self.wiki_store)

        # Censors
        self.censor_registry = CensorRegistry()

        # Proof cache
        self.proof_cache = FixRegistry(cache_path)

        # Validator committee
        self.committee = ValidatorCommittee([
            SemanticValidator(),
            EntityValidator(),
            # SemanticProbingValidator added separately
        ])

        # Statistics
        self._stats = {
            "total": 0,
            "cached": 0,
            "suppressed": 0,
            "verified": 0,
            "rejected": 0,
            "review": 0,
        }

    def _claim_to_cache_sig(self, claim: str) -> ErrorSignature:
        """Convert claim to cache signature."""
        import hashlib
        claim_hash = hashlib.sha256(claim.encode()).hexdigest()[:16]
        return ErrorSignature(
            severity=ErrorSeverity.MINOR,
            error_type=ErrorType.PARTIAL_DEFINITION,
            context=f"verified:{claim_hash}",
            affected_categories=["verification_cache"],
            delta=0.0,
        )

    def _claim_to_censor_context(self, claim: str) -> CensorContext:
        """Convert claim to censor context."""
        # Compute semantic signature
        vector = self.encoder.encode(claim)
        sig = self.analyzer.compute_signature(vector)

        return CensorContext(
            perceptual={
                "primary_dimension": sig.primary_dimension,
                "entropy": sig.entropy,
            },
            sequential={
                "claim_length": len(claim),
            },
        )

    def verify(self, claim: str) -> VerificationResult:
        """Run full verification pipeline."""
        start = time.perf_counter()
        self._stats["total"] += 1

        # 1. Check proof cache (O(1))
        cache_sig = self._claim_to_cache_sig(claim)
        cached = self.proof_cache.lookup(cache_sig)

        if cached:
            self._stats["cached"] += 1
            latency = (time.perf_counter() - start) * 1000
            cached_data = json.loads(cached.definition_supplement)
            return VerificationResult(
                status=VerificationStatus.CACHED,
                confidence=cached_data.get("confidence", 1.0),
                details={"source": "cache", "original_status": cached_data.get("status")},
                latency_ms=latency,
            )

        # 2. Check censors
        context = self._claim_to_censor_context(claim)
        suppression = self.censor_registry.query(context, "verify")

        if suppression > 0.9:
            self._stats["suppressed"] += 1
            latency = (time.perf_counter() - start) * 1000
            return VerificationResult(
                status=VerificationStatus.SUPPRESSED,
                confidence=suppression,
                details={"reason": "Known failure pattern"},
                latency_ms=latency,
            )

        # 3. Run validator committee
        committee_result = self.committee.validate(claim)

        # 4. Handle outcome
        if committee_result.auto_accept:
            # Cache successful verification
            self._stats["verified"] += 1
            confidence_score = committee_result.weighted_agreement
            fix = Fix(
                fix_type=FixType.COMPLETE_DEFINITION,
                definition_supplement=json.dumps({
                    "status": "verified",
                    "confidence": confidence_score,
                }),
                elements_to_add={},
                elements_to_remove=[],
                value_adjustments={},
                category_moves={},
                decomposition=None,
            )
            self.proof_cache.register(cache_sig, fix)

            latency = (time.perf_counter() - start) * 1000
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=confidence_score,
                details={"validators": committee_result.votes},
                latency_ms=latency,
            )

        elif committee_result.all_zero_margin:
            # Validators disagree - needs human review
            self._stats["review"] += 1
            latency = (time.perf_counter() - start) * 1000
            return VerificationResult(
                status=VerificationStatus.REVIEW,
                confidence=0.5,
                details={
                    "reason": "Validators disagree",
                    "validators": committee_result.votes,
                },
                latency_ms=latency,
            )

        else:
            # Rejection - learn from failure
            self._stats["rejected"] += 1
            self.censor_registry.learn(context, "verify", success=False)
            confidence_score = committee_result.weighted_agreement

            latency = (time.perf_counter() - start) * 1000
            return VerificationResult(
                status=VerificationStatus.REJECTED,
                confidence=confidence_score,  # Used to be 1 - confidence, but weighted agreement reflects consensus strength
                details={"validators": committee_result.votes},
                latency_ms=latency,
            )

    def batch_verify(self, claims: List[str]) -> List[VerificationResult]:
        """Verify multiple claims."""
        return [self.verify(claim) for claim in claims]

    def stats(self) -> Dict:
        """Get verification statistics."""
        total = self._stats["total"]
        if total == 0:
            return self._stats

        return {
            **self._stats,
            "cache_rate": self._stats["cached"] / total,
            "suppression_rate": self._stats["suppressed"] / total,
            "auto_accept_rate": self._stats["verified"] / total,
            "review_rate": self._stats["review"] / total,
            "rejection_rate": self._stats["rejected"] / total,
        }
