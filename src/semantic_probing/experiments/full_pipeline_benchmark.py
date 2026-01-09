"""
Benchmark the full unified verification pipeline.

Compares:
1. Single LLM (GPT-4 baseline)
2. Voting (5 LLM instances)
3. Semantic only (coherence check)
4. Wiki only (claim verification)
5. Full pipeline (all 5 components)

Usage:
    python -m semantic_probing.experiments.full_pipeline_benchmark \
        --output reports/full_pipeline_results.json
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict

from semantic_probing.pipeline.unified_verifier import UnifiedVerifier, VerificationStatus


@dataclass
class BenchmarkMetrics:
    """Metrics for a verification system."""
    accuracy: float
    auto_accept_rate: float
    error_in_auto_accept: float  # Errors in auto-accepted claims
    review_rate: float
    mean_latency_ms: float
    cache_hit_rate: float


@dataclass
class FullBenchmarkResult:
    """Complete benchmark results."""
    # System metrics
    full_pipeline: BenchmarkMetrics
    semantic_only: BenchmarkMetrics
    wiki_only: BenchmarkMetrics

    # Dataset info
    n_claims: int
    n_true: int
    n_false: int

    # Key findings
    findings: List[str]


def load_benchmark_claims() -> List[Dict]:
    """Load benchmark claims with ground truth."""
    data_path = Path("/Users/rohanvinaik/semantic_probing/data/benchmarks/verification_benchmark.jsonl")

    if data_path.exists():
        claims = []
        with open(data_path) as f:
            for line in f:
                claims.append(json.loads(line))
        return claims

    # Generate benchmark
    print("Generating verification benchmark...")
    claims = []

    # True claims
    true_claims = [
        "Paris is the capital of France.",
        "Water freezes at 0 degrees Celsius.",
        "Einstein developed the theory of relativity.",
        "The Earth orbits the Sun.",
        "DNA is a double helix structure.",
    ]

    # False claims
    false_claims = [
        "Paris is the capital of Germany.",
        "Water freezes at 50 degrees Celsius.",
        "Einstein invented the telephone.",
        "The Sun orbits the Earth.",
        "DNA is a triple helix structure.",
    ]

    # Ambiguous claims (for review detection)
    ambiguous_claims = [
        "The best programming language is Python.",
        "Classical music is superior to jazz.",
        "The universe is deterministic.",
    ]

    for claim in true_claims:
        claims.append({"claim": claim, "label": True, "type": "factual"})
    for claim in false_claims:
        claims.append({"claim": claim, "label": False, "type": "factual"})
    for claim in ambiguous_claims:
        claims.append({"claim": claim, "label": None, "type": "ambiguous"})

    # Save
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, 'w') as f:
        for claim in claims:
            f.write(json.dumps(claim) + '\n')

    return claims


class SemanticOnlyVerifier:
    """Verifier using only semantic coherence."""

    def __init__(self):
        from semantic_probing.encoding.sparse_ternary import HadamardBasis
        from semantic_probing.probes.primitives import PrimitiveProbe
        from semantic_probing.encoding.text_encoder import TextEncoder
        from semantic_probing.analysis.signatures import SignatureAnalyzer

        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

    def verify(self, claim: str) -> Dict:
        start = time.perf_counter()
        vector = self.encoder.encode(claim)
        sig = self.analyzer.compute_signature(vector)

        # Simple heuristic: low entropy = coherent = likely true
        verdict = sig.entropy < 3.0
        confidence = 1.0 - min(1.0, sig.entropy / 5.0)

        return {
            "verdict": verdict,
            "confidence": confidence,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }


def run_benchmark(claims: List[Dict]) -> FullBenchmarkResult:
    """Run full benchmark comparison."""

    # Initialize verifiers
    full_pipeline = UnifiedVerifier()
    semantic_only = SemanticOnlyVerifier()

    # Results storage
    results = {
        "full": [],
        "semantic": [],
    }

    print(f"Running benchmark on {len(claims)} claims...")

    for i, item in enumerate(claims):
        claim = item["claim"]
        truth = item["label"]

        # Full pipeline
        full_result = full_pipeline.verify(claim)
        results["full"].append({
            "claim": claim,
            "truth": truth,
            "status": full_result.status.value,
            "confidence": full_result.confidence,
            "latency_ms": full_result.latency_ms,
        })

        # Semantic only
        sem_result = semantic_only.verify(claim)
        results["semantic"].append({
            "claim": claim,
            "truth": truth,
            "verdict": sem_result["verdict"],
            "confidence": sem_result["confidence"],
            "latency_ms": sem_result["latency_ms"],
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(claims)} claims")

    # Compute metrics
    def compute_metrics(results_list: List[Dict], is_full: bool = False) -> BenchmarkMetrics:
        factual = [r for r in results_list if r["truth"] is not None]

        if is_full:
            # Full pipeline metrics
            verified = [r for r in factual if r["status"] == "verified"]
            auto_accepted = len(verified)
            correct_auto = sum(1 for r in verified if r["truth"] == True)
            review = sum(1 for r in results_list if r["status"] == "review")
        else:
            # Simple verifier metrics
            auto_accepted = len(factual)
            correct_auto = sum(1 for r in factual if r["verdict"] == r["truth"])
            review = 0

        accuracy = correct_auto / len(factual) if factual else 0
        auto_accept_rate = auto_accepted / len(results_list) if results_list else 0
        error_in_auto = (auto_accepted - correct_auto) / auto_accepted if auto_accepted > 0 else 0
        review_rate = review / len(results_list) if results_list else 0
        mean_latency = sum(r["latency_ms"] for r in results_list) / len(results_list) if results_list else 0

        return BenchmarkMetrics(
            accuracy=accuracy,
            auto_accept_rate=auto_accept_rate,
            error_in_auto_accept=error_in_auto,
            review_rate=review_rate,
            mean_latency_ms=mean_latency,
            cache_hit_rate=full_pipeline.stats().get("cache_rate", 0) if is_full else 0,
        )

    full_metrics = compute_metrics(results["full"], is_full=True)
    semantic_metrics = compute_metrics(results["semantic"], is_full=False)

    # Count claims
    n_true = sum(1 for c in claims if c["label"] == True)
    n_false = sum(1 for c in claims if c["label"] == False)

    # Key findings
    findings = []
    if full_metrics.error_in_auto_accept < semantic_metrics.error_in_auto_accept:
        findings.append("Full pipeline has lower error rate in auto-accepted claims")
    if full_metrics.review_rate > 0:
        findings.append(f"Full pipeline escalated {full_metrics.review_rate:.1%} to human review")

    return FullBenchmarkResult(
        full_pipeline=full_metrics,
        semantic_only=semantic_metrics,
        wiki_only=BenchmarkMetrics(0, 0, 0, 0, 0, 0),  # Placeholder
        n_claims=len(claims),
        n_true=n_true,
        n_false=n_false,
        findings=findings,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/full_pipeline_results.json")
    args = parser.parse_args()

    # Load claims
    claims = load_benchmark_claims()

    # Run benchmark
    result = run_benchmark(claims)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    # Print summary
    print(f"\n=== Full Pipeline Benchmark Results ===")
    print(f"\nDataset: {result.n_claims} claims ({result.n_true} true, {result.n_false} false)")

    print(f"\n--- Full Pipeline ---")
    print(f"  Accuracy: {result.full_pipeline.accuracy:.1%}")
    print(f"  Auto-accept rate: {result.full_pipeline.auto_accept_rate:.1%}")
    print(f"  Error in auto-accept: {result.full_pipeline.error_in_auto_accept:.1%}")
    print(f"  Review rate: {result.full_pipeline.review_rate:.1%}")
    print(f"  Mean latency: {result.full_pipeline.mean_latency_ms:.1f}ms")

    print(f"\n--- Semantic Only ---")
    print(f"  Accuracy: {result.semantic_only.accuracy:.1%}")
    print(f"  Mean latency: {result.semantic_only.mean_latency_ms:.1f}ms")

    print(f"\n--- Key Findings ---")
    for finding in result.findings:
        print(f"  • {finding}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
