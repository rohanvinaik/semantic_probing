"""
Establish that semantic coherence metrics predict reasoning quality.

Benchmark Suite:
- 200 correct reasoning traces (from GSM8K solutions)
- 200 incorrect reasoning traces (model failures)
- 200 hallucinated traces (fabricated facts)

Usage:
    python -m semantic_probing.experiments.coherence_benchmark \
        --output reports/coherence_baseline.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from semantic_probing.encoding.sparse_ternary import HadamardBasis
from semantic_probing.probes.primitives import PrimitiveProbe
from semantic_probing.encoding.text_encoder import TextEncoder
from semantic_probing.analysis.signatures import SignatureAnalyzer


@dataclass
class CoherenceMetrics:
    """Coherence metrics for a reasoning trace."""
    primitive_stability: float  # Consistency across steps
    dimension_drift: float      # Change from start to end
    entropy_variance: float     # Stability of focus
    mean_entropy: float
    primary_dimension: str


@dataclass
class BenchmarkResult:
    """Results from coherence benchmark."""
    # Per-category metrics
    correct_metrics: List[CoherenceMetrics]
    incorrect_metrics: List[CoherenceMetrics]
    hallucinated_metrics: List[CoherenceMetrics]

    # Aggregate statistics
    correct_mean_stability: float
    incorrect_mean_stability: float
    hallucinated_mean_stability: float

    # Separability
    stability_auroc: float  # AUROC for correct vs incorrect using stability
    drift_auroc: float      # AUROC using dimension drift

    # Recommendations
    optimal_stability_threshold: float
    optimal_drift_threshold: float


class CoherenceBenchmark:
    """Benchmark semantic coherence as predictor of reasoning quality."""

    def __init__(self):
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

    def compute_trace_coherence(self, reasoning_steps: List[str]) -> CoherenceMetrics:
        """Compute coherence metrics for a reasoning trace."""
        if not reasoning_steps:
            return CoherenceMetrics(0, 0, 0, 0, "UNKNOWN")

        # Compute signature for each step
        signatures = []
        for step in reasoning_steps:
            vector = self.encoder.encode(step)
            sig = self.analyzer.compute_signature(vector)
            signatures.append(sig)

        # Primitive stability: correlation of activations across steps
        if len(signatures) > 1:
            profiles = [sig.dimension_profile for sig in signatures]
            stability = self._compute_profile_stability(profiles)
        else:
            stability = 1.0

        # Dimension drift: change from first to last step
        if len(signatures) > 1:
            first = signatures[0].dimension_profile
            last = signatures[-1].dimension_profile
            drift = self._compute_profile_distance(first, last)
        else:
            drift = 0.0

        # Entropy variance
        entropies = [sig.entropy for sig in signatures]
        entropy_var = np.var(entropies) if len(entropies) > 1 else 0.0

        # Primary dimension (most common)
        dims = [sig.primary_dimension for sig in signatures]
        primary = max(set(dims), key=dims.count) if dims else "UNKNOWN"

        return CoherenceMetrics(
            primitive_stability=stability,
            dimension_drift=drift,
            entropy_variance=entropy_var,
            mean_entropy=np.mean(entropies),
            primary_dimension=primary,
        )

    def _compute_profile_stability(self, profiles: List[Dict[str, float]]) -> float:
        """Compute stability of dimension profiles across steps."""
        if len(profiles) < 2:
            return 1.0

        # Get all dimensions
        all_dims = set()
        for p in profiles:
            all_dims.update(p.keys())

        # Compute variance per dimension
        variances = []
        for dim in all_dims:
            values = [p.get(dim, 0) for p in profiles]
            variances.append(np.var(values))

        # Stability = 1 - mean variance (higher = more stable)
        return 1.0 - min(1.0, np.mean(variances))

    def _compute_profile_distance(self, p1: Dict, p2: Dict) -> float:
        """Euclidean distance between dimension profiles."""
        all_dims = set(p1.keys()) | set(p2.keys())
        dist_sq = sum((p1.get(d, 0) - p2.get(d, 0))**2 for d in all_dims)
        return np.sqrt(dist_sq)

    def parse_reasoning_steps(self, trace: str) -> List[str]:
        """Split reasoning trace into steps."""
        # Split on common step markers
        import re
        steps = re.split(r'\n(?=Step|\d+\.|•|-)', trace)
        return [s.strip() for s in steps if s.strip()]

    def run_benchmark(
        self,
        correct_traces: List[str],
        incorrect_traces: List[str],
        hallucinated_traces: List[str],
    ) -> BenchmarkResult:
        """Run full benchmark."""

        # Compute metrics for each category
        correct_metrics = [
            self.compute_trace_coherence(self.parse_reasoning_steps(t))
            for t in correct_traces
        ]
        incorrect_metrics = [
            self.compute_trace_coherence(self.parse_reasoning_steps(t))
            for t in incorrect_traces
        ]
        hallucinated_metrics = [
            self.compute_trace_coherence(self.parse_reasoning_steps(t))
            for t in hallucinated_traces
        ]

        # Aggregate statistics
        correct_stab = np.mean([m.primitive_stability for m in correct_metrics])
        incorrect_stab = np.mean([m.primitive_stability for m in incorrect_metrics])
        halluc_stab = np.mean([m.primitive_stability for m in hallucinated_metrics])

        # Compute AUROC for separability
        stability_auroc = self._compute_auroc(
            [m.primitive_stability for m in correct_metrics],
            [m.primitive_stability for m in incorrect_metrics],
        )
        drift_auroc = self._compute_auroc(
            [-m.dimension_drift for m in correct_metrics],  # Negate: lower drift = better
            [-m.dimension_drift for m in incorrect_metrics],
        )

        # Find optimal thresholds
        opt_stability = self._find_optimal_threshold(
            [m.primitive_stability for m in correct_metrics],
            [m.primitive_stability for m in incorrect_metrics],
        )
        opt_drift = self._find_optimal_threshold(
            [-m.dimension_drift for m in correct_metrics],
            [-m.dimension_drift for m in incorrect_metrics],
        )

        return BenchmarkResult(
            correct_metrics=correct_metrics,
            incorrect_metrics=incorrect_metrics,
            hallucinated_metrics=hallucinated_metrics,
            correct_mean_stability=correct_stab,
            incorrect_mean_stability=incorrect_stab,
            hallucinated_mean_stability=halluc_stab,
            stability_auroc=stability_auroc,
            drift_auroc=drift_auroc,
            optimal_stability_threshold=opt_stability,
            optimal_drift_threshold=-opt_drift,
        )

    def _compute_auroc(self, positive: List[float], negative: List[float]) -> float:
        """Compute AUROC for binary classification."""
        from sklearn.metrics import roc_auc_score
        y_true = [1] * len(positive) + [0] * len(negative)
        y_score = positive + negative
        try:
            return roc_auc_score(y_true, y_score)
        except:
            return 0.5

    def _find_optimal_threshold(self, pos: List[float], neg: List[float]) -> float:
        """Find threshold that maximizes F1."""
        all_vals = sorted(set(pos + neg))
        best_f1 = 0
        best_thresh = 0
        for thresh in all_vals:
            tp = sum(1 for v in pos if v >= thresh)
            fp = sum(1 for v in neg if v >= thresh)
            fn = sum(1 for v in pos if v < thresh)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh


def load_benchmark_data() -> Tuple[List[str], List[str], List[str]]:
    """Load or generate benchmark data."""
    data_dir = Path("/Users/rohanvinaik/semantic_probing/data/benchmarks")

    # Try to load existing benchmark
    bench_file = data_dir / "coherence_traces.jsonl"
    if bench_file.exists():
        correct, incorrect, hallucinated = [], [], []
        with open(bench_file) as f:
            for line in f:
                item = json.loads(line)
                if item["category"] == "correct":
                    correct.append(item["trace"])
                elif item["category"] == "incorrect":
                    incorrect.append(item["trace"])
                else:
                    hallucinated.append(item["trace"])
        return correct, incorrect, hallucinated

    # Generate from GSM8K
    print("Generating benchmark data from GSM8K...")
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")

    correct = [item["answer"] for item in ds.select(range(200))]

    # Generate incorrect by truncating or corrupting
    incorrect = []
    for item in ds.select(range(200, 400)):
        trace = item["answer"]
        # Truncate reasoning
        lines = trace.split('\n')
        if len(lines) > 2:
            incorrect.append('\n'.join(lines[:len(lines)//2]) + "\nTherefore the answer is 42.")
        else:
            incorrect.append("The answer is obviously 42.")

    # Generate hallucinated with fabricated facts
    hallucinated = []
    for item in ds.select(range(400, 600)):
        hallucinated.append(
            f"Based on the theorem of quantum arithmetic, {item['question'].split()[0]} "
            f"equals approximately 3.14159. By applying the Henderson-Smith conjecture, "
            f"we get the final answer: 999."
        )

    # Save for future use
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(bench_file, 'w') as f:
        for trace in correct:
            f.write(json.dumps({"category": "correct", "trace": trace}) + '\n')
        for trace in incorrect:
            f.write(json.dumps({"category": "incorrect", "trace": trace}) + '\n')
        for trace in hallucinated:
            f.write(json.dumps({"category": "hallucinated", "trace": trace}) + '\n')

    return correct, incorrect, hallucinated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/coherence_baseline.json")
    args = parser.parse_args()

    # Load data
    correct, incorrect, hallucinated = load_benchmark_data()
    print(f"Loaded {len(correct)} correct, {len(incorrect)} incorrect, {len(hallucinated)} hallucinated traces")

    # Run benchmark
    benchmark = CoherenceBenchmark()
    result = benchmark.run_benchmark(correct, incorrect, hallucinated)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    result_dict = {
        "correct_mean_stability": result.correct_mean_stability,
        "incorrect_mean_stability": result.incorrect_mean_stability,
        "hallucinated_mean_stability": result.hallucinated_mean_stability,
        "stability_auroc": result.stability_auroc,
        "drift_auroc": result.drift_auroc,
        "optimal_stability_threshold": result.optimal_stability_threshold,
        "optimal_drift_threshold": result.optimal_drift_threshold,
        "n_correct": len(result.correct_metrics),
        "n_incorrect": len(result.incorrect_metrics),
        "n_hallucinated": len(result.hallucinated_metrics),
    }

    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n=== Coherence Benchmark Results ===")
    print(f"Correct mean stability:      {result.correct_mean_stability:.3f}")
    print(f"Incorrect mean stability:    {result.incorrect_mean_stability:.3f}")
    print(f"Hallucinated mean stability: {result.hallucinated_mean_stability:.3f}")
    print(f"\nStability AUROC: {result.stability_auroc:.3f}")
    print(f"Drift AUROC:     {result.drift_auroc:.3f}")
    print(f"\nOptimal thresholds:")
    print(f"  Stability >= {result.optimal_stability_threshold:.3f}")
    print(f"  Drift <= {result.optimal_drift_threshold:.3f}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
