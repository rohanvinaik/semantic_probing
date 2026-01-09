"""
Probe trained specialists to extract interpretable functional networks.

This experiment identifies what "circuits" each specialist has learned
by comparing primitive activation patterns to a generalist baseline.

Usage:
    python -m semantic_probing.experiments.functional_extraction \
        --specialist_path /path/to/checkpoint \
        --dimension LOGICAL \
        --output reports/functional_networks/logical_specialist.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np

from semantic_probing.encoding.sparse_ternary import HadamardBasis
from semantic_probing.probes.primitives import PrimitiveProbe
from semantic_probing.encoding.text_encoder import TextEncoder
from semantic_probing.analysis.signatures import SignatureAnalyzer


@dataclass
class PrimitiveActivation:
    """Activation pattern for a single primitive."""
    primitive: str
    mean_activation: float
    std_activation: float
    activation_frequency: float  # % of probes where active


@dataclass
class FunctionalNetwork:
    """Extracted functional network from a specialist."""
    dimension: str
    specialist_path: str

    # Activation patterns
    strong_primitives: List[PrimitiveActivation]  # Much stronger than baseline
    weak_primitives: List[PrimitiveActivation]    # Much weaker than baseline

    # Network statistics
    specialization_score: float  # How focused vs generalist
    dimension_alignment: float   # Correlation with expected dimension
    entropy: float               # Average entropy across probes

    # Interpretable description
    functional_description: str


class FunctionalNetworkExtractor:
    """Extract functional networks by comparing specialist to baseline."""

    # Probe sets for each dimension
    PROBE_SETS = {
        "LOGICAL": [
            "If A then B. A is true. What can we conclude?",
            "All dogs are mammals. Fido is a dog. Is Fido a mammal?",
            "Not all birds can fly. Penguins are birds. Can penguins fly?",
            "If it rains, the ground is wet. The ground is wet. Did it rain?",
            "A or B is true. A is false. What must be true?",
        ],
        "QUANTITY": [
            "What is 15 multiplied by 7?",
            "If I have 23 apples and give away 8, how many remain?",
            "What is 144 divided by 12?",
            "I have twice as many books as you. You have 15. How many do I have?",
            "What percentage of 80 is 20?",
        ],
        "TEMPORAL": [
            "Event A happened before B, and B before C. What is the order?",
            "If today is Tuesday, what day was it 3 days ago?",
            "The meeting starts at 2pm and lasts 90 minutes. When does it end?",
            "John was born in 1990. How old was he in 2010?",
            "First X, then Y, finally Z. What happened second?",
        ],
        "SPATIAL": [
            "X is north of Y, Y is north of Z. Where is X relative to Z?",
            "The ball is on the table, the table is in the room. Where is the ball?",
            "A is left of B, C is right of B. What is between A and C?",
            "Go 3 blocks east, then 2 blocks north. Where are you?",
            "The cat is under the bed, the bed is in the bedroom. Where is the cat?",
        ],
        "MENTAL": [
            "John thinks Mary knows the secret. Does Mary actually know it?",
            "She believes he is lying. Is he actually lying?",
            "Tom wants to go but thinks he shouldn't. What will Tom likely do?",
            "Alice knows that Bob doesn't know her name. What does Bob know?",
            "He pretended to be happy. Was he actually happy?",
        ],
    }

    def __init__(self):
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

    def probe_model(self, model, probes: List[str]) -> List[Dict]:
        """Run probes through model and collect activation patterns."""
        patterns = []
        for probe_text in probes:
            # Get model response
            response = model.generate(probe_text, max_tokens=256)

            # Compute semantic signature of response
            combined = f"{probe_text} {response}"
            vector = self.encoder.encode(combined)
            sig = self.analyzer.compute_signature(vector)

            patterns.append({
                "probe": probe_text,
                "response": response,
                "activations": activations,
                "dimension_profile": sig.dimension_profile,
                "entropy": sig.entropy,
                "primary_dimension": sig.primary_dimension,
            })
        return patterns

    def compare_to_baseline(
        self,
        specialist_patterns: List[Dict],
        baseline_patterns: List[Dict],
    ) -> Dict[str, float]:
        """Compute difference in activation patterns."""
        # Aggregate activations
        spec_agg = self._aggregate_activations(specialist_patterns)
        base_agg = self._aggregate_activations(baseline_patterns)

        # Compute differences
        differences = {}
        all_primitives = set(spec_agg.keys()) | set(base_agg.keys())
        for prim in all_primitives:
            spec_val = spec_agg.get(prim, 0)
            base_val = base_agg.get(prim, 0)
            differences[prim] = spec_val - base_val

        return differences

    def _aggregate_activations(self, patterns: List[Dict]) -> Dict[str, float]:
        """Compute mean activation per primitive."""
        all_activations = {}
        for p in patterns:
            for prim, val in p["activations"].items():
                if prim not in all_activations:
                    all_activations[prim] = []
                all_activations[prim].append(abs(val))

        return {prim: np.mean(vals) for prim, vals in all_activations.items()}

    def extract_network(
        self,
        specialist_model,
        baseline_model,
        dimension: str,
        specialist_path: str,
    ) -> FunctionalNetwork:
        """Extract functional network by comparing specialist to baseline."""

        probes = self.PROBE_SETS.get(dimension, self.PROBE_SETS["LOGICAL"])

        # Probe both models
        spec_patterns = self.probe_model(specialist_model, probes)
        base_patterns = self.probe_model(baseline_model, probes)

        # Compare activations
        differences = self.compare_to_baseline(spec_patterns, base_patterns)

        # Identify strong/weak primitives
        sorted_diffs = sorted(differences.items(), key=lambda x: x[1], reverse=True)

        strong = [
            PrimitiveActivation(
                primitive=p,
                mean_activation=differences[p],
                std_activation=0.0,  # Would compute from raw data
                activation_frequency=1.0,
            )
            for p, _ in sorted_diffs[:5] if differences[p] > 0.1
        ]

        weak = [
            PrimitiveActivation(
                primitive=p,
                mean_activation=differences[p],
                std_activation=0.0,
                activation_frequency=1.0,
            )
            for p, _ in sorted_diffs[-5:] if differences[p] < -0.1
        ]

        # Compute statistics
        spec_entropies = [p["entropy"] for p in spec_patterns]
        base_entropies = [p["entropy"] for p in base_patterns]

        specialization = np.std(list(differences.values()))
        alignment = np.mean([
            p["dimension_profile"].get(dimension, 0)
            for p in spec_patterns
        ])

        # Generate description
        strong_names = [s.primitive for s in strong]
        weak_names = [w.primitive for w in weak]
        description = (
            f"The {dimension} specialist shows increased activation in "
            f"{', '.join(strong_names)} and decreased activation in "
            f"{', '.join(weak_names)}. This suggests the model has developed "
            f"specialized circuits for {dimension.lower()} reasoning."
        )

        return FunctionalNetwork(
            dimension=dimension,
            specialist_path=specialist_path,
            strong_primitives=strong,
            weak_primitives=weak,
            specialization_score=specialization,
            dimension_alignment=alignment,
            entropy=np.mean(spec_entropies),
            functional_description=description,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist_path", required=True)
    parser.add_argument("--baseline_path", default=None,
                       help="Path to baseline model. If None, uses base model.")
    parser.add_argument("--dimension", required=True,
                       choices=["LOGICAL", "QUANTITY", "TEMPORAL", "SPATIAL", "MENTAL"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load models (placeholder - actual loading depends on Tinker API)
    # from tinker_cookbook.utils import load_model
    class MockModel:
        def generate(self, text, max_tokens):
            return "Mock response"
    
    specialist = MockModel() 
    baseline = MockModel()

    # Extract network
    extractor = FunctionalNetworkExtractor()
    network = extractor.extract_network(
        specialist, baseline, args.dimension, args.specialist_path
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(asdict(network), f, indent=2, default=str)

    print(f"Extracted functional network saved to {output_path}")
    print(f"\nFunctional Description:\n{network.functional_description}")


if __name__ == "__main__":
    main()
