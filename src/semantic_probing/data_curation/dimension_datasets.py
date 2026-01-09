"""
Filter datasets by dominant semantic dimension using NSM primitives.

Usage:
    python -m semantic_probing.data_curation.dimension_datasets \
        --dimension LOGICAL \
        --source gsm8k \
        --output data/specialists/logical_specialist.jsonl \
        --threshold 0.5
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Iterator
from dataclasses import dataclass

from semantic_probing.encoding.sparse_ternary import HadamardBasis
from semantic_probing.probes.primitives import PrimitiveProbe
from semantic_probing.encoding.text_encoder import TextEncoder


@dataclass
class DimensionFilter:
    """Filter examples by semantic dimension dominance."""

    target_dimension: str
    threshold: float = 0.5

    # Semantic dimensions and their constituent primitives
    DIMENSIONS = {
        "LOGICAL": ["NOT", "IF", "BECAUSE", "MAYBE", "CAN", "TRUE"],
        "QUANTITY": ["ONE", "TWO", "SOME", "ALL", "MANY", "MORE", "MUCH"],
        "TEMPORAL": ["NOW", "BEFORE", "AFTER", "WHEN", "MOMENT", "LONG_TIME"],
        "SPATIAL": ["WHERE", "HERE", "ABOVE", "BELOW", "NEAR", "FAR", "SIDE"],
        "MENTAL": ["THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR"],
        "SUBSTANTIVES": ["SOMETHING", "SOMEONE", "PEOPLE", "BODY", "PART", "KIND"],
        "EVALUATORS": ["GOOD", "BAD", "BIG", "SMALL"],
        "ACTION": ["DO", "HAPPEN", "MOVE", "SAY", "WORD", "LIVE", "DIE"],
    }

    def __post_init__(self):
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()

        if self.target_dimension not in self.DIMENSIONS:
            raise ValueError(f"Unknown dimension: {self.target_dimension}")

    def compute_dimension_score(self, text: str) -> float:
        """Compute activation score for target dimension."""
        # Encode text to sparse ternary vector
        vector = self.encoder.encode(text)

        # Probe for primitive activations
        activations = self.probe.probe_vector(vector)

        # Sum activations for primitives in target dimension
        target_primitives = self.DIMENSIONS[self.target_dimension]
        dim_score = sum(
            abs(activations.get(p, 0))
            for p in target_primitives
        ) / len(target_primitives)

        # Normalize by total activation
        total = sum(abs(v) for v in activations.values()) + 1e-8
        return dim_score / total if total > 0 else 0

    def filter_dataset(self, examples: Iterator[Dict]) -> Iterator[Dict]:
        """Yield examples dominated by target dimension."""
        for item in examples:
            text = f"{item.get('problem', '')} {item.get('solution', '')}"
            score = self.compute_dimension_score(text)

            if score >= self.threshold:
                item['dimension_score'] = score
                item['target_dimension'] = self.target_dimension
                yield item


def load_source_dataset(source: str) -> Iterator[Dict]:
    """Load examples from various source datasets."""
    if source == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
        for item in ds:
            yield {
                "problem": item["question"],
                "solution": item["answer"],
                "source": "gsm8k"
            }
    elif source == "folio":
        from datasets import load_dataset
        ds = load_dataset("yale-nlp/FOLIO", split="train")
        for item in ds:
            yield {
                "problem": item["premises"] + " " + item["conclusion"],
                "solution": item["label"],
                "source": "folio"
            }
    elif source == "math":
        from datasets import load_dataset
        ds = load_dataset("lighteval/MATH", split="train")
        for item in ds:
            yield {
                "problem": item["problem"],
                "solution": item["solution"],
                "source": "math"
            }
    else:
        # Load from local JSONL
        path = Path(source)
        if path.exists():
            with open(path) as f:
                for line in f:
                    yield json.loads(line)
        else:
            raise ValueError(f"Unknown source: {source}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", required=True,
                       choices=list(DimensionFilter.DIMENSIONS.keys()))
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_examples", type=int, default=10000)
    args = parser.parse_args()

    # Create filter
    dim_filter = DimensionFilter(args.dimension, args.threshold)

    # Process dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, 'w') as f:
        for item in dim_filter.filter_dataset(load_source_dataset(args.source)):
            f.write(json.dumps(item) + '\n')
            count += 1
            if count >= args.max_examples:
                break

    print(f"Wrote {count} {args.dimension}-dominant examples to {output_path}")


if __name__ == "__main__":
    main()
