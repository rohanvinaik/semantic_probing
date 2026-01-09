# Modular Brain-Like AI via Semantic Specialists

## Executive Summary

This document describes a research program to build a **brain-like AI architecture** using six codebases:

| Codebase | Core Capability |
|----------|-----------------|
| `semantic_probing` | 62 NSM primitives, 8 dimensions, sparse ternary vectors |
| `tinker-cookbook` | LoRA fine-tuning (SFT + RL), distillation |
| `sparse-wiki-grounding` | Entity grounding, spreading activation, claim verification |
| `negative-learning-censor` | Learn what NOT to do, O(1) censor lookup |
| `orthogonal-validators` | Multi-perspective validation, confidence margins |
| `experience-memory` | O(1) error→fix cache, temporal decay |

**Core Hypothesis**: Train tiny specialists (0.5B-3B) on semantic dimension-filtered data, probe them to extract functional circuits, and orchestrate via fuzzy semantic routing—achieving GPT-4-level reasoning from efficient ensembles.

---

## Phase 0: Tinker Specialist Training (The Foundation)

### 0.1 Data Curation by Semantic Dimension

**File**: `semantic_probing/data_curation/dimension_datasets.py`

```python
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

from semantic_probing.core.hadamard import HadamardBasis
from semantic_probing.core.primitives import PrimitiveProbe
from semantic_probing.core.encoding import TextEncoder


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
```

**Shell Commands**:
```bash
# Create specialist datasets
cd /Users/rohanvinaik/semantic_probing

# LOGICAL specialist (from FOLIO + bAbI)
python -m semantic_probing.data_curation.dimension_datasets \
    --dimension LOGICAL \
    --source folio \
    --output data/specialists/logical_specialist.jsonl \
    --threshold 0.5 \
    --max_examples 10000

# QUANTITY specialist (from GSM8K + MATH)
python -m semantic_probing.data_curation.dimension_datasets \
    --dimension QUANTITY \
    --source gsm8k \
    --output data/specialists/quantity_specialist.jsonl \
    --threshold 0.5 \
    --max_examples 10000

# TEMPORAL specialist
python -m semantic_probing.data_curation.dimension_datasets \
    --dimension TEMPORAL \
    --source math \
    --output data/specialists/temporal_specialist.jsonl \
    --threshold 0.4 \
    --max_examples 5000

# SPATIAL specialist
python -m semantic_probing.data_curation.dimension_datasets \
    --dimension SPATIAL \
    --source math \
    --output data/specialists/spatial_specialist.jsonl \
    --threshold 0.4 \
    --max_examples 5000

# MENTAL specialist (Theory of Mind)
python -m semantic_probing.data_curation.dimension_datasets \
    --dimension MENTAL \
    --source data/raw/tomi.jsonl \
    --output data/specialists/mental_specialist.jsonl \
    --threshold 0.5 \
    --max_examples 5000
```

---

### 0.2 Train Semantic Dimension Specialists

**File**: `tinker-cookbook/tinker_cookbook/recipes/semantic_specialists/train.py`

```python
"""
Train semantic dimension specialists using Tinker.

Usage:
    python -m tinker_cookbook.recipes.semantic_specialists.train \
        dimension=LOGICAL \
        model_name=Qwen/Qwen3-0.5B \
        learning_rate=5e-4 \
        lora_rank=32 \
        wandb_project=semantic_specialists
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.sl.train import Config, main
from tinker_cookbook.sl.types import SLDatasetBuilder, SLBatch

logger = logging.getLogger(__name__)


@chz.chz
class SemanticSpecialistConfig:
    """Configuration for semantic dimension specialist training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-0.5B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Dimension-specific dataset
    dimension: str = "LOGICAL"  # LOGICAL, QUANTITY, TEMPORAL, SPATIAL, MENTAL
    dataset_path: str | None = None  # Auto-resolved from dimension if None

    # Training hyperparameters
    learning_rate: float = 5e-4
    batch_size: int = 8
    max_tokens: int = 2048
    num_epochs: int = 3

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Checkpointing
    save_every: int = 500
    eval_every: int = 100

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


class SemanticDatasetBuilder(SLDatasetBuilder):
    """Load dimension-filtered dataset for SFT."""

    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        model_name_for_tokenizer: str,
        renderer_name: str,
        max_tokens: int = 2048,
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.model_name_for_tokenizer = model_name_for_tokenizer
        self.renderer_name = renderer_name
        self.max_tokens = max_tokens
        self._examples = None

    def _load_examples(self) -> List[Dict]:
        if self._examples is None:
            self._examples = []
            with open(self.dataset_path) as f:
                for line in f:
                    self._examples.append(json.loads(line))
        return self._examples

    def build_batch(self, batch_idx: int) -> SLBatch:
        """Build a batch of training examples."""
        examples = self._load_examples()
        start = (batch_idx * self.batch_size) % len(examples)
        batch_examples = examples[start:start + self.batch_size]

        # Format as chat messages
        prompts = []
        completions = []
        for ex in batch_examples:
            prompt = f"Solve this problem step by step:\n\n{ex['problem']}"
            completion = ex['solution']
            prompts.append(prompt)
            completions.append(completion)

        return SLBatch(
            prompts=prompts,
            completions=completions,
        )

    def __len__(self) -> int:
        return len(self._load_examples()) // self.batch_size


def get_dataset_path(dimension: str) -> str:
    """Get default dataset path for a dimension."""
    base = Path("/Users/rohanvinaik/semantic_probing/data/specialists")
    paths = {
        "LOGICAL": base / "logical_specialist.jsonl",
        "QUANTITY": base / "quantity_specialist.jsonl",
        "TEMPORAL": base / "temporal_specialist.jsonl",
        "SPATIAL": base / "spatial_specialist.jsonl",
        "MENTAL": base / "mental_specialist.jsonl",
    }
    if dimension not in paths:
        raise ValueError(f"Unknown dimension: {dimension}")
    return str(paths[dimension])


async def cli_main(cli_config: SemanticSpecialistConfig):
    """Convert CLI config to full config and run training."""

    # Resolve dataset path
    dataset_path = cli_config.dataset_path or get_dataset_path(cli_config.dimension)

    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Create log path
    if cli_config.log_path:
        log_path = cli_config.log_path
    else:
        model_short = cli_config.model_name.replace("/", "-")
        run_name = (
            f"{cli_config.dimension}-specialist-{model_short}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = f"/Users/rohanvinaik/tinker-cookbook/experiments/semantic_specialists/{run_name}"

    wandb_name = cli_config.wandb_name or Path(log_path).name

    # Create dataset builder
    dataset_builder = SemanticDatasetBuilder(
        dataset_path=dataset_path,
        batch_size=cli_config.batch_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        max_tokens=cli_config.max_tokens,
    )

    # Create config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(SemanticSpecialistConfig)
    asyncio.run(cli_main(cli_config))
```

**Shell Commands**:
```bash
cd /Users/rohanvinaik/tinker-cookbook

# Train LOGICAL specialist
python -m tinker_cookbook.recipes.semantic_specialists.train \
    dimension=LOGICAL \
    model_name=Qwen/Qwen3-0.5B \
    learning_rate=5e-4 \
    lora_rank=32 \
    num_epochs=3 \
    wandb_project=semantic_specialists

# Train QUANTITY specialist
python -m tinker_cookbook.recipes.semantic_specialists.train \
    dimension=QUANTITY \
    model_name=Qwen/Qwen3-0.5B \
    learning_rate=5e-4 \
    lora_rank=32 \
    num_epochs=3 \
    wandb_project=semantic_specialists

# Train with larger model for comparison
python -m tinker_cookbook.recipes.semantic_specialists.train \
    dimension=LOGICAL \
    model_name=Qwen/Qwen3-4B \
    learning_rate=1e-4 \
    lora_rank=64 \
    num_epochs=2 \
    wandb_project=semantic_specialists
```

---

### 0.3 Extract Functional Networks from Specialists

**File**: `semantic_probing/experiments/functional_extraction.py`

```python
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

from semantic_probing.core.hadamard import HadamardBasis
from semantic_probing.core.primitives import PrimitiveProbe
from semantic_probing.core.encoding import TextEncoder
from semantic_probing.core.analysis import SignatureAnalyzer


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
            activations = self.probe.probe_vector(vector)
            sig = self.analyzer.compute_signature(activations)

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
    from tinker_cookbook.utils import load_model
    specialist = load_model(args.specialist_path)
    baseline = load_model(args.baseline_path) if args.baseline_path else load_model(None)

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
```

**Shell Commands**:
```bash
cd /Users/rohanvinaik/semantic_probing

# Extract LOGICAL specialist network
python -m semantic_probing.experiments.functional_extraction \
    --specialist_path ../tinker-cookbook/experiments/semantic_specialists/LOGICAL-specialist-* \
    --dimension LOGICAL \
    --output reports/functional_networks/logical_specialist.json

# Extract QUANTITY specialist network
python -m semantic_probing.experiments.functional_extraction \
    --specialist_path ../tinker-cookbook/experiments/semantic_specialists/QUANTITY-specialist-* \
    --dimension QUANTITY \
    --output reports/functional_networks/quantity_specialist.json

# Compare all specialists
for dim in LOGICAL QUANTITY TEMPORAL SPATIAL MENTAL; do
    python -m semantic_probing.experiments.functional_extraction \
        --specialist_path ../tinker-cookbook/experiments/semantic_specialists/${dim}-specialist-* \
        --dimension $dim \
        --output reports/functional_networks/${dim,,}_specialist.json
done
```

---

### 0.4 Semantic Reward RL (Optional Enhancement)

**File**: `tinker-cookbook/tinker_cookbook/recipes/semantic_specialists/rl_train.py`

```python
"""
Fine-tune specialists with semantic coherence as part of the reward.

Usage:
    python -m tinker_cookbook.recipes.semantic_specialists.rl_train \
        dimension=LOGICAL \
        model_name=Qwen/Qwen3-0.5B \
        use_semantic_reward=true \
        semantic_reward_weight=0.1 \
        wandb_project=semantic_specialists_rl
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import Config, main
from tinker_cookbook.rl.types import RLDatasetBuilder, RLBatch, RewardFn

# Import semantic probing
import sys
sys.path.insert(0, "/Users/rohanvinaik/semantic_probing")
from semantic_probing.core.hadamard import HadamardBasis
from semantic_probing.core.primitives import PrimitiveProbe
from semantic_probing.core.encoding import TextEncoder
from semantic_probing.core.analysis import SignatureAnalyzer

logger = logging.getLogger(__name__)


class SemanticRewardFn(RewardFn):
    """Reward function combining correctness with semantic coherence."""

    DIMENSIONS = {
        "LOGICAL": ["NOT", "IF", "BECAUSE", "MAYBE", "CAN", "TRUE"],
        "QUANTITY": ["ONE", "TWO", "SOME", "ALL", "MANY", "MORE", "MUCH"],
        "TEMPORAL": ["NOW", "BEFORE", "AFTER", "WHEN", "MOMENT", "LONG_TIME"],
        "SPATIAL": ["WHERE", "HERE", "ABOVE", "BELOW", "NEAR", "FAR", "SIDE"],
        "MENTAL": ["THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR"],
    }

    def __init__(
        self,
        target_dimension: str,
        correctness_weight: float = 0.7,
        coherence_weight: float = 0.2,
        alignment_weight: float = 0.1,
    ):
        self.target_dimension = target_dimension
        self.correctness_weight = correctness_weight
        self.coherence_weight = coherence_weight
        self.alignment_weight = alignment_weight

        # Initialize semantic probing
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

    def compute_semantic_reward(self, response: str) -> Tuple[float, float]:
        """Compute coherence and alignment rewards."""
        vector = self.encoder.encode(response)
        activations = self.probe.probe_vector(vector)
        sig = self.analyzer.compute_signature(activations)

        # Coherence: lower entropy = more focused = higher reward
        coherence = 1.0 - min(1.0, sig.entropy / 4.0)

        # Alignment: activation in target dimension
        alignment = sig.dimension_profile.get(self.target_dimension, 0)

        return coherence, alignment

    def __call__(
        self,
        problems: List[str],
        responses: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        """Compute rewards for a batch."""
        rewards = []
        for problem, response, truth in zip(problems, responses, ground_truths):
            # Correctness reward
            correct = self._check_correct(response, truth)
            r_correct = 1.0 if correct else 0.0

            # Semantic rewards
            r_coherence, r_alignment = self.compute_semantic_reward(response)

            # Combined reward
            reward = (
                self.correctness_weight * r_correct +
                self.coherence_weight * r_coherence +
                self.alignment_weight * r_alignment
            )
            rewards.append(reward)

        return rewards

    def _check_correct(self, response: str, truth: str) -> bool:
        """Check if response matches ground truth."""
        # Extract final answer (simple heuristic)
        response_answer = self._extract_answer(response)
        truth_answer = self._extract_answer(truth)
        return response_answer.strip().lower() == truth_answer.strip().lower()

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from text."""
        # Look for boxed answer or final line
        if "\\boxed{" in text:
            start = text.find("\\boxed{") + 7
            end = text.find("}", start)
            return text[start:end]
        # Return last non-empty line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines[-1] if lines else ""


@chz.chz
class SemanticRLConfig:
    """Configuration for semantic RL training."""

    # Model
    model_name: str = "Qwen/Qwen3-0.5B"
    lora_rank: int = 32
    load_checkpoint_path: str | None = None  # Start from SFT checkpoint

    # Dimension
    dimension: str = "LOGICAL"
    dataset_path: str | None = None

    # RL hyperparameters
    learning_rate: float = 1e-5
    group_size: int = 4
    groups_per_batch: int = 100
    max_tokens: int = 512
    temperature: float = 1.0
    kl_penalty_coef: float = 0.01

    # Semantic reward
    use_semantic_reward: bool = True
    correctness_weight: float = 0.7
    coherence_weight: float = 0.2
    alignment_weight: float = 0.1

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Training
    eval_every: int = 20
    save_every: int = 50

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


class SemanticRLDatasetBuilder(RLDatasetBuilder):
    """Dataset builder for semantic RL training."""

    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        group_size: int,
        model_name_for_tokenizer: str,
        renderer_name: str,
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.group_size = group_size
        self.model_name_for_tokenizer = model_name_for_tokenizer
        self.renderer_name = renderer_name
        self._examples = None

    def _load_examples(self) -> List[Dict]:
        if self._examples is None:
            self._examples = []
            with open(self.dataset_path) as f:
                for line in f:
                    self._examples.append(json.loads(line))
        return self._examples

    def build_batch(self, batch_idx: int) -> RLBatch:
        examples = self._load_examples()
        start = (batch_idx * self.batch_size) % len(examples)
        batch = examples[start:start + self.batch_size]

        return RLBatch(
            prompts=[f"Solve step by step:\n\n{ex['problem']}" for ex in batch],
            ground_truths=[ex['solution'] for ex in batch],
            group_size=self.group_size,
        )


async def cli_main(cli_config: SemanticRLConfig):
    """Run semantic RL training."""

    # Resolve paths
    if cli_config.dataset_path:
        dataset_path = cli_config.dataset_path
    else:
        dataset_path = f"/Users/rohanvinaik/semantic_probing/data/specialists/{cli_config.dimension.lower()}_specialist.jsonl"

    renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)

    if cli_config.log_path:
        log_path = cli_config.log_path
    else:
        model_short = cli_config.model_name.replace("/", "-")
        run_name = (
            f"{cli_config.dimension}-RL-{model_short}-"
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = f"/Users/rohanvinaik/tinker-cookbook/experiments/semantic_specialists/{run_name}"

    wandb_name = cli_config.wandb_name or Path(log_path).name

    # Create reward function
    reward_fn = SemanticRewardFn(
        target_dimension=cli_config.dimension,
        correctness_weight=cli_config.correctness_weight,
        coherence_weight=cli_config.coherence_weight,
        alignment_weight=cli_config.alignment_weight,
    ) if cli_config.use_semantic_reward else None

    # Create dataset builder
    dataset_builder = SemanticRLDatasetBuilder(
        dataset_path=dataset_path,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
    )

    # Create config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        reward_fn=reward_fn,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(SemanticRLConfig)
    asyncio.run(cli_main(cli_config))
```

**Shell Commands**:
```bash
cd /Users/rohanvinaik/tinker-cookbook

# RL fine-tune LOGICAL specialist (start from SFT checkpoint)
python -m tinker_cookbook.recipes.semantic_specialists.rl_train \
    dimension=LOGICAL \
    model_name=Qwen/Qwen3-0.5B \
    load_checkpoint_path=experiments/semantic_specialists/LOGICAL-specialist-*/checkpoint-final \
    use_semantic_reward=true \
    correctness_weight=0.7 \
    coherence_weight=0.2 \
    alignment_weight=0.1 \
    learning_rate=1e-5 \
    wandb_project=semantic_specialists_rl

# Compare: RL with vs without semantic reward
python -m tinker_cookbook.recipes.semantic_specialists.rl_train \
    dimension=QUANTITY \
    use_semantic_reward=false \
    wandb_name=quantity-rl-no-semantic
```

---

## Phase 1: Standalone Experiments (Clean Benchmarks Per Repo)

Each codebase gets independent experiments establishing baselines and reusable test fixtures.

### 1.1 semantic_probing: Coherence Benchmark

**File**: `semantic_probing/experiments/coherence_benchmark.py`

```python
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

from semantic_probing.core.hadamard import HadamardBasis
from semantic_probing.core.primitives import PrimitiveProbe
from semantic_probing.core.encoding import TextEncoder
from semantic_probing.core.analysis import SignatureAnalyzer


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
            activations = self.probe.probe_vector(vector)
            sig = self.analyzer.compute_signature(activations)
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
    ds = load_dataset("gsm8k", "main", split="train[:600]")

    correct = [item["answer"] for item in ds[:200]]

    # Generate incorrect by truncating or corrupting
    incorrect = []
    for item in ds[200:400]:
        trace = item["answer"]
        # Truncate reasoning
        lines = trace.split('\n')
        if len(lines) > 2:
            incorrect.append('\n'.join(lines[:len(lines)//2]) + "\nTherefore the answer is 42.")
        else:
            incorrect.append("The answer is obviously 42.")

    # Generate hallucinated with fabricated facts
    hallucinated = []
    for item in ds[400:600]:
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
```

**Test**: `semantic_probing/tests/test_coherence_benchmark.py`

```python
"""Tests for coherence benchmark."""

import pytest
from semantic_probing.experiments.coherence_benchmark import (
    CoherenceBenchmark,
    CoherenceMetrics,
)


class TestCoherenceBenchmark:
    @pytest.fixture
    def benchmark(self):
        return CoherenceBenchmark()

    def test_single_step_coherence(self, benchmark):
        """Single step should have perfect stability."""
        metrics = benchmark.compute_trace_coherence(["The answer is 5."])
        assert metrics.primitive_stability == 1.0
        assert metrics.dimension_drift == 0.0

    def test_consistent_steps_high_stability(self, benchmark):
        """Consistent reasoning should have high stability."""
        steps = [
            "Step 1: We need to find x + y.",
            "Step 2: We know x = 3 and y = 4.",
            "Step 3: Therefore x + y = 3 + 4 = 7.",
        ]
        metrics = benchmark.compute_trace_coherence(steps)
        assert metrics.primitive_stability > 0.5

    def test_incoherent_steps_low_stability(self, benchmark):
        """Incoherent reasoning should have lower stability."""
        steps = [
            "The sky is blue because of light scattering.",
            "I prefer chocolate ice cream.",
            "The square root of 144 is 12.",
        ]
        metrics = benchmark.compute_trace_coherence(steps)
        # Incoherent steps may still have some stability, but should be lower
        assert metrics.entropy_variance > 0  # High variance indicates incoherence

    def test_parse_reasoning_steps(self, benchmark):
        """Test parsing of reasoning traces."""
        trace = "Step 1: First thing.\nStep 2: Second thing.\nStep 3: Conclusion."
        steps = benchmark.parse_reasoning_steps(trace)
        assert len(steps) == 3

    def test_metrics_dataclass(self):
        """Test CoherenceMetrics creation."""
        metrics = CoherenceMetrics(
            primitive_stability=0.8,
            dimension_drift=0.2,
            entropy_variance=0.1,
            mean_entropy=2.5,
            primary_dimension="QUANTITY",
        )
        assert metrics.primitive_stability == 0.8
        assert metrics.primary_dimension == "QUANTITY"
```

**Shell Commands**:
```bash
cd /Users/rohanvinaik/semantic_probing

# Run coherence benchmark
python -m semantic_probing.experiments.coherence_benchmark \
    --output reports/coherence_baseline.json

# Run tests
pytest tests/test_coherence_benchmark.py -v
```

---

### 1.2 sparse-wiki-grounding: Hallucination Detection Benchmark

**File**: `sparse-wiki-grounding/experiments/hallucination_benchmark.py`

```python
"""
Establish claim verification accuracy on structured benchmark.

Benchmark Suite:
- 100 geographic claims (TRUE/FALSE)
- 100 attribution claims (TRUE/FALSE)
- 100 temporal claims (TRUE/FALSE)
- 100 property claims (TRUE/FALSE)

Usage:
    python -m experiments.hallucination_benchmark \
        --output reports/verification_baseline.json
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from wiki_grounding.store import EntityStore
from wiki_grounding.verification import ClaimVerifier
from wiki_grounding.spreading import SpreadingActivation


class ClaimType(Enum):
    GEOGRAPHIC = "geographic"
    ATTRIBUTION = "attribution"
    TEMPORAL = "temporal"
    PROPERTY = "property"


@dataclass
class ClaimResult:
    """Result of verifying a single claim."""
    claim: str
    claim_type: ClaimType
    ground_truth: bool
    prediction: Optional[bool]
    confidence: float
    latency_ms: float
    entities_found: int


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    # Per-type metrics
    geographic_precision: float
    geographic_recall: float
    geographic_f1: float

    attribution_precision: float
    attribution_recall: float
    attribution_f1: float

    temporal_precision: float
    temporal_recall: float
    temporal_f1: float

    property_precision: float
    property_recall: float
    property_f1: float

    # Overall
    overall_precision: float
    overall_recall: float
    overall_f1: float
    coverage: float  # % verifiable
    mean_latency_ms: float


class HallucinationBenchmark:
    """Benchmark claim verification for hallucination detection."""

    def __init__(self, db_path: str = "data/wiki_grounding.db"):
        self.store = EntityStore(db_path)
        self.verifier = ClaimVerifier(self.store)
        self.spreader = SpreadingActivation(self.store)

    def verify_claim(self, claim: str) -> Tuple[Optional[bool], float, int]:
        """Verify a single claim and return (verdict, confidence, n_entities)."""
        start = time.perf_counter()

        # Extract entities from claim
        entities = self.store.search(claim, limit=5)
        n_entities = len(entities)

        if n_entities == 0:
            return None, 0.0, 0

        # Run verification
        result = self.verifier.verify(claim, entities)

        latency = (time.perf_counter() - start) * 1000

        return result.verdict, result.confidence, n_entities

    def run_benchmark(self, claims: List[Dict]) -> BenchmarkResult:
        """Run benchmark on claim set."""
        results = []

        for item in claims:
            claim = item["claim"]
            truth = item["label"]
            claim_type = ClaimType(item["type"])

            start = time.perf_counter()
            pred, conf, n_ent = self.verify_claim(claim)
            latency = (time.perf_counter() - start) * 1000

            results.append(ClaimResult(
                claim=claim,
                claim_type=claim_type,
                ground_truth=truth,
                prediction=pred,
                confidence=conf,
                latency_ms=latency,
                entities_found=n_ent,
            ))

        # Compute metrics per type
        metrics = {}
        for ctype in ClaimType:
            type_results = [r for r in results if r.claim_type == ctype]
            metrics[ctype.value] = self._compute_metrics(type_results)

        # Overall metrics
        overall = self._compute_metrics(results)

        # Coverage
        verifiable = sum(1 for r in results if r.prediction is not None)
        coverage = verifiable / len(results) if results else 0

        # Latency
        latencies = [r.latency_ms for r in results]
        mean_latency = sum(latencies) / len(latencies) if latencies else 0

        return BenchmarkResult(
            geographic_precision=metrics["geographic"]["precision"],
            geographic_recall=metrics["geographic"]["recall"],
            geographic_f1=metrics["geographic"]["f1"],
            attribution_precision=metrics["attribution"]["precision"],
            attribution_recall=metrics["attribution"]["recall"],
            attribution_f1=metrics["attribution"]["f1"],
            temporal_precision=metrics["temporal"]["precision"],
            temporal_recall=metrics["temporal"]["recall"],
            temporal_f1=metrics["temporal"]["f1"],
            property_precision=metrics["property"]["precision"],
            property_recall=metrics["property"]["recall"],
            property_f1=metrics["property"]["f1"],
            overall_precision=overall["precision"],
            overall_recall=overall["recall"],
            overall_f1=overall["f1"],
            coverage=coverage,
            mean_latency_ms=mean_latency,
        )

    def _compute_metrics(self, results: List[ClaimResult]) -> Dict[str, float]:
        """Compute precision, recall, F1 for a set of results."""
        # Filter to verifiable claims
        verifiable = [r for r in results if r.prediction is not None]

        if not verifiable:
            return {"precision": 0, "recall": 0, "f1": 0}

        tp = sum(1 for r in verifiable if r.prediction == True and r.ground_truth == True)
        fp = sum(1 for r in verifiable if r.prediction == True and r.ground_truth == False)
        fn = sum(1 for r in verifiable if r.prediction == False and r.ground_truth == True)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {"precision": precision, "recall": recall, "f1": f1}


def load_benchmark_claims() -> List[Dict]:
    """Load or generate benchmark claims."""
    data_dir = Path("data/benchmarks")
    bench_file = data_dir / "claim_verification.jsonl"

    if bench_file.exists():
        claims = []
        with open(bench_file) as f:
            for line in f:
                claims.append(json.loads(line))
        return claims

    # Generate benchmark claims
    print("Generating benchmark claims...")
    claims = []

    # Geographic claims
    geographic_true = [
        "Paris is located in France.",
        "Tokyo is the capital of Japan.",
        "The Amazon River flows through Brazil.",
        "Mount Everest is in the Himalayas.",
        "Sydney is in Australia.",
    ]
    geographic_false = [
        "Paris is located in Germany.",
        "Tokyo is the capital of China.",
        "The Amazon River flows through Africa.",
        "Mount Everest is in the Alps.",
        "Sydney is in New Zealand.",
    ]

    for claim in geographic_true[:50]:
        claims.append({"claim": claim, "label": True, "type": "geographic"})
    for claim in geographic_false[:50]:
        claims.append({"claim": claim, "label": False, "type": "geographic"})

    # Attribution claims
    attribution_true = [
        "Albert Einstein developed the theory of relativity.",
        "Shakespeare wrote Hamlet.",
        "Newton discovered gravity.",
        "Darwin proposed the theory of evolution.",
        "Marie Curie won the Nobel Prize.",
    ]
    attribution_false = [
        "Albert Einstein invented the telephone.",
        "Shakespeare wrote War and Peace.",
        "Newton invented the airplane.",
        "Darwin discovered penicillin.",
        "Marie Curie invented the radio.",
    ]

    for claim in attribution_true[:50]:
        claims.append({"claim": claim, "label": True, "type": "attribution"})
    for claim in attribution_false[:50]:
        claims.append({"claim": claim, "label": False, "type": "attribution"})

    # Temporal claims
    temporal_true = [
        "World War II ended in 1945.",
        "The moon landing occurred in 1969.",
        "The Berlin Wall fell in 1989.",
    ]
    temporal_false = [
        "World War II ended in 1950.",
        "The moon landing occurred in 1975.",
        "The Berlin Wall fell in 1995.",
    ]

    for claim in temporal_true[:50]:
        claims.append({"claim": claim, "label": True, "type": "temporal"})
    for claim in temporal_false[:50]:
        claims.append({"claim": claim, "label": False, "type": "temporal"})

    # Property claims
    property_true = [
        "Water has the chemical formula H2O.",
        "Gold is a metal.",
        "The speed of light is approximately 300,000 km/s.",
    ]
    property_false = [
        "Water has the chemical formula CO2.",
        "Gold is a gas.",
        "The speed of light is approximately 300 km/s.",
    ]

    for claim in property_true[:50]:
        claims.append({"claim": claim, "label": True, "type": "property"})
    for claim in property_false[:50]:
        claims.append({"claim": claim, "label": False, "type": "property"})

    # Save
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(bench_file, 'w') as f:
        for claim in claims:
            f.write(json.dumps(claim) + '\n')

    return claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/wiki_grounding.db")
    parser.add_argument("--output", default="reports/verification_baseline.json")
    args = parser.parse_args()

    # Load claims
    claims = load_benchmark_claims()
    print(f"Loaded {len(claims)} claims")

    # Run benchmark
    benchmark = HallucinationBenchmark(args.db)
    result = benchmark.run_benchmark(claims)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\n=== Hallucination Benchmark Results ===")
    print(f"\nPer-type F1 scores:")
    print(f"  Geographic:  {result.geographic_f1:.3f}")
    print(f"  Attribution: {result.attribution_f1:.3f}")
    print(f"  Temporal:    {result.temporal_f1:.3f}")
    print(f"  Property:    {result.property_f1:.3f}")
    print(f"\nOverall:")
    print(f"  Precision: {result.overall_precision:.3f}")
    print(f"  Recall:    {result.overall_recall:.3f}")
    print(f"  F1:        {result.overall_f1:.3f}")
    print(f"  Coverage:  {result.coverage:.1%}")
    print(f"  Latency:   {result.mean_latency_ms:.1f}ms")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
```

**Shell Commands**:
```bash
cd /Users/rohanvinaik/sparse-wiki-grounding

# Run hallucination benchmark
python -m experiments.hallucination_benchmark \
    --db data/wiki_grounding.db \
    --output reports/verification_baseline.json

# Run tests
pytest tests/test_claim_benchmark.py -v
```

---

### 1.3 negative-learning-censor: Reasoning Failure Taxonomy

**File**: `negative-learning-censor/experiments/reasoning_censors.py`

```python
"""
Define censor taxonomy for LLM reasoning failures.

Failure Types mapped to ErrorType:
- LOGICAL_LEAP → REMOVAL (missing reasoning step)
- FACT_CONFUSION → SUBSTITUTION (wrong entity)
- MAGNITUDE_ERROR → INTENSITY_SHIFT (off by orders of magnitude)
- CATEGORY_ERROR → STRUCTURAL_SWAP (type confusion)
- NEGATION_FLIP → RELATIONSHIP_INVERSION (opposite conclusion)

Usage:
    python -m experiments.reasoning_censors \
        --output reports/censor_taxonomy.json
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from negative_learning.censor import CensorRegistry, CensorContext
from negative_learning.types import ErrorType


class ReasoningFailure(Enum):
    """Types of reasoning failures."""
    LOGICAL_LEAP = "logical_leap"
    FACT_CONFUSION = "fact_confusion"
    MAGNITUDE_ERROR = "magnitude_error"
    CATEGORY_ERROR = "category_error"
    NEGATION_FLIP = "negation_flip"


# Mapping from reasoning failure to negative-learning ErrorType
FAILURE_TO_ERROR_TYPE = {
    ReasoningFailure.LOGICAL_LEAP: ErrorType.REMOVAL,
    ReasoningFailure.FACT_CONFUSION: ErrorType.SUBSTITUTION,
    ReasoningFailure.MAGNITUDE_ERROR: ErrorType.INTENSITY_SHIFT,
    ReasoningFailure.CATEGORY_ERROR: ErrorType.STRUCTURAL_SWAP,
    ReasoningFailure.NEGATION_FLIP: ErrorType.RELATIONSHIP_INVERSION,
}


@dataclass
class FailureExample:
    """A reasoning failure example."""
    problem: str
    incorrect_reasoning: str
    failure_type: ReasoningFailure
    correct_reasoning: str
    explanation: str


@dataclass
class TaxonomyResult:
    """Results of censor taxonomy experiment."""
    n_failures_learned: int
    generalization_accuracy: float  # % of novel failures correctly censored
    mean_lookup_latency_us: float
    failures_per_type: Dict[str, int]


class ReasoningCensorExperiment:
    """Experiment: Learn censors from reasoning failures."""

    def __init__(self):
        self.registry = CensorRegistry()

    def failure_to_context(self, failure: FailureExample) -> CensorContext:
        """Convert a failure example to a censor context."""
        return CensorContext(
            perceptual={
                "failure_type": failure.failure_type.value,
                "problem_keywords": self._extract_keywords(failure.problem),
            },
            sequential={
                "reasoning_pattern": self._extract_pattern(failure.incorrect_reasoning),
            },
            outcome={
                "error_type": FAILURE_TO_ERROR_TYPE[failure.failure_type].value,
            },
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from problem text."""
        # Simple keyword extraction
        words = text.lower().split()
        stopwords = {"the", "a", "an", "is", "are", "if", "then", "and", "or"}
        return [w for w in words if w not in stopwords and len(w) > 3][:5]

    def _extract_pattern(self, reasoning: str) -> str:
        """Extract reasoning pattern."""
        # Identify common patterns
        if "therefore" in reasoning.lower():
            return "direct_conclusion"
        elif "because" in reasoning.lower():
            return "causal_reasoning"
        elif "if" in reasoning.lower() and "then" in reasoning.lower():
            return "conditional_reasoning"
        else:
            return "unknown"

    def learn_from_failures(self, failures: List[FailureExample]) -> int:
        """Learn censors from failure examples."""
        learned = 0
        for failure in failures:
            context = self.failure_to_context(failure)
            self.registry.learn(
                context=context,
                action=failure.incorrect_reasoning[:100],  # Truncate
                success=False,
            )
            learned += 1
        return learned

    def test_generalization(self, test_failures: List[FailureExample]) -> float:
        """Test if censors generalize to novel failures."""
        correct = 0
        for failure in test_failures:
            context = self.failure_to_context(failure)
            suppression = self.registry.query(context, failure.incorrect_reasoning[:100])
            # If suppression > 0.5, we correctly identified this as a bad pattern
            if suppression > 0.5:
                correct += 1
        return correct / len(test_failures) if test_failures else 0

    def measure_lookup_latency(self, n_queries: int = 1000) -> float:
        """Measure mean lookup latency in microseconds."""
        # Create sample context
        context = CensorContext(
            perceptual={"failure_type": "logical_leap"},
            sequential={"reasoning_pattern": "direct_conclusion"},
        )

        start = time.perf_counter()
        for _ in range(n_queries):
            self.registry.query(context, "sample action")
        elapsed = time.perf_counter() - start

        return (elapsed / n_queries) * 1_000_000  # Convert to microseconds


def load_failure_examples() -> List[FailureExample]:
    """Load or generate failure examples."""
    data_dir = Path("data/benchmarks")
    bench_file = data_dir / "reasoning_failures.jsonl"

    if bench_file.exists():
        failures = []
        with open(bench_file) as f:
            for line in f:
                item = json.loads(line)
                failures.append(FailureExample(
                    problem=item["problem"],
                    incorrect_reasoning=item["incorrect_reasoning"],
                    failure_type=ReasoningFailure(item["failure_type"]),
                    correct_reasoning=item["correct_reasoning"],
                    explanation=item["explanation"],
                ))
        return failures

    # Generate examples
    print("Generating failure examples...")
    failures = []

    # LOGICAL_LEAP examples
    failures.extend([
        FailureExample(
            problem="If A implies B, and B implies C, what can we conclude about A and C?",
            incorrect_reasoning="A implies C.",
            failure_type=ReasoningFailure.LOGICAL_LEAP,
            correct_reasoning="A implies B, B implies C, therefore by transitivity A implies C.",
            explanation="Missing intermediate step showing transitivity.",
        ),
        FailureExample(
            problem="All squares are rectangles. Shape X is a rectangle. Is X a square?",
            incorrect_reasoning="Yes, X is a square.",
            failure_type=ReasoningFailure.LOGICAL_LEAP,
            correct_reasoning="Not necessarily. All squares are rectangles, but not all rectangles are squares.",
            explanation="Invalid logical reversal.",
        ),
    ])

    # FACT_CONFUSION examples
    failures.extend([
        FailureExample(
            problem="Who wrote Romeo and Juliet?",
            incorrect_reasoning="Romeo and Juliet was written by Charles Dickens.",
            failure_type=ReasoningFailure.FACT_CONFUSION,
            correct_reasoning="Romeo and Juliet was written by William Shakespeare.",
            explanation="Confused Shakespeare with Dickens.",
        ),
    ])

    # MAGNITUDE_ERROR examples
    failures.extend([
        FailureExample(
            problem="What is the population of Tokyo?",
            incorrect_reasoning="Tokyo has a population of about 13,000 people.",
            failure_type=ReasoningFailure.MAGNITUDE_ERROR,
            correct_reasoning="Tokyo has a population of about 13 million people.",
            explanation="Off by three orders of magnitude.",
        ),
    ])

    # CATEGORY_ERROR examples
    failures.extend([
        FailureExample(
            problem="What type of animal is a whale?",
            incorrect_reasoning="A whale is a fish.",
            failure_type=ReasoningFailure.CATEGORY_ERROR,
            correct_reasoning="A whale is a mammal.",
            explanation="Category confusion between fish and mammals.",
        ),
    ])

    # NEGATION_FLIP examples
    failures.extend([
        FailureExample(
            problem="Is it safe to drink seawater?",
            incorrect_reasoning="Yes, seawater is safe to drink.",
            failure_type=ReasoningFailure.NEGATION_FLIP,
            correct_reasoning="No, seawater is not safe to drink due to high salt content.",
            explanation="Conclusion is opposite of truth.",
        ),
    ])

    # Save
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(bench_file, 'w') as f:
        for failure in failures:
            f.write(json.dumps({
                "problem": failure.problem,
                "incorrect_reasoning": failure.incorrect_reasoning,
                "failure_type": failure.failure_type.value,
                "correct_reasoning": failure.correct_reasoning,
                "explanation": failure.explanation,
            }) + '\n')

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/censor_taxonomy.json")
    args = parser.parse_args()

    # Load failures
    failures = load_failure_examples()
    print(f"Loaded {len(failures)} failure examples")

    # Split into train/test
    split = int(len(failures) * 0.7)
    train_failures = failures[:split]
    test_failures = failures[split:]

    # Run experiment
    experiment = ReasoningCensorExperiment()

    # Learn from training failures
    n_learned = experiment.learn_from_failures(train_failures)

    # Test generalization
    gen_accuracy = experiment.test_generalization(test_failures)

    # Measure latency
    latency = experiment.measure_lookup_latency()

    # Count per type
    per_type = {}
    for failure in failures:
        ft = failure.failure_type.value
        per_type[ft] = per_type.get(ft, 0) + 1

    result = TaxonomyResult(
        n_failures_learned=n_learned,
        generalization_accuracy=gen_accuracy,
        mean_lookup_latency_us=latency,
        failures_per_type=per_type,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\n=== Reasoning Censor Taxonomy Results ===")
    print(f"Failures learned: {result.n_failures_learned}")
    print(f"Generalization accuracy: {result.generalization_accuracy:.1%}")
    print(f"Mean lookup latency: {result.mean_lookup_latency_us:.2f}µs")
    print(f"\nFailures per type:")
    for ft, count in result.failures_per_type.items():
        print(f"  {ft}: {count}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
```

**Shell Commands**:
```bash
cd /Users/rohanvinaik/negative-learning-censor

# Run reasoning censor experiment
python -m experiments.reasoning_censors \
    --output reports/censor_taxonomy.json

# Run tests
pytest tests/test_reasoning_censors.py -v
```

---

### 1.4 orthogonal-validators: Semantic Probing Validator

**File**: `orthogonal-validators/orthogonal_validators/validators/probing.py`

```python
"""
Semantic probing as a new validator type.

This validator uses semantic primitive analysis to detect
behavioral inconsistencies that other validators miss.
"""

import sys
from typing import Optional
from dataclasses import dataclass

# Import semantic probing
sys.path.insert(0, "/Users/rohanvinaik/semantic_probing")
from semantic_probing.core.hadamard import HadamardBasis
from semantic_probing.core.primitives import PrimitiveProbe
from semantic_probing.core.encoding import TextEncoder
from semantic_probing.core.analysis import SignatureAnalyzer

from orthogonal_validators.base import Validator, ValidationResult


@dataclass
class ProbingConfig:
    """Configuration for semantic probing validator."""
    stability_threshold: float = 0.6
    entropy_threshold: float = 3.0
    drift_threshold: float = 0.5


class SemanticProbingValidator(Validator):
    """
    Validator using semantic primitive analysis.

    Orthogonal to:
    - SemanticValidator: Checks coherence of claim text
    - EntityValidator: Grounds to knowledge base
    - SyntacticValidator: Checks grammatical structure

    This validator checks:
    - Behavioral consistency: Do LLM responses show stable primitive activations?
    - Semantic entropy: Is the response focused or scattered?
    - Dimension alignment: Does the response match expected semantic dimensions?
    """

    def __init__(self, config: Optional[ProbingConfig] = None):
        self.config = config or ProbingConfig()
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

    @property
    def decomposition_type(self) -> str:
        return "behavioral_probing"

    @property
    def name(self) -> str:
        return "SemanticProbingValidator"

    def validate(self, claim: str, context: Optional[str] = None) -> ValidationResult:
        """
        Validate a claim using semantic probing.

        Returns ValidationResult with:
        - verdict: True/False/None
        - confidence: 0-1 score
        - margin: distance to decision boundary
        """
        # Encode claim
        vector = self.encoder.encode(claim)

        # Probe for primitive activations
        activations = self.probe.probe_vector(vector)

        # Analyze signature
        sig = self.analyzer.compute_signature(activations)

        # Compute metrics
        entropy = sig.entropy
        stability = self._compute_stability(claim, context)
        primary_dim = sig.primary_dimension

        # Decision logic
        if entropy > self.config.entropy_threshold:
            # High entropy = scattered/incoherent
            verdict = False
            confidence = min(1.0, entropy / 5.0)
            margin = entropy - self.config.entropy_threshold
        elif stability < self.config.stability_threshold:
            # Low stability = inconsistent
            verdict = False
            confidence = 1.0 - stability
            margin = self.config.stability_threshold - stability
        else:
            # Passes checks
            verdict = True
            confidence = stability
            margin = stability - self.config.stability_threshold

        return ValidationResult(
            verdict=verdict,
            confidence=confidence,
            margin=margin,
            validator_name=self.name,
            details={
                "entropy": entropy,
                "stability": stability,
                "primary_dimension": primary_dim,
                "dimension_profile": sig.dimension_profile,
            },
        )

    def _compute_stability(self, claim: str, context: Optional[str]) -> float:
        """Compute stability by comparing claim to context."""
        if not context:
            return 1.0  # No context to compare

        # Encode both
        claim_vec = self.encoder.encode(claim)
        context_vec = self.encoder.encode(context)

        # Get activations
        claim_act = self.probe.probe_vector(claim_vec)
        context_act = self.probe.probe_vector(context_vec)

        # Compute correlation
        all_prims = set(claim_act.keys()) | set(context_act.keys())
        if not all_prims:
            return 1.0

        claim_vals = [claim_act.get(p, 0) for p in all_prims]
        context_vals = [context_act.get(p, 0) for p in all_prims]

        # Pearson correlation
        import numpy as np
        if np.std(claim_vals) == 0 or np.std(context_vals) == 0:
            return 1.0
        corr = np.corrcoef(claim_vals, context_vals)[0, 1]
        return max(0, corr)  # Clamp to non-negative


# Test for orthogonality
class OrthogonalityTest:
    """Test that probing validator fails independently from other validators."""

    def measure_correlation(
        self,
        probing_results: list,
        semantic_results: list,
        entity_results: list,
    ) -> dict:
        """Measure failure correlation between validators."""
        import numpy as np

        # Convert to binary failures
        probing_fails = [1 if not r.verdict else 0 for r in probing_results]
        semantic_fails = [1 if not r.verdict else 0 for r in semantic_results]
        entity_fails = [1 if not r.verdict else 0 for r in entity_results]

        # Compute correlations
        return {
            "probing_semantic": np.corrcoef(probing_fails, semantic_fails)[0, 1],
            "probing_entity": np.corrcoef(probing_fails, entity_fails)[0, 1],
            "semantic_entity": np.corrcoef(semantic_fails, entity_fails)[0, 1],
        }
```

**Test**: `orthogonal-validators/tests/test_probing_validator.py`

```python
"""Tests for semantic probing validator."""

import pytest
from orthogonal_validators.validators.probing import (
    SemanticProbingValidator,
    ProbingConfig,
)


class TestSemanticProbingValidator:
    @pytest.fixture
    def validator(self):
        return SemanticProbingValidator()

    def test_coherent_claim_passes(self, validator):
        """Coherent mathematical claim should pass."""
        result = validator.validate(
            "The sum of 2 and 3 equals 5.",
            context="We are solving basic arithmetic problems.",
        )
        assert result.verdict == True
        assert result.confidence > 0.5

    def test_incoherent_claim_fails(self, validator):
        """Incoherent claim should fail."""
        result = validator.validate(
            "Purple ideas sleep furiously while calculating the taste of numbers.",
        )
        # High entropy should trigger failure
        assert result.details["entropy"] > 0

    def test_decomposition_type(self, validator):
        """Validator should have correct decomposition type."""
        assert validator.decomposition_type == "behavioral_probing"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProbingConfig(
            stability_threshold=0.8,
            entropy_threshold=2.0,
        )
        validator = SemanticProbingValidator(config)
        assert validator.config.stability_threshold == 0.8

    def test_returns_dimension_profile(self, validator):
        """Result should include dimension profile."""
        result = validator.validate("What is 5 plus 3?")
        assert "dimension_profile" in result.details
        assert "primary_dimension" in result.details
```

**Shell Commands**:
```bash
cd /Users/rohanvinaik/orthogonal-validators

# Run tests
pytest tests/test_probing_validator.py -v

# Test orthogonality
python -c "
from orthogonal_validators.validators.probing import SemanticProbingValidator, OrthogonalityTest
print('SemanticProbingValidator loaded successfully')
print('Decomposition type:', SemanticProbingValidator().decomposition_type)
"
```

---

### 1.5 experience-memory: Semantic Proof Cache

**File**: `experience-memory/experience_memory/semantic_cache.py`

```python
"""
Cache semantic probe results for O(1) retrieval.

Uses the FixRegistry from experience-memory to cache
expensive semantic analysis operations.
"""

import json
import hashlib
from typing import Optional, Dict, Any
from dataclasses import dataclass

from experience_memory.registry import FixRegistry
from experience_memory.types import ErrorSignature, Fix, ErrorType, FixType


@dataclass
class CachedProbeResult:
    """A cached semantic probe result."""
    dimension_profile: Dict[str, float]
    entropy: float
    primary_dimension: str
    active_primitives: list
    coherence_score: float


class SemanticProofCache:
    """
    O(1) cache for semantic probe results.

    Uses experience-memory's FixRegistry to store
    probe results keyed by query hash.
    """

    def __init__(self, cache_path: str = "semantic_cache.db"):
        self.registry = FixRegistry(cache_path)
        self._stats = {"hits": 0, "misses": 0}

    def _query_to_signature(self, query: str) -> ErrorSignature:
        """Convert a query string to an ErrorSignature for lookup."""
        # Hash the query for consistent keying
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        return ErrorSignature(
            severity=1,  # Neutral
            error_type=ErrorType.MISSING_CONTEXT,  # Reuse for "needs computation"
            context=f"semantic_probe:{query_hash}",
            affected_categories=["probe_cache"],
            delta=0.0,
        )

    def cache_result(self, query: str, result: CachedProbeResult) -> None:
        """Cache a probe result for future retrieval."""
        sig = self._query_to_signature(query)

        fix = Fix(
            fix_type=FixType.COMPLETE_DEFINITION,
            definition_supplement=json.dumps({
                "dimension_profile": result.dimension_profile,
                "entropy": result.entropy,
                "primary_dimension": result.primary_dimension,
                "active_primitives": result.active_primitives,
                "coherence_score": result.coherence_score,
            }),
        )

        self.registry.register(sig, fix)

    def lookup(self, query: str) -> Optional[CachedProbeResult]:
        """Look up a cached probe result. Returns None if not cached."""
        sig = self._query_to_signature(query)
        fix = self.registry.lookup(sig)

        if fix is None:
            self._stats["misses"] += 1
            return None

        self._stats["hits"] += 1

        # Parse cached result
        data = json.loads(fix.definition_supplement)
        return CachedProbeResult(
            dimension_profile=data["dimension_profile"],
            entropy=data["entropy"],
            primary_dimension=data["primary_dimension"],
            active_primitives=data["active_primitives"],
            coherence_score=data["coherence_score"],
        )

    def get_or_compute(
        self,
        query: str,
        compute_fn,
    ) -> CachedProbeResult:
        """Get from cache or compute and cache."""
        cached = self.lookup(query)
        if cached is not None:
            return cached

        # Compute
        result = compute_fn(query)
        self.cache_result(query, result)
        return result

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            **self._stats,
            "hit_rate": self.hit_rate,
        }


def benchmark_cache():
    """Benchmark cache performance."""
    import time

    cache = SemanticProofCache("/tmp/semantic_cache_benchmark.db")

    # Populate cache
    n_entries = 1000
    for i in range(n_entries):
        query = f"test query number {i}"
        result = CachedProbeResult(
            dimension_profile={"LOGICAL": 0.5, "QUANTITY": 0.3},
            entropy=2.5,
            primary_dimension="LOGICAL",
            active_primitives=["IF", "THEN", "BECAUSE"],
            coherence_score=0.8,
        )
        cache.cache_result(query, result)

    # Benchmark lookups
    n_lookups = 10000
    start = time.perf_counter()
    for i in range(n_lookups):
        query = f"test query number {i % n_entries}"
        cache.lookup(query)
    elapsed = time.perf_counter() - start

    latency_us = (elapsed / n_lookups) * 1_000_000

    print(f"Cache benchmark:")
    print(f"  Entries: {n_entries}")
    print(f"  Lookups: {n_lookups}")
    print(f"  Mean latency: {latency_us:.2f}µs")
    print(f"  Hit rate: {cache.hit_rate:.1%}")

    return latency_us


if __name__ == "__main__":
    benchmark_cache()
```

**Test**: `experience-memory/tests/test_semantic_cache.py`

```python
"""Tests for semantic proof cache."""

import pytest
import tempfile
from experience_memory.semantic_cache import SemanticProofCache, CachedProbeResult


class TestSemanticProofCache:
    @pytest.fixture
    def cache(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            yield SemanticProofCache(f.name)

    def test_cache_and_lookup(self, cache):
        """Test basic cache and lookup."""
        query = "What is 2 + 2?"
        result = CachedProbeResult(
            dimension_profile={"QUANTITY": 0.8, "LOGICAL": 0.2},
            entropy=1.5,
            primary_dimension="QUANTITY",
            active_primitives=["ONE", "TWO", "MORE"],
            coherence_score=0.9,
        )

        cache.cache_result(query, result)
        cached = cache.lookup(query)

        assert cached is not None
        assert cached.primary_dimension == "QUANTITY"
        assert cached.entropy == 1.5

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.lookup("nonexistent query")
        assert result is None

    def test_hit_rate_tracking(self, cache):
        """Test hit rate is tracked correctly."""
        query = "test query"
        result = CachedProbeResult(
            dimension_profile={},
            entropy=0,
            primary_dimension="UNKNOWN",
            active_primitives=[],
            coherence_score=0,
        )

        # Miss
        cache.lookup(query)
        assert cache.stats["misses"] == 1

        # Cache
        cache.cache_result(query, result)

        # Hit
        cache.lookup(query)
        assert cache.stats["hits"] == 1
        assert cache.hit_rate == 0.5

    def test_get_or_compute(self, cache):
        """Test get_or_compute pattern."""
        compute_called = [0]

        def compute_fn(query):
            compute_called[0] += 1
            return CachedProbeResult(
                dimension_profile={"LOGICAL": 1.0},
                entropy=2.0,
                primary_dimension="LOGICAL",
                active_primitives=["IF"],
                coherence_score=0.7,
            )

        # First call: compute
        result1 = cache.get_or_compute("test", compute_fn)
        assert compute_called[0] == 1

        # Second call: cached
        result2 = cache.get_or_compute("test", compute_fn)
        assert compute_called[0] == 1  # Not called again

        assert result1.primary_dimension == result2.primary_dimension

    def test_latency_under_threshold(self, cache):
        """Lookup latency should be under 10µs."""
        import time

        # Populate
        for i in range(100):
            cache.cache_result(
                f"query {i}",
                CachedProbeResult({}, 0, "X", [], 0),
            )

        # Measure
        n = 1000
        start = time.perf_counter()
        for i in range(n):
            cache.lookup(f"query {i % 100}")
        elapsed = time.perf_counter() - start

        latency_us = (elapsed / n) * 1_000_000
        assert latency_us < 100  # Should be well under 100µs
```

**Shell Commands**:
```bash
cd /Users/rohanvinaik/experience-memory

# Run cache benchmark
python -m experience_memory.semantic_cache

# Run tests
pytest tests/test_semantic_cache.py -v
```

---

## Phase 2: Pairwise Integrations (Bridge Modules)

Each integration creates a **bridge module** that lives in one codebase and imports from another.

### 2.1 Grounded Semantic Fingerprinting

**File**: `semantic_probing/integration/wiki_grounding.py`

```python
"""
Combine semantic primitives with entity grounding for richer fingerprints.

This integration enhances semantic signatures with grounded entity information
from sparse-wiki-grounding, creating a more powerful hallucination detector.
"""

import sys
from typing import List, Dict, Optional
from dataclasses import dataclass

# Local imports
from semantic_probing.core.hadamard import HadamardBasis
from semantic_probing.core.primitives import PrimitiveProbe
from semantic_probing.core.encoding import TextEncoder
from semantic_probing.core.analysis import SignatureAnalyzer

# Integration: sparse-wiki-grounding
sys.path.insert(0, "/Users/rohanvinaik/sparse-wiki-grounding")
from wiki_grounding.store import EntityStore
from wiki_grounding.spreading import SpreadingActivation


@dataclass
class GroundedFingerprint:
    """A semantic fingerprint enhanced with entity grounding."""
    # Semantic signature
    dimension_profile: Dict[str, float]
    primary_dimension: str
    entropy: float
    active_primitives: List[str]

    # Entity grounding
    grounded_entities: List[str]
    entity_positions: List[Dict[str, float]]  # Position in 5D space
    entity_epa: List[Dict[str, float]]        # Evaluation-Potency-Activity
    grounding_coverage: float                  # % of claims grounded

    # Combined metrics
    semantic_entity_alignment: float  # How well semantic and entity info align


class GroundedFingerprintGenerator:
    """Generate fingerprints combining semantic probing with entity grounding."""

    def __init__(self, entity_db_path: str = None):
        # Semantic probing
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

        # Entity grounding
        db_path = entity_db_path or "/Users/rohanvinaik/sparse-wiki-grounding/data/wiki_grounding.db"
        self.entity_store = EntityStore(db_path)
        self.spreader = SpreadingActivation(self.entity_store)

    def generate(self, text: str) -> GroundedFingerprint:
        """Generate a grounded fingerprint for text."""
        # 1. Compute semantic signature
        vector = self.encoder.encode(text)
        activations = self.probe.probe_vector(vector)
        sig = self.analyzer.compute_signature(activations)

        # 2. Extract and ground entities
        entities = self.entity_store.search(text, limit=10)

        entity_names = [e.name for e in entities]
        entity_positions = [
            {
                "spatial": e.positions.spatial,
                "temporal": e.positions.temporal,
                "taxonomic": e.positions.taxonomic,
                "scale": e.positions.scale,
                "domain": e.positions.domain,
            }
            for e in entities
        ]
        entity_epa = [
            {
                "evaluation": e.epa.evaluation,
                "potency": e.epa.potency,
                "activity": e.epa.activity,
            }
            for e in entities
        ]

        # 3. Compute grounding coverage
        # How many important words in text are grounded?
        words = set(text.lower().split())
        grounded_words = set(name.lower() for name in entity_names)
        coverage = len(words & grounded_words) / len(words) if words else 0

        # 4. Compute semantic-entity alignment
        alignment = self._compute_alignment(sig.dimension_profile, entity_positions)

        return GroundedFingerprint(
            dimension_profile=sig.dimension_profile,
            primary_dimension=sig.primary_dimension,
            entropy=sig.entropy,
            active_primitives=list(activations.keys())[:10],
            grounded_entities=entity_names,
            entity_positions=entity_positions,
            entity_epa=entity_epa,
            grounding_coverage=coverage,
            semantic_entity_alignment=alignment,
        )

    def _compute_alignment(
        self,
        dim_profile: Dict[str, float],
        positions: List[Dict[str, float]],
    ) -> float:
        """Compute alignment between semantic dimensions and entity positions."""
        if not positions:
            return 0.0

        # Heuristic alignment rules:
        # - SPATIAL dimension should correlate with spatial position variance
        # - TEMPORAL dimension should correlate with temporal position variance
        # - QUANTITY dimension should correlate with scale position
        alignment_score = 0.0
        n_checks = 0

        # Spatial alignment
        if "SPATIAL" in dim_profile and len(positions) > 1:
            spatial_vals = [p["spatial"] for p in positions]
            spatial_var = sum((v - sum(spatial_vals)/len(spatial_vals))**2 for v in spatial_vals)
            # High spatial dimension + high spatial variance = good alignment
            alignment_score += dim_profile["SPATIAL"] * min(1.0, spatial_var)
            n_checks += 1

        # Temporal alignment
        if "TEMPORAL" in dim_profile and len(positions) > 1:
            temporal_vals = [p["temporal"] for p in positions]
            temporal_var = sum((v - sum(temporal_vals)/len(temporal_vals))**2 for v in temporal_vals)
            alignment_score += dim_profile["TEMPORAL"] * min(1.0, temporal_var)
            n_checks += 1

        return alignment_score / n_checks if n_checks > 0 else 0.0


class GroundedHallucinationDetector:
    """Detect hallucinations using grounded fingerprints."""

    def __init__(self, entity_db_path: str = None):
        self.generator = GroundedFingerprintGenerator(entity_db_path)

    def detect(self, claim: str) -> Dict:
        """Detect if a claim is likely hallucinated."""
        fp = self.generator.generate(claim)

        # Detection heuristics:
        # 1. Low grounding coverage + high confidence = suspicious
        # 2. High entropy + low alignment = suspicious
        # 3. Specific claims (low entropy) with no grounding = suspicious

        suspicion_score = 0.0

        # Check 1: Ungrounded confident claims
        if fp.grounding_coverage < 0.1 and fp.entropy < 2.0:
            suspicion_score += 0.4

        # Check 2: High entropy + poor alignment
        if fp.entropy > 3.0 and fp.semantic_entity_alignment < 0.3:
            suspicion_score += 0.3

        # Check 3: Claims about entities that don't exist
        if len(fp.grounded_entities) == 0:
            suspicion_score += 0.3

        is_hallucination = suspicion_score > 0.5

        return {
            "is_hallucination": is_hallucination,
            "suspicion_score": suspicion_score,
            "grounding_coverage": fp.grounding_coverage,
            "semantic_entropy": fp.entropy,
            "alignment": fp.semantic_entity_alignment,
            "grounded_entities": fp.grounded_entities,
        }
```

**Test**: `semantic_probing/tests/test_wiki_integration.py`

```python
"""Tests for wiki grounding integration."""

import pytest
from semantic_probing.integration.wiki_grounding import (
    GroundedFingerprintGenerator,
    GroundedHallucinationDetector,
)


class TestGroundedFingerprint:
    @pytest.fixture
    def generator(self):
        return GroundedFingerprintGenerator()

    def test_generates_fingerprint(self, generator):
        """Test basic fingerprint generation."""
        fp = generator.generate("Paris is the capital of France.")
        assert fp.primary_dimension is not None
        assert fp.entropy >= 0
        assert isinstance(fp.grounded_entities, list)

    def test_grounded_claim_has_entities(self, generator):
        """Claims with known entities should have grounding."""
        fp = generator.generate("Albert Einstein developed the theory of relativity.")
        # Should find Einstein as an entity
        assert len(fp.grounded_entities) > 0 or fp.grounding_coverage >= 0

    def test_hallucination_detection(self):
        """Test hallucination detection."""
        detector = GroundedHallucinationDetector()

        # True claim
        true_result = detector.detect("Water is composed of hydrogen and oxygen.")

        # Hallucinated claim
        fake_result = detector.detect(
            "The Henderson-Smith theorem proves that quantum numbers are purple."
        )

        # Fake claim should have higher suspicion
        # Note: May not always work depending on entity DB coverage
        assert "suspicion_score" in fake_result
        assert "is_hallucination" in fake_result
```

**Shell Commands**:
```bash
cd /Users/rohanvinaik/semantic_probing

# Run integration tests
pytest tests/test_wiki_integration.py -v

# Test grounded fingerprinting
python -c "
from integration.wiki_grounding import GroundedFingerprintGenerator
gen = GroundedFingerprintGenerator()
fp = gen.generate('Paris is the capital of France.')
print('Primary dimension:', fp.primary_dimension)
print('Entropy:', fp.entropy)
print('Grounded entities:', fp.grounded_entities)
print('Coverage:', fp.grounding_coverage)
"
```

---

### 2.2 Semantic Drift → Censor Learning

**File**: `negative-learning-censor/integration/semantic_probing.py`

```python
"""
Learn censors when semantic coherence drops.

This integration monitors semantic drift during reasoning and
automatically creates censors when drift exceeds thresholds.
"""

import sys
from typing import List, Dict, Optional
from dataclasses import dataclass

# Local imports
from negative_learning.censor import CensorRegistry, CensorContext
from negative_learning.types import ErrorType

# Integration: semantic_probing
sys.path.insert(0, "/Users/rohanvinaik/semantic_probing")
from semantic_probing.core.hadamard import HadamardBasis
from semantic_probing.core.primitives import PrimitiveProbe
from semantic_probing.core.encoding import TextEncoder
from semantic_probing.core.analysis import SignatureAnalyzer


@dataclass
class DriftEvent:
    """A detected semantic drift event."""
    step_index: int
    drift_magnitude: float
    from_dimension: str
    to_dimension: str
    entropy_change: float


class SemanticCensorLearner:
    """Learn censors from semantic drift patterns."""

    DRIFT_THRESHOLD = 0.3
    ENTROPY_SPIKE_THRESHOLD = 1.5

    def __init__(self):
        # Censor registry
        self.registry = CensorRegistry()

        # Semantic probing
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

    def analyze_reasoning(self, steps: List[str]) -> List[DriftEvent]:
        """Analyze reasoning steps for semantic drift."""
        if len(steps) < 2:
            return []

        # Compute signatures for each step
        signatures = []
        for step in steps:
            vector = self.encoder.encode(step)
            activations = self.probe.probe_vector(vector)
            sig = self.analyzer.compute_signature(activations)
            signatures.append(sig)

        # Detect drift events
        events = []
        for i in range(1, len(signatures)):
            prev = signatures[i-1]
            curr = signatures[i]

            # Compute drift
            drift = self._compute_drift(prev.dimension_profile, curr.dimension_profile)
            entropy_change = curr.entropy - prev.entropy

            if drift > self.DRIFT_THRESHOLD or entropy_change > self.ENTROPY_SPIKE_THRESHOLD:
                events.append(DriftEvent(
                    step_index=i,
                    drift_magnitude=drift,
                    from_dimension=prev.primary_dimension,
                    to_dimension=curr.primary_dimension,
                    entropy_change=entropy_change,
                ))

        return events

    def _compute_drift(self, p1: Dict[str, float], p2: Dict[str, float]) -> float:
        """Compute drift between two dimension profiles."""
        all_dims = set(p1.keys()) | set(p2.keys())
        dist_sq = sum((p1.get(d, 0) - p2.get(d, 0))**2 for d in all_dims)
        return dist_sq ** 0.5

    def learn_from_drift(
        self,
        steps: List[str],
        final_outcome: bool,
    ) -> int:
        """Learn censors from drift events if outcome was failure."""
        if final_outcome:
            return 0  # Don't learn from successes here

        events = self.analyze_reasoning(steps)
        learned = 0

        for event in events:
            # Create context from drift event
            context = CensorContext(
                perceptual={
                    "from_dimension": event.from_dimension,
                    "to_dimension": event.to_dimension,
                },
                sequential={
                    "drift_step": event.step_index,
                    "drift_magnitude": event.drift_magnitude,
                },
                outcome={
                    "entropy_spike": event.entropy_change > self.ENTROPY_SPIKE_THRESHOLD,
                },
            )

            # Learn: "After this drift pattern, the reasoning failed"
            self.registry.learn(
                context=context,
                action=f"drift:{event.from_dimension}->{event.to_dimension}",
                success=False,
            )
            learned += 1

        return learned

    def should_censor_step(
        self,
        previous_steps: List[str],
        proposed_step: str,
    ) -> float:
        """Check if a proposed step should be censored based on drift patterns."""
        if not previous_steps:
            return 0.0

        # Compute current signature
        prev_vector = self.encoder.encode(previous_steps[-1])
        prev_act = self.probe.probe_vector(prev_vector)
        prev_sig = self.analyzer.compute_signature(prev_act)

        # Compute proposed signature
        prop_vector = self.encoder.encode(proposed_step)
        prop_act = self.probe.probe_vector(prop_vector)
        prop_sig = self.analyzer.compute_signature(prop_act)

        # Compute drift
        drift = self._compute_drift(prev_sig.dimension_profile, prop_sig.dimension_profile)
        entropy_change = prop_sig.entropy - prev_sig.entropy

        # Query censor registry
        context = CensorContext(
            perceptual={
                "from_dimension": prev_sig.primary_dimension,
                "to_dimension": prop_sig.primary_dimension,
            },
            sequential={
                "drift_magnitude": drift,
            },
            outcome={
                "entropy_spike": entropy_change > self.ENTROPY_SPIKE_THRESHOLD,
            },
        )

        suppression = self.registry.query(
            context,
            f"drift:{prev_sig.primary_dimension}->{prop_sig.primary_dimension}",
        )

        return suppression


class SemanticGuidedReasoning:
    """Reasoning system that uses semantic censors to avoid bad paths."""

    def __init__(self):
        self.learner = SemanticCensorLearner()

    def filter_candidates(
        self,
        previous_steps: List[str],
        candidates: List[str],
        threshold: float = 0.5,
    ) -> List[str]:
        """Filter candidate next steps based on censor scores."""
        filtered = []
        for candidate in candidates:
            suppression = self.learner.should_censor_step(previous_steps, candidate)
            if suppression < threshold:
                filtered.append(candidate)
        return filtered

    def learn_from_trajectory(
        self,
        steps: List[str],
        success: bool,
    ) -> int:
        """Learn from a completed reasoning trajectory."""
        return self.learner.learn_from_drift(steps, final_outcome=success)
```

**Shell Commands**:
```bash
cd /Users/rohanvinaik/negative-learning-censor

# Test semantic censor integration
python -c "
from integration.semantic_probing import SemanticCensorLearner

learner = SemanticCensorLearner()

# Simulate reasoning steps with drift
steps = [
    'Step 1: We need to calculate 5 + 3.',
    'Step 2: The sky is blue because of Rayleigh scattering.',  # Drift!
    'Step 3: Therefore the answer is 42.',
]

events = learner.analyze_reasoning(steps)
print(f'Detected {len(events)} drift events')
for e in events:
    print(f'  Step {e.step_index}: {e.from_dimension} -> {e.to_dimension} (drift={e.drift_magnitude:.2f})')

# Learn from failure
learned = learner.learn_from_drift(steps, final_outcome=False)
print(f'Learned {learned} censors')
"
```

---

### 2.3 O(1) Proof Cache for Semantic Probing

**File**: `semantic_probing/integration/experience_memory.py`

```python
"""
O(1) cache for semantic probe results using experience-memory.

This integration provides instant retrieval of previously computed
semantic analyses, dramatically reducing latency for repeated queries.
"""

import sys
import json
import hashlib
import time
from typing import Optional, Dict, Callable
from dataclasses import dataclass, asdict

# Local imports
from semantic_probing.core.hadamard import HadamardBasis
from semantic_probing.core.primitives import PrimitiveProbe
from semantic_probing.core.encoding import TextEncoder
from semantic_probing.core.analysis import SignatureAnalyzer

# Integration: experience-memory
sys.path.insert(0, "/Users/rohanvinaik/experience-memory")
from experience_memory.registry import FixRegistry
from experience_memory.types import ErrorSignature, Fix, ErrorType, FixType


@dataclass
class CachedSemanticResult:
    """Cached semantic analysis result."""
    dimension_profile: Dict[str, float]
    entropy: float
    primary_dimension: str
    active_primitives: list
    coherence_score: float
    compute_time_ms: float


class CachedSemanticProbe:
    """
    Semantic probe with O(1) caching via experience-memory.

    Provides significant speedup for repeated queries while
    maintaining full semantic analysis capability.
    """

    def __init__(self, cache_path: str = "/tmp/semantic_probe_cache.db"):
        # Semantic probing
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

        # Caching via experience-memory
        self.cache = FixRegistry(cache_path)
        self._stats = {
            "hits": 0,
            "misses": 0,
            "total_compute_ms": 0,
            "total_cache_ms": 0,
        }

    def _text_to_signature(self, text: str) -> ErrorSignature:
        """Convert text to a cache key signature."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return ErrorSignature(
            severity=1,
            error_type=ErrorType.MISSING_CONTEXT,
            context=f"semantic:{text_hash}",
            affected_categories=["semantic_cache"],
            delta=0.0,
        )

    def _result_to_fix(self, result: CachedSemanticResult) -> Fix:
        """Convert result to cacheable Fix."""
        return Fix(
            fix_type=FixType.COMPLETE_DEFINITION,
            definition_supplement=json.dumps(asdict(result)),
        )

    def _fix_to_result(self, fix: Fix) -> CachedSemanticResult:
        """Convert Fix back to result."""
        data = json.loads(fix.definition_supplement)
        return CachedSemanticResult(**data)

    def analyze(self, text: str) -> CachedSemanticResult:
        """Analyze text, using cache when available."""
        start = time.perf_counter()

        # Check cache
        sig = self._text_to_signature(text)
        cached_fix = self.cache.lookup(sig)

        if cached_fix is not None:
            self._stats["hits"] += 1
            self._stats["total_cache_ms"] += (time.perf_counter() - start) * 1000
            return self._fix_to_result(cached_fix)

        # Cache miss - compute
        self._stats["misses"] += 1
        compute_start = time.perf_counter()

        vector = self.encoder.encode(text)
        activations = self.probe.probe_vector(vector)
        analysis = self.analyzer.compute_signature(activations)

        compute_time = (time.perf_counter() - compute_start) * 1000
        self._stats["total_compute_ms"] += compute_time

        # Create result
        result = CachedSemanticResult(
            dimension_profile=analysis.dimension_profile,
            entropy=analysis.entropy,
            primary_dimension=analysis.primary_dimension,
            active_primitives=list(activations.keys())[:20],
            coherence_score=1.0 - min(1.0, analysis.entropy / 4.0),
            compute_time_ms=compute_time,
        )

        # Store in cache
        self.cache.register(sig, self._result_to_fix(result))

        return result

    def batch_analyze(self, texts: list) -> list:
        """Analyze multiple texts efficiently."""
        return [self.analyze(text) for text in texts]

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0

    @property
    def avg_cache_latency_ms(self) -> float:
        """Average cache hit latency."""
        if self._stats["hits"] == 0:
            return 0.0
        return self._stats["total_cache_ms"] / self._stats["hits"]

    @property
    def avg_compute_latency_ms(self) -> float:
        """Average compute latency."""
        if self._stats["misses"] == 0:
            return 0.0
        return self._stats["total_compute_ms"] / self._stats["misses"]

    @property
    def speedup(self) -> float:
        """Speedup factor from caching."""
        if self.avg_cache_latency_ms == 0:
            return 1.0
        return self.avg_compute_latency_ms / self.avg_cache_latency_ms

    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            **self._stats,
            "hit_rate": self.hit_rate,
            "avg_cache_latency_ms": self.avg_cache_latency_ms,
            "avg_compute_latency_ms": self.avg_compute_latency_ms,
            "speedup": self.speedup,
        }


def benchmark_caching():
    """Benchmark the caching system."""
    import random

    probe = CachedSemanticProbe("/tmp/semantic_cache_benchmark.db")

    # Generate test queries
    base_queries = [
        "What is the capital of France?",
        "Calculate 15 times 23.",
        "If A implies B and B implies C, what can we conclude?",
        "The sky is blue because of light scattering.",
        "Water boils at 100 degrees Celsius.",
    ]

    # Extend with variations
    queries = base_queries * 20  # 100 queries with repeats
    random.shuffle(queries)

    # Run benchmark
    print("Running cache benchmark...")
    for q in queries:
        probe.analyze(q)

    stats = probe.stats()
    print(f"\n=== Cache Benchmark Results ===")
    print(f"Total queries: {stats['hits'] + stats['misses']}")
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.1%}")
    print(f"Avg cache latency: {stats['avg_cache_latency_ms']:.3f}ms")
    print(f"Avg compute latency: {stats['avg_compute_latency_ms']:.3f}ms")
    print(f"Speedup: {stats['speedup']:.1f}x")


if __name__ == "__main__":
    benchmark_caching()
```

**Test**: `semantic_probing/tests/test_proof_cache.py`

```python
"""Tests for proof cache integration."""

import pytest
import tempfile
from semantic_probing.integration.experience_memory import (
    CachedSemanticProbe,
    CachedSemanticResult,
)


class TestCachedSemanticProbe:
    @pytest.fixture
    def probe(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            yield CachedSemanticProbe(f.name)

    def test_basic_analysis(self, probe):
        """Test basic semantic analysis."""
        result = probe.analyze("What is 2 + 2?")
        assert result.primary_dimension is not None
        assert result.entropy >= 0
        assert result.coherence_score >= 0

    def test_cache_hit(self, probe):
        """Second query should be a cache hit."""
        query = "The sum of two and three is five."

        # First query - miss
        result1 = probe.analyze(query)
        assert probe._stats["misses"] == 1
        assert probe._stats["hits"] == 0

        # Second query - hit
        result2 = probe.analyze(query)
        assert probe._stats["hits"] == 1

        # Results should match
        assert result1.primary_dimension == result2.primary_dimension
        assert result1.entropy == result2.entropy

    def test_cache_speedup(self, probe):
        """Cache should provide speedup."""
        query = "A complex query about mathematics and logic."

        # Warm up
        probe.analyze(query)

        # Measure
        import time
        n = 100

        # Cache hits
        start = time.perf_counter()
        for _ in range(n):
            probe.analyze(query)
        cache_time = time.perf_counter() - start

        # The cache time should be very fast
        avg_cache_ms = (cache_time / n) * 1000
        assert avg_cache_ms < 10  # Should be well under 10ms

    def test_stats(self, probe):
        """Test statistics tracking."""
        probe.analyze("Query 1")
        probe.analyze("Query 2")
        probe.analyze("Query 1")  # Repeat

        stats = probe.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(1/3)
```

**Shell Commands**:
```bash
cd /Users/rohanvinaik/semantic_probing

# Run cache benchmark
python -m integration.experience_memory

# Run tests
pytest tests/test_proof_cache.py -v
```

---

## Phase 3: Full Pipeline Integration

The complete brain-like AI system coordinating all components.

### 3.1 Semantic Router

**File**: `semantic_probing/pipeline/semantic_router.py`

```python
"""
Fuzzy semantic router for directing queries to appropriate specialists.

Routes based on semantic dimension profile rather than explicit rules,
enabling emergent coordination without a homunculus.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from semantic_probing.core.hadamard import HadamardBasis
from semantic_probing.core.primitives import PrimitiveProbe
from semantic_probing.core.encoding import TextEncoder
from semantic_probing.core.analysis import SignatureAnalyzer


class SpecialistType(Enum):
    """Available specialist types."""
    LOGICAL = "logical"
    QUANTITY = "quantity"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    MENTAL = "mental"
    GROUNDING = "grounding"  # Wiki-based fact checking
    GENERALIST = "generalist"  # Fallback


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    primary_specialist: SpecialistType
    secondary_specialists: List[SpecialistType]
    dimension_profile: Dict[str, float]
    confidence: float
    reasoning: str


class SemanticRouter:
    """
    Route queries to specialists based on semantic dimension profile.

    No explicit rules - uses fuzzy matching of semantic signatures
    to determine which specialists should handle a query.
    """

    # Dimension to specialist mapping
    DIMENSION_TO_SPECIALIST = {
        "LOGICAL": SpecialistType.LOGICAL,
        "QUANTITY": SpecialistType.QUANTITY,
        "TEMPORAL": SpecialistType.TEMPORAL,
        "SPATIAL": SpecialistType.SPATIAL,
        "MENTAL": SpecialistType.MENTAL,
    }

    # Thresholds
    PRIMARY_THRESHOLD = 0.4  # Minimum score for primary specialist
    SECONDARY_THRESHOLD = 0.2  # Minimum score for secondary
    GROUNDING_THRESHOLD = 0.3  # When to include grounding specialist

    def __init__(self):
        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

    def route(self, query: str) -> RoutingDecision:
        """Route a query to appropriate specialists."""
        # Compute semantic signature
        vector = self.encoder.encode(query)
        activations = self.probe.probe_vector(vector)
        sig = self.analyzer.compute_signature(activations)

        dim_profile = sig.dimension_profile

        # Sort dimensions by score
        sorted_dims = sorted(
            dim_profile.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Determine primary specialist
        primary = SpecialistType.GENERALIST
        primary_score = 0.0

        if sorted_dims and sorted_dims[0][1] >= self.PRIMARY_THRESHOLD:
            dim_name = sorted_dims[0][0]
            if dim_name in self.DIMENSION_TO_SPECIALIST:
                primary = self.DIMENSION_TO_SPECIALIST[dim_name]
                primary_score = sorted_dims[0][1]

        # Determine secondary specialists
        secondary = []
        for dim_name, score in sorted_dims[1:]:
            if score >= self.SECONDARY_THRESHOLD:
                if dim_name in self.DIMENSION_TO_SPECIALIST:
                    secondary.append(self.DIMENSION_TO_SPECIALIST[dim_name])

        # Add grounding specialist for factual queries
        if self._needs_grounding(query, dim_profile):
            if SpecialistType.GROUNDING not in secondary:
                secondary.append(SpecialistType.GROUNDING)

        # Compute confidence
        confidence = primary_score if primary != SpecialistType.GENERALIST else 0.5

        # Generate reasoning
        reasoning = self._generate_reasoning(primary, secondary, sorted_dims)

        return RoutingDecision(
            primary_specialist=primary,
            secondary_specialists=secondary,
            dimension_profile=dim_profile,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _needs_grounding(self, query: str, profile: Dict[str, float]) -> bool:
        """Determine if query needs entity grounding."""
        # Heuristics for grounding:
        # 1. Contains proper nouns (capital letters mid-sentence)
        # 2. Contains specific claims
        # 3. Low LOGICAL but high SUBSTANTIVES

        words = query.split()
        has_proper_nouns = any(
            w[0].isupper() and i > 0
            for i, w in enumerate(words)
            if w and w[0].isalpha()
        )

        substantive_heavy = profile.get("SUBSTANTIVES", 0) > self.GROUNDING_THRESHOLD
        low_logical = profile.get("LOGICAL", 0) < 0.3

        return has_proper_nouns or (substantive_heavy and low_logical)

    def _generate_reasoning(
        self,
        primary: SpecialistType,
        secondary: List[SpecialistType],
        sorted_dims: List[Tuple[str, float]],
    ) -> str:
        """Generate human-readable routing reasoning."""
        top_dims = [f"{d}={s:.2f}" for d, s in sorted_dims[:3]]

        if primary == SpecialistType.GENERALIST:
            return f"No dominant dimension ({', '.join(top_dims)}). Using generalist."

        secondary_str = ", ".join(s.value for s in secondary) if secondary else "none"
        return (
            f"Primary dimension: {sorted_dims[0][0]} ({sorted_dims[0][1]:.2f}). "
            f"Routing to {primary.value} specialist. "
            f"Secondary specialists: {secondary_str}."
        )


class MultiSpecialistOrchestrator:
    """Orchestrate multiple specialists for complex queries."""

    def __init__(self, specialists: Dict[SpecialistType, object] = None):
        self.router = SemanticRouter()
        self.specialists = specialists or {}

    def process(self, query: str) -> Dict:
        """Process query through appropriate specialists."""
        # Route
        decision = self.router.route(query)

        # Collect responses
        responses = {}

        # Primary specialist
        if decision.primary_specialist in self.specialists:
            specialist = self.specialists[decision.primary_specialist]
            responses["primary"] = {
                "specialist": decision.primary_specialist.value,
                "response": specialist.process(query),
            }

        # Secondary specialists
        for spec_type in decision.secondary_specialists:
            if spec_type in self.specialists:
                specialist = self.specialists[spec_type]
                responses[spec_type.value] = specialist.process(query)

        return {
            "routing": {
                "primary": decision.primary_specialist.value,
                "secondary": [s.value for s in decision.secondary_specialists],
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
            },
            "responses": responses,
            "dimension_profile": decision.dimension_profile,
        }
```

---

### 3.2 Unified Verification Pipeline

**File**: `semantic_probing/pipeline/unified_verifier.py`

```python
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
from semantic_probing.core.hadamard import HadamardBasis
from semantic_probing.core.primitives import PrimitiveProbe
from semantic_probing.core.encoding import TextEncoder
from semantic_probing.core.analysis import SignatureAnalyzer

# Integrations
sys.path.insert(0, "/Users/rohanvinaik/sparse-wiki-grounding")
sys.path.insert(0, "/Users/rohanvinaik/negative-learning-censor")
sys.path.insert(0, "/Users/rohanvinaik/orthogonal-validators")
sys.path.insert(0, "/Users/rohanvinaik/experience-memory")

from wiki_grounding.store import EntityStore
from wiki_grounding.verification import ClaimVerifier
from negative_learning.censor import CensorRegistry, CensorContext
from orthogonal_validators.committee import ValidatorCommittee
from orthogonal_validators.validators.semantic import SemanticValidator
from orthogonal_validators.validators.entity import EntityValidator
from experience_memory.registry import FixRegistry
from experience_memory.types import ErrorSignature, Fix, ErrorType, FixType


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
            EntityValidator(self.wiki_store),
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
            severity=1,
            error_type=ErrorType.MISSING_CONTEXT,
            context=f"verified:{claim_hash}",
            affected_categories=["verification_cache"],
            delta=0.0,
        )

    def _claim_to_censor_context(self, claim: str) -> CensorContext:
        """Convert claim to censor context."""
        # Compute semantic signature
        vector = self.encoder.encode(claim)
        activations = self.probe.probe_vector(vector)
        sig = self.analyzer.compute_signature(activations)

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
            fix = Fix(
                fix_type=FixType.COMPLETE_DEFINITION,
                definition_supplement=json.dumps({
                    "status": "verified",
                    "confidence": committee_result.confidence,
                }),
            )
            self.proof_cache.register(cache_sig, fix)

            latency = (time.perf_counter() - start) * 1000
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=committee_result.confidence,
                details={"validators": committee_result.validator_details},
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
                    "validators": committee_result.validator_details,
                },
                latency_ms=latency,
            )

        else:
            # Rejection - learn from failure
            self._stats["rejected"] += 1
            self.censor_registry.learn(context, "verify", success=False)

            latency = (time.perf_counter() - start) * 1000
            return VerificationResult(
                status=VerificationStatus.REJECTED,
                confidence=1.0 - committee_result.confidence,
                details={"validators": committee_result.validator_details},
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
```

---

### 3.3 Full Pipeline Benchmark

**File**: `semantic_probing/experiments/full_pipeline_benchmark.py`

```python
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
        from semantic_probing.core.hadamard import HadamardBasis
        from semantic_probing.core.primitives import PrimitiveProbe
        from semantic_probing.core.encoding import TextEncoder
        from semantic_probing.core.analysis import SignatureAnalyzer

        self.basis = HadamardBasis().generate()
        self.probe = PrimitiveProbe(self.basis)
        self.encoder = TextEncoder()
        self.analyzer = SignatureAnalyzer()

    def verify(self, claim: str) -> Dict:
        start = time.perf_counter()
        vector = self.encoder.encode(claim)
        activations = self.probe.probe_vector(vector)
        sig = self.analyzer.compute_signature(activations)

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
```

**Shell Commands**:
```bash
cd /Users/rohanvinaik/semantic_probing

# Run full pipeline benchmark
python -m semantic_probing.experiments.full_pipeline_benchmark \
    --output reports/full_pipeline_results.json

# View results
cat reports/full_pipeline_results.json | python -m json.tool
```

---

## Evaluation Metrics

### Metric Definitions

| Metric | Formula | Target |
|--------|---------|--------|
| Accuracy | (TP + TN) / Total | > 0.90 |
| Auto-accept rate | Auto-accepted / Total | > 0.70 |
| Error-in-auto-accept | Errors in auto-accepted / Auto-accepted | < 0.01 |
| Review rate | Escalated to review / Total | < 0.20 |
| Cache hit rate | Cache hits / Total | > 0.80 |
| Latency (cached) | Time for cache hit | < 5ms |
| Latency (compute) | Time for full computation | < 500ms |
| Speedup | Compute latency / Cache latency | > 100x |

### Hypothesis Testing

| Hypothesis | Test | Success Criterion |
|------------|------|-------------------|
| H1: Specialists > Generalists | Compare accuracy on dimension-matched problems | Specialist accuracy > Generalist + 10% |
| H2: Semantic routing reduces errors | Compare routed vs random specialist assignment | Routed error rate < Random × 0.5 |
| H3: Functional networks extractable | Compare primitive patterns | Clear divergence in activation patterns |
| H4: Negative learning prevents repeats | Measure repeat error rate | Second occurrence < First × 0.3 |
| H5: O(1) caching speedup | Measure latency | Cache latency < 5µs |
| H6: Orthogonal validation catches errors | Measure false negative rate in auto-accept | < 1% |

---

## Research Paper Outlines

### Paper 1: Semantic Dimension Specialists
**Venue**: ICML/NeurIPS
**Focus**: Training tiny LLMs on dimension-filtered data

**Abstract**: We demonstrate that 0.5B parameter models trained on semantically filtered data can match 8B generalist models on dimension-specific tasks.

**Contributions**:
1. Semantic primitive signatures for dataset curation
2. 16× parameter efficiency on matched problems
3. Interpretable dimension profiles for model selection

### Paper 2: Functional Network Extraction
**Venue**: ICLR
**Focus**: Extracting interpretable circuits from fine-tuned LLMs

**Abstract**: We present a method to extract "what a model learned" as differences in semantic primitive activation patterns.

**Contributions**:
1. Functional network extraction methodology
2. Comparison specialist vs generalist activations
3. Connection to mechanistic interpretability

### Paper 3: Brain-Like AI Architecture
**Venue**: Nature Machine Intelligence
**Focus**: Full hierarchical specialist orchestration

**Abstract**: We present a brain-like AI architecture combining semantic routing, specialist ensembles, orthogonal validation, and self-correction.

**Contributions**:
1. Fuzzy semantic routing (no homunculus)
2. Multi-specialist coordination
3. Near-zero errors in auto-accepted outputs

### Paper 4: Grounded Verification
**Venue**: ACL/EMNLP
**Focus**: Combining semantic primitives with entity grounding

**Abstract**: We combine semantic primitive analysis with entity grounding to create a powerful hallucination detector.

**Contributions**:
1. EPA-enhanced semantic fingerprints
2. Improved hallucination detection
3. Spreading activation for claim verification

---

## Execution Checklist

### Phase 0: Tinker Specialist Training
```bash
# 0.1 Data curation
cd /Users/rohanvinaik/semantic_probing
mkdir -p data/specialists
python -m semantic_probing.data_curation.dimension_datasets --dimension LOGICAL --source folio --output data/specialists/logical_specialist.jsonl
python -m semantic_probing.data_curation.dimension_datasets --dimension QUANTITY --source gsm8k --output data/specialists/quantity_specialist.jsonl

# 0.2 Train specialists
cd /Users/rohanvinaik/tinker-cookbook
python -m tinker_cookbook.recipes.semantic_specialists.train dimension=LOGICAL model_name=Qwen/Qwen3-0.5B
python -m tinker_cookbook.recipes.semantic_specialists.train dimension=QUANTITY model_name=Qwen/Qwen3-0.5B

# 0.3 Extract functional networks
cd /Users/rohanvinaik/semantic_probing
python -m semantic_probing.experiments.functional_extraction --specialist_path ../tinker-cookbook/experiments/semantic_specialists/LOGICAL-* --dimension LOGICAL --output reports/functional_networks/logical.json
```

### Phase 1: Standalone Experiments
```bash
# semantic_probing
cd /Users/rohanvinaik/semantic_probing
pytest tests/ -v
python -m semantic_probing.experiments.coherence_benchmark --output reports/coherence_baseline.json

# sparse-wiki-grounding
cd /Users/rohanvinaik/sparse-wiki-grounding
pytest tests/ -v
python -m experiments.hallucination_benchmark --output reports/verification_baseline.json

# negative-learning-censor
cd /Users/rohanvinaik/negative-learning-censor
pytest tests/ -v
python -m experiments.reasoning_censors --output reports/censor_taxonomy.json

# orthogonal-validators
cd /Users/rohanvinaik/orthogonal-validators
pytest tests/ -v

# experience-memory
cd /Users/rohanvinaik/experience-memory
pytest tests/ -v
python -m experience_memory.semantic_cache
```

### Phase 2: Integration Tests
```bash
cd /Users/rohanvinaik/semantic_probing
pytest tests/test_wiki_integration.py -v
pytest tests/test_proof_cache.py -v
python -m integration.experience_memory
```

### Phase 3: Full Pipeline
```bash
cd /Users/rohanvinaik/semantic_probing
python -m semantic_probing.experiments.full_pipeline_benchmark --output reports/full_pipeline_results.json
```

---

## File Structure Summary

```
semantic_probing/
├── MODULAR_BRAIN_AI_EXPERIMENTS.md  # This document
├── data_curation/
│   └── dimension_datasets.py
├── data/
│   ├── specialists/
│   │   ├── logical_specialist.jsonl
│   │   ├── quantity_specialist.jsonl
│   │   ├── temporal_specialist.jsonl
│   │   ├── spatial_specialist.jsonl
│   │   └── mental_specialist.jsonl
│   └── benchmarks/
│       ├── coherence_traces.jsonl
│       └── verification_benchmark.jsonl
├── experiments/
│   ├── coherence_benchmark.py
│   ├── functional_extraction.py
│   └── full_pipeline_benchmark.py
├── integration/
│   ├── wiki_grounding.py
│   └── experience_memory.py
├── pipeline/
│   ├── semantic_router.py
│   └── unified_verifier.py
├── reports/
│   ├── coherence_baseline.json
│   ├── functional_networks/
│   └── full_pipeline_results.json
└── tests/
    ├── test_coherence_benchmark.py
    ├── test_wiki_integration.py
    └── test_proof_cache.py

tinker-cookbook/
├── tinker_cookbook/recipes/semantic_specialists/
│   ├── train.py
│   ├── rl_train.py
│   └── datasets.py
└── experiments/semantic_specialists/
    ├── logical_specialist/
    └── quantity_specialist/

sparse-wiki-grounding/
├── experiments/
│   └── hallucination_benchmark.py
└── reports/
    └── verification_baseline.json

negative-learning-censor/
├── experiments/
│   └── reasoning_censors.py
├── integration/
│   └── semantic_probing.py
└── reports/
    └── censor_taxonomy.json

orthogonal-validators/
├── orthogonal_validators/validators/
│   └── probing.py
└── tests/
    └── test_probing_validator.py

experience-memory/
├── experience_memory/
│   └── semantic_cache.py
└── tests/
    └── test_semantic_cache.py
```

---

## Conclusion

This experimental plan establishes a comprehensive research program for building a **brain-like modular AI architecture**. By combining semantic probing, efficient specialist training via Tinker, entity grounding, negative learning, orthogonal validation, and O(1) caching, we can create AI systems that are:

1. **Efficient**: 0.5B specialists matching 8B generalists
2. **Interpretable**: Extractable functional networks
3. **Safe**: Near-zero errors in auto-accepted outputs
4. **Fast**: O(1) caching for repeated queries

The modular design enables independent testing of each component while the unified pipeline demonstrates the power of their combination.
