"""Entity grounding module."""

from __future__ import annotations
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union, Any
from pathlib import Path
import sqlite3

from ..encoding import SemanticDimension

# =============================================================================
# DIMENSION ENUMS
# =============================================================================

class GroundingDimension(str, Enum):
    """
    The 5 hierarchical dimension trees for entity grounding.
    """
    SPATIAL = "SPATIAL"       # Earth -> continents -> countries -> regions -> cities
    TEMPORAL = "TEMPORAL"     # Present -> centuries -> decades -> years
    TAXONOMIC = "TAXONOMIC"   # Thing -> Person/Place/Event/... -> subtypes
    SCALE = "SCALE"           # Regional -> Local/National/Global
    DOMAIN = "DOMAIN"         # Knowledge -> fields -> subfields -> topics


class TernaryValue(IntEnum):
    """Balanced ternary for EPA values."""
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1


# =============================================================================
# CORE ENTITY MODELS
# =============================================================================

@dataclass(frozen=True)
class Entity:
    """Core entity from sparse wiki."""
    id: str
    wikipedia_title: str
    label: str
    description: Optional[str] = None
    vital_level: Optional[int] = None
    pagerank: Optional[float] = None


@dataclass(frozen=True)
class DimensionPosition:
    """Position in one dimension tree."""
    dimension: GroundingDimension
    path_sign: int
    path_depth: int
    path_nodes: List[str]
    zero_state: str
    merkle_hash: Optional[str] = None

    @property
    def formatted_path(self) -> str:
        """Return formatted notation."""
        sign_str = "+" if self.path_sign > 0 else ("-" if self.path_sign < 0 else "")
        return f"{sign_str}{self.path_depth}:{self.dimension.value}/{'/'.join(self.path_nodes)}"


@dataclass(frozen=True)
class EPAValues:
    """Evaluation, Potency, Activity values."""
    evaluation: TernaryValue = TernaryValue.NEUTRAL
    potency: TernaryValue = TernaryValue.NEUTRAL
    activity: TernaryValue = TernaryValue.NEUTRAL
    confidence: float = 1.0


@dataclass
class EntityProfile:
    """Complete entity profile."""
    entity: Entity
    positions: List[DimensionPosition] = field(default_factory=list)
    epa: EPAValues = field(default_factory=lambda: EPAValues())
    properties: Dict[str, Union[str, List[str]]] = field(default_factory=dict)

    def get_positions(self, dimension: GroundingDimension) -> List[DimensionPosition]:
        """Get all positions for a dimension."""
        return [pos for pos in self.positions if pos.dimension == dimension]


@dataclass
class GroundingCandidate:
    """One possible grounding for an ambiguous mention."""
    entity: Entity
    positions: List[DimensionPosition]
    epa: EPAValues
    score: float


@dataclass
class GroundingResult:
    """Result of grounding a text mention."""
    mention: str
    candidates: List[GroundingCandidate] = field(default_factory=list)
    best_match: Optional[GroundingCandidate] = None


# =============================================================================
# GROUNDED ENTITY MODELS (Enhanced)
# =============================================================================

# Mapping from Grounding Dimensions to Semantic Dimensions
DIMENSION_MAPPING = {
    "SPATIAL": SemanticDimension.SPATIAL,
    "TEMPORAL": SemanticDimension.TEMPORAL,
    "TAXONOMIC": SemanticDimension.SUBSTANTIVES,
    "SCALE": SemanticDimension.QUANTITY,
    "DOMAIN": SemanticDimension.MENTAL,
}

# Mapping from semantic anchor categories to Semantic Dimensions
ANCHOR_MAPPING = {
    "TYPE": SemanticDimension.SUBSTANTIVES,
    "GEOGRAPHY": SemanticDimension.SPATIAL,
    "HISTORY": SemanticDimension.TEMPORAL,
    "KNOWN_FOR": SemanticDimension.MENTAL,
    "SCOPE": SemanticDimension.QUANTITY,
}

@dataclass
class GroundedEntity:
    """A grounded entity with semantic contributions."""
    term: str
    lemma: str
    entity: Entity
    positions: List[DimensionPosition]
    epa: EPAValues
    confidence: float
    disambiguation_needed: bool
    semantic_anchors: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def dimension_contributions(self) -> Dict[SemanticDimension, float]:
        """Compute contributions to semantic dimensions."""
        contributions = {}

        # From dimension positions
        for pos in self.positions:
            dim_name = pos.dimension.value
            sem_dim = DIMENSION_MAPPING.get(dim_name)
            if sem_dim is not None:
                # Weight by depth
                weight = min(1.0, pos.path_depth / 4.0)
                contributions[sem_dim] = max(contributions.get(sem_dim, 0.0), weight)

        # From semantic anchors
        for category, anchors in self.semantic_anchors.items():
            sem_dim = ANCHOR_MAPPING.get(category)
            if sem_dim is not None and anchors:
                anchor_weight = min(1.0, len(anchors) / 5.0) * 0.5
                contributions[sem_dim] = max(contributions.get(sem_dim, 0.0), anchor_weight)

        return contributions


@dataclass
class EntityEnhancement:
    """Enhancement data from entity grounding."""
    grounded_entities: List[GroundedEntity] = field(default_factory=list)
    ungrounded_terms: List[str] = field(default_factory=list)
    dimension_boosts: Dict[SemanticDimension, float] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)
    attributes: Dict[str, float] = field(default_factory=dict)
    confidence_boost: float = 0.0
    context_used: Optional[str] = None


# =============================================================================
# ENTITY GROUNDER
# =============================================================================

class EntityGrounder:
    """Entity grounding using sparse world knowledge."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def ground(
        self,
        mention: str,
        context: Optional[str] = None,
        max_candidates: int = 10,
    ) -> GroundingResult:
        """Ground a mention to entity candidates."""
        # Finds entities by label
        cursor = self.conn.execute("SELECT * FROM entities WHERE label LIKE ? LIMIT ?", (mention, max_candidates * 2))
        rows = cursor.fetchall()
        
        candidates = []
        for row in rows:
            entity = Entity(
                id=row["id"],
                wikipedia_title=row["wikipedia_title"],
                label=row["label"],
                description=row["description"],
                vital_level=row["vital_level"],
                pagerank=row["pagerank"],
            )
            
            # Get positions
            positions = self._get_positions(entity.id)
            # Get EPA
            epa = self._get_epa(entity.id)
            
            # Score
            score = 0.5 # Default
            if context:
                score += self._compute_context_score(positions, context)
                
            candidates.append(GroundingCandidate(entity, positions, epa, score))
            
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:max_candidates]
        
        best = candidates[0] if candidates and (len(candidates) == 1 or candidates[0].score > candidates[1].score + 0.2) else None
        
        return GroundingResult(mention, candidates, best)

    def _get_positions(self, entity_id: str) -> List[DimensionPosition]:
        """Get dimension positions from DB."""
        # Assuming a schema where positions are stored or linked
        # The schema in 'prepare_entities.py' implies 'dimension_positions' table exists.
        # columns: entity_id, dimension, path_sign, path_depth, path_nodes_json, zero_state ...
        try:
            cursor = self.conn.execute("SELECT * FROM dimension_positions WHERE entity_id = ?", (entity_id,))
            positions = []
            import json
            for row in cursor.fetchall():
                positions.append(DimensionPosition(
                    dimension=GroundingDimension(row["dimension"]),
                    path_sign=row["path_sign"],
                    path_depth=row["path_depth"],
                    path_nodes=json.loads(row["path_nodes"]),
                    zero_state=row["zero_state"],
                ))
            return positions
        except Exception:
            return []

    def _get_epa(self, entity_id: str) -> EPAValues:
        """Get EPA values."""
        try:
            row = self.conn.execute("SELECT * FROM epa_values WHERE entity_id = ?", (entity_id,)).fetchone()
            if row:
                return EPAValues(
                    evaluation=TernaryValue(row["evaluation"]),
                    potency=TernaryValue(row["potency"]),
                    activity=TernaryValue(row["activity"]),
                    confidence=row["confidence"],
                )
        except Exception:
            pass
        return EPAValues()

    def _compute_context_score(self, positions: List[DimensionPosition], context: str) -> float:
        """Score based on context overlap."""
        score = 0.0
        context_words = set(context.lower().split())
        for pos in positions:
            for node in pos.path_nodes:
                if node.lower() in context_words:
                    score += 1.0
        return score
