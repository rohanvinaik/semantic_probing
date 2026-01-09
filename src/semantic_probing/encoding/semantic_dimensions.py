"""Semantic dimensions and primitive registry."""

from dataclasses import dataclass
from enum import IntEnum, Enum
from typing import Dict, List, Optional, Set, Tuple

# =============================================================================
# CORE CONSTANTS
# =============================================================================

D = 8192  # Total dimension (2^13, optimal for Hadamard)
N_DIMENSIONS = 8  # Number of semantic dimensions (formerly banks)
DIMS_PER_DIMENSION = D // N_DIMENSIONS  # 1024 dims per dimension


# =============================================================================
# SEMANTIC DIMENSIONS
# =============================================================================

class SemanticDimension(IntEnum):
    """The 8 semantic dimensions, each with 1024 dimensions."""
    SUBSTANTIVES = 0   # Entities and reference
    QUANTITY = 1       # Amount and cardinality (with hierarchy)
    EVALUATORS = 2     # Value and magnitude judgments (bipolar)
    MENTAL = 3         # Cognitive and perceptual states
    ACTION = 4         # Events, states, and existence
    TEMPORAL = 5       # Time relations and duration (bipolar)
    SPATIAL = 6        # Location and spatial relations (bipolar)
    LOGICAL = 7        # Modality, causation, hierarchy markers


DIMENSION_NAMES = {
    0: "SUBSTANTIVES",
    1: "QUANTITY",
    2: "EVALUATORS",
    3: "MENTAL",
    4: "ACTION",
    5: "TEMPORAL",
    6: "SPATIAL",
    7: "LOGICAL",
}


# =============================================================================
# BIPOLAR (ANTONYM) PAIRS
# =============================================================================

@dataclass(frozen=True)
class BipolarPair:
    """
    A bipolar pair where antonyms occupy the same space with opposite signs.

    Example: EVALUATION dimension
        - GOOD encodes as +1 in these dimensions
        - BAD encodes as -1 in these dimensions
        - cos(GOOD, BAD) = -1.0 (perfect opposition)
    """
    name: str           # Dimension name (e.g., "EVALUATION")
    positive: str       # Positive pole label (e.g., "GOOD")
    negative: str       # Negative pole label (e.g., "BAD")
    dimension: SemanticDimension  # Which dimension this belongs to

    def get_polarity(self, primitive: str) -> int:
        """Get polarity (+1, -1, or 0 if not this dimension)."""
        if primitive == self.positive:
            return 1
        elif primitive == self.negative:
            return -1
        return 0


# The 14 true antonym pairs collapsed as bipolar dimensions
BIPOLAR_PAIRS: List[BipolarPair] = [
    # Dimension 0: SUBSTANTIVES
    BipolarPair("IDENTITY", "SAME", "OTHER", SemanticDimension.SUBSTANTIVES),

    # Dimension 2: EVALUATORS
    BipolarPair("EVALUATION", "GOOD", "BAD", SemanticDimension.EVALUATORS),
    BipolarPair("SIZE", "BIG", "SMALL", SemanticDimension.EVALUATORS),

    # Dimension 4: ACTION
    BipolarPair("VITALITY", "ALIVE", "DEAD", SemanticDimension.ACTION),

    # Dimension 5: TEMPORAL
    BipolarPair("DURATION", "LONG_TIME", "SHORT_TIME", SemanticDimension.TEMPORAL),
    BipolarPair("TEMPORAL_ORDER", "BEFORE", "AFTER", SemanticDimension.TEMPORAL),
    BipolarPair("TEMPORAL_REF", "THEN", "NOW", SemanticDimension.TEMPORAL),
    BipolarPair("TEMPORAL_BOUND", "START", "END", SemanticDimension.TEMPORAL),

    # Dimension 6: SPATIAL
    BipolarPair("VERTICAL", "ABOVE", "BELOW", SemanticDimension.SPATIAL),
    BipolarPair("DISTANCE", "FAR", "NEAR", SemanticDimension.SPATIAL),
    BipolarPair("CONTAINMENT", "OUTSIDE", "INSIDE", SemanticDimension.SPATIAL),
    BipolarPair("DIRECTION", "TOWARD", "AWAY", SemanticDimension.SPATIAL),
    BipolarPair("ACCOMPANIMENT", "WITH", "WITHOUT", SemanticDimension.SPATIAL),

    # Dimension 7: LOGICAL
    BipolarPair("TRUTH", "TRUE", "FALSE", SemanticDimension.LOGICAL),
]

# Quick lookup: primitive name -> BipolarPair
BIPOLAR_LOOKUP: Dict[str, BipolarPair] = {}
for pair in BIPOLAR_PAIRS:
    BIPOLAR_LOOKUP[pair.positive] = pair
    BIPOLAR_LOOKUP[pair.negative] = pair

# Set of all bipolar primitive names (both poles)
BIPOLAR_PRIMITIVES: Set[str] = set()
for pair in BIPOLAR_PAIRS:
    BIPOLAR_PRIMITIVES.add(pair.positive)
    BIPOLAR_PRIMITIVES.add(pair.negative)


def is_bipolar(primitive: str) -> bool:
    """Check if a primitive is part of a bipolar pair."""
    return primitive in BIPOLAR_PRIMITIVES


def get_bipolar_pair(primitive: str) -> Optional[BipolarPair]:
    """Get the bipolar pair for a primitive, if any."""
    return BIPOLAR_LOOKUP.get(primitive)


def get_canonical_pole(primitive: str) -> Tuple[str, int]:
    """
    Get canonical (positive) form and polarity for a primitive.

    Returns:
        (canonical_name, polarity) where polarity is +1 or -1
        For non-bipolar primitives, returns (primitive, +1)
    """
    pair = get_bipolar_pair(primitive)
    if pair is None:
        return (primitive, 1)
    return (pair.positive, pair.get_polarity(primitive))


# =============================================================================
# HIERARCHICAL MODIFIERS
# =============================================================================

@dataclass(frozen=True)
class HierarchyLevel:
    """A level in an ordinal hierarchy."""
    name: str       # English word (e.g., "MANY")
    level: int      # Ordinal level (-3 to +3)
    markers: int    # Number of HIER_PLUS or HIER_MINUS markers


class HierarchyType(Enum):
    """Types of hierarchical encoding."""
    QUANTITY = "quantity"     # Amount/cardinality
    INTENSITY = "intensity"   # Degree/strength
    CERTAINTY = "certainty"   # Epistemic modality


# Quantity Hierarchy (7 levels, middle-anchored)
QUANTITY_HIERARCHY: Dict[str, HierarchyLevel] = {
    "NONE":   HierarchyLevel("NONE", -3, 3),    # [−][−][−][QUANTITY]
    "FEW":    HierarchyLevel("FEW", -2, 2),     # [−][−][QUANTITY]
    "SOME":   HierarchyLevel("SOME", -1, 1),    # [−][QUANTITY]
    "MIDDLE": HierarchyLevel("MIDDLE", 0, 0),   # [QUANTITY]
    "MORE":   HierarchyLevel("MORE", 1, 1),     # [+][QUANTITY]
    "MANY":   HierarchyLevel("MANY", 2, 2),     # [+][+][QUANTITY]
    "ALL":    HierarchyLevel("ALL", 3, 3),      # [+][+][+][QUANTITY]
}

# Intensity Hierarchy (5 levels)
INTENSITY_HIERARCHY: Dict[str, HierarchyLevel] = {
    "BARELY":    HierarchyLevel("BARELY", -2, 2),    # [−][−][INTENSITY]
    "SLIGHTLY":  HierarchyLevel("SLIGHTLY", -1, 1),  # [−][INTENSITY]
    "MODERATE":  HierarchyLevel("MODERATE", 0, 0),   # [INTENSITY]
    "VERY":      HierarchyLevel("VERY", 1, 1),       # [+][INTENSITY]
    "EXTREMELY": HierarchyLevel("EXTREMELY", 2, 2),  # [+][+][INTENSITY]
}

# Certainty Hierarchy (7 levels, epistemic modality)
CERTAINTY_HIERARCHY: Dict[str, HierarchyLevel] = {
    "IMPOSSIBLE": HierarchyLevel("IMPOSSIBLE", -3, 3),  # [−][−][−][CERTAINTY]
    "UNLIKELY":   HierarchyLevel("UNLIKELY", -2, 2),    # [−][−][CERTAINTY]
    "DOUBTFUL":   HierarchyLevel("DOUBTFUL", -1, 1),    # [−][CERTAINTY]
    "POSSIBLE":   HierarchyLevel("POSSIBLE", 0, 0),     # [CERTAINTY] (= MAYBE)
    "PROBABLE":   HierarchyLevel("PROBABLE", 1, 1),     # [+][CERTAINTY]
    "LIKELY":     HierarchyLevel("LIKELY", 2, 2),       # [+][+][CERTAINTY]
    "CERTAIN":    HierarchyLevel("CERTAIN", 3, 3),      # [+][+][+][CERTAINTY]
}


# =============================================================================
# UNIPOLAR PRIMITIVES (No antonym)
# =============================================================================

# Dimension 0: SUBSTANTIVES (7 unipolar + 1 bipolar pair)
DIM_0_UNIPOLAR = [
    "I",          # First person
    "YOU",        # Second person
    "SOMEONE",    # Animate entity
    "SOMETHING",  # Inanimate entity
    "PEOPLE",     # Collective animate
    "BODY",       # Physical embodiment
    "KIND",       # Category/type
    "PART",       # Part-whole relation
]

# Dimension 1: QUANTITY (2 cardinal + 1 base + 1 category)
DIM_1_UNIPOLAR = [
    "ONE",        # Singular
    "TWO",        # Dual (minimal plural)
    "QUANTITY",   # Base for hierarchy
    "NUMBER",     # Category marker for all numerals
]

# Dimension 2: EVALUATORS - all bipolar (no unipolar)
DIM_2_UNIPOLAR: List[str] = []

# Dimension 3: MENTAL (7 unipolar)
DIM_3_UNIPOLAR = [
    "THINK",      # Cognitive processing
    "KNOW",       # Epistemic state
    "WANT",       # Volition
    "FEEL",       # Affective/sensory
    "SEE",        # Visual perception
    "HEAR",       # Auditory perception
    "INFER",      # Epistemic inference
]

# Dimension 4: ACTION (8 unipolar)
DIM_4_UNIPOLAR = [
    "SAY",        # Speech act
    "WORDS",      # Linguistic content
    "DO",         # Agentive action
    "HAPPEN",     # Non-agentive event
    "MOVE",       # Change of location
    "BE",         # Stative existence
    "HAVE",       # Possession
    "EXIST",      # Ontological status
]

# Dimension 5: TEMPORAL (1 unipolar + 3 bipolar)
DIM_5_UNIPOLAR = [
    "MOMENT",     # Punctual time
    "WHEN",       # Temporal deixis
]

# Dimension 6: SPATIAL (3 unipolar + 3 bipolar)
DIM_6_UNIPOLAR = [
    "WHERE",      # Spatial deixis
    "HERE",       # Proximal location
    "SIDE",       # Lateral relation
    "TOUCH",      # Contact relation
]

# Dimension 7: LOGICAL (7 unipolar + hierarchy + 1 bipolar)
DIM_7_UNIPOLAR = [
    "NOT",        # Negation
    "MAYBE",      # Epistemic possibility
    "CAN",        # Ability/possibility
    "MUST",       # Deontic necessity/obligation
    "BECAUSE",    # Retrospective causation/reasoning
    "CAUSE",      # Force dynamics
    "IF",         # Conditionality
    "LIKE",       # Similarity
    "HIER_PLUS",  # Positive hierarchy marker [+]
    "HIER_MINUS", # Negative hierarchy marker [−]
    "INTENSITY",  # Base for intensity hierarchy
    "CERTAINTY",  # Base for certainty hierarchy
]





# =============================================================================
# SEMANTIC ROLES
# =============================================================================

SEMANTIC_ROLES: List[str] = [
    "ARG0",  # Agent / Experiencer
    "ARG1",  # Patient / Theme
    "ARG2",  # Goal / Recipient / Beneficiary
    "ARG3",  # Instrument / Source
    "ARG4",  # Location / Time
    "MOD",   # Modifier (Adjective/Adverb)
]

# =============================================================================
# SCALES
# =============================================================================

class Scale(Enum):
    """Hierarchical scales for semantic encoding."""
    TEXT = "text"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    WORD = "word"


# =============================================================================
# SCALAR BANKS (SENTENCE/TEXT/PARAGRAPH)
# =============================================================================

class SentenceBank(IntEnum):
    """Sentence-scale semantic banks."""
    CORE_STRUCTURE = 0  # SUBJ, PRED, OBJ, COMPL
    MODIFICATION = 1    # ADJ, ADV, DET
    TAM = 2             # Tense, Aspect, Mood
    COMP_0 = 3          # Compositional Operators 0
    COMP_1 = 4          # Compositional Operators 1
    COMP_2 = 5          # Compositional Operators 2
    COMP_3 = 6          # Compositional Operators 3
    RESERVED = 7        # Future use

class TextBank(IntEnum):
    """Text-scale semantic banks."""
    FRAME = 0
    NARRATIVE = 1
    STATE = 2
    RHETORIC = 3
    ARGUMENT = 4
    SOURCE = 5
    STYLE = 6
    META = 7

class ParagraphBank(IntEnum):
    """Paragraph-scale semantic banks."""
    TOPIC = 0
    TRANSITION = 1
    ELABORATION = 2
    EVIDENCE = 3
    SUMMARY = 4
    CONNECTIVE = 5
    DISCOURSE = 6
    MODALITY = 7


# =============================================================================
# SCALAR PRIMITIVES
# =============================================================================

# Sentence-scale (23 primitives)
SENTENCE_PRIMITIVES = [
    # Bank 0: CORE_STRUCTURE
    "SYN_SUBJ", "SYN_PRED", "SYN_OBJ", "SYN_COMPL",
    # Bank 1: MODIFICATION
    "SYN_ADJ_MOD", "SYN_ADV_MOD", "SYN_DET",
    # Bank 2: TAM
    "SYN_TENSE_PAST", "SYN_TENSE_PRES", "SYN_TENSE_FUT",
    "SYN_ASPECT_PERF", "SYN_ASPECT_PROG",
    "SYN_MOOD_INTER", "SYN_MOOD_IMPER",
    # Bank 3-6: COMPOSITIONAL OPERATORS
    "COMP_THAN", "COMP_AS",           # Comparison
    "COMP_HOW", "COMP_WHAT",          # Quantification
    "COMP_JUST", "COMP_ONLY", "COMP_EVEN",  # Constraint
    "COMP_BUT", "COMP_SO", "COMP_BECAUSE",  # Discourse
]

TEXT_PRIMITIVES = [
    "TXT_INTRO", "TXT_BODY", "TXT_CONCL",
    "TXT_NARR_FIRST", "TXT_NARR_THIRD",
    "TXT_ARG_PRO", "TXT_ARG_CON",
    "TXT_STYLE_FORMAL", "TXT_STYLE_INFORMAL",
    "TXT_META_TITLE", "TXT_META_AUTHOR", "TXT_META_DATE",
]

PARAGRAPH_PRIMITIVES = [
    "PARA_TOPIC_SENT", "PARA_SUPPORT", "PARA_CONCL",
    "PARA_TRANS_NEXT", "PARA_TRANS_PREV",
    "PARA_ELAB_DEF", "PARA_ELAB_EX",
    "PARA_EVID_QUOTE", "PARA_EVID_STAT",
    "PARA_SUM_RESTATE",
    "PARA_CONN_AND", "PARA_CONN_BUT", "PARA_CONN_SO",
    "PARA_DISC_QUES", "PARA_DISC_ANS",
]

# Compositional operator metadata
COMP_METADATA = {
    "COMP_THAN": {"arity": 2, "operation": "SET_DIFF"},
    "COMP_AS": {"arity": 2, "operation": "BIND_EQ"},
    "COMP_HOW": {"arity": 1, "operation": "DEGREE_QUERY"},
    "COMP_WHAT": {"arity": 1, "operation": "TYPE_QUERY"},
    "COMP_JUST": {"arity": 2, "operation": "CARD_LOWER"},
    "COMP_ONLY": {"arity": 1, "operation": "SET_RESTRICT"},
    "COMP_EVEN": {"arity": 1, "operation": "SCALAR_EXTREME"},
    "COMP_BUT": {"arity": 2, "operation": "CONTRAST"},
    "COMP_SO": {"arity": 2, "operation": "CONSEQUENCE"},
    "COMP_BECAUSE": {"arity": 2, "operation": "CAUSAL_SOURCE"},
}

def is_compositional_operator(primitive: str) -> bool:
    """Check if primitive is a compositional operator."""
    return primitive in COMP_METADATA

def get_compositional_metadata(primitive: str) -> Optional[Dict]:
    """Get metadata for compositional operator."""
    return COMP_METADATA.get(primitive)


# =============================================================================
# SCALAR REGISTRIES
# =============================================================================

@dataclass
class SentencePrimitiveInfo:
    """Information for sentence primitives."""
    name: str
    bank: SentenceBank
    is_syntactic: bool
    is_compositional: bool
    arity: int = 0
    operation: Optional[str] = None

def _build_sentence_registry() -> Dict[str, SentencePrimitiveInfo]:
    registry = {}
    
    # Map primitives to banks
    bank_map = {
        SentenceBank.CORE_STRUCTURE: ["SYN_SUBJ", "SYN_PRED", "SYN_OBJ", "SYN_COMPL"],
        SentenceBank.MODIFICATION: ["SYN_ADJ_MOD", "SYN_ADV_MOD", "SYN_DET"],
        SentenceBank.TAM: [
            "SYN_TENSE_PAST", "SYN_TENSE_PRES", "SYN_TENSE_FUT",
            "SYN_ASPECT_PERF", "SYN_ASPECT_PROG",
            "SYN_MOOD_INTER", "SYN_MOOD_IMPER"
        ],
        # Combine COMP banks for simplifiction in lookup, though they are distinct in encoding
        SentenceBank.COMP_0: ["COMP_THAN", "COMP_AS"],
        SentenceBank.COMP_1: ["COMP_HOW", "COMP_WHAT"],
        SentenceBank.COMP_2: ["COMP_JUST", "COMP_ONLY", "COMP_EVEN"],
        SentenceBank.COMP_3: ["COMP_BUT", "COMP_SO", "COMP_BECAUSE"],
    }
    
    # Invert map
    prim_to_bank = {}
    for bank, prims in bank_map.items():
        for p in prims:
            prim_to_bank[p] = bank
            
    for prim in SENTENCE_PRIMITIVES:
        bank = prim_to_bank.get(prim, SentenceBank.RESERVED)
        is_comp = prim.startswith("COMP_")
        meta = COMP_METADATA.get(prim, {})
        
        registry[prim] = SentencePrimitiveInfo(
            name=prim,
            bank=bank,
            is_syntactic=prim.startswith("SYN_"),
            is_compositional=is_comp,
            arity=meta.get("arity", 0),
            operation=meta.get("operation")
        )
        
    return registry

SENTENCE_PRIMITIVE_REGISTRY = _build_sentence_registry()


# =============================================================================
# COMPLETE PRIMITIVE REGISTRY
# =============================================================================

@dataclass
class PrimitiveInfo:
    """Full information about a primitive."""
    name: str
    dimension: SemanticDimension
    is_bipolar: bool = False
    bipolar_dim: Optional[str] = None  # Dimension name if bipolar
    is_hierarchy_base: bool = False
    is_hierarchy_marker: bool = False


def _build_primitive_registry() -> Dict[str, PrimitiveInfo]:
    """Build complete primitive registry."""
    registry: Dict[str, PrimitiveInfo] = {}

    # Add unipolar primitives
    dim_primitives = [
        (SemanticDimension.SUBSTANTIVES, DIM_0_UNIPOLAR),
        (SemanticDimension.QUANTITY, DIM_1_UNIPOLAR),
        (SemanticDimension.EVALUATORS, DIM_2_UNIPOLAR),
        (SemanticDimension.MENTAL, DIM_3_UNIPOLAR),
        (SemanticDimension.ACTION, DIM_4_UNIPOLAR),
        (SemanticDimension.TEMPORAL, DIM_5_UNIPOLAR),
        (SemanticDimension.SPATIAL, DIM_6_UNIPOLAR),
        (SemanticDimension.LOGICAL, DIM_7_UNIPOLAR),
    ]

    for dim, primitives in dim_primitives:
        for prim in primitives:
            is_hier_base = prim in ("QUANTITY", "INTENSITY", "CERTAINTY")
            is_hier_marker = prim in ("HIER_PLUS", "HIER_MINUS")
            registry[prim] = PrimitiveInfo(
                name=prim,
                dimension=dim,
                is_bipolar=False,
                is_hierarchy_base=is_hier_base,
                is_hierarchy_marker=is_hier_marker,
            )

    # Add bipolar primitives (both poles)
    for pair in BIPOLAR_PAIRS:
        registry[pair.positive] = PrimitiveInfo(
            name=pair.positive,
            dimension=pair.dimension,
            is_bipolar=True,
            bipolar_dim=pair.name,
        )
        registry[pair.negative] = PrimitiveInfo(
            name=pair.negative,
            dimension=pair.dimension,
            is_bipolar=True,
            bipolar_dim=pair.name,
        )

    return registry


PRIMITIVE_REGISTRY: Dict[str, PrimitiveInfo] = _build_primitive_registry()
