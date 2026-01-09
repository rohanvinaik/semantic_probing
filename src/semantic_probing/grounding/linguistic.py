"""Linguistic grounding module."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import sqlite3
import re

from ..encoding import (
    SparseVector,
    HadamardBasis,
    SentenceScaleBasis,
    encode_word_full,
    create_sentence_basis,
    SEMANTIC_ROLES,
    bind,
    bundle,
    SentenceBank,
    SENTENCE_PRIMITIVES,
    SemanticDimension,
)

@dataclass
class LinguisticProfile:
    """Semantic profile from linguistic grounding."""
    primitive_activations: Dict[str, float] = field(default_factory=dict)
    dimension_activations: Dict[str, float] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)
    attributes: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    token_coverage: float = 0.0


class LinguisticGrounder:
    """Ground text to semantic primitives."""

    def __init__(self, lexicon_path: Optional[str] = None):
        self.basis = HadamardBasis().generate()
        self.lexicon_path = lexicon_path
        self._conn = None
        if lexicon_path:
            try:
                self._conn = sqlite3.connect(lexicon_path)
                # Enable dictionary cursor access effectively
                self._conn.row_factory = sqlite3.Row
            except Exception as e:
                print(f"Warning: Could not connect to lexicon at {lexicon_path}: {e}")

    def __del__(self):
        if self._conn:
            self._conn.close()

    def encode_word(self, word: str, pos: str = "n") -> Optional[SparseVector]:
        """Encode a single word to its semantic vector."""
        if not self._conn:
            return None

        cursor = self._conn.cursor()
        
        # Try to find word in lexemes
        # Try exact match first, then case-insensitive
        query = "SELECT id, decomposition FROM lexemes WHERE lemma = ? AND pos = ?"
        cursor.execute(query, (word, pos))
        rows = cursor.fetchall()
        
        if not rows:
            cursor.execute(query, (word.lower(), pos))
            rows = cursor.fetchall()
            
        if not rows:
            return None
            
        # Use first match
        row = rows[0]
        decomposition_json = row["decomposition"]
        
        if not decomposition_json:
            return None
            
        import json
        try:
            decomposition = json.loads(decomposition_json)
        except:
            return None
            
        return self._compose_decomposition(decomposition)

    def _compose_decomposition(
        self,
        decomposition: List[Tuple[str, str, int]]
    ) -> SparseVector:
        """Compose vector from decomposition list."""
        components = []
        
        for prim, role, polarity in decomposition:
            # Get primitive vector
            prim_vec = self.basis.get_primitive(prim)
            if prim_vec is None:
                continue

            # Apply polarity
            if polarity == -1:
                prim_vec = prim_vec.negate()

            # Bind with role if available
            role_vec = self.basis.get_role(role)
            if role_vec:
                bound = bind(prim_vec, role_vec)
                components.append(bound)
            else:
                components.append(prim_vec)

            # Also add raw primitive (for direct similarity)
            components.append(prim_vec)

        if not components:
            return SparseVector.zeros(self.basis.dimension)
            
        return bundle(components)

    def create_profile(self, text: str) -> LinguisticProfile:
        """
        Create a semantic profile for a text.
        
        Simplified: word-level only, no sentence structure.
        """
        words = self._tokenize(text)
        if not words:
            return LinguisticProfile()
            
        vectors = []
        primitive_activations = {}
        dimension_counts = {dim.name: 0.0 for dim in SemanticDimension}
        
        covered_tokens = 0
        
        for word in words:
            # Simple heuristic: try 'n', if fail try 'v', 'adj'
            vec = self.encode_word(word, "n")
            if not vec:
                vec = self.encode_word(word, "v")
            if not vec:
                vec = self.encode_word(word, "adj")
                
            if vec:
                vectors.append(vec)
                covered_tokens += 1
                
                # Analyze activations (simplified)
                # In a real system, we'd probe the vector. 
                # Here, since we just constructed it, we could use the decomposition if we had it.
                # But encode_word returns vector.
                # Let's probe the vector against the basis primitives.
                
                # Optimization: For now, just accumulation of raw primitive hits isn't efficient 
                # if we do it for every word by dot product.
                # But we are building a profile.
                pass 
        
        # Aggregate logic
        token_coverage = covered_tokens / len(words) if words else 0.0
        
        # If we have vectors, bundle them to get a text vector
        if vectors:
            text_vec = bundle(vectors)
            
            # Probe against all primitives to get activations
            for prim_name, prim_vec in self.basis.primitives.items():
                activation = text_vec.cosine(prim_vec)
                if abs(activation) > 0.05: # Threshold
                    primitive_activations[prim_name] = activation
                    
                    # Add to dimension aggregation
                    info = PRIMITIVE_REGISTRY.get(prim_name)
                    if info:
                        dim_name = info.dimension.name
                        dimension_counts[dim_name] = dimension_counts.get(dim_name, 0.0) + abs(activation)
        
        return LinguisticProfile(
            primitive_activations=primitive_activations,
            dimension_activations=dimension_counts,
            confidence=token_coverage, # Proxy
            token_coverage=token_coverage,
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer."""
        # Split by non-alphanumeric, remove empty
        return [w for w in re.split(r'[^a-zA-Z0-9]+', text) if w]


@dataclass
class SentenceGrounder:
    """
    Grounds conceptual content into sentence-scale semantic vectors.
    
    Binds word-scale vectors with syntactic roles and compositional operators.
    """
    word_basis: HadamardBasis
    sentence_basis: SentenceScaleBasis = field(default_factory=create_sentence_basis)
    
    def encode_sentence(
        self,
        words: List[str],
        roles: List[str],
        operators: List[Tuple[int, str]] = None, # (index, operator_name)
    ) -> SparseVector:
        """
        Encode a sequence of words with their syntactic roles.
        
        Args:
            words: List of word strings
            roles: List of syntactic role strings (e.g. "SYN_SUBJ")
            operators: Optional list of (index, operator_name) tuples to bind to specific words
            
        Returns:
            Bundled sentence vector
        """
        if len(words) != len(roles):
            raise ValueError("Words and roles must have same length")
            
        operators_map = dict(operators) if operators else {}
        
        components = []
        for i, (word, role) in enumerate(zip(words, roles)):
            word_vec = self.word_basis.get_primitive(word)
            if word_vec is None:
                continue # Skip unknown words or handle as OOV
                
            role_vec = self.sentence_basis.get_primitive(role)
            if role_vec is None:
                # Fallback to generic semantic roles if sentence role not found
                # but here we expect SYN_* primitives
                continue
                
            op_vec = None
            if i in operators_map:
                op_vec = self.sentence_basis.get_primitive(operators_map[i])
                
            # Bind Word * Role (* Op)
            encoded = encode_word_full(word_vec, role_vec, op_vec)
            components.append(encoded)
            
        if not components:
            return SparseVector.zeros(self.word_basis.dimension)
            
        return bundle(components)
