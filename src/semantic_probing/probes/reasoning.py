"""Reasoning probing logic (relations, logic)."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import hashlib

from ..encoding import (
    SparseVector,
    HadamardBasis,
    bind,
    bundle,
    generate_role_pattern,
    D,
    N_DIMENSIONS,
    DIMS_PER_DIMENSION,
)

@dataclass
class Fact:
    """A ground fact (predicate + arguments)."""
    predicate: str
    arguments: List[str]
    weight: float = 1.0


class ReasoningProbe:
    """
    Probes for logical and relational reasoning.
    
    Adapts HDC-Neurosymbolic concepts:
    - Facts encoded as Predicate + Role-Filler bindings
    - Rules encoded as Head + Body bindings
    """

    def __init__(self, basis: Optional[HadamardBasis] = None):
        if basis is None:
            basis = HadamardBasis().generate()
        self.basis = basis
        
        # Cache for generated role/predicate vectors not in basis
        self._cache: Dict[str, SparseVector] = {}

    def _get_vector(self, name: str, is_role: bool = False) -> SparseVector:
        """Get vector for name, generating if needed."""
        # Check standard basis (primitives/roles)
        if is_role:
            vec = self.basis.get_role(name)
        else:
            vec = self.basis.get_primitive(name)
            
        if vec:
            return vec
            
        # Check cache
        cache_key = f"role:{name}" if is_role else f"sym:{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Generate new stable vector
        # Use hash of name to seed generation
        seed = int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**32)
        
        if is_role:
            # Generate generic role pattern (like arg_0)
            # Use seed as index substitute
            vec = generate_role_pattern(
                role_index=seed,
                dimension=self.basis.dimension,
                n_banks=self.basis.n_banks,
                dims_per_bank=self.basis.dims_per_bank,
            )
        else:
            # Generate generic symbol pattern (like a primitive)
            # For random symbols, we map to a random bank?
            # Or use role pattern for symbols too? 
            # Ideally symbols are orthogonal.
            # Let's use role pattern but maybe sparser?
            # Or just re-use role pattern generator which makes random vectors spanning all banks.
            # This makes symbols distributed.
            vec = generate_role_pattern(
                role_index=seed,
                dimension=self.basis.dimension,
                n_banks=self.basis.n_banks,
                dims_per_bank=self.basis.dims_per_bank,
            )
            
        self._cache[cache_key] = vec
        return vec

    def encode_fact(
        self,
        fact: Fact,
        entity_vectors: Optional[Dict[str, SparseVector]] = None
    ) -> SparseVector:
        """
        Encode a fact into a hypervector.
        
        Structure: Predicate + Bundle(Bind(Arg_i, Role_i))
        """
        # Encode predicate
        pred_vec = self._get_vector(fact.predicate)
        
        # Encode arguments
        arg_components = []
        for i, arg in enumerate(fact.arguments):
            # Get semantic role (arg_0, arg_1...) or use named args if provided?
            # We'll use positional arg_i for generality
            role_vec = self._get_vector(f"arg_{i}", is_role=True)
            
            # Get filler vector
            if entity_vectors and arg in entity_vectors:
                filler_vec = entity_vectors[arg]
            else:
                filler_vec = self._get_vector(arg)
                
            # Bind role * filler
            bound = bind(filler_vec, role_vec)
            arg_components.append(bound)
            
        # Bundle arguments
        if arg_components:
            args_vec = bundle(arg_components)
            # Bind predicate with arguments (or just bundle? Usually Pred * Args or Pred + Args)
            # Relational-AI uses Pred + Args for simple facts?
            # Actually, (Pred * Roles) or similar.
            # We'll use Pred + Args to keep them separable if we query?
            # But Pred + Args mixes them in superposition.
            # Ideally: Bind(Pred, Args) implies Predicate applies to Args.
            # Let's use bundle([pred_vec, args_vec]) -> superposition.
            return bundle([pred_vec, args_vec])
            
        return pred_vec

    def query_fact(
        self,
        query_vec: SparseVector,
        kb_vector: SparseVector,
        threshold: float = 0.1
    ) -> float:
        """
        Query if a fact is present in the knowledge base vector.
        
        Args:
            query_vec: Encoded fact vector (the hypothesis)
            kb_vector: Superposition of all known facts
            
        Returns:
            Confidence score (cosine similarity)
        """
        return query_vec.cosine(kb_vector)

    def probe_relation(
        self,
        subject_vec: SparseVector,
        object_vec: SparseVector,
        relation_name: str
    ) -> float:
        """
        Probe if a specific relation holds between subject and object.
        
        Hypothesis: Relation + (Subj * Arg0) + (Obj * Arg1)
        """
        fact = Fact(relation_name, ["SUBJ", "OBJ"])
        # We manually construct the vectors
        # Use dummy dict
        entity_vectors = {"SUBJ": subject_vec, "OBJ": object_vec}
        
        hypothesis_vec = self.encode_fact(fact, entity_vectors)
        
        # We don't have a KB here?
        # This function seems to assume we are probing if Subj/Obj are related by Relation
        # relative to EACH OTHER?
        # Or relative to a KB?
        # If no KB, this is meaningless unless we check intrinsic property.
        
        # If we interpret typical "relation probing", we might check if 
        # bind(subject, relation) is close to object?
        # E.g. Translation: bind(King, Man-to-Woman) ~ Queen
        
        # Let's try transformational probing:
        # Relation vector mapping Arg0 to Arg1.
        # VectorR * Subj ~ Obj ?
        # In sparse ternary: bind(Subj, Relation) ~ Obj
        
        rel_vec = self._get_vector(relation_name) # Or _get_vector(relation_name, is_role=True)
        # We'll assume relation is an operator
        
        transformed = bind(subject_vec, rel_vec)
        return transformed.cosine(object_vec)
