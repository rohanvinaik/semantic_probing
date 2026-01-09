"""
Text encoder for semantic probing.
Converts text to SparseVectors using primitive mapping.
"""

import re
import sqlite3
from typing import List, Optional

from .sparse_ternary import SparseVector, HadamardBasis, bundle, D
from .semantic_dimensions import PRIMITIVE_REGISTRY

class TextEncoder:
    """Encodes text into semantic vectors."""

    def __init__(self, db_path: str = "data/lexicon.db"):
        self.db_path = db_path
        self.basis = HadamardBasis().generate()
        self._cache = {}

    def encode(self, text: str) -> SparseVector:
        """Encode text string into a single SparseVector."""
        words = self._tokenize(text)
        vectors = []
        
        for word in words:
            vec = self._encode_word(word)
            if vec:
                vectors.append(vec)
                
        if not vectors:
            return SparseVector.zeros(D)
            
        return bundle(vectors)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace/punctuation tokenization."""
        # Split by non-alphanumeric, keep only valid words
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _encode_word(self, word: str) -> Optional[SparseVector]:
        """Encode a single word."""
        if word in self._cache:
            return self._cache[word]

        # 1. Check if it's a primitive name directly
        upper = word.upper()
        if upper in PRIMITIVE_REGISTRY:
            vec = self.basis.get_primitive(upper)
            if vec:
                self._cache[word] = vec
                return vec

        # 2. Look up in lexicon DB (if exists)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT primitive, weight FROM activation WHERE word = ?", 
                    (word,)
                )
                rows = cursor.fetchall()
                
                if rows:
                    vecs = []
                    # Weighted bundle? bundles are usually unweighted in ternary
                    # We can repeat vectors to approximate weight?
                    # Or just bundle the primitives.
                    for prim, weight in rows:
                        pvec = self.basis.get_primitive(prim)
                        if pvec:
                            # Simple approach: add primitive
                            # Better: could implement weighted superposition if SparseVector allowed
                            # For now, just bundle unique primitives
                            vecs.append(pvec)
                    
                    if vecs:
                        res = bundle(vecs)
                        self._cache[word] = res
                        return res
        except sqlite3.Error:
            pass
            
        return None
