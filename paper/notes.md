# Paper Notes

This directory contains notes and drafts for the associated research paper.

## Key Concepts

-   **Structural Antonymy**: Defining antonyms not by distributional statistics (co-occurrence) but by geometric opposition in specific subspaces.
-   **Sparse Ternary Codes**: Using {-1, 0, 1} allows for "binding through overlay" and efficient superposition without noise explosion (unlike dense vectors).
-   **NSM Basis**: grounding the vector space in Wierzbicka's Natural Semantic Metalanguage guarantees universality and interpretability.

## Experiments

1.  **Antonym Detection**:
    -   Validate that `semantic-probing` can identify antonyms purely geometrically.
    -   Compare with Cosine Similarity in Word2Vec/GloVe (which often confuses antonyms/synonyms due to similar context).

2.  **Logical Consistency**:
    -   Can we enable/disable logical reasoning by flipping bits in the logical dimension?
