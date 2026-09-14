"""Analysis for M-SLICE.

Spec §7.3: the analysis layer consumes ``blinded_label`` only. The unblinded label is written into
manifests but must never be read here, and ``tests/test_blinding_boundary.py`` enforces that by
scanning this package for the forbidden identifier.
"""
