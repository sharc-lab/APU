"""Minimal agent harness for M-SLICE (minimal M3.1).

A fixed scaffold in which **only the model endpoint swaps**. Identical prompts, tools, stopping
criteria, and retry logic across every condition - spec §7 M3.1. Per-model prompt tailoring
invalidates H1 and is forbidden.
"""
