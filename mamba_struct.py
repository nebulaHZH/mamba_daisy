"""
Backward-compatible wrapper for legacy imports.

The shared model definitions now live in mamba_models.py.
Keep this module as a thin re-export layer so older code does not break.
"""

from mamba_models import MambaBlock, MambaSequenceClassifier, ResidualBlock

__all__ = ["MambaBlock", "ResidualBlock", "MambaSequenceClassifier"]
