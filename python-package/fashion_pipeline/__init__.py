"""
This package provides reusable components for a machine learning pipeline
focused on text classification based on customer reviews.

Modules:
- pipeline: Constructs the full preprocessing and classification pipeline.
- transformers: Custom scikit-learn-compatible transformers.
- spacy_transformers: spaCy-based text feature extraction.
- utils: Utility functions for loading models and preprocessing data.
"""

from .pipeline import *
from .utils import *
from .transformers import *
from .spacy_transformers import *
