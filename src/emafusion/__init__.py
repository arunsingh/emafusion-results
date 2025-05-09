# EMAFusion – Core library implementation
# MIT License © 2025 Arun Singh <arunsingh.in@gmail.com>

"""Top‑level package exports."""
# src/emafusion/__init__.py
"""Top‑level package exports for emafusion."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("emafusion")
except PackageNotFoundError:  # local editable install
    __version__ = "0.0.0"

from .router import TaxonomyRouter, LearnedRouter  # noqa: F401
from .cascade import Cascade  # noqa: F401
from .judge import LLMJudge  # noqa: F401

__all__ = [
    "TaxonomyRouter",
    "LearnedRouter",
    "Cascade",
    "LLMJudge",
    "__version__",
]