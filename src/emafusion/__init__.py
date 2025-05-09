# EMAFusion – Core library implementation
# MIT License © 2025 Arun Singh <arunsingh.in@gmail.com>

"""Top‑level package exports."""
# src/emafusion/__init__.py
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("emafusion")
except PackageNotFoundError:  # local editable install
    __version__ = "0.0.0"

from .router import TaxonomyRouter, LearnedRouter  # noqa: F401
from .cascade import Cascade  # noqa: F401
from .judge import Judge  # noqa: F401
from .fusion import fuse_answers  # noqa: F401
from .datasets import load_dataset  # noqa: F401