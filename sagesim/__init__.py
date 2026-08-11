"""SAGESim - Scalable Agent-based GPU-Enabled Simulator."""

from importlib.metadata import version as _version, PackageNotFoundError as _PackageNotFound

import cupy
cupy.disable_experimental_feature_warning = True

from sagesim.model import Model
from sagesim.breed import Breed
from sagesim.space import NetworkSpace
from sagesim.utils import get_num_neighbors, get_neighbor

try:
    __version__ = _version("sagesim")
except _PackageNotFound:  # source tree that was never pip-installed
    __version__ = "unknown"
__all__ = ["Model", "Breed", "NetworkSpace", "get_num_neighbors", "get_neighbor"]
