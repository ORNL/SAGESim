"""Shared fixtures for the SAGESim test suite.

Every test here executes GPU kernels, so the whole suite is skipped when no
device is visible rather than erroring at import time.
"""

import cupy as cp
import pytest

from sagesim.jit_extensions import install_jit_extensions


def _device_count() -> int:
    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


@pytest.fixture(scope="session", autouse=True)
def jit_extensions():
    """Install the ``jit.threadfence`` builtin once, before any test runs.

    SAGESim's generated step-function files install this themselves (see
    ``sagesim/model.py``), but the grid-barrier tests hand-write rawkernels that
    need it too. Installing here rather than inside individual tests keeps the
    monkeypatch from being a mid-suite side effect, so results don't depend on
    collection order. ``install_jit_extensions`` is idempotent.
    """
    install_jit_extensions()


def pytest_collection_modifyitems(config, items):
    """Skip the whole suite when no GPU is visible.

    SAGESim compiles its step functions to CuPy kernels and has no CPU backend,
    so without a device there is nothing meaningful to assert.
    """
    if _device_count() > 0:
        return
    skip_gpu = pytest.mark.skip(reason="no GPU visible; SAGESim executes only on GPU")
    for item in items:
        item.add_marker(skip_gpu)
