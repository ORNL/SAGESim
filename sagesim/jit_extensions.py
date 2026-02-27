"""
CuPy JIT extensions for SAGESim.

Adds __threadfence() support to CuPy JIT rawkernels, required for
software grid barriers. CuPy doesn't expose this builtin, so we
monkeypatch it following the same pattern as SyncThreads.
"""

from cupyx.jit._internal_types import BuiltinFunc, Data
from cupyx.jit import _cuda_types, _builtin_funcs
import cupyx.jit as jit_module

_installed = False


class ThreadFence(BuiltinFunc):
    """Emits ``__threadfence()`` — ensures all global memory writes by
    the calling thread are visible to all other threads in the device."""

    def __call__(self):
        super().__call__()

    def call_const(self, env):
        return Data('__threadfence()', _cuda_types.void)


def install_jit_extensions():
    """Idempotent installer — safe to call multiple times."""
    global _installed
    if _installed:
        return
    _builtin_funcs.threadfence = ThreadFence()
    jit_module.threadfence = _builtin_funcs.threadfence
    _installed = True
