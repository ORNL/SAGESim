"""Launch a compiled cupyx.jit raw kernel without re-deriving its signature.

``cupyx.jit`` RawKernel's ``__call__`` rebuilds a type signature for every argument on
every call (a ``CArray`` object per array, a scalar rule per scalar), hashes it to find
the compiled kernel, then casts the scalars and launches. For SAGESim's generated step
kernel, with ~60 arguments, that is 0.26 ms per launch on a laptop CPU, several times
the launch itself (0.02 ms) and about as much as a whole sparse tick of the kernel. It
is paid once per launch, i.e. once per tick whenever ticks are not fused (every
multi-rank run, every ``simulate()`` call of a driver that steps presentation by
presentation).

After the first call the compiled ``cupy.cuda.function.Function`` sits in the jit
kernel's cache keyed by that signature. ``DirectLaunch`` takes it out once, keeps the
scalar casts the wrapper would apply, and calls the ``Function`` directly.

The signature is a property of the argument list, not of the data: for an array its
dtype, ndim, contiguity and 32-bit indexability; for a scalar its type. ``accepts()``
checks an argument list against the one the launcher was built for, by identity first
(a buffer that is the same object has the same signature) and by those four facts for
any array that was replaced, and adopts the replacement. A scalar is cast to the type
the kernel was compiled with, as the wrapper would do for a matching signature.

This reaches into private cupyx modules (``_cuda_types``, ``_cuda_typerules``, the
kernel's ``_cache`` / ``_mode``). ``build()`` returns None when anything about them is
not as expected, and the caller keeps using the wrapper. Set ``SAGESIM_DIRECT_LAUNCH=0``
to disable it.
"""

import os

import cupy
import numpy


class DirectLaunch:

    @classmethod
    def build(cls, jit_kernel, args):
        """A launcher for `jit_kernel` with `args`' signature, or None if the wrapper has
        not compiled that signature yet or cupyx's internals are not the expected ones."""
        if os.environ.get("SAGESIM_DIRECT_LAUNCH", "1") == "0":
            return None
        try:
            return cls(jit_kernel, args)
        except (KeyError, AttributeError, ImportError, TypeError):
            return None

    def __init__(self, jit_kernel, args):
        from cupyx.jit import _cuda_types, _cuda_typerules
        from cupyx.jit._cuda_types import Scalar

        self.jit_kernel = jit_kernel
        in_types = []
        for x in args:
            if isinstance(x, cupy.ndarray):
                in_types.append(_cuda_types.CArray.from_ndarray(x))
            else:
                in_types.append(_cuda_typerules.get_ctype_from_scalar(jit_kernel._mode, x))
        key = (tuple(in_types), cupy.cuda.get_device_id())
        self.kern, self.enable_cg = jit_kernel._cache[key]
        self.n_args = len(args)
        self._casts = [(i, numpy.float32 if t.dtype.char == 'e' else t.dtype.type)
                       for i, t in enumerate(in_types) if isinstance(t, Scalar)]
        self._array_pos = [i for i, t in enumerate(in_types) if not isinstance(t, Scalar)]
        # strong references: while held, no other array can get one of these ids
        self._arrays = [args[i] for i in self._array_pos]
        self._facts = [self._fact(a) for a in self._arrays]

    @staticmethod
    def _fact(a):
        return (a.dtype, a.ndim, a._c_contiguous, a._index_32_bits)

    def accepts(self, args):
        """True when `args` has the signature this launcher was compiled for; adopts any
        replaced array that has the same dtype / ndim / contiguity / indexability."""
        if len(args) != self.n_args:
            return False
        arrays, facts = self._arrays, self._facts
        for j, i in enumerate(self._array_pos):
            a = args[i]
            if a is arrays[j]:
                continue
            if not isinstance(a, cupy.ndarray) or self._fact(a) != facts[j]:
                return False
            arrays[j] = a
        return True

    def __call__(self, grid, block, args, shared_mem=0, stream=None):
        """Launch with `grid` and `block` as (x, y, z) tuples, as the wrapper normalises them."""
        a = list(args)
        for i, cast in self._casts:
            a[i] = cast(a[i])
        self.kern(grid, block, tuple(a), shared_mem, stream, self.enable_cg)
