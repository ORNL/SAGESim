"""sagesim.launch.DirectLaunch: the compiled kernel called without cupyx.jit's per-call
type inference gives the same result as the wrapper, is adopted by the model after its
first launch, follows a replaced buffer of the same signature, and refuses a changed one.
"""
import os
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from cupyx import jit

from sagesim.breed import Breed
from sagesim.launch import DirectLaunch
from sagesim.model import Model
from sagesim.space import NetworkSpace


@jit.rawkernel(device="cuda")
def _scale(n, scale, x, out):
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < n:
        out[i] = x[i] * scale


def test_direct_call_matches_wrapper_and_tracks_signature():
    x = cp.arange(1000, dtype=cp.float32)
    out_w, out_d = cp.zeros_like(x), cp.zeros_like(x)
    _scale[(4,), (256,)](1000, cp.float32(2.0), x, out_w)
    args = (1000, cp.float32(2.0), x, out_d)
    direct = DirectLaunch.build(_scale, args)
    assert direct is not None
    direct((4, 1, 1), (256, 1, 1), args)
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_array_equal(out_d.get(), out_w.get())

    # a scalar arrives as a Python int: cast to the compiled type (int32 here) like the wrapper
    direct((4, 1, 1), (256, 1, 1), (500, 3.0, x, out_d))
    cp.cuda.Stream.null.synchronize()
    assert out_d[499] == 499 * 3.0 and out_d[500] == 500 * 2.0

    # same signature, different buffer: adopted; different dtype / ndim / count: refused
    assert direct.accepts((1000, cp.float32(1.0), x, cp.zeros_like(x)))
    assert not direct.accepts((1000, cp.float32(1.0), x, cp.zeros(1000, dtype=cp.float64)))
    assert not direct.accepts((1000, cp.float32(1.0), x, cp.zeros((10, 100), dtype=cp.float32)))
    assert not direct.accepts((1000, cp.float32(1.0), x))
    # a signature the wrapper has not compiled: no launcher, caller keeps the wrapper
    assert DirectLaunch.build(_scale, (1000, cp.float32(1.0), x, cp.zeros(1000, dtype=cp.float64))) is None


# --- through the model: per-tick launches (verbose timing) use the launcher from tick 2 ---

@jit.rawkernel(device="cuda")
def _count(tick, agent_index, agent_ids, breeds, locations, x):
    x[agent_index] = x[agent_index] + 1.0 + 0.001 * tick


class _B(Breed):
    def __init__(self):
        super().__init__("B")
        self.register_property("x", 0.0)
        self.register_step_func(_count, Path(__file__).resolve(), 0, no_double_buffer=["x"])


class _M(Model):
    def __init__(self, tag):
        super().__init__(NetworkSpace(), step_function_file_path=f"step_func_code_dl_{tag}.py")
        self.b = _B()
        self.register_breed(self.b)


def _run(tag, verbose, ticks=12):
    m = _M(tag)
    ids = [m.create_agent_of_breed(m.b, x=0.0) for _ in range(300)]
    m.setup()
    m.verbose_timing = verbose
    for _ in range(ticks):                   # one launch per call
        m.simulate(1, sync_workers_every_n_ticks=1)
    return m, np.array([m.get_agent_property_value(i, "x") for i in ids[:5]])


def test_model_adopts_direct_launch_and_results_match():
    m_direct, x_direct = _run("on", verbose=True)
    assert getattr(m_direct, "_direct_launch", None) is not None, "launcher not adopted"
    os.environ["SAGESIM_DIRECT_LAUNCH"] = "0"
    try:
        m_wrap, x_wrap = _run("off", verbose=True)
    finally:
        os.environ.pop("SAGESIM_DIRECT_LAUNCH", None)
    assert getattr(m_wrap, "_direct_launch", None) is None
    np.testing.assert_array_equal(x_direct, x_wrap)
    expected = sum(1.0 + 0.001 * t for t in range(12))
    np.testing.assert_allclose(x_direct, expected, rtol=1e-6)
