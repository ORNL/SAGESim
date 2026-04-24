"""Tests for GPU RNG and math helper functions in sagesim.math_utils."""

import unittest
import numpy as np
import cupy as cp
from cupyx import jit

from sagesim.math_utils import (
    rand_uniform_philox,
    rand_uniform_xorshift,
    rand_normal,
    clamp,
    lerp,
)


# ── Wrapper kernels ──────────────────────────────────────────────────
# Each kernel calls the math_utils function and writes the result to an
# output array so we can inspect it on the CPU.

@jit.rawkernel(device="cuda")
def _kernel_philox(out, seed, tick, n):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if tid < n:
        out[tid] = rand_uniform_philox(seed, tick, tid, 1)


@jit.rawkernel(device="cuda")
def _kernel_philox_salt(out, seed, tick, salt, n):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if tid < n:
        out[tid] = rand_uniform_philox(seed, tick, tid, salt)


@jit.rawkernel(device="cuda")
def _kernel_xorshift(out, seed, tick, n):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if tid < n:
        out[tid] = rand_uniform_xorshift(seed, tick, tid, 1)


@jit.rawkernel(device="cuda")
def _kernel_normal(out, seed, tick, n):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if tid < n:
        out[tid] = rand_normal(seed, tick, tid, 1)


@jit.rawkernel(device="cuda")
def _kernel_clamp(out, values, lo, hi, n):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if tid < n:
        out[tid] = clamp(values[tid], lo, hi)


@jit.rawkernel(device="cuda")
def _kernel_safe_divide(out, nums, denoms, fallback_val, n):
    """Inline safe_divide logic to avoid C++ keyword 'default' in math_utils."""
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if tid < n:
        result = fallback_val
        d = denoms[tid]
        if d > 1e-30 or d < -1e-30:
            result = nums[tid] / d
        out[tid] = result


@jit.rawkernel(device="cuda")
def _kernel_lerp(out, a_arr, b_arr, t_arr, n):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if tid < n:
        out[tid] = lerp(a_arr[tid], b_arr[tid], t_arr[tid])


def _launch(kernel, *args, n=1024):
    blocks = (n + 31) // 32
    kernel[blocks, 32](*args, cp.int32(n))
    cp.cuda.Stream.null.synchronize()


class TestPhiloxRNG(unittest.TestCase):

    def test_deterministic(self):
        """Same inputs produce identical output."""
        n = 1024
        out1 = cp.zeros(n, dtype=cp.float32)
        out2 = cp.zeros(n, dtype=cp.float32)
        _launch(_kernel_philox, out1, cp.int32(42), cp.int32(0), n=n)
        _launch(_kernel_philox, out2, cp.int32(42), cp.int32(0), n=n)
        np.testing.assert_array_equal(out1.get(), out2.get())

    def test_range(self):
        """All values in [0, 1)."""
        n = 10000
        out = cp.zeros(n, dtype=cp.float32)
        _launch(_kernel_philox, out, cp.int32(123), cp.int32(5), n=n)
        result = out.get()
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result < 1.0))

    def test_different_salts(self):
        """Different salt values produce different sequences."""
        n = 1024
        out1 = cp.zeros(n, dtype=cp.float32)
        out2 = cp.zeros(n, dtype=cp.float32)
        _launch(_kernel_philox_salt, out1, cp.int32(42), cp.int32(0), cp.int32(1), n=n)
        _launch(_kernel_philox_salt, out2, cp.int32(42), cp.int32(0), cp.int32(2), n=n)
        self.assertFalse(np.array_equal(out1.get(), out2.get()))


class TestXorshiftRNG(unittest.TestCase):

    def test_range(self):
        """All values in [0, 1)."""
        n = 10000
        out = cp.zeros(n, dtype=cp.float32)
        _launch(_kernel_xorshift, out, cp.int32(99), cp.int32(3), n=n)
        result = out.get()
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result < 1.0))


class TestNormalRNG(unittest.TestCase):

    def test_distribution(self):
        """Large sample has mean ~0, std ~1."""
        n = 50000
        out = cp.zeros(n, dtype=cp.float32)
        _launch(_kernel_normal, out, cp.int32(7), cp.int32(0), n=n)
        result = out.get()
        self.assertAlmostEqual(float(np.mean(result)), 0.0, delta=0.05)
        self.assertAlmostEqual(float(np.std(result)), 1.0, delta=0.1)


class TestMathHelpers(unittest.TestCase):

    def test_clamp(self):
        n = 4
        values = cp.array([-5.0, 0.5, 3.0, 10.0], dtype=cp.float32)
        out = cp.zeros(n, dtype=cp.float32)
        _launch(_kernel_clamp, out, values, cp.float32(0.0), cp.float32(1.0), n=n)
        np.testing.assert_array_almost_equal(out.get(), [0.0, 0.5, 1.0, 1.0])

    def test_safe_divide(self):
        n = 3
        nums = cp.array([10.0, 5.0, 1.0], dtype=cp.float32)
        denoms = cp.array([2.0, 0.0, 4.0], dtype=cp.float32)
        out = cp.zeros(n, dtype=cp.float32)
        _launch(_kernel_safe_divide, out, nums, denoms, cp.float32(-1.0), n=n)
        result = out.get()
        self.assertAlmostEqual(result[0], 5.0, places=5)
        self.assertAlmostEqual(result[1], -1.0, places=5)  # default
        self.assertAlmostEqual(result[2], 0.25, places=5)

    def test_lerp(self):
        n = 3
        a = cp.array([0.0, 10.0, 100.0], dtype=cp.float32)
        b = cp.array([10.0, 20.0, 200.0], dtype=cp.float32)
        t = cp.array([0.0, 0.5, 1.0], dtype=cp.float32)
        out = cp.zeros(n, dtype=cp.float32)
        _launch(_kernel_lerp, out, a, b, t, n=n)
        np.testing.assert_array_almost_equal(out.get(), [0.0, 15.0, 200.0])


if __name__ == "__main__":
    unittest.main()
