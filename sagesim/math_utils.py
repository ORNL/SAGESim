"""
SAGESim Math Utility Functions

GPU-compatible helper functions for use inside @jit.rawkernel step functions.
These provide common mathematical operations that respect CuPy JIT constraints.

Usage in step functions:
    from sagesim.math_utils import rand_uniform_philox, rand_normal, clamp

All functions are decorated with @jit.rawkernel(device="cuda") and can be
called directly from within step function kernels.

Random Number Generation
========================

Two PRNG algorithms are provided — choose based on your quality/speed needs:

    rand_uniform_philox (Philox-2x32-10) — RECOMMENDED
        Same algorithm used by PyTorch, JAX, TensorFlow for GPU RNG.
        Passes BigCrush (TestU01). 10 rounds, 23-bit output precision.
        Best for: scientific simulations, mortality, recruitment.

    rand_uniform_xorshift (xorshift-multiply)
        3 rounds of xorshift + multiply. Good uniformity, ~3x faster.
        20-bit output precision.
        Best for: high-frequency calls (e.g., daily loops).

All RNG functions take 4 arguments: (seed, tick, agent_index, salt)

    seed         Set once per simulation run. Controls which random stream
                 is used. Different seeds → different simulation outcomes.
                 Same seed → reproducible results.

    tick         SAGESim's built-in simulation step counter. Advances
                 automatically each step, ensuring the same agent gets
                 different random values on different ticks.

    agent_index  SAGeSim's built-in per-agent index (= GPU thread ID).
                 Each agent runs as a separate GPU thread. This ensures
                 parallel agents get different random values on the same tick.

    salt         Manual per-call-site differentiator (1, 2, 3, ...).
                 Within one step function execution, seed/tick/agent_index
                 are all fixed. Salt is the ONLY thing that changes between
                 calls. Each call site must use a unique salt value.

                 Example:
                     r1 = rand_uniform_philox(seed, tick, agent_index, 1)  # mortality
                     r2 = rand_uniform_philox(seed, tick, agent_index, 2)  # recruitment
                     r3 = rand_uniform_philox(seed, tick, agent_index, 3)  # species

Why tick and agent_index as counters
------------------------------------
SAGeSim step functions run once per tick per agent on the GPU. Each execution
is a unique (tick, agent_index) pair — tick advances across time steps, and
agent_index identifies which parallel GPU thread (= which agent) is running.
Together they form a natural 2D counter that guarantees every agent on every
tick gets a unique random stream, without any mutable state or synchronization.

This mirrors how PyTorch uses Philox internally:
    PyTorch             SAGeSim
    ──────────────────  ──────────────────
    seed                seed
    global offset       tick
    thread_id           agent_index
    draw_index          salt
"""

import cupy as cp  # noqa: F401
from cupyx import jit


# ─── Philox-2x32-10 ──────────────────────────────────────────────────
#
# Counter-based RNG (Salmon et al., SC 2011). Uses only integer multiply,
# add, and XOR — all natively supported in CuPy JIT rawkernels.
# 10 rounds unrolled because CuPy JIT does not reliably support variable
# reassignment inside for-loop bodies.
#
# Constants:
#   Multiplier M = 0xD256D193 = 3529668003
#   Weyl key bump W = 0x9E3779B9 = 2654435769 (golden ratio)
#   MOD32 = 2^32 = 4294967296


@jit.rawkernel(device="cuda")
def rand_uniform_philox(seed, tick, agent_index, salt):
    """
    Generate a uniform random float in [0, 1) using Philox-2x32-10.

    Recommended PRNG for scientific simulations. Passes BigCrush (TestU01).

    Args:
        seed: Per-run seed for non-determinism / reproducibility (int)
        tick: Simulation tick — auto-advances each step (int)
        agent_index: Agent buffer index / GPU thread ID (int)
        salt: Unique per-call-site differentiator, e.g. 1, 2, 3 (int)

    Returns:
        float in [0.0, 1.0)
    """
    # Philox-2x32 counter and key initialization
    # counter0: combines tick (time axis) with salt (draw index)
    # counter1: agent_index (thread axis)
    # key: seed (stream selector)
    c0 = (int(tick) * 1000003 + int(salt)) % 4294967296
    c1 = int(agent_index) % 4294967296
    key = int(seed) % 4294967296

    # Round 1
    product = c0 * 3529668003
    hi = (product // 4294967296) % 4294967296
    lo = product % 4294967296
    c0 = (hi ^ key ^ c1) % 4294967296
    c1 = lo
    key = (key + 2654435769) % 4294967296

    # Round 2
    product = c0 * 3529668003
    hi = (product // 4294967296) % 4294967296
    lo = product % 4294967296
    c0 = (hi ^ key ^ c1) % 4294967296
    c1 = lo
    key = (key + 2654435769) % 4294967296

    # Round 3
    product = c0 * 3529668003
    hi = (product // 4294967296) % 4294967296
    lo = product % 4294967296
    c0 = (hi ^ key ^ c1) % 4294967296
    c1 = lo
    key = (key + 2654435769) % 4294967296

    # Round 4
    product = c0 * 3529668003
    hi = (product // 4294967296) % 4294967296
    lo = product % 4294967296
    c0 = (hi ^ key ^ c1) % 4294967296
    c1 = lo
    key = (key + 2654435769) % 4294967296

    # Round 5
    product = c0 * 3529668003
    hi = (product // 4294967296) % 4294967296
    lo = product % 4294967296
    c0 = (hi ^ key ^ c1) % 4294967296
    c1 = lo
    key = (key + 2654435769) % 4294967296

    # Round 6
    product = c0 * 3529668003
    hi = (product // 4294967296) % 4294967296
    lo = product % 4294967296
    c0 = (hi ^ key ^ c1) % 4294967296
    c1 = lo
    key = (key + 2654435769) % 4294967296

    # Round 7
    product = c0 * 3529668003
    hi = (product // 4294967296) % 4294967296
    lo = product % 4294967296
    c0 = (hi ^ key ^ c1) % 4294967296
    c1 = lo
    key = (key + 2654435769) % 4294967296

    # Round 8
    product = c0 * 3529668003
    hi = (product // 4294967296) % 4294967296
    lo = product % 4294967296
    c0 = (hi ^ key ^ c1) % 4294967296
    c1 = lo
    key = (key + 2654435769) % 4294967296

    # Round 9
    product = c0 * 3529668003
    hi = (product // 4294967296) % 4294967296
    lo = product % 4294967296
    c0 = (hi ^ key ^ c1) % 4294967296
    c1 = lo
    key = (key + 2654435769) % 4294967296

    # Round 10
    product = c0 * 3529668003
    hi = (product // 4294967296) % 4294967296
    lo = product % 4294967296
    c0 = (hi ^ key ^ c1) % 4294967296

    # Convert to float [0, 1) — use 23 bits (float32 mantissa precision)
    return float(c0 % 8388608) / 8388608.0


# ─── xorshift-multiply ───────────────────────────────────────────────
#
# Lightweight 3-round hash PRNG. Good uniformity, ~3x faster than Philox.
# Does not pass full BigCrush. Suitable for high-frequency draws where
# speed matters more than statistical rigor.


@jit.rawkernel(device="cuda")
def rand_uniform_xorshift(seed, tick, agent_index, salt):
    """
    Generate a uniform random float in [0, 1) using xorshift-multiply hash.

    Faster alternative to Philox. Good uniformity but does not pass full
    BigCrush. Use for high-frequency draws (e.g., daily precipitation loops).

    Args:
        seed: Per-run seed for non-determinism / reproducibility (int)
        tick: Simulation tick — auto-advances each step (int)
        agent_index: Agent buffer index / GPU thread ID (int)
        salt: Unique per-call-site differentiator, e.g. 1, 2, 3 (int)

    Returns:
        float in [0.0, 1.0)
    """
    h = int(seed) * 1013904243 + int(tick) * 374761393 + int(agent_index) * 668265263 + int(salt) * 2531011
    # Round 1
    h = (h ^ ((h // 65536) % 65536)) % 2147483648
    h = (h * 73244475) % 2147483648
    # Round 2
    h = (h ^ ((h // 65536) % 65536)) % 2147483648
    h = (h * 73244475) % 2147483648
    # Round 3
    h = (h ^ ((h // 65536) % 65536)) % 2147483648
    return float(h % 1000000) / 1000000.0


# ─── Normal Distribution ─────────────────────────────────────────────


@jit.rawkernel(device="cuda")
def rand_normal(seed, tick, agent_index, salt):
    """
    Generate a standard normal sample N(0,1) using the Box-Muller transform.

    Draws two independent Philox uniforms and converts them to an exact
    normal variate via: z = sqrt(-2 ln u1) * cos(2π u2).

    Uses two salt values (salt and salt + 73856093) for uncorrelated draws.

    Args:
        seed: Per-run seed (int)
        tick: Simulation tick (int)
        agent_index: Agent buffer index (int)
        salt: Per-call-site differentiator (int)

    Returns:
        float distributed as N(0,1), practically in [-6, 6]
    """
    u1 = rand_uniform_philox(seed, tick, agent_index, salt)
    u2 = rand_uniform_philox(seed, tick, agent_index, salt + 73856093)
    # Guard against log(0)
    u1_safe = u1 + 1e-10
    z = cp.sqrt(-2.0 * cp.log(u1_safe)) * cp.cos(6.283185307179586 * u2)
    return z


@jit.rawkernel(device="cuda")
def rand_normal_bounded(seed, tick, agent_index, salt, lo, hi):
    """
    Generate a bounded normal sample, clamped to [lo, hi].

    Convenience wrapper combining rand_normal with clamping.

    Args:
        seed: Per-run seed (int)
        tick: Simulation tick (int)
        agent_index: Agent buffer index (int)
        salt: Per-call-site differentiator (int)
        lo: Lower clamp bound (float)
        hi: Upper clamp bound (float)

    Returns:
        float N(0,1) clamped to [lo, hi]
    """
    z = rand_normal(seed, tick, agent_index, salt)
    result = z
    if result < lo:
        result = lo
    if result > hi:
        result = hi
    return result


# ─── Math Helpers ─────────────────────────────────────────────────────


@jit.rawkernel(device="cuda")
def clamp(value, lo, hi):
    """
    Constrain a value to the range [lo, hi].

    Args:
        value: Input value (float)
        lo: Lower bound (float)
        hi: Upper bound (float)

    Returns:
        float clamped to [lo, hi]
    """
    result = value
    if result < lo:
        result = lo
    if result > hi:
        result = hi
    return result


@jit.rawkernel(device="cuda")
def safe_divide(numerator, denominator, default):
    """
    Divide numerator by denominator, returning default if denominator is zero.

    Avoids NaN/Inf on GPU from division by zero.

    Args:
        numerator: Dividend (float)
        denominator: Divisor (float)
        default: Value to return when denominator == 0 (float)

    Returns:
        numerator / denominator, or default if denominator == 0
    """
    result = default
    if denominator > 1e-30 or denominator < -1e-30:
        result = numerator / denominator
    return result


@jit.rawkernel(device="cuda")
def lerp(a, b, t):
    """
    Linear interpolation between a and b.

    Args:
        a: Start value (float)
        b: End value (float)
        t: Interpolation factor, typically in [0, 1] (float)

    Returns:
        a + t * (b - a)
    """
    return a + t * (b - a)


@jit.rawkernel(device="cuda")
def kronecker(x):
    """
    Kronecker delta (step function).

    Args:
        x: Input value (float)

    Returns:
        1.0 if x >= 0, else 0.0
    """
    result = 0.0
    if x >= 0.0:
        result = 1.0
    return result
