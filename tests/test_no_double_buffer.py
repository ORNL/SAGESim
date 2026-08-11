"""
Test for no_double_buffer feature.

This test demonstrates the difference between:
1. With double buffering (default): Priority 1 reads value from START of tick
2. Without double buffering: Priority 1 reads value AFTER Priority 0's write (same-tick visibility)

Test scenario:
- Property "counter" starts at 0
- Priority 0 step function: writes counter = 10
- Priority 1 step function: reads counter and writes result = counter * 2

Expected results:
- WITH double buffering: result = 0 * 2 = 0 (reads old value)
- WITHOUT double buffering: result = 10 * 2 = 20 (reads new value from same tick)
"""

import sys

import pytest
from pathlib import Path

import cupy as cp
from cupyx import jit

from sagesim.model import Model
from sagesim.space import NetworkSpace
from sagesim.breed import Breed


# Priority 0: writes counter = 10
@jit.rawkernel(device="cuda")
def write_counter_step_func(
    tick,
    agent_index,
    agent_ids,
    breeds,
    locations,
    counter,
    result,
):
    """Write a fixed value to counter."""
    counter[agent_index] = 10


# Priority 1: reads counter and writes result = counter * 2
@jit.rawkernel(device="cuda")
def read_and_multiply_step_func(
    tick,
    agent_index,
    agent_ids,
    breeds,
    locations,
    counter,
    result,
):
    """Read counter and write result = counter * 2."""
    result[agent_index] = counter[agent_index] * 2


class DoubleBufferBreed(Breed):
    """Breed WITH double buffering (default behavior)."""

    def __init__(self) -> None:
        super().__init__("DoubleBufferBreed")
        self.register_property("counter", 0)
        self.register_property("result", 0)

        curr_fpath = Path(__file__).resolve()
        # Default: double buffering enabled for all properties
        self.register_step_func(write_counter_step_func, curr_fpath, priority=0)
        self.register_step_func(read_and_multiply_step_func, curr_fpath, priority=1)


class NoDoubleBufferBreed(Breed):
    """Breed WITHOUT double buffering for counter."""

    def __init__(self) -> None:
        super().__init__("NoDoubleBufferBreed")
        self.register_property("counter", 0)
        self.register_property("result", 0)

        curr_fpath = Path(__file__).resolve()
        # Disable double buffering for counter - changes visible within same tick
        self.register_step_func(
            write_counter_step_func,
            curr_fpath,
            priority=0,
            no_double_buffer=["counter"]
        )
        self.register_step_func(read_and_multiply_step_func, curr_fpath, priority=1)


class DoubleBufferModel(Model):
    """Model WITH double buffering."""

    def __init__(self) -> None:
        space = NetworkSpace()
        super().__init__(space, step_function_file_path="step_func_code_db.py")
        self._breed = DoubleBufferBreed()
        self.register_breed(breed=self._breed)

    def create_agent(self):
        return self.create_agent_of_breed(self._breed, counter=0, result=0)


class NoDoubleBufferModel(Model):
    """Model WITHOUT double buffering for counter."""

    def __init__(self) -> None:
        space = NetworkSpace()
        super().__init__(space, step_function_file_path="step_func_code_nodb.py")
        self._breed = NoDoubleBufferBreed()
        self.register_breed(breed=self._breed)

    def create_agent(self):
        return self.create_agent_of_breed(self._breed, counter=0, result=0)


class ReaderBeforeWriterDoubleBufferBreed(Breed):
    """Breed WITH double buffering, but reader priority < writer priority."""

    def __init__(self) -> None:
        super().__init__("ReaderBeforeWriterDoubleBufferBreed")
        self.register_property("counter", 0)
        self.register_property("result", 0)

        curr_fpath = Path(__file__).resolve()
        # Reader runs FIRST (priority 0), writer runs SECOND (priority 1)
        self.register_step_func(read_and_multiply_step_func, curr_fpath, priority=0)
        self.register_step_func(write_counter_step_func, curr_fpath, priority=1)


class ReaderBeforeWriterNoDoubleBufferBreed(Breed):
    """Breed WITHOUT double buffering, reader priority < writer priority."""

    def __init__(self) -> None:
        super().__init__("ReaderBeforeWriterNoDoubleBufferBreed")
        self.register_property("counter", 0)
        self.register_property("result", 0)

        curr_fpath = Path(__file__).resolve()
        # Reader runs FIRST (priority 0), writer runs SECOND (priority 1)
        self.register_step_func(read_and_multiply_step_func, curr_fpath, priority=0)
        self.register_step_func(
            write_counter_step_func,
            curr_fpath,
            priority=1,
            no_double_buffer=["counter"]
        )


class ReaderBeforeWriterDoubleBufferModel(Model):
    """Model WITH double buffering, reader < writer priority."""

    def __init__(self) -> None:
        space = NetworkSpace()
        super().__init__(space, step_function_file_path="step_func_code_rbw_db.py")
        self._breed = ReaderBeforeWriterDoubleBufferBreed()
        self.register_breed(breed=self._breed)

    def create_agent(self):
        return self.create_agent_of_breed(self._breed, counter=0, result=0)


class ReaderBeforeWriterNoDoubleBufferModel(Model):
    """Model WITHOUT double buffering, reader < writer priority."""

    def __init__(self) -> None:
        space = NetworkSpace()
        super().__init__(space, step_function_file_path="step_func_code_rbw_nodb.py")
        self._breed = ReaderBeforeWriterNoDoubleBufferBreed()
        self.register_breed(breed=self._breed)

    def create_agent(self):
        return self.create_agent_of_breed(self._breed, counter=0, result=0)


@pytest.fixture(autouse=True)
def clear_step_func_cache():
    """Drop generated step-function modules so each test regenerates its own."""
    yield
    for name in [k for k in sys.modules if k.startswith("step_func_code")]:
        del sys.modules[name]


def test_with_double_buffer_default():
    """
    WITH double buffering (default):
    - Priority 0 writes counter = 10 to write buffer
    - Priority 1 reads counter from read buffer (still 0)
    - result = 0 * 2 = 0
    """
    model = DoubleBufferModel()
    agent_id = model.create_agent()

    model.setup()
    model.simulate(1, sync_workers_every_n_ticks=1)

    counter = model.get_agent_property_value(agent_id, "counter")
    result = model.get_agent_property_value(agent_id, "result")

    # Counter is updated (write buffer copied to read buffer at end of tick)
    assert counter == 10, "counter should be 10 after tick"
    # Result is 0 because priority 1 read from the read buffer (old value)
    assert result == 0, (
        "with double buffering, result should be 0 (read old counter value)"
    )


def test_without_double_buffer():
    """
    WITHOUT double buffering for counter:
    - Priority 0 writes counter = 10 directly to read buffer
    - Priority 1 reads counter from read buffer (now 10)
    - result = 10 * 2 = 20
    """
    model = NoDoubleBufferModel()
    agent_id = model.create_agent()

    model.setup()
    model.simulate(1, sync_workers_every_n_ticks=1)

    counter = model.get_agent_property_value(agent_id, "counter")
    result = model.get_agent_property_value(agent_id, "result")

    assert counter == 10, "counter should be 10 after tick"
    # Result is 20 because priority 1 read the updated counter value
    assert result == 20, (
        "without double buffering, result should be 20 (read new counter value)"
    )


def test_reader_before_writer_with_double_buffer():
    """
    Reader priority (0) < writer priority (1), WITH double buffering:
    - Priority 0 reads counter = 0, writes result = 0
    - Priority 1 writes counter = 10
    - result = 0 * 2 = 0

    Double buffering doesn't matter here because the reader runs first.
    """
    model = ReaderBeforeWriterDoubleBufferModel()
    agent_id = model.create_agent()

    model.setup()
    model.simulate(1, sync_workers_every_n_ticks=1)

    counter = model.get_agent_property_value(agent_id, "counter")
    result = model.get_agent_property_value(agent_id, "result")

    assert counter == 10, "counter should be 10 after tick"
    assert result == 0, "reader runs before writer, result should be 0"


def test_reader_before_writer_without_double_buffer():
    """
    Reader priority (0) < writer priority (1), WITHOUT double buffering:
    - Priority 0 reads counter = 0, writes result = 0
    - Priority 1 writes counter = 10
    - result = 0 * 2 = 0

    Same result as with double buffering: double buffering is unnecessary when
    reader priority < writer priority.
    """
    model = ReaderBeforeWriterNoDoubleBufferModel()
    agent_id = model.create_agent()

    model.setup()
    model.simulate(1, sync_workers_every_n_ticks=1)

    counter = model.get_agent_property_value(agent_id, "counter")
    result = model.get_agent_property_value(agent_id, "result")

    assert counter == 10, "counter should be 10 after tick"
    assert result == 0, (
        "reader runs before writer, result should be 0 (same as with double buffer)"
    )
