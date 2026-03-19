import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec("cupy") is None:
    fake_cupy = types.ModuleType("cupy")
    fake_cupy.array = lambda *args, **kwargs: None
    fake_cupy.concatenate = lambda *args, **kwargs: None
    fake_cupy.empty = lambda *args, **kwargs: None
    fake_cupy.full = lambda *args, **kwargs: None
    fake_cupy.zeros = lambda *args, **kwargs: None
    fake_cupy.ndarray = object
    fake_cupy.float32 = float
    fake_cupy.int32 = int
    fake_cupy.int64 = int
    fake_cupy.uint32 = int
    fake_cupy.nan = float("nan")
    fake_cupy.disable_experimental_feature_warning = True
    sys.modules["cupy"] = fake_cupy

if importlib.util.find_spec("awkward") is None:
    fake_awkward = types.ModuleType("awkward")
    fake_awkward.from_iter = lambda *args, **kwargs: types.SimpleNamespace(
        layout=types.SimpleNamespace(minmax_depth=(1, 1))
    )
    fake_awkward.to_regular = lambda value, axis=None: value
    fake_awkward.to_numpy = lambda value: value
    fake_awkward.to_cupy = lambda value: value
    fake_awkward.from_cupy = lambda value: value
    sys.modules["awkward"] = fake_awkward

if importlib.util.find_spec("mpi4py") is None:
    class _FakeComm:
        def Get_size(self):
            return 1

        def Get_rank(self):
            return 0

        def barrier(self):
            return None

    fake_mpi4py = types.ModuleType("mpi4py")
    fake_mpi4py.MPI = types.SimpleNamespace(COMM_WORLD=_FakeComm())
    sys.modules["mpi4py"] = fake_mpi4py

if importlib.util.find_spec("cupyx") is None:
    def _rawkernel(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator

    fake_cupyx = types.ModuleType("cupyx")
    fake_cupyx_jit = types.ModuleType("cupyx.jit")
    fake_cupyx_jit.rawkernel = _rawkernel
    fake_cupyx.jit = fake_cupyx_jit
    sys.modules["cupyx"] = fake_cupyx
    sys.modules["cupyx.jit"] = fake_cupyx_jit

from sagesim.model import Model
from sagesim.space import NetworkSpace


class TestStepFunctionRuntimePath(unittest.TestCase):
    def test_setup_writes_generated_step_module_into_current_working_directory(self):
        generated_module_source = "stepfunc = 'runtime kernel'\n"

        with tempfile.TemporaryDirectory() as source_dir_str, tempfile.TemporaryDirectory() as run_dir_str:
            source_dir = Path(source_dir_str)
            run_dir = Path(run_dir_str)
            configured_step_module_path = source_dir / "step_func_code.py"
            model_pickle_path = source_dir / "model.pkl"

            model = Model(
                NetworkSpace(),
                step_function_file_path=str(configured_step_module_path),
            )
            model.save(model, str(model_pickle_path))
            loaded_model = model.load(str(model_pickle_path))

            original_cwd = Path.cwd()
            try:
                os.chdir(run_dir)
                with patch(
                    "sagesim.model.generate_gpu_func",
                    return_value=generated_module_source,
                ), patch.object(
                    loaded_model._agent_factory,
                    "_generate_agent_data_tensors",
                    return_value=[],
                ):
                    loaded_model.setup(use_gpu=False)
            finally:
                os.chdir(original_cwd)

            generated_step_module_path = Path(
                loaded_model._generated_step_function_file_path
            )
            self.assertEqual(generated_step_module_path.parent, run_dir.resolve())
            self.assertTrue(generated_step_module_path.exists())
            self.assertRegex(
                generated_step_module_path.name,
                r"^step_func_code_[0-9a-f]{16}\.py$",
            )
            self.assertEqual(
                generated_step_module_path.read_text(encoding="utf-8"),
                generated_module_source,
            )
            self.assertFalse(configured_step_module_path.exists())
            self.assertEqual(loaded_model._step_func, "runtime kernel")
