"""CPU-only checks: python3 -m unittest tests.test_generated_module."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import chdir
import importlib.util
from pathlib import Path
import tempfile
import unittest


# Load the stdlib-only helper without importing SAGESim's GPU dependencies.
spec = importlib.util.spec_from_file_location(
    "_generated_module", Path(__file__).parents[1] / "sagesim/_generated_module.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class Comm:
    def __init__(self, rank=0, result=None):
        self.rank = rank
        self.result = result

    def Get_rank(self):
        return self.rank

    def bcast(self, value, root):
        assert root == 0
        if self.rank == 0:
            self.result = value
            if value[0] is not None:
                assert Path(value[0]).read_text()  # closed and readable at publication
        return self.result


def publish(args):
    directory, configured_path, value = args
    with chdir(directory):
        return helper.write_step_module(
            lambda: f"stepfunc = {value}\n", configured_path, Comm()
        )


class GeneratedModuleTests(unittest.TestCase):
    def test_concurrent_runs_use_distinct_importable_files_in_runtime_directory(self):
        with tempfile.TemporaryDirectory() as original, tempfile.TemporaryDirectory() as runtime:
            configured = str(Path(original) / "step_func_code.py")
            Path(configured).write_text("original source\n")
            with ProcessPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(publish, [(runtime, configured, i) for i in range(12)]))
            self.assertEqual(len(set(paths)), 12)
            self.assertEqual(Path(configured).read_text(), "original source\n")
            for i, path in enumerate(paths):
                self.assertEqual(Path(path).parent.resolve(), Path(runtime).resolve())
                spec = importlib.util.spec_from_file_location(Path(path).stem, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertEqual(module.stepfunc, i)

    def test_peer_uses_root_path_without_generating_source(self):
        with tempfile.TemporaryDirectory() as runtime, chdir(runtime):
            root = Comm()
            path = helper.write_step_module(lambda: "stepfunc = 1\n", "step_func_code.py", root)
            peer = Comm(1, root.result)
            def unexpected():
                self.fail("Only rank zero should generate source")
            self.assertEqual(helper.write_step_module(unexpected, "ignored.py", peer), path)

    def test_generation_error_is_reported_to_all_ranks(self):
        def fail():
            raise ValueError("invalid kernel")
        root = Comm()
        with self.assertRaisesRegex(RuntimeError, "invalid kernel"):
            helper.write_step_module(fail, "step_func_code.py", root)
        with self.assertRaisesRegex(RuntimeError, "invalid kernel"):
            helper.write_step_module(fail, "step_func_code.py", Comm(1, root.result))
