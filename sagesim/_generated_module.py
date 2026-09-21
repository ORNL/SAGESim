"""Publish generated Python source for all ranks in an MPI communicator."""

from pathlib import Path
import tempfile


def write_step_module(source_factory, configured_path, comm):
    """Return a unique shared source path rooted in rank zero's current directory.

    Keep the source on disk: CuPy inspects it when compiling kernels lazily.
    All ranks must be able to access rank zero's working directory.
    """
    result = None
    if comm.Get_rank() == 0:
        path = None
        try:
            source = source_factory()
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=Path.cwd(),
                prefix=Path(configured_path).stem + "_", suffix=".py", delete=False,
            ) as stream:
                path = Path(stream.name)
                stream.write(source)
            result = (str(path), None)
        except Exception as exc:
            if path is not None:
                path.unlink(missing_ok=True)
            result = (None, f"{type(exc).__name__}: {exc}")
    # Broadcast only after closing the file; peers never see partial source.
    path, error = comm.bcast(result, root=0)
    if error is not None:
        raise RuntimeError(f"Could not generate step module: {error}")
    return path
