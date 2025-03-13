# SAGESim

Scalable Agent-based GPU Enabled Simulator

# Requirements

 - Python 3.7+
 - `conda create -n sagesimenv python=3.9`
 - `conda activate sagesimenv`
 - Install CUDA toolkit
 - - https://developer.nvidia.com/cuda-toolkit
 - - https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.0.2/
 - - [Install using Anaconda](`conda install -c anaconda cudatoolkit`)
 - install mpi4py (Recommend using conda): https://mpi4py.readthedocs.io/en/stable/install.html#using-conda
 - [Follow the instructions here to install CuPy](https://docs.cupy.dev/en/stable/install.html)
 - `pip install -r requirements.txt`
 
# Run Example

 - `git clone https://code.ornl.gov/sagesim/sagesim`
 - `export PYTHONPATH=/path/to/clone_repo`
 - `cd /path/to/clone_repo/examples/sir`
 - `mpiexec -n 4 python run.py`


# There are some unfortunate quirks to using CuPyx `jit.rawkernel`:
 - nan checked by inequality to self. Unfortunate limitation of cupyx.
 - Dicts and objects are unsupported.
 - *args and **kwargs are unsupported.
 - nested functions are unsupported.
 - Be sure to use `cupy` data types and array routines in favor of `numpy`: [https://docs.cupy.dev/en/stable/reference/routines.html]
 - `for` loops must use range iterator only. No 'for each' style loops.
 - `return` does not seem to work well either
 - `break` and `continue` are unsupported!
 - Cannot reassign variables within `if` or `for` statements. Must be assigned at top level of function or new variable declared under subscope.
 -  `-1` indexing does not necessarily work as expected, as it will access the last element of the memory block of the array instead of the logical array. Use `len(my_array) - 1` instead
