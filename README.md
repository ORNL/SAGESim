![# SAGESim](SAGESim-inline-tag-color.png)


# Requirements

 - Python 3.11+
 - NVIDIA GPU with CUDA drivers or AMD GPU with ROCm 5.7.1+ 

# Installation
Your system might require specific steps to installing `mpi4py` and/or `cupy` depending on your hardware.
In that case use your systems recommended instructions to install `mpi4py` and `cupy` and execute:

`pip install sagesim` 

# Run Example

 - `git clone https://code.ornl.gov/sagesim/sagesim`
 - `cd /path/to/clone_repo/examples/sir`
 - `mpirun -n 4 python run.py --num_agents 10000 --percent_init_connections 0.1 --num_nodes 1`


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
