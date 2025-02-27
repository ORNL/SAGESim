from math import isnan, nan

# from numba import cuda


# @cuda.jit
def step_func_helper_get_agent_index(id, id2index):
    if isnan(id):
        return nan
    if isnan(id2index[int(id)]):
        return nan
    return int(id2index[int(id)])


# @cuda.jit
# def step_func_helper_get_random_float(rng_states, idx):
#    return cuda.random.xoroshiro128p_uniform_float32(rng_states, idx)
