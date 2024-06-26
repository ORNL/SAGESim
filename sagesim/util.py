import time
from typing import Iterable, List, Any, Tuple, Optional
import math

import numpy as np

from dask.distributed import Client, LocalCluster


def convert_to_equal_side_tensor(
    tensor: List[Any], max_dims: Optional[Tuple[int]] = None
) -> np.array:
    dim2maxlen = {}

    def find_max_depth(l, curr_depth=0):
        dim2maxlen[curr_depth] = max(dim2maxlen.get(curr_depth, 0), len(l))
        max_depth = curr_depth
        for item in l:
            if type(item) == list:
                max_depth = max(max_depth, find_max_depth(item, curr_depth + 1))
        return max_depth

    start = time.time()
    if not max_dims:
        print("no dims specified... running find max depth")
        find_max_depth(tensor)
        max_dims = tuple(list(dim2maxlen.values()))

    if len(max_dims) > 1 and max_dims[1] == 0:
        return np.full(shape=max_dims[0], fill_value=math.nan)

    answer = np.full(shape=max_dims, fill_value=math.nan)

    def fill_arr(arr, coord):
        if len(coord) == len(max_dims):
            if type(arr) == list:
                raise TypeError()
            answer[tuple(coord)] = arr
        else:
            for i, item in enumerate(arr):
                new_coord = coord + [i]
                fill_arr(item, new_coord)

    start = time.time()
    fill_arr(tensor, [])
    return answer


def compress_tensor(arr: Iterable, level: int = 0):
    if not hasattr(arr, "__iter__") and not hasattr(arr, "__cuda_array_interface__"):
        if not np.isnan(arr):
            return arr
        else:
            return None
    else:
        new_arr = []
        for item in arr:
            new_item = compress_tensor(item, level + 1)
            if (
                (not isinstance(new_item, Iterable) and new_item != None)
                or (isinstance(new_item, Iterable) and len(new_item))
                or level <= 0
            ):
                new_arr.append(new_item)
        if len(new_arr):
            return new_arr
        else:
            return [] if level else None


def init_dask_cluster(
    num_workers: int, scheduler_fpath: Optional[str] = None
) -> Client:
    cluster = LocalCluster(n_workers=num_workers)
    # 1. Connects to the dask-cuda-cluster
    client = (
        Client(scheduler_file=scheduler_fpath)
        if scheduler_fpath
        else cluster.get_client()
    )
    print("client information ", client)

    # 2. Blocks until num_workers are ready
    print("Waiting for " + str(num_workers) + " workers...")
    client.wait_for_workers(n_workers=num_workers)

    workers_info = client.scheduler_info()["workers"]
    connected_workers = len(workers_info)

    print(str(connected_workers) + " workers connected")
    print(f"workers_info: {workers_info}")

    return client


def submit_to_dask_cluster():
    pass
