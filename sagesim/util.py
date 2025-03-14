from typing import List, Any

import awkward as ak
import cupy as cp


def convert_to_equal_side_tensor(ragged_list: List[Any]) -> cp.ndarray:
    awkward_array = ak.from_iter(ragged_list)
    assert (
        awkward_array.layout.minmax_depth[0] == awkward_array.layout.minmax_depth[1]
    ), "Tensor is of unequal depth"

    i = 1
    while i < awkward_array.layout.minmax_depth[0]:
        awkward_array = ak.fill_none(awkward_array, [], axis=i - 1)
        nums_in_level = ak.fill_none(ak.ravel(ak.num(awkward_array, axis=i)), value=0)
        awkward_array = ak.pad_none(
            awkward_array, int(max(nums_in_level)), axis=i, clip=True
        )
        i += 1

    awkward_array = ak.fill_none(awkward_array, cp.nan, axis=-1)

    return ak.to_cupy(awkward_array)


def compress_tensor(regular_tensor: cp.ndarray, min_axis: int = 1) -> List[Any]:
    awkward_tensor = ak.from_cupy(regular_tensor)
    awkward_tensor = ak.nan_to_none(awkward_tensor)
    awkward_tensor = ak.drop_none(awkward_tensor)

    i = -1
    while awkward_tensor.layout.minmax_depth[0] + i > min_axis:
        awkward_tensor = ak.mask(awkward_tensor, ak.num(awkward_tensor, axis=i) > 0)
        awkward_tensor = ak.drop_none(awkward_tensor)
        i -= 1

    return ak.to_list(awkward_tensor)
