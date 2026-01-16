from typing import Any, Callable, List, Dict, Optional, Union
from collections import OrderedDict
from math import nan


class Breed:
    def __init__(self, name: str) -> None:
        # self._properties is a dict with keys as property name and
        #   values are properties.
        #   properties themselves are list of type and default value.
        self._properties: OrderedDict[str, List[Any, Any]] = OrderedDict()
        self._prop2pos: Dict[str, int] = {}
        self._name: str = name
        self._step_funcs: Dict[int, Callable] = {}
        self._breedidx: int = -1
        self._num_properties: int = 0
        self._prop2maxdims: Dict[str, List[int]] = {}
        self._prop2neighbor_visible: Dict[str, bool] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def properties(self) -> Dict[str, Any]:
        return self._properties

    @property
    def step_funcs(self) -> Dict[int, Callable]:
        return self._step_funcs

    @property
    def prop2neighbor_visible(self) -> Dict[str, bool]:
        """Returns mapping of property names to their neighbor_visible flag."""
        return self._prop2neighbor_visible

    def register_property(
        self,
        name: str,
        default: Union[int, float, List] = nan,
        max_dims: Optional[List[int]] = None,
        neighbor_visible: bool = True,
    ) -> None:
        """
        Register a property for this breed.

        :param name: Property name
        :param default: Default value for the property
        :param max_dims: Optional maximum dimensions for the property
        :param neighbor_visible: If True (default), this property will be sent to
            neighboring workers during MPI synchronization. Set to False for properties
            that are never read by neighbors to reduce communication overhead.
        """
        self._properties[name] = default
        self._prop2pos[name] = self._num_properties
        self._num_properties += 1
        self._prop2maxdims[name] = max_dims
        self._prop2neighbor_visible[name] = neighbor_visible

    def register_step_func(
        self, step_func: Callable, module_fpath: str, priority: int = 0
    ):
        """
        What the agent is supposed to do during a simulation step.

        """
        self._step_funcs[priority] = (step_func, module_fpath)
