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

    @property
    def name(self) -> str:
        return self._name

    @property
    def properties(self) -> Dict[str, Any]:
        return self._properties

    @property
    def step_funcs(self) -> Dict[int, Callable]:
        return self._step_funcs

    def register_property(
        self,
        name: str,
        default: Union[int, float, List] = nan,
        max_dims: Optional[List[int]] = None,
    ) -> None:
        self._properties[name] = default
        self._prop2pos[name] = self._num_properties
        self._num_properties += 1
        self._prop2maxdims[name] = max_dims

    def register_step_func(self, step_func: Callable, priority: int = 0):
        """
        What the agent is supposed to do during a simulation step.

        """
        self._step_funcs[priority] = step_func
