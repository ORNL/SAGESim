from typing import Any, Callable, List, Dict, Optional, Union, Set
from collections import OrderedDict
from math import nan
from pathlib import Path
import inspect


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
        self._no_double_buffer_props: Set[str] = set()  # Properties that should not use double buffering

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

    @property
    def no_double_buffer_props(self) -> Set[str]:
        """Returns the set of property names that should not use double buffering."""
        return self._no_double_buffer_props

    def register_step_func(
        self,
        step_func: Callable,
        module_fpath: str = None,
        priority: int = 0,
        no_double_buffer: Optional[List[str]] = None,
    ):
        """
        What the agent is supposed to do during a simulation step.

        :param step_func: The step function to execute
        :param module_fpath: Path to the module containing the step function.
            If None, auto-detected via inspect.getfile(step_func).
        :param priority: Execution priority (lower values execute first)
        :param no_double_buffer: List of property names that should NOT use double
            buffering in this step function. These properties will write directly
            to the read buffer, making changes visible to subsequent priorities
            within the same tick. Default is None (all written properties use
            double buffering for safety).

            Use this when:
            - Reader priority < Writer priority (reader runs before writer)
            - Accessing different indices (e.g., temporal arrays with t vs t-1)
            - You explicitly want same-tick visibility between priorities
        """
        if module_fpath is None:
            module_fpath = inspect.getfile(step_func)
        self._step_funcs[priority] = (step_func, str(Path(module_fpath).resolve()))

        # Accumulate no_double_buffer properties from all step functions
        if no_double_buffer:
            self._no_double_buffer_props.update(no_double_buffer)
