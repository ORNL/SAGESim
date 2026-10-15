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
        # priority -> list of breed-local array names the step function takes (None = all)
        self._step_func_blas: Dict[int, Optional[List[str]]] = {}
        # priority -> neighbour roles whose events wake the step function (None = clocked)
        self._step_func_wake_roles: Dict[int, Optional[List[int]]] = {}

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

    @property
    def step_func_blas(self) -> Dict[int, Optional[List[str]]]:
        """priority -> breed-local array names the step function declares (None = all)."""
        return self._step_func_blas

    @property
    def step_func_wake_roles(self) -> Dict[int, Optional[List[int]]]:
        """priority -> neighbour roles whose events wake the step function (None = clocked)."""
        return self._step_func_wake_roles

    def register_step_func(
        self,
        step_func: Callable,
        module_fpath: str = None,
        priority: int = 0,
        no_double_buffer: Optional[List[str]] = None,
        breed_local_arrays: Optional[List[str]] = None,
        wake_on_roles: Optional[List[int]] = None,
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
        :param breed_local_arrays: Names of the breed-local arrays this step function
            takes, in registration order, as `<name>, <name>_idx` pairs after the
            property parameters. Default None = every registered array (the historical
            behaviour). Pass a list so a kernel that uses none of them, or only some,
            keeps a signature that does not change when another array is registered.
        :param wake_on_roles: None (default): the step function runs for every agent of
            the breed every tick. A list of neighbour slots, e.g. [0]: the agents are
            event-woken instead -- an agent is stepped at tick t only if, at tick t-1,
            one of the neighbours in those slots raised the model's event property
            (see Model.configure_wake), or it was woken explicitly. Every breed stepping
            at this priority must then be event-woken. An agent queued twice in a tick
            runs once per woken priority (a per-priority stamp). A kernel that must
            catch state up after sleeping keeps the tick it last ran in its own state.
        """
        if module_fpath is None:
            module_fpath = inspect.getfile(step_func)
        self._step_funcs[priority] = (step_func, str(Path(module_fpath).resolve()))
        self._step_func_blas[priority] = (
            list(breed_local_arrays) if breed_local_arrays is not None else None)
        self._step_func_wake_roles[priority] = (
            [int(r) for r in wake_on_roles] if wake_on_roles is not None else None)

        # Accumulate no_double_buffer properties from all step functions
        if no_double_buffer:
            self._no_double_buffer_props.update(no_double_buffer)
