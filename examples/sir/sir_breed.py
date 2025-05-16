from enum import Enum
from random import random

from sagesim.breed import Breed

from state import SIRState
from sir_step_func import step_func


# Define the SIRState enumeration for agent states
class SIRState(Enum):
    SUSCEPTIBLE = 1
    INFECTED = 2
    RECOVERED = 3


class SIRBreed(Breed):
    """
    SIRBreed class the SIR model.
    Inherits from the Breed class in the sagesim library.
    """

    def __init__(self) -> None:
        name = "SIR"
        super().__init__(name)
        # Register properties for the breed
        self.register_property("state", SIRState.SUSCEPTIBLE.value)
        self.register_property("preventative_measures", [random() for _ in range(100)])
        # Register the step function
        self.register_step_func(step_func, "sir_step_func.py", 0)
