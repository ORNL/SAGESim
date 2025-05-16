from sagesim.model import Model
from sagesim.space import NetworkSpace
from sir_breed import SIRBreed


class SIRModel(Model):
    """
    SIRModel class for the SIR model.
    Inherits from the Model class in the sagesim library.
    """

    def __init__(self, p_infection=0.2, p_recovery=0.2) -> None:
        space = NetworkSpace()
        super().__init__(space)
        self._sir_breed = SIRBreed()

        # Register the breed
        self.register_breed(breed=self._sir_breed)

        # register user-defined global properties
        self.register_global_property("p_infection", p_infection)
        self.register_global_property("p_recovery", p_recovery)

    # create_agent method takes user-defined properties, that is, the state and preventative_measures, to create an agent
    def create_agent(self, state, preventative_measures):
        agent_id = self.create_agent_of_breed(
            self._sir_breed, state=state, preventative_measures=preventative_measures
        )
        return agent_id

    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)
