from sagesim.model import Model
from sagesim.space import NetworkSpace
from sir_breed import SIRBreed
from state import SIRState


class SIRModel(Model):

    def __init__(self, p_infection=0.5) -> None:
        space = NetworkSpace()
        super().__init__(space)
        self._sir_breed = SIRBreed()
        self.register_breed(breed=self._sir_breed)
        self.register_global_property("p_infection", p_infection)

    def create_agent(self, state):
        agent_id = self.create_agent_of_breed(self._sir_breed, state=state)
        self.get_space().add_agent(agent_id)
        return agent_id

    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)
