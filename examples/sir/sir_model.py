from sagesim.model import Model
from sagesim.space import NetworkSpace
from sir_breed import SIRBreed
from state import SIRState


def reduce_agent_data_tensors_(adts_A, adts_B):
    result = []
    # breed would be same as first
    result.append(adts_A[0])
    # network would be same as first
    result.append(adts_A[1])
    # state would be max of both
    result.append(max(adts_A[2], adts_B[2]))
    return result


class SIRModel(Model):

    def __init__(self, p_infection=0.5) -> None:
        space = NetworkSpace()
        super().__init__(space)
        self._sir_breed = SIRBreed()
        self.register_breed(breed=self._sir_breed)
        self.register_global_property("p_infection", p_infection)
        self.register_reduce_function(reduce_agent_data_tensors_)

    def create_agent(self, state):
        agent_id = self.create_agent_of_breed(self._sir_breed, state=state)
        self.get_space().add_agent(agent_id)
        return agent_id

    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)
