from sagesim.model import Model
from sagesim.space import NetworkSpace
from sfr_breed import SFRBreed


class SFRModel(Model):

    def __init__(self) -> None:
        space = NetworkSpace()
        super().__init__(space)
        self._sfr_breed = SFRBreed()
        self.register_breed(breed=self._sfr_breed)

    def create_agent(self, popularity, vehicle_num):
        agent_id = self.create_agent_of_breed(self._sfr_breed,
                                                popularity=popularity, 
                                                vehicle_num=vehicle_num
                                            )
        self.get_space().add_agent(agent_id)
        return agent_id

    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)
