from sagesim.model import Model
from sagesim.space import NetworkSpace
from cosine_doi_breed import CosineDOIBreed
from state import SIRState


class SIRModel(Model):
    """
    SIRModel class for the SIR model.
    Inherits from the Model class in the sagesim library.
    """

    def __init__(self, p_infection=0.2, p_recovery=0.2) -> None:
        space = NetworkSpace()
        super().__init__(space)
        self._sir_breed = CosineDOIBreed()

        # Register the breed
        self.register_breed(breed=self._sir_breed)

        # register user-defined global properties
        self.register_global_property("p_infection", p_infection)
        self.register_global_property("p_recovery", p_recovery)

        self.num_agents = 0

    # create_agent method takes user-defined properties, that is, the state, preventative_measures, and embedding, to create an agent
    def create_agent(self, state, preventative_measures, embedding=None):
        if embedding is None:
            embedding = []
        agent_id = self.create_agent_of_breed(
            self._sir_breed,
            state=state,
            preventative_measures=preventative_measures,
            embedding=embedding,
        )
        self.num_agents += 1
        return agent_id

    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)

    def simulate(self, num_steps: int, sync_workers_every_n_ticks: int = 1):
        """
        Run the simulation for a specified number of steps.

        Args:
            num_steps (int): Number of simulation steps to run.
        """
        for agent in range(self.num_agents):
            # Initialize infection history for each agent
            self.set_agent_property_value(
                agent,
                "infection_history",
                [SIRState.SUSCEPTIBLE.value for _ in range(num_steps)],
            )
        super().simulate(num_steps, sync_workers_every_n_ticks)

    def get_infection_history(self, agent_id):
        """
        Retrieve the infection history for a specific agent.

        Args:
            agent_id (int): The ID of the agent.

        Returns:
            list: A list representing the infection history of the agent.
        """
        return self.get_agent_property_value(agent_id, "infection_history")
