from sagesim.model import Model
from sagesim.space import NetworkSpace
from sir_breed import SIRBreed


class SIRModel(Model):
    """
    SIRModel class for the SIR model.
    Inherits from the Model class in the sagesim library.
    """

    def __init__(self, p_infection=0.2, p_recovery=0.2, enable_state_tracking=True) -> None:
        space = NetworkSpace()
        super().__init__(space)
        self._sir_breed = SIRBreed()

        # Register the breed
        self.register_breed(breed=self._sir_breed)

        # register user-defined global properties
        self.register_global_property("p_infection", p_infection)
        self.register_global_property("p_recovery", p_recovery)

        # Store state tracking setting
        self.enable_state_tracking = enable_state_tracking

    # create_agent method takes user-defined properties, that is, the state and preventative_measures, to create an agent
    def create_agent(self, state, preventative_measures):
        agent_id = self.create_agent_of_breed(
            self._sir_breed, state=state, preventative_measures=preventative_measures
        )
        return agent_id

    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)

    def simulate(self, ticks: int, sync_workers_every_n_ticks: int = 1) -> None:
        """
        Override simulate to allocate state_history_buffer before simulation.
        """
        # Allocate state_history_buffer for all agents
        for agent_id in range(self._agent_factory.num_agents):
            current_state = self.get_agent_property_value(agent_id, "state")

            if self.enable_state_tracking:
                # Allocate buffer for all ticks
                state_history_buffer = [current_state for _ in range(ticks)]
            else:
                # Allocate single-element buffer that will be overwritten
                state_history_buffer = [current_state]

            self.set_agent_property_value(
                id=agent_id,
                property_name="state_history_buffer",
                value=state_history_buffer
            )

        # Call parent simulate
        super().simulate(ticks, sync_workers_every_n_ticks)

    def get_state_history(self, agent_id: int):
        """
        Get the state history for a specific agent.

        Returns:
            List of states for each tick if tracking enabled, otherwise list with final state
        """
        if not self.enable_state_tracking:
            return None
        return self.get_agent_property_value(agent_id, "state_history_buffer")
